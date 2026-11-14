"""The host pass holds an installed package's ``desktop/`` folder to the installers' plain-tree rule.

A link in it would publish whatever it points at on the host into the served ``desktop-plugins/``
folder. A dev checkout linked into ``plugins/`` still publishes its ``desktop/`` folder as it stands.
A read-only package (the Nix store, a deb/rpm install) publishes too, and no staging folder outlives a
publish (#154579).
"""

import json
import os
import stat
import time
from pathlib import Path

import pytest

from hermes_cli.desktop_plugin_halves import (
    PACKAGE_MARKER,
    publish_desktop_tree,
    reconcile_desktop_halves,
    reconcile_desktop_halves_report,
)

pytestmark = pytest.mark.platforms("posix")


def _package(folder: Path, outside: Path, link: str = "leak.txt") -> None:
    (folder / "desktop" / "__pycache__").mkdir(parents=True)
    (folder / "plugin.yaml").write_text("name: media\n", encoding="utf-8")
    (folder / "desktop" / "plugin.js").write_text("export default {}", encoding="utf-8")
    (folder / "desktop" / link).symlink_to(outside)


# A root-layout half skips .git/__pycache__ when it copies and checks; a desktop/ folder copies them, so it checks them.
@pytest.mark.parametrize("link", ["leak.txt", "__pycache__/leak.txt"])
def test_an_installed_package_whose_desktop_folder_holds_a_link_is_not_published(tmp_path, link):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret", encoding="utf-8")
    _package(root / "plugins" / "media", secret, link)

    report = reconcile_desktop_halves_report(root, app_root)
    assert report["touched"] == []
    assert [entry["name"] for entry in report["refused"]] == ["media"]
    assert not (app_root / "media").exists()


def test_a_linked_dev_checkout_still_publishes_its_desktop_folder(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    asset = tmp_path / "asset.txt"
    asset.write_text("checkout file", encoding="utf-8")
    checkout = tmp_path / "src" / "media"
    _package(checkout, asset)
    (root / "plugins").mkdir(parents=True)
    (root / "plugins" / "media").symlink_to(checkout, target_is_directory=True)

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "export default {}"


@pytest.fixture
def read_only():
    """Strip the write bits from a tree as a package manager installs it; restore them so tmp cleanup works."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write permissions")
    trees = []

    def _make(tree: Path) -> Path:
        trees.append(tree)
        for folder, _dirs, files in os.walk(tree, topdown=False):
            for name in files:
                os.chmod(Path(folder) / name, stat.S_IRUSR)
            os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
        return tree

    yield _make
    for tree in trees:
        for folder, _dirs, files in os.walk(tree):
            os.chmod(folder, stat.S_IRWXU)
            for name in files:
                os.chmod(Path(folder) / name, stat.S_IRUSR | stat.S_IWUSR)


def test_a_read_only_package_publishes_its_desktop_half_and_leaves_no_staging_folder(tmp_path, read_only):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    package = root / "plugins" / "media"
    (package / "desktop" / "assets").mkdir(parents=True)
    (package / "plugin.yaml").write_text("name: media\n", encoding="utf-8")
    (package / "desktop" / "plugin.js").write_text("export default {}", encoding="utf-8")
    (package / "desktop" / "assets" / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (package / "desktop" / PACKAGE_MARKER).write_text('{"package": "shipped"}', encoding="utf-8")
    read_only(package)

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    assert json.loads((app_root / "media" / PACKAGE_MARKER).read_text(encoding="utf-8"))["package"] == "media"
    assert sorted(p.name for p in app_root.iterdir()) == ["media"]


def test_a_publish_that_fails_on_a_read_only_tree_leaves_no_staging_folder(tmp_path, read_only):
    source, app_root = tmp_path / "src", tmp_path / "desktop-plugins"
    (source / "assets").mkdir(parents=True)
    (source / "plugin.js").write_text("export default {}", encoding="utf-8")
    (source / "assets" / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (source / "unreadable.js").write_text("secret", encoding="utf-8")
    read_only(source)
    os.chmod(source / "unreadable.js", 0)

    with pytest.raises(OSError):
        publish_desktop_tree(source, app_root / "media", {"package": "media"})
    assert list(app_root.iterdir()) == []


def test_reconcile_removes_abandoned_read_only_staging_without_touching_recent_or_user_folders(tmp_path, read_only):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    abandoned = app_root / ".media.staging-abcdefgh"
    abandoned_electron = app_root / ".media.staging-AbC12x"
    recent = app_root / ".media.staging-ijklmnop"
    lookalike = app_root / ".media.staging-AbC12"
    for staging in (abandoned, abandoned_electron, recent, lookalike):
        (staging / "media").mkdir(parents=True)
        (staging / "media" / "plugin.js").write_text("export default {}", encoding="utf-8")
    (app_root / "notes").mkdir()
    old = time.time() - 2 * 60 * 60  # Older than the real one-hour cleanup threshold.
    for staging in (abandoned, abandoned_electron, lookalike):
        read_only(staging)
        os.utime(staging, (old, old))

    assert reconcile_desktop_halves_report(root, app_root) == {"touched": [], "refused": []}
    assert not abandoned.exists()
    assert not abandoned_electron.exists()
    assert (recent / "media" / "plugin.js").exists()
    assert (lookalike / "media" / "plugin.js").exists()
    assert (app_root / "notes").is_dir()
