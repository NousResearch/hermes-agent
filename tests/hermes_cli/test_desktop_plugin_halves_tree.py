"""The host pass holds an installed package's ``desktop/`` folder to the installers' plain-tree rule.

A link in it would publish whatever it points at on the host into the served ``desktop-plugins/``
folder. A dev checkout linked into ``plugins/`` still publishes its ``desktop/`` folder as it stands.
"""

from pathlib import Path

import pytest

from hermes_cli.desktop_plugin_halves import reconcile_desktop_halves, reconcile_desktop_halves_report

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
