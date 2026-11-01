"""Host filesystem behaviour of the desktop-half pass.

The host pass treats a Windows junction as the link it is, as Electron's ``lstat`` does.

``Path.is_symlink()`` is False for a junction, and a junction needs no privilege to make on Windows,
so a check that only asks ``is_symlink()`` publishes whatever folder a junction points at. The
real-junction tests run on the Windows lane; the stubbed ones run the same contract on every host.
On macOS and Windows a ``Desktop/`` folder answers for ``desktop/``; that half must publish.
"""

import json
import os
from pathlib import Path

import pytest

from hermes_cli import desktop_plugin_halves as halves
from hermes_cli.desktop_plugin_halves import PACKAGE_MARKER, reconcile_desktop_halves_report


def _package(folder: Path) -> None:
    (folder / "desktop").mkdir(parents=True)
    (folder / "plugin.yaml").write_text("name: media\n", encoding="utf-8")
    (folder / "desktop" / "plugin.js").write_text("export default {}", encoding="utf-8")


def _junction(target: Path, link: Path) -> None:
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def _stub_junctions(monkeypatch, *paths: Path) -> None:
    """Report *paths* as junctions (a plain folder on this host), the reparse tag Windows would give."""
    wanted = {os.fspath(p) for p in paths}
    monkeypatch.setattr(halves, "is_junction", lambda p: os.fspath(p) in wanted)


def _secret_folder(tmp_path: Path) -> Path:
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "token.txt").write_text("host secret", encoding="utf-8")
    return secret


@pytest.mark.platforms("windows")
def test_a_junction_inside_an_installed_desktop_folder_is_not_published(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    package = root / "plugins" / "media"
    _package(package)
    _junction(_secret_folder(tmp_path), package / "desktop" / "assets")

    report = reconcile_desktop_halves_report(root, app_root)
    assert report["touched"] == []
    assert [entry["name"] for entry in report["refused"]] == ["media"]
    assert not (app_root / "media").exists()


@pytest.mark.platforms("windows")
def test_a_dev_checkout_junctioned_into_plugins_is_published_as_linked(tmp_path):
    """Junctions are how a checkout is linked in on Windows without Developer Mode. Electron's lstat
    calls the folder a link and stamps ``linked``; the host must agree, or each writer re-copies
    the other's half on every pass."""
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    checkout = tmp_path / "src" / "media"
    _package(checkout)
    (root / "plugins").mkdir(parents=True)
    _junction(checkout, root / "plugins" / "media")

    assert reconcile_desktop_halves_report(root, app_root) == {"touched": [str(app_root / "media")], "refused": []}
    marker = json.loads((app_root / "media" / PACKAGE_MARKER).read_text(encoding="utf-8"))
    assert marker["linked"] is True


@pytest.mark.platforms("windows")
def test_pruning_a_junctioned_copy_removes_the_junction_not_the_folder_it_points_at(tmp_path):
    """An old copy that kept a link to a checkout, its marker inside the checkout: once the package is
    gone only the junction goes. ``shutil.rmtree`` refuses a junction, so it used to stay forever."""
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    checkout = tmp_path / "src" / "media"
    checkout.mkdir(parents=True)
    (checkout / "plugin.js").write_text("export default {}", encoding="utf-8")
    marker = {"package": "media", "source": str(tmp_path / "gone" / "desktop"), "sourceMtimeMs": 1}
    (checkout / PACKAGE_MARKER).write_text(json.dumps(marker), encoding="utf-8")
    app_root.mkdir(parents=True)
    _junction(checkout, app_root / "media")

    assert reconcile_desktop_halves_report(root, app_root)["touched"] == [str(app_root / "media")]
    assert not os.path.lexists(app_root / "media")
    assert (checkout / "plugin.js").read_text(encoding="utf-8") == "export default {}"


@pytest.mark.parametrize("junction", ["desktop", "desktop/assets"])
def test_a_folder_the_host_reports_as_a_junction_is_refused_like_a_link(tmp_path, monkeypatch, junction):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    package = root / "plugins" / "media"
    _package(package)
    (package / "desktop" / "assets").mkdir()
    _stub_junctions(monkeypatch, package / junction)

    report = reconcile_desktop_halves_report(root, app_root)
    assert [entry["name"] for entry in report["refused"]] == ["media"]
    assert "link (desktop" in report["refused"][0]["reason"]
    assert not (app_root / "media").exists()


def test_a_package_folder_the_host_reports_as_a_junction_is_a_linked_checkout(tmp_path, monkeypatch):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    package = root / "plugins" / "media"
    _package(package)
    _stub_junctions(monkeypatch, package)

    assert reconcile_desktop_halves_report(root, app_root)["touched"] == [str(app_root / "media")]
    marker = json.loads((app_root / "media" / PACKAGE_MARKER).read_text(encoding="utf-8"))
    assert marker["linked"] is True


def test_a_plain_folder_the_host_reports_as_a_junction_at_the_target_is_left_alone(tmp_path, monkeypatch):
    """A junction at ``<app_root>/<name>`` with no marker is the user's own linked plugin: the pass
    neither publishes over it nor deletes it, as it does for a symlink."""
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    _package(root / "plugins" / "media")
    (app_root / "media").mkdir(parents=True)
    (app_root / "media" / "mine.txt").write_text("mine", encoding="utf-8")
    _stub_junctions(monkeypatch, app_root / "media")

    assert reconcile_desktop_halves_report(root, app_root) == {"touched": [], "refused": []}
    assert (app_root / "media" / "mine.txt").read_text(encoding="utf-8") == "mine"


@pytest.mark.platforms("macos", "windows")
def test_a_desktop_folder_spelled_with_a_capital_publishes_on_a_case_insensitive_volume(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    package = root / "plugins" / "media"
    (package / "Desktop").mkdir(parents=True)
    (package / "Desktop" / "plugin.js").write_text("export default {}", encoding="utf-8")
    if not (package / "desktop" / "plugin.js").is_file():
        pytest.skip("this temporary folder is on a case-sensitive volume")

    assert reconcile_desktop_halves_report(root, app_root) == {"touched": [str(app_root / "media")], "refused": []}
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "export default {}"
