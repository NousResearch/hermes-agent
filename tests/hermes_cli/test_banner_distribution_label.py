"""The banner names the desktop shell shape, not just "desktop-app".

An old-style installer artifact (payload ``bootstrap``, mechanism ``self``)
runs its CLI backend from the same stamp as a bundled release build; the
banner must not present both as identical packaged builds — the installer
shell never updates itself, the managed checkout under it does.
"""

import json

import pytest


@pytest.fixture
def desktop_stamp(tmp_path, monkeypatch):
    def _stamp(payload, update_mechanism, tag=None):
        root = tmp_path / f"app-{payload}-{update_mechanism}"
        root.mkdir()
        (root / "install-stamp.json").write_text(json.dumps({
            "source": "local", "distribution": "desktop-app", "payload": payload,
            "updateMechanism": update_mechanism, "commit": "a" * 40,
            "displayVersion": "0.17.6", "baseVersion": "0.17.6", "tag": tag,
        }))
        monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
        monkeypatch.setattr("pm.paths.repo_root", lambda: root)
        monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
        from hermes_cli.version_info import _reset_version_info_cache
        _reset_version_info_cache()
        return root

    yield _stamp
    from hermes_cli.version_info import _reset_version_info_cache
    _reset_version_info_cache()


def test_bootstrap_shell_banner_says_installer(desktop_stamp):
    from hermes_cli import banner

    desktop_stamp("bootstrap", "self")
    label = banner.format_banner_version_label()
    assert "0.17.6" in label
    assert "installer" in label


def test_bundled_release_banner_does_not_say_installer(desktop_stamp):
    from hermes_cli import banner

    desktop_stamp("bundled", "electron-updater")
    assert "installer" not in banner.format_banner_version_label()
