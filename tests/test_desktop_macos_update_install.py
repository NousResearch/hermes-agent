from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.main as main


def _make_app(root: Path, marker: str) -> Path:
    app = root / "Hermes.app"
    executable = app / "Contents" / "MacOS" / "Hermes"
    executable.parent.mkdir(parents=True)
    executable.write_text(marker)
    return app


def test_standalone_cli_update_syncs_the_rebuilt_installed_app(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_UPDATE_OWNER", raising=False)
    rebuilt = _make_app(tmp_path / "release" / "mac-arm64", "new renderer")
    target = _make_app(tmp_path / "Applications", "old renderer")

    with (
        mock.patch.object(main.sys, "platform", "darwin"),
        mock.patch.object(
            main, "_desktop_macos_built_app_bundle", return_value=rebuilt
        ),
        mock.patch.object(
            main, "_desktop_macos_installed_app_candidates", return_value=[target]
        ),
        mock.patch.object(main.subprocess, "run"),
    ):
        main._sync_macos_installed_desktop_app(tmp_path / "apps" / "desktop")

    assert (target / "Contents" / "MacOS" / "Hermes").read_text() == "new renderer"


def test_electron_owned_update_leaves_the_running_installed_app_alone(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_DESKTOP_UPDATE_OWNER", "electron")
    rebuilt = _make_app(tmp_path / "release" / "mac-arm64", "new renderer")
    target = _make_app(tmp_path / "Applications", "old renderer")

    with (
        mock.patch.object(main.sys, "platform", "darwin"),
        mock.patch.object(
            main, "_desktop_macos_built_app_bundle", return_value=rebuilt
        ) as built,
        mock.patch.object(
            main, "_desktop_macos_installed_app_candidates", return_value=[target]
        ),
    ):
        main._sync_macos_installed_desktop_app(tmp_path / "apps" / "desktop")

    assert (target / "Contents" / "MacOS" / "Hermes").read_text() == "old renderer"
    built.assert_not_called()


def test_installed_app_swap_rolls_back_if_the_final_rename_fails(tmp_path):
    rebuilt = _make_app(tmp_path / "release" / "mac-arm64", "new renderer")
    target = _make_app(tmp_path / "Applications", "old renderer")
    original_rename = Path.rename

    def fail_staged_rename(path: Path, destination: Path):
        if path.name == "Hermes.app.hermes-update-new":
            raise OSError("simulated final swap failure")
        return original_rename(path, destination)

    with (
        mock.patch.object(main.sys, "platform", "darwin"),
        mock.patch.object(Path, "rename", fail_staged_rename),
        pytest.raises(OSError, match="simulated final swap failure"),
    ):
        main._install_macos_desktop_app_bundle(rebuilt, target)

    assert (target / "Contents" / "MacOS" / "Hermes").read_text() == "old renderer"
    assert not (target.parent / "Hermes.app.hermes-update-new").exists()
