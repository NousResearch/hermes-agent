"""Retired dependency handoff stays inert; queued artifact recovery stays live."""
from pathlib import Path

import pytest

from hermes_cli import main as cli_main, main_install_repair


def test_pending_rename_filter_drops_only_our_shim_pairs():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [
        r"\??\C:\other\thing.dll", r"!\??\C:\other\thing.dll.bak",
        r"\??\C:\hermes\venv\Scripts\hermes.exe",
        r"!\??\C:\hermes\venv\Scripts\hermes.exe.old.1755624735000",
    ]
    kept, removed = main_install_repair._filter_pending_shim_renames(entries, shims)
    assert removed == 1
    assert kept == entries[:2]


def test_pending_rename_filter_keeps_a_shim_pair_with_a_foreign_target():
    shims = [Path(r"C:\hermes\venv\Scripts\hermes.exe")]
    entries = [r"\??\C:\hermes\venv\Scripts\hermes.exe", r"!\??\C:\somewhere\else.exe"]
    assert main_install_repair._filter_pending_shim_renames(entries, shims) == (entries, 0)


def test_pending_rename_filter_preserves_a_trailing_delete_entry():
    entries = [r"\??\C:\other\thing.dll", "", r"\??\C:\other\orphan.dll"]
    assert main_install_repair._filter_pending_shim_renames(entries, []) == (entries, 0)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
def test_legacy_shim_recovery_finds_both_layouts(tmp_path, monkeypatch, venv_name):
    scripts = tmp_path / venv_name / "Scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    assert main_install_repair._venv_scripts_dir() == scripts
