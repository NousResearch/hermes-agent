"""Pending-rename filter for the Windows console-shim update self-lock (#88838, #89599, #86093).

``_filter_pending_shim_renames`` is a pure function over registry
``PendingFileRenameOperations`` entries, so it runs on any host.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import main_install_repair
from hermes_cli import main as cli_main
from hermes_constants import venv_bin_dir
import pytest


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


# ---------------------------------------------------------------------------
# venv layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
def test_venv_scripts_dir_finds_both_layouts(tmp_path, monkeypatch, venv_name):
    """uv writes .venv; our installers write venv. Both must resolve (#79542).

    A ``venv``-only lookup silently returned None on a ``.venv`` install, so the
    whole Windows shim-lock preflight skipped itself. Uses the host's real bin
    dir name (``Scripts``/``bin``), so no OS is faked.
    """
    scripts = venv_bin_dir(tmp_path / venv_name, windows=main_install_repair._is_windows())
    scripts.mkdir(parents=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    assert main_install_repair._venv_scripts_dir() == scripts
