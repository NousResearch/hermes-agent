"""Tests for _refresh_bootstrap_cache_scripts (the stale-installer-cache fix).

Pre-#67193 hermes-setup binaries reuse ``bootstrap-cache/install-<branch>.ps1``
forever without re-downloading, so a script cached at install time executes
months-stale code on every GUI update/repair (the 2026-08-09 incident: a
June 4 cached venv stage without the #81327 tree-kill sweep died on
``Access denied``). Those binaries have no self-update path, so ``hermes
update`` now rewrites the mutable-ref cache entries from the fresh checkout —
the stale binary's unconditional reuse then always executes current code.

Contract under test:
- ONLY the cache key for the update-target branch is overwritten with the
  checkout's scripts/install.* content (sanitize_ref parity: bb/gui →
  install-bb_gui.ps1); sibling mutable refs keep their own scripts
- .ps1 gets a UTF-8 BOM (installer cache format, #67193); .sh does not
- commit-pin entries are immutable: is_valid_commit() parity means 7-40
  hex chars — abbreviated SHAs included — are never touched
- missing cache dir / missing sources / IO errors are silent no-ops
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli import main as cli_main

BOM = b"\xef\xbb\xbf"


def _setup(tmp_path, *, ps1=b"WRITE-HOST current\n", sh=b"echo current\n"):
    home = tmp_path / "hermes_home"
    root = tmp_path / "hermes-agent"
    (root / "scripts").mkdir(parents=True)
    (home / "bootstrap-cache").mkdir(parents=True)
    if ps1 is not None:
        (root / "scripts" / "install.ps1").write_bytes(ps1)
    if sh is not None:
        (root / "scripts" / "install.sh").write_bytes(sh)
    return home, root


def _run(home, root, branch="main"):
    with patch.object(cli_main, "get_hermes_home", return_value=str(home)), patch.object(
        cli_main, "PROJECT_ROOT", root
    ):
        cli_main._refresh_bootstrap_cache_scripts(branch)


def test_branch_ref_ps1_is_overwritten_with_bom(tmp_path, capsys):
    home, root = _setup(tmp_path)
    stale = home / "bootstrap-cache" / "install-main.ps1"
    stale.write_bytes(BOM + b"WRITE-HOST stale june script\n")
    _run(home, root)
    assert stale.read_bytes() == BOM + b"WRITE-HOST current\n"
    assert "install-main.ps1" in capsys.readouterr().out


def test_branch_ref_sh_is_overwritten_without_bom(tmp_path):
    home, root = _setup(tmp_path)
    stale = home / "bootstrap-cache" / "install-main.sh"
    stale.write_bytes(b"echo stale\n")
    _run(home, root)
    assert stale.read_bytes() == b"echo current\n"


def test_source_with_existing_bom_not_doubled(tmp_path):
    home, root = _setup(tmp_path, ps1=BOM + b"WRITE-HOST current\n")
    stale = home / "bootstrap-cache" / "install-main.ps1"
    stale.write_bytes(b"old")
    _run(home, root)
    assert stale.read_bytes() == BOM + b"WRITE-HOST current\n"


def test_commit_sha_entries_left_alone(tmp_path):
    home, root = _setup(tmp_path)
    pinned = home / "bootstrap-cache" / ("install-" + "a" * 40 + ".ps1")
    pinned.write_bytes(b"pinned bytes")
    _run(home, root)
    assert pinned.read_bytes() == b"pinned bytes"


def test_abbreviated_commit_pin_left_alone(tmp_path):
    # install_script.rs::is_valid_commit() accepts 7-40 hex chars — an
    # abbreviated pin like install-4ce1994.ps1 is immutable too, and must
    # not be rewritten even when a caller passes it as the branch.
    home, root = _setup(tmp_path)
    pinned = home / "bootstrap-cache" / "install-4ce1994.ps1"
    pinned.write_bytes(b"pinned bytes")
    _run(home, root, branch="4ce1994")
    assert pinned.read_bytes() == b"pinned bytes"


def test_only_target_branch_key_is_refreshed(tmp_path):
    # Coexisting mutable refs cache DIFFERENT branches' scripts: updating
    # main must not clobber install-bb_gui.ps1 with main's script.
    home, root = _setup(tmp_path)
    main_entry = home / "bootstrap-cache" / "install-main.ps1"
    gui_entry = home / "bootstrap-cache" / "install-bb_gui.ps1"
    main_entry.write_bytes(b"stale main")
    gui_entry.write_bytes(b"bb/gui branch script")
    _run(home, root, branch="main")
    assert main_entry.read_bytes() == BOM + b"WRITE-HOST current\n"
    assert gui_entry.read_bytes() == b"bb/gui branch script"


def test_branch_ref_is_sanitized_like_the_installer(tmp_path):
    # sanitize_ref parity: bb/gui -> install-bb_gui.ps1.
    home, root = _setup(tmp_path)
    gui_entry = home / "bootstrap-cache" / "install-bb_gui.ps1"
    gui_entry.write_bytes(b"stale")
    _run(home, root, branch="bb/gui")
    assert gui_entry.read_bytes() == BOM + b"WRITE-HOST current\n"


def test_uncached_branch_is_noop(tmp_path, capsys):
    # A ref the installer never cached has nothing to heal — do not create
    # new cache entries the bootstrapper didn't write itself.
    home, root = _setup(tmp_path)
    _run(home, root, branch="main")
    assert not (home / "bootstrap-cache" / "install-main.ps1").exists()
    assert "Refreshed installer bootstrap-cache" not in capsys.readouterr().out


def test_already_current_entry_not_reported(tmp_path, capsys):
    home, root = _setup(tmp_path)
    current = home / "bootstrap-cache" / "install-main.ps1"
    current.write_bytes(BOM + b"WRITE-HOST current\n")
    _run(home, root)
    assert "Refreshed installer bootstrap-cache" not in capsys.readouterr().out


def test_missing_cache_dir_is_noop(tmp_path):
    home, root = _setup(tmp_path)
    import shutil

    shutil.rmtree(home / "bootstrap-cache")
    _run(home, root)  # must not raise


def test_missing_sources_is_noop(tmp_path):
    home, root = _setup(tmp_path, ps1=None, sh=None)
    stale = home / "bootstrap-cache" / "install-main.ps1"
    stale.write_bytes(b"stale")
    _run(home, root)
    assert stale.read_bytes() == b"stale"


def test_never_raises_on_io_error(tmp_path):
    home, root = _setup(tmp_path)
    with patch.object(cli_main, "get_hermes_home", side_effect=OSError("boom")):
        cli_main._refresh_bootstrap_cache_scripts()  # must not raise
