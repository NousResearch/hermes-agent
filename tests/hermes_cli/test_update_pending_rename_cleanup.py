"""Tests for #85839 — orphaned PendingFileRenameOperations cleanup.

``_schedule_replace_on_reboot`` queues ``MoveFileExW(MOVEFILE_DELAY_UNTIL_REBOOT)``
pairs into the Session Manager registry.  Neither success nor failure of the
subsequent recovery install removes the queued entries, so across repeated
failed boot-recoveries they accumulate (one per failed boot).  On the next
reboot the Session Manager applies entry #1, renaming the current, healthy
``hermes.exe`` → ``hermes.exe.old.<ts>`` — the shim vanishes after a reboot
that was supposed to fix things.

``_cleanup_pending_file_rename_operations`` (called from
``_cleanup_quarantined_exes`` on every launch) scans the registry value and
removes hermes-shim pairs whose source file no longer exists (the shim was
rewritten by a later install — the pending rename is a booby trap) or whose
target ``.old.`` backup no longer exists (stale pair from a failed cycle).
Pairs for other applications are left untouched.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#

# NT path prefixes used by MoveFileExW in the registry value.
_NT_PREFIX = "\\??\\"


def _nt(path: str) -> str:
    """Wrap a path in the NT ``\\??\\`` prefix used by the Session Manager."""
    return _NT_PREFIX + path


def _make_winreg_mock(
    *,
    initial_entries: list[str] | None = None,
    reg_type: int = 11,  # REG_MULTI_SZ
):
    """Build a winreg mock with read/write tracking.

    ``initial_entries`` is the full REG_MULTI_SZ list (including the trailing
    empty string).  When ``None``, the value does not exist (FileNotFoundError).
    """
    if initial_entries is not None:
        stored = {"value": list(initial_entries), "type": reg_type}
    else:
        stored = None

    class _Key:
        def CloseKey(self):
            pass

    key_handle = _Key()

    # Sentinel: use a unique object to represent FileNotFoundError on
    # QueryValueEx so the mock can distinguish "value missing" from "value
    # present but empty".
    _NOT_FOUND = object()

    def _query_value_ex(key, name):
        if stored is None:
            raise FileNotFoundError(name)
        return (list(stored["value"]), stored["type"])

    def _set_value_ex(key, name, reserved, vtype, value):
        stored["value"] = list(value)
        stored["type"] = vtype

    winreg_mod = types.ModuleType("winreg")
    winreg_mod.HKEY_LOCAL_MACHINE = 0x80000002
    winreg_mod.KEY_READ = 0x20019
    winreg_mod.KEY_WRITE = 0x20006
    winreg_mod.REG_MULTI_SZ = 11
    winreg_mod.REG_SZ = 1
    winreg_mod.QueryValueEx = _query_value_ex
    winreg_mod.SetValueEx = _set_value_ex

    open_calls = []

    def _open_key(root, subpath, reserved, access):
        open_calls.append((root, subpath, reserved, access))
        return key_handle

    winreg_mod.OpenKey = _open_key
    winreg_mod.CloseKey = lambda key: None

    return winreg_mod, open_calls, stored


def _patch_windows_and_winreg(monkeypatch, winreg_mod):
    """Patch _is_windows → True and inject the winreg mock."""
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    # Inject winreg into the sys.modules so `import winreg` inside the
    # function body picks up the mock.
    import sys
    monkeypatch.setitem(sys.modules, "winreg", winreg_mod)


# ---------------------------------------------------------------------------#
# Tests
# ---------------------------------------------------------------------------#


def test_removes_orphaned_pair_when_source_no_longer_exists(tmp_path, monkeypatch):
    """The canonical #85839 scenario: the shim was rewritten by a later
    install, but the pending rename entry survived.  It must be removed so
    the next reboot does not rename the healthy shim away."""
    healthy_shim = tmp_path / "Scripts" / "hermes.exe"
    healthy_shim.parent.mkdir(parents=True)
    healthy_shim.write_bytes(b"\x00")  # exists — but the pending entry targets a DIFFERENT path

    # The pending entry's source path points to a file that no longer exists
    # (the .exe.old shim was swept by _cleanup_quarantined_exes).
    orphan_src = str(tmp_path / "Scripts" / "hermes.exe.old.1786674911157")
    orphan_tgt = str(tmp_path / "Scripts" / "hermes.exe.old.1786670000000")

    entries = [_nt(orphan_src), _nt(orphan_tgt), ""]

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    # The orphaned pair should be gone; only the trailing "" remains.
    assert stored is not None
    assert stored["value"] == [""]


def test_removes_pair_when_target_old_backup_gone(tmp_path, monkeypatch):
    """Source still exists but target .old. backup was already swept —
    the pending rename has no valid target and is stale."""
    src = tmp_path / "Scripts" / "hermes.exe"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x00")

    # Target .old. does NOT exist.
    orphan_tgt = str(tmp_path / "Scripts" / "hermes.exe.old.9999999999999")

    entries = [_nt(str(src)), _nt(orphan_tgt), ""]

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    assert stored is not None
    assert stored["value"] == [""]


def test_removes_armed_pair_healthy_shim_with_old_target(tmp_path, monkeypatch):
    """The #85839 booby trap: healthy hermes.exe exists AND .old. target exists.
    This is the ARMED rename-away that will clobber the healthy shim on next boot.
    Once we're running, the install is good — disarm it regardless of target existence."""
    healthy_shim = tmp_path / "Scripts" / "hermes.exe"
    old_backup = tmp_path / "Scripts" / "hermes.exe.old.1786674911157"
    healthy_shim.parent.mkdir(parents=True)
    healthy_shim.write_bytes(b"new")
    old_backup.write_bytes(b"old")

    # Both exist — source is the HEALTHY shim (no .old. in name), target is .old.
    entries = [_nt(str(healthy_shim)), _nt(str(old_backup)), ""]

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    # The armed pair should be DISARMED (removed) — healthy shim must not be renamed away.
    assert stored is not None
    assert stored["value"] == [""]


def test_removes_armed_pair_both_exist_legacy_case(tmp_path, monkeypatch):
    """Legacy test case: source=hermes.exe + target=hermes.exe.old.* both exist.
    This IS the armed trap — the source is the healthy shim (no .old.) and target
    is .old. backup. Should be removed to prevent next-boot clobber."""
    src = tmp_path / "Scripts" / "hermes.exe"
    tgt = tmp_path / "Scripts" / "hermes.exe.old.1234567890"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x00")
    tgt.write_bytes(b"\x00")

    entries = [_nt(str(src)), _nt(str(tgt)), ""]

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    # Armed pair removed — healthy shim must not be renamed away on next boot.
    assert stored is not None
    assert stored["value"] == [""]


def test_preserves_non_hermes_entries(tmp_path, monkeypatch):
    """Entries for other applications (not hermes*.exe) must never be touched."""
    healthy = tmp_path / "Scripts" / "hermes.exe"
    healthy.parent.mkdir(parents=True)
    healthy.write_bytes(b"\x00")

    orphan_hermes = str(tmp_path / "Scripts" / "hermes.exe.old.111")
    other_app_src = str(tmp_path / "other_app.exe")
    other_app_tgt = str(tmp_path / "other_app.exe.old.222")

    # REG_MULTI_SZ: pairs are consecutive, only a single trailing "" terminator.
    entries = [
        _nt(orphan_hermes), _nt(str(tmp_path / "Scripts" / "hermes.exe.old.000")),
        _nt(other_app_src), _nt(other_app_tgt),
        "",
    ]

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    assert stored is not None
    # The hermes pair is removed (orphan source gone); the other-app pair stays.
    assert stored["value"] == [_nt(other_app_src), _nt(other_app_tgt), ""]


def test_noop_when_registry_value_missing(tmp_path, monkeypatch):
    """When PendingFileRenameOperations does not exist, the function is a
    silent no-op."""
    winreg_mod, open_calls, stored = _make_winreg_mock(initial_entries=None)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    # Should not raise.
    cli_main._cleanup_pending_file_rename_operations()
    assert stored is None  # nothing was written


def test_noop_on_non_windows(monkeypatch):
    """On non-Windows the function is a silent no-op."""
    monkeypatch.setattr(cli_main, "_is_windows", lambda: False)
    # Should not raise, should not try to import winreg.
    cli_main._cleanup_pending_file_rename_operations()


def test_noop_on_wrong_reg_type(tmp_path, monkeypatch):
    """If the value exists but is not REG_MULTI_SZ, do not touch it."""
    winreg_mod, _, stored = _make_winreg_mock(
        initial_entries=["C:\\hermes.exe", ""],
        reg_type=1,  # REG_SZ, not REG_MULTI_SZ
    )
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    # Stored value is unchanged.
    assert stored is not None
    assert stored["value"] == ["C:\\hermes.exe", ""]
    assert stored["type"] == 1


def test_noop_on_malformed_entries(tmp_path, monkeypatch):
    """Odd number of entries (malformed) — don't risk corrupting it."""
    src = tmp_path / "Scripts" / "hermes.exe.old.111"
    src.parent.mkdir(parents=True)

    entries = [_nt(str(src))]  # odd — no trailing "" even

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=entries)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    # Unchanged.
    assert stored is not None
    assert stored["value"] == entries


def test_integration_called_from_cleanup_quarantined_exes(tmp_path, monkeypatch):
    """_cleanup_quarantined_exes calls _cleanup_pending_file_rename_operations
    on Windows (integration check via patching the inner call)."""
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)

    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir(parents=True)

    # Need to mock winreg so the real function doesn't touch the actual registry.
    winreg_mod, _, _ = _make_winreg_mock(initial_entries=[""])
    import sys
    monkeypatch.setitem(sys.modules, "winreg", winreg_mod)

    called = False
    original = cli_main._cleanup_pending_file_rename_operations

    def _spy():
        nonlocal called
        called = True
        # Call the real one too so it doesn't break anything.
        original()

    monkeypatch.setattr(cli_main, "_cleanup_pending_file_rename_operations", _spy)

    # Need to patch venv scripts dir resolution.
    monkeypatch.setattr(cli_main, "_venv_scripts_dir", lambda: scripts_dir)

    cli_main._cleanup_quarantined_exes(scripts_dir)

    assert called, "cleanup_quarantined_exes did not call _cleanup_pending_file_rename_operations"


def test_multiple_orphaned_pairs_all_removed(tmp_path, monkeypatch):
    """The #85839 report: 12 accumulated entries from failed boot-recoveries.
    All orphaned hermes pairs should be cleaned in one sweep."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir(parents=True)
    # None of the source paths exist.
    pairs = []
    for i in range(12):
        src = str(scripts / f"hermes.exe.old.{1786674911157 + i}")
        tgt = str(scripts / f"hermes.exe.old.{1786670000000 + i}")
        pairs.extend([_nt(src), _nt(tgt)])
    pairs.append("")

    winreg_mod, _, stored = _make_winreg_mock(initial_entries=pairs)
    _patch_windows_and_winreg(monkeypatch, winreg_mod)

    cli_main._cleanup_pending_file_rename_operations()

    assert stored is not None
    assert stored["value"] == [""]
