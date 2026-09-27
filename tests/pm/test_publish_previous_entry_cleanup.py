"""A locked DLL in the displaced ``.previous-*`` entry must not fail a successful
install (#124807).

``_publish_entry`` publishes and verifies the NEW entry first; only then does it
remove the displaced ``.previous-*`` bytes. On Windows a process/AV handle can hold
``DLLs/libcrypto-3-x64.dll`` far longer than ``_remove_entry``'s ~2s retry window,
aborting ``hermes update`` with ``[WinError 5]`` even though the new Python is
installed and verified. ``_settle_previous_entry`` already cleans a leftover
``.previous-*`` on the next PM run, so the stale bytes are eventually reclaimed.

The removal after a verified publish is best-effort: unreachable bytes are garbage,
not corruption of the live entry. Removal failures during rollback/staging
(``_restore_previous_entry``, ``_settle_previous_entry``'s restore branch) still
propagate — there the surviving bytes ARE the live entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.install import _publish_entry, _settle_previous_entry  # noqa: E402


class _FakeStore:
    """Minimal stand-in for pm.store.Store: entries live under root."""

    def __init__(self, root: Path):
        self.root = root

    def entry(self, name: str) -> Path:
        return self.root / name

    def publish(self, staged: Path, name: str) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(target)


class _FakePackage:
    name = "python"

    def verify(self, entry: Path, target: str):
        return None  # verified


def _populate(entry: Path) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "DLLs").mkdir()
    (entry / "DLLs" / "libcrypto-3-x64.dll").write_bytes(b"MZ")


def test_locked_previous_entry_does_not_fail_verified_publish(tmp_path, monkeypatch):
    """WinError-5-style persistent lock on the displaced entry: publish succeeds."""
    import pm.install as install_mod

    store = _FakeStore(tmp_path)
    staged = tmp_path / ".scratch" / "tree"
    staged.mkdir(parents=True)
    _populate(staged)
    entry = store.entry("python-3.14.7-win32-x64")
    previous_entry = store.entry(".previous-python-3.14.7-win32-x64")
    _populate(previous_entry)

    calls = {"n": 0}

    def locked_remove(store_arg, name):
        calls["n"] += 1
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(install_mod, "_remove_entry", locked_remove)

    with _publish_entry(_FakePackage(), store, staged, entry, previous_entry, "win32-x64"):
        pass  # caller's facts commit

    # The new entry is live and the install completed instead of raising.
    assert entry.is_dir()
    assert (entry / "DLLs" / "libcrypto-3-x64.dll").exists()
    # Cleanup was attempted, and the stale bytes stay for the next settle pass.
    assert calls["n"] >= 1
    assert previous_entry.exists()


def test_next_run_settles_the_leftover_previous_entry(tmp_path, monkeypatch):
    """The leftover .previous-* from a locked publish is reclaimed by settle."""
    import pm.install as install_mod

    store = _FakeStore(tmp_path)
    entry = store.entry("python-3.14.7-win32-x64")
    previous_entry = store.entry(".previous-python-3.14.7-win32-x64")
    _populate(entry)
    _populate(previous_entry)

    removed = []

    def recording_remove(store_arg, name):
        removed.append(name)

    monkeypatch.setattr(install_mod, "_remove_entry", recording_remove)

    # previous facts say the live entry was published from this exact install.
    previous = {"entry": entry.name}
    monkeypatch.setattr(install_mod, "_entry_verified", lambda *a, **k: True)

    _settle_previous_entry(_FakePackage(), store, entry, previous_entry, previous, "win32-x64")

    assert removed == [previous_entry.name]


def test_rollback_removal_failure_still_propagates(tmp_path, monkeypatch):
    """Inside the exception path the displaced bytes ARE the live entry: a failed
    removal must still raise (behavior unchanged)."""
    import pm.install as install_mod

    store = _FakeStore(tmp_path)
    staged = tmp_path / ".scratch" / "tree"
    staged.mkdir(parents=True)
    _populate(staged)
    entry = store.entry("python-3.14.7-win32-x64")
    previous_entry = store.entry(".previous-python-3.14.7-win32-x64")
    _populate(previous_entry)

    def locked_remove(store_arg, name):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(install_mod, "_remove_entry", locked_remove)

    with pytest.raises(PermissionError):
        with _publish_entry(_FakePackage(), store, staged, entry, previous_entry, "win32-x64"):
            raise RuntimeError("facts commit blew up")
