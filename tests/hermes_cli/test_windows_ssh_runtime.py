"""Tests for hermes_cli.windows_ssh_runtime._open's path-escape guard.

No test file existed for this module before; these are scoped to the specific bug fixed here
(ancestor-junction false positive in the escape check) rather than a full module test suite.
The whole pywin32 surface is faked in-process since this module only runs on native Windows.
"""

import ntpath
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

import hermes_cli.windows_ssh_runtime as windows_ssh_runtime


class _FakeHandle:
    def __init__(self, resolved: str, is_reparse_point: bool = False):
        self.resolved = resolved
        self.is_reparse_point = is_reparse_point


def _make_fake_win32(resolved_by_path: dict, reparse_points: set = frozenset()):
    """A fake win32file/win32con pair. resolved_by_path maps the exact string passed to
    CreateFile to the final resolved path GetFinalPathNameByHandle should report for it -
    this is how we simulate a junction: the string handed to CreateFile and the string
    GetFinalPathNameByHandle reports back can differ."""

    closed = []

    class FakeWin32File:
        FILE_ATTRIBUTE_REPARSE_POINT = 0x400

        def __init__(self):
            self.calls = []

        def CreateFile(self, path, access, share, sa, creation, flags, template):
            self.calls.append(
                {"path": path, "access": access, "share": share, "sa": sa,
                 "creation": creation, "flags": flags}
            )
            if path not in resolved_by_path:
                raise AssertionError(f"unexpected CreateFile({path!r})")
            return _FakeHandle(resolved_by_path[path], is_reparse_point=path in reparse_points)

        def GetFinalPathNameByHandle(self, handle, flags):
            return "\\\\?\\" + handle.resolved

        def GetFileInformationByHandle(self, handle):
            attrs = self.FILE_ATTRIBUTE_REPARSE_POINT if handle.is_reparse_point else 0
            return (attrs,)

        def CloseHandle(self, handle):
            closed.append(handle)

    win32con = SimpleNamespace(
        GENERIC_READ=1, FILE_SHARE_READ=1, FILE_SHARE_WRITE=2,
        OPEN_EXISTING=3, FILE_FLAG_BACKUP_SEMANTICS=0x02000000,
    )
    return SimpleNamespace(win32file=FakeWin32File(), win32con=win32con), closed


def _patch_common(monkeypatch, fake_w):
    monkeypatch.setattr(windows_ssh_runtime, "_win32", lambda: fake_w)
    monkeypatch.setattr(windows_ssh_runtime, "_security_attributes", lambda: None)
    monkeypatch.setattr(windows_ssh_runtime, "_verify_security", lambda handle: None)
    monkeypatch.setattr(windows_ssh_runtime.os, "path", ntpath)


def test_open_accepts_ancestor_junction(monkeypatch):
    """A relocated %LOCALAPPDATA%\\hermes (ancestor junction) must not trip the escape guard
    when the leaf itself is untouched - this was the false positive DeepSeek flagged."""

    path = PureWindowsPath(r"C:\Users\x\AppData\Local\hermes\ssh\token")
    resolved = {
        str(path): r"D:\real-hermes\ssh\token",
        str(path.parent): r"D:\real-hermes\ssh",
    }
    fake_w, closed = _make_fake_win32(resolved)
    _patch_common(monkeypatch, fake_w)

    handle = windows_ssh_runtime._open(path, access=1, creation=3, flags=0)

    assert handle.resolved == r"D:\real-hermes\ssh\token"
    assert handle not in closed
    assert len(closed) == 1

    # DeepSeek review Q6: lock in the exact CreateFile flags _resolved_parent uses for the
    # parent-probe - FILE_FLAG_BACKUP_SEMANTICS (required to open a directory) but NOT
    # _OPEN_REPARSE_POINT (which would stop the parent junction itself from being followed,
    # reintroducing the asymmetry this fix closes). Also pins access/creation/share/sa.
    win32con = fake_w.win32con
    parent_call = next(c for c in fake_w.win32file.calls if c["path"] == str(path.parent))
    assert parent_call["flags"] == win32con.FILE_FLAG_BACKUP_SEMANTICS
    assert not (parent_call["flags"] & 0x00200000)  # _OPEN_REPARSE_POINT must be absent
    assert parent_call["access"] == win32con.GENERIC_READ
    assert parent_call["share"] == win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE
    assert parent_call["creation"] == win32con.OPEN_EXISTING
    assert parent_call["sa"] is None


def test_open_still_rejects_genuine_escape(monkeypatch):
    """A real escape - the leaf resolves somewhere its own resolved parent doesn't explain -
    must still raise. Guards against the fix accidentally widening the check into a no-op."""

    path = PureWindowsPath(r"C:\Users\x\AppData\Local\hermes\ssh\token")
    resolved = {
        str(path): r"D:\attacker-controlled\token",
        str(path.parent): r"C:\Users\x\AppData\Local\hermes\ssh",
    }
    fake_w, closed = _make_fake_win32(resolved)
    _patch_common(monkeypatch, fake_w)

    with pytest.raises(OSError, match="escaped its expected path"):
        windows_ssh_runtime._open(path, access=1, creation=3, flags=0)

    assert len(closed) == 2


def test_open_still_rejects_leaf_reparse_point(monkeypatch):
    """The leaf itself being a reparse point must still raise, even though its resolved path
    now matches its resolved parent + name (the ancestor-junction fix must not swallow this)."""

    path = PureWindowsPath(r"C:\Users\x\AppData\Local\hermes\ssh\token")
    resolved = {
        str(path): r"C:\Users\x\AppData\Local\hermes\ssh\token",
        str(path.parent): r"C:\Users\x\AppData\Local\hermes\ssh",
    }
    fake_w, closed = _make_fake_win32(resolved, reparse_points={str(path)})
    _patch_common(monkeypatch, fake_w)

    with pytest.raises(OSError, match="reparse point"):
        windows_ssh_runtime._open(path, access=1, creation=3, flags=0)

    assert len(closed) == 2
