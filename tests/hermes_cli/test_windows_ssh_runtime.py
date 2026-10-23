"""Path-escape guard of ``hermes_cli.windows_ssh_runtime._open``.

``%LOCALAPPDATA%\\hermes`` may be a junction to another drive; only that root may resolve
elsewhere, never a Hermes-created path below it. pywin32 is faked: CreateFile follows the given
junctions as NTFS does, except a final component opened with ``_OPEN_REPARSE_POINT``, which
yields the link itself with its reparse attribute.
"""

import os
from types import SimpleNamespace

import pytest

import hermes_cli.windows_ssh_runtime as windows_ssh_runtime

_OWNER = "0" * 32
_NONCE = "0123456789abcdef"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_WIN32CON = SimpleNamespace(
    GENERIC_READ=0x80000000, READ_CONTROL=0x00020000, DELETE=0x00010000, OPEN_EXISTING=3,
    FILE_ATTRIBUTE_NORMAL=0x80, FILE_FLAG_BACKUP_SEMANTICS=0x02000000,
    FILE_SHARE_READ=1, FILE_SHARE_WRITE=2, FILE_SHARE_DELETE=4)
# read_token's open of the session token.
_TOKEN_ACCESS = _WIN32CON.GENERIC_READ | _WIN32CON.READ_CONTROL | _WIN32CON.DELETE
_TOKEN_FLAGS = (_WIN32CON.FILE_ATTRIBUTE_NORMAL | windows_ssh_runtime._OPEN_REPARSE_POINT
                | windows_ssh_runtime._DELETE_ON_CLOSE)


class _Handle:
    def __init__(self, final: str, attributes: int):
        self.final = final
        self.attributes = attributes


def _fake_win32(junctions: dict):
    opened, closed = [], []

    def final_path(path: str) -> str:
        for link in sorted(junctions, key=len, reverse=True):
            if path == link or path.startswith(link + os.sep):
                return final_path(junctions[link] + path[len(link):])
        return path

    class FakeWin32File:
        def CreateFile(self, path, access, share, sa, creation, flags, template):
            if flags & windows_ssh_runtime._OPEN_REPARSE_POINT and path in junctions:
                parent, name = os.path.split(path)
                opened.append(_Handle(os.path.join(final_path(parent), name), _FILE_ATTRIBUTE_REPARSE_POINT))
            else:
                opened.append(_Handle(final_path(path), 0))
            return opened[-1]

        def GetFinalPathNameByHandle(self, handle, flags):
            return "\\\\?\\" + handle.final

        def GetFileInformationByHandle(self, handle):
            return (handle.attributes,)

        def CloseHandle(self, handle):
            closed.append(handle)

    return SimpleNamespace(win32file=FakeWin32File(), win32con=_WIN32CON), opened, closed


def _token_with_junctions(monkeypatch, tmp_path, junctions: dict):
    """``junctions`` maps a link relative to the Hermes root to a target relative to ``tmp_path``."""
    root = tmp_path / "hermes"
    monkeypatch.setattr(windows_ssh_runtime, "get_default_hermes_root", lambda: root)
    fake, opened, closed = _fake_win32(
        {str(root / link): str(tmp_path / target) for link, target in junctions.items()})
    monkeypatch.setattr(windows_ssh_runtime, "_win32", lambda: fake)
    monkeypatch.setattr(windows_ssh_runtime, "_security_attributes", lambda: None)
    monkeypatch.setattr(windows_ssh_runtime, "_verify_security", lambda handle: None)
    return windows_ssh_runtime._token_path(_OWNER, _NONCE), opened, closed


@pytest.mark.parametrize("junctions", [{}, {"": "relocated-hermes"}], ids=["plain-root", "junctioned-root"])
def test_open_accepts_token_under_hermes_root(tmp_path, monkeypatch, junctions):
    token, opened, closed = _token_with_junctions(monkeypatch, tmp_path, junctions)

    handle = windows_ssh_runtime._open(token, _TOKEN_ACCESS, _WIN32CON.OPEN_EXISTING, _TOKEN_FLAGS)

    assert len(closed) == len(opened) - 1
    assert set(closed) == set(opened) - {handle}


@pytest.mark.parametrize("junctions", [
    {"": "relocated-hermes", os.path.join("desktop-ssh", _OWNER): "outside"},
    {"desktop-ssh": os.path.join("hermes", "sibling")},
], ids=["ownership-dir-outside-root", "desktop-ssh-to-sibling-in-root"])
def test_open_rejects_redirect_below_hermes_root(tmp_path, monkeypatch, junctions):
    token, opened, closed = _token_with_junctions(monkeypatch, tmp_path, junctions)

    with pytest.raises(OSError, match="escaped its expected path"):
        windows_ssh_runtime._open(token, _TOKEN_ACCESS, _WIN32CON.OPEN_EXISTING, _TOKEN_FLAGS)

    assert len(closed) == len(opened)
    assert set(closed) == set(opened)


def test_open_rejects_junction_as_token(tmp_path, monkeypatch):
    link = os.path.join("desktop-ssh", _OWNER, f"{_NONCE}.token")
    token, opened, closed = _token_with_junctions(monkeypatch, tmp_path, {link: "outside"})

    with pytest.raises(OSError, match="contains a reparse point"):
        windows_ssh_runtime._open(token, _TOKEN_ACCESS, _WIN32CON.OPEN_EXISTING, _TOKEN_FLAGS)

    assert len(closed) == len(opened)
    assert set(closed) == set(opened)
