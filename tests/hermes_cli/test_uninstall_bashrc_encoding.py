"""Regression tests for shell-config editing encoding on Windows.

``remove_path_from_shell_configs`` reads/writes ``~/.bashrc`` and friends.
A bare ``read_text()``/``write_text()`` uses ``locale.getpreferredencoding()``
(cp1252 on Windows). On a Git Bash install — which the Windows installer
explicitly supports (``install.ps1`` sets ``HERMES_GIT_BASH_PATH``) — users
commonly have non-ASCII bytes in their ``.bashrc`` (CJK comments, accented
paths, Powerline glyphs, emoji prompts). The pre-fix behavior on Windows:

  * if the byte can't decode as cp1252 → ``UnicodeDecodeError``, swallowed by
    the broad ``except``, PATH cleanup silently skipped;
  * if it *can* decode (many UTF-8 multibyte sequences are legal-but-wrong in
    cp1252) → ``write_text()`` re-encoded the mojibake, **permanently
    corrupting** the user's shell config.

The fix reads/writes with ``encoding="utf-8", errors="surrogateescape"`` so
the round-trip is byte-for-byte lossless regardless of the file's encoding.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import hermes_cli.uninstall as uninstall


# --- helpers ---------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point HOME at a tmp dir so find_shell_configs() reads our fixtures."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() caches nothing per-call on POSIX, but be explicit on Windows.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


@pytest.fixture
def windows_default_encoding(monkeypatch):
    """Simulate the Windows locale default for read_text/write_text.

    POSIX runners default to UTF-8, so the cp1252 corruption is invisible
    there. This forces a no-encoding read_text/write_text to use cp1252 (the
    Windows default), so the regression tests catch the bug on any platform.
    Calls that pass an explicit encoding are left untouched.
    """
    real_read = Path.read_text
    real_write = Path.write_text

    def _read(self, *args, **kwargs):
        if "encoding" not in kwargs and not args:
            kwargs["encoding"] = "cp1252"
        return real_read(self, *args, **kwargs)

    def _write(self, data, *args, **kwargs):
        if "encoding" not in kwargs and not args:
            kwargs["encoding"] = "cp1252"
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read)
    monkeypatch.setattr(Path, "write_text", _write)


# --- the bug: a non-ASCII .bashrc must round-trip losslessly ---------------

class TestShellConfigEncodingRoundTrip:
    def test_cjk_bashrc_round_trips_losslessly(self, home, windows_default_encoding):
        """A CJK .bashrc with a hermes PATH line is edited, not corrupted.

        Pre-fix on Windows, the CJK bytes (0xE5..) either raised
        UnicodeDecodeError (skipping cleanup) or mojibake'd the file on write.
        With surrogateescape the non-ASCII bytes survive byte-for-byte. The
        windows_default_encoding fixture makes a no-encoding read/write behave
        like cp1252, so this catches the bug on POSIX runners too.
        """
        bashrc = home / ".bashrc"
        original = (
            "# 我的配置\n"
            'export PATH="$HOME/.local/bin:$PATH"\n'
            'export PATH="/Users/x/.hermes/bin:$PATH"  # Hermes Agent\n'
            "alias ll='ls -la'\n"
            "export PS1='🦊 > '\n"
        ).encode("utf-8")
        _write_bytes(bashrc, original)

        removed_from = uninstall.remove_path_from_shell_configs()

        # The hermes PATH line was removed (so the file changed)…
        assert bashrc in removed_from
        new = bashrc.read_bytes()
        assert b".hermes/bin" not in new
        # …and every non-ASCII byte that wasn't part of the removed line is
        # preserved exactly — no mojibake, no loss.
        assert "我的配置".encode("utf-8") in new
        assert "🦊 > ".encode("utf-8") in new
        assert b"alias ll='ls -la'" in new

    def test_non_utf8_bytes_survive_round_trip(self, home):
        """A legacy ISO-8859 byte must survive even though it isn't valid UTF-8.

        surrogateescape round-trips undecodable bytes through the surrogate
        half, so a file Hermes doesn't fully understand is still written back
        byte-identically (minus the removed line).
        """
        bashrc = home / ".bashrc"
        # 0xE9 is 'é' in ISO-8859-1 but invalid as a standalone UTF-8 byte.
        original = b"# caf\xe9 notes\nexport PATH=/x  # Hermes Agent\n"
        _write_bytes(bashrc, original)

        uninstall.remove_path_from_shell_configs()

        new = bashrc.read_bytes()
        assert b"caf\xe9 notes" in new  # the exotic byte survived
        assert b"# Hermes Agent" not in new

    def test_pure_ascii_bashrc_unchanged_behavior(self, home):
        """The common case (pure ASCII) still removes the hermes line."""
        bashrc = home / ".bashrc"
        original = (
            "# my config\n"
            'export PATH="/Users/x/.hermes/bin:$PATH"  # Hermes Agent\n'
        ).encode("ascii")
        _write_bytes(bashrc, original)

        removed_from = uninstall.remove_path_from_shell_configs()

        assert bashrc in removed_from
        assert b".hermes/bin" not in bashrc.read_bytes()

    def test_file_without_hermes_entries_is_not_rewritten(self, home):
        """A .bashrc with no hermes PATH line is left byte-identical (no write)."""
        bashrc = home / ".bashrc"
        original = "# 我的配置\nalias ll='ls -la'\n".encode("utf-8")
        _write_bytes(bashrc, original)

        removed_from = uninstall.remove_path_from_shell_configs()

        assert removed_from == []
        assert bashrc.read_bytes() == original


# --- the fix: read/write pass an explicit encoding -------------------------

class TestExplicitEncodingPassed:
    """Guard against a future refactor dropping the encoding kwargs."""

    def test_read_and_write_pass_utf8_encoding(self, home, monkeypatch):
        bashrc = home / ".bashrc"
        _write_bytes(
            bashrc,
            b"# config\nexport PATH=/x  # Hermes Agent\n",
        )

        read_calls = []
        write_calls = []

        real_read = Path.read_text
        real_write = Path.write_text

        def spy_read(self, *args, **kwargs):
            read_calls.append(kwargs)
            return real_read(self, *args, **kwargs)

        def spy_write(self, data, *args, **kwargs):
            write_calls.append(kwargs)
            return real_write(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", spy_read)
        monkeypatch.setattr(Path, "write_text", spy_write)

        uninstall.remove_path_from_shell_configs()

        # At least one read happened (and it passed encoding).
        assert read_calls, "expected a read of the .bashrc"
        assert all("encoding" in c for c in read_calls), \
            "read_text() must pass an explicit encoding"
        assert all("utf-8" in str(c["encoding"]).lower() for c in read_calls)
        # The hermes line was present, so a write happened too — with encoding.
        assert write_calls, "expected a rewrite of the .bashrc"
        assert all("encoding" in c for c in write_calls), \
            "write_text() must pass an explicit encoding"
