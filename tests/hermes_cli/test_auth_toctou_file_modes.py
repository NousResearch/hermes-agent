"""Regression tests for TOCTOU-safe credential file writers in ``hermes_cli.auth``.

Background
==========
The three writers below used to create a temp file via ``Path.write_text`` /
``Path.open('w')`` and only ``chmod``'d it to ``0o600`` afterward. Between
create and chmod the file existed at the process umask (typically ``0o644``),
briefly exposing OAuth tokens to other local users on multi-user hosts. The
fix switches them to ``os.open(O_EXCL, mode=0o600)`` + ``os.fdopen`` +
``fsync`` so the file is atomic at ``0o600`` on creation. Mirrors the fixes
shipped for ``agent/google_oauth.py`` (#19673) and ``tools/mcp_oauth.py``
(#21148).

These tests stay green only while the token file and its parent directory
end up at ``0o600`` / ``0o700`` after every write. POSIX-only — the mode-bit
enforcement does not exist on Windows.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)


class _StatWithOwner:
    """Keep the real stat payload while overriding only ownership fields."""

    def __init__(self, result, uid, gid):
        self._result = result
        self.st_uid = uid
        self.st_gid = gid

    def __getattr__(self, name):
        return getattr(self._result, name)


# ---------------------------------------------------------------------------
# _save_auth_store  (~/.hermes/auth.json — every native OAuth provider)
# ---------------------------------------------------------------------------


def test_save_auth_store_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """``_save_auth_store`` must land ``auth.json`` at 0o600 and parent at 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old_umask = os.umask(0o022)  # make the race observable if it regresses
    try:
        from hermes_cli import auth as auth_mod

        auth_store = {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {"openai-codex": {"tokens": {"access_token": "secret-x"}}},
            "active_provider": "openai-codex",
        }
        auth_path = auth_mod._save_auth_store(auth_store)
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(auth_path.stat().st_mode)
    parent_mode = stat.S_IMODE(auth_path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"auth.json mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"auth.json parent dir mode 0o{parent_mode:o} != 0o700 — siblings can traverse"
    )

    # Content survived the rewrite
    data = json.loads(auth_path.read_text())
    assert data["providers"]["openai-codex"]["tokens"]["access_token"] == "secret-x"


@pytest.mark.skipif(not hasattr(os, "fchown"), reason="fchown is unavailable on Windows")
def test_save_auth_store_preserves_existing_owner_before_atomic_replace(tmp_path, monkeypatch):
    """The replacement inode must retain the existing store's service owner.

    This models an administrator rewriting a 0600 store owned by the gateway
    account: ownership must be applied to the temporary inode *before*
    ``atomic_replace`` exposes it.
    """
    from hermes_cli import auth as auth_mod

    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"version": 1, "providers": {}}\n')
    owner = (1234, 5678)

    real_stat = os.stat

    def stat_with_service_owner(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(auth_path):
            return _StatWithOwner(result, *owner)
        return result

    captured = []
    monkeypatch.setattr(auth_mod.os, "stat", stat_with_service_owner)
    monkeypatch.setattr(auth_mod.os, "fchown", lambda fd, uid, gid: captured.append((uid, gid)))

    auth_mod._save_auth_store({"providers": {}}, target_path=auth_path)

    assert captured == [owner]


@pytest.mark.skipif(not hasattr(os, "fchown"), reason="fchown is unavailable on Windows")
def test_save_auth_store_first_write_inherits_parent_owner(tmp_path, monkeypatch):
    """A first store write belongs to the HERMES_HOME owner, not the CLI."""
    from hermes_cli import auth as auth_mod

    auth_path = tmp_path / "auth.json"
    owner = (2345, 6789)
    real_stat = os.stat

    def stat_with_service_owner(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(auth_path.parent):
            return _StatWithOwner(result, *owner)
        return result

    captured = []
    monkeypatch.setattr(auth_mod.os, "stat", stat_with_service_owner)
    monkeypatch.setattr(auth_mod.os, "fchown", lambda fd, uid, gid: captured.append((uid, gid)))

    auth_mod._save_auth_store({"providers": {}}, target_path=auth_path)

    assert captured == [owner]


@pytest.mark.skipif(not hasattr(os, "fchown"), reason="fchown is unavailable on Windows")
def test_save_auth_store_skips_fchown_for_current_owner(tmp_path, monkeypatch):
    """A same-user rewrite must not depend on a needless ownership syscall."""
    from hermes_cli import auth as auth_mod

    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"providers": {}}\n')
    monkeypatch.setattr(auth_mod.os, "fchown", lambda *_args: pytest.fail("unexpected fchown"))

    auth_mod._save_auth_store({"providers": {}}, target_path=auth_path)


@pytest.mark.skipif(not hasattr(os, "fchown"), reason="fchown is unavailable on Windows")
def test_save_auth_store_does_not_replace_when_owner_transfer_fails(tmp_path, monkeypatch):
    """A failed fchown must leave the old credential store untouched."""
    from hermes_cli import auth as auth_mod

    auth_path = tmp_path / "auth.json"
    original = '{"version": 1, "providers": {"old": {}}}\n'
    auth_path.write_text(original)
    real_stat = os.stat

    def stat_with_service_owner(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(auth_path):
            return _StatWithOwner(result, 1234, 5678)
        return result

    monkeypatch.setattr(auth_mod.os, "stat", stat_with_service_owner)
    monkeypatch.setattr(auth_mod.os, "fchown", lambda *_args: (_ for _ in ()).throw(PermissionError("denied")))

    with pytest.raises(PermissionError, match="denied"):
        auth_mod._save_auth_store({"providers": {"new": {}}}, target_path=auth_path)

    assert auth_path.read_text() == original
    assert not list(tmp_path.glob("auth.json.tmp.*"))


# ---------------------------------------------------------------------------
# _save_qwen_cli_tokens  (Qwen CLI OAuth tokens)
# ---------------------------------------------------------------------------


def test_save_qwen_cli_tokens_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """``_save_qwen_cli_tokens`` must land the token file at 0o600 and parent at 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The Qwen CLI auth path lives under $HOME/.qwen by default — isolate it.
    monkeypatch.setenv("HOME", str(tmp_path))
    old_umask = os.umask(0o022)
    try:
        from hermes_cli import auth as auth_mod

        tokens = {
            "access_token": "qwen-secret",
            "refresh_token": "qwen-refresh",
            "token_type": "Bearer",
            "expiry_date": 123,
        }
        auth_path = auth_mod._save_qwen_cli_tokens(tokens)
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(auth_path.stat().st_mode)
    parent_mode = stat.S_IMODE(auth_path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"Qwen token file mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"Qwen token parent dir mode 0o{parent_mode:o} != 0o700"
    )

    data = json.loads(auth_path.read_text())
    assert data["access_token"] == "qwen-secret"


# ---------------------------------------------------------------------------
# Nous shared-credential store write (inside _write_shared_nous_state)
# ---------------------------------------------------------------------------


def test_shared_nous_store_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """The Nous shared-credential store must land at 0o600 / parent 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # _nous_shared_store_path() refuses to touch the real shared store during
    # pytest runs; redirect it into tmp_path explicitly. Use a distinct
    # subdirectory name (``shared_override``) so the guard's "real user
    # home" reference — which currently tracks HERMES_HOME via
    # get_default_hermes_root() — can't collide with our override and
    # falsely claim we're writing to the real user's shared store.
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared_override"))
    old_umask = os.umask(0o022)
    try:
        from hermes_cli import auth as auth_mod

        state = {
            "access_token": "nous-access-xxx",
            "refresh_token": "nous-refresh-xxx",
            "token_type": "Bearer",
            "scope": "openid profile",
            "client_id": "test-client",
            "obtained_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-01-01T01:00:00Z",
        }
        auth_mod._write_shared_nous_state(state)
        path = auth_mod._nous_shared_store_path()
    finally:
        os.umask(old_umask)

    assert path.exists(), "shared Nous store was not written"
    mode = stat.S_IMODE(path.stat().st_mode)
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)

    assert mode == 0o600, (
        f"Nous shared store mode 0o{mode:o} != 0o600 — TOCTOU race regressed"
    )
    assert parent_mode == 0o700, (
        f"Nous shared store parent dir mode 0o{parent_mode:o} != 0o700"
    )

    data = json.loads(path.read_text())
    assert data["refresh_token"] == "nous-refresh-xxx"


# ---------------------------------------------------------------------------
# Atomicity: verify ``os.open`` is called with an explicit 0o600 mode.
# ---------------------------------------------------------------------------


def test_save_auth_store_uses_os_open_with_0o600_mode(tmp_path, monkeypatch):
    """Regression: the writer must call ``os.open`` with an explicit restricted
    mode so the file is created at 0o600 atomically — closing the TOCTOU
    window the previous ``Path.open('w')`` left open (fd inherited process
    umask and was briefly 0o644 before post-write chmod)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    observed_opens: list[tuple[str, int, int]] = []
    real_os_open = os.open

    def spying_os_open(path, flags, mode=0o777, *args, **kwargs):
        observed_opens.append((str(path), flags, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    with patch.object(os, "open", spying_os_open):
        from hermes_cli import auth as auth_mod

        auth_mod._save_auth_store(
            {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
        )

    auth_tmp_opens = [
        (p, fl, m) for (p, fl, m) in observed_opens if "auth.json.tmp" in p
    ]
    assert auth_tmp_opens, (
        f"os.open was never called for the auth.json temp file; "
        f"observed={observed_opens!r}"
    )
    for path, flags, mode in auth_tmp_opens:
        assert flags & os.O_CREAT, f"auth.json temp open missing O_CREAT: path={path}"
        assert flags & os.O_EXCL, (
            f"auth.json temp open missing O_EXCL — TOCTOU-safe pattern regressed: "
            f"path={path}, flags={flags}"
        )
        # Must be exactly S_IRUSR | S_IWUSR (0o600) — no group/other bits.
        expected = stat.S_IRUSR | stat.S_IWUSR
        assert mode == expected, (
            f"auth.json temp open mode 0o{mode:o} != 0o{expected:o} — "
            f"umask would apply and potentially expose tokens"
        )
