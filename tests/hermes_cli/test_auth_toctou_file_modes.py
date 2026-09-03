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


# ---------------------------------------------------------------------------
# _save_auth_store  (~/.hermes/auth.json — every native OAuth provider)
# ---------------------------------------------------------------------------


def test_save_auth_store_writes_0o600_with_0o700_parent(tmp_path, monkeypatch):
    """``_save_auth_store`` must land ``auth.json`` at 0o600 and parent at 0o700."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old_umask = os.umask(0o022)  # make the race observable if it regresses
    try:
        from hermes_cli import auth as auth_mod

        parent_owner = (tmp_path.stat().st_uid, tmp_path.stat().st_gid)

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
    auth_stat = auth_path.stat()
    assert (auth_stat.st_uid, auth_stat.st_gid) == parent_owner, (
        "a new auth.json must inherit its HERMES_HOME directory's owner"
    )

    # Content survived the rewrite
    data = json.loads(auth_path.read_text())
    assert data["providers"]["openai-codex"]["tokens"]["access_token"] == "secret-x"


@pytest.mark.parametrize(
    ("existing", "expected_owner"),
    [
        (True, (1101, 1102)),
        (False, (2201, 2202)),
    ],
    ids=("existing-auth-file", "new-auth-file"),
)
def test_save_auth_store_sets_owner_before_atomic_replace(
    tmp_path, monkeypatch, existing, expected_owner
):
    """A mismatched writer must assign the target owner on the temp fd.

    Fake stat values exercise the privileged-writer branch without requiring
    root or changing real filesystem ownership.
    """
    from hermes_cli import auth as auth_mod

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    auth_path = hermes_home / "auth.json"
    if existing:
        auth_path.write_text(
            json.dumps(
                {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
            )
        )

    real_stat = os.stat
    real_fstat = os.fstat

    class _StatWithOwner:
        def __init__(self, wrapped, uid, gid):
            self._wrapped = wrapped
            self.st_uid = uid
            self.st_gid = gid

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        candidate = os.fspath(path)
        if existing and candidate == os.fspath(auth_path):
            return _StatWithOwner(result, 1101, 1102)
        if candidate == os.fspath(hermes_home):
            return _StatWithOwner(result, 2201, 2202)
        return result

    def fake_fstat(fd):
        return _StatWithOwner(real_fstat(fd), 9901, 9902)

    fchown_calls = []
    replace_observations = []
    real_replace = auth_mod.atomic_replace

    def fake_fchown(fd, uid, gid):
        fchown_calls.append((uid, gid))
        # Some POSIX systems clear permission bits after ownership changes.
        # Make the replace-time assertion depend on the production fchmod.
        os.fchmod(fd, 0o640)

    def spying_replace(tmp, target):
        replace_observations.append(
            (list(fchown_calls), stat.S_IMODE(real_stat(tmp).st_mode))
        )
        return real_replace(tmp, target)

    monkeypatch.setattr(auth_mod.os, "stat", fake_stat)
    monkeypatch.setattr(auth_mod.os, "fstat", fake_fstat)
    monkeypatch.setattr(auth_mod.os, "fchown", fake_fchown)
    monkeypatch.setattr(auth_mod, "atomic_replace", spying_replace)

    auth_mod._save_auth_store(
        {"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}
    )

    assert fchown_calls == [expected_owner]
    assert replace_observations == [([expected_owner], 0o600)], (
        "ownership and mode must be applied to the temp file before replace"
    )
    assert stat.S_IMODE(real_stat(auth_path).st_mode) == 0o600


def test_save_auth_store_recovers_when_owner_cannot_be_preserved(
    tmp_path, monkeypatch
):
    """A foreign-owned store remains replaceable by its directory owner."""
    from hermes_cli import auth as auth_mod

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps({"version": auth_mod.AUTH_STORE_VERSION, "providers": {}})
    )

    real_stat = os.stat

    class _StatWithOwner:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_uid = wrapped.st_uid + 1
            self.st_gid = wrapped.st_gid + 1

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(auth_path):
            return _StatWithOwner(result)
        return result

    def reject_fchown(_fd, _uid, _gid):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(auth_mod.os, "stat", fake_stat)
    monkeypatch.setattr(auth_mod.os, "fchown", reject_fchown)

    saved = auth_mod._save_auth_store(
        {
            "version": auth_mod.AUTH_STORE_VERSION,
            "providers": {"openai-codex": {"tokens": {"access_token": "new"}}},
        }
    )

    assert saved == auth_path
    assert json.loads(auth_path.read_text())["providers"]["openai-codex"]["tokens"][
        "access_token"
    ] == "new"
    assert stat.S_IMODE(real_stat(auth_path).st_mode) == 0o600


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
