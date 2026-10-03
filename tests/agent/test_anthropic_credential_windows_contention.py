"""Windows contended-rename coverage for the Anthropic refresh *commit* step.

CPython opens files without ``FILE_SHARE_DELETE`` on Windows, so ``os.replace``
onto ``~/.claude/.credentials.json`` is denied with winerror 5 while ANY other
handle holds it (Claude Code itself, an AV scanner mid-scan, a sibling Hermes).

That denial is maximally expensive here: the commit runs *after* the refresh POST
has already spent the single-use refresh token, so a dropped write quarantines
the pair via ``mark_rotation_consumed_uncommitted`` and the account can only be
recovered with a full ``claude setup-token`` / ``claude /login``.  Observed in
the wild on Windows 11 build 26200 (#109799): one transient winerror 5 at 04:52
bricked every later turn with "No Anthropic credentials found" while the
credentials file on disk was intact.

#109555 moved ``_atomic_write_private_json`` onto ``utils.atomic_json_write``,
which commits through ``utils.atomic_replace`` (#57775) — bounded retries for
winerror 5/32/33, then an in-place rewrite.  These tests pin the *observable*
contract that fix gives this store, so a future writer change cannot silently
reintroduce the brick:

1. a contended target costs a retry, never a re-login;
2. the pair that reaches disk is complete — valid JSON, scopes kept, unrelated
   top-level entries in the file untouched, no half-written copy left behind;
3. a symlinked credentials path keeps the link and the real file is rewritten
   (the deliberate behaviour change documented in #109555).
"""

from __future__ import annotations

import errno
import json
import os
import time

import pytest

from agent import anthropic_credentials as AA

# Synthetic, non-functional token material.
_STALE_ACCESS = "sk-ant-oat01-contended-stale"
_STALE_REFRESH = "sk-ant-ort01-contended-stale"
_ROTATED_ACCESS = "sk-ant-oat01-contended-rotated"
_ROTATED_REFRESH = "sk-ant-ort01-contended-rotated"

_SCOPES = ["user:inference", "user:profile"]

# A neighbour entry Claude Code owns and Hermes must never drop.
_UNRELATED_KEY = "claudeAiSubscription"
_UNRELATED_VALUE = {"tier": "max", "seatId": "seat-42"}

# Far enough in the past that the resolver always refreshes.
_EXPIRED_MS = 1_000


@pytest.fixture(autouse=True)
def _clean_spent_registry():
    """Isolate the process-global consumed-rotation registry between tests."""
    AA._SPENT_ROTATION_FINGERPRINTS.clear()
    yield
    AA._SPENT_ROTATION_FINGERPRINTS.clear()


@pytest.fixture(autouse=True)
def _fast_replace_retries(monkeypatch):
    """Collapse the jittered backoff so the retry budget costs ~0ms."""
    monkeypatch.setattr("utils._REPLACE_RETRY_BASE_DELAY_S", 0.001)
    monkeypatch.setattr("utils._REPLACE_RETRY_MAX_DELAY_S", 0.001)


@pytest.fixture
def claude_credentials(tmp_path, monkeypatch):
    """Point the ``claude_code`` singleton at a tmp file holding a stale pair."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    cred_path = tmp_path / "claude" / ".credentials.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": _STALE_ACCESS,
                    "refreshToken": _STALE_REFRESH,
                    "expiresAt": _EXPIRED_MS,
                    "scopes": _SCOPES,
                },
                _UNRELATED_KEY: _UNRELATED_VALUE,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(AA, "claude_code_credentials_path", lambda: cred_path)
    # The Keychain reader shadows the file on macOS; keep the file the only
    # source so this suite behaves identically on every platform.
    monkeypatch.setattr(AA, "_read_claude_code_credentials_from_keychain", lambda: None)
    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _rotating_refresh)
    monkeypatch.setattr("utils._IS_WINDOWS", True)
    return cred_path


def _rotating_refresh(*_a, **_kw):
    """Stand-in for the token endpoint: always rotates the pair."""
    return {
        "access_token": _ROTATED_ACCESS,
        "refresh_token": _ROTATED_REFRESH,
        "expires_at_ms": int(time.time() * 1000) + 3_600_000,
    }


def _sharing_error() -> PermissionError:
    """The PermissionError shape Windows raises for a target held elsewhere."""
    exc = PermissionError(errno.EACCES, "contended", "src", None, "dst")
    exc.winerror = 5
    return exc


def _assert_store_is_whole(cred_path, expected_pair):
    """The committed file parses, carries *expected_pair*, keeps the scopes Claude
    Code gates on, and still holds the unrelated entry it started with."""
    raw = cred_path.read_text(encoding="utf-8")
    data = json.loads(raw)  # a torn/truncated rewrite fails here
    oauth = data["claudeAiOauth"]
    assert (oauth["accessToken"], oauth["refreshToken"]) == expected_pair
    assert oauth["scopes"] == _SCOPES
    assert data[_UNRELATED_KEY] == _UNRELATED_VALUE


def _quarantined(secret, cred_path) -> bool:
    return AA.is_rotation_consumed_uncommitted(secret, source_path=cred_path)


def _stray_files(cred_dir):
    """Everything beside the credentials file, its lock and its quarantine sidecar —
    writer-agnostic, so any temp naming scheme counts as a stray."""
    expected = {".credentials.json", ".credentials.lock"}
    return sorted(
        p.name
        for p in cred_dir.iterdir()
        if p.name not in expected and "spent-rotations" not in p.name
    )


def test_transient_contention_still_commits_the_rotated_pair(
    claude_credentials, monkeypatch
):
    """A holder that lets go inside the retry budget must cost nothing."""
    real_replace = os.replace
    calls = {"n": 0}

    def contended_twice(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2 and os.path.basename(os.fspath(dst)) == ".credentials.json":
            raise _sharing_error()
        return real_replace(src, dst)

    monkeypatch.setattr("utils.os.replace", contended_twice)

    assert AA._resolve_claude_code_token_from_credentials() == _ROTATED_ACCESS
    _assert_store_is_whole(claude_credentials, (_ROTATED_ACCESS, _ROTATED_REFRESH))
    assert not _quarantined(_STALE_REFRESH, claude_credentials)
    assert _stray_files(claude_credentials.parent) == []


def test_target_held_for_the_whole_call_does_not_force_a_relogin(
    claude_credentials, monkeypatch
):
    """A holder that never releases the target.

    The refresh token is already spent by the time the commit runs, so refusing to
    land it is not a safe no-op: it is the brick from #109799. The rotated pair must
    reach disk whole — and, because the writer's last resort rewrites the live file
    instead of renaming onto it, "whole" is asserted explicitly: parseable JSON,
    scopes kept, the unrelated neighbour entry intact, no partial copy left over.
    """

    def always_contended(src, dst):
        raise _sharing_error()

    monkeypatch.setattr("utils.os.replace", always_contended)

    assert AA._resolve_claude_code_token_from_credentials() == _ROTATED_ACCESS
    _assert_store_is_whole(claude_credentials, (_ROTATED_ACCESS, _ROTATED_REFRESH))
    assert not _quarantined(_STALE_REFRESH, claude_credentials)
    assert not _quarantined(_STALE_ACCESS, claude_credentials)
    assert _stray_files(claude_credentials.parent) == []


def test_symlinked_credentials_path_keeps_the_link_and_rewrites_the_target(
    claude_credentials,
):
    """A symlinked ``.credentials.json`` (dotfile managers, shared stores) survives
    the commit: the link is not replaced by a regular file and the pair lands in the
    file it points at."""
    real_store = claude_credentials.parent.parent / "real-store" / ".credentials.json"
    real_store.parent.mkdir(parents=True, exist_ok=True)
    claude_credentials.replace(real_store)
    try:
        claude_credentials.symlink_to(real_store)
    except (OSError, NotImplementedError) as exc:  # Windows without the privilege
        pytest.skip(f"symlink creation not permitted on this host: {exc}")

    assert AA._resolve_claude_code_token_from_credentials() == _ROTATED_ACCESS
    assert claude_credentials.is_symlink()
    assert os.path.realpath(claude_credentials) == os.path.realpath(real_store)
    _assert_store_is_whole(real_store, (_ROTATED_ACCESS, _ROTATED_REFRESH))
