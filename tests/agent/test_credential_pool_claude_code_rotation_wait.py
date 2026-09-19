"""A borrowed claude_code credential survives an out-of-band rotation by the `claude` CLI.

Regression for #105797. ``claude_code`` pool entries are borrowed: the single
shared ``~/.claude/.credentials.json`` written by the official ``claude`` CLI,
not Hermes's auth.json, is their token authority. The CLI revokes the old
access token the instant it starts rotating, but writes the replacement pair
some way after — in the reported trace both Hermes refresh attempts had
already failed 1.4 s after the first 401 while the file landed at ~1.8 s.
Every re-read therefore replayed the same consumed refresh token, the entry
was declared unrecoverable, and the worker served the remaining ~30 calls on
its fallback provider although a valid credential existed one second later.

These are contracts between two pieces of state, not snapshots: what recovery
returns is tied to what the file actually holds, and the wait is tied to the
module constant that bounds it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from agent.credential_pool import (
    _CLAUDE_CODE_ROTATION_POLL_SECONDS,
    _CLAUDE_CODE_ROTATION_WAIT_SECONDS,
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)

_HOUR_MS = 3_600_000


def _future_expiry_ms() -> int:
    """An access token that is revoked but NOT expired — the shape in the report."""
    return int(time.time() * 1000) + _HOUR_MS


def _entry(*, access_token: str, refresh_token: str) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id="borrowed-1",
        label="claude code",
        auth_type=AUTH_TYPE_OAUTH,
        priority=1,
        source="claude_code",
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=_future_expiry_ms(),
    )


class _FakeClock:
    """Drives the poll loop without spending real wall-clock time.

    ``slept`` is what the production code asked for in total, which is the
    quantity the bounded-wait contract is about.
    """

    def __init__(self, monkeypatch, *, on_sleep=None) -> None:
        self.now = 1_000.0
        self.slept = 0.0
        self._on_sleep = on_sleep
        monkeypatch.setattr(time, "monotonic", lambda: self.now)
        monkeypatch.setattr(time, "sleep", self._sleep)

    def _sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds
        if self._on_sleep is not None:
            self._on_sleep(self.slept)


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    """A temp ``~/.claude/.credentials.json`` reached through the real path resolver."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # Keychain is a second, host-specific authority; this test is about the file.
    monkeypatch.setattr(
        "agent.anthropic_credentials._read_claude_code_credentials_from_keychain", lambda: None
    )
    path = tmp_path / ".claude" / ".credentials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_credentials(path: Path, access_token: str, refresh_token: str, *, mtime: float) -> None:
    """Write the file the way the `claude` CLI does, with an explicit mtime.

    The mtime is set rather than inherited so the test never depends on the
    filesystem's timestamp granularity.
    """
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": access_token,
                    "refreshToken": refresh_token,
                    "expiresAt": _future_expiry_ms(),
                }
            }
        ),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))


@pytest.fixture(autouse=True)
def _fake_pool_store(monkeypatch):
    """In-memory stand-in for ~/.hermes/auth.json so _adopt() can persist."""
    store: Dict[str, list] = {}

    def _write(provider, entries, *, removed_ids=None, status_cleared_ids=None):
        store[provider] = list(entries)

    def _read(provider=None):
        return dict(store) if provider is None else list(store.get(provider, []))

    monkeypatch.setattr("agent.credential_pool.write_credential_pool", _write)
    monkeypatch.setattr("agent.credential_pool.read_credential_pool", _read)
    return store


@pytest.mark.parametrize(
    "cli_writes",
    [
        # One atomic write, a full second into the window.
        pytest.param({1.0: ("cli-at", "cli-rt")}, id="single-write"),
        # The usual shape on disk: the file is truncated (or a temp file lands)
        # before the real pair is written. The first bump carries no usable
        # access token, so the window must keep polling for the second one.
        pytest.param(
            {0.5: ("", "cli-rt"), 2.0: ("cli-at", "cli-rt")}, id="two-phase-write"
        ),
    ],
)
def test_rotation_landing_during_the_wait_recovers_instead_of_benching(
    claude_home, monkeypatch, cli_writes
):
    """The CLI's rotation arrives mid-wait -> recovery adopts it, no bench, no POST.

    The refresh token in hand is already spent, so any POST must fail; the only
    way out is the file itself changing. Recovery must return the pair the file
    ends up holding, and must NOT spend the CLI's freshly written single-use
    refresh token to get there — including when the CLI takes two writes to get
    there, since an interim write that carries nothing usable must not forfeit
    the rest of the window.
    """
    _write_credentials(claude_home, "revoked-at", "spent-rt", mtime=1_700_000_000.0)
    posted: List[str] = []

    def _refuse(refresh_token, use_json=False):
        posted.append(refresh_token)
        raise ValueError("invalid_grant: refresh token already used")

    monkeypatch.setattr("agent.anthropic_credentials.refresh_anthropic_oauth_pure", _refuse)

    pending = dict(cli_writes)

    def _cli_rotates_after_one_second(slept: float) -> None:
        for at in sorted(pending):
            if slept >= at:
                access_token, refresh_token = pending.pop(at)
                _write_credentials(
                    claude_home, access_token, refresh_token, mtime=1_700_000_000.0 + at
                )
                break

    clock = _FakeClock(monkeypatch, on_sleep=_cli_rotates_after_one_second)

    pool = CredentialPool("anthropic", [_entry(access_token="revoked-at", refresh_token="spent-rt")])
    recovered = pool._recover_failed_refresh(
        pool.entries()[0], ValueError("invalid_grant: refresh token already used")
    )

    on_disk = json.loads(claude_home.read_text(encoding="utf-8"))["claudeAiOauth"]
    assert clock.slept > 0 and on_disk["accessToken"] == "cli-at", (
        "the recovery path never waited, so the CLI's rotation never landed "
        "inside it — without that wait this test would prove nothing"
    )
    assert recovered is not None, (
        "regression #105797: the borrowed claude_code entry was declared "
        "unrecoverable even though the `claude` CLI wrote a valid pair to "
        "~/.claude/.credentials.json while we were still allowed to wait for it."
    )
    assert recovered.access_token != "revoked-at", (
        "recovery handed back the revoked token that just 401'd, not the "
        "rotated one the CLI wrote during the wait"
    )
    assert recovered.access_token == on_disk["accessToken"], (
        "recovery must hand back exactly the token the credentials file holds"
    )
    assert recovered.refresh_token == on_disk["refreshToken"]
    assert posted == [], (
        "the file's pair was valid, so nothing needed POSTing; spending the "
        "CLI's single-use refresh token would rotate it out of its own session"
    )
    assert pool.entries()[0].last_status != STATUS_EXHAUSTED
    assert not pending, "the test's own CLI writes did not all land inside the window"
    assert clock.slept <= _CLAUDE_CODE_ROTATION_WAIT_SECONDS, (
        "the wait must stop as soon as a usable pair is on disk, not run the window out"
    )


@pytest.mark.parametrize(
    "mid_wait_write",
    [
        # Nothing ever touches the file.
        pytest.param(None, id="no-write"),
        # A write lands but carries the same spent pair (a metadata touch or a
        # same-token rewrite). It is not a recovery, and it must not be
        # mistaken for one and end the window early either.
        pytest.param(("revoked-at", "spent-rt"), id="useless-rewrite"),
        # The file's refreshToken rotated but its accessToken is still the one
        # that just 401'd: the sync adopts the rotated refreshToken while the
        # accessToken it already had stays put. ``expiresAt`` is an hour out,
        # so an expiry check calls it healthy — it is not: a revoked token is
        # not an expired token, and handing it back as recovered buys another 401.
        pytest.param(("revoked-at", "cli-rt"), id="new-refresh-token-same-revoked-access"),
    ],
)
def test_credentials_file_that_never_rotates_is_unrecoverable_within_the_bounded_wait(
    claude_home, monkeypatch, mid_wait_write
):
    """No usable rotation ever lands -> still unrecoverable, and the wait is bounded.

    The wait must not become an open-ended retry loop: a genuinely dead
    borrowed credential has to reach its verdict inside the module's own
    window so the fallback can take over.
    """
    _write_credentials(claude_home, "revoked-at", "spent-rt", mtime=1_700_000_000.0)
    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: (_ for _ in ()).throw(ValueError("invalid_grant")),
    )
    written: List[float] = []

    def _touch_once(slept: float) -> None:
        if mid_wait_write is not None and slept >= 0.5 and not written:
            written.append(slept)
            _write_credentials(claude_home, *mid_wait_write, mtime=1_700_000_001.0)

    clock = _FakeClock(monkeypatch, on_sleep=_touch_once)

    pool = CredentialPool("anthropic", [_entry(access_token="revoked-at", refresh_token="spent-rt")])
    recovered = pool._recover_failed_refresh(
        pool.entries()[0], ValueError("invalid_grant: refresh token already used")
    )

    assert recovered is None, (
        "a credentials file still holding the exact pair that just 401'd is "
        "not a recovery — handing it back only buys another 401"
    )
    assert clock.slept >= _CLAUDE_CODE_ROTATION_WAIT_SECONDS, (
        "the wait ended before its window was spent; a write that carries no "
        "usable pair must not forfeit the poll time the late rotation needs"
    )
    assert clock.slept <= _CLAUDE_CODE_ROTATION_WAIT_SECONDS + _CLAUDE_CODE_ROTATION_POLL_SECONDS, (
        "the wait overran the window it is bounded by; a borrowed credential "
        "that is really dead must not hold the turn open indefinitely"
    )


def test_benched_borrowed_entry_is_retried_once_the_credentials_file_rotates(
    claude_home, monkeypatch
):
    """Being benched is a statement about a moment, not about the credential.

    Once the shared file has a new mtime there is something new to try, so the
    same-entry refresh cap must not keep serving the rest of the run on the
    fallback provider (#105797, second half).
    """
    from agent.agent_runtime_helpers import _MAX_AUTH_REFRESH_ATTEMPTS, _recover_auth_failure

    _write_credentials(claude_home, "revoked-at", "spent-rt", mtime=1_700_000_000.0)
    refreshed_entry = _entry(access_token="revoked-at", refresh_token="spent-rt")
    swaps: List[str] = []
    agent = SimpleNamespace(
        provider="anthropic",
        api_mode="chat_completions",
        _auth_pool_refresh_counts={},
        _auth_pool_unrecoverable_mtimes={},
        _is_entitlement_failure=lambda error_context, status_code: False,
        _swap_credential=lambda entry: swaps.append(entry.id),
    )
    pool = SimpleNamespace(try_refresh_matching=lambda **kw: refreshed_entry)

    def _recover():
        return _recover_auth_failure(
            agent, pool, status_code=401, has_retried_429=False, error_context={},
            api_key_hint=None, credential_id=refreshed_entry.id,
            rotate_and_swap=lambda *a: False,
        )[0]

    for _ in range(_MAX_AUTH_REFRESH_ATTEMPTS):
        assert _recover() is True, "the capped attempts themselves must still be tried"
    assert _recover() is False, "the cap must fire once the allowed attempts are spent"

    # The `claude` CLI rotates: same entry, new token material on disk.
    _write_credentials(claude_home, "cli-at", "cli-rt", mtime=1_700_000_002.0)

    assert _recover() is True, (
        "regression #105797: the run stayed pinned to the fallback provider "
        "although the credentials file had been rotated since the entry was "
        "benched — the primary must get another turn once the file changes"
    )
    assert swaps, "a retried primary must actually swap the credential back in"
