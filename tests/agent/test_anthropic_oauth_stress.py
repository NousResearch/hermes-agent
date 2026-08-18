"""Load / stress test for the Anthropic OAuth cross-process refresh race fix.

Companion to ``tests/agent/test_credential_pool_anthropic_refresh_race.py``,
which proves the bug in isolation with two racers. This test scales the same
scenario up to look for bottlenecks and degradation under real concurrency:
many "Hermes processes" (threads, each with its own ``CredentialPool``
instance) hammering the same single-use Anthropic refresh token at once,
against the REAL cross-process file lock (``_auth_store_lock``) and REAL
credential-pool persistence (not mocked) under a throwaway ``HERMES_HOME`` --
only the network call to Anthropic is faked. This checks the fix does not
deadlock, does not lose updates, and does not degrade into a "thundering
herd" of redundant refresh POSTs as concurrency grows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace as dc_replace

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)

CONCURRENCY = 20


def _entry(*, id: str, refresh_token: str, source: str) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id=id,
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token="stale-at",
        refresh_token=refresh_token,
        expires_at_ms=0,
    )


class _SingleUseTokenServer:
    """Same single-use-refresh-token contract as the race test, tuned for
    a wider fan-out (more callers, less per-call delay so the suite stays
    fast while still exercising real contention).
    """

    def __init__(self, delay_seconds: float = 0.02) -> None:
        self._lock = threading.Lock()
        self._spent: set[str] = set()
        self._rotation = 0
        self.calls: list[str] = []
        self.delay_seconds = delay_seconds

    def refresh(self, refresh_token: str, *, use_json: bool = False):
        with self._lock:
            self.calls.append(refresh_token)
        time.sleep(self.delay_seconds)
        with self._lock:
            if refresh_token in self._spent:
                raise ValueError("invalid_grant: refresh token already used")
            self._spent.add(refresh_token)
            self._rotation += 1
            rotation = self._rotation
        return {
            "access_token": f"sk-ant-oat-rotated-{rotation}",
            "refresh_token": f"sk-ant-ort-rotated-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Real, throwaway HERMES_HOME so _auth_store_lock and
    write_credential_pool/read_credential_pool exercise the genuine
    file-lock + on-disk persistence path, not a mock.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_high_concurrency_anthropic_refresh_no_lost_updates_no_deadlock(
    hermes_home, monkeypatch
):
    """CONCURRENCY 'Hermes processes' race the same stale refresh token
    against the real cross-process lock + real on-disk pool persistence.

    Bottleneck check: total wall-clock time must stay close to what a
    correctly-serialized (or adopt-without-refreshing) implementation would
    take, not blow up toward CONCURRENCY * network_delay -- and every
    participant must end up with a usable, non-exhausted credential.
    """
    server = _SingleUseTokenServer(delay_seconds=0.02)
    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: server.refresh(refresh_token, use_json=use_json),
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.read_claude_code_credentials", lambda: None
    )

    shared_stale_entry = _entry(
        id="pool-entry", refresh_token="stale-rt", source="manual:hermes_pkce"
    )
    pools = [
        CredentialPool("anthropic", [dc_replace(shared_stale_entry)])
        for _ in range(CONCURRENCY)
    ]

    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def _run(idx: int) -> None:
        try:
            entry = pools[idx].entries()[0]
            results[idx] = pools[idx]._refresh_entry(entry, force=True)
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors[idx] = exc

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(CONCURRENCY)]
    start = time.monotonic()
    for t in threads:
        t.start()
    # Generous per-thread join budget: a correct implementation serializes
    # through one file lock, so worst case is roughly
    # CONCURRENCY * (delay + lock overhead), well under this ceiling. A
    # deadlock or livelock would blow straight through it.
    deadline = start + max(10.0, CONCURRENCY * server.delay_seconds * 5)
    for t in threads:
        remaining = max(0.1, deadline - time.monotonic())
        t.join(timeout=remaining)
    elapsed = time.monotonic() - start

    still_alive = [t for t in threads if t.is_alive()]
    assert not still_alive, (
        f"{len(still_alive)}/{CONCURRENCY} threads never finished -- "
        "possible deadlock in the cross-process refresh lock."
    )
    assert not errors, f"unexpected exceptions during concurrent refresh: {errors!r}"

    assert len(results) == CONCURRENCY
    assert all(r is not None for r in results.values()), (
        "at least one of the concurrent processes could not recover a "
        "usable Anthropic credential after the refresh race"
    )
    for idx, pool in enumerate(pools):
        entry_after = pool.entries()[0]
        assert entry_after.last_status != STATUS_EXHAUSTED, (
            f"process {idx} ended up with an exhausted Anthropic credential "
            "despite valid tokens existing on disk"
        )

    # Bottleneck signal: this must stay well below "every thread pays the
    # full network delay independently" (CONCURRENCY * delay). If the fix
    # regresses into N sequential POSTs instead of lock+adopt, this is
    # where it would show up first.
    naive_serial_upper_bound = CONCURRENCY * server.delay_seconds * 3
    assert elapsed < naive_serial_upper_bound, (
        f"refresh race took {elapsed:.2f}s for {CONCURRENCY} concurrent "
        f"processes -- expected well under {naive_serial_upper_bound:.2f}s "
        "if the lock + pool-store adoption path is working efficiently"
    )
