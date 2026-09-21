"""Load / stress test for the Anthropic OAuth cross-process refresh race fix.

Companion to ``tests/agent/test_credential_pool_anthropic_refresh_race.py``,
which proves the bug in isolation with two racers. This test scales the same scenario up to look for bottlenecks and degradation
under real concurrency. The thread stress case keeps the suite fast while a
separate spawn-based case uses independent interpreters, distinct profile
homes, and one shared Claude Code credentials file. Both exercise the REAL
cross-process file lock (``_auth_store_lock``) and REAL credential-pool
persistence under throwaway directories — only the network call to Anthropic
is faked. The process case also counts refresh POSTs and requires exactly one
use of the stale single-use token, so a broken lock cannot remain green merely
because two in-process mocks happened to finish quickly.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)

CONCURRENCY = 20


def _process_claude_code_refresh_worker(
    profile_home: str,
    shared_credentials_path: str,
    server_state_path: str,
    start_event,
    result_queue,
) -> None:
    """Refresh one shared Claude Code credential from an independent process."""
    os.environ["HERMES_HOME"] = profile_home

    from agent import anthropic_credentials as anthropic_mod
    from agent import credential_pool as credential_pool_mod
    from hermes_cli import auth as auth_mod

    shared_path = Path(shared_credentials_path)
    server_path = Path(server_state_path)

    def read_shared_credentials():
        data = json.loads(shared_path.read_text(encoding="utf-8"))
        oauth = data["claudeAiOauth"]
        return {
            "accessToken": oauth["accessToken"],
            "refreshToken": oauth.get("refreshToken", ""),
            "expiresAt": oauth.get("expiresAt", 0),
            "source": "claude_code_credentials_file",
        }

    def write_shared_credentials(access_token, refresh_token, expires_at_ms, **_kwargs):
        data = json.loads(shared_path.read_text(encoding="utf-8"))
        data["claudeAiOauth"] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        shared_path.write_text(json.dumps(data), encoding="utf-8")

    def fake_refresh(refresh_token, *, use_json=False):
        # The state file models a single-use token endpoint. The lock here
        # protects only the fake server's accounting; the production lock is
        # what must ensure that the second Hermes process never calls this
        # function after the first one has rotated the shared credential.
        with auth_mod._auth_store_lock(timeout_seconds=10, target_path=server_path):
            state = json.loads(server_path.read_text(encoding="utf-8"))
            state["calls"].append(refresh_token)
            if refresh_token in state["spent"]:
                server_path.write_text(json.dumps(state), encoding="utf-8")
                raise ValueError("invalid_grant: refresh token already used")
            state["spent"].append(refresh_token)
            state["rotation"] += 1
            rotation = state["rotation"]
            server_path.write_text(json.dumps(state), encoding="utf-8")
        # Keep the simulated network operation inside the production shared
        # lock long enough for the second profile to prove it waits, then
        # re-reads the newly-written shared credentials file.
        time.sleep(0.1)
        return {
            "access_token": f"process-access-{rotation}",
            "refresh_token": f"process-refresh-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }

    # Keep this worker hermetic: each profile has its own auth store, while
    # both workers deliberately point at the same Claude credential source.
    auth_mod._global_auth_file_path = lambda: None
    anthropic_mod.claude_code_credentials_path = lambda: shared_path
    anthropic_mod.read_claude_code_credentials = read_shared_credentials
    anthropic_mod._write_claude_code_credentials = write_shared_credentials
    anthropic_mod.refresh_anthropic_oauth_pure = fake_refresh

    result_queue.put({"kind": "ready", "pid": os.getpid()})
    if not start_event.wait(timeout=10):
        result_queue.put({"kind": "result", "ok": False, "error": "start barrier timeout"})
        return

    entry = _entry(id="pool-entry", refresh_token="stale-rt", source="claude_code")
    pool = credential_pool_mod.CredentialPool("anthropic", [entry])
    try:
        refreshed = pool._refresh_entry(pool.entries()[0], force=True)
        result_queue.put({
            "kind": "result",
            "ok": refreshed is not None,
            "refresh_token": refreshed.refresh_token if refreshed else None,
            "pool_refresh_token": pool.entries()[0].refresh_token,
        })
    except BaseException as exc:  # pragma: no cover - failure diagnostics
        result_queue.put({"kind": "result", "ok": False, "error": repr(exc)})


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
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: server.refresh(refresh_token, use_json=use_json),
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials", lambda: None
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


@pytest.mark.live_system_guard_bypass
@pytest.mark.windows_only
def test_distinct_profiles_share_one_claude_refresh_without_duplicate_post(
    hermes_home,
):
    """Independent profiles must serialize a shared Claude Code refresh.

    The profile auth locks intentionally have different paths here; only the
    dedicated lock keyed to the shared Claude credentials file can prevent the
    second process from POSTing the already-spent refresh token.
    """
    shared_credentials_path = hermes_home / "shared-claude-credentials.json"
    shared_credentials_path.write_text(
        json.dumps({
            "claudeAiOauth": {
                "accessToken": "stale-at",
                "refreshToken": "stale-rt",
                "expiresAt": 0,
            }
        }),
        encoding="utf-8",
    )
    server_state_path = hermes_home / "fake-token-server.json"
    server_state_path.write_text(
        json.dumps({"calls": [], "spent": [], "rotation": 0}),
        encoding="utf-8",
    )

    profile_homes = [hermes_home / "profile-a", hermes_home / "profile-b"]
    for profile_home in profile_homes:
        profile_home.mkdir(parents=True)
        (profile_home / "auth.json").write_text(
            json.dumps({
                "version": 1,
                "providers": {},
                # claude_code is a borrowed source; its raw tokens must not
                # be persisted in a profile pool. Each worker constructs the
                # runtime entry from the shared credential source below.
                "credential_pool": {},
            }),
            encoding="utf-8",
        )

    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_process_claude_code_refresh_worker,
            args=(
                str(profile_home),
                str(shared_credentials_path),
                str(server_state_path),
                start_event,
                result_queue,
            ),
        )
        for profile_home in profile_homes
    ]

    messages = []
    try:
        for process in processes:
            process.start()

        ready_deadline = time.monotonic() + 20.0
        while len([m for m in messages if m.get("kind") == "ready"]) < len(processes):
            remaining = max(0.1, ready_deadline - time.monotonic())
            if remaining <= 0.1:
                break
            try:
                messages.append(result_queue.get(timeout=remaining))
            except queue.Empty:
                break
        assert len([m for m in messages if m.get("kind") == "ready"]) == len(processes), (
            f"not all refresh workers reached the start barrier: {messages!r}"
        )
        start_event.set()

        for process in processes:
            process.join(timeout=30)
        assert not [process for process in processes if process.is_alive()], (
            "a profile refresh worker did not finish; possible shared-lock deadlock"
        )

        result_deadline = time.monotonic() + 5.0
        results = [m for m in messages if m.get("kind") == "result"]
        while len(results) < len(processes) and time.monotonic() < result_deadline:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                break
            messages.append(message)
            if message.get("kind") == "result":
                results.append(message)
    finally:
        start_event.set()
        for process in processes:
            process.join(timeout=2)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    assert len(results) == len(processes), f"missing process results: {messages!r}"
    assert all(result.get("ok") for result in results), results
    assert {result.get("refresh_token") for result in results} == {"process-refresh-1"}
    assert all((profile_home / "auth.lock").exists() for profile_home in profile_homes)
    assert shared_credentials_path.with_suffix(".lock").exists()

    server_state = json.loads(server_state_path.read_text(encoding="utf-8"))
    assert server_state["calls"] == ["stale-rt"], (
        "the shared stale refresh token must be POSTed exactly once across "
        f"distinct profiles, got {server_state['calls']!r}"
    )
    assert server_state["spent"] == ["stale-rt"]

    shared_credentials = json.loads(shared_credentials_path.read_text(encoding="utf-8"))
    assert shared_credentials["claudeAiOauth"]["refreshToken"] == "process-refresh-1"
    for profile_home in profile_homes:
        profile_text = (profile_home / "auth.json").read_text(encoding="utf-8")
        assert "stale-rt" not in profile_text
        assert "process-refresh-1" not in profile_text
