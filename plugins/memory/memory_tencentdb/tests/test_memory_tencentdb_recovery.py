"""Tests for memory-tencentdb provider self-healing.

Verifies the fixes that prevent the "tdai dies and is never resurrected"
class of failures:

  1. Watchdog thread starts on initialize() and resurrects a dead Gateway
     even when no business request triggers the failure path.
  2. Lazy probe (_ensure_alive_for_request) lets a request short-circuit
     guard self-heal before returning empty, breaking the
     "_gateway_available stuck at False" deadlock.
  3. is_process_alive() correctly distinguishes "child has exited" from
     "child still running but unhealthy".
  4. shutdown() cleanly stops the watchdog and drops the supervisor so
     subsequent recovery attempts are no-ops.

These tests use mocks for the supervisor / client so they neither spawn
real Node processes nor open network sockets.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Inject plugin + hermes-agent roots into sys.path so the provider module
# can be imported regardless of whether tests are invoked from the
# tdai-memory-openclaw-plugin tree (where this file lives at
# hermes-plugin/memory/memory_tencentdb/tests/) or from a hermes-agent
# checkout (where the same file is under tests/plugins/memory/). Mirrors
# the layout used by ``test_gateway_shutdown_leak.py`` next door.
_THIS_FILE = pathlib.Path(__file__).resolve()
_HERE = _THIS_FILE.parent
# When checked into the plugin repo: parents[4] = repo root,
# hermes-plugin/ holds the importable ``plugins`` package.
# When checked into hermes-agent: the tests/ tree already sits under a
# repo root that exposes ``plugins`` directly, so the extra insertion is
# harmless (sys.path lookups stop at the first match).
for candidate in (
    _HERE.parents[3] if len(_HERE.parents) >= 4 else None,    # plugin repo: hermes-plugin/
    _HERE.parents[4] if len(_HERE.parents) >= 5 else None,    # hermes-agent root
    _HERE.parents[2] if len(_HERE.parents) >= 3 else None,    # fallback
):
    if candidate is not None and (candidate / "plugins").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

# Optional: hermes-agent provides ``agent.memory_provider``. Tests can set
# HERMES_AGENT_ROOT to point at a sibling checkout if needed.
_hermes_root = os.environ.get("HERMES_AGENT_ROOT")
if not _hermes_root:
    # Try the canonical sibling layout used by this monorepo.
    sibling = _HERE.parents[4] / "hermes-agent" if len(_HERE.parents) >= 5 else None
    if sibling is not None and (sibling / "agent").is_dir():
        _hermes_root = str(sibling)
if _hermes_root and _hermes_root not in sys.path:
    sys.path.insert(0, _hermes_root)

try:
    from plugins.memory.memory_tencentdb import MemoryTencentdbProvider
    from plugins.memory.memory_tencentdb import supervisor as supervisor_module
except ImportError as e:  # pragma: no cover — env-dependent
    pytest.skip(
        f"memory_tencentdb provider not importable ({e}); set HERMES_AGENT_ROOT "
        "to a hermes-agent checkout if running from the plugin repo.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSupervisor:
    """In-memory stand-in for GatewaySupervisor.

    Lets tests script the sequence of (alive?, healthy?, ensure_running()
    outcome) values without spawning subprocesses or opening sockets.
    """

    def __init__(self) -> None:
        self.alive = True
        self.healthy = True
        # If set, the next ensure_running() call flips alive+healthy back on.
        self.respawn_succeeds = True
        self.client = MagicMock(name="MemoryTencentdbSdkClient")
        self.ensure_running_calls = 0
        self.is_running_calls = 0
        self.is_process_alive_calls = 0
        self.shutdown_calls = 0

    def is_running(self) -> bool:
        self.is_running_calls += 1
        return self.healthy

    def is_process_alive(self) -> bool:
        self.is_process_alive_calls += 1
        return self.alive

    def ensure_running(self) -> bool:
        self.ensure_running_calls += 1
        if self.respawn_succeeds:
            self.alive = True
            self.healthy = True
            return True
        return False

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.alive = False
        self.healthy = False


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def fast_watchdog(monkeypatch):
    """Make the watchdog poll every 50 ms instead of 10 s.

    Tests can then trigger a state change and assert the watchdog reacts
    within a tight bound, keeping the suite fast.
    """
    import plugins.memory.memory_tencentdb as mod

    monkeypatch.setattr(mod, "_WATCHDOG_INTERVAL_SECS", 0.05)
    monkeypatch.setattr(mod, "_WATCHDOG_SHUTDOWN_TIMEOUT_SECS", 0.5)
    # Also collapse the request-path cooldown so tests do not need to wait
    # 15 s between recovery attempts triggered from prefetch / sync_turn.
    monkeypatch.setattr(mod, "_RECOVER_COOLDOWN_SECS", 0)
    yield


@pytest.fixture()
def provider_with_fake_supervisor(monkeypatch, fast_watchdog):
    """Yield a MemoryTencentdbProvider wired to a FakeSupervisor.

    We monkey-patch the GatewaySupervisor symbol used inside the provider
    module so initialize() builds a FakeSupervisor instead of the real one.
    The FakeSupervisor is exposed on the provider as ``_fake`` for tests
    to manipulate.
    """
    import plugins.memory.memory_tencentdb as mod

    fake = FakeSupervisor()

    def _factory(*args, **kwargs):
        return fake

    monkeypatch.setattr(mod, "GatewaySupervisor", _factory)
    # Make the auto-discovery happy: pretend an env var is set so the
    # provider does not try to walk the filesystem looking for server.ts.
    monkeypatch.setenv("MEMORY_TENCENTDB_GATEWAY_CMD", "fake-cmd")

    provider = MemoryTencentdbProvider()
    provider.initialize(session_id="test-session", user_id="test-user")
    provider._fake = fake  # attach for test access

    # initialize() may have spawned _background_start in another thread.
    # Wait until the provider settles into the "available" state before
    # tests start poking at it. The FakeSupervisor reports healthy from
    # the get-go, so this should be quick.
    _wait_until(lambda: provider._gateway_available, timeout=2.0)

    try:
        yield provider
    finally:
        provider.shutdown()


# ---------------------------------------------------------------------------
# Supervisor.is_process_alive
# ---------------------------------------------------------------------------


class _FakePopen:
    def __init__(self, returncode: Optional[int] = None) -> None:
        self._returncode = returncode

    def poll(self):
        return self._returncode

    @property
    def returncode(self):
        return self._returncode


def test_is_process_alive_returns_false_without_spawn():
    sup = supervisor_module.GatewaySupervisor(gateway_cmd="")
    assert sup.is_process_alive() is False


def test_is_process_alive_true_when_running():
    sup = supervisor_module.GatewaySupervisor(gateway_cmd="")
    sup._process = _FakePopen(returncode=None)
    assert sup.is_process_alive() is True


def test_is_process_alive_false_after_exit():
    sup = supervisor_module.GatewaySupervisor(gateway_cmd="")
    sup._process = _FakePopen(returncode=137)
    assert sup.is_process_alive() is False


def test_reap_dead_process_drops_handle():
    sup = supervisor_module.GatewaySupervisor(gateway_cmd="")
    sup._process = _FakePopen(returncode=137)
    sup._reap_dead_process()
    assert sup._process is None


def test_reap_dead_process_keeps_alive_handle():
    sup = supervisor_module.GatewaySupervisor(gateway_cmd="")
    alive = _FakePopen(returncode=None)
    sup._process = alive
    sup._reap_dead_process()
    assert sup._process is alive


# ---------------------------------------------------------------------------
# Watchdog: detects death, resurrects, and reattaches
# ---------------------------------------------------------------------------


def test_watchdog_starts_after_initialize(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    assert provider._watchdog_thread is not None
    assert provider._watchdog_thread.is_alive()


def test_watchdog_detects_dead_gateway_and_resurrects(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    fake = provider._fake

    # Simulate "tdai got SIGKILL'd": process is dead, port is silent.
    fake.alive = False
    fake.healthy = False
    fake.respawn_succeeds = True
    # Also force the provider into the "stuck" state to mimic the
    # production deadlock described in the issue.
    provider._gateway_available = False

    # The watchdog (interval=50ms) should pick this up well within 1s.
    assert _wait_until(
        lambda: fake.ensure_running_calls >= 1, timeout=2.0
    ), "watchdog never called ensure_running on a dead Gateway"

    # And after the respawn succeeds it must flip availability back on.
    assert _wait_until(
        lambda: provider._gateway_available, timeout=2.0
    ), "watchdog respawned but never restored _gateway_available"

    # Client was reattached from the (post-respawn) supervisor instance.
    assert provider._client is fake.client


def test_watchdog_picks_up_external_restart_without_respawning(
    provider_with_fake_supervisor,
):
    """If something external (systemd, operator) brings tdai back, the
    watchdog should NOT spawn a duplicate — it should just notice health
    is back and reattach."""
    provider = provider_with_fake_supervisor
    fake = provider._fake

    # Mark provider as "stuck False" but keep the Gateway healthy.
    provider._gateway_available = False
    fake.alive = True
    fake.healthy = True
    initial_respawns = fake.ensure_running_calls

    assert _wait_until(
        lambda: provider._gateway_available, timeout=2.0
    ), "watchdog never reattached to an externally-healthy Gateway"

    assert fake.ensure_running_calls == initial_respawns, (
        "watchdog spawned a duplicate even though the Gateway was healthy"
    )


def test_watchdog_resets_circuit_breaker_on_recovery(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    fake = provider._fake

    # Trip the breaker manually (mimics 5 consecutive request failures).
    provider._consecutive_failures = 999
    provider._breaker_open_until = time.monotonic() + 60
    provider._gateway_available = False
    fake.alive = False
    fake.healthy = False
    fake.respawn_succeeds = True

    assert _wait_until(
        lambda: provider._gateway_available and not provider._is_breaker_open(),
        timeout=2.0,
    ), "watchdog recovered Gateway but did not reset the breaker"


def test_watchdog_stops_on_shutdown(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    thread = provider._watchdog_thread
    assert thread is not None and thread.is_alive()

    provider.shutdown()

    # After shutdown, the thread must wind down promptly.
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "watchdog kept running after shutdown()"


# ---------------------------------------------------------------------------
# Lazy probe: request path self-heals when stuck-False
# ---------------------------------------------------------------------------


def test_prefetch_recovers_when_stuck_false_and_breaker_closed(
    provider_with_fake_supervisor,
):
    """The original bug: prefetch sees _gateway_available==False and
    short-circuits to "" forever, never giving recovery a chance. After
    the fix, prefetch should attempt a one-shot recovery and proceed."""
    provider = provider_with_fake_supervisor
    fake = provider._fake

    # Stop the watchdog so it cannot sneak in and do the recovery for us;
    # we want to assert that the *request path* is what triggers the heal.
    provider._stop_watchdog()

    # Park the provider in the stuck state.
    provider._gateway_available = False
    provider._client = None
    fake.alive = False
    fake.healthy = False
    fake.respawn_succeeds = True
    fake.client.recall.return_value = {"context": "memories from tdai"}

    result = provider.prefetch(query="hello", session_id="test-session")

    assert "memories from tdai" in result, (
        "prefetch should self-heal and return real memories, got: %r" % result
    )
    assert provider._gateway_available
    assert fake.ensure_running_calls >= 1


def test_prefetch_respects_open_breaker(provider_with_fake_supervisor):
    """Breaker should still take precedence — the lazy probe must not
    turn every request into a respawn attempt during a confirmed outage."""
    provider = provider_with_fake_supervisor
    fake = provider._fake
    provider._stop_watchdog()

    provider._gateway_available = False
    provider._consecutive_failures = 999
    provider._breaker_open_until = time.monotonic() + 60
    initial_respawns = fake.ensure_running_calls

    assert provider.prefetch(query="hello") == ""
    assert fake.ensure_running_calls == initial_respawns, (
        "lazy probe ran ensure_running while breaker was open"
    )


def test_handle_tool_call_recovers_when_stuck_false(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    fake = provider._fake
    provider._stop_watchdog()

    provider._gateway_available = False
    provider._client = None
    fake.respawn_succeeds = True
    fake.client.search_memories.return_value = {"results": ["m1", "m2"]}

    out = provider.handle_tool_call(
        "memory_tencentdb_memory_search", {"query": "anything"}
    )

    assert "results" in out
    assert provider._gateway_available


def test_sync_turn_recovers_when_stuck_false(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    fake = provider._fake
    provider._stop_watchdog()

    provider._gateway_available = False
    provider._client = None
    fake.respawn_succeeds = True
    capture_called = threading.Event()
    fake.client.capture.side_effect = lambda **kw: capture_called.set()

    provider.sync_turn(user_content="u", assistant_content="a")

    assert capture_called.wait(timeout=2.0), (
        "sync_turn never reached the Gateway after lazy recovery"
    )
    assert provider._gateway_available


# ---------------------------------------------------------------------------
# Shutdown safety
# ---------------------------------------------------------------------------


def test_shutdown_drops_supervisor_blocks_recovery(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    fake = provider._fake
    provider.shutdown()

    # Even if a stale request came in after shutdown, _try_recover_gateway
    # must refuse to run (supervisor is None).
    before = fake.ensure_running_calls
    assert provider._try_recover_gateway() is False
    assert fake.ensure_running_calls == before


def test_shutdown_is_idempotent(provider_with_fake_supervisor):
    provider = provider_with_fake_supervisor
    provider.shutdown()
    provider.shutdown()  # must not raise
