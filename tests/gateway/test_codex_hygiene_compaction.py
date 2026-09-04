"""Regression tests for #73503 — codex_app_server compression must not be a no-op.

On the codex_app_server runtime the model's real working context is the
app-server's server-side thread (CodexAppServerSession is constructed with no
history and each turn submits only the new user message), so:

* the old detached-agent hygiene path could only rewrite the transcript
  mirror — a guaranteed no-op ("compressed 150 -> 150 msgs") — and then
  evicted the live cached agent, destroying the thread that held the only
  real context;
* the fix routes hygiene and manual /compress to the LIVE cached agent's
  ``thread/compact/start`` (asserted here at the compact_thread RPC-stub
  boundary) and keeps that agent cached;
* ``compression.codex_app_server_auto`` semantics hold: only ``hermes``
  lets Hermes' threshold start a compaction; ``native``/``off`` skip
  cleanly, and no mode ever runs the local transcript compressor.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agent.conversation_compression import compress_context
from agent.transports.codex_app_server_session import TurnResult
from gateway.run import run_codex_hygiene_compaction


class FakeCodexSession:
    """RPC-stub boundary: stands in for the app-server thread client."""

    def __init__(self, result=None):
        self.result = result or TurnResult(
            thread_id="thread-1", turn_id="compact-1", compacted=True
        )
        self.compact_calls = 0
        self.closed = False

    def compact_thread(self):
        self.compact_calls += 1
        return self.result

    def close(self):
        self.closed = True


class LiveCodexAgent:
    """Minimal live cached agent whose _compress_context is the REAL routing.

    The forwarder mirrors run_agent.AIAgent._compress_context: it calls the
    module-level compress_context(), so these tests exercise the genuine
    codex route (mode gate, cooldown, compact_thread RPC) rather than a stub
    of the code under test.
    """

    def __init__(self, mode="hermes", session=None):
        self.api_mode = "codex_app_server"
        self.codex_app_server_auto_compaction = mode
        self.session_id = "sess-1"
        self.platform = "telegram"
        self._cached_system_prompt = "cached prompt"
        self._codex_session = session if session is not None else FakeCodexSession()
        self.context_compressor = SimpleNamespace(
            compression_count=0,
            last_compression_rough_tokens=0,
            last_prompt_tokens=100,
            last_completion_tokens=10,
            awaiting_real_usage_after_compression=False,
        )
        self.local_compress_calls = 0
        self.warnings = []

    # -- surface used by compress_context --------------------------------
    def _touch_activity(self, *a, **k):
        pass

    def _emit_status(self, m):
        pass

    def _emit_warning(self, m):
        self.warnings.append(m)

    def _build_system_prompt(self, s):
        return "built prompt"

    def _compress_context(self, messages, system_message, **kwargs):
        return compress_context(self, messages, system_message, **kwargs)


def _history(n=150):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}" * 50}
        for i in range(n)
    ]


def _gateway(tmp_path, session_key="tg:123", agent=None):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    gw = SimpleNamespace(
        _agent_cache={} if agent is None else {session_key: (agent, 0.0)},
        _agent_cache_lock=None,
        _session_db=db,
    )
    return gw, db


# ---------------------------------------------------------------------------
# Core no-op regression: hermes mode + live thread => thread/compact runs
# ---------------------------------------------------------------------------

def test_hermes_mode_compacts_live_thread_at_rpc_boundary(tmp_path):
    agent = LiveCodexAgent(mode="hermes")
    key = "tg:123"
    gw, db = _gateway(tmp_path, key, agent)
    # Pre-arm a persisted failure streak so success provably resets it
    # through the REAL SessionDB (hygiene path involves the DB).
    assert db.increment_hygiene_failure_streak(key) == 1

    outcome = asyncio.run(
        run_codex_hygiene_compaction(
            gw,
            key,
            agent.session_id,
            auto_mode="hermes",
            history=_history(),
            approx_tokens=345_000,
            timeout_seconds=30.0,
        )
    )

    assert outcome == "compacted"
    # The no-op is gone: the thread genuinely shrank via the app-server's own
    # mechanism — asserted at the RPC-stub boundary.
    assert agent._codex_session.compact_calls == 1
    # A compaction boundary was recorded on the live agent's compressor.
    assert agent.context_compressor.compression_count == 1
    # The live agent is STILL cached — hygiene must not evict the thread owner.
    assert gw._agent_cache[key][0] is agent
    # Success reset the persisted hygiene failure streak.
    assert db.increment_hygiene_failure_streak(key) == 1  # was cleared to 0


def test_hermes_mode_without_cached_agent_skips_without_local_compression(tmp_path):
    # Detached case: no live agent → no thread → nothing real to compact.
    # The old behavior "compressed" the mirror (a no-op) and then evicted;
    # the new behavior is an honest skip with the transcript untouched.
    gw, _ = _gateway(tmp_path, agent=None)
    history = _history()
    before = [dict(m) for m in history]

    outcome = asyncio.run(
        run_codex_hygiene_compaction(
            gw,
            "tg:123",
            "sess-1",
            auto_mode="hermes",
            history=history,
            approx_tokens=345_000,
            timeout_seconds=5.0,
        )
    )

    assert outcome == "skipped:no-cached-agent"
    assert history == before


# ---------------------------------------------------------------------------
# Mode semantics: native / off never touch the thread nor run local fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["native", "off"])
def test_native_and_off_modes_skip_cleanly(tmp_path, mode):
    agent = LiveCodexAgent(mode=mode)
    key = "tg:123"
    gw, _ = _gateway(tmp_path, key, agent)

    outcome = asyncio.run(
        run_codex_hygiene_compaction(
            gw,
            key,
            agent.session_id,
            auto_mode=mode,
            history=_history(),
            approx_tokens=345_000,
            timeout_seconds=5.0,
        )
    )

    assert outcome == f"skipped:mode={mode}"
    # off must not silently compress; native leaves scheduling to codex.
    assert agent._codex_session.compact_calls == 0
    assert agent.context_compressor.compression_count == 0
    # And the agent stays cached in every skip path.
    assert gw._agent_cache[key][0] is agent


def test_unknown_mode_falls_back_to_native_semantics(tmp_path):
    agent = LiveCodexAgent(mode="banana")
    gw, _ = _gateway(tmp_path, "tg:123", agent)
    outcome = asyncio.run(
        run_codex_hygiene_compaction(
            gw,
            "tg:123",
            agent.session_id,
            auto_mode="banana",
            history=_history(),
            approx_tokens=345_000,
            timeout_seconds=5.0,
        )
    )
    assert outcome == "skipped:mode=native"
    assert agent._codex_session.compact_calls == 0


# ---------------------------------------------------------------------------
# force=True (manual /compress) must never violate the "no local fallback"
# contract: it compacts the THREAD in every mode, and never rewrites the
# transcript mirror.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["native", "hermes", "off"])
def test_force_compacts_thread_never_local_fallback(mode):
    agent = LiveCodexAgent(mode=mode)
    messages = _history()
    before = [dict(m) for m in messages]

    returned, prompt = compress_context(
        agent, messages, "system", approx_tokens=345_000, force=True
    )

    # Thread compaction ran (manual force is an explicit user decision)...
    assert agent._codex_session.compact_calls == 1
    # ...and the local transcript mirror was NOT rewritten in any mode.
    assert returned is messages
    assert returned == before
    assert prompt == "cached prompt"


@pytest.mark.parametrize("mode", ["native", "off"])
def test_force_without_live_thread_does_not_run_local_compressor(mode):
    # The #73715 branch let force=True fall through to the local Hermes
    # compressor in native/off when no thread existed. That is wrong on this
    # runtime in every mode: rewriting the mirror cannot shrink the thread.
    agent = LiveCodexAgent(mode=mode, session=None)
    agent._codex_session = None
    messages = _history()
    before = [dict(m) for m in messages]

    returned, _ = compress_context(
        agent, messages, "system", approx_tokens=345_000, force=True
    )

    assert returned is messages
    assert returned == before
    assert agent.context_compressor.compression_count == 0


# ---------------------------------------------------------------------------
# Failure handling: a wedged compaction records a DB cooldown (real SessionDB)
# ---------------------------------------------------------------------------

def test_timeout_records_persistent_cooldown(tmp_path):
    class HangingSession(FakeCodexSession):
        def compact_thread(self):
            self.compact_calls += 1
            import time

            time.sleep(3.0)
            return self.result

    agent = LiveCodexAgent(mode="hermes", session=HangingSession())
    key = "tg:123"
    gw, db = _gateway(tmp_path, key, agent)
    db.create_session(agent.session_id, "gateway")

    outcome = asyncio.run(
        run_codex_hygiene_compaction(
            gw,
            key,
            agent.session_id,
            auto_mode="hermes",
            history=_history(),
            approx_tokens=345_000,
            timeout_seconds=1.0,
            failure_cooldown_seconds=300.0,
        )
    )

    assert outcome == "failed:timeout"
    state = db.get_compression_failure_cooldown(agent.session_id)
    assert state is not None and state.get("remaining_seconds", 0) > 0
    # Even on failure the live agent stays cached — its thread still holds
    # the only real context.
    assert gw._agent_cache[key][0] is agent


# ---------------------------------------------------------------------------
# Manual /compress gateway surface
# ---------------------------------------------------------------------------

def _slash_host(agent, session_key="tg:123"):
    from gateway.slash_commands import GatewaySlashCommandsMixin

    host = SimpleNamespace(
        _agent_cache={session_key: (agent, 0.0)} if agent is not None else {},
        _agent_cache_lock=None,
    )

    async def _run_in_executor_with_context(fn):
        return await asyncio.get_running_loop().run_in_executor(None, fn)

    host._run_in_executor_with_context = _run_in_executor_with_context
    host._compress_codex_app_server_session = (
        GatewaySlashCommandsMixin._compress_codex_app_server_session.__get__(host)
    )
    return host


def test_manual_compress_routes_to_live_thread():
    agent = LiveCodexAgent(mode="off")  # even 'off': manual is a user decision
    host = _slash_host(agent)

    reply = asyncio.run(
        host._compress_codex_app_server_session("tg:123", agent.session_id)
    )

    assert agent._codex_session.compact_calls == 1
    assert "compacted" in reply


def test_manual_compress_without_live_thread_reports_honestly():
    host = _slash_host(None)
    reply = asyncio.run(
        host._compress_codex_app_server_session("tg:123", "sess-1")
    )
    assert "Nothing to compact" in reply


# ---------------------------------------------------------------------------
# Multiplexed gateway: the hygiene worker must see the caller's ContextVars
# (profile secret scope / HERMES_HOME override). A bare run_in_executor worker
# starts with an EMPTY Context, so get_secret(<PROVIDER>_API_KEY) inside the
# summary path fails closed and every hygiene compaction degrades to a lossy
# truncation (#100849 bundle).
# ---------------------------------------------------------------------------

def test_hygiene_worker_inherits_caller_contextvars(tmp_path):
    import contextvars
    import threading

    marker = contextvars.ContextVar("hygiene_scope_marker", default=None)
    seen = {}

    class ScopeProbeAgent(LiveCodexAgent):
        def _compress_context(self, messages, system_message, **kwargs):
            seen["value"] = marker.get()
            seen["thread"] = threading.current_thread().name
            return super()._compress_context(messages, system_message, **kwargs)

    agent = ScopeProbeAgent(mode="hermes")
    key = "tg:ctx"
    gw, _db = _gateway(tmp_path, key, agent)

    async def _scoped():
        token = marker.set("profile-scope")
        try:
            return await run_codex_hygiene_compaction(
                gw, key, agent.session_id, auto_mode="hermes",
                history=_history(), approx_tokens=345_000, timeout_seconds=30.0,
            )
        finally:
            marker.reset(token)

    assert asyncio.run(_scoped()) == "compacted"
    assert seen["thread"] != "MainThread", "compaction must still run off-loop"
    assert seen["value"] == "profile-scope", (
        "hygiene worker lost the caller's ContextVars — under multiplex_profiles "
        "this is the UnscopedSecretError / lossy-truncation regression"
    )
