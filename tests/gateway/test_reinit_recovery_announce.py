"""Recovery-announce architecture — three announce sites, one shared predicate.

Restructured from #238 (which emitted recovery at the end-of-turn persist site,
keyed on the FINAL served route — invisible for a restore that re-failed-over
mid-turn, e.g. the refusing-/model-pin case, SPEC 2026-07-08 prologue-recovery):

  • failover        → mid-turn switch site (untouched here)
  • restore leg     → INLINE in restore_primary_runtime (rider "restore")
  • re-init snap-back → PRE-RUN gateway site _announce_reinit_recovery
                        (rider "re-init")
  • end-of-turn _announce_and_persist_served_route → PERSIST-ONLY

All gates route through agent.agent_runtime_helpers.recovery_should_announce
(Momus MB-1: ONE predicate, no drift).

Covers SPEC ACs: AC-1a/1b (restore + re-init legs under a refusing pin),
AC-2 (durable sink), AC-3 (manual-dance suppression), AC-4 (chat gate off →
sink still written), AC-5 (no prior route → silent), AC-8 (shared predicate,
both sites in lockstep) + persist-only invariants of the end-of-turn site.
"""

import threading
import types
from datetime import datetime, timezone

import pytest

import gateway.run as gateway_run
from agent.agent_runtime_helpers import recovery_should_announce
from gateway.session import SessionEntry, SessionStore


def _entry(session_key="agent:main:discord:c1:c1", session_id="s1", last_served=None,
           override=None):
    now = datetime.now(timezone.utc)
    e = SessionEntry(session_key=session_key, session_id=session_id, created_at=now, updated_at=now)
    e.last_served_identity = last_served
    e.model_override_identity = override
    return e


def _store(tmp_path):
    store = object.__new__(SessionStore)
    store._entries = {}
    store.sessions_dir = tmp_path
    store._lock = threading.RLock()
    store._loaded = True
    store._record_gateway_session_peer = lambda *a, **k: None
    return store


def _runner(store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = store
    return runner


def _agent(provider=None, model=None):
    a = types.SimpleNamespace()
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)
    a._last_fallback_announced = None
    if provider is not None:
        a.provider = provider
    if model is not None:
        a.model = model
    return a


def _sink_lines(home):
    import pathlib
    p = pathlib.Path(home) / "state" / "model-route-changes.log"
    if not p.exists():
        return []
    return [ln for ln in p.read_text().splitlines() if ln.strip()]


def _reinit(runner, agent, key, provider, model):
    runner._announce_reinit_recovery(
        agent=agent,
        session_key=key,
        applied_provider=provider,
        applied_model=model,
    )


def _persist(runner, agent, key, provider, model, was_reinit=False):
    runner._announce_and_persist_served_route(
        agent=agent,
        session_key=key,
        served_provider=provider,
        served_model=model,
        was_reinit=was_reinit,
    )


OPUS = ("claude-apx-3", "claude-opus-4-8")
FABLE = ("claude-apx-5", "claude-fable-5")
PIN = {"provider": FABLE[0], "model": FABLE[1]}
SERVED_OPUS = {"provider": OPUS[0], "model": OPUS[1]}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
    store = _store(tmp_path)
    runner = _runner(store)
    return tmp_path, store, runner


class TestSharedPredicate:
    """AC-8: the ONE decision function, unit-tested directly."""

    def test_announces_restore_to_standing_pin(self):
        # The refusing-pin case: restore opus→fable on a STANDING pin announces.
        assert recovery_should_announce(
            OPUS, FABLE, override_target=FABLE, override_target_changed=False,
        )

    def test_suppresses_manual_retarget(self):
        # /model fable issued THIS turn: candidate==fresh target → user-driven.
        assert not recovery_should_announce(
            OPUS, FABLE, override_target=FABLE, override_target_changed=True,
        )

    def test_suppresses_manual_clear(self):
        # /model reset this turn: snap to config default is user-driven.
        assert not recovery_should_announce(
            OPUS, FABLE, override_target=None, override_target_changed=True,
        )

    def test_same_route_silent(self):
        assert not recovery_should_announce(FABLE, FABLE)

    def test_unknown_prev_silent(self):
        assert not recovery_should_announce((None, None), FABLE)
        assert not recovery_should_announce("junk", FABLE)  # type: ignore[arg-type]

    def test_no_override_announces(self):
        assert recovery_should_announce(OPUS, FABLE)


class TestReinitPreRunAnnounce:
    """AC-1b: the re-init snap-back leg at the PRE-RUN site."""

    def test_reinit_snapback_announces_under_standing_pin(self, env):
        # AC-1b core: pinned fable, prior turn ended on opus (refusal),
        # cache evicted → fresh agent starts on fable → announce, even though
        # this turn will later refuse back to opus.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS), override=dict(PIN))
        agent = _agent()
        _reinit(runner, agent, key, *FABLE)
        assert len(agent._announced) == 1
        line = agent._announced[0]
        assert line.startswith("🔄 Model recovery (re-init): ")
        assert "claude-apx-3/claude-opus-4-8" in line and "claude-apx-5/claude-fable-5" in line
        # AC-2: durable sink line written.
        lines = _sink_lines(str(tmp_path))
        assert len(lines) == 1 and " recovery " in lines[0]
        # INV-5: pre-run site does NOT persist.
        assert store._entries[key].last_served_identity == SERVED_OPUS

    def test_reinit_announces_without_any_override(self, env):
        # The original #238 re-init case (no pin) still works at the new site.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS))
        agent = _agent()
        _reinit(runner, agent, key, "yunwu", "claude-fable-5")
        assert len(agent._announced) == 1
        assert agent._announced[0].startswith("🔄 Model recovery (re-init): ")

    def test_manual_retarget_is_suppressed_and_stamp_consumed_once(self, env):
        # AC-3: /model fable issued → stamp set → suppressed exactly ONE turn.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS), override=dict(PIN))
        runner._override_target_just_changed = {key: True}
        agent = _agent()
        _reinit(runner, agent, key, *FABLE)
        assert agent._announced == []          # the user's own switch: silent
        assert _sink_lines(str(tmp_path)) == []  # no misleading audit line
        assert key not in runner._override_target_just_changed  # consumed
        # Next re-init (stamp gone) announces again — stale stamps can't mask.
        agent2 = _agent()
        _reinit(runner, agent2, key, *FABLE)
        assert len(agent2._announced) == 1

    def test_manual_clear_is_suppressed(self, env):
        # AC-3b: /model reset (override → None) + snap to default: silent.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS), override=None)
        runner._override_target_just_changed = {key: True}
        agent = _agent()
        _reinit(runner, agent, key, "yunwu", "claude-fable-5")
        assert agent._announced == []

    def test_same_model_reinit_is_silent(self, env):
        # AC: a re-init that lands on the same route is a non-event.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served={"provider": "yunwu", "model": "claude-fable-5"})
        agent = _agent()
        _reinit(runner, agent, key, "yunwu", "claude-fable-5")
        assert agent._announced == []
        assert _sink_lines(str(tmp_path)) == []

    def test_no_prior_served_route_is_silent(self, env):
        # AC-5: first turn ever — nothing to recover from.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=None)
        agent = _agent()
        _reinit(runner, agent, key, "yunwu", "claude-fable-5")
        assert agent._announced == []

    def test_sink_written_when_chat_gate_off(self, env):
        # AC-4: announce_recovery=false → no chat emit, sink line still lands.
        tmp_path, store, runner = env
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: false\n")
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS), override=dict(PIN))
        agent = _agent()
        _reinit(runner, agent, key, *FABLE)
        assert agent._announced == []
        lines = _sink_lines(str(tmp_path))
        assert len(lines) == 1 and " recovery " in lines[0]

    def test_missing_agent_or_key_is_noop(self, env):
        tmp_path, store, runner = env
        agent = _agent()
        _reinit(runner, agent, "", "yunwu", "claude-fable-5")
        _reinit(runner, None, "agent:main:discord:c1:c1", "yunwu", "claude-fable-5")
        assert agent._announced == []

    def test_unknown_session_is_noop(self, env):
        tmp_path, store, runner = env
        agent = _agent()
        _reinit(runner, agent, "agent:main:discord:missing:missing", "yunwu", "claude-fable-5")
        assert agent._announced == []


class TestRestoreInlineAnnounce:
    """AC-1a: the restore leg fires INLINE in restore_primary_runtime — at the
    only moment the restored route exists when the turn later re-fails-over."""

    def _restorable_agent(self, tmp_path):
        # A minimal real-ish agent that exercises the actual restore path.
        from agent.agent_runtime_helpers import restore_primary_runtime  # noqa: F401
        a = types.SimpleNamespace()
        a._announced = []
        a._emit_status = lambda m: a._announced.append(m)
        a._last_fallback_announced = None
        a._fallback_activated = True
        a._fallback_index = 2
        a._rate_limited_until = 0
        # Currently ON the fallback (opus); primary snapshot is the pin (fable).
        a.provider, a.model = OPUS
        a.base_url = "http://x"
        a.api_mode = "chat_completions"
        a.api_key = "k"
        a._client_kwargs = {}
        a._use_prompt_caching = False
        a._use_native_cache_layout = False
        a._transport_cache = {}
        a._credential_pool = None
        a._create_openai_client = lambda *args, **kw: object()
        cc = types.SimpleNamespace(
            update_model=lambda **kw: None, model=FABLE[1], base_url="http://x",
            api_key="k", provider=FABLE[0], context_length=100000, threshold_tokens=50000,
        )
        a.context_compressor = cc
        a._primary_runtime = {
            "model": FABLE[1], "provider": FABLE[0], "base_url": "http://x",
            "api_mode": "chat_completions", "api_key": "k", "client_kwargs": {},
            "use_prompt_caching": False, "use_native_cache_layout": False,
            "compressor_model": FABLE[1], "compressor_base_url": "http://x",
            "compressor_api_key": "k", "compressor_provider": FABLE[0],
            "compressor_context_length": 100000, "compressor_threshold_tokens": 50000,
        }
        return a

    def test_restore_emits_inline_with_restore_rider(self, tmp_path, monkeypatch):
        # AC-1a: cache-warm turn under the refusing pin — the opus→fable restore
        # announces AT THE RESTORE, independent of how the turn ends.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        from agent.agent_runtime_helpers import restore_primary_runtime
        # rewrite_prompt_model_identity needs a real-enough message list; stub it.
        import agent.chat_completion_helpers as cch
        monkeypatch.setattr(cch, "rewrite_prompt_model_identity", lambda *a, **k: None)
        agent = self._restorable_agent(tmp_path)
        assert restore_primary_runtime(agent) is True
        assert agent.model == FABLE[1] and agent.provider == FABLE[0]
        assert len(agent._announced) == 1
        line = agent._announced[0]
        assert line.startswith("🔄 Model recovery (restore): ")
        assert "claude-apx-3/claude-opus-4-8" in line and "claude-apx-5/claude-fable-5" in line
        # AC-2: durable sink line for the restore leg.
        lines = _sink_lines(str(tmp_path))
        assert len(lines) == 1 and " recovery " in lines[0]
        # Per-turn marker set (end-of-turn parity/observability).
        assert agent._recovery_emitted_this_turn == (OPUS, FABLE)

    def test_restore_gate_off_still_writes_sink(self, tmp_path, monkeypatch):
        # AC-4 on the restore leg.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: false\n")
        from agent.agent_runtime_helpers import restore_primary_runtime
        import agent.chat_completion_helpers as cch
        monkeypatch.setattr(cch, "rewrite_prompt_model_identity", lambda *a, **k: None)
        agent = self._restorable_agent(tmp_path)
        assert restore_primary_runtime(agent) is True
        assert agent._announced == []
        assert len(_sink_lines(str(tmp_path))) == 1

    def test_no_fallback_active_no_emit(self, tmp_path, monkeypatch):
        # A turn with no fallback to restore: first-guard return, no announce.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        from agent.agent_runtime_helpers import restore_primary_runtime
        agent = self._restorable_agent(tmp_path)
        agent._fallback_activated = False
        assert restore_primary_runtime(agent) is False
        assert agent._announced == []
        assert _sink_lines(str(tmp_path)) == []

    def test_restore_emit_failure_never_breaks_restore(self, tmp_path, monkeypatch):
        # INV-3: a broken sink/emit must not break the restore itself.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        from agent.agent_runtime_helpers import restore_primary_runtime
        import agent.chat_completion_helpers as cch
        monkeypatch.setattr(cch, "rewrite_prompt_model_identity", lambda *a, **k: None)

        def _boom(*a, **k):
            raise RuntimeError("sink down")

        monkeypatch.setattr(cch, "_append_route_change", _boom)
        agent = self._restorable_agent(tmp_path)
        assert restore_primary_runtime(agent) is True  # restore still succeeds
        assert agent.model == FABLE[1]


class TestPersistOnlyEndOfTurn:
    """The end-of-turn site is persist-only — no announce, ever (the old
    end-of-turn recovery emit produced a spurious/backwards line when the
    FAILOVER changed the final route)."""

    def test_persists_served_route(self, env):
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS))
        agent = _agent()
        _persist(runner, agent, key, "yunwu", "claude-fable-5")
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_never_announces_even_on_route_change(self, env):
        # Mutation-guard for the restructure: end-of-turn route change (e.g. a
        # failover changed the final route) does NOT emit and does NOT sink.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served={"provider": "claude-apx-5", "model": "claude-fable-5"})
        agent = _agent()
        _persist(runner, agent, key, *OPUS, was_reinit=True)
        assert agent._announced == []
        assert _sink_lines(str(tmp_path)) == []
        assert store._entries[key].last_served_identity == SERVED_OPUS

    def test_missing_bits_noop(self, env):
        tmp_path, store, runner = env
        agent = _agent()
        _persist(runner, agent, "", "p", "m")
        _persist(runner, None, "agent:main:discord:c1:c1", "p", "m")
        assert agent._announced == []


class TestManualSwitchStamp:
    """The _set_session_model_override door stamps the one-turn suppression."""

    def test_set_override_stamps(self, env):
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        runner.session_store.entry_for = lambda k: store._entries.get(k)
        runner.session_store.persist = lambda: None
        runner._model_override_is_persistable = lambda o: None
        runner._set_session_model_override(key, {"model": FABLE[1], "provider": FABLE[0]})
        assert runner._override_target_just_changed.get(key) is True

    def test_clear_override_stamps(self, env):
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        runner.session_store.entry_for = lambda k: store._entries.get(k)
        runner.session_store.persist = lambda: None
        runner._session_model_overrides = {key: {"model": "x"}}
        runner._set_session_model_override(key, None)
        assert runner._override_target_just_changed.get(key) is True

    def test_stamp_cleared_at_end_of_every_turn(self, env):
        # Greptile #249 P2: the stamp must not outlive its one-turn window even
        # if the following turns land on a CACHED agent (pre-run site skipped).
        # The end-of-turn persist site pops it unconditionally.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS))
        runner._override_target_just_changed = {key: True}
        agent = _agent()
        # A cached-agent turn: only the end-of-turn persist runs (no pre-run pop).
        _persist(runner, agent, key, *OPUS)
        assert key not in runner._override_target_just_changed  # stamp cleared

    def test_lingering_stamp_cannot_suppress_a_later_recovery(self, env):
        # End-to-end of the P2 concern: switch turn stamps → a cached turn ends
        # and clears the stamp → a LATER fresh-agent recovery is NOT suppressed.
        tmp_path, store, runner = env
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=dict(SERVED_OPUS), override=dict(PIN))
        runner._override_target_just_changed = {key: True}
        # Cached turn clears the stamp at end-of-turn.
        _persist(runner, _agent(), key, *OPUS)
        # Later fresh-agent turn: genuine refusing-pin recovery announces.
        agent2 = _agent()
        _reinit(runner, agent2, key, *FABLE)
        assert len(agent2._announced) == 1
