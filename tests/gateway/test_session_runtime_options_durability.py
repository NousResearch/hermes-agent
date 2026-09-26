"""Durable-first session runtime options for /model, /reasoning and /fast (#92185, PR #92187).

F1: a slash write used to move live state first and swallow a failed save, so the process ran one
configuration while a restart brought back another. Every session write now persists first and
assigns live state from the write's own completion; a failed save raises with nothing changed.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from gateway.session_state import SERVICE_TIER_UNSET

MODEL_X = {
    "model": "gpt-5.5", "provider": "openai", "api_key": "sk-live-only",
    "base_url": "https://api.openai.example/v1", "api_mode": "responses",
}


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1")


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source())


def _no_db(**_kw):
    raise RuntimeError("SQLite disabled in test")


@pytest.fixture
def home(tmp_path, monkeypatch):
    import hermes_state

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-5.4\n  provider: openai\nagent:\n  reasoning_effort: medium\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(hermes_state, "SessionDB", _no_db)
    return tmp_path


def _store(tmp_path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


def _runner(store: SessionStore):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_db = None
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_model_notes = {}
    runner._reasoning_config = None
    runner._service_tier = None
    runner._show_reasoning = False
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    return runner


def _switch_result(model="gpt-5.5"):
    return SimpleNamespace(
        new_model=model, target_provider="openai", provider_label="OpenAI", api_key="sk-live-only",
        base_url="https://api.openai.example/v1", api_mode="responses", request_overrides={},
        runtime_capabilities={},
    )


def _switch_ctx(session_key):
    from gateway.slash_commands_model import _ModelSwitchContext

    return _ModelSwitchContext(
        session_key=session_key, source=_source(), config_path=None, persist_global=False,
        current_model="gpt-5.4", current_provider="openai",
    )


def _durable(tmp_path, session_key):
    """What a restarted gateway reads back: (model, reasoning, tier) from a fresh store."""
    entry = _store(tmp_path).lookup_by_session_key(session_key)
    return (
        (entry.model_override or {}).get("model"),
        entry.reasoning_override,
        getattr(entry, "service_tier_override", None),
    )


def _live(runner, session_key):
    conv = runner._session_state(session_key).conversation
    model = (conv.model_override or {}).get("model")
    tier = conv.service_tier_override
    tier = None if tier is SERVICE_TIER_UNSET else ("normal" if tier is None else tier)
    return model, conv.reasoning_override, tier


class _ParkedWrites:
    """Park ``set_runtime_options`` on its worker thread until released."""

    def __init__(self, store):
        self.started = threading.Event()
        self.release = threading.Event()
        real = store.set_runtime_options

        def _parked(*args, **kwargs):
            self.started.set()
            assert self.release.wait(10), "parked write was never released"
            return real(*args, **kwargs)

        store.set_runtime_options = _parked

    async def wait_started(self):
        assert await asyncio.to_thread(self.started.wait, 5), "no durable write was submitted"


# -- C3-1 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["reasoning", "fast_cold", "model"])
async def test_slash_write_failure_leaves_state_untouched(home, monkeypatch, surface):
    store = _store(home)
    source = _source()
    session_key = store.get_or_create_session(source).session_key
    runner = _runner(store)
    # A committed starting point, so memory == disk before the failing write.
    await runner._handle_reasoning_command(_event("/reasoning low"))
    before_live = _live(runner, session_key)
    before_runner_wide = (runner._reasoning_config, runner._service_tier)
    assert before_live == _durable(home, session_key)
    # A /moa or /model --once armed for the next turn must survive the failed write.
    restore = {"had_override": False, "override": None}
    runner._session_state(session_key).conversation.one_turn_restore = dict(restore)

    def _disk_full(_data):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(store, "_save_sessions_json", _disk_full)
        if surface == "reasoning":
            await runner._handle_reasoning_command(_event("/reasoning high"))
        elif surface == "fast_cold":
            await runner._handle_fast_command(_event("/fast cold"))
        else:
            await runner._record_model_switch(
                _switch_result(), _switch_ctx(session_key), source=source, one_turn=False, picker=False)

    assert _live(runner, session_key) == before_live
    # The runner-wide copies used for display and the next turn follow only a durable write.
    assert (runner._reasoning_config, runner._service_tier) == before_runner_wide
    assert session_key not in runner._pending_model_notes
    assert runner._session_state(session_key).conversation.one_turn_restore == restore
    assert _durable(home, session_key) == before_live


# -- C3-2 -------------------------------------------------------------------------------------


def test_runner_rehydrates_all_runtime_options_live_wins(home):
    store = _store(home)
    session_key = store.get_or_create_session(_source()).session_key
    assert store.set_runtime_options(
        session_key, model_override=MODEL_X, reasoning_override={"enabled": True, "effort": "high"},
        service_tier_override="cold",
    )

    # Restart: fresh store, fresh runner. A live value set before the first read wins.
    runner = _runner(_store(home))
    runner._session_state(session_key).conversation.reasoning_override = {"enabled": False}
    with patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
               return_value={"api_key": "sk-fresh", "api_mode": "responses"}):
        assert runner._resolve_session_service_tier(session_key=session_key) == "cold"
    conv = runner._session_state(session_key).conversation
    assert conv.model_override["model"] == "gpt-5.5"
    assert conv.model_override["api_key"] == "sk-fresh"  # credentials re-resolved, never read back
    assert runner._resolve_session_reasoning_config(session_key=session_key) == {"enabled": False}

    # A failed read is retried: the once-per-process flag is set only after a good read.
    flaky = _runner(_store(home))
    calls = {"n": 0}
    real = flaky.session_store.get_runtime_options

    def _once_broken(key):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return real(key)

    flaky.session_store.get_runtime_options = _once_broken
    flaky.session_store.get_model_override = lambda _key: None
    assert flaky._resolve_session_service_tier(session_key=session_key) is None  # config default
    assert flaky._resolve_session_service_tier(session_key=session_key) == "cold"
    assert flaky._resolve_session_reasoning_config(session_key=session_key) == {
        "enabled": True, "effort": "high"}


# -- C3-3 -------------------------------------------------------------------------------------


def _drive_api_model_pick(runner, monkeypatch):
    """The structured host API (#92185) picking gpt-5.5, with model resolution stubbed."""
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model=kw["raw_input"], target_provider="openai", api_key="sk-live-only",
        api_mode="responses"))
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning", lambda *a, **k: None)
    runner._resolve_session_agent_runtime = lambda **_kw: ("gpt-5.4", {"provider": "openai"})
    return runner.apply_session_options(_source(), {"model": "gpt-5.5"})


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["typed", "picker", "api"])
async def test_model_commit_pops_armed_one_shot(home, monkeypatch, surface):
    store = _store(home)
    source = _source()
    session_key = store.get_or_create_session(source).session_key
    runner = _runner(store)

    # 1) A durable model pick supersedes an armed /moa one-shot on every surface: otherwise the
    #    post-turn restore reverts memory to the pre-moa model while disk holds the pick.
    runner._claim_one_turn_restore(session_key)
    runner._session_state(session_key).conversation.model_override = {"provider": "moa", "model": "default"}
    if surface == "api":
        result = await _drive_api_model_pick(runner, monkeypatch)
        assert result["status"] == "accepted", result
    else:
        await runner._record_model_switch(
            _switch_result(), _switch_ctx(session_key), source=source, one_turn=False,
            picker=surface == "picker")
    conv = runner._session_state(session_key).conversation
    assert conv.one_turn_restore is None
    assert _live(runner, session_key)[0] == _durable(home, session_key)[0] == "gpt-5.5"

    # 2) A /moa armed while a reasoning commit is parked in settle survives it: the reasoning
    #    write names no model, so neither disk nor the restore is touched.
    parked = _ParkedWrites(store)
    task = asyncio.create_task(runner._handle_reasoning_command(_event("/reasoning high")))
    await parked.wait_started()
    runner._claim_one_turn_restore(session_key)
    moa = {"provider": "moa", "model": "default", "base_url": "moa://local"}
    conv.model_override = dict(moa)
    parked.release.set()
    await task

    assert conv.model_override == moa
    assert conv.one_turn_restore is not None
    assert _durable(home, session_key)[:2] == ("gpt-5.5", {"enabled": True, "effort": "high"})
    runner._restore_session_model_override(session_key, conv.one_turn_restore)
    assert _live(runner, session_key) == _durable(home, session_key)


# -- C3-4 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["clear_before_write", "reset_before_write"])
async def test_boundary_during_parked_commit_leaks_nothing(home, order):
    """/new while a commit is parked: /new clears the conversation scope, then publishes a fresh
    entry (reset_session). Whichever side the write lands on, the value reaches neither memory nor
    the new entry. The commit here read the conversation before the clear; the real-/new test
    below covers a commit that reads it between the clear and the fresh entry."""
    from gateway.run_session_options import SessionMissing

    store = _store(home)
    session_key = store.get_or_create_session(_source()).session_key
    runner = _runner(store)
    parked = _ParkedWrites(store)
    task = asyncio.create_task(runner._handle_reasoning_command(_event("/reasoning xhigh")))
    await parked.wait_started()

    runner._clear_conversation_scope(session_key, reason="session_reset")
    if order == "reset_before_write":
        await asyncio.to_thread(store.reset_session, session_key)
    parked.release.set()
    with pytest.raises(SessionMissing):
        await task
    if order == "clear_before_write":
        await asyncio.to_thread(store.reset_session, session_key)

    assert runner._session_state(session_key).conversation.reasoning_override is None
    assert store.lookup_by_session_key(session_key).reasoning_override is None
    assert _durable(home, session_key)[1] is None


@pytest.mark.asyncio
async def test_new_command_and_parked_commit_leak_nothing(home):
    """The real /new against a commit parked while it resolves the route. Without one step for
    "clear, then publish a fresh entry", the commit can read the cleared conversation, write to
    the old entry, assign live, and then lose its entry to the reset: memory holds the value for
    the new conversation while disk does not."""
    store = _store(home)
    source = _source()
    session_key = store.get_or_create_session(source).session_key
    runner = _runner(store)
    runner.adapters = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._cleanup_old_agent_for_reset = AsyncMock()
    runner._fire_session_reset_hooks = AsyncMock()
    runner._reset_notice_session_info = lambda _source: ""
    runner._telegram_topic_new_header = lambda _source: None
    runner._is_telegram_topic_lane = lambda _source: False
    route_started, route_release = threading.Event(), threading.Event()
    reset_release = threading.Event()
    real_route, real_reset = store.get_or_create_session, store.reset_session

    def _parked_route(*args, **kwargs):
        route_started.set()
        assert route_release.wait(10)
        return real_route(*args, **kwargs)

    def _parked_reset(*args, **kwargs):
        assert reset_release.wait(10)
        return real_reset(*args, **kwargs)

    store.get_or_create_session = _parked_route
    store.reset_session = _parked_reset
    commit = asyncio.create_task(
        runner._commit_session_runtime_options(source, {"reasoning_override": {"enabled": True, "effort": "max"}}))
    reset = None
    try:
        assert await asyncio.to_thread(route_started.wait, 5)
        reset = asyncio.create_task(runner._handle_reset_command(_event("/new")))
        await asyncio.sleep(0.05)  # /new runs as far as it can while the commit is parked
        route_release.set()
        await asyncio.wait_for(commit, 5)
    finally:
        route_release.set()
        reset_release.set()
    if reset is not None:
        await asyncio.wait_for(reset, 5)

    assert store.lookup_by_session_key(session_key).reasoning_override is None
    assert _durable(home, session_key)[1] is None
    assert runner._session_state(session_key).conversation.reasoning_override is None
    assert runner._resolve_session_reasoning_config(session_key=session_key) != {"enabled": True, "effort": "max"}


# -- C3-5 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_write_on_expired_route_survives_next_message_and_restart(home):
    """A /reasoning on a suspended (or idle-expired) route used to be wiped by the next message:
    the command never resolved the route, so the boundary fired on the NEXT message and cleared
    the override it had just set. The commit resolves the route and consumes the boundary first."""
    store = _store(home)
    source = _source()
    old = store.get_or_create_session(source)
    session_key = old.session_key
    runner = _runner(store)
    runner._session_state(session_key).conversation.reasoning_override = {"enabled": False}  # old convo
    store.suspend_session(session_key)

    await runner._handle_reasoning_command(_event("/reasoning high"))

    # The next message resolves the route and runs the boundary handling.
    entry = await runner.async_session_store.get_or_create_session(source)
    await runner._hmwa_open_session(entry, session_key, source)
    assert entry.session_id != old.session_id
    assert runner._resolve_session_reasoning_config(session_key=session_key) == {
        "enabled": True, "effort": "high"}
    assert _durable(home, session_key)[1] == {"enabled": True, "effort": "high"}
