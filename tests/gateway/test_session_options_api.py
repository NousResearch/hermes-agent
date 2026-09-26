"""``GatewayRunner.apply_session_options``: silent per-session model / reasoning / fast (#92185).

Hosts (e.g. OcuClaw's G2 adapter, ``sessions.options.apply``) drive a session's runtime options
from their own UI instead of injecting visible slash commands. The whole patch is validated
before any state moves, the write is the slash commands' durable-first commit, and every outcome
is a structured result.
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource, SessionStore
from gateway.session_state import SERVICE_TIER_UNSET
from hermes_cli.model_switch import ModelSwitchResult

CONFIG_MODEL = "gpt-5.4"


def _no_db(**_kw):
    raise RuntimeError("SQLite disabled in test")


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")


def _event(text="hello") -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source())


def _switch(**kw):
    model = kw["raw_input"]
    if model == "bad-model":
        return ModelSwitchResult(success=False, error_message="model not found: bad-model")
    return ModelSwitchResult(
        success=True, new_model=model, target_provider=kw.get("explicit_provider") or "openai",
        api_key="sk-SECRET-live-only", base_url="", api_mode="responses", provider_label="OpenAI",
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    import hermes_state

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"model:\n  default: {CONFIG_MODEL}\n  provider: openai\n", encoding="utf-8")
    monkeypatch.setattr(hermes_state, "SessionDB", _no_db)
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda **_kw: {"model": {"default": CONFIG_MODEL, "provider": "openai"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: CONFIG_MODEL)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway", lambda *a, **k: None)
    guard = {"warning": None}
    monkeypatch.setattr(
        "hermes_cli.model_selection_guards.combined_selection_warning",
        lambda *a, **k: guard["warning"])
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    return SimpleNamespace(tmp_path=tmp_path, store=store, guard=guard)


def _runner(store):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._draining = False
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.delivery_router = MagicMock()

    def _runtime(*, source=None, session_key=None, user_config=None):
        override = runner._session_state(session_key).conversation.model_override or {}
        return override.get("model") or CONFIG_MODEL, {"provider": override.get("provider") or "openai"}

    runner._resolve_session_agent_runtime = _runtime
    return runner


def _snapshot(env, runner, key):
    """(live, durable-after-restart, queued note) for the three options, both in the store's
    encoding (tier: None = inherit, "normal" = explicit normal)."""
    conv = runner._session_state(key).conversation
    tier = conv.service_tier_override
    live = ((conv.model_override or {}).get("model"), conv.reasoning_override,
            None if tier is SERVICE_TIER_UNSET else ("normal" if tier is None else tier))
    entry = SessionStore(sessions_dir=env.tmp_path / "sessions", config=GatewayConfig()).lookup_by_session_key(key)
    durable = ((entry.model_override or {}).get("model"), entry.reasoning_override, entry.service_tier_override)
    return live, durable, runner._pending_model_notes.get(key)


class _ParkedWrites:
    """Park ``set_runtime_options`` on its worker thread until released (optionally failing)."""

    def __init__(self, store, fail=None):
        self.started = threading.Event()
        self.release = threading.Event()
        real = store.set_runtime_options

        def _parked(*args, **kwargs):
            self.started.set()
            assert self.release.wait(10), "parked write was never released"
            if fail is not None:
                raise fail
            return real(*args, **kwargs)

        store.set_runtime_options = _parked

    async def wait_started(self):
        assert await asyncio.to_thread(self.started.wait, 5), "no durable write was submitted"


# -- C5-1 -------------------------------------------------------------------------------------


def _row_rejected_route(env, runner, source):
    source.profile_route_rejected = True
    runner._canonicalize = lambda _source, **_kw: None


def _row_busy(env, runner, source):
    runner._session_state(runner._session_key_for_source(source)).turn.agent = _AGENT_PENDING_SENTINEL


def _row_confirmation(env, runner, source):
    env.guard["warning"] = SimpleNamespace(title="Expensive model", message="gpt-5.5 costs 10x more.")


def _row_disk_full(env, runner, source):
    def _disk_full(_data):
        raise OSError(28, "No space left on device")
    env.store._save_sessions_json = _disk_full


def _row_entry_replaced(env, runner, source):
    real = env.store.set_runtime_options

    def _replaced_first(key, **kwargs):
        env.store.reset_session(key)  # a boundary publishes a fresh entry before the write
        return real(key, **kwargs)
    env.store.set_runtime_options = _replaced_first


_HOSTED_ROOM = {"entered": 0}


@contextmanager
def _launch_scope():
    _HOSTED_ROOM["entered"] += 1
    yield


def _row_hosted_room(env, runner, source):
    # A standalone gateway after a hosted room flipped the process-wide credential guard: the API
    # must resolve under the launch profile's scope (#112878), like a turn would.
    _HOSTED_ROOM["entered"] = 0
    runner._standalone_launch_scope = _launch_scope


_ROWS = [
    ("unknown_key", {"temperature": 0.2}, None, "rejected", "invalid_options"),
    ("provider_without_model", {"provider": "openai"}, None, "rejected", "invalid_options"),
    ("rejected_route", {"reasoning_effort": "high"}, _row_rejected_route, "rejected", "invalid_session"),
    ("busy", {"reasoning_effort": "high"}, _row_busy, "rejected", "session_busy"),
    ("bad_model", {"model": "bad-model", "reasoning_effort": "high"}, None, "rejected", "model_rejected"),
    ("bad_effort", {"model": "gpt-5.5", "reasoning_effort": "turbo"}, None, "rejected", "reasoning_rejected"),
    ("fast_not_bool", {"reasoning_effort": "high", "fast": "yes"}, None, "rejected", "fast_rejected"),
    ("fast_unsupported", {"model": "tiny-local", "reasoning_effort": "high", "fast": True}, None,
     "rejected", "fast_unsupported"),
    ("confirmation", {"model": "gpt-5.5", "reasoning_effort": "high"}, _row_confirmation,
     "confirmation_required", "model_confirmation_required"),
    ("disk_full", {"model": "gpt-5.5", "reasoning_effort": "high"}, _row_disk_full,
     "rejected", "durable_write_failed"),
    ("entry_replaced", {"reasoning_effort": "high"}, _row_entry_replaced, "rejected", "session_missing"),
    ("hosted_room_scope", {"model": "gpt-5.5"}, _row_hosted_room, "accepted", None),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _ROWS, ids=[r[0] for r in _ROWS])
async def test_rejections_persist_nothing(env, row):
    name, options, arrange, status, code = row
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    before = _snapshot(env, runner, key)
    if arrange is not None:
        arrange(env, runner, source)

    result = await runner.apply_session_options(source, options)

    assert result["status"] == status, result
    if status == "accepted":  # the hosted-room row: resolved and committed under the launch scope
        assert _HOSTED_ROOM["entered"] == 1
        assert _snapshot(env, runner, key)[1][0] == "gpt-5.5"
        return
    assert result["code"] == code
    assert isinstance(result.get("error"), str) and result["error"]
    if code == "session_missing":
        live, durable, note = _snapshot(env, runner, key)
        assert (live, durable, note) == ((None, None, None), (None, None, None), None)
    else:
        assert _snapshot(env, runner, key) == before


# -- C5-2 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepts_without_a_turn_and_is_idempotent(env):
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    writes = []
    real = env.store.set_runtime_options
    env.store.set_runtime_options = lambda k, **kw: writes.append(kw) or real(k, **kw)

    with patch.object(GatewayRunner, "_handle_message_with_agent", AsyncMock()) as turn:
        result = await runner.apply_session_options(
            source, {"model": "gpt-5.5", "provider": "openai", "reasoning_effort": "high", "fast": True,
                     "confirm_model_selection": False, "initial": False})
        turn.assert_not_called()  # silent: no chat turn
    assert result["status"] == "accepted"
    assert result["applied"] == ["model", "reasoning_effort", "fast"]
    assert result["effective"] == {"model": "gpt-5.5", "provider": "openai", "reasoning_effort": "high",
                                   "fast": True}
    live, durable, note = _snapshot(env, runner, key)
    assert live == (("gpt-5.5", {"enabled": True, "effort": "high"}, "priority"))
    assert durable == ("gpt-5.5", {"enabled": True, "effort": "high"}, "priority")
    assert "gpt-5.5" in note
    assert len(writes) == 1  # one store write for all three

    # Re-asserting the same values: no write, no eviction, no "switched from X to X" note.
    runner._pending_model_notes.clear()
    again = await runner.apply_session_options(
        source, {"model": "gpt-5.5", "provider": "openai", "reasoning_effort": "high", "fast": True})
    assert again["status"] == "accepted" and again["applied"] == []
    assert len(writes) == 1 and key not in runner._pending_model_notes

    # "" means inherit: the overrides are cleared, live and on disk.
    cleared = await runner.apply_session_options(source, {"model": "", "reasoning_effort": ""})
    assert cleared["applied"] == ["model", "reasoning_effort"]
    assert cleared["effective"]["model"] == CONFIG_MODEL
    live, durable, _ = _snapshot(env, runner, key)
    assert live[:2] == (None, None) and durable[:2] == (None, None)

    # initial: a host restoring its saved choice at session start queues no note.
    runner._pending_model_notes.clear()
    restored = await runner.apply_session_options(source, {"model": "gpt-5.5", "initial": True})
    assert restored["applied"] == ["model"] and key not in runner._pending_model_notes


@pytest.mark.asyncio
async def test_model_mirror_writes_the_whole_route_to_the_session_row(env):
    """The dashboard mirror passes base_url/api_mode: update_session_model deletes a route key
    it is not given, so a model-only write would strand the new model on no endpoint."""
    source = _source()
    env.store.get_or_create_session(source)
    runner = _runner(env.store)
    runner._session_db = MagicMock()
    runner._session_db.update_session_model = AsyncMock()

    result = await runner.apply_session_options(source, {"model": "gpt-5.5", "provider": "openai"})

    assert result["applied"] == ["model"]
    runner._session_db.update_session_model.assert_awaited_once()
    _, model = runner._session_db.update_session_model.await_args.args
    assert model == "gpt-5.5"
    assert runner._session_db.update_session_model.await_args.kwargs == {
        "provider": "openai", "base_url": None, "api_mode": "responses"}


# -- C5-3 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_expired_session_consumes_boundary_and_drops_old_scope(env):
    source = _source()
    old = env.store.get_or_create_session(source)
    key = old.session_key
    runner = _runner(env.store)
    conv = runner._session_state(key).conversation
    conv.reasoning_override = {"enabled": False}  # the old conversation's options
    runner._pending_model_notes[key] = "[Note: stale switch note]"
    env.store.suspend_session(key)

    result = await runner.apply_session_options(source, {"fast": True})

    assert result["status"] == "accepted" and result["applied"] == ["fast"]
    entry = env.store.lookup_by_session_key(key)
    assert entry.session_id != old.session_id and entry.was_auto_reset is False
    assert conv.reasoning_override is None and key not in runner._pending_model_notes
    # The next message runs no boundary cleanup, so the option survives it and a restart.
    await runner._hmwa_open_session(await runner.async_session_store.get_or_create_session(source), key, source)
    assert runner._resolve_session_service_tier(session_key=key) == "priority"
    assert _snapshot(env, runner, key)[1] == (None, None, "priority")


# -- C5-4 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turn_admitted_during_validation_is_rejected_busy(env, monkeypatch):
    """F2 barrier: pause after the API saw "idle", admit a real turn, resume: the API must
    reject under the lock instead of committing under a running turn."""
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    resolving, resume = threading.Event(), threading.Event()

    def _slow_switch(**kw):
        resolving.set()
        assert resume.wait(10)
        return _switch(**kw)

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _slow_switch)
    turn_running, turn_done = asyncio.Event(), asyncio.Event()

    async def _turn(_self, _event, _source, _key, _generation):
        turn_running.set()
        await turn_done.wait()
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", _turn):
        api = asyncio.create_task(runner.apply_session_options(source, {"model": "gpt-5.5"}))
        try:
            assert await asyncio.to_thread(resolving.wait, 5)
            turn = asyncio.create_task(runner._handle_message(_event()))
            await asyncio.wait_for(turn_running.wait(), 5)
        finally:
            resume.set()
        result = await asyncio.wait_for(api, 5)
        turn_done.set()
        await turn

    assert result["status"] == "rejected" and result["code"] == "session_busy"
    assert _snapshot(env, runner, key)[1] == (None, None, None)


# -- C5-5 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_slash_during_api_validation_stays_authoritative(env, monkeypatch):
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    resolving, resume = threading.Event(), threading.Event()

    def _slow_switch(**kw):
        resolving.set()
        assert resume.wait(10)
        return _switch(**kw)

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _slow_switch)
    api = asyncio.create_task(runner.apply_session_options(source, {"model": "gpt-5.5", "fast": True}))
    try:
        assert await asyncio.to_thread(resolving.wait, 5)
        # The user's own command lands while the host request is still validating.
        await runner._set_session_service_tier_override(key, "cold", source=source)
    finally:
        resume.set()
    result = await asyncio.wait_for(api, 5)

    assert result["status"] == "rejected" and result["code"] == "conflict"
    live, durable, note = _snapshot(env, runner, key)
    assert live == (None, None, "cold") and durable == (None, None, "cold") and note is None


# -- C5-6 -------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["repeated_cancel", "enclosing_timeout", "failure_behind_cancel"])
async def test_cancellation_settles_before_lock_release(env, mode, caplog):
    """F3: a cancelled caller must not release the admission lock while its write is in flight,
    across ANY number of cancels; the cancel surfaces afterwards (never uncancelled), live ends
    equal to disk, and a write failure behind the cancel is logged, not lost."""
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    fail = OSError("database or disk is full") if mode == "failure_behind_cancel" else None
    parked = _ParkedWrites(env.store, fail=fail)
    competitor_in = asyncio.Event()

    async def _competing_admission():
        async with runner._session_admission_lock(key):
            competitor_in.set()

    if mode == "enclosing_timeout":
        async def _bounded():
            async with asyncio.timeout(0.05):
                await runner.apply_session_options(source, {"reasoning_effort": "max"})
        api = asyncio.create_task(_bounded())
    else:
        api = asyncio.create_task(runner.apply_session_options(source, {"reasoning_effort": "max"}))
    await parked.wait_started()
    competitor = asyncio.create_task(_competing_admission())
    try:
        if mode != "enclosing_timeout":
            for _ in range(3):
                api.cancel()
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.1)  # the timeout has fired and cancelled the call
        assert not api.done() and not competitor_in.is_set()  # still settling, lock still held
    finally:
        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            parked.release.set()
            if mode == "enclosing_timeout":
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(api, 5)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(api, 5)
    await asyncio.wait_for(competitor, 5)

    live, durable, _ = _snapshot(env, runner, key)
    assert live == durable
    if mode == "failure_behind_cancel":
        assert durable[1] is None
        assert any(
            "failed while the caller was cancelled" in r.getMessage() and r.exc_info
            for r in caplog.records)
    else:
        assert durable[1] == {"enabled": True, "effort": "max"}


# -- C5-7 -------------------------------------------------------------------------------------


def test_loop_teardown_cannot_separate_write_from_live_assignment(env):
    """asyncio.run tears the loop down (cancel-all) with the apply still pending and its write in
    flight: the write lands, live follows it, and the caller ends cancelled."""
    source = _source()
    key = env.store.get_or_create_session(source).session_key
    runner = _runner(env.store)
    parked = _ParkedWrites(env.store)
    tasks = []

    async def _main():
        tasks.append(asyncio.create_task(runner.apply_session_options(source, {"reasoning_effort": "ultra"})))
        await parked.wait_started()
        threading.Timer(0.2, parked.release.set).start()
        # Return with the apply pending: asyncio.run now cancels it and waits for it.

    asyncio.run(_main())

    assert tasks[0].cancelled()
    live, durable, _ = _snapshot(env, runner, key)
    assert durable[1] == {"enabled": True, "effort": "ultra"}
    assert live == durable
