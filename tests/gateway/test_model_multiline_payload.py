"""``/model <name>`` with an inline payload (``/model <name>\\n<prompt>``).

A successful typed switch delivers the confirmation out-of-band and routes the payload to the
agent as the user turn on the new model; every other response (error, picker, help, plain str)
keeps the pre-existing behavior. Routing keys off the structural ``ModelSwitchConfirmation``
type — never off localized confirmation wording.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.model_switch import ModelSwitchResult

pytestmark = pytest.mark.asyncio

CONFIG_YAML = """\
model:
  default: old-model
  provider: openrouter
providers: {}
"""


def _make_source(thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        user_id="u1",
        chat_id="!room:example.org",
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
    )


def _make_event(text: str, thread_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(thread_id),
        message_id="$event",
        internal=True,
    )


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.MATRIX,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.MATRIX: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock()
    adapter.send_slash_confirm = AsyncMock(return_value=None)  # no buttons → text fallback
    adapter._pending_messages = {}
    runner.adapters = {Platform.MATRIX: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    _facade = MagicMock()
    _facade._store = runner.session_store
    _facade.set_model_override = AsyncMock()
    runner._async_session_store = _facade
    runner._evict_cached_agent = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._update_prompt_pending = {}
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._pending_native_image_paths_by_session = {}
    runner._background_tasks = {}
    runner._background_task_counter = 0
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._session_service_tier_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._pending_model_notes = {}
    runner._service_tier = None
    runner._fast_mode_by_session = {}
    runner._goal_state_by_session = {}
    runner._goal_runs_in_progress = set()
    runner._goal_queued_by_session = set()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._check_slash_access = lambda _source, _command: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda key: runner._running_agents.pop(key, None)
    runner._handle_message_with_agent = AsyncMock(return_value="agent reply")
    return runner, adapter


def _setup_isolated_home(tmp_path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)


def _quiet_model_switch_guards(monkeypatch) -> None:
    """Neutralize environment-dependent gates around the switch itself."""
    import gateway.code_skew as code_skew
    import hermes_cli.model_selection_guards as model_selection_guards

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda *a, **k: None)
    monkeypatch.setattr(
        model_selection_guards, "combined_selection_warning", lambda *a, **k: None
    )


def _fire_selection_guard(monkeypatch) -> None:
    """Make the selection guard fire for every model (cost-guard style warning)."""
    import hermes_cli.model_selection_guards as model_selection_guards

    warning = SimpleNamespace(title="Cost guard", message="This model is expensive.")
    monkeypatch.setattr(
        model_selection_guards, "combined_selection_warning", lambda *a, **k: warning
    )


def _fake_switch_model(monkeypatch, *, success: bool = True, seen_raw_inputs: list):
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(model_switch, "resolve_display_context_length", lambda *a, **k: 0)

    def fake_switch_model(**kwargs):
        seen_raw_inputs.append(kwargs["raw_input"])
        if not success:
            return ModelSwitchResult(success=False, error_message="bad model")
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider="openrouter",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr(model_switch, "switch_model", fake_switch_model)


def _command_model_hook_ctx(runner) -> dict:
    """Return the hook ctx of the single ``command:model`` interceptor emission."""
    hook_calls = [
        call
        for call in runner.hooks.emit_collect.await_args_list
        if call.args and call.args[0] == "command:model"
    ]
    assert len(hook_calls) == 1
    return hook_calls[0].args[1]


async def test_model_command_with_inline_payload_switches_then_routes_payload(
    tmp_path, monkeypatch
):
    _setup_isolated_home(tmp_path, monkeypatch)
    _quiet_model_switch_guards(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    result = await runner._handle_message(
        _make_event(
            "/model ollama-cloud/glm-5.1\nBonjour test, repond OK",
            thread_id="thread-1",
        )
    )

    assert result == "agent reply"
    assert seen_raw_inputs == ["ollama-cloud/glm-5.1"]

    hook_ctx = _command_model_hook_ctx(runner)
    assert hook_ctx["raw_args"] == "ollama-cloud/glm-5.1"
    assert hook_ctx["args"] == "ollama-cloud/glm-5.1"

    # The switch confirmation is delivered out-of-band before the payload is
    # routed to the agent (the /blueprint ack pattern).
    adapter.send.assert_awaited_once()
    send_args = adapter.send.await_args
    assert send_args.args[0] == "!room:example.org"
    assert "Model switched to `ollama-cloud/glm-5.1`" in send_args.args[1]
    metadata = send_args.kwargs.get("metadata") or {}
    assert metadata.get("thread_id") == "thread-1"

    runner._handle_message_with_agent.assert_awaited_once()
    routed_event = runner._handle_message_with_agent.await_args.args[0]
    assert routed_event.text == "Bonjour test, repond OK"


async def test_model_command_confirmation_is_structural_not_text(tmp_path, monkeypatch):
    """Routing keys off ModelSwitchConfirmation, not the localized wording."""
    from gateway.slash_commands_model import ModelSwitchConfirmation

    runner, adapter = _make_runner()
    localized = ModelSwitchConfirmation("Modèle remplacé par `m1` (libellé localisé)")
    runner._handle_model_command = AsyncMock(return_value=localized)

    result = await runner._handle_message(_make_event("/model m1\ncorps du prompt"))

    assert result == "agent reply"
    # Handler received only the command line.
    handled_event = runner._handle_model_command.await_args.args[0]
    assert handled_event.text == "/model m1"
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[1] == str(localized)
    routed_event = runner._handle_message_with_agent.await_args.args[0]
    assert routed_event.text == "corps du prompt"


async def test_model_command_plain_string_response_does_not_route(tmp_path, monkeypatch):
    """A non-confirmation response (help text, picker, error) is returned as-is."""
    runner, adapter = _make_runner()
    runner._handle_model_command = AsyncMock(
        return_value="Model switched to `m1`"  # looks like success, but plain str
    )

    result = await runner._handle_message(_make_event("/model m1\nprompt body"))

    assert result == "Model switched to `m1`"
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


async def test_model_command_with_inline_payload_does_not_route_after_switch_error(
    tmp_path, monkeypatch
):
    _setup_isolated_home(tmp_path, monkeypatch)
    _quiet_model_switch_guards(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, success=False, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    result = await runner._handle_message(_make_event("/model bad-model\nQuestion body"))

    assert result == "Error: bad model"
    assert seen_raw_inputs == ["bad-model"]
    adapter.send.assert_not_awaited()
    adapter._send_with_retry.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


async def test_model_command_without_payload_behaves_as_before(tmp_path, monkeypatch):
    """Single-line /model keeps returning the confirmation as the reply."""
    _setup_isolated_home(tmp_path, monkeypatch)
    _quiet_model_switch_guards(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    result = await runner._handle_message(_make_event("/model ollama-cloud/glm-5.1"))

    assert isinstance(result, str)
    assert "Model switched to `ollama-cloud/glm-5.1`" in result
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


# ---- Approval path: the payload survives a selection-guard confirmation ----


async def test_model_inline_payload_routes_once_after_approval(tmp_path, monkeypatch):
    """Guard fires on ``/model <name>\n<payload>``: /approve switches the model and routes the
    original payload exactly once; a second /approve routes nothing."""
    from tools import slash_confirm as slash_confirm_mod

    _setup_isolated_home(tmp_path, monkeypatch)
    _fire_selection_guard(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    session_key = runner._session_key_for_source(_make_source(thread_id="thread-1"))
    slash_confirm_mod.clear(session_key)  # module-global confirm state; fresh per test

    # Turn 1: the guard fires — nothing is committed and the payload is retained session-scoped.
    prompt = await runner._handle_message(
        _make_event("/model expensive-model\npayload body", thread_id="thread-1")
    )
    assert "Cost guard" in prompt
    assert seen_raw_inputs == ["expensive-model"]
    assert runner._session_model_overrides == {}
    assert runner._model_inline_payload_stash() == {session_key: "payload body"}
    pending = slash_confirm_mod.get_pending(session_key)
    assert pending is not None and pending["command"] == "model"

    # Turn 2: !approve commits the switch, delivers the confirmation out-of-band, and the
    # retained payload runs as the user turn on the new model.
    result = await runner._handle_message(_make_event("!approve", thread_id="thread-1"))
    assert result == "agent reply"
    assert runner._session_model_overrides[session_key]["model"] == "expensive-model"
    adapter.send.assert_awaited_once()
    assert "Model switched to `expensive-model`" in adapter.send.await_args.args[1]
    runner._handle_message_with_agent.assert_awaited_once()
    assert runner._handle_message_with_agent.await_args.args[0].text == "payload body"
    assert runner._model_inline_payload_stash() == {}

    # Turn 3: a second /approve routes nothing (confirm resolved, stash empty).
    await runner._handle_message(_make_event("/approve", thread_id="thread-1"))
    assert runner._handle_message_with_agent.await_count == 1
    assert adapter.send.await_count == 1


async def test_model_inline_payload_not_routed_on_cancel(tmp_path, monkeypatch):
    """Cancel/deny drops the retained payload: nothing is routed and nothing is committed."""
    from tools import slash_confirm as slash_confirm_mod

    _setup_isolated_home(tmp_path, monkeypatch)
    _fire_selection_guard(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    session_key = runner._session_key_for_source(_make_source(thread_id="thread-1"))
    slash_confirm_mod.clear(session_key)

    await runner._handle_message(
        _make_event("/model expensive-model\npayload body", thread_id="thread-1")
    )
    result = await runner._handle_message(_make_event("!cancel", thread_id="thread-1"))

    assert "cancelled" in result.lower()
    assert runner._session_model_overrides == {}
    assert runner._model_inline_payload_stash() == {}
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


async def test_model_inline_payload_dropped_when_approved_switch_fails(
    tmp_path, monkeypatch
):
    """A switch that fails on approval drops the payload — no routing, no replay."""
    from tools import slash_confirm as slash_confirm_mod

    _setup_isolated_home(tmp_path, monkeypatch)
    _fire_selection_guard(monkeypatch)
    seen_raw_inputs: list[str] = []
    _fake_switch_model(monkeypatch, seen_raw_inputs=seen_raw_inputs)

    runner, adapter = _make_runner()
    session_key = runner._session_key_for_source(_make_source(thread_id="thread-1"))
    slash_confirm_mod.clear(session_key)

    class _FailingAgent:
        def __init__(self):
            self.model = "old-model"
            self.provider = "openrouter"

        def switch_model(self, **kwargs):
            # Mirrors agent_runtime_helpers.switch_model: the real method restores old
            # state then re-raises; the gateway must surface the error, not commit.
            raise RuntimeError("connection refused: bad base_url")

    import threading

    runner._agent_cache = {session_key: [_FailingAgent(), None]}
    runner._agent_cache_lock = threading.Lock()
    evicted = []
    runner._evict_cached_agent = lambda sk: evicted.append(sk)

    await runner._handle_message(
        _make_event("/model expensive-model\npayload body", thread_id="thread-1")
    )
    result = await runner._handle_message(_make_event("!approve", thread_id="thread-1"))

    assert "failed" in result.lower()
    assert runner._handle_message_with_agent.await_count == 0
    assert runner._model_inline_payload_stash() == {}
    assert runner._session_model_overrides == {}
    assert evicted == []
    adapter.send.assert_not_awaited()


async def test_new_clears_staged_model_inline_payload():
    """The stash is conversation-scoped: /new clears it for the calling session only."""
    runner, _adapter = _make_runner()
    session_key = runner._session_key_for_source(_make_source())
    # Reset-shaped store with the real async facade (the property rebuilds it), per
    # test_session_model_reset.py.
    session_entry = _session_entry()
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = session_key
    runner._async_session_store = None
    runner._session_db = None
    stash = runner._model_inline_payload_stash()
    stash[session_key] = "stale payload"
    stash["other_key"] = "other payload"

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in stash
    assert stash["other_key"] == "other payload"
