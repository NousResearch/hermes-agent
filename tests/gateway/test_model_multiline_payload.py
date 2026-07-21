from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.model_switch import ModelSwitchResult

pytestmark = __import__("pytest").mark.asyncio


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
    return MessageEvent(text=text, source=_make_source(thread_id), message_id="$event")


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
    runner.adapters = {Platform.MATRIX: adapter}
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
    runner._session_model_overrides = {}
    runner._voice_mode = {}
    runner._draining = False
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
    runner._handle_message_with_agent = AsyncMock(return_value="agent reply")
    return runner, adapter


def _quiet_model_switch_guards(monkeypatch):
    """Neutralize environment-dependent gates around the switch itself."""
    import gateway.code_skew as code_skew
    import hermes_cli.model_cost_guard as model_cost_guard

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda *a, **k: None)
    monkeypatch.setattr(
        model_cost_guard, "expensive_model_warning", lambda *a, **k: None
    )


async def test_model_command_with_inline_payload_switches_then_routes_payload(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(model_switch, "resolve_display_context_length", lambda *a, **k: 0)
    _quiet_model_switch_guards(monkeypatch)

    seen_raw_inputs: list[str] = []

    def fake_switch_model(**kwargs):
        seen_raw_inputs.append(kwargs["raw_input"])
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider="openrouter",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr(model_switch, "switch_model", fake_switch_model)

    runner, adapter = _make_runner()
    result = await runner._handle_message(
        _make_event(
            "/model ollama-cloud/glm-5.1\nBonjour test, repond OK",
            thread_id="thread-1",
        )
    )

    assert result == "agent reply"
    assert seen_raw_inputs == ["ollama-cloud/glm-5.1"]

    runner.hooks.emit_collect.assert_awaited_once()
    hook_name, hook_ctx = runner.hooks.emit_collect.await_args.args
    assert hook_name == "command:model"
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
    from gateway.slash_commands import ModelSwitchConfirmation

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

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
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

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
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    _quiet_model_switch_guards(monkeypatch)

    seen_raw_inputs: list[str] = []

    def fake_switch_model(**kwargs):
        seen_raw_inputs.append(kwargs["raw_input"])
        return ModelSwitchResult(success=False, error_message="bad model")

    monkeypatch.setattr(model_switch, "switch_model", fake_switch_model)

    runner, adapter = _make_runner()
    result = await runner._handle_message(_make_event("/model bad-model\nQuestion body"))

    assert result == "Error: bad model"
    assert seen_raw_inputs == ["bad-model"]
    adapter.send.assert_not_awaited()
    adapter._send_with_retry.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


async def test_model_command_without_payload_behaves_as_before(tmp_path, monkeypatch):
    """Single-line /model keeps returning the confirmation as the reply."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(model_switch, "resolve_display_context_length", lambda *a, **k: 0)
    _quiet_model_switch_guards(monkeypatch)

    def fake_switch_model(**kwargs):
        return ModelSwitchResult(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider="openrouter",
            provider_label="OpenRouter",
        )

    monkeypatch.setattr(model_switch, "switch_model", fake_switch_model)

    runner, adapter = _make_runner()
    result = await runner._handle_message(_make_event("/model ollama-cloud/glm-5.1"))

    assert isinstance(result, str)
    assert "Model switched to `ollama-cloud/glm-5.1`" in result
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()
