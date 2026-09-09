"""Title input is user intent, not the gateway's model-facing skill payload."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import title_generator
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionEntry
from gateway.turn_context import TurnContext
from run_agent import AIAgent


class _TurnObserved(BaseException):
    """Stop after the real prologue, before any main-model network request."""


@pytest.mark.asyncio
@pytest.mark.parametrize("skills", [[], "alpha", ["alpha"], ["alpha", "beta"]])
async def test_gateway_titles_original_request_without_changing_model_input(tmp_path, monkeypatch, skills):
    question = "Why does my connection pool exhaust after a deployment?"
    names = [skills] if isinstance(skills, str) else skills
    payloads = {name: f"# {name}\n" + (f"Follow {name} procedures.\n" * 100) for name in names}
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", lambda name, **kw: (
        {"name": name, "content": payloads[name]}, tmp_path / name, name,
    ))
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel", user_id="user", user_name="Example")
    event = MessageEvent(text=question, source=source, auto_skill=skills,
                         channel_context="[Discord channel context: synthetic metadata]")
    entry = SessionEntry(session_id="title-test", session_key="discord:title-test",
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._get_proxy_url = lambda: None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, entry, entry.session_key))
    runner._hmwa_open_session = AsyncMock(return_value=(False, True))
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=AsyncMock(return_value=[]))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._consume_pending_native_image_paths = lambda key: []
    runner._adapter_for_source = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None
    async def propagate_error(exc, *args):
        raise exc
    runner._hmwa_agent_error_reply = propagate_error

    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    agent = AIAgent(session_db=db, model="test-model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                    platform="discord", session_id=entry.session_id, enabled_toolsets=[],
                    quiet_mode=True, skip_memory=True, skip_context_files=True)
    agent.compression_enabled = False
    title_inputs, title_requests, model_inputs = [], [], []

    def title_request(**kwargs):
        title_requests.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"title": "Diagnose connection pool exhaustion"}'))])

    def title_at_turn_start(db, sid, text, **kwargs):
        title_inputs.append(text)
        title_generator.generate_title(text)

    monkeypatch.setattr(title_generator, "maybe_auto_title", title_at_turn_start)
    monkeypatch.setattr(title_generator, "call_llm", title_request)

    def observe_model(agent, messages):
        model_inputs.append(messages[-1]["content"])
        raise _TurnObserved

    monkeypatch.setattr("agent.conversation_loop.begin_iteration", observe_model)

    defaults = TurnContext()
    display = SimpleNamespace(**{name: getattr(defaults, name) for name in runner._DISPLAY_TO_TURN_CTX})
    display.platform_key = "discord"
    display.resolve_display_setting = lambda *args: False
    runner._run_agent_display_settings = lambda source: display

    def run_without_delivery(ctx, worker, *args):
        # Execute at the existing wiring seam: real _run_agent_inner/context/agent
        # forwarding, without starting the executor or platform delivery tasks.
        worker._native_image_run_message = lambda: ctx.message
        worker._run_conversation_with_approval(agent, [], None, ctx.persist_user_message, ctx.persist_user_timestamp)

    runner._run_agent_bind_turn_wiring = run_without_delivery
    runner._run_agent = runner._run_agent_inner
    try:
        with pytest.raises(_TurnObserved):
            await runner._handle_message_with_agent(event, source, entry.session_key, 1)
        assert title_inputs == [question]
        assert title_requests == [question]
        # Both independent per-turn carriers must survive the gateway/facade/loop
        # handoff: title text must not displace upstream memory author attribution.
        assert agent._turn_author == {"id": "user", "name": "Example", "is_bot": False}
        assert question in model_inputs[0]
        for payload in payloads.values():
            assert payload in model_inputs[0]
        if skills:
            assert model_inputs[0].index(question) > title_generator.MAX_TITLE_INPUT_CHARS
        stored = [row["content"] for row in db.get_messages(entry.session_id) if row["role"] == "user"]
        assert stored and question in stored[0]
        for payload in payloads.values():
            assert payload in stored[0]
        # The ordinary Discord metadata is still model-facing, too.
        assert model_inputs[0] != question
        # The cached agent must not reuse this title override on a later caller
        # that does not supply one.
        with pytest.raises(_TurnObserved):
            agent.run_conversation("Explain transaction isolation")
        assert title_inputs[-1] == "Explain transaction isolation"
        assert title_requests[-1] == "Explain transaction isolation"
        # Relay metadata belongs to upstream's facade, while title input belongs
        # to the conversation prologue. Both must survive the same admission.
        from agent import relay_runtime
        from unittest.mock import Mock
        begin_turn = Mock(wraps=relay_runtime.SESSION_COORDINATOR.begin_turn)
        monkeypatch.setattr(relay_runtime.SESSION_COORDINATOR, "begin_turn", begin_turn)
        metadata = {"source": "title-regression"}
        with pytest.raises(_TurnObserved):
            agent.run_conversation("Enriched relay request", relay_metadata=metadata,
                                   title_user_message="Original relay request")
        assert begin_turn.call_args.kwargs["metadata"] is metadata
        assert title_inputs[-1] == "Original relay request"
        assert model_inputs[-1] == "Enriched relay request"
    finally:
        agent.close()
        db.close()
