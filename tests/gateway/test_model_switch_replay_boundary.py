"""Regression tests for gateway /model switches replaying opaque provider state.

A gateway session can start on one Responses provider (for example Codex/GPT)
and then switch to another (for example xAI/Grok) with /model.  Opaque
reasoning blobs such as ``encrypted_content`` are provider-specific; replaying
old blobs into the new provider causes HTTP 400 decrypt failures until the user
/resets.  A model switch should therefore act as a replay boundary for opaque
provider state while preserving visible transcript text.
"""

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _build_gateway_agent_history
from gateway.session import SessionSource


OLD_TS = 1_700_000_000.0
SWITCH_TS = OLD_TS + 10.0
NEW_TS = OLD_TS + 20.0


def test_model_switch_boundary_strips_old_opaque_assistant_replay_fields():
    history = [
        {"role": "user", "content": "old question", "timestamp": OLD_TS},
        {
            "role": "assistant",
            "content": "old answer",
            "timestamp": OLD_TS + 1.0,
            "reasoning_details": [
                {"type": "reasoning", "encrypted_content": "old-openrouter-blob"},
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "old-codex-blob", "summary": []},
            ],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_old",
                    "phase": "final",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "old answer"}],
                },
            ],
            "finish_reason": "stop",
        },
    ]

    agent_history, observed = _build_gateway_agent_history(
        history,
        provider_state_boundary_ts=SWITCH_TS,
    )

    assert observed is None
    assert agent_history == [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": "old answer",
            "finish_reason": "stop",
        },
    ]


def test_model_switch_boundary_preserves_new_provider_state_after_boundary():
    history = [
        {
            "role": "assistant",
            "content": "new answer",
            "timestamp": NEW_TS,
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "new-provider-blob", "summary": []},
            ],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_new",
                    "phase": "final",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "new answer"}],
                },
            ],
        },
    ]

    agent_history, _ = _build_gateway_agent_history(
        history,
        provider_state_boundary_ts=SWITCH_TS,
    )

    assert agent_history[0]["codex_reasoning_items"][0]["encrypted_content"] == "new-provider-blob"
    assert agent_history[0]["codex_message_items"][0]["id"] == "msg_new"


def test_model_switch_boundary_strips_unknown_timestamp_opaque_state():
    history = [
        {
            "role": "assistant",
            "content": "legacy answer",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "legacy-provider-blob", "summary": []},
            ],
        },
    ]

    agent_history, _ = _build_gateway_agent_history(
        history,
        provider_state_boundary_ts=SWITCH_TS,
    )

    assert agent_history == [{"role": "assistant", "content": "legacy answer"}]


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_model_replay_boundaries = {}
    runner._running_agents = {}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="grok-4.3",
        target_provider="xai-oauth",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://api.x.ai/v1",
        api_mode="codex_responses",
        provider_label="xAI",
        is_global=False,
    )


@pytest.mark.asyncio
async def test_model_command_records_replay_boundary(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"default": "gpt-5.5", "provider": "openai-codex"},
                "providers": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _fake_switch_result())

    runner = _make_runner()
    event = _make_event("/model grok-4.3")
    session_key = runner._session_key_for_source(event.source)

    result = await runner._handle_model_command(event)

    assert result is not None
    assert "grok-4.3" in result
    assert session_key in runner._session_model_replay_boundaries
