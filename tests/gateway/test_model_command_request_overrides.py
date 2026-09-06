"""Regression tests for gateway /model preserving named-custom request_overrides."""

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_db = None
    runner._evict_cached_agent = lambda _session_key: None
    runner.session_store = None
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="ou_test",
            chat_type="dm",
            user_id="user-1",
        ),
    )


@pytest.mark.asyncio
async def test_handle_model_command_stores_request_overrides_for_named_custom_provider(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
model:
  default: gpt-5.4
  provider: openai-codex
providers: {}
custom_providers:
  - name: Local (127.0.0.1:4141)
    base_url: http://127.0.0.1:4141/v1
    model: rotator-openrouter-coding
    extra_body:
      text:
        verbosity: low
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: ModelSwitchResult(
            success=True,
            new_model="rotator-openrouter-coding",
            target_provider="custom:local-(127.0.0.1:4141)",
            provider_changed=True,
            api_key="no-key-required",
            base_url="http://127.0.0.1:4141/v1",
            api_mode="codex_responses",
            request_overrides={
                "extra_body": {"text": {"verbosity": "low"}},
            },
            provider_label="Local (127.0.0.1:4141)",
            is_global=False,
        ),
    )

    runner = _make_runner()
    event = _make_event("/model rotator-openrouter-coding --provider custom:local-(127.0.0.1:4141)")

    result = await runner._handle_model_command(event)

    assert result is not None
    session_key = runner._session_key_for_source(event.source)
    assert runner._session_model_overrides[session_key]["request_overrides"] == {
        "extra_body": {"text": {"verbosity": "low"}},
    }
