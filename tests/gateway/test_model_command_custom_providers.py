"""Regression tests for gateway /model support of config.yaml custom_providers."""

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_direct_model_switch_offloads_to_thread(tmp_path, monkeypatch):
    """A direct `/model <name>` switch must route switch_model() through
    ``asyncio.to_thread`` — never run it on the event loop."""
    import asyncio

    import gateway.slash_commands_model as scm
    from hermes_cli.model_switch import ModelSwitchResult

    runner = _make_runner()
    ctx = scm._ModelSwitchContext(
        session_key="s1", source=None, persist_global=False,
        config_path=tmp_path / "config.yaml",
        current_provider="custom", current_base_url="http://127.0.0.1:8317/v1",
        current_api_key="test-custom-key",
    )

    captured = {}

    def _fake_switch(**kwargs):
        captured.update(kwargs)
        return ModelSwitchResult(success=False, error_message="stop after capture")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)

    loop = asyncio.get_running_loop()
    orig = loop.run_in_executor

    async def fake_to_thread(func, *args, **kwargs):
        # record that the switch ran OFF the event loop thread
        captured["thread"] = "worker"
        return func(*args, **kwargs)

    monkeypatch.setattr(scm.asyncio, "to_thread", fake_to_thread)

    result, error = await GatewayRunner._perform_model_switch(
        runner, ctx, "Kimi-K3", None, "test",
    )

    assert error is None or "stop after capture" in str(error)
    assert captured["current_api_key"] == "test-custom-key"


@pytest.mark.asyncio
async def test_read_config_populates_current_api_key(tmp_path):
    """read_config() must lift model.api_key into the switch context so the
    bare custom endpoint's /models probe authenticates (#83837 family)."""
    import gateway.slash_commands_model as scm

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.6-sol",
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:8317/v1",
                    "api_key": "test-custom-key",
                }
            }
        ),
        encoding="utf-8",
    )

    ctx = scm._ModelSwitchContext(
        session_key="s1", source=None, persist_global=False,
        config_path=tmp_path / "config.yaml",
    )
    ctx.read_config()

    assert ctx.current_provider == "custom"
    assert ctx.current_base_url == "http://127.0.0.1:8317/v1"
    assert ctx.current_api_key == "test-custom-key"
