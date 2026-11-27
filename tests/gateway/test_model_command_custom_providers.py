"""Regression tests for gateway /model support of config.yaml custom_providers."""

import threading

import pytest
import hermes_yaml as yaml

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
async def test_direct_model_switch_runs_off_the_event_loop(tmp_path, monkeypatch):
    """A direct `/model <name>` switch must run switch_model() on a worker thread so the
    blocking models.dev HTTP fetch can't freeze the gateway event loop (#20525)."""
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5.4", "provider": "openrouter"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    switch_threads: list[int] = []

    # Fail the switch so the handler returns before _finish_switch (which needs
    # full runner state) — only where the switch ran matters here.
    def _fake_switch(**kwargs):
        switch_threads.append(threading.get_ident())
        return ModelSwitchResult(success=False, error_message="nope")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)

    result = await _make_runner()._handle_model_command(_make_event("/model gpt-5.4"))

    assert switch_threads, "switch_model never ran"
    assert threading.get_ident() not in switch_threads, "switch_model ran inline on the event-loop thread"
    assert result is not None and "nope" in result


@pytest.mark.asyncio
async def test_direct_custom_model_switch_uses_configured_api_key(tmp_path, monkeypatch):
    """Gateway /model must authenticate the custom /models probe: read_config()
    lifts model.api_key into the switch context so switch_model() receives it
    (#83837)."""
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {
                "default": "gpt-5.6-sol",
                "provider": "custom",
                "base_url": "http://127.0.0.1:8317/v1",
                "api_key": "test-custom-key",
            }
        }),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    captured = {}

    def _fake_switch(**kwargs):
        captured.update(kwargs)
        return ModelSwitchResult(success=False, error_message="stop after capture")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)

    result = await _make_runner()._handle_model_command(_make_event("/model Kimi-K3"))

    assert result is not None and "stop after capture" in result
    assert captured["current_api_key"] == "test-custom-key"
