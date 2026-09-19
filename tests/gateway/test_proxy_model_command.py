"""Proxy-mode /model must not resolve against the sidecar catalog.

The Matrix E2EE sidecar has GATEWAY_PROXY_URL and an empty providers: block.
Local switch_model() raises Unknown provider 'Local'/'ollama'. The sidecar
stores a session override; the next proxied turn sends model+provider to the host.
"""

from unittest.mock import MagicMock

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:server.org",
            chat_type="dm",
            user_id="@greg:nevfi",
            user_name="greg",
        ),
    )


def _runner(tmp_path, monkeypatch, *, proxy_url="http://host:8642"):
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"proxy_url": proxy_url}, "platforms": {"matrix": {"enabled": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("GATEWAY_PROXY_URL", proxy_url)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    called = []

    def _should_not_switch(**kwargs):
        called.append(kwargs)
        raise AssertionError("switch_model must not run on a proxy-mode sidecar")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _should_not_switch)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.multiplex_profiles = False
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._switch_calls = called
    return runner


@pytest.mark.asyncio
async def test_proxy_model_switch_stores_override_without_local_resolve(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    result = await runner._handle_model_command(
        _event("/model cf-turbo-q8-dual --provider ollama")
    )
    assert runner._switch_calls == []
    assert result is not None and "cf-turbo-q8-dual" in result
    assert "ollama" in result
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "cf-turbo-q8-dual"
    assert overrides[0]["provider"] == "ollama"


@pytest.mark.asyncio
async def test_proxy_model_switch_accepts_display_name_local(tmp_path, monkeypatch):
    """The host catalog shows name Local for providers.local — users type that."""
    runner = _runner(tmp_path, monkeypatch)
    result = await runner._handle_model_command(
        _event("/model cf-turbo-q8-dual --provider Local")
    )
    assert runner._switch_calls == []
    assert result is not None and "Unknown provider" not in result
    override = next(iter(runner._session_model_overrides.values()))
    assert override["provider"] == "Local"


@pytest.mark.asyncio
async def test_proxy_model_requires_provider_flag(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model cf-turbo-q8-dual"))
    assert runner._switch_calls == []
    assert result is not None and "--provider" in result
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_proxy_model_bare_lists_usage(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model"))
    assert runner._switch_calls == []
    assert result is not None and "proxy mode" in result.lower()
    assert "--provider" in result
