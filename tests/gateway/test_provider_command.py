from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/provider"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace()
    return runner


@pytest.mark.asyncio
async def test_provider_command_shows_supported_model_switch_syntax(monkeypatch, tmp_path):
    from gateway.run import _hermes_home
    import gateway.run as gateway_run

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"provider": "openai"}}), encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        "hermes_cli.models.list_available_providers",
        lambda: [{"id": "openai", "label": "OpenAI", "aliases": [], "authenticated": True}],
    )

    runner = _make_runner()
    result = await runner._handle_provider_command(_make_event())

    assert "Switch: `/model model-name --provider provider`" in result
    assert "provider:model-name" not in result
