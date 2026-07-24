"""Regression tests: /model status/info should not switch to literal models.

Adapted from closed PR #28498 after the gateway model handler moved from
``gateway/run.py`` into ``gateway/slash_commands.py``.
"""

import yaml
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
    runner._running_agents = {}
    return runner


def _make_event(text="/model status"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gemini-3.1-pro-preview",
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:8317/v1",
                },
                "providers": {},
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize("subcommand", ["status", "info", "current", "show"])
async def test_model_status_alias_shows_info_instead_of_switching(
    _isolated_home,
    monkeypatch,
    subcommand,
):
    """Info aliases should display the current model, not call switch_model()."""
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: pytest.fail(f"/model {subcommand} must not call switch_model"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **_kwargs: pytest.fail(f"/model {subcommand} must not open the picker"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: pytest.fail(f"/model {subcommand} must not list provider catalogs"),
    )

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event(f"/model {subcommand}"))

    session_key = runner._session_key_for_source(
        SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    )
    override = runner._session_model_overrides.get(session_key, {})
    assert override.get("model") != subcommand
    assert result is not None
    assert "Current: `gemini-3.1-pro-preview` on custom" in result
    assert "`/model <name>`" in result


@pytest.mark.asyncio
async def test_model_status_uses_session_override(_isolated_home):
    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_event("/model status").source)
    runner._session_model_overrides[session_key] = {
        "model": "anthropic/claude-sonnet-4",
        "provider": "anthropic",
    }

    result = await runner._handle_model_command(_make_event("/model status"))

    assert result is not None
    assert "Current: `anthropic/claude-sonnet-4` on anthropic" in result
