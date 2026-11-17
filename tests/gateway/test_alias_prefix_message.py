"""``+<alias> rest`` on a messaging chat runs one turn on the alias, then the chat's model again.

Drives the real ``GatewayRunner._handle_message`` (idle path, real ``/model`` handler and turn
finalizer) with the model resolution and the agent run stubbed at their boundaries.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.model_switch import ModelSwitchResult

SOURCE = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
KEY = build_session_key(SOURCE)


def _runner(tmp_path, monkeypatch, *, prefix_enabled):
    import hermes_yaml
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(hermes_yaml.safe_dump({
        "model": {"default": "base-model", "provider": "openrouter"},
        "model_aliases": {"fast": {"model": "fast-model", "provider": "openrouter"}},
        "display": {"model_alias_prefix": prefix_enabled},
    }), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="fast-model", target_provider="openrouter", provider_changed=False,
        api_key="sk-test", base_url="https://openrouter.ai/api/v1", api_mode="chat_completions",
        provider_label="OpenRouter"))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    entry = SessionEntry(session_key=KEY, session_id="sess-1", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner._async_session_store = MagicMock(set_model_override=AsyncMock(),
                                            get_or_create_session=AsyncMock(return_value=entry))
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    turns = []

    async def _run_turn(event, source, quick_key, run_generation):
        override = runner._session_model_override(quick_key) or {}
        turns.append((event.text, override.get("model", "base-model")))
        return "ok"

    runner._handle_message_with_agent = _run_turn
    return runner, turns


@pytest.mark.asyncio
async def test_alias_prefix_runs_one_turn_on_the_alias_then_restores(tmp_path, monkeypatch):
    runner, turns = _runner(tmp_path, monkeypatch, prefix_enabled=True)

    await runner._handle_message(MessageEvent(text="+fast hi there", source=SOURCE, message_id="m1"))
    await runner._handle_message(MessageEvent(text="and again", source=SOURCE, message_id="m2"))
    await runner._handle_message(MessageEvent(text="+unknownword stays text", source=SOURCE, message_id="m3"))

    assert turns == [("hi there", "fast-model"), ("and again", "base-model"),
                     ("+unknownword stays text", "base-model")]


@pytest.mark.asyncio
async def test_alias_prefix_is_off_by_default(tmp_path, monkeypatch):
    runner, turns = _runner(tmp_path, monkeypatch, prefix_enabled=False)

    await runner._handle_message(MessageEvent(text="+fast hi there", source=SOURCE, message_id="m1"))

    assert turns == [("+fast hi there", "base-model")]
