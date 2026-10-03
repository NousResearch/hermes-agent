"""Telegram receiving the literal Discord-rendered ``/model name:<provider>/<model>``
must switch provider AND model, confirm the resolved pair, and change nothing
when the provider is unknown.

Repro: on Telegram, current provider ``my-apr``, ``/model name:my-bpx/claude-fable-5-1``
left the session on ``my-apr`` with model ``name:my-bpx/claude-fable-5-1``.

The gateway handler drives the REAL parser and the REAL switch_model; only the
network-touching resolution steps are patched.
"""

import threading
import types
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.models import _resolve_provider_prefix


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._evict_cached_agent = lambda _session_key: None
    runner.session_store = None
    return runner


def _telegram_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="user-1",
        ),
    )


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: my-apr\n"
        "providers:\n"
        "  my-apr:\n    base_url: http://apr.invalid/v1\n    models:\n      claude-opus-5: {}\n"
        "  my-bpx:\n    base_url: http://bpx.invalid/v1\n    models:\n      claude-fable-5-1: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    runner = _make_runner()
    calls = []
    agent = types.SimpleNamespace(
        model="claude-opus-5",
        provider="my-apr",
        reasoning_config=None,
        context_compressor=None,
        switch_model=lambda **kw: calls.append(kw),
    )

    async def _noop(*_a, **_k):
        return None

    runner._announce_model_switch = _noop
    runner._announce_switch = _noop
    runner._switch_reasoning_kwargs = lambda **_k: {}
    return runner, agent, calls


def _patched_resolution(validation):
    return [
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch("hermes_cli.model_switch.normalize_model_for_provider",
              side_effect=lambda model, provider: model),
        patch("hermes_cli.models_validate.validate_requested_model", return_value=validation),
        patch("hermes_cli.models._configured_provider_ids", return_value={"my-apr", "my-bpx"}),
        patch("hermes_cli.models.detect_provider_for_model",
              side_effect=lambda name, _current: _resolve_provider_prefix(name)),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider",
              return_value={"api_key": "***", "base_url": "http://resolved/v1", "api_mode": ""}),
    ]


async def _run(runner, agent, text, validation):
    event = _telegram_event(text)
    session_key = runner._session_key_for_source(event.source)
    runner._agent_cache[session_key] = (agent, 0.0)
    patches = _patched_resolution(validation)
    for p in patches:
        p.start()
    try:
        reply = await runner._handle_model_command(event)
    finally:
        for p in reversed(patches):
            p.stop()
    return reply, session_key


@pytest.mark.asyncio
async def test_telegram_discord_name_form_switches_provider_and_model(runner_env):
    runner, agent, calls = runner_env
    reply, session_key = await _run(
        runner, agent, "/model name:my-bpx/claude-fable-5-1",
        {"accepted": True, "persist": True, "recognized": True, "message": None},
    )

    assert calls, reply
    assert calls[0]["new_provider"] == "my-bpx"
    assert calls[0]["new_model"] == "claude-fable-5-1"
    override = runner._session_model_overrides[session_key]
    assert (override["provider"], override["model"]) == ("my-bpx", "claude-fable-5-1")
    # The confirmation names the resolved pair so a mis-split is visible.
    assert "`my-bpx/claude-fable-5-1`" in reply, reply
    assert "name:" not in reply, reply


@pytest.mark.asyncio
async def test_telegram_unknown_provider_prefix_refused_without_state_change(runner_env):
    runner, agent, calls = runner_env
    reply, session_key = await _run(
        runner, agent, "/model name:no-such-prov/claude-fable-5-1",
        {"accepted": True, "persist": True, "recognized": False, "message": "Note: could not verify"},
    )

    assert "Unknown provider 'no-such-prov'" in reply, reply
    assert not calls
    assert session_key not in runner._session_model_overrides
