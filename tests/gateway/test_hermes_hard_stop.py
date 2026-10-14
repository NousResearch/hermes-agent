"""Tests for the Home Assistant-owned Hermes first-turn hard stop."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
import gateway.run as gateway_run


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _clear_cache():
    gateway_run._HERMES_HARD_STOP_CACHE.update(
        {"ts": 0.0, "state": None, "reason": ""}
    )


def test_explicit_ha_on_state_blocks(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(
        gateway_run,
        "_hermes_hard_stop_ha_config",
        lambda: ("http://ha.test", "token"),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(
            {"state": "on", "attributes": {"reason": "test threshold"}}
        ),
    )

    assert gateway_run._check_hermes_first_turn_hard_stop() == (
        True,
        "test threshold",
    )


def test_ha_error_fails_open(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(
        gateway_run,
        "_hermes_hard_stop_ha_config",
        lambda: ("http://ha.test", "token"),
    )

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("HA unavailable")

    monkeypatch.setattr("urllib.request.urlopen", _timeout)
    assert gateway_run._check_hermes_first_turn_hard_stop() == (False, "")


def _source():
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )


def _runner_with_session(session_entry):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = None
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda _source: session_entry
    )
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._is_telegram_topic_lane = lambda _source: False
    return runner


@pytest.mark.asyncio
async def test_new_first_turn_is_blocked_before_context_build(monkeypatch):
    session = SimpleNamespace(
        session_key="discord:test",
        session_id="session-1",
        created_at=100,
        updated_at=100,
        was_auto_reset=False,
        is_fresh_reset=False,
    )
    runner = _runner_with_session(session)
    monkeypatch.setattr(
        gateway_run,
        "_check_hermes_first_turn_hard_stop",
        lambda: (True, "OpenAI test threshold"),
    )

    result = await runner._handle_message_with_agent(
        SimpleNamespace(text="hello"), _source(), "quick", 1
    )

    assert "hard stop is active" in result
    assert "OpenAI test threshold" in result
    runner.hooks.emit.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_subturn_skips_hard_stop_check(monkeypatch):
    session = SimpleNamespace(
        session_key="discord:test",
        session_id="session-1",
        created_at=100,
        updated_at=101,
        was_auto_reset=False,
        is_fresh_reset=False,
    )
    runner = _runner_with_session(session)

    def _must_not_run():
        raise AssertionError("hard-stop check ran for an existing sub-turn")

    monkeypatch.setattr(
        gateway_run, "_check_hermes_first_turn_hard_stop", _must_not_run
    )

    class _ReachedContext(Exception):
        pass

    def _context(*_args, **_kwargs):
        raise _ReachedContext

    monkeypatch.setattr(gateway_run, "build_session_context", _context)

    with pytest.raises(_ReachedContext):
        await runner._handle_message_with_agent(
            SimpleNamespace(text="continue"), _source(), "quick", 1
        )
