"""Regression tests for /model command blocking the async event loop.

Bug (#41289): ``list_picker_providers()`` and
``list_authenticated_providers()`` are synchronous functions that may
perform blocking I/O (e.g. ``urllib.request.urlopen`` to fetch provider
models).  When called directly from ``_handle_model_command`` (an async
method running on the gateway event loop), they freeze the loop for the
duration of the network request — 2–3 minutes in the worst case.

Fix: wrap both calls in ``await asyncio.to_thread(...)`` so the blocking
work runs in a worker thread and the event loop stays responsive.
"""

import asyncio
import time
from unittest.mock import patch, MagicMock

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


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_text_fallback_does_not_block_event_loop(tmp_path, monkeypatch):
    """``list_authenticated_providers`` (text fallback path) must run in a
    worker thread so the event loop stays responsive during blocking I/O."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: gpt-5.5\nproviders: {}\n")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    call_started = asyncio.Event()
    call_finished = asyncio.Event()

    def slow_list_authenticated_providers(**kwargs):
        call_started.set()
        time.sleep(0.5)  # Blocking sleep — would freeze the loop if not in a thread
        call_finished.set()
        return []

    # No adapter → has_picker=False → falls through to text list path
    runner = _make_runner()

    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        side_effect=slow_list_authenticated_providers,
    ), patch(
        "hermes_cli.model_switch.list_picker_providers",
        return_value=[],
    ):
        loop_responsive = False

        async def check_loop():
            nonlocal loop_responsive
            await call_started.wait()
            # If we get here while the blocking call is still running,
            # the event loop is NOT blocked — success!
            if not call_finished.is_set():
                loop_responsive = True

        checker = asyncio.create_task(check_loop())
        result = await runner._handle_model_command(_make_event("/model"))
        await checker

    assert loop_responsive, (
        "Event loop was blocked during list_authenticated_providers — "
        "the call must be wrapped in asyncio.to_thread()"
    )
    # Text fallback returns a string (the model list)
    assert result is not None


@pytest.mark.asyncio
async def test_text_fallback_returns_correct_result(tmp_path, monkeypatch):
    """Sanity check: the text fallback still returns the expected provider
    listing after being wrapped in asyncio.to_thread."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: gpt-5.5\nproviders: {}\n")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    fake_providers = [
        {
            "name": "OpenRouter",
            "slug": "openrouter",
            "is_current": True,
            "models": ["gpt-5.5", "claude-4"],
            "total_models": 2,
        },
    ]

    runner = _make_runner()

    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=fake_providers,
    ), patch(
        "hermes_cli.model_switch.list_picker_providers",
        return_value=[],
    ):
        result = await runner._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "OpenRouter" in result
    assert "gpt-5.5" in result
