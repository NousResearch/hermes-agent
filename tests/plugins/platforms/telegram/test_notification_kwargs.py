"""Telegram notification-mode gating (#132516): "important" mode silences ordinary sends, but a
human-decision prompt (``is_approval_prompt``) must always push — a silently delivered prompt is
indistinguishable from "no prompt" and costs the full approvals.timeout before the command is refused.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.parametrize("mode,metadata,expected", [
    ("important", None, {"disable_notification": True}),
    ("important", {"thread_id": "t1"}, {"disable_notification": True}),
    ("important", {"notify": True}, {}),
    ("important", {"thread_id": "t1", "is_approval_prompt": True}, {}),
    ("all", None, {}),
    ("all", {"is_approval_prompt": True}, {}),
])
def test_notification_kwargs(mode, metadata, expected):
    adapter = object.__new__(TelegramAdapter)
    adapter._notifications_mode = mode
    assert adapter._notification_kwargs(metadata) == expected


def test_clarify_text_fallback_pushes_in_important_mode():
    """A failed native clarify card retries as base numbered text via adapter.send(); that
    human-decision prompt must not go out with disable_notification=True."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from gateway.config import PlatformConfig
    from gateway.run_turn_runner_clarify_delivery import text_fallback_coro

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot, adapter._app = AsyncMock(), MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    adapter._notifications_mode = "important"

    asyncio.run(text_fallback_coro(
        adapter, chat_id="123", question="Which env?", choices=None, clarify_id="c1", session_key="s", metadata=None))

    assert adapter._bot.send_message.await_args.kwargs.get("disable_notification") is None


@pytest.mark.parametrize("yaml_mode,env_mode,expected", [
    ("all", None, "all"),
    ("important", None, "important"),
    (None, None, "important"),
    ("all", "important", "important"),
])
def test_resolve_notifications_mode_reads_config_yaml(tmp_path, monkeypatch, yaml_mode, env_mode, expected):
    """display.platforms.telegram.notifications in config.yaml selects the mode (#144895);
    HERMES_TELEGRAM_NOTIFICATIONS still overrides it."""
    import gateway.run as gateway_run
    from plugins.platforms.telegram.adapter import _resolve_notifications_mode

    if yaml_mode is not None:
        (tmp_path / "config.yaml").write_text(
            f"display:\n  platforms:\n    telegram:\n      notifications: {yaml_mode}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    if env_mode is None:
        monkeypatch.delenv("HERMES_TELEGRAM_NOTIFICATIONS", raising=False)
    else:
        monkeypatch.setenv("HERMES_TELEGRAM_NOTIFICATIONS", env_mode)

    assert _resolve_notifications_mode() == expected
