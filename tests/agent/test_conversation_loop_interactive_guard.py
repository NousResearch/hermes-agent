from types import SimpleNamespace

from agent.conversation_loop import (
    _interactive_turn_wall_seconds,
    _interactive_turn_guard_enabled,
    _interactive_wall_clock_guard_message,
)


def test_telegram_wall_clock_guard_message_preserves_background_state() -> None:
    message = _interactive_wall_clock_guard_message("telegram", 900)

    assert "Telegram turn hit Hermes' 15-minute interactive safety guard" in message
    assert "Background jobs, if any, were not intentionally stopped" in message
    assert "verify the live state before continuing" in message


def test_non_telegram_wall_clock_guard_message_is_not_misbranded() -> None:
    message = _interactive_wall_clock_guard_message("slack", 900)

    assert message.startswith("This interactive turn hit")
    assert "Telegram" not in message


def test_interactive_turn_wall_seconds_from_env(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_INTERACTIVE_TURN_TIMEOUT", "1200")

    assert _interactive_turn_wall_seconds() == 1200


def test_interactive_turn_wall_seconds_rejects_disabled_env(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_INTERACTIVE_TURN_TIMEOUT", "0")

    assert _interactive_turn_wall_seconds() == 900


def test_interactive_turn_guard_scope() -> None:
    assert _interactive_turn_guard_enabled(SimpleNamespace(platform="telegram"))
    assert _interactive_turn_guard_enabled(SimpleNamespace(platform="slack"))
    assert not _interactive_turn_guard_enabled(SimpleNamespace(platform="cli"))
    assert not _interactive_turn_guard_enabled(SimpleNamespace(platform="cron"))
