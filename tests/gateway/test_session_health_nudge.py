from datetime import datetime, timedelta
from typing import Any, cast

from gateway.config import Platform
from gateway.run import _build_session_health_nudge
from gateway.session import SessionEntry


def _make_entry(**overrides: Any) -> SessionEntry:
    now = datetime.now()
    data: dict[str, Any] = {
        "session_key": "telegram:dm:u1:c1",
        "session_id": "sess-1",
        "created_at": now - timedelta(hours=13),
        "updated_at": now,
        "platform": Platform.TELEGRAM,
        "chat_type": "dm",
    }
    data.update(overrides)
    return cast(SessionEntry, SessionEntry(**data))


def test_session_health_nudge_triggers_for_old_heavy_telegram_session():
    entry = _make_entry()
    history = [{"role": "user", "content": f"message {i}"} for i in range(130)]

    result = _build_session_health_nudge(
        platform=Platform.TELEGRAM,
        session_entry=entry,
        history=history,
        approx_tokens=70000,
        context_length=100000,
        now=datetime.now(),
    )

    assert result is not None
    notice, reason = result
    assert reason == "context"
    assert "thread already has 130 messages" in notice
    assert "session has been open for" in notice
    assert "/new" in notice
    assert "/compress" in notice


def test_session_health_nudge_skips_non_telegram_and_already_nudged():
    history = [{"role": "user", "content": f"message {i}"} for i in range(130)]

    non_telegram = _build_session_health_nudge(
        platform=Platform.DISCORD,
        session_entry=_make_entry(platform=Platform.DISCORD),
        history=history,
        approx_tokens=70000,
        context_length=100000,
        now=datetime.now(),
    )
    assert non_telegram is None

    already_nudged = _build_session_health_nudge(
        platform=Platform.TELEGRAM,
        session_entry=_make_entry(session_health_nudged=True),
        history=history,
        approx_tokens=70000,
        context_length=100000,
        now=datetime.now(),
    )
    assert already_nudged is None


def test_session_entry_round_trips_health_nudge_fields():
    entry = _make_entry(
        session_health_nudged=True,
        session_health_nudge_reason="messages",
    )

    restored = SessionEntry.from_dict(entry.to_dict())

    assert restored.session_health_nudged is True
    assert restored.session_health_nudge_reason == "messages"
