from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.reminder_shortcuts import (
    extract_subject_from_reminder_text,
    is_reminder_intent,
    parse_simple_reminder_batch,
    parse_simple_relative_reminder,
    reminder_list_output,
)
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str, *, reply_to_text: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        reply_to_message_id="quoted1" if reply_to_text else None,
        reply_to_text=reply_to_text,
        source=SessionSource(
            platform=Platform.WHATSAPP,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace(send=MagicMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompt_pending = {}
    runner._last_reminder_subject_by_chat = {}
    runner._session_model_overrides = {}
    return runner


def test_parse_new_reminder_uses_only_current_line_not_old_history():
    text = """laundry in 1 minute
reminder in 2 minutes
laundry in 1 minute
reminder in 2 minutes
Remind me test 01 in 1 minute"""

    parsed = parse_simple_relative_reminder(text)

    assert parsed is not None
    assert parsed.subject == "test 01"
    assert parsed.schedule == "1m"
    assert parsed.display_time == "in 1 minute"


def test_reply_to_reminder_reuses_quoted_subject():
    parsed = parse_simple_relative_reminder(
        "Remind me in 2 min",
        reply_to_text="⚕ *Hermes Agent*\n────────────\n📅 REMINDER:\nlaundry",
    )

    assert parsed is not None
    assert parsed.subject == "laundry"
    assert parsed.schedule == "2m"
    assert parsed.display_time == "in 2 minutes"
    assert parsed.reused_reply_subject is True


def test_reply_to_reminder_next_week_reuses_quoted_subject():
    parsed = parse_simple_relative_reminder(
        "Remind me next week",
        reply_to_text="📅 REMINDER:\ncontact Ami warehouse",
    )

    assert parsed is not None
    assert parsed.subject == "contact Ami warehouse"
    assert parsed.schedule == "7d"
    assert parsed.display_time == "in 1 week"


def test_parse_voice_transcript_subject_after_delay():
    parsed = parse_simple_relative_reminder(
        '[The user sent a voice message~ Here\'s what they said: "Remind me in 2 minutes, test 03."]'
    )

    assert parsed is not None
    assert parsed.subject == "test 03"
    assert parsed.schedule == "2m"
    assert parsed.display_time == "in 2 minutes"



def test_voice_reply_to_reminder_reuses_quoted_subject():
    parsed = parse_simple_relative_reminder(
        '[The user sent a voice message~ Here\'s what they said: "Remind me in 2 minutes."]',
        reply_to_text="📅 REMINDER:\ncontact Ami warehouse",
    )

    assert parsed is not None
    assert parsed.subject == "contact Ami warehouse"
    assert parsed.schedule == "2m"
    assert parsed.display_time == "in 2 minutes"
    assert parsed.reused_reply_subject is True


def test_parse_text_subject_after_delay():
    parsed = parse_simple_relative_reminder("Remind me in 2 minutes, test 03.")

    assert parsed is not None
    assert parsed.subject == "test 03"
    assert parsed.schedule == "2m"


def test_parse_text_subject_after_delay_without_punctuation():
    parsed = parse_simple_relative_reminder("Remind me in 2 minutes test 03")

    assert parsed is not None
    assert parsed.subject == "test 03"
    assert parsed.schedule == "2m"


def test_parse_tomorrow_batch_creates_independent_absolute_reminders():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    parsed = parse_simple_reminder_batch(
        "Remind me tomorrow at 5am acrylic\nRemind me tomorrow at 5:05am graphics",
        now=datetime(2026, 6, 17, 18, 54, tzinfo=ZoneInfo("America/Chicago")),
    )

    assert parsed is not None
    assert [(r.subject, r.display_time) for r in parsed.reminders] == [
        ("acrylic", "Tomorrow 5:00 AM"),
        ("graphics", "Tomorrow 5:05 AM"),
    ]
    assert parsed.reminders[0].schedule == "2026-06-18T05:00:00-05:00"
    assert parsed.reminders[1].schedule == "2026-06-18T05:05:00-05:00"


def test_non_reminder_current_message_does_not_parse_even_with_old_history_terms():
    assert parse_simple_relative_reminder("test 02") is None
    assert parse_simple_relative_reminder("Here?") is None
    assert parse_simple_relative_reminder("What did you do to WhatsApp messaging?") is None
    assert not is_reminder_intent("test 02")


def test_parse_relative_reminder_can_parse_quoted_reminder_text_without_icon():
    parsed = parse_simple_relative_reminder(
        "Remind me in 2 min",
        reply_to_text="REMINDER:\ntest B2",
    )

    assert parsed is not None
    assert parsed.subject == "test B2"
    assert parsed.reused_reply_subject is True


@pytest.mark.asyncio
async def test_whatsapp_reply_without_quoted_text_uses_cached_subject(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": "job1"})
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [])

    runner = _make_runner()
    runner._last_reminder_subject_by_chat = {"15551234567@s.whatsapp.net": ("testB3", time.time())}

    result = await runner._handle_message(_make_event("Remind me in 2 min"))

    assert result == 'Done — rescheduled "testB3" to in 2 minutes. Reply "done" when handled.'
    assert created and created[0]["prompt"] == "📅 REMINDER:\ntestB3"


@pytest.mark.asyncio
async def test_whatsapp_reply_with_quoted_reminder_text_without_icon_uses_subject(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": "job1"})
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [])

    runner = _make_runner()
    result = await runner._handle_message(
        _make_event("Remind me in 2 min", reply_to_text="REMINDER:\ntestB3")
    )

    assert result == 'Done — rescheduled "testB3" to in 2 minutes. Reply "done" when handled.'
    assert created and created[0]["prompt"] == "📅 REMINDER:\ntestB3"


def test_explicit_reminder_uses_latest_current_line_only():
    parsed = parse_simple_relative_reminder(
        "📅 REMINDER:\nlaundry\nRemind me in 1 minute test 05"
    )

    assert parsed is not None
    assert parsed.subject == "test 05"
    assert parsed.schedule == "1m"


def test_extract_subject_from_reminder_text():
    assert extract_subject_from_reminder_text("📅 REMINDER:\nquotation Amin") == "quotation Amin"


def test_reminder_list_output_labels_once_and_recurring():
    jobs = [
        {
            "name": "Reminder: test a2",
            "prompt": "📅 REMINDER:\ntest a2",
            "schedule_display": "Today 9:47 PM",
            "schedule": {"kind": "once"},
            "repeat": {"times": 1, "completed": 0},
            "next_run_at": "2026-06-17T21:47:00-05:00",
        },
        {
            "name": "Reminder: Zepbound Shot",
            "prompt": "📅 REMINDER:\nZepbound Shot",
            "schedule_display": "Every Wednesday 4:00 PM",
            "schedule": {"kind": "cron"},
            "repeat": {"times": None, "completed": 0},
            "next_run_at": "2026-06-24T16:00:00-05:00",
        },
    ]

    output = reminder_list_output(jobs)

    assert output == (
        "Here are your future reminders:\n"
        "Once — Today 9:47 PM — test a2\n"
        "Recurring — Every Wednesday 4:00 PM — Zepbound Shot"
    )


@pytest.mark.asyncio
async def test_whatsapp_reminder_list_shortcut_returns_labelled_jobs(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])
    monkeypatch.setattr(
        "cron.jobs.list_jobs",
        lambda include_disabled=False: [
            {
                "name": "Reminder: test a2",
                "prompt": "📅 REMINDER:\ntest a2",
                "schedule_display": "Today 9:47 PM",
                "schedule": {"kind": "once"},
                "repeat": {"times": 1, "completed": 0},
                "next_run_at": "2026-06-17T21:47:00-05:00",
            },
            {
                "name": "Reminder: Zepbound Shot",
                "prompt": "📅 REMINDER:\nZepbound Shot",
                "schedule_display": "Every Wednesday 4:00 PM",
                "schedule": {"kind": "cron"},
                "repeat": {"times": None, "completed": 0},
                "next_run_at": "2026-06-24T16:00:00-05:00",
            },
        ],
    )

    runner = _make_runner()
    result = await runner._handle_message(_make_event("show me the list of all future reminders you have on record"))

    assert result == (
        "Here are your future reminders:\n"
        "Once — Today 9:47 PM — test a2\n"
        "Recurring — Every Wednesday 4:00 PM — Zepbound Shot"
    )


@pytest.mark.asyncio
async def test_whatsapp_simple_reminder_shortcut_creates_one_job(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []

    def _fake_create_job(**kwargs):
        created.append(kwargs)
        return {"id": "job1", **kwargs}

    monkeypatch.setattr("cron.jobs.create_job", _fake_create_job)

    runner = _make_runner()
    result = await runner._handle_message(_make_event("Remind me test 01 in 1 minute"))

    assert result == "Done. Reminder set: in 1 minute — test 01."
    assert len(created) == 1
    assert created[0]["prompt"] == "📅 REMINDER:\ntest 01"
    assert created[0]["schedule"] == "1m"
    assert created[0]["name"] == "Reminder: test 01"
    assert created[0]["deliver"] == "origin"


@pytest.mark.asyncio
async def test_whatsapp_reply_reminder_shortcut_reuses_subject(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": "job1"})
    # Reply path scans pending jobs to reschedule in place; no live job here.
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [])

    runner = _make_runner()
    result = await runner._handle_message(
        _make_event(
            "Remind me in 2 min",
            reply_to_text="📅 REMINDER:\nlaundry",
        )
    )

    # Redesign: a reply to a reminder reschedules THAT reminder (keeping its
    # subject) rather than "reusing" it as a fresh creation.
    assert result == 'Done — rescheduled "laundry" to in 2 minutes. Reply "done" when handled.'
    assert len(created) == 1
    assert created[0]["prompt"] == "📅 REMINDER:\nlaundry"
    assert created[0]["schedule"] == "2m"


@pytest.mark.asyncio
async def test_whatsapp_tomorrow_batch_shortcut_creates_two_jobs(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": f"job{len(created)}"})

    from datetime import datetime
    from zoneinfo import ZoneInfo
    import gateway.reminder_shortcuts as shortcuts

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = datetime(2026, 6, 17, 18, 54, tzinfo=ZoneInfo("America/Chicago"))
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(shortcuts, "datetime", FixedDateTime)

    runner = _make_runner()
    result = await runner._handle_message(
        _make_event("Remind me tomorrow at 5am acrylic\nRemind me tomorrow at 5:05am graphics")
    )

    assert result == (
        "Done. Reminder set: Tomorrow 5:00 AM — acrylic.\n"
        "Done. Reminder set: Tomorrow 5:05 AM — graphics."
    )
    assert len(created) == 2
    assert [job["name"] for job in created] == ["Reminder: acrylic", "Reminder: graphics"]
    assert [job["schedule"] for job in created] == [
        "2026-06-18T05:00:00-05:00",
        "2026-06-18T05:05:00-05:00",
    ]
    assert [job["prompt"] for job in created] == [
        "📅 REMINDER:\nacrylic",
        "📅 REMINDER:\ngraphics",
    ]


# ── Redesign: reply-to-reminder = reschedule/snooze/close ──────────────────────

from datetime import datetime as _dt
from gateway.reminder_shortcuts import (
    parse_snooze_reply,
    is_reminder_close_intent,
    ParsedSnooze,
)

_FIRED = "📅 REMINDER:\nReceipts Brussels airlines"
# Saturday 2026-06-20 14:09 local
_NOW = _dt(2026, 6, 20, 14, 9).astimezone()


@pytest.mark.parametrize(
    "reply,expect_schedule_kind,expect_display",
    [
        ("Remind me Monday at 5:35am", "iso", "Monday 5:35 AM"),
        ("Monday 5:35am", "iso", "Monday 5:35 AM"),
        ("Monday", "iso", "Monday 9:00 AM"),
        ("5m", "5m", "in 5 minutes"),
        ("10m", "10m", "in 10 minutes"),
        ("30m", "30m", "in 30 minutes"),
        ("1h", "1h", "in 1 hour"),
        ("2 hours", "2h", "in 2 hours"),
        ("in 1 hour", "1h", "in 1 hour"),
        ("tomorrow", "iso", "Tomorrow 9:00 AM"),
        ("tomorrow at 5:35am", "iso", "Tomorrow 5:35 AM"),
        ("next week", "7d", "in 1 week"),
        ("1 day", "1d", "in 1 day"),
        ("3 days", "3d", "in 3 days"),
    ],
)
def test_parse_snooze_reply_inherits_subject_and_resolves_time(
    reply, expect_schedule_kind, expect_display
):
    parsed = parse_snooze_reply(reply, reply_to_text=_FIRED, now=_NOW)
    assert parsed is not None, f"reply {reply!r} should parse as a snooze"
    assert parsed.subject == "Receipts Brussels airlines"
    assert parsed.display_time == expect_display
    if expect_schedule_kind == "iso":
        # ISO timestamp for weekday/clock targets.
        assert "T" in parsed.schedule
    else:
        assert parsed.schedule == expect_schedule_kind


def test_parse_snooze_reply_requires_inherited_subject():
    # No quote and no fallback -> cannot identify the reminder -> None.
    assert parse_snooze_reply("Monday 5:35am", now=_NOW) is None
    # Fallback subject still works.
    parsed = parse_snooze_reply(
        "Monday 5:35am", reply_to_subject_fallback="Receipts Brussels airlines", now=_NOW
    )
    assert parsed is not None
    assert parsed.subject == "Receipts Brussels airlines"


@pytest.mark.parametrize(
    "word",
    ["done", "Done.", "completed", "complete", "received", "resolved",
     "cancel", "cancelled", "stop", "close", "finished", "ok done"],
)
def test_is_reminder_close_intent_recognizes_close_words(word):
    assert is_reminder_close_intent(word) is True


@pytest.mark.parametrize("word", ["Receipts Brussels airlines", "blah", "Monday", "1h", ""])
def test_is_reminder_close_intent_rejects_non_close(word):
    assert is_reminder_close_intent(word) is False


@pytest.mark.asyncio
async def test_reply_monday_time_reschedules_keeping_subject(monkeypatch):
    """The exact reported failure: reply 'Remind me Monday at 5:35am' to a fired
    reminder must reschedule THAT reminder, keep its text, and not clarify."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    created = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": "jobX"})
    # Fired one-shot reminders are deleted on fire -> no live job to update.
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [])

    runner = _make_runner()
    result = await runner._handle_message(
        _make_event("Remind me Monday at 5:35am", reply_to_text=_FIRED)
    )

    assert result is not None
    assert "Receipts Brussels airlines" in result
    assert "rescheduled" in result.lower()
    assert len(created) == 1
    assert created[0]["prompt"] == "📅 REMINDER:\nReceipts Brussels airlines"
    assert "T05:35" in created[0]["schedule"]


@pytest.mark.asyncio
async def test_reply_reschedules_pending_job_in_place_no_duplicate(monkeypatch):
    """When the reminder is still pending, the reply updates it in place rather
    than creating a duplicate job."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    pending = {
        "id": "pending1",
        "prompt": "📅 REMINDER:\nReceipts Brussels airlines",
        "origin": {"chat_id": "15551234567@s.whatsapp.net"},
    }
    created = []
    updates = []
    monkeypatch.setattr("cron.jobs.create_job", lambda **kwargs: created.append(kwargs) or {"id": "new"})
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [pending])
    monkeypatch.setattr(
        "cron.jobs.update_job",
        lambda job_id, upd: updates.append((job_id, upd)) or {"id": job_id, **upd},
    )

    runner = _make_runner()
    result = await runner._handle_message(_make_event("1h", reply_to_text=_FIRED))

    assert result is not None and "rescheduled" in result.lower()
    # Updated in place, NOT duplicated.
    assert updates == [("pending1", {"schedule": "1h"})]
    assert created == []


@pytest.mark.asyncio
async def test_reply_done_closes_reminder(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    pending = {
        "id": "pending1",
        "prompt": "📅 REMINDER:\nReceipts Brussels airlines",
        "origin": {"chat_id": "15551234567@s.whatsapp.net"},
    }
    removed = []
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [pending])
    monkeypatch.setattr("cron.jobs.remove_job", lambda job_id: removed.append(job_id) or True)

    runner = _make_runner()
    result = await runner._handle_message(_make_event("done", reply_to_text=_FIRED))

    assert result is not None
    assert "Closed" in result
    assert "Receipts Brussels airlines" in result
    assert removed == ["pending1"]


@pytest.mark.asyncio
async def test_bare_done_without_reply_does_not_close(monkeypatch):
    """A standalone 'done' with no reminder reply context must not hijack."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])

    removed = []
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=False: [])
    monkeypatch.setattr("cron.jobs.remove_job", lambda job_id: removed.append(job_id) or True)

    runner = _make_runner()
    # No reply_to_text -> reminder reply handler must return None (fall through).
    handled = runner._try_handle_reminder_reply(_make_event("done"))
    assert handled is None
    assert removed == []
