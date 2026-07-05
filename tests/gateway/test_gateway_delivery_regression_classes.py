"""Executable regressions from t_c4ebf796 gateway-delivery checklist.

The checklist started as a staged OpenClaw-derived prompt.  These tests pin the
Hermes contracts that prevent the same failure classes in the live gateway.
"""
from cron import scheduler as cron_scheduler
from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
    _thread_metadata_for_source,
)
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _telegram_source(*, chat_type="group", thread_id="17", message_id="9001"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003912140421",
        chat_type=chat_type,
        thread_id=thread_id,
        message_id=message_id,
    )


def test_quoted_forum_reply_routes_by_topic_not_reply_anchor():
    """A quoted/threaded forum message must produce one topic-routed reply path.

    Telegram forum topics are routed by ``message_thread_id`` metadata.  Replying
    to the triggering/quoted message as well would create a second routing
    surface and risks duplicate or ambient bot-only continuations.
    """
    source = _telegram_source(chat_type="group", thread_id="17", message_id="9001")
    event = MessageEvent(
        text="Hermes, check this",
        message_type=MessageType.TEXT,
        source=source,
        message_id="9001",
        reply_to_message_id="8888",
    )

    assert _thread_metadata_for_source(source) == {"thread_id": "17"}
    assert _reply_anchor_for_event(event) is None


def test_private_dm_topic_requires_single_visible_reply_anchor():
    """DM topic fallback uses the triggering user message as its reply anchor."""
    source = _telegram_source(chat_type="dm", thread_id="42", message_id="9001")
    event = MessageEvent(
        text="continue",
        message_type=MessageType.TEXT,
        source=source,
        message_id="9001",
        reply_to_message_id="seed-message",
    )

    metadata = _thread_metadata_for_source(source)

    assert metadata == {
        "thread_id": "42",
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": "42",
        "telegram_reply_to_message_id": "9001",
    }
    assert _reply_anchor_for_event(event) == "9001"
    assert TelegramAdapter._thread_kwargs_for_send(
        "8059005725",
        "42",
        metadata,
        reply_to_message_id=9001,
    ) == {"message_thread_id": 42}


def test_background_delivery_error_stays_separate_from_process_success(monkeypatch):
    """A completed job with a delivery failure is not reported as execution failure."""
    calls = []

    monkeypatch.setattr(
        cron_scheduler,
        "run_job",
        lambda job: (True, "full output", "artifact summary", None),
    )
    monkeypatch.setattr(cron_scheduler, "save_job_output", lambda jid, out: "/tmp/job-output.md")
    monkeypatch.setattr(
        cron_scheduler,
        "_deliver_result",
        lambda job, content, adapters=None, loop=None: "telegram unavailable",
    )
    monkeypatch.setattr(
        cron_scheduler,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: calls.append(
            (jid, ok, err, delivery_error)
        ),
    )

    assert cron_scheduler.run_one_job({"id": "job-1", "name": "background artifact report"}) is True
    assert calls == [("job-1", True, None, "telegram unavailable")]
