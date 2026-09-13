"""Cron thread-loss warnings only apply to the origin conversation."""

from __future__ import annotations

import logging
from types import SimpleNamespace


def _prepare(job, target):
    from cron import scheduler_delivery as delivery

    original = delivery._resolve_target_transport
    delivery._resolve_target_transport = lambda *_args, **_kwargs: (
        ("standalone", SimpleNamespace(extra={}), None, {}), None
    )
    try:
        return delivery._prepare_target_delivery(
            job,
            target,
            adapters=None,
            loop=None,
            config=SimpleNamespace(),
            notify_delivery=True,
            mirror_enabled=False,
            mirror_text="report",
            delivery_errors=[],
        )
    finally:
        delivery._resolve_target_transport = original


def test_cross_platform_target_without_thread_is_not_a_lost_origin_thread(caplog):
    job = {
        "id": "daily-report",
        "deliver": "email",
        "origin": {"platform": "discord", "chat_id": "discord-thread-parent", "thread_id": "42"},
    }

    with caplog.at_level(logging.WARNING, logger="cron.scheduler_delivery"):
        target = _prepare(job, {"platform": "email", "chat_id": "grant@example.com", "thread_id": None})

    assert target is not None
    assert "delivery target lost it" not in caplog.text


def test_same_conversation_target_without_thread_still_warns(caplog):
    job = {
        "id": "daily-report",
        "deliver": "discord",
        "origin": {"platform": "discord", "chat_id": "discord-thread-parent", "thread_id": "42"},
    }

    with caplog.at_level(logging.WARNING, logger="cron.scheduler_delivery"):
        target = _prepare(job, {"platform": "discord", "chat_id": "discord-thread-parent", "thread_id": None})

    assert target is not None
    assert "delivery target lost it" in caplog.text
