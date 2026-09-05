"""Regression test for #103429 — queued Telegram follow-ups lose the reply anchor.

When ``busy_input_mode: queue`` answers a follow-up that arrived mid-turn, the
follow-up's answer (and the typing/reaction lane for it) is routed through
``_thread_metadata_for_target`` with the follow-up's own anchor as
``reply_to_message_id``. For a plain (thread-less) Telegram DM that helper
returns ``None`` outright — the anchor is dropped before the adapter ever sees
it — so the answer quotes the turn-opening message instead of the follow-up.

Contract: a thread-less target that carries an explicit reply anchor must get
metadata that preserves the anchor (``telegram_reply_to_message_id`` on
Telegram) instead of ``None``.
"""

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


def _runner():
    # Bypass the heavy GatewayRunner.__init__ — the helper under test only
    # reads thread-routing state we do not touch here.
    return object.__new__(GatewayRunner)


def test_plain_telegram_dm_with_reply_anchor_carries_reply_to_metadata():
    """A thread-less Telegram DM answered with an explicit anchor must carry
    telegram_reply_to_message_id so the answer quotes the anchor."""
    runner = _runner()
    meta = runner._thread_metadata_for_target(
        Platform.TELEGRAM,
        "555",
        None,               # thread_id — a plain DM has none
        chat_type="dm",
        reply_to_message_id="202",
    )
    assert meta is not None, (
        "#103429: plain Telegram DM with an explicit reply anchor returned "
        "None metadata, so telegram_reply_to_message_id is never set and the "
        "answer quotes the turn-opening message instead of the follow-up"
    )
    assert meta.get("telegram_reply_to_message_id") == "202"


def test_plain_telegram_dm_without_anchor_still_returns_none():
    """No thread and no anchor => nothing to route on; keep the old None."""
    runner = _runner()
    assert runner._thread_metadata_for_target(
        Platform.TELEGRAM, "555", None, chat_type="dm",
    ) is None


def test_threadless_target_keeps_anchor_for_other_platforms():
    """A thread-less non-Telegram target with an explicit anchor must not
    lose it either (generic reply_to_message_id key)."""
    runner = _runner()
    meta = runner._thread_metadata_for_target(
        Platform.DISCORD,
        "999",
        None,
        reply_to_message_id="77",
    )
    assert meta is not None
    assert meta.get("reply_to_message_id") == "77"


def test_threaded_target_behavior_unchanged():
    """The existing thread path is untouched: thread metadata is produced as
    before and the anchor is preserved for Telegram DM topics."""
    runner = _runner()
    meta = runner._thread_metadata_for_target(
        Platform.TELEGRAM,
        "555",
        "42",
        chat_type="dm",
        reply_to_message_id="202",
    )
    assert meta is not None
    assert meta.get("thread_id") == "42"
    assert meta.get("telegram_reply_to_message_id") == "202"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
