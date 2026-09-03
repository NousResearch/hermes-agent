"""Tests for Matrix adapter phantom self-loop filtering.

Phantom interrupt notices (text emitted by the gateway's own orchestrator
or by upstream runtime scaffolding) leak into the chat surface as if they
were user messages. When the bot's own gateway picks them back up via
``_on_room_message`` they are dispatched as new user turns, triggering
another interrupt notice, forming a self-perpetuating loop.

Filter these at intake in ``_on_room_message`` so the loop is broken
without relying on the model's own silence-token detection.

Phantom shapes observed when a misconfigured bot fell into a self-loop
emitting these messages in rapid succession until rate-limited:

  - ``[This response was interrupted by a user correction.]``
  - ``↪ Redirected current run (iteration N/500). I'll adjust using your correction.``
  - ``⚡ Interrupting current task. I'll respond to your message shortly.``
  - ``⚡ Stopped. You can continue this session.``
  - ``No active task to stop.``
  - ``♻️ Recovered reply ... the gateway restarted during delivery...``
  - ``💾 Self-improvement review: ...``
  - ``💭 Reasoning: ...``

Some of these (``♻️ Recovered reply``, ``↪ Redirected``, ``⚡
Interrupting``) are wrapped in OOB control messages that *should* be
display_kind=hidden; their appearance in chat bodies is an upstream bug.
For the matrix adapter we defend at the intake gate so the bot stops
feeding its own phantom notices back into the model.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import time

import pytest


PHANTOM_BODIES = [
    "[This response was interrupted by a user correction.]",
    "↪ Redirected current run (iteration 3/500). I'll adjust using your correction.",
    "⚡ Interrupting current task. I'll respond to your message shortly.",
    # Status-detail variant (running: <tool>) — same "Interrupting current task"
    # prefix as the period form, but followed by " (running: web_search)."
    # rather than "." or ":". The bare-prefix gate covers this; if it
    # regresses to require trailing punctuation the line below leaks through.
    "⚡ Interrupting current task (running: web_search). I'll respond to your message shortly.",
    "⚡ Stopped. You can continue this session.",
    "No active task to stop.",
    "♻️ Recovered reply — the gateway restarted during delivery, retrying.",
    "💾 Self-improvement review: User profile updated (compressed).",
    "💭 Reasoning: checking the sender field before responding",
    # Edge case: phantom text appearing as a substring of a real message
    # should NOT be filtered — the gate only matches full-prefix phantoms.
    # Covered by the negative test below.
]


def _make_adapter():
    """Same minimal adapter harness as test_matrix_message_event_metadata."""
    import os

    os.environ["MATRIX_REQUIRE_MENTION"] = "false"
    os.environ["MATRIX_AUTO_THREAD"] = "false"

    from plugins.platforms.matrix.adapter import MatrixAdapter

    from gateway.config import PlatformConfig

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._client = None
    identity = SimpleNamespace(
        display_name="Test Room",
        room_topic=None,
        server_name="example.org",
        chat_type="dm",
    )
    adapter._resolve_room_identity = AsyncMock(return_value=identity)
    return adapter


def _make_event(body, sender="@alice:example.org", event_id="$evt1"):
    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id="!room1:example.org",
        timestamp=int(time.time() * 1000),
        content={"body": body, "msgtype": "m.text"},
    )


# ---------------------------------------------------------------------------
# Phantom intake filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phantom_interrupt_text_is_dropped_at_intake():
    """A phantom interrupt message must NOT be dispatched to handle_message."""
    adapter = _make_adapter()
    adapter._startup_ts = time.time() - 10

    event = _make_event("[This response was interrupted by a user correction.]")
    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("phantom_body", PHANTOM_BODIES)
async def test_all_phantom_shapes_are_dropped(phantom_body):
    """Every documented phantom shape must be filtered.

    Parametrized so a regression on any single shape fails loudly with the
    specific body that leaked through.
    """
    adapter = _make_adapter()
    adapter._startup_ts = time.time() - 10

    event = _make_event(phantom_body)
    await adapter._on_room_message(event)

    assert adapter.handle_message.await_count == 0, (
        f"phantom body leaked through: {phantom_body!r}"
    )


@pytest.mark.asyncio
async def test_real_message_with_phantom_substring_passes():
    """Substring phantom text inside a real message must NOT be filtered.

    Only full-prefix phantom shapes (or messages whose first non-whitespace
    token is a phantom marker) are filtered. A user message like
    '⚡ Interrupting my workout to ask you this...' must still reach the
    model. This test pins that the filter is precise, not greedy.
    """
    adapter = _make_adapter()
    adapter._startup_ts = time.time() - 10

    event = _make_event(
        "⚡ Interrupting my morning to ask — should I deploy the patch now?"
    )
    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text.startswith("⚡ Interrupting my morning")


@pytest.mark.asyncio
async def test_plain_user_message_still_dispatched():
    """The filter must not regress normal user text."""
    adapter = _make_adapter()
    adapter._startup_ts = time.time() - 10

    event = _make_event("hey hermes, what's the disk usage on the K3s nodes?")
    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    msg = adapter.handle_message.await_args.args[0]
    assert msg.text == "hey hermes, what's the disk usage on the K3s nodes?"