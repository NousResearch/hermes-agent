"""Tests for WhatsApp bridge read-receipt behavior."""

from pathlib import Path


BRIDGE_JS = Path(__file__).resolve().parents[2] / "scripts" / "whatsapp-bridge" / "bridge.js"


def test_bridge_marks_messages_read_after_gating_and_before_queueing():
    """Only messages that will be queued for Hermes should be marked read.

    The bridge gives senders WhatsApp read receipts (blue ticks / seen) once it
    has accepted the message for Hermes. The read-receipt call must run *after*
    the empty-body / unsupported-content gating (so skipped messages are not
    marked read) but *before* the message is pushed onto the queue.
    """
    source = BRIDGE_JS.read_text(encoding="utf-8")

    read_call = "await sock.readMessages([msg.key]);"
    queue_call = "messageQueue.push(event);"
    empty_gate = "if (!body && !hasMedia) {"

    assert read_call in source
    assert empty_gate in source
    # Read receipt must happen after the empty/unsupported-content gating...
    assert source.index(empty_gate) < source.index(read_call)
    # ...and before the message is queued for the agent.
    assert source.index(read_call) < source.index(queue_call)
