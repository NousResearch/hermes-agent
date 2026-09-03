"""Lease-contention lifecycle statuses respect interim_assistant_messages mute.

#94658: "Another Hermes process is using this session", "Still waiting …"
and "Session is free …" are emitted via ``_emit_status`` →
``status_callback("lifecycle", …)`` unconditionally. On chat gateways they
bypassed ``display.platforms.*.interim_assistant_messages: false`` because
``_prepare_gateway_status_message`` had no interim-setting gate. Routine
compression progress keeps its own opt-in gate (``compression.progress_notices``)
and must NOT be suppressed by the interim mute.
"""

import pytest

from agent.conversation_compression import ROUTINE_COMPRESSION_STATUS_SAMPLES
from gateway.run import _prepare_gateway_status_message

# Exact lease-wait strings from run_agent.py acquire_session_turn_lease flow
# (plus the post-acquire "Session is free" status).
LEASE_WAIT_LIFECYCLE = [
    (
        "⏳ Another Hermes process is using this session; "
        "waiting for it to finish before starting your turn..."
    ),
    "⏳ Still waiting for the other Hermes process on this session (42s)...",
    "🎭 Session is free; loading the latest transcript...",
]

CHAT_PLATFORMS = ["telegram", "discord", "slack", "whatsapp"]


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("message", LEASE_WAIT_LIFECYCLE, ids=lambda m: m[:28])
def test_lease_wait_suppressed_when_interim_muted(platform, message):
    """interim_assistant_messages: false silences lease-contention chatter."""
    assert (
        _prepare_gateway_status_message(
            platform,
            "lifecycle",
            message,
            interim_assistant_messages_enabled=False,
        )
        is None
    ), f"lease-wait lifecycle must be suppressed for {platform}"


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize("message", LEASE_WAIT_LIFECYCLE, ids=lambda m: m[:28])
def test_lease_wait_delivered_when_interim_enabled(platform, message):
    """interim_assistant_messages: true (default) keeps the status visible."""
    assert (
        _prepare_gateway_status_message(
            platform,
            "lifecycle",
            message,
            interim_assistant_messages_enabled=True,
        )
        == message
    )
    # Default (None) keeps legacy behavior byte-identical.
    assert (
        _prepare_gateway_status_message(platform, "lifecycle", message)
        == message
    )


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_compression_progress_not_suppressed_by_interim_mute(monkeypatch, platform):
    """Routine compression progress keeps its own opt-in gate (#52995)."""
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"compression": {"progress_notices": True}},
    )
    for message in ROUTINE_COMPRESSION_STATUS_SAMPLES:
        # When progress_notices is open, interim mute must not swallow it.
        assert (
            _prepare_gateway_status_message(
                platform,
                "lifecycle",
                message,
                interim_assistant_messages_enabled=False,
            )
            == message
        ), f"compression progress must survive interim mute on {platform}"


@pytest.mark.parametrize("message", LEASE_WAIT_LIFECYCLE, ids=lambda m: m[:28])
def test_lease_wait_raw_surfaces_unaffected(message):
    """Programmatic surfaces keep raw text regardless of the interim mute."""
    for platform in ("local", "api_server", "webhook"):
        assert (
            _prepare_gateway_status_message(
                platform,
                "lifecycle",
                message,
                interim_assistant_messages_enabled=False,
            )
            == message
        )
