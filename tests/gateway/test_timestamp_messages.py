"""Tests for the per-message timestamp prefix feature.

When ``GatewayConfig.timestamp_messages`` is True, the gateway prepends each
inbound user message with an ISO-8601 timestamp (rendered in the host's
local timezone) before it enters the agent loop. This gives the model a
clock signal on every turn in long-running sessions where the system
prompt's "session start" timestamp is the only built-in clock.
"""

import re
from datetime import datetime, timezone, timedelta

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_ISO_PREFIX_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)\] "
)


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="504464026",
        chat_name="Simon",
        chat_type="private",
        user_name="Simon",
    )


@pytest.mark.asyncio
async def test_timestamp_prefix_added_when_enabled():
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
            timestamp_messages=True,
        )
    )
    # Fixed UTC time so we can assert the rendered offset is well-formed.
    fixed_ts = datetime(2026, 5, 15, 18, 25, 0, tzinfo=timezone.utc)
    source = _make_source()
    event = MessageEvent(text="hello", source=source, timestamp=fixed_ts)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert _ISO_PREFIX_RE.match(result), f"missing/malformed iso prefix: {result!r}"
    assert result.endswith(" hello")


@pytest.mark.asyncio
async def test_timestamp_prefix_skipped_when_disabled_by_default():
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
            # timestamp_messages defaults to False
        )
    )
    source = _make_source()
    event = MessageEvent(
        text="hello",
        source=source,
        timestamp=datetime(2026, 5, 15, 18, 25, 0, tzinfo=timezone.utc),
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_timestamp_prefix_preserves_numeric_offset():
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
            timestamp_messages=True,
        )
    )
    # Build a tz-aware datetime in a non-UTC zone (BST-equivalent +01:00) and
    # confirm the rendered prefix carries a numeric offset (could be either
    # the original or the host-local one — astimezone() converts to local).
    bst = timezone(timedelta(hours=1))
    event_ts = datetime(2026, 5, 15, 19, 25, 0, tzinfo=bst)
    source = _make_source()
    event = MessageEvent(text="hi", source=source, timestamp=event_ts)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    # Must contain a numeric offset (+HH:MM or -HH:MM); never a bare time.
    assert re.search(r"[+-]\d{2}:\d{2}\]", result), f"no numeric offset: {result!r}"
    assert result.endswith(" hi")


@pytest.mark.asyncio
async def test_timestamp_prefix_applies_before_sender_prefix_in_shared_group():
    """When both shared-group sender prefix and timestamp_messages are on,
    the timestamp wraps the already-prefixed message (one prefix per line)."""
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
            group_sessions_per_user=False,
            timestamp_messages=True,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(
        text="hello",
        source=source,
        timestamp=datetime(2026, 5, 15, 18, 25, 0, tzinfo=timezone.utc),
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert _ISO_PREFIX_RE.match(result), f"missing iso prefix: {result!r}"
    # The sender prefix should still be present, after the timestamp.
    assert "[Alice] hello" in result


@pytest.mark.asyncio
async def test_timestamp_prefix_survives_malformed_timestamp():
    """A bad timestamp must never block delivery — it should silently no-op."""
    runner = _make_runner(
        GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
            timestamp_messages=True,
        )
    )
    source = _make_source()
    # Provide an object that masquerades as a datetime but raises on astimezone.
    class _BadTs:
        tzinfo = None

        def astimezone(self, *args, **kwargs):
            raise RuntimeError("boom")

    event = MessageEvent(text="hello", source=source)
    event.timestamp = _BadTs()  # type: ignore[assignment]

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    # Original message comes through unchanged.
    assert result == "hello"
