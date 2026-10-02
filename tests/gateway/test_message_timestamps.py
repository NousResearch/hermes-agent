from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from hermes_time import safe_strftime

from gateway.message_timestamps import (
    render_user_content_with_timestamp,
)


BERLIN = ZoneInfo("Europe/Berlin")


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


@pytest.mark.parametrize("epoch", [1.0, 1_000_000_000.0])
def test_render_numeric_timestamp_preserves_instant_in_system_timezone(epoch):
    # Epoch 1 is still in 1969 west of UTC. Windows rejects a naive
    # astimezone() conversion there, although the Unix timestamp is positive.
    local = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()
    prefix = safe_strftime(local, "%a %Y-%m-%d %H:%M:%S %Z")

    assert render_user_content_with_timestamp("hello", epoch) == f"[{prefix}] hello"


def test_render_user_content_deduplicates_existing_timestamp_and_preserves_embedded_time():
    db_processing_ts = _epoch(2026, 4, 27, 15, 55, 36)
    stored_content = (
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    rendered = render_user_content_with_timestamp(
        stored_content,
        db_processing_ts,
        tz=BERLIN,
    )

    assert rendered == stored_content
    assert rendered.count("2026-04-27") == 1


# ---------------------------------------------------------------------------
# Opt-in gate: gateway.message_timestamps.enabled (default OFF)
# ---------------------------------------------------------------------------




def test_build_history_injects_only_when_enabled():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "hello", "timestamp": _epoch(2026, 4, 28, 13, 40, 53)},
        {"role": "assistant", "content": "hi"},
    ]

    # Default (off): user content stays clean, no timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history)
    assert agent_history[0]["content"] == "hello"

    # Enabled: user content gets exactly one timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    assert agent_history[0]["content"].startswith("[")
    assert agent_history[0]["content"].endswith("hello")
    # Assistant message is never timestamped.
    assert agent_history[1]["content"] == "hi"
