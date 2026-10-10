"""Regression tests for #130510: handoff notices carry creation time."""

from datetime import datetime, timezone

from gateway.partner_handoff import agent_notice, format_handoff_created, human_notice


def _epoch(year, month, day, hour, minute, second, ms=0):
    return datetime(year, month, day, hour, minute, second, ms * 1000, tzinfo=timezone.utc).timestamp()


def test_human_notice_header_matches_proposed_format():
    ts = _epoch(2026, 10, 1, 13, 45, 12)
    text = human_notice("5b00c286deadbeef", "wren", "peer", "key is at ~/peer-keys/money.key", ts)
    assert "[Handoff 5b00c286 · created 2026-10-01T13:45:12.000Z · from your conversation with wren]" in text
    assert "key is at ~/peer-keys/money.key" in text


def test_agent_notice_carries_id_and_created():
    ts = _epoch(2026, 10, 1, 13, 49, 51, ms=399)
    text = agent_notice("287323dbdeadbeef", "wren", "peer", "HOLD — don't copy until Will decides", ts)
    assert "287323db" in text
    assert "created 2026-10-01T13:49:51.399Z" in text
    assert "HOLD" in text


def test_missing_ts_omits_created_segment_backward_compatible():
    # Old callers pass no ts: header keeps the pre-fix shape, no "created".
    text = human_notice("717cb738deadbeef", "wren", "peer", "new standing rule")
    assert "[Handoff 717cb738 · from your conversation with wren]" in text
    assert "created" not in text
    # Explicit None behaves the same.
    assert "created" not in human_notice("717cb738deadbeef", "wren", "peer", "x", None)
    assert "created" not in agent_notice("717cb738deadbeef", "wren", "peer", "x", None)


def test_ts_accepts_float_int_numeric_string_and_iso_string():
    epoch = _epoch(2026, 10, 1, 13, 45, 12)
    iso = "2026-10-01T13:45:12.000Z"
    for ts in (epoch, int(epoch), str(epoch), iso):
        assert format_handoff_created(ts) == iso
        assert f"created {iso}" in human_notice("5b00c28600", "wren", "p", "i", ts)
        assert f"created {iso}" in agent_notice("5b00c28600", "wren", "p", "i", ts)


def test_subsecond_precision_preserved_and_zero_millis_not_trimmed():
    assert format_handoff_created(_epoch(2026, 10, 1, 13, 43, 6, ms=401)) == "2026-10-01T13:43:06.401Z"
    assert format_handoff_created(_epoch(2026, 10, 1, 13, 45, 12)) == "2026-10-01T13:45:12.000Z"
    assert format_handoff_created("") is None
    assert format_handoff_created(None) is None


def test_receiver_can_recover_creation_order_despite_delivery_order():
    # Issue shape: 5b00c286 created second, delivered fourth — after the hold.
    # The created stamps must order opposite to delivery order.
    # Edge case: same second with .000Z vs .001Z to ensure constant-width
    # string sorting matches chronological order.
    older = human_notice("5b00c28600", "wren", "p", "scp it", _epoch(2026, 10, 1, 13, 45, 12))
    newer = human_notice("287323db00", "wren", "p", "HOLD", _epoch(2026, 10, 1, 13, 45, 12, ms=1))
    assert "created 2026-10-01T13:45:12.000Z" in older
    assert "created 2026-10-01T13:45:12.001Z" in newer
    assert older.split("created ")[1] < newer.split("created ")[1]


def test_id_truncated_to_eight_and_empty_requester_falls_back():
    text = human_notice("abcdef1234567890", "", "peer", "intent", _epoch(2026, 10, 1, 13, 45, 12))
    assert "[Handoff abcdef12 ·" in text
    assert "from your conversation with peer]" in text
    text = human_notice("abcdef12", "", "", "intent", _epoch(2026, 10, 1, 13, 45, 12))
    assert "from your conversation with someone]" in text


def test_datetime_ts_supported():
    dt = datetime(2026, 10, 1, 13, 45, 12, tzinfo=timezone.utc)
    assert format_handoff_created(dt) == "2026-10-01T13:45:12.000Z"
    assert "created 2026-10-01T13:45:12.000Z" in human_notice("5b00c28600", "wren", "p", "i", dt)


def test_iso_string_without_millis_normalizes_to_constant_width():
    # Follow-up on #130673: a legacy no-millisecond ISO string must not pass
    # through as 20 chars, or lexicographic order breaks again.
    assert format_handoff_created("2026-10-01T13:45:12Z") == "2026-10-01T13:45:12.000Z"
    assert format_handoff_created("2026-10-01T13:45:12.001Z") == "2026-10-01T13:45:12.001Z"
    assert format_handoff_created("2026-10-01T13:45:12+00:00") == "2026-10-01T13:45:12.000Z"
    older = format_handoff_created("2026-10-01T13:45:12Z")
    newer = format_handoff_created("2026-10-01T13:45:12.001Z")
    assert older is not None and newer is not None
    assert older < newer
    # Truly unrecognized strings keep the documented passthrough.
    assert format_handoff_created("not-a-timestamp") == "not-a-timestamp"
