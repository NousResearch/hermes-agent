from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gateway.whatsapp_message_store import (
    append_whatsapp_record,
    query_whatsapp_records,
)


def _record(
    *,
    record_id: str,
    effective_event_at_utc: str,
    record_sequence: int,
    conversation_key: str,
    destination_key: str,
    destination_context_type: str,
    direction: str,
    group_chat_id: str | None = None,
    dm_counterparty_id: str | None = None,
    record_kind: str = "message_record",
) -> dict[str, object]:
    return {
        "record_kind": record_kind,
        "record_id": record_id,
        "conversation_key": conversation_key,
        "destination_key": destination_key,
        "destination_context_type": destination_context_type,
        "group_chat_id": group_chat_id,
        "dm_counterparty_id": dm_counterparty_id,
        "direction": direction,
        "effective_event_at_utc": effective_event_at_utc,
        "record_sequence": record_sequence,
    }


def _append(base_dir, record: dict[str, object], effective_event_at: datetime) -> None:
    append_whatsapp_record(
        record, effective_event_at=effective_event_at, base_dir=base_dir
    )


def test_query_whatsapp_records_requires_destination_scope(tmp_path):
    with pytest.raises(ValueError, match="requires at least one destination scope"):
        query_whatsapp_records(
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 2, tzinfo=timezone.utc),
            base_dir=tmp_path,
        )


def test_query_whatsapp_records_scans_overlapping_days_and_orders_chronologically(
    tmp_path,
):
    base_dir = tmp_path / "records"

    _append(
        base_dir,
        _record(
            record_id="late-day-one",
            effective_event_at_utc="2024-06-01T23:59:50Z",
            record_sequence=50,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="inbound",
        ),
        datetime(2024, 6, 1, 23, 59, 50, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="same-time-lower-seq",
            effective_event_at_utc="2024-06-02T00:00:10Z",
            record_sequence=10,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="outbound",
        ),
        datetime(2024, 6, 2, 0, 0, 10, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="same-time-higher-seq",
            effective_event_at_utc="2024-06-02T00:00:10Z",
            record_sequence=30,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="inbound",
        ),
        datetime(2024, 6, 2, 0, 0, 10, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="excluded-other-destination",
            effective_event_at_utc="2024-06-02T00:00:20Z",
            record_sequence=1,
            conversation_key="whatsapp:dm:19998887777",
            destination_key="whatsapp:dm:19998887777",
            destination_context_type="direct_message",
            dm_counterparty_id="19998887777",
            direction="inbound",
        ),
        datetime(2024, 6, 2, 0, 0, 20, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="excluded-exclusive-upper-bound",
            effective_event_at_utc="2024-06-02T00:01:00Z",
            record_sequence=99,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="outbound",
        ),
        datetime(2024, 6, 2, 0, 1, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="excluded-non-message-row",
            effective_event_at_utc="2024-06-02T00:00:15Z",
            record_sequence=5,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="outbound",
            record_kind="delivery_event",
        ),
        datetime(2024, 6, 2, 0, 0, 15, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="excluded-non-overlap-day",
            effective_event_at_utc="2024-06-03T00:00:01Z",
            record_sequence=1,
            conversation_key="whatsapp:dm:15551230000",
            destination_key="whatsapp:dm:15551230000",
            destination_context_type="direct_message",
            dm_counterparty_id="15551230000",
            direction="inbound",
        ),
        datetime(2024, 6, 3, 0, 0, 1, tzinfo=timezone.utc),
    )

    records = query_whatsapp_records(
        datetime(2024, 6, 1, 23, 59, tzinfo=timezone.utc),
        datetime(2024, 6, 2, 0, 1, tzinfo=timezone.utc),
        base_dir=base_dir,
        conversation_key="whatsapp:dm:15551230000",
    )

    assert [record["record_id"] for record in records] == [
        "late-day-one",
        "same-time-lower-seq",
        "same-time-higher-seq",
    ]


def test_query_whatsapp_records_filters_by_canonical_destination_fields_and_direction(
    tmp_path,
):
    base_dir = tmp_path / "records"

    _append(
        base_dir,
        _record(
            record_id="group-inbound",
            effective_event_at_utc="2024-06-02T09:00:00Z",
            record_sequence=1,
            conversation_key="whatsapp:group:group-123@g.us",
            destination_key="whatsapp:group:group-123@g.us",
            destination_context_type="group_chat",
            group_chat_id="group-123@g.us",
            direction="inbound",
        ),
        datetime(2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="group-outbound",
            effective_event_at_utc="2024-06-02T09:05:00Z",
            record_sequence=2,
            conversation_key="whatsapp:group:group-123@g.us",
            destination_key="whatsapp:group:group-123@g.us",
            destination_context_type="group_chat",
            group_chat_id="group-123@g.us",
            direction="outbound",
        ),
        datetime(2024, 6, 2, 9, 5, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="dm-outbound",
            effective_event_at_utc="2024-06-02T10:00:00Z",
            record_sequence=3,
            conversation_key="whatsapp:dm:15551234567",
            destination_key="whatsapp:dm:15551234567",
            destination_context_type="direct_message",
            dm_counterparty_id="15551234567",
            direction="outbound",
        ),
        datetime(2024, 6, 2, 10, 0, 0, tzinfo=timezone.utc),
    )

    group_records = query_whatsapp_records(
        datetime(2024, 6, 2, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 6, 3, 0, 0, 0, tzinfo=timezone.utc),
        base_dir=base_dir,
        group_chat_id="group-123@g.us",
        direction="outbound",
    )
    dm_records = query_whatsapp_records(
        datetime(2024, 6, 2, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 6, 3, 0, 0, 0, tzinfo=timezone.utc),
        base_dir=base_dir,
        destination_key="whatsapp:dm:15551234567",
        dm_counterparty_id="15551234567",
    )

    assert [record["record_id"] for record in group_records] == ["group-outbound"]
    assert [record["record_id"] for record in dm_records] == ["dm-outbound"]
