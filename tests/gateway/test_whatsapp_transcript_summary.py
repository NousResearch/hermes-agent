from __future__ import annotations

from datetime import datetime, timezone

from gateway.whatsapp_message_store import append_whatsapp_record
from gateway.whatsapp_transcript_summary import (
    format_whatsapp_transcript_summary,
    generate_whatsapp_transcript_summary,
)


def _append(base_dir, record: dict[str, object], effective_event_at: datetime) -> None:
    append_whatsapp_record(
        record,
        effective_event_at=effective_event_at,
        base_dir=base_dir,
    )


def _record(
    *,
    record_id: str,
    effective_event_at_utc: str,
    record_sequence: int,
    participant_role: str,
    message_classification: str,
    text: str | None,
    sender_id: str | None = None,
    sender_name: str | None = None,
    media_types: list[str] | None = None,
    record_kind: str = "conversation_record",
) -> dict[str, object]:
    return {
        "record_kind": record_kind,
        "record_id": record_id,
        "conversation_key": "whatsapp:dm:15551230000",
        "destination_key": "whatsapp:dm:15551230000",
        "destination_context_type": "direct_message",
        "destination_chat_id": "15551230000@s.whatsapp.net",
        "destination_target_id": "15551230000",
        "group_chat_id": None,
        "dm_counterparty_id": "15551230000",
        "direction": "inbound" if participant_role != "agent" else "outbound",
        "effective_event_at_utc": effective_event_at_utc,
        "record_sequence": record_sequence,
        "participant_role": participant_role,
        "message_classification": message_classification,
        "command_authority_scope": "owner_only"
        if participant_role == "owner_operator"
        else "none",
        "sender_id": sender_id,
        "sender_name": sender_name,
        "text": text,
        "media_types": media_types or [],
    }


def test_generate_whatsapp_transcript_summary_separates_operator_context_and_recap(
    tmp_path,
):
    base_dir = tmp_path / "records"

    _append(
        base_dir,
        _record(
            record_id="operator-1",
            effective_event_at_utc="2024-06-02T09:00:00Z",
            record_sequence=1,
            participant_role="owner_operator",
            message_classification="command_capable",
            text="Summarize this supplier thread for me.",
            sender_id="15550000001",
            sender_name="Founder",
        ),
        datetime(2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="external-1",
            effective_event_at_utc="2024-06-02T09:01:00Z",
            record_sequence=2,
            participant_role="external_party",
            message_classification="conversational_only",
            text=(
                "We can deliver 200 units by Friday. Do you still need the "
                "revised quote?"
            ),
            sender_id="15551230000",
            sender_name="Vendor",
        ),
        datetime(2024, 6, 2, 9, 1, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="agent-1",
            effective_event_at_utc="2024-06-02T09:02:00Z",
            record_sequence=3,
            participant_role="agent",
            message_classification="conversational_only",
            text=(
                "Yes — please send the revised quote and confirm the final unit price."
            ),
            sender_id="agent",
            sender_name="Hermes",
        ),
        datetime(2024, 6, 2, 9, 2, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="external-media",
            effective_event_at_utc="2024-06-02T09:03:00Z",
            record_sequence=4,
            participant_role="external_party",
            message_classification="conversational_only",
            text=None,
            sender_id="15551230000",
            sender_name="Vendor",
            media_types=["image"],
        ),
        datetime(2024, 6, 2, 9, 3, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="edit-outcome",
            effective_event_at_utc="2024-06-02T09:04:00Z",
            record_sequence=5,
            participant_role="agent",
            message_classification="conversational_only",
            text="ignored edit",
            record_kind="edit_outcome",
        ),
        datetime(2024, 6, 2, 9, 4, 0, tzinfo=timezone.utc),
    )

    summary = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
            "include_operator_context": True,
        },
        authorized=True,
        base_dir=base_dir,
    )

    assert summary["summary_status"] == "ready"
    assert summary["coverage_start_utc"] == "2024-06-02T09:00:00Z"
    assert summary["coverage_end_utc"] == "2024-06-02T09:03:00Z"
    assert summary["covered_record_count"] == 4
    assert summary["operator_context"] == [
        "Founder: Summarize this supplier thread for me."
    ]
    assert summary["conversation_recap"] == [
        (
            "Vendor: We can deliver 200 units by Friday. Do you still need the "
            "revised quote?"
        ),
        (
            "Hermes: Yes — please send the revised quote and confirm the final "
            "unit price."
        ),
        "Vendor sent non-text media (image)",
    ]
    assert summary["open_items"] == [
        (
            "Unresolved question from Vendor: We can deliver 200 units by "
            "Friday. Do you still need the revised quote?"
        )
    ]
    assert summary["uncertainties"] == [
        (
            "At least one non-text media turn was preserved without attachment "
            "contents, so the summary only reports that media was exchanged."
        )
    ]
    assert (
        "WhatsApp transcript summary for whatsapp:dm:15551230000"
        in summary["narrative_text"]
    )
    assert "Vendor" in summary["narrative_text"]
    assert "Hermes" in summary["narrative_text"]


def test_generate_whatsapp_transcript_summary_can_exclude_operator_context(tmp_path):
    base_dir = tmp_path / "records"

    _append(
        base_dir,
        _record(
            record_id="operator-1",
            effective_event_at_utc="2024-06-02T09:00:00Z",
            record_sequence=1,
            participant_role="owner_operator",
            message_classification="command_capable",
            text="Summarize this supplier thread for me.",
        ),
        datetime(2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="external-1",
            effective_event_at_utc="2024-06-02T09:01:00Z",
            record_sequence=2,
            participant_role="external_party",
            message_classification="conversational_only",
            text="Please confirm whether you want delivery on Friday.",
            sender_name="Vendor",
        ),
        datetime(2024, 6, 2, 9, 1, 0, tzinfo=timezone.utc),
    )

    summary = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
            "include_operator_context": False,
        },
        authorized=True,
        base_dir=base_dir,
    )

    assert summary["summary_status"] == "ready"
    assert summary["operator_context"] == []
    assert summary["covered_record_count"] == 1
    assert summary["coverage_start_utc"] == "2024-06-02T09:01:00Z"


def test_generate_whatsapp_transcript_summary_returns_no_records_when_only_operator_rows_exist(
    tmp_path,
):
    base_dir = tmp_path / "records"
    _append(
        base_dir,
        _record(
            record_id="operator-only",
            effective_event_at_utc="2024-06-02T09:00:00Z",
            record_sequence=1,
            participant_role="owner_operator",
            message_classification="command_capable",
            text="Summarize the thread.",
        ),
        datetime(2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc),
    )

    summary = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
        },
        authorized=True,
        base_dir=base_dir,
    )

    assert summary["summary_status"] == "no_conversation_records"
    assert summary["covered_record_count"] == 0
    assert summary["conversation_recap"] == []


def test_generate_whatsapp_transcript_summary_rejects_invalid_request_and_unauthorized(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / "records"
    observed = {"queries": 0}

    def _fail_if_queried(*_args, **_kwargs):
        observed["queries"] += 1
        raise AssertionError("query_whatsapp_records should not run")

    monkeypatch.setattr(
        "gateway.whatsapp_transcript_summary.query_whatsapp_records",
        _fail_if_queried,
    )

    forbidden = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
        },
        authorized=False,
        base_dir=base_dir,
    )
    invalid = generate_whatsapp_transcript_summary(
        {
            "range_start_utc": "2024-06-02T10:00:00Z",
            "range_end_utc": "2024-06-02T09:00:00Z",
        },
        authorized=True,
        base_dir=base_dir,
    )

    assert forbidden["summary_status"] == "forbidden"
    assert invalid["summary_status"] == "invalid_request"
    assert observed["queries"] == 0


def test_generate_whatsapp_transcript_summary_fails_when_window_too_large(tmp_path):
    base_dir = tmp_path / "records"

    for index in range(3):
        _append(
            base_dir,
            _record(
                record_id=f"record-{index}",
                effective_event_at_utc=f"2024-06-02T09:0{index}:00Z",
                record_sequence=index + 1,
                participant_role="external_party",
                message_classification="conversational_only",
                text=f"message {index}",
            ),
            datetime(2024, 6, 2, 9, index, 0, tzinfo=timezone.utc),
        )

    summary = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
        },
        authorized=True,
        base_dir=base_dir,
        max_records=2,
    )

    assert summary["summary_status"] == "summary_window_too_large"


def test_format_whatsapp_transcript_summary_renders_founder_visible_output():
    rendered = format_whatsapp_transcript_summary({
        "summary_status": "ready",
        "scope": {
            "conversation_key": None,
            "destination_key": "whatsapp:dm:15551230000",
            "group_chat_id": None,
            "dm_counterparty_id": None,
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
        },
        "coverage_start_utc": "2024-06-02T09:01:00Z",
        "coverage_end_utc": "2024-06-02T09:03:00Z",
        "covered_record_count": 2,
        "participants": [
            {
                "participant_role": "external_party",
                "participant_id": "15551230000",
                "participant_name": "Vendor",
            },
            {
                "participant_role": "agent",
                "participant_id": "agent",
                "participant_name": "Hermes",
            },
        ],
        "operator_context": ["Operator: summarize the quote thread"],
        "conversation_recap": [
            "Vendor: quoted 200 units.",
            "Hermes: requested final price.",
        ],
        "open_items": [
            "Unresolved question from Vendor: Do you still need the revised quote?"
        ],
        "uncertainties": ["A media attachment was present without visible contents."],
        "narrative_text": "WhatsApp transcript summary for whatsapp:dm:15551230000.",
    })

    assert "WhatsApp transcript summary" in rendered
    assert "Thread: whatsapp:dm:15551230000" in rendered
    assert "Operator context:" in rendered
    assert "Conversation recap:" in rendered
    assert "Open items:" in rendered
    assert "Uncertainties:" in rendered


def test_generate_whatsapp_transcript_summary_preserves_query_ordering(tmp_path):
    base_dir = tmp_path / "records"

    _append(
        base_dir,
        _record(
            record_id="later-sequence",
            effective_event_at_utc="2024-06-02T09:01:00Z",
            record_sequence=20,
            participant_role="external_party",
            message_classification="conversational_only",
            text="Second by sequence.",
            sender_name="Vendor",
        ),
        datetime(2024, 6, 2, 9, 1, 0, tzinfo=timezone.utc),
    )
    _append(
        base_dir,
        _record(
            record_id="earlier-sequence",
            effective_event_at_utc="2024-06-02T09:01:00Z",
            record_sequence=10,
            participant_role="agent",
            message_classification="conversational_only",
            text="First by sequence.",
            sender_name="Hermes",
        ),
        datetime(2024, 6, 2, 9, 1, 0, tzinfo=timezone.utc),
    )

    summary = generate_whatsapp_transcript_summary(
        {
            "destination_key": "whatsapp:dm:15551230000",
            "range_start_utc": "2024-06-02T09:00:00Z",
            "range_end_utc": "2024-06-02T10:00:00Z",
        },
        authorized=True,
        base_dir=base_dir,
    )

    assert summary["conversation_recap"] == [
        "Hermes: First by sequence.",
        "Vendor: Second by sequence.",
    ]
