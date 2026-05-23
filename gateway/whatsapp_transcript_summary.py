from __future__ import annotations

import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.whatsapp_message_store import query_whatsapp_records

TRANSCRIPT_SUMMARY_SCOPE_FIELDS = (
    "conversation_key",
    "destination_key",
    "group_chat_id",
    "dm_counterparty_id",
    "range_start_utc",
    "range_end_utc",
)
TRANSCRIPT_SUMMARY_OPTIONAL_FIELDS = ("include_operator_context",)
TRANSCRIPT_SUMMARY_ALLOWED_FIELDS = set(
    TRANSCRIPT_SUMMARY_SCOPE_FIELDS + TRANSCRIPT_SUMMARY_OPTIONAL_FIELDS
)
DEFAULT_INCLUDE_OPERATOR_CONTEXT = True
DEFAULT_MAX_SUMMARY_RECORDS = 200
_READY_SUMMARY_STATUSES = {
    "ready",
    "no_conversation_records",
    "summary_window_too_large",
    "invalid_request",
    "forbidden",
}


def _default_scope() -> dict[str, str | None]:
    return {
        "conversation_key": None,
        "destination_key": None,
        "group_chat_id": None,
        "dm_counterparty_id": None,
        "range_start_utc": None,
        "range_end_utc": None,
    }


def _bool_from_request(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("include_operator_context must be true or false")


def _parse_explicit_utc_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("timestamps must be explicit UTC datetimes")
        return value.astimezone(timezone.utc)

    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamps are required")

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamps must be ISO-8601 UTC values") from exc

    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamps must be explicit UTC values")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_transcript_summary_request(
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_request = request or {}
    scope = _default_scope()
    for field in (
        "conversation_key",
        "destination_key",
        "group_chat_id",
        "dm_counterparty_id",
    ):
        value = raw_request.get(field)
        if value is None:
            continue
        text = str(value).strip()
        scope[field] = text or None

    include_operator_context = _bool_from_request(
        raw_request.get("include_operator_context"),
        default=DEFAULT_INCLUDE_OPERATOR_CONTEXT,
    )

    range_start_utc = _parse_explicit_utc_timestamp(raw_request.get("range_start_utc"))
    range_end_utc = _parse_explicit_utc_timestamp(raw_request.get("range_end_utc"))

    scope["range_start_utc"] = _iso_utc(range_start_utc)
    scope["range_end_utc"] = _iso_utc(range_end_utc)

    if not any(
        scope[field]
        for field in (
            "conversation_key",
            "destination_key",
            "group_chat_id",
            "dm_counterparty_id",
        )
    ):
        raise ValueError(
            "request requires at least one exact destination scope: "
            "conversation_key, destination_key, group_chat_id, or dm_counterparty_id"
        )
    if range_start_utc >= range_end_utc:
        raise ValueError("range_start_utc must be earlier than range_end_utc")

    return {
        **scope,
        "range_start_dt": range_start_utc,
        "range_end_dt": range_end_utc,
        "include_operator_context": include_operator_context,
    }


def parse_transcript_summary_command_args(raw_args: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for token in shlex.split(raw_args or ""):
        if "=" not in token:
            raise ValueError("command arguments must use field=value syntax")
        field, value = token.split("=", 1)
        field = field.strip()
        if field not in TRANSCRIPT_SUMMARY_ALLOWED_FIELDS:
            raise ValueError(f"unsupported field: {field}")
        parsed[field] = value.strip()
    return parsed


def _speaker_label(row: dict[str, Any]) -> str:
    participant_role = row.get("participant_role")
    sender_name = str(row.get("sender_name") or "").strip()
    sender_id = str(row.get("sender_id") or "").strip()

    if participant_role == "owner_operator":
        return sender_name or "Operator"
    if participant_role == "agent":
        return "Hermes"
    if sender_name:
        return sender_name
    if sender_id:
        return f"External party ({sender_id})"
    return "External party"


def _participant_entry(row: dict[str, Any]) -> tuple[str, str | None, str | None]:
    participant_role = str(row.get("participant_role") or "") or None
    participant_id = str(row.get("sender_id") or "").strip() or None
    participant_name = str(row.get("sender_name") or "").strip() or None
    if participant_role == "agent":
        participant_id = participant_id or "agent"
        participant_name = participant_name or "Hermes"
    elif participant_role == "owner_operator" and not participant_name:
        participant_name = "Operator"
    return participant_role or "external_party", participant_id, participant_name


def _trim_text(value: str, *, limit: int = 240) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _row_fact(row: dict[str, Any]) -> dict[str, Any] | None:
    speaker = _speaker_label(row)
    text = str(row.get("text") or "").strip()
    media_types = [
        str(item).strip()
        for item in (row.get("media_types") or [])
        if str(item).strip()
    ]

    if text:
        return {
            "speaker": speaker,
            "kind": "text",
            "text": _trim_text(text),
        }
    if media_types:
        joined_media_types = ", ".join(media_types)
        return {
            "speaker": speaker,
            "kind": "media",
            "text": f"sent non-text media ({joined_media_types})",
        }
    return None


def _merge_fact_rows(rows: list[dict[str, Any]]) -> list[str]:
    recap: list[str] = []
    last_speaker: str | None = None
    last_kind: str | None = None

    for row in rows:
        fact = _row_fact(row)
        if fact is None:
            continue
        speaker = fact["speaker"]
        kind = fact["kind"]
        text = fact["text"]
        if recap and speaker == last_speaker and kind == last_kind == "text":
            recap[-1] = recap[-1] + f" {text}"
            continue
        if kind == "text":
            recap.append(f"{speaker}: {text}")
        else:
            recap.append(f"{speaker} {text}")
        last_speaker = speaker
        last_kind = kind
    return recap


def _collect_open_items(rows: list[dict[str, Any]]) -> list[str]:
    open_items: list[str] = []
    pending_question: dict[str, Any] | None = None

    for row in rows:
        text = str(row.get("text") or "").strip()
        if text.endswith("?"):
            pending_question = {
                "participant_role": str(row.get("participant_role") or ""),
                "speaker": _speaker_label(row),
                "text": _trim_text(text),
            }
            continue
        if (
            pending_question is not None
            and str(row.get("participant_role") or "") == pending_question["participant_role"]
            and text
        ):
            pending_question = None

    if pending_question is not None:
        open_items.append(
            f"Unresolved question from {pending_question['speaker']}: {pending_question['text']}"
        )
    return open_items


def _collect_uncertainties(rows: list[dict[str, Any]]) -> list[str]:
    uncertainties: list[str] = []
    saw_media_only = False
    saw_blank_turn = False

    for row in rows:
        text = str(row.get("text") or "").strip()
        media_types = row.get("media_types") or []
        if not text and media_types:
            saw_media_only = True
        if not text and not media_types:
            saw_blank_turn = True

    if saw_media_only:
        uncertainties.append(
            "At least one non-text media turn was preserved without attachment "
            "contents, so the summary only reports that media was exchanged."
        )
    if saw_blank_turn:
        uncertainties.append(
            "At least one preserved turn had no stored text or media metadata, "
            "so it could not be described in detail."
        )
    return uncertainties


def _scope_label(scope: dict[str, Any]) -> str:
    if scope.get("conversation_key"):
        return str(scope["conversation_key"])
    if scope.get("destination_key"):
        return str(scope["destination_key"])
    if scope.get("group_chat_id"):
        return f"group {scope['group_chat_id']}"
    if scope.get("dm_counterparty_id"):
        return f"DM {scope['dm_counterparty_id']}"
    return "requested WhatsApp conversation"


def _summary_output(
    *,
    summary_status: str,
    scope: dict[str, Any],
    coverage_start_utc: str | None = None,
    coverage_end_utc: str | None = None,
    covered_record_count: int = 0,
    participants: list[dict[str, Any]] | None = None,
    operator_context: list[str] | None = None,
    conversation_recap: list[str] | None = None,
    open_items: list[str] | None = None,
    uncertainties: list[str] | None = None,
    narrative_text: str = "",
) -> dict[str, Any]:
    if summary_status not in _READY_SUMMARY_STATUSES:
        raise ValueError(f"unsupported summary status: {summary_status}")
    return {
        "summary_status": summary_status,
        "scope": {
            key: scope.get(key)
            for key in TRANSCRIPT_SUMMARY_SCOPE_FIELDS
        },
        "coverage_start_utc": coverage_start_utc,
        "coverage_end_utc": coverage_end_utc,
        "covered_record_count": covered_record_count,
        "participants": participants or [],
        "operator_context": operator_context or [],
        "conversation_recap": conversation_recap or [],
        "open_items": open_items or [],
        "uncertainties": uncertainties or [],
        "narrative_text": narrative_text,
    }


def generate_whatsapp_transcript_summary(
    request: dict[str, Any] | None,
    *,
    authorized: bool,
    base_dir: Path | None = None,
    max_records: int = DEFAULT_MAX_SUMMARY_RECORDS,
) -> dict[str, Any]:
    if not authorized:
        return _summary_output(
            summary_status="forbidden",
            scope=request or _default_scope(),
        )

    try:
        normalized_request = normalize_transcript_summary_request(request)
    except ValueError:
        return _summary_output(
            summary_status="invalid_request",
            scope=request or _default_scope(),
        )

    records = query_whatsapp_records(
        normalized_request["range_start_dt"],
        normalized_request["range_end_dt"],
        base_dir=base_dir,
        conversation_key=normalized_request["conversation_key"],
        destination_key=normalized_request["destination_key"],
        group_chat_id=normalized_request["group_chat_id"],
        dm_counterparty_id=normalized_request["dm_counterparty_id"],
    )

    if len(records) > max_records:
        return _summary_output(
            summary_status="summary_window_too_large",
            scope=normalized_request,
        )

    operator_rows = [
        row
        for row in records
        if row.get("participant_role") == "owner_operator"
        and row.get("message_classification") == "command_capable"
    ]
    conversation_rows = [
        row
        for row in records
        if row.get("participant_role") in {"external_party", "agent"}
        and row.get("message_classification") == "conversational_only"
    ]

    if not conversation_rows:
        return _summary_output(
            summary_status="no_conversation_records",
            scope=normalized_request,
        )

    included_rows = list(conversation_rows)
    operator_context = []
    if normalized_request["include_operator_context"]:
        operator_context = _merge_fact_rows(operator_rows)
        included_rows = list(operator_rows) + included_rows

    participants: list[dict[str, Any]] = []
    seen_participants: set[tuple[str, str | None, str | None]] = set()
    for row in included_rows:
        participant = _participant_entry(row)
        if participant in seen_participants:
            continue
        seen_participants.add(participant)
        participants.append(
            {
                "participant_role": participant[0],
                "participant_id": participant[1],
                "participant_name": participant[2],
            }
        )

    conversation_recap = _merge_fact_rows(conversation_rows)
    open_items = _collect_open_items(conversation_rows)
    uncertainties = _collect_uncertainties(conversation_rows)

    coverage_rows = included_rows if included_rows else conversation_rows
    coverage_start = coverage_rows[0].get("effective_event_at_utc")
    coverage_end = coverage_rows[-1].get("effective_event_at_utc")
    covered_record_count = len(coverage_rows)

    participant_labels = [
        participant.get("participant_name")
        or participant.get("participant_id")
        or participant.get("participant_role")
        for participant in participants
    ]
    recap_preview = "; ".join(conversation_recap[:3])
    if len(conversation_recap) > 3:
        recap_preview += f"; +{len(conversation_recap) - 3} more turn(s)"

    narrative_parts = [
        (
            "WhatsApp transcript summary for "
            f"{_scope_label(normalized_request)} covering {coverage_start} "
            f"to {coverage_end}."
        ),
        f"Participants: {', '.join(participant_labels)}.",
        f"Key developments: {recap_preview}.",
    ]
    if open_items:
        narrative_parts.append(f"Open items: {'; '.join(open_items)}.")
    if uncertainties:
        narrative_parts.append(f"Uncertainties: {'; '.join(uncertainties)}.")

    return _summary_output(
        summary_status="ready",
        scope=normalized_request,
        coverage_start_utc=coverage_start,
        coverage_end_utc=coverage_end,
        covered_record_count=covered_record_count,
        participants=participants,
        operator_context=operator_context,
        conversation_recap=conversation_recap,
        open_items=open_items,
        uncertainties=uncertainties,
        narrative_text=" ".join(narrative_parts),
    )


def format_whatsapp_transcript_summary(summary: dict[str, Any]) -> str:
    lines = [
        "WhatsApp transcript summary",
        f"Status: {summary.get('summary_status', 'invalid_request')}",
    ]

    summary_status = summary.get("summary_status")
    if summary_status == "ready":
        scope = summary.get("scope") or {}
        lines.extend(
            [
                f"Thread: {_scope_label(scope)}",
                (
                    "Coverage: "
                    f"{summary.get('coverage_start_utc')} → {summary.get('coverage_end_utc')} "
                    f"({summary.get('covered_record_count', 0)} records)"
                ),
            ]
        )
        participants = summary.get("participants") or []
        if participants:
            rendered_participants = []
            for participant in participants:
                label = (
                    participant.get("participant_name")
                    or participant.get("participant_id")
                    or participant.get("participant_role")
                )
                rendered_participants.append(
                    f"{label} [{participant.get('participant_role')}]"
                )
            lines.append("Participants: " + "; ".join(rendered_participants))

        def _append_section(title: str, entries: list[str]) -> None:
            if not entries:
                return
            lines.append("")
            lines.append(title)
            for entry in entries:
                lines.append(f"- {entry}")

        _append_section("Operator context:", list(summary.get("operator_context") or []))
        _append_section("Conversation recap:", list(summary.get("conversation_recap") or []))
        _append_section("Open items:", list(summary.get("open_items") or []))
        _append_section("Uncertainties:", list(summary.get("uncertainties") or []))
        lines.append("")
        lines.append(summary.get("narrative_text") or "")
        return "\n".join(line for line in lines if line is not None)

    if summary_status == "forbidden":
        lines.append("This request is not authorized to generate a WhatsApp transcript summary.")
    elif summary_status == "summary_window_too_large":
        lines.append("The requested window is too large to summarize safely without dropping evidence.")
    elif summary_status == "no_conversation_records":
        lines.append("No conversational WhatsApp records matched the requested scope and timeframe.")
    else:
        lines.append(
            "The request was invalid. Provide an exact scope plus explicit UTC "
            "range_start_utc and range_end_utc values."
        )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_SUMMARY_RECORDS",
    "DEFAULT_INCLUDE_OPERATOR_CONTEXT",
    "TRANSCRIPT_SUMMARY_ALLOWED_FIELDS",
    "TRANSCRIPT_SUMMARY_OPTIONAL_FIELDS",
    "TRANSCRIPT_SUMMARY_SCOPE_FIELDS",
    "format_whatsapp_transcript_summary",
    "generate_whatsapp_transcript_summary",
    "normalize_transcript_summary_request",
    "parse_transcript_summary_command_args",
]
