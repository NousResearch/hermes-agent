"""Compression-safe dedup helpers for side-effecting tool calls.

The first guarded surface is ``send_message`` because duplicate outbound
messages are user-visible and harmful. We persist a compact action log across
context compaction and consult it before re-executing the same send.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

ACTION_LOG_BEGIN = "[HERMES_ACTION_LOG_BEGIN]"
ACTION_LOG_END = "[HERMES_ACTION_LOG_END]"


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _send_message_hash(message: Any) -> str:
    normalized = _normalize_space(message)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_send_message_record(
    function_args: Mapping[str, Any] | None,
    result_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    args = dict(function_args or {})
    action = _normalize_space(args.get("action") or "send").lower()
    target = _normalize_space(args.get("target"))
    message = args.get("message")
    if action != "send" or not target or not _normalize_space(message):
        return None

    message_sha = _send_message_hash(message)
    record: dict[str, Any] = {
        "tool": "send_message",
        "target": target,
        "message_sha256": message_sha,
        "fingerprint": f"send_message:{target}:{message_sha}",
    }
    if isinstance(result_payload, Mapping):
        for src, dst in (
            ("platform", "platform"),
            ("chat_id", "chat_id"),
            ("thread_id", "thread_id"),
            ("message_id", "message_id"),
        ):
            value = result_payload.get(src)
            if value not in (None, ""):
                record[dst] = str(value)
    return record


def _coerce_action_log_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = None
    if not isinstance(records, list):
        return []
    out: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict) and record.get("fingerprint"):
            out.append(record)
    return out


def extract_action_log_records(text: Any) -> list[dict[str, Any]]:
    rendered = _message_content_text(text)
    records: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = rendered.find(ACTION_LOG_BEGIN, cursor)
        if start == -1:
            break
        payload_start = start + len(ACTION_LOG_BEGIN)
        end = rendered.find(ACTION_LOG_END, payload_start)
        if end == -1:
            break
        payload_text = rendered[payload_start:end].strip()
        try:
            payload = json.loads(payload_text)
        except (TypeError, ValueError):
            payload = None
        records.extend(_coerce_action_log_payload(payload))
        cursor = end + len(ACTION_LOG_END)
    return records


def strip_action_log_blocks(text: Any) -> str:
    rendered = _message_content_text(text)
    pattern = re.compile(
        rf"\n?{re.escape(ACTION_LOG_BEGIN)}\n.*?\n{re.escape(ACTION_LOG_END)}",
        re.DOTALL,
    )
    return pattern.sub("", rendered).rstrip()


def merge_action_log_records(*record_lists: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for records in record_lists:
        for record in records:
            if not isinstance(record, dict):
                continue
            fingerprint = str(record.get("fingerprint") or "").strip()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(record)
    return merged


def render_action_log_block(records: Iterable[dict[str, Any]]) -> str:
    merged = merge_action_log_records(records)
    if not merged:
        return ""
    payload = {"version": 1, "records": merged}
    return (
        f"\n\n{ACTION_LOG_BEGIN}\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
        f"{ACTION_LOG_END}"
    )


def collect_successful_send_message_records(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    call_id_to_args: dict[str, Mapping[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if fn.get("name") != "send_message":
                continue
            call_id = tc.get("id")
            if not call_id:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            if isinstance(args, dict):
                call_id_to_args[call_id] = args

    records: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if not call_id or call_id not in call_id_to_args:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except (TypeError, ValueError):
                payload = None
        elif isinstance(content, dict):
            payload = content
        else:
            payload = None
        if not isinstance(payload, dict):
            continue
        if not payload.get("success") or payload.get("skipped"):
            continue
        record = build_send_message_record(call_id_to_args[call_id], payload)
        if record is not None:
            records.append(record)
    return merge_action_log_records(records)


def collect_summary_action_log_records(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for msg in messages:
        records.extend(extract_action_log_records(msg.get("content")))
    return merge_action_log_records(records)


def find_prior_duplicate_send_message(
    messages: Iterable[Mapping[str, Any]],
    function_args: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    current = build_send_message_record(function_args)
    if current is None:
        return None
    fingerprint = current["fingerprint"]
    for record in collect_summary_action_log_records(messages):
        if record.get("fingerprint") == fingerprint:
            return record
    return None


def build_duplicate_send_message_skip_result(
    function_args: Mapping[str, Any] | None,
    prior_record: Mapping[str, Any] | None,
) -> str | None:
    if prior_record is None:
        return None
    record = build_send_message_record(function_args)
    if record is None:
        return None
    target = str(prior_record.get("target") or record["target"])
    return json.dumps(
        {
            "success": True,
            "skipped": True,
            "reason": "compression_duplicate_send_message",
            "target": target,
            "fingerprint": record["fingerprint"],
            "note": (
                "Skipped duplicate send_message. An identical target/message pair "
                "already succeeded earlier in this session and was preserved in "
                "the context-compaction action log."
            ),
        },
        ensure_ascii=False,
    )
