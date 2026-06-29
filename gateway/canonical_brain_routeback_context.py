"""Exact Canonical Brain route-back context for gateway turns.

This module is intentionally read-only and state-driven. It does not classify
messages, infer business meaning, or send anything. It only tells the model
when the current Discord thread is already an exact route-back target for an
existing Canonical Brain case, so the next answer can continue that case
instead of creating a duplicate.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - import-safe during tool discovery
    load_config = None  # type: ignore[assignment]

from gateway.config import Platform
from gateway.session import SessionContext

logger = logging.getLogger(__name__)

CANONICAL_BRAIN_ROOT = Path("/opt/adventico-ai-platform/canonical-brain")
CLOUD_SQL_HELPER = CANONICAL_BRAIN_ROOT / "bin" / "cloud_sql_synthetic_write_gate.py"
EVENT_TABLE = "canonical_event_log"
MAX_CONTEXT_CASES = 3


@dataclass(frozen=True)
class RouteBackCaseContext:
    case_id: str
    source_thread_id: str


def _load_helper() -> Any:
    if not CLOUD_SQL_HELPER.exists():
        raise RuntimeError("canonical brain Cloud SQL helper missing")
    spec = importlib.util.spec_from_file_location(
        "canonical_brain_cloud_sql_helper",
        CLOUD_SQL_HELPER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical brain Cloud SQL helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _routeback_context_enabled() -> bool:
    if load_config is None:
        return False
    try:
        cfg = load_config() or {}
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    cb = cfg.get("canonical_brain")
    if not isinstance(cb, dict):
        return False
    routeback = cb.get("route_back_context")
    if isinstance(routeback, dict) and "enabled" in routeback:
        return bool(routeback.get("enabled"))
    audit = cb.get("audit_bridge")
    return bool(cb.get("tools_enabled") or (isinstance(audit, dict) and audit.get("enabled")))


def _helper_available() -> bool:
    return CLOUD_SQL_HELPER.exists()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _row_get(row: Any, columns: list[str], name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    try:
        idx = columns.index(name)
    except ValueError:
        return None
    if isinstance(row, (list, tuple)) and idx < len(row):
        return row[idx]
    return None


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _same_thread(value: Any, current_thread_id: str) -> bool:
    return bool(value) and str(value) == current_thread_id


def _source_refs(source: Mapping[str, Any]) -> dict[str, Any]:
    refs = source.get("source_refs")
    return refs if isinstance(refs, dict) else {}


def _route_back_target(payload: Mapping[str, Any]) -> dict[str, Any]:
    target = _nested_get(payload, "route_back", "target_ref")
    if isinstance(target, dict):
        return target
    target = payload.get("target_ref")
    return target if isinstance(target, dict) else {}


def _receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = payload.get("receipt")
    if isinstance(receipt, dict):
        return receipt
    receipt = payload.get("delivery_receipt")
    if isinstance(receipt, dict):
        return receipt
    receipt = _nested_get(payload, "route_back", "receipt")
    return receipt if isinstance(receipt, dict) else {}


def _row_targets_current_thread(
    source: Mapping[str, Any],
    payload: Mapping[str, Any],
    current_thread_id: str,
) -> bool:
    target = _route_back_target(payload)
    receipt = _receipt(payload)
    return any(
        _same_thread(value, current_thread_id)
        for value in (
            target.get("id"),
            target.get("thread_id"),
            target.get("channel_id"),
            receipt.get("chat_id"),
            receipt.get("thread_id"),
        )
    )


def _row_source_thread(source: Mapping[str, Any]) -> str:
    refs = _source_refs(source)
    return str(refs.get("thread_id") or refs.get("chat_id") or "").strip()


def _query_linked_rows(current_thread_id: str) -> list[Any]:
    helper = _load_helper()
    password = helper.get_secret_value()
    try:
        sock = helper.connect(password)
        try:
            thread_sql = helper.sql_quote(current_thread_id)
            sql = f"""
SELECT event_id::text, event_type, case_id, occurred_at::text, source, payload
FROM {EVENT_TABLE}
WHERE case_id IN (
  SELECT DISTINCT case_id
  FROM {EVENT_TABLE}
  WHERE source->'source_refs'->>'thread_id' = {thread_sql}
     OR source->'source_refs'->>'chat_id' = {thread_sql}
     OR payload->'route_back'->'target_ref'->>'id' = {thread_sql}
     OR payload->'route_back'->'target_ref'->>'thread_id' = {thread_sql}
     OR payload->'route_back'->'target_ref'->>'channel_id' = {thread_sql}
     OR payload->'receipt'->>'chat_id' = {thread_sql}
     OR payload->'receipt'->>'thread_id' = {thread_sql}
     OR payload->'receipt'->>'channel_id' = {thread_sql}
     OR payload->'delivery_receipt'->>'chat_id' = {thread_sql}
     OR payload->'delivery_receipt'->>'thread_id' = {thread_sql}
     OR payload->'delivery_receipt'->>'channel_id' = {thread_sql}
     OR payload->'route_back'->'receipt'->>'chat_id' = {thread_sql}
     OR payload->'route_back'->'receipt'->>'thread_id' = {thread_sql}
     OR payload->'route_back'->'receipt'->>'channel_id' = {thread_sql}
)
ORDER BY occurred_at DESC
LIMIT 80;
"""
            result = helper.query(sock, sql)
            rows = result.get("rows", []) if isinstance(result, dict) else []
            return rows if isinstance(rows, list) else []
        finally:
            try:
                sock.close()
            except Exception:
                pass
    finally:
        password = ""


def lookup_routeback_cases_for_thread(current_thread_id: str) -> list[RouteBackCaseContext]:
    """Return exact cases where ``current_thread_id`` is a route-back target.

    The current thread must appear in a route-back target/receipt, and the same
    case must have a different source/requester thread. Source-only matches are
    deliberately ignored here; this context is for owner/resolver answer turns.
    """
    current_thread_id = str(current_thread_id or "").strip()
    if not current_thread_id:
        return []

    columns = ["event_id", "event_type", "case_id", "occurred_at", "source", "payload"]
    grouped: dict[str, dict[str, Any]] = {}
    for row in _query_linked_rows(current_thread_id):
        case_id = str(_row_get(row, columns, "case_id") or "").strip()
        if not case_id:
            continue
        source = _coerce_mapping(_row_get(row, columns, "source"))
        payload = _coerce_mapping(_row_get(row, columns, "payload"))
        entry = grouped.setdefault(case_id, {"is_target": False, "source_threads": []})
        if _row_targets_current_thread(source, payload, current_thread_id):
            entry["is_target"] = True
        source_thread = _row_source_thread(source)
        if source_thread and source_thread != current_thread_id:
            entry["source_threads"].append(source_thread)

    contexts: list[RouteBackCaseContext] = []
    for case_id, data in grouped.items():
        if not data.get("is_target"):
            continue
        source_threads = list(dict.fromkeys(data.get("source_threads") or []))
        if not source_threads:
            continue
        contexts.append(RouteBackCaseContext(case_id=case_id, source_thread_id=source_threads[0]))
        if len(contexts) >= MAX_CONTEXT_CASES:
            break
    return contexts


def build_routeback_context_prompt(contexts: Iterable[RouteBackCaseContext]) -> str:
    cases = list(contexts)
    if not cases:
        return ""
    lines = [
        "## Canonical Brain Route-Back Context",
        "",
        "The current Discord thread is an exact route-back target for existing Canonical Brain case(s):",
    ]
    for item in cases:
        lines.append(f"- `{item.case_id}` (source/requester thread: `{item.source_thread_id}`)")
    lines.extend(
        [
            "",
            "If this turn contains an owner/resolver answer, delivery result, or status update:",
            "- Continue the same `case_id`; do not create a new duplicate case.",
            "- Record the answer/status as durable case state before any requester closeout.",
            "- Notify the source/requester thread at most once, with only the actionable delta.",
            "- Record `route_back.sent` only after a real delivery receipt/message_id.",
            "- Muncho must not send route-backs by DM. Use only public approved Discord channels/threads; if no approved public target is available, record/report `route_back.blocked`.",
            "- If a resolver asks you to forward/notify the requester, either actually notify the source/requester thread and record `route_back.sent`, or record/report `route_back.blocked` with the blocker. A reply like 'noted', 'marked', or 'for forwarding' is not a terminal outcome.",
            "- A `route_back.required` or `route_back.intent.created` tool result is not completion. Keep working in the same turn until there is a `route_back.sent` receipt or `route_back.blocked` blocker.",
            "- Never answer the resolver with only 'noted/marked for forwarding' after a concrete forward request; that leaves the requester uninformed.",
            "- Do not use cron for immediate route-back delivery; use direct Discord delivery when available. Cron is only for future reminders/watchers, and never both create+run for the same immediate message.",
            "- Do not repeat the owner/resolver request after the owner/resolver has answered.",
            "- If durable route-back recording fails after a send, do not send duplicate public corrections; record/report the state blocker separately.",
            "- Include a concrete next-action artifact for the requester when useful: email subject/body, code snippet, checklist, decision options, or precise next steps. Do not only forward content.",
        ]
    )
    return "\n".join(lines)


def build_routeback_context_prompt_for_session(context: SessionContext) -> str:
    """Build a fail-soft prompt fragment for the current gateway session."""
    try:
        if context.source.platform != Platform.DISCORD:
            return ""
        current_thread_id = str(context.source.thread_id or context.source.chat_id or "").strip()
        if not current_thread_id:
            return ""
        if not _routeback_context_enabled() or not _helper_available():
            return ""
        cases = lookup_routeback_cases_for_thread(current_thread_id)
        return build_routeback_context_prompt(cases)
    except Exception as exc:
        logger.debug("Canonical Brain route-back context lookup failed: %s", exc)
        return ""
