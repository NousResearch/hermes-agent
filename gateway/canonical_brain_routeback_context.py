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
import re
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
MAX_CONTEXT_QUERY_CASES = MAX_CONTEXT_CASES + 1
CANONICAL_BRAIN_IO_TIMEOUT_SECONDS = 12
ROUTE_BACK_LIFECYCLE_TYPES = (
    "route_back.required",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
)
_CONTEXT_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$")


@dataclass(frozen=True)
class RouteBackCaseContext:
    case_id: str
    source_thread_id: str


@dataclass(frozen=True)
class RouteBackContextLookup:
    cases: tuple[RouteBackCaseContext, ...]
    truncated: bool = False


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


def _same_thread(value: Any, current_thread_id: str) -> bool:
    return bool(value) and str(value) == current_thread_id


def _safe_context_identifier(value: Any) -> str:
    """Return a bounded inert identifier, or fail closed for legacy bad rows."""
    candidate = str(value or "").strip()
    return candidate if _CONTEXT_IDENTIFIER_RE.fullmatch(candidate) else ""


def _observed_session(source: Mapping[str, Any]) -> dict[str, Any]:
    observed = source.get("observed_session")
    return observed if isinstance(observed, dict) else {}


def _row_targets_current_thread(
    payload: Mapping[str, Any],
    current_thread_id: str,
) -> bool:
    route_back = payload.get("route_back")
    route_back = route_back if isinstance(route_back, Mapping) else {}
    surfaces = (
        payload.get("target_ref"),
        route_back.get("target_ref"),
        payload.get("receipt"),
        payload.get("delivery_receipt"),
        route_back.get("receipt"),
    )
    return any(
        isinstance(surface, Mapping)
        and any(
            _same_thread(surface.get(key), current_thread_id)
            for key in ("id", "chat_id", "thread_id", "channel_id")
        )
        for surface in surfaces
    )


def _row_source_thread(source: Mapping[str, Any]) -> str:
    observed = _observed_session(source)
    if str(observed.get("platform") or "").strip().casefold() != "discord":
        return ""
    return str(observed.get("thread_id") or observed.get("chat_id") or "").strip()


def _query_linked_rows(current_thread_id: str) -> list[Any]:
    helper = _load_helper()
    password = helper.get_secret_value()
    try:
        sock = helper.connect(password)
        setter = getattr(sock, "settimeout", None)
        if callable(setter):
            setter(CANONICAL_BRAIN_IO_TIMEOUT_SECONDS)
        try:
            thread_sql = helper.sql_quote(current_thread_id)
            lifecycle_sql = ", ".join(
                helper.sql_quote(event_type) for event_type in ROUTE_BACK_LIFECYCLE_TYPES
            )
            sql = f"""
WITH secure_linked_cases AS (
  SELECT DISTINCT e.case_id
  FROM {EVENT_TABLE} e
  WHERE (
    e.source->'observed_session'->>'platform' = 'discord'
    AND (
      e.source->'observed_session'->>'thread_id' = {thread_sql}
      OR e.source->'observed_session'->>'chat_id' = {thread_sql}
    )
  ) OR (
    e.event_type = 'route_back.sent'
    AND e.evidence @> '[{{"verified":true,"attestation":"deterministic_runtime_receipt"}}]'::jsonb
    AND (
      e.payload->'target_ref'->>'id' = {thread_sql}
      OR e.payload->'target_ref'->>'thread_id' = {thread_sql}
      OR e.payload->'target_ref'->>'channel_id' = {thread_sql}
      OR e.payload->'route_back'->'target_ref'->>'id' = {thread_sql}
      OR e.payload->'route_back'->'target_ref'->>'thread_id' = {thread_sql}
      OR e.payload->'route_back'->'target_ref'->>'channel_id' = {thread_sql}
      OR e.payload->'receipt'->>'chat_id' = {thread_sql}
      OR e.payload->'receipt'->>'thread_id' = {thread_sql}
      OR e.payload->'receipt'->>'channel_id' = {thread_sql}
      OR e.payload->'delivery_receipt'->>'chat_id' = {thread_sql}
      OR e.payload->'delivery_receipt'->>'thread_id' = {thread_sql}
      OR e.payload->'delivery_receipt'->>'channel_id' = {thread_sql}
      OR e.payload->'route_back'->'receipt'->>'chat_id' = {thread_sql}
      OR e.payload->'route_back'->'receipt'->>'thread_id' = {thread_sql}
      OR e.payload->'route_back'->'receipt'->>'channel_id' = {thread_sql}
    )
  )
), fair_case_rows AS (
  SELECT
    latest_route.event_id AS event_id,
    latest_route.event_type,
    linked.case_id,
    latest_route.occurred_at AS occurred_at,
    latest_route.source,
    latest_route.payload
  FROM secure_linked_cases linked
  JOIN LATERAL (
    SELECT e.event_id, e.event_type, e.occurred_at, e.source, e.payload
    FROM {EVENT_TABLE} e
    WHERE e.case_id = linked.case_id
      AND e.event_type IN ({lifecycle_sql})
    ORDER BY e.occurred_at DESC, e.event_id DESC
    LIMIT 1
  ) latest_route ON TRUE
  WHERE latest_route.source->'observed_session'->>'platform' = 'discord'
    AND COALESCE(
      NULLIF(latest_route.source->'observed_session'->>'thread_id', ''),
      NULLIF(latest_route.source->'observed_session'->>'chat_id', '')
    ) <> {thread_sql}
    AND (
    latest_route.payload->'target_ref'->>'id' = {thread_sql}
    OR latest_route.payload->'target_ref'->>'thread_id' = {thread_sql}
    OR latest_route.payload->'target_ref'->>'channel_id' = {thread_sql}
    OR latest_route.payload->'route_back'->'target_ref'->>'id' = {thread_sql}
    OR latest_route.payload->'route_back'->'target_ref'->>'thread_id' = {thread_sql}
    OR latest_route.payload->'route_back'->'target_ref'->>'channel_id' = {thread_sql}
    OR latest_route.payload->'receipt'->>'chat_id' = {thread_sql}
    OR latest_route.payload->'receipt'->>'thread_id' = {thread_sql}
    OR latest_route.payload->'receipt'->>'channel_id' = {thread_sql}
    OR latest_route.payload->'delivery_receipt'->>'chat_id' = {thread_sql}
    OR latest_route.payload->'delivery_receipt'->>'thread_id' = {thread_sql}
    OR latest_route.payload->'delivery_receipt'->>'channel_id' = {thread_sql}
    OR latest_route.payload->'route_back'->'receipt'->>'chat_id' = {thread_sql}
    OR latest_route.payload->'route_back'->'receipt'->>'thread_id' = {thread_sql}
    OR latest_route.payload->'route_back'->'receipt'->>'channel_id' = {thread_sql}
  )
)
SELECT event_id::text, event_type, case_id, occurred_at::text, source, payload
FROM fair_case_rows
ORDER BY occurred_at DESC, event_id DESC, case_id
LIMIT {MAX_CONTEXT_QUERY_CASES};
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


def lookup_routeback_context_for_thread(
    current_thread_id: str,
) -> RouteBackContextLookup:
    """Return bounded exact cases plus an explicit incomplete-context signal.

    The current thread must appear in a route-back target/receipt, and the same
    exact route lifecycle row must have a different observed requester thread.
    Source-only matches and arbitrary older case participants are deliberately
    ignored here; this context is for owner/resolver answer turns.
    """
    current_thread_id = str(current_thread_id or "").strip()
    if not current_thread_id:
        return RouteBackContextLookup(cases=())

    columns = ["event_id", "event_type", "case_id", "occurred_at", "source", "payload"]
    contexts: list[RouteBackCaseContext] = []
    seen_case_ids: set[str] = set()
    rows = _query_linked_rows(current_thread_id)
    truncated = len(rows) > MAX_CONTEXT_CASES
    for row in rows:
        case_id = _safe_context_identifier(_row_get(row, columns, "case_id"))
        if not case_id or case_id in seen_case_ids:
            continue
        source = _coerce_mapping(_row_get(row, columns, "source"))
        payload = _coerce_mapping(_row_get(row, columns, "payload"))
        event_type = str(_row_get(row, columns, "event_type") or "")
        if event_type not in ROUTE_BACK_LIFECYCLE_TYPES:
            continue
        if not _row_targets_current_thread(payload, current_thread_id):
            continue
        source_thread = _safe_context_identifier(_row_source_thread(source))
        if not source_thread or source_thread == current_thread_id:
            continue
        seen_case_ids.add(case_id)
        contexts.append(RouteBackCaseContext(case_id=case_id, source_thread_id=source_thread))
        if len(contexts) >= MAX_CONTEXT_CASES:
            break
    return RouteBackContextLookup(cases=tuple(contexts), truncated=truncated)


def lookup_routeback_cases_for_thread(current_thread_id: str) -> list[RouteBackCaseContext]:
    """Backward-compatible case-only view of the bounded context lookup."""
    return list(lookup_routeback_context_for_thread(current_thread_id).cases)


def build_routeback_context_prompt(
    contexts: Iterable[RouteBackCaseContext],
    *,
    truncated: bool = False,
) -> str:
    cases = list(contexts)
    if not cases and not truncated:
        return ""
    lines = [
        "## Canonical Brain Route-Back Context",
        "",
        "The current Discord thread is an exact route-back target for existing Canonical Brain case(s):",
    ]
    for item in cases:
        lines.append(f"- `{item.case_id}` (source/requester thread: `{item.source_thread_id}`)")
    if truncated:
        lines.extend(
            [
                "",
                "INCOMPLETE CONTEXT: Additional exact linked cases exist beyond this bounded snapshot.",
                "Do not assume the listed cases are exhaustive or select a case by keywords. "
                "Use `canonical_brain_query` for the exact current Discord thread before "
                "updating or routing a case; if the complete state cannot be read, report "
                "that concrete blocker.",
            ]
        )
    lines.extend(
        [
            "",
            "If this turn contains an owner/resolver answer, delivery result, or status update:",
            "- Continue the same `case_id`; do not create a new duplicate case.",
            "- Record the answer/status as durable case state before any requester closeout.",
            "- Notify the source/requester thread at most once, with only the actionable delta.",
            "- Record `route_back.sent` only after a real delivery receipt/message_id.",
            "- Muncho must not send route-backs by DM. Use only public approved Discord channels/threads; if no approved public target is available, record/report `route_back.blocked`.",
            "- When the target is exact and public, prefer `route_back_execute`: it sends directly, then records `route_back.sent` with the real receipt, or records `route_back.blocked` if delivery cannot be completed.",
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
        lookup = lookup_routeback_context_for_thread(current_thread_id)
        return build_routeback_context_prompt(
            lookup.cases,
            truncated=lookup.truncated,
        )
    except Exception as exc:
        logger.debug("Canonical Brain route-back context lookup failed: %s", exc)
        return ""


def attach_routeback_context_to_user_turn(
    message_text: Any,
    persist_user_message: Any,
    context_prompt: str,
) -> tuple[Any, Any]:
    """Attach changing Brain state as a replayable current-turn snapshot.

    No synthetic role is inserted and the frozen system prompt is untouched.
    The exact decorated user content is persisted because replacing it with a
    raw-only variant on the next turn would mutate the cached conversation
    prefix.  Newer snapshots arrive as newer user turns; past API content stays
    byte-stable.
    """
    if not context_prompt or not isinstance(message_text, str):
        return message_text, persist_user_message
    decorated = context_prompt + "\n\n## Current user message\n" + message_text
    return decorated, decorated


__all__ = [
    "RouteBackCaseContext",
    "RouteBackContextLookup",
    "lookup_routeback_context_for_thread",
    "lookup_routeback_cases_for_thread",
    "build_routeback_context_prompt",
    "build_routeback_context_prompt_for_session",
    "attach_routeback_context_to_user_turn",
]
