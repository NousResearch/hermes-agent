"""Runtime-issued evidence receipts for delegated child tool results.

Receipts are minted only inside ``delegated_child_context`` and are committed to
the child agent's private in-memory ledger only after the canonical tool-result
row was durably flushed. Parent aggregation reads this ledger directly; it never
trusts receipt-shaped strings reconstructed from child prose or transcript.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from agent.delegation_context import is_delegated_child_context

_RECEIPT_RE = re.compile(r"\bdr_[0-9a-f]{24}\b")
_TARGET_KEYS = frozenset({
    "cwd", "destination_path", "directory", "dst", "endpoint", "file_path",
    "new_path", "old_path", "path", "source_path", "src", "target_path", "url", "urls",
})
_URL_KEYS = frozenset({"endpoint", "url", "urls"})


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def _safe_target(key: str, value: Any) -> Any:
    if isinstance(value, list):
        kept = [_safe_target(key, item) for item in value[:16]]
        return [item for item in kept if item is not None]
    if not isinstance(value, str) or not value:
        return None
    bounded = value[:1024]
    if key not in _URL_KEYS:
        return bounded
    try:
        parsed = urlsplit(bounded)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return None
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None


def _input_summary(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        return {"argument_keys": [], "targets": {}}
    targets: dict[str, Any] = {}
    for raw_key, value in arguments.items():
        key = str(raw_key).lower()
        if key not in _TARGET_KEYS:
            continue
        clean = _safe_target(key, value)
        if clean not in (None, [], {}):
            targets[key] = clean
    return {
        "argument_keys": sorted(str(key)[:128] for key in arguments)[:64],
        "targets": targets,
    }


def prepare_runtime_receipt(
    agent: Any, *, tool_name: str, tool_call_id: str, arguments: Any,
    result: Any, status: str, effect_disposition: Any,
) -> dict[str, Any] | None:
    """Mint one pending receipt inside a delegated child; no ledger mutation yet."""
    if not is_delegated_child_context():
        return None
    session_id = str(getattr(agent, "session_id", "") or "")
    subagent_id = str(getattr(agent, "_subagent_id", "") or "")
    if not session_id or not tool_call_id:
        return None
    normalized_status = status if status in {"ok", "error", "blocked", "timeout", "cancelled"} else "unknown"
    return {
        "receipt_id": f"dr_{uuid.uuid4().hex[:24]}",
        "child_session_id": session_id,
        "child_subagent_id": subagent_id or None,
        "tool_call_id": str(tool_call_id),
        "tool_name": str(tool_name or "tool"),
        "status": normalized_status,
        "effect_disposition": str(effect_disposition or "unknown"),
        "input_summary": _input_summary(arguments),
        "output_sha256": hashlib.sha256(_stable_text(result).encode("utf-8", errors="replace")).hexdigest(),
    }


def append_receipt_marker(content: Any, receipt_id: str) -> Any:
    """Append the child-visible citation marker without changing non-text blocks."""
    marker = f"[Runtime receipt: {receipt_id}]"
    if isinstance(content, str):
        return f"{content}\n\n{marker}" if content else marker
    if isinstance(content, list):
        return [*content, {"type": "text", "text": marker}]
    return f"{_stable_text(content)}\n\n{marker}"


def commit_runtime_receipt(agent: Any, receipt: dict[str, Any] | None) -> None:
    """Commit only after the matching canonical tool row has durably flushed."""
    if not isinstance(receipt, dict):
        return
    ledger = getattr(agent, "_delegate_runtime_receipts", None)
    if not isinstance(ledger, list):
        ledger = []
        setattr(agent, "_delegate_runtime_receipts", ledger)
    ledger.append(dict(receipt))


def parent_visible_receipts(child: Any, summary: Any) -> dict[str, Any]:
    """Validate summary citations against the Runtime-only child ledger."""
    raw = getattr(child, "_delegate_runtime_receipts", None)
    ledger = [dict(item) for item in raw if isinstance(item, dict) and _RECEIPT_RE.fullmatch(str(item.get("receipt_id") or ""))] if isinstance(raw, list) else []
    by_id = {item["receipt_id"]: item for item in ledger}
    cited = []
    if isinstance(summary, str):
        for receipt_id in _RECEIPT_RE.findall(summary):
            if receipt_id in by_id and receipt_id not in cited:
                cited.append(receipt_id)
    claimed = _RECEIPT_RE.findall(summary) if isinstance(summary, str) else []
    fabricated = sorted({receipt_id for receipt_id in claimed if receipt_id not in by_id})
    if cited:
        status = "verified_citations"
    elif ledger:
        status = "missing_citation"
    else:
        status = "no_runtime_evidence"
    return {
        "runtime_receipts": ledger,
        "cited_receipt_ids": cited,
        "fabricated_receipt_ids": fabricated,
        "provenance_status": status,
    }
