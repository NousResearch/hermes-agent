"""Bounded recall over the LCM store.

The recall contract, stated once:

* **Current session by default, and only by default.**  Every entry point
  takes the caller's session and searches *that* session.  There is no
  "search everything" flag here: broad cross-session history search is Hermes'
  ``session_search`` tool, and keeping the two surfaces distinct is the point
  of the recall policy (see ``skills/hermes-lcm/references/recall-policy.md``).
* **Every response is a page.**  A result list reports ``returned``,
  ``total``, ``page_size`` and ``has_more``; a node expansion reports the same
  over its lineage.  Message bodies are capped at ``body_chars``.
* **No unbounded transcript is ever assembled.**  ``lsm_page`` walks the store
  one page at a time with a cursor, so even "read the whole session" is a
  sequence of bounded calls the caller must make deliberately.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .storage import LCMStore, schema_version

# Request-level ceilings, mirroring config.HARD_* so the tool schema itself
# cannot be talked into a wider bound than the store enforces.
from .config import HARD_MAX_BODY_CHARS, HARD_MAX_PAGE_SIZE, HARD_MAX_SEARCH_RESULTS

SCOPE_SESSION = "session"


def _require_store(store: Optional[LCMStore]) -> LCMStore:
    if store is None or not store.is_open:
        raise RuntimeError("LCM store is not open")
    return store


def policy() -> Dict[str, Any]:
    """Machine-readable description of the recall policy (for diagnostics)."""
    return {
        "scope": SCOPE_SESSION,
        "scope_note": (
            "LCM recall is scoped to the current session and DAG lineage. "
            "Broad cross-session history search is Hermes session_search, not lcm_*."
        ),
        "bounded": True,
        "max_page_size": HARD_MAX_PAGE_SIZE,
        "max_search_results": HARD_MAX_SEARCH_RESULTS,
        "max_body_chars": HARD_MAX_BODY_CHARS,
        "unbounded_transcript_loads": False,
    }


def search(
    store: LCMStore,
    session_id: str,
    query: str,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Full-text search over this session's raw messages (bounded)."""
    store = _require_store(store)
    hits = store.search(session_id, query, limit)
    return {
        "session_id": session_id,
        "query": str(query),
        "returned": len(hits),
        "results": hits,
        "bounded": True,
        "has_more": len(hits) >= store.max_search_results,
        "note": (
            "Matches carry message_id; call lcm_expand(message_id=...) to read "
            "the stored message."
        ),
    }


def expand(
    store: LCMStore,
    session_id: str,
    *,
    node_id: Optional[str] = None,
    message_id: Optional[str] = None,
    page: int = 0,
    page_size: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve one summary node or one raw message, in bounded pages.

    Exactly one of ``node_id`` / ``message_id`` is required.
    """
    store = _require_store(store)
    if node_id and message_id:
        return _error("pass either node_id or message_id, not both")
    if node_id:
        return _expand_node(store, session_id, node_id, page=page, page_size=page_size, max_chars=max_chars)
    if message_id:
        return _expand_message(store, session_id, message_id, max_chars=max_chars)
    return _error("one of node_id or message_id is required")


def _expand_node(
    store: LCMStore,
    session_id: str,
    node_id: str,
    *,
    page: int,
    page_size: Optional[int],
    max_chars: Optional[int],
) -> Dict[str, Any]:
    node = store.node(node_id)
    if node is None or node["session_id"] != session_id:
        return _error(f"unknown node for this session: {node_id}")

    body_cap = _body_cap(max_chars, store.body_chars)
    try:
        lineage = store.page_node_sources(node_id, page=page, page_size=page_size)
    except Exception as exc:  # pragma: no cover - defensive
        return _error(f"lineage page failed: {exc}")

    resolved = []
    for edge in lineage["items"]:
        if edge["kind"] == "message":
            record = store.message(session_id, edge["id"])
            if record is None:
                resolved.append({"kind": "message", "id": edge["id"], "missing": True})
                continue
            resolved.append(
                {
                    "kind": "message",
                    "id": edge["id"],
                    "seq": record["seq"],
                    "role": record["role"],
                    "body": record["body"][:body_cap],
                    "truncated": len(record["body"]) > body_cap,
                }
            )
        else:
            child = store.node(edge["id"])
            resolved.append(
                {
                    "kind": "node",
                    "id": edge["id"],
                    "level": child["level"] if child else None,
                    "note": "superseded summary node — expand it to descend the DAG",
                }
            )

    return {
        "session_id": session_id,
        "node": {
            "node_id": node["node_id"],
            "level": node["level"],
            "source_count": node["source_count"],
            "summary": node["summary"],
        },
        "lineage": resolved,
        "returned": len(resolved),
        "total": lineage["total"],
        "page": lineage["page"],
        "page_size": lineage["page_size"],
        "has_more": lineage["has_more"],
        "next_page": lineage["page"] + 1 if lineage["has_more"] else None,
        "body_chars": body_cap,
        "bounded": True,
    }


def _expand_message(
    store: LCMStore, session_id: str, message_id: str, *, max_chars: Optional[int]
) -> Dict[str, Any]:
    record = store.message(session_id, message_id)
    if record is None:
        return _error(f"unknown message for this session: {message_id}")
    body_cap = _body_cap(max_chars, store.body_chars)
    return {
        "session_id": session_id,
        "message": {
            "message_id": record["message_id"],
            "seq": record["seq"],
            "role": record["role"],
            "compacted": record["compacted"],
            "body": record["body"][:body_cap],
            "truncated": len(record["body"]) > body_cap,
        },
        "body_chars": body_cap,
        "bounded": True,
    }


def page(
    store: LCMStore,
    session_id: str,
    *,
    cursor: int = 0,
    page_size: Optional[int] = None,
) -> Dict[str, Any]:
    """One bounded page of raw messages, ascending by ``seq``."""
    store = _require_store(store)
    try:
        window = store.page_window(session_id, after_seq=int(cursor or 0), page_size=page_size)
    except (TypeError, ValueError):
        return _error("cursor must be an integer seq offset")
    return {
        "session_id": session_id,
        "items": window["items"],
        "returned": window["returned"],
        "total": window["total"],
        "page_size": window["page_size"],
        "has_more": window["has_more"],
        "next_cursor": window["next_cursor"],
        "bounded": True,
        "note": "pass next_cursor as cursor to continue; the session is never returned whole.",
    }


def status(store: LCMStore, session_id: str) -> Dict[str, Any]:
    """Diagnostics: store shape, schema version, bounds, and the policy."""
    store = _require_store(store)
    return {
        "session_id": session_id,
        "db_path": str(store.db_path),
        "policy": policy(),
        "bounds": {
            "page_size": store.page_size,
            "max_page_size": store.max_page_size,
            "max_search_results": store.max_search_results,
            "body_chars": store.body_chars,
        },
        "stats": store.stats(session_id),
    }


def _body_cap(requested: Optional[int], default: int) -> int:
    if requested is None:
        return default
    try:
        parsed = int(requested)
    except (TypeError, ValueError):
        return default
    return max(200, min(parsed, HARD_MAX_BODY_CHARS))


def _error(message: str) -> Dict[str, Any]:
    return {"error": message, "bounded": True}


__all__ = [
    "SCOPE_SESSION",
    "policy",
    "search",
    "expand",
    "page",
    "status",
    "schema_version",
]
