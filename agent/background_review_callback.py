"""Background-review improvement_record callback support.

Sibling module to ``agent.background_review``. The runtime wiring (the
inline block in ``_run_review_in_thread``) calls into
:func:`run_callback_pipeline` here; everything else in this module is
the supporting machinery.

Responsibility split:
  - ``background_review.py`` owns the review-thread lifecycle and the
    user-facing summary surfacing.
  - ``background_review_callback.py`` (this module) owns tool-call
    classification, improvement_record construction, callback
    resolution + caching, and bounded-timeout dispatch.

Downstream consumers subscribe via config ``background_review.callback``
(dotted path). Callback failures are swallowed; a misbehaving callback
never crashes the review thread or the parent agent's turn.
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_MEMORY_TOOL_NAMES = ("memory", "categorical_memory")
_SKILL_TOOL_NAMES = ("skill_manage",)
# Combined set used wherever the classifier needs "is this a write tool?".
_WRITE_TOOL_NAMES = _MEMORY_TOOL_NAMES + _SKILL_TOOL_NAMES


def _build_improvement_records_from_review_messages(
    messages: List[Dict],
    *,
    session_id: str,
    ts: str,
) -> List[Dict]:
    """Walk the review_agent's message buffer and emit improvement_records.

    Supports two message shapes:

    1. **Hermes runtime shape (production)** — assistant messages with a
       ``tool_calls`` array (each containing ``id`` + ``function.name`` +
       ``function.arguments`` JSON string), followed by ``role: "tool"``
       results keyed by ``tool_call_id``.

    2. **Abstract test shape (unit tests only)** — ``role: "tool_use"`` and
       ``role: "tool_result"`` messages. Hermes never produces this shape;
       it exists so tests can be written without constructing full
       runtime-shape fixtures.

    Both shapes feed the same ``pending_tool_calls`` dict and
    ``_flush_tool`` reconciliation.
    """
    records: List[Dict] = []
    plan_buf: List[str] = []
    pending_tool_calls: Dict[str, Dict] = {}  # tool_call_id -> normalized tool_use

    def _flush_tool(tool_use: Dict, tool_result: Dict) -> None:
        op = _classify_op(tool_use)
        if op is None:
            plan_buf.clear()
            return
        write = _build_write_entry(tool_use, tool_result, op)
        if write is None:
            plan_buf.clear()
            return
        records.append({
            "schema_version": 1,
            "session_id": session_id,
            "ts": ts,
            "origin": "background_review",
            "plan": "\n".join(plan_buf).strip(),
            "writes": [write],
        })
        plan_buf.clear()

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                plan_buf.append(content)
            # Hermes runtime shape: tool_calls live on the assistant message.
            for tc in msg.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {}) or {}
                fn_name = fn.get("name", "")
                if fn_name not in _WRITE_TOOL_NAMES:
                    continue
                tcid = tc.get("id")
                if not tcid:
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                pending_tool_calls[tcid] = {
                    "name": fn_name,
                    "input": args,
                    "tool_call_id": tcid,
                }
        elif role == "tool_use":
            # Abstract test shape.
            if msg.get("name") in _WRITE_TOOL_NAMES:
                tcid = msg.get("tool_call_id")
                if tcid:
                    pending_tool_calls[tcid] = msg
        elif role in ("tool_result", "tool"):
            tcid = msg.get("tool_call_id")
            if tcid in pending_tool_calls:
                _flush_tool(pending_tool_calls.pop(tcid), msg)

    return records


def _classify_op(tool_use: Dict) -> Optional[str]:
    """Map a tool_use to a write-op label, or None if not a tracked write."""
    name = tool_use.get("name", "")
    inp = tool_use.get("input") or {}
    action = inp.get("action", "")
    if name in _SKILL_TOOL_NAMES and action in ("write_file", "patch", "patch_file"):
        # Intentionally collapse patch + patch_file → "patch": downstream
        # consumers care about the semantic operation (file mutation via
        # diff) rather than the specific skill_manage subaction.
        return "patch" if "patch" in action else "write_file"
    if name in _MEMORY_TOOL_NAMES:
        if action == "add":
            return "memory_add"
        if action in ("update", "replace"):
            return "memory_update"
        if action in ("remove", "delete"):
            return "memory_delete"
    return None


def _build_write_entry(tool_use: Dict, tool_result: Dict, op: str) -> Optional[Dict]:
    """Construct the per-write entry attached to an improvement_record."""
    inp = tool_use.get("input") or {}
    raw = tool_result.get("content")
    try:
        result = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        result = {"success": False, "error": (raw or "")[:200]}
    if not isinstance(result, dict):
        result = {"success": False, "error": str(result)[:200]}
    success = bool(result.get("success", False))
    err = (result.get("error") or "")[:200] if not success else None
    path = result.get("path") or inp.get("path") or ""
    diff = result.get("diff") if success and op in ("write_file", "patch") else None
    post = inp.get("content") if success and op in ("write_file", "patch") else None
    return {
        "path": path,
        "op": op,
        "success": success,
        "error_preview": err,
        "diff": diff,
        "post_content": post,
    }


def _invoke_callback_with_timeout(
    callback: Callable[[Dict], None],
    record: Dict,
    *,
    timeout: float,
) -> None:
    """Run ``callback(record)`` under a timeout; swallow all exceptions.

    Failures are logged at WARNING; nothing propagates. A hung callback is
    abandoned after ``timeout`` seconds (the daemon thread is left to
    finish on its own).
    """
    def _safe() -> None:
        try:
            callback(record)
        except Exception as e:
            logger.warning("background_review callback raised: %s", e)

    t = threading.Thread(target=_safe, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.warning(
            "background_review callback timed out after %.1fs", timeout
        )


# Module-level cache of resolved callbacks, keyed by dotted path. Read and
# written from the daemon review thread; guard with ``_callback_cache_lock``.
# Failed loads are cached as ``None`` so a misconfigured dotted path is
# importlib-resolved at most once per process.
_callback_cache: Dict[str, Optional[Callable[[Dict], None]]] = {}
_callback_cache_lock = threading.Lock()


def _load_callback(dotted: str) -> Optional[Callable[[Dict], None]]:
    """Resolve a dotted module path to a callable. Cached. None on failure."""
    with _callback_cache_lock:
        if dotted in _callback_cache:
            return _callback_cache[dotted]
    try:
        mod_path, _, attr = dotted.rpartition(".")
        if not mod_path:
            logger.warning(
                "background_review callback path %r missing module prefix", dotted
            )
            with _callback_cache_lock:
                _callback_cache[dotted] = None
            return None
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, attr, None)
        if fn is None or not callable(fn):
            logger.warning(
                "background_review callback %r not a callable", dotted
            )
            with _callback_cache_lock:
                _callback_cache[dotted] = None
            return None
        with _callback_cache_lock:
            _callback_cache[dotted] = fn
        return fn
    except Exception as e:
        logger.warning(
            "background_review callback load failed for %r: %s", dotted, e
        )
        with _callback_cache_lock:
            _callback_cache[dotted] = None
        return None


def run_callback_pipeline(agent: Any, review_messages: List[Dict]) -> None:
    """Build improvement_records from a review_agent's messages and dispatch
    each to the configured callback.

    Safe to call unconditionally — no-ops when no callback is configured.
    Never raises.
    """
    try:
        cfg = (getattr(agent, "config", None) or {}).get("background_review") or {}
        dotted = cfg.get("callback")
        if not dotted:
            return
        fn = _load_callback(dotted)
        if fn is None:
            return
        # Build records once per review batch — the same ts is shared by all
        # records produced in this review (review batch timestamp, not
        # per-write).
        records = _build_improvement_records_from_review_messages(
            review_messages,
            session_id=getattr(agent, "session_id", "") or "unknown",
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        timeout = float(cfg.get("callback_timeout_seconds", 5))
        for rec in records:
            _invoke_callback_with_timeout(fn, rec, timeout=timeout)
    except Exception as e:
        logger.warning("background_review callback pipeline failed: %s", e)
