"""Model Observability Plugin v2 — evidence-based model tracking + enforcement.

Subscribes to four hooks:

1. post_api_request (v1, unchanged)
   Fires after every LLM API call (parent + subagents). Writes a JSONL record
   to ~/.hermes/logs/model_usage.jsonl.

2. on_session_start (v1, unchanged)
   Writes a session boundary marker to the JSONL log.

3. pre_tool_call (v2, NEW)
   Fires BEFORE delegate_task executes. Serves two purposes:
     a) Scoping anchor: captures the JSONL log byte offset so transform_tool_result
        reads only records written by THIS delegation's children.
     b) Missing-pin detection: flags whether the caller specified a model.
   Always returns None — never blocks (soft enforcement only).
   Also evicts TTL-expired stash entries on each fire.

4. transform_tool_result (v2, NEW)
   Fires after delegate_task completes, before the result goes back to the LLM.
   Reads the stash entry set by pre_tool_call, reads only JSONL entries past
   the saved byte offset (perfect scoping — no prior-session bleed), builds
   an observability block, and injects it into the result string.
   - If missing_pin=True: notes auto-router resolution (soft warning, no ⚠️)
   - If match=False on an explicitly pinned model: injects a prominent WARNING
   Pops the stash entry after use to prevent memory leak.

Timing guarantee (verified from delegate_tool.py):
   delegate_task uses ThreadPoolExecutor and blocks (while pending:) until ALL
   child futures complete. Each child's post_api_request (JSONL write) fires
   inside _run_single_child on the worker thread before that future resolves.
   By the time transform_tool_result fires in the parent, all subagent JSONL
   writes are complete. No race condition.

Record schema (one JSON object per line in model_usage.jsonl):
    ts             — ISO 8601 UTC timestamp
    session_id     — parent session ID
    task_id        — delegated task ID (None for parent calls)
    agent_type     — "parent" | "subagent"
    model_request  — model the agent was configured with (what we asked for)
    model_response — model field echoed back in API response (what answered)
    provider       — provider the agent was configured with
    api_mode       — e.g. "chat_completions", "anthropic_messages"
    api_call       — which API call within this agent's lifetime (1-indexed)
    duration_s     — wall clock duration of this single API call
    finish_reason  — stop / tool_calls / length / etc.
    tokens_in      — prompt token count
    tokens_out     — completion token count
    assistant_chars — length of assistant text output
    tool_calls     — number of tool calls in this response
    platform       — cli / telegram / discord / etc.
    match          — True if model_request == model_response (normalized)

Query with: python3 ~/.hermes/scripts/model_usage.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Resolve log path once. ~/.hermes/logs/ is the established convention.
LOG_PATH = Path(os.path.expanduser("~/.hermes/logs/model_usage.jsonl"))

# Thread lock for concurrent subagent writes.
_write_lock = threading.Lock()

# Stash for pre_tool_call → transform_tool_result coordination.
# Key: tool_call_id (str)
# Value: {"offset": int, "missing_pin": bool, "ts": float (monotonic)}
_delegate_stash: dict[str, dict] = {}
_stash_lock = threading.Lock()

# Stash entries older than this are evicted as dead-letter cleanup.
_STASH_TTL_S: float = 120.0


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _normalize_model(model: Any) -> str:
    """Lowercase + strip for case-insensitive comparison."""
    if not model:
        return ""
    return str(model).strip().lower()


def _models_match(req: Any, resp: Any) -> bool:
    """True if the requested model and the response model are effectively the same.

    Two cases we treat as a match:
    1. Exact equality after normalize (the common case).
    2. The response model is a date-versioned alias of the requested model —
       e.g. "anthropic/claude-sonnet-4.6" vs "anthropic/claude-4.6-sonnet-20260217",
       or "openai/gpt-5.1-codex" vs "openai/gpt-5.1-codex-20251113".
       Detection heuristic: the response model starts with the requested model
       string (after stripping any provider prefix), OR the requested model
       string appears as a substring of the response model after the last '/'.

    Intentionally NOT a match:
    - openrouter/auto → <anything>  (auto-router resolution is real signal)
    - genuinely different models
    """
    r = _normalize_model(req)
    s = _normalize_model(resp)

    if not r or not s:
        return False

    if r in ("openrouter/auto", "auto"):
        return False

    if r == s:
        return True

    def _bare(m: str) -> str:
        return m.rsplit("/", 1)[-1] if "/" in m else m

    r_bare = _bare(r)
    s_bare = _bare(s)

    import re

    # Date-suffix heuristic
    if s_bare.startswith(r_bare) and len(s_bare) > len(r_bare):
        suffix = s_bare[len(r_bare):]
        if re.match(r"^-\d{6,}", suffix):
            return True

    if r_bare.startswith(s_bare) and len(r_bare) > len(s_bare):
        suffix = r_bare[len(s_bare):]
        if re.match(r"^-\d{6,}", suffix):
            return True

    # Token-set heuristic: handles Anthropic's name reordering
    def _non_date_tokens(name: str) -> frozenset:
        parts = re.split(r"[.-]", name)
        return frozenset(p for p in parts if p and not re.fullmatch(r"\d{6,}", p))

    if _non_date_tokens(r_bare) == _non_date_tokens(s_bare) and _non_date_tokens(r_bare):
        return True

    return False


def _extract_tokens(usage: Any) -> tuple[int, int]:
    """Pull prompt/completion tokens from the usage dict robustly."""
    if not isinstance(usage, dict):
        return 0, 0
    tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    tokens_out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    try:
        return int(tokens_in), int(tokens_out)
    except (ValueError, TypeError):
        return 0, 0


def _log_byte_offset() -> int:
    """Return the current byte size of LOG_PATH, or 0 if it doesn't exist."""
    try:
        return LOG_PATH.stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def _read_log_from_offset(offset: int) -> list[dict]:
    """Read JSONL records written after the given byte offset."""
    try:
        if not LOG_PATH.exists():
            return []
        with open(LOG_PATH, "rb") as f:
            f.seek(offset)
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records
    except Exception:
        return []


def _is_auto_router(model_request: str) -> bool:
    return _normalize_model(model_request) in ("openrouter/auto", "auto", "")


def _is_pareto_router(model_request: str) -> bool:
    return _normalize_model(model_request) in ("openrouter/pareto-code", "pareto-code")


def _detect_missing_pin(args: Any) -> bool:
    """Return True if delegate_task was called without a model pin."""
    if not isinstance(args, dict):
        return True
    # Single-task form: check top-level "model"
    if "goal" in args or "context" in args:
        return not bool(args.get("model"))
    # Batch form: check each task
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        return any(not isinstance(t, dict) or not t.get("model") for t in tasks)
    # Empty or unrecognized args
    return True


def _evict_expired_stash() -> None:
    """Remove stash entries older than _STASH_TTL_S."""
    now = time.monotonic()
    with _stash_lock:
        expired = [k for k, v in _delegate_stash.items()
                   if now - v.get("ts", 0) > _STASH_TTL_S]
        for k in expired:
            del _delegate_stash[k]


# ---------------------------------------------------------------------------
# Hook: post_api_request (v1, unchanged)
# ---------------------------------------------------------------------------

def _on_post_api_request(**kwargs: Any) -> None:
    """Write a JSONL record for one LLM API call."""
    try:
        task_id = kwargs.get("task_id") or None
        model_request = kwargs.get("model") or ""
        model_response = kwargs.get("response_model") or ""
        provider = kwargs.get("provider") or ""

        tokens_in, tokens_out = _extract_tokens(kwargs.get("usage"))

        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": kwargs.get("session_id") or "",
            "task_id": task_id,
            "agent_type": "subagent" if task_id else "parent",
            "model_request": model_request,
            "model_response": model_response,
            "provider": provider,
            "api_mode": kwargs.get("api_mode") or "",
            "api_call": kwargs.get("api_call_count") or 0,
            "duration_s": round(float(kwargs.get("api_duration") or 0.0), 3),
            "finish_reason": kwargs.get("finish_reason") or "",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "assistant_chars": int(kwargs.get("assistant_content_chars") or 0),
            "tool_calls": int(kwargs.get("assistant_tool_call_count") or 0),
            "platform": kwargs.get("platform") or "",
            "match": _models_match(model_request, model_response),
        }

        line = json.dumps(record, ensure_ascii=False)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.debug("model_observability write failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook: on_session_start (v1, unchanged)
# ---------------------------------------------------------------------------

def _on_session_start(**kwargs: Any) -> None:
    """Write a session boundary marker so logs are groupable."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "session_start",
            "session_id": kwargs.get("session_id") or "",
            "platform": kwargs.get("platform") or "",
            "model": kwargs.get("model") or "",
            "provider": kwargs.get("provider") or "",
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("model_observability session_start write failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook: pre_tool_call (v2, NEW)
# ---------------------------------------------------------------------------

def _on_pre_tool_call(**kwargs: Any) -> Optional[dict]:
    """Scoping anchor + missing-pin detection for delegate_task.

    Always returns None — never blocks or warns directly.
    Side effects only: stashes byte offset + missing_pin flag keyed on tool_call_id,
    and evicts expired stash entries.
    """
    try:
        _evict_expired_stash()

        tool_name = kwargs.get("tool_name") or ""
        if tool_name != "delegate_task":
            return None

        tool_call_id = kwargs.get("tool_call_id") or ""
        args = kwargs.get("args") or {}

        offset = _log_byte_offset()
        missing_pin = _detect_missing_pin(args)

        with _stash_lock:
            _delegate_stash[tool_call_id] = {
                "offset": offset,
                "missing_pin": missing_pin,
                "ts": time.monotonic(),
            }
    except Exception as exc:
        logger.debug("model_observability pre_tool_call failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Hook: transform_tool_result (v2, NEW)
# ---------------------------------------------------------------------------

def _on_transform_tool_result(**kwargs: Any) -> Optional[str]:
    """Enrich delegate_task results with model observability data.

    Reads the stash entry set by pre_tool_call, reads only JSONL entries past
    the saved byte offset, builds an observability block, and injects it into
    the result string. Pops the stash entry after use.

    Returns the enriched result string, or None to leave the result unchanged.
    """
    try:
        tool_name = kwargs.get("tool_name") or ""
        if tool_name != "delegate_task":
            return None

        tool_call_id = kwargs.get("tool_call_id") or ""
        original_result = kwargs.get("result") or ""

        # Retrieve and pop stash entry
        with _stash_lock:
            stash_entry = _delegate_stash.pop(tool_call_id, None)

        if stash_entry is None:
            # pre_tool_call didn't fire (e.g. plugin loaded mid-session) — degrade gracefully
            return None

        offset = stash_entry["offset"]
        missing_pin = stash_entry["missing_pin"]

        # Read only records written after the scoping anchor
        records = _read_log_from_offset(offset)
        subagent_records = [
            r for r in records
            if isinstance(r, dict) and r.get("agent_type") == "subagent"
        ]

        if not subagent_records:
            # No subagent calls — delegation may have failed before children ran,
            # or no children wrote JSONL. Inject a minimal note if pin was missing,
            # but never a false mismatch warning.
            if missing_pin:
                block = "\n\n[model_observability] No model pin specified. No subagent telemetry available."
                return original_result + block
            return None

        # Classify records
        mismatches = [
            r for r in subagent_records
            if (
                not r.get("match", True)
                and not _is_auto_router(r.get("model_request", ""))
                and not _is_pareto_router(r.get("model_request", ""))
            )
        ]
        auto_resolutions = [
            r for r in subagent_records
            if not r.get("match", True) and _is_auto_router(r.get("model_request", ""))
        ]
        pareto_resolutions = [
            r for r in subagent_records
            if not r.get("match", True) and _is_pareto_router(r.get("model_request", ""))
        ]
        matched = [
            r for r in subagent_records
            if r.get("match", True)
        ]

        # Build observability block
        lines = ["\n\n[model_observability]"]

        # Aggregate stats
        total_tokens_in = sum(r.get("tokens_in", 0) for r in subagent_records)
        total_tokens_out = sum(r.get("tokens_out", 0) for r in subagent_records)
        total_duration = sum(r.get("duration_s", 0.0) for r in subagent_records)
        lines.append(
            f"  subagents: {len(subagent_records)} | "
            f"tokens in: {total_tokens_in:,} | tokens out: {total_tokens_out:,} | "
            f"duration: {total_duration:.1f}s"
        )

        # Matched pins
        if matched:
            unique_matched = sorted({r.get("model_request", "") for r in matched})
            lines.append(f"  ✓ matched: {', '.join(unique_matched)}")

        # Auto-router resolutions (informational, no warning)
        if auto_resolutions:
            for r in auto_resolutions:
                lines.append(
                    f"  → auto-router resolved to: {r.get('model_response', 'unknown')}"
                )

        # Pareto-router resolutions (informational, no warning)
        if pareto_resolutions:
            for r in pareto_resolutions:
                lines.append(
                    f"  → pareto-router resolved to: {r.get('model_response', 'unknown')}"
                )

        # Missing pin warning (soft)
        if missing_pin and not mismatches:
            lines.append("  ℹ no model pin specified — auto-router was used")

        # Override mismatches (prominent WARNING)
        if mismatches:
            lines.append("  ⚠️ WARNING: model override mismatch detected")
            for r in mismatches:
                lines.append(
                    f"    requested: {r.get('model_request', '?')} "
                    f"→ actual: {r.get('model_response', '?')}"
                )

        block = "\n".join(lines)
        return original_result + block

    except Exception as exc:
        logger.debug("model_observability transform_tool_result failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Entry point called by the plugin manager at load time."""
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    logger.info("model_observability plugin v2 registered (log: %s)", LOG_PATH)
