"""Structured audit log for Hermes Agent.

Writes JSONL (one JSON object per line) to ~/.hermes/logs/audit/ with
per-session files.  Designed for ``tail -f`` in a second terminal:

    tail -f ~/.hermes/logs/audit/latest.jsonl | jq .

Events are categorized by type using a dotted namespace convention:

    session.start    -- session created (model, provider, platform, tools)
    session.end      -- session ended (duration, token totals, cost)
    api.request      -- LLM API call (model, provider, tokens, cost, duration)
    tool.call        -- tool invoked (name, detail, result summary, duration)
    tool.error       -- tool execution failed (name, error)
    cli.launch       -- CLI started (model, session, toolsets)
    context.*        -- context assembly and compression events
    memory.*         -- local memory operations
    honcho.*         -- honcho/memory operations
    cron.*           -- cron job executions
    mcp.*            -- MCP server/tool events
    skill.*          -- skill load/manage events
    plugin.*         -- external plugin events

The source is derived from the event type prefix (text before the first
dot).  Each source can be independently enabled/disabled via the
``audit.sources`` config block:

    audit:
      sources:
        core: true      # session, api, tool, cli, context events
        honcho: true    # honcho.* events
        memory: true    # memory.* events
        cron: true      # cron.* events
        mcp: true       # MCP tool/server events
        skill: true     # skill load/manage events
        plugin: true    # external plugin events

Writes are async (background thread with queue) to avoid blocking the agent.

Config (config.yaml):

    audit:
      enabled: true              # master switch (default: true)
      redact: true               # redact sensitive args (default: true)
      retention_days: 30         # auto-clean logs older than N days (0 = keep forever)
      jsonl: true                # write JSONL files for hermes tail
      sources: {...}             # per-source enable/disable (all true by default)
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_AUDIT_DIR = _HERMES_HOME / "logs" / "audit"
_LATEST_LINK = _AUDIT_DIR / "latest.jsonl"

# Module-level state
_log_file = None
_log_path: Optional[Path] = None
_session_id: Optional[str] = None
_enabled: Optional[bool] = None
_redact: bool = True
_retention_days: int = 30

# SQLite audit DB (SessionDB instance, shared with hermes_state)
_db = None  # type: Any  # SessionDB | None
_user_id: Optional[str] = None
_platform: Optional[str] = None

# Per-source enable/disable (all True by default).
# Keys are source names; values are booleans.
_sources: Dict[str, bool] = {}

# Async write queue + worker
_write_queue: queue.Queue = queue.Queue()
_writer_thread: Optional[threading.Thread] = None
_writer_stop = threading.Event()

# Prefixes that map to the "core" source.  Everything else uses its own
# prefix as the source name (honcho.* → "honcho", cron.* → "cron", etc.).
_CORE_PREFIXES = frozenset({
    "session", "api", "tool", "cli", "context", "user",
})


# Keys whose values should be redacted when audit_log_redact is true
_SENSITIVE_KEYS = frozenset({
    "password", "secret", "token", "api_key", "apikey",
    "authorization", "credential", "private_key",
})


def _resolve_source(event_type: str) -> str:
    """Derive the audit source from an event type string.

    Core events (session.*, api.*, tool.*, cli.*, context.*) map to
    ``"core"``.  Everything else uses the prefix before the first dot
    (e.g. ``"honcho.operation"`` → ``"honcho"``).

    Unknown or un-dotted types default to ``"core"``.
    """
    prefix = event_type.split(".", 1)[0] if "." in event_type else event_type
    if prefix in _CORE_PREFIXES:
        return "core"
    return prefix


def _source_enabled(event_type: str) -> bool:
    """Return True if the source for *event_type* is enabled.

    When no sources dict is configured (the default), everything is
    enabled.  When a sources dict is present, missing keys default True
    so new sources are opt-out rather than opt-in.
    """
    if not _sources:
        return True
    source = _resolve_source(event_type)
    return _sources.get(source, True)


def _matches_source_filter(event_type: str, source: Optional[str]) -> bool:
    """Return True if *event_type* matches the requested *source* filter.

    ``source=None`` means no filtering.  The matching logic uses the same
    source derivation as ``_resolve_source`` so old records without an
    explicit ``source`` field still filter correctly.
    """
    if not source:
        return True
    return _resolve_source(event_type) == source


def _should_redact(key: str) -> bool:
    return _redact and key.lower() in _SENSITIVE_KEYS


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of args with sensitive values replaced."""
    if not _redact or not isinstance(args, dict):
        return args
    out = {}
    for k, v in args.items():
        if _should_redact(k):
            out[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 2000:
            out[k] = v[:2000] + f"... ({len(v)} chars)"
        else:
            out[k] = v
    return out


def _sanitize_result(result: str, max_len: int = 2000) -> str:
    """Truncate long results for readability."""
    if not isinstance(result, str):
        result = str(result)
    if len(result) > max_len:
        return result[:max_len] + f"... ({len(result)} chars)"
    return result


def is_enabled() -> bool:
    """Return True if audit logging is active."""
    return _enabled is True and _log_file is not None


def emit(event_type: str, **kwargs) -> None:
    """General-purpose audit event emitter.

    Any subsystem can call this to log an event.  The event_type is a
    dotted string (e.g. 'memory.loaded', 'honcho.operation',
    'mcp.tool_call') and kwargs become the JSON payload.

    The source is derived from the event type prefix (see
    ``_resolve_source``).  If the source is disabled via the
    ``audit.sources`` config block, the event is silently dropped.

    Values that look sensitive are auto-redacted.  Long strings are
    truncated.  This is the primary interface — the specialised
    log_*() helpers are thin wrappers around this.
    """
    if not is_enabled():
        return
    if not _source_enabled(event_type):
        return
    # Light sanitisation: redact known-sensitive keys, truncate long values
    safe = {}
    for k, v in kwargs.items():
        if _should_redact(k):
            safe[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 2000:
            safe[k] = v[:2000] + f"... ({len(v)} chars)"
        elif isinstance(v, dict):
            safe[k] = _sanitize_args(v)
        else:
            safe[k] = v
    _write(event_type, **safe)


def configure(*, enabled: bool = False, redact: bool = True, retention_days: int = 30,
              user_id: Optional[str] = None, platform: Optional[str] = None,
              sources: Optional[Dict[str, bool]] = None) -> None:
    """Set module-level config.  Called once at session start.

    ``sources`` is an optional dict mapping source names to booleans.
    When provided, only events whose source is enabled will be written.
    Missing keys default to ``True`` (opt-out, not opt-in).  Example::

        configure(enabled=True, sources={"core": True, "mcp": False})
    """
    global _enabled, _redact, _retention_days, _user_id, _platform, _db, _sources
    _enabled = enabled
    _redact = redact
    _retention_days = retention_days
    if user_id is not None:
        _user_id = user_id
    if platform is not None:
        _platform = platform
    if sources is not None and isinstance(sources, dict):
        _sources = sources
    else:
        _sources = {}
    # Reset DB reference when disabled so tests/reconfigure get clean state
    if not enabled:
        _db = None


def start_session(session_id: str, *, user_id: Optional[str] = None,
                  platform: Optional[str] = None, **metadata) -> None:
    """Open the audit log file for a new session.

    Idempotent: if the log file is already open for this session_id, this
    is a no-op.  This allows both the CLI (early init) and AIAgent.__init__
    to call start_session without double-opening files or writer threads.
    """
    global _log_file, _log_path, _session_id, _writer_thread, _db

    if not _enabled:
        return

    # Already running for this session — nothing to do.
    if _log_file is not None and _session_id == session_id:
        return

    _session_id = session_id
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    _rotate_logs()

    log_path = _AUDIT_DIR / f"{session_id}.jsonl"
    _log_path = log_path
    _log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    # Update the "latest" symlink
    try:
        if _LATEST_LINK.is_symlink() or _LATEST_LINK.exists():
            _LATEST_LINK.unlink()
        _LATEST_LINK.symlink_to(log_path.name)
    except OSError:
        pass

    # Initialize SQLite audit DB (in the audit dir alongside JSONL files)
    try:
        if _db is None:
            from hermes_state import SessionDB
            audit_db_path = _AUDIT_DIR / "audit.db"
            _db = SessionDB(audit_db_path)
    except Exception:
        pass  # SQLite unavailable — JSONL-only mode

    # Store user_id / platform from params or module defaults
    if user_id is not None:
        metadata["user_id"] = user_id
    elif _user_id is not None:
        metadata["user_id"] = _user_id
    if platform is not None:
        metadata["platform"] = platform
    elif _platform is not None:
        metadata["platform"] = _platform

    # Start async writer thread
    _writer_stop.clear()
    _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
    _writer_thread.start()

    _write("session.start", session_id=session_id, **metadata)


def end_session(**metadata) -> Optional[Path]:
    """Close the audit log for the current session.  Returns the log file path."""
    global _log_file, _writer_thread
    if not is_enabled():
        return None
    _write("session.end", log_path=str(_log_path) if _log_path else "", **metadata)
    # Drain the queue before closing
    _writer_stop.set()
    if _writer_thread and _writer_thread.is_alive():
        _writer_thread.join(timeout=2.0)
    # Flush remaining
    _drain_queue()
    path = _log_path
    try:
        _log_file.close()
    except Exception:
        pass
    _log_file = None
    # If another session is still running, repoint the symlink to it
    # so that hermes tail doesn't go dead
    try:
        if path and _AUDIT_DIR.exists():
            candidates = [
                f for f in _AUDIT_DIR.glob("*.jsonl")
                if f.name != "latest.jsonl" and f.name != path.name
            ]
            if candidates:
                newest = max(candidates, key=lambda f: f.stat().st_mtime)
                # Only repoint if the candidate was modified very recently (likely still active)
                if time.time() - newest.stat().st_mtime < 10:
                    try:
                        if _LATEST_LINK.is_symlink() or _LATEST_LINK.exists():
                            _LATEST_LINK.unlink()
                        _LATEST_LINK.symlink_to(newest.name)
                    except OSError:
                        pass
    except Exception:
        pass
    # Prune old audit events in SQLite
    try:
        if _db is not None and _retention_days > 0:
            _db.prune_audit_events(retention_days=_retention_days)
    except Exception:
        pass
    return path


def log_api_request(
    *,
    model: str = "",
    provider: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cache_read_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost_usd: Optional[float] = None,
    duration_ms: Optional[float] = None,
    finish_reason: str = "",
    tool_calls_count: int = 0,
) -> None:
    """Log an LLM API request/response."""
    if not is_enabled():
        return
    _write(
        "api.request",
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        finish_reason=finish_reason,
        tool_calls_count=tool_calls_count,
    )


def log_tool_call(
    *,
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: str = "",
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log a tool call with rich per-tool detail extracted from args/result."""
    if not is_enabled():
        return
    event_type = "tool.error" if error else "tool.call"
    safe_args = _sanitize_args(args or {})
    detail = _extract_tool_detail(tool_name, safe_args, result)

    payload = {
        "tool": tool_name,
        "args": safe_args,
        "detail": detail,
        "duration_ms": duration_ms,
    }
    if error:
        payload["error"] = str(error)
    else:
        payload["result_summary"] = _sanitize_result(result)
    _write(event_type, **payload)


def log_honcho_operation(
    *,
    operation: str,
    payload: Optional[Dict[str, Any]] = None,
    result: str = "",
) -> None:
    """Log a honcho/memory operation.  Thin wrapper around emit()."""
    emit(
        "honcho.operation",
        operation=operation,
        payload=payload or {},
        result_summary=_sanitize_result(result, max_len=500),
    )


def log_cron_run(
    *,
    job_id: str,
    job_name: str,
    success: bool,
    duration_s: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log a cron job execution.  Thin wrapper around emit()."""
    emit(
        "cron.run",
        job_id=job_id,
        job_name=job_name,
        success=success,
        duration_s=duration_s,
        error=error,
    )


# -- Rich tool detail extraction --------------------------------------------

def _extract_tool_detail(tool_name: str, args: dict, result: str) -> dict:
    """Extract structured detail per tool type for display in hermes tail."""
    detail = {}
    try:
        if tool_name == "terminal":
            detail["command"] = args.get("command", "")[:200]
            # Parse exit code from result
            _parse_terminal_result(result, detail)

        elif tool_name == "execute_code":
            code = args.get("code", "")
            detail["language"] = args.get("language", "python")
            detail["code_preview"] = code[:150] + ("..." if len(code) > 150 else "")
            _parse_terminal_result(result, detail)

        elif tool_name == "read_file":
            detail["path"] = args.get("file_path", args.get("path", ""))
            detail["offset"] = args.get("offset")
            detail["limit"] = args.get("limit")
            # Extract size from result
            if result:
                detail["result_bytes"] = len(result)

        elif tool_name == "write_file":
            detail["path"] = args.get("file_path", args.get("path", ""))
            content = args.get("content", "")
            detail["content_bytes"] = len(content) if isinstance(content, str) else 0

        elif tool_name in ("patch", "patch_file"):
            detail["path"] = args.get("path", args.get("file_path", ""))
            mode = args.get("mode", "replace")
            detail["mode"] = mode
            if mode == "replace":
                old = args.get("old_string", "")
                new = args.get("new_string", "")
                detail["old_bytes"] = len(old) if isinstance(old, str) else 0
                detail["new_bytes"] = len(new) if isinstance(new, str) else 0
            else:
                diff = args.get("patch", "")
                detail["diff_bytes"] = len(diff) if isinstance(diff, str) else 0

        elif tool_name == "search_files":
            detail["query"] = args.get("query", args.get("pattern", ""))[:100]
            detail["path"] = args.get("path", args.get("directory", ""))

        elif tool_name in ("fetch_url", "url_fetch"):
            detail["url"] = args.get("url", "")[:200]
            if result:
                detail["result_bytes"] = len(result)

        elif tool_name in ("browser_navigate", "browser_snapshot"):
            detail["url"] = args.get("url", "")[:200]
            detail["action"] = args.get("action", "")

        elif tool_name == "web_search":
            detail["query"] = args.get("query", "")[:100]
            # Extract URLs from result
            if result:
                import re as _re
                urls = _re.findall(r'https?://[^\s"\'<>\]]+', result[:2000])
                if urls:
                    detail["result_urls"] = urls[:10]

        elif tool_name.startswith("honcho_"):
            detail["query"] = args.get("query", args.get("question", ""))[:200]
            detail["conclusion"] = args.get("conclusion", "")[:200]

        elif tool_name.startswith("mcp_"):
            # MCP tool calls -- extract server and method + sanitized args
            detail["mcp_tool"] = tool_name
            # Parse server/method from mcp__server__method pattern
            parts = tool_name.split("__")
            if len(parts) >= 3:
                detail["mcp_server"] = parts[1]
                detail["mcp_method"] = "__".join(parts[2:])
            detail["args"] = {k: str(v)[:200] for k, v in list(args.items())[:10]}

        elif tool_name == "send_message":
            detail["platform"] = args.get("platform", "")
            detail["chat_id"] = args.get("chat_id", "")

        elif tool_name == "delegate_task":
            detail["task"] = args.get("task", "")[:150]

        elif tool_name == "memory":
            detail["action"] = args.get("action", "")
            detail["content"] = args.get("content", "")[:150]

        elif tool_name == "todo":
            detail["action"] = args.get("action", "")
            detail["item"] = args.get("item", args.get("text", ""))[:100]

        elif tool_name == "skill_manage":
            detail["action"] = args.get("action", "")
            detail["skill"] = args.get("name", "")
            detail["category"] = args.get("category", "")
            if args.get("file_path"):
                detail["file_path"] = args.get("file_path", "")

    except Exception:
        pass
    return detail


def _parse_terminal_result(result: str, detail: dict) -> None:
    """Extract exit code and output summary from terminal/execute_code result."""
    try:
        if not result:
            return
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict):
            detail["exit_code"] = parsed.get("exit_code", parsed.get("returncode"))
            output = parsed.get("output", parsed.get("stdout", ""))
            stderr = parsed.get("stderr", "")
            if output:
                detail["output_preview"] = output[:200]
            if stderr:
                detail["stderr_preview"] = stderr[:200]
    except (json.JSONDecodeError, TypeError):
        pass


# -- Log rotation -----------------------------------------------------------

def _rotate_logs() -> None:
    """Remove audit log files older than retention_days."""
    if _retention_days <= 0:
        return
    try:
        cutoff = time.time() - (_retention_days * 86400)
        for f in _AUDIT_DIR.glob("*.jsonl"):
            if f.name == "latest.jsonl":
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
    except Exception:
        pass


# -- Query API (for /audit gateway command) ----------------------------------

def get_audit_dir() -> Path:
    """Return the audit log directory path."""
    return _AUDIT_DIR


def list_sessions() -> List[Dict[str, Any]]:
    """List all audit log sessions with metadata."""
    sessions = []
    if not _AUDIT_DIR.exists():
        return sessions
    for f in sorted(_AUDIT_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.name == "latest.jsonl":
            continue
        session_id = f.stem
        stat = f.stat()
        meta = {}
        try:
            with open(f, "r", encoding="utf-8") as fh:
                first = fh.readline().strip()
                if first:
                    rec = json.loads(first)
                    meta = {k: v for k, v in rec.items() if k not in ("ts", "iso", "type", "session")}
        except Exception:
            pass
        sessions.append({
            "session_id": session_id,
            "file": str(f),
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            **meta,
        })
    return sessions


def query_events(
    *,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    tool_name: Optional[str] = None,
    source: Optional[str] = None,
    keyword: Optional[str] = None,
    user_id: Optional[str] = None,
    after: Optional[float] = None,
    before: Optional[float] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Query audit events across sessions with optional filters."""
    # Prefer SQLite when available
    if _db is not None:
        try:
            return _db.query_audit_events(
                session_id=session_id,
                event_type=event_type,
                tool_name=tool_name,
                source=source,
                user_id=user_id,
                keyword=keyword,
                after=after,
                before=before,
                limit=limit,
            )
        except Exception:
            pass  # fall through to JSONL
    # Fallback to JSONL file scanning
    if not _AUDIT_DIR.exists():
        return []
    if session_id:
        files = list(_AUDIT_DIR.glob(f"*{session_id}*.jsonl"))
    else:
        files = sorted(
            [f for f in _AUDIT_DIR.glob("*.jsonl") if f.name != "latest.jsonl"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    results = []
    for f in files:
        if len(results) >= limit:
            break
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event_type and rec.get("type") != event_type:
                        continue
                    if tool_name and rec.get("tool") != tool_name:
                        continue
                    if not _matches_source_filter(rec.get("type", ""), source):
                        continue
                    if keyword and keyword.lower() not in line.lower():
                        continue
                    if "source" not in rec:
                        rec["source"] = _resolve_source(rec.get("type", ""))
                    if user_id and rec.get("user_id") != user_id:
                        continue
                    if after is not None and rec.get("ts", 0) < after:
                        continue
                    if before is not None and rec.get("ts", float("inf")) > before:
                        continue
                    results.append(rec)
                    if len(results) >= limit:
                        break
        except Exception:
            continue
    return results


def export_events(
    *,
    session_id: Optional[str] = None,
    format: str = "jsonl",
) -> str:
    """Export audit events as JSONL or CSV string."""
    events = query_events(session_id=session_id, limit=100_000)
    if format == "csv":
        import csv
        import io
        if not events:
            return ""
        buf = io.StringIO()
        all_keys = []
        for e in events:
            for k in e:
                if k not in all_keys:
                    all_keys.append(k)
        writer = csv.DictWriter(buf, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            flat = {}
            for k, v in e.items():
                flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            writer.writerow(flat)
        return buf.getvalue()
    else:
        return "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in events)


# -- Summary & problems -----------------------------------------------------

def audit_summary(
    *,
    session_id: Optional[str] = None,
    after: Optional[float] = None,
    before: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate stats across audit events.  Returns a dict of metrics."""
    events = query_events(
        session_id=session_id, after=after, before=before, limit=100_000,
    )
    if not events:
        return {"total": 0}

    api_calls = [e for e in events if e.get("type") == "api.request"]
    tool_calls = [e for e in events if e.get("type") == "tool.call"]
    errors = [e for e in events if e.get("type") == "tool.error" or e.get("error")]
    sessions = {e.get("session", "") for e in events if e.get("session")}

    total_tokens = sum(e.get("total_tokens", 0) or 0 for e in api_calls)
    total_cost = sum(e.get("cost_usd", 0) or 0 for e in api_calls)
    prompt_tokens = sum(e.get("prompt_tokens", 0) or 0 for e in api_calls)
    completion_tokens = sum(e.get("completion_tokens", 0) or 0 for e in api_calls)

    # Top tools by frequency
    tool_freq: Dict[str, int] = {}
    tool_dur: Dict[str, List[float]] = {}
    for e in tool_calls:
        t = e.get("tool", "?")
        tool_freq[t] = tool_freq.get(t, 0) + 1
        dur = e.get("duration_ms")
        if dur is not None:
            tool_dur.setdefault(t, []).append(dur)

    top_tools = sorted(tool_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    top_tools_detail = [
        {"tool": t, "count": c, "avg_ms": round(sum(tool_dur.get(t, [0])) / max(len(tool_dur.get(t, [1])), 1))}
        for t, c in top_tools
    ]

    # Top errors by frequency
    err_freq: Dict[str, int] = {}
    for e in errors:
        msg = (e.get("error") or "")[:80]
        if msg:
            err_freq[msg] = err_freq.get(msg, 0) + 1
    top_errors = sorted(err_freq.items(), key=lambda x: x[1], reverse=True)[:3]

    operational = len(api_calls) + len(tool_calls)
    error_rate = round(len(errors) / max(operational, 1) * 100, 1)

    ts_vals = [e.get("ts", 0) for e in events if e.get("ts")]
    return {
        "total": len(events),
        "sessions": len(sessions),
        "api_calls": len(api_calls),
        "tool_calls": len(tool_calls),
        "errors": len(errors),
        "error_rate": error_rate,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_cost_usd": round(total_cost, 4),
        "top_tools": top_tools_detail,
        "top_errors": [{"error": e, "count": c} for e, c in top_errors],
        "first_event": min(ts_vals) if ts_vals else None,
        "last_event": max(ts_vals) if ts_vals else None,
    }


def audit_problems(
    *,
    session_id: Optional[str] = None,
    after: Optional[float] = None,
    before: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Auto-detect anomalies in audit events.  Returns list of findings."""
    events = query_events(
        session_id=session_id, after=after, before=before, limit=100_000,
    )
    if not events:
        return []

    findings: List[Dict[str, str]] = []

    # Repeated API errors (same message 3+ times)
    api_errors: Dict[str, int] = {}
    for e in events:
        if e.get("type") in ("api.request",) and e.get("error"):
            msg = (e.get("error") or "")[:80]
            api_errors[msg] = api_errors.get(msg, 0) + 1
    for msg, count in api_errors.items():
        if count >= 3:
            findings.append({"rule": "repeated_api_error", "message": f"\"{msg}\" occurred {count} times"})

    # High error rate (>25%)
    operational = sum(1 for e in events if e.get("type") in ("api.request", "tool.call"))
    error_count = sum(1 for e in events if e.get("type") == "tool.error" or e.get("error"))
    if operational > 0 and (error_count / operational) > 0.25:
        rate = round(error_count / operational * 100, 1)
        findings.append({"rule": "high_error_rate", "message": f"{rate}% error rate ({error_count}/{operational} events)"})

    # Rate limiting (429)
    rate_limited = sum(1 for e in events if "429" in str(e.get("error", "")))
    if rate_limited > 0:
        findings.append({"rule": "rate_limiting", "message": f"{rate_limited} rate-limited requests (429)"})

    # Auth failures (401/403)
    auth_fails = sum(1 for e in events if any(c in str(e.get("error", "")) for c in ("401", "403")))
    if auth_fails > 0:
        findings.append({"rule": "auth_failure", "message": f"{auth_fails} auth failures (401/403)"})

    # Slow tools (avg > 10s)
    tool_dur: Dict[str, List[float]] = {}
    for e in events:
        if e.get("type") == "tool.call" and e.get("duration_ms"):
            tool_dur.setdefault(e.get("tool", "?"), []).append(e["duration_ms"])
    for tool, durs in tool_dur.items():
        avg = sum(durs) / len(durs)
        if avg > 10000:
            findings.append({"rule": "slow_tools", "message": f"{tool} avg {avg:.0f}ms ({len(durs)} calls)"})

    # Repeated tool failures (same tool errors 3+ times)
    tool_errors: Dict[str, int] = {}
    for e in events:
        if e.get("type") == "tool.error":
            tool_errors[e.get("tool", "?")] = tool_errors.get(e.get("tool", "?"), 0) + 1
    for tool, count in tool_errors.items():
        if count >= 3:
            findings.append({"rule": "repeated_tool_failure", "message": f"{tool} failed {count} times"})

    return findings


# -- Async writer -----------------------------------------------------------

def _write_to_sqlite(line: str) -> None:
    """Write a JSONL line to the SQLite audit_events table.  Best-effort."""
    if _db is None:
        return
    try:
        rec = json.loads(line)
        ts = rec.get("ts", time.time())
        iso = rec.get("iso", "")
        event_type = rec.get("type", "")
        session_id = rec.get("session", _session_id)

        # Extract indexed columns from the record
        tool = rec.get("tool")
        user_id = rec.get("user_id", _user_id)
        platform_val = rec.get("platform", _platform)
        detail = rec.get("detail")
        if isinstance(detail, dict):
            detail = json.dumps(detail, ensure_ascii=False, default=str)
        duration_ms = rec.get("duration_ms")
        error = rec.get("error")

        _db.insert_audit_event(
            session_id=session_id,
            ts=ts,
            iso=iso,
            event_type=event_type,
            tool=tool,
            user_id=user_id,
            platform=platform_val,
            detail=detail,
            duration_ms=duration_ms,
            error=error,
            payload_json=line,
            commit=False,
        )
    except Exception:
        pass  # SQLite failure must never crash the agent

_BATCH_SIZE = 10          # commit after this many events
_BATCH_INTERVAL = 2.0     # or after this many seconds, whichever first


def _writer_loop():
    """Background thread that drains the write queue to disk.

    Batches SQLite commits: flushes every ``_BATCH_SIZE`` events or every
    ``_BATCH_INTERVAL`` seconds, whichever comes first.  JSONL writes are
    immediate (OS-buffered via line-buffered file handle).
    """
    pending = 0
    last_commit = time.monotonic()
    while not _writer_stop.is_set():
        try:
            line = _write_queue.get(timeout=0.1)
            if _log_file and not _log_file.closed:
                _log_file.write(line + "\n")
            _write_to_sqlite(line)
            pending += 1
            if pending >= _BATCH_SIZE:
                _commit_sqlite_batch()
                pending = 0
                last_commit = time.monotonic()
        except queue.Empty:
            # Flush on idle if there are pending writes
            if pending > 0 and (time.monotonic() - last_commit) >= _BATCH_INTERVAL:
                _commit_sqlite_batch()
                pending = 0
                last_commit = time.monotonic()
            continue
        except Exception:
            pass


def _commit_sqlite_batch():
    """Commit pending SQLite inserts.  Best-effort."""
    try:
        if _db is not None:
            _db.commit_audit_batch()
    except Exception:
        pass


def _drain_queue():
    """Flush remaining items in the queue (called on session end)."""
    while not _write_queue.empty():
        try:
            line = _write_queue.get_nowait()
            if _log_file and not _log_file.closed:
                _log_file.write(line + "\n")
            _write_to_sqlite(line)
        except queue.Empty:
            break
        except Exception:
            pass
    _commit_sqlite_batch()


def _write(event_type: str, **data) -> None:
    """Enqueue a JSONL record for async writing.

    Source filtering is applied here so that specialised helpers
    (``log_tool_call``, ``log_api_request``) that bypass ``emit()``
    still respect the per-source config.
    """
    if _log_file is None:
        return
    if not _source_enabled(event_type):
        return
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": event_type,
        "source": _resolve_source(event_type),
        "session": _session_id,
        **data,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        _write_queue.put_nowait(line)
    except Exception:
        pass  # audit logging must never crash the agent
