"""token-consumption-tracker plugin — track token usage across models and providers.

Hooks into ``post_api_request`` to record per-call token consumption with
estimated cost, and stores the data in-memory (per-session counters) plus a
persistent JSONL log at ``$HERMES_HOME/token-tracker/usage.jsonl``.

Provides two interfaces:
- ``/token`` slash command — in-session query of current session usage.
- ``hermes tokens`` CLI command — cross-session historic usage reports.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATE_DIR_NAME = "token-tracker"
_USAGE_FILE_NAME = "usage.jsonl"
_SESSION_FILE_NAME = "session.jsonl"

# ---------------------------------------------------------------------------
# Thread-safe in-memory store
# ---------------------------------------------------------------------------

_lock = threading.Lock()
# Per-session usage records (list of dicts) — keyed by session_id.
_session_records: Dict[str, List[Dict[str, Any]]] = {}
# Running totals per session.
_session_totals: Dict[str, Dict[str, Any]] = {}


def _get_hermes_home() -> Path:
    """Resolve HERMES_HOME (fallback: ~/.hermes)."""
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


def _get_state_dir() -> Path:
    """State directory for token-tracker data."""
    return _get_hermes_home() / _STATE_DIR_NAME


def _get_usage_file() -> Path:
    """Global usage history file (append-only JSONL)."""
    return _get_state_dir() / _USAGE_FILE_NAME


def _get_session_file() -> Path:
    """Per-session usage file (JSONL)."""
    return _get_state_dir() / _SESSION_FILE_NAME


def _ensure_state_dir() -> None:
    """Create the state directory if it doesn't exist."""
    _get_state_dir().mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_key(session_id: str) -> str:
    return session_id or "default"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def _estimate_cost_for_entry(
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> tuple[Optional[float], str, str]:
    """Estimate USD cost for a usage record.

    Returns (amount_usd, status, source).
    """
    try:
        from agent.usage_pricing import (
            CanonicalUsage,
            estimate_usage_cost,
        )

        usage = CanonicalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        result = estimate_usage_cost(
            model,
            usage,
            provider=provider,
        )
        if result.amount_usd is not None:
            return float(result.amount_usd), result.status, result.source
        return None, result.status, result.source
    except Exception as exc:
        logger.debug("token-tracker: cost estimation failed: %s", exc)
        return None, "unknown", "none"


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _record_usage(record: Dict[str, Any]) -> None:
    """Store a usage record in-memory and append to the JSONL file."""
    session_id = record.get("session_id", "")
    key = _session_key(session_id)

    with _lock:
        _session_records.setdefault(key, []).append(record)
        totals = _session_totals.setdefault(
            key,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        totals["input_tokens"] += record.get("input_tokens", 0)
        totals["output_tokens"] += record.get("output_tokens", 0)
        totals["cache_read_tokens"] += record.get("cache_read_tokens", 0)
        totals["cache_write_tokens"] += record.get("cache_write_tokens", 0)
        totals["reasoning_tokens"] += record.get("reasoning_tokens", 0)
        totals["total_tokens"] += record.get("total_tokens", 0) or (
            record.get("input_tokens", 0)
            + record.get("output_tokens", 0)
            + record.get("cache_read_tokens", 0)
            + record.get("cache_write_tokens", 0)
        )
        totals["api_calls"] += 1
        cost = record.get("estimated_cost_usd")
        if cost is not None:
            totals["estimated_cost_usd"] += cost

    # Append to persistent JSONL (best-effort, never raise).
    try:
        _ensure_state_dir()
        with open(_get_usage_file(), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.debug("token-tracker: failed to persist record: %s", exc)


def _get_session_summary(session_id: str) -> Dict[str, Any]:
    """Return summary for a given session."""
    key = _session_key(session_id)
    with _lock:
        totals = _session_totals.get(key)
        if not totals:
            return {
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
        return dict(totals)


def _get_all_sessions() -> Dict[str, Dict[str, Any]]:
    """Return summaries for all tracked sessions."""
    with _lock:
        return {k: dict(v) for k, v in _session_totals.items()}


def _get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Read the last N records from the global usage file."""
    try:
        usage_file = _get_usage_file()
        if not usage_file.exists():
            return []
        with open(usage_file) as f:
            lines = f.readlines()
        records = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
    except Exception as exc:
        logger.debug("token-tracker: failed to read history: %s", exc)
        return []


def _reset_session(session_id: str) -> None:
    """Reset in-memory counters for a session."""
    key = _session_key(session_id)
    with _lock:
        _session_records.pop(key, None)
        _session_totals.pop(key, None)


def _reset_all() -> None:
    """Reset all in-memory counters."""
    with _lock:
        _session_records.clear()
        _session_totals.clear()
    try:
        usage_file = _get_usage_file()
        if usage_file.exists():
            usage_file.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_post_api_request(**kwargs: Any) -> None:
    """Record token consumption from each API request."""
    usage = kwargs.get("usage")
    if not usage:
        return

    model = kwargs.get("model") or ""
    provider = kwargs.get("provider") or ""

    # Extract token counts from usage dict (normalized via CanonicalUsage).
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read_tokens = usage.get("cache_read_tokens", 0) or 0
    cache_write_tokens = usage.get("cache_write_tokens", 0) or 0
    reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
    prompt_tokens = usage.get("prompt_tokens", 0) or (
        input_tokens + cache_read_tokens + cache_write_tokens
    )
    total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + output_tokens)

    # Estimate cost.
    cost, cost_status, cost_source = _estimate_cost_for_entry(
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    record = {
        "timestamp": _now_iso(),
        "session_id": kwargs.get("session_id", ""),
        "turn_id": kwargs.get("turn_id", ""),
        "api_request_id": kwargs.get("api_request_id", ""),
        "model": model,
        "provider": provider,
        "base_url": kwargs.get("base_url", ""),
        "api_mode": kwargs.get("api_mode", ""),
        "platform": kwargs.get("platform", ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "api_duration_ms": int(kwargs.get("api_duration", 0) * 1000),
        "finish_reason": kwargs.get("finish_reason", ""),
        "response_model": kwargs.get("response_model", ""),
        "estimated_cost_usd": cost,
        "cost_status": cost_status,
        "cost_source": cost_source,
    }

    _record_usage(record)


def _on_session_start(**kwargs: Any) -> None:
    """Initialize tracking for a new session."""
    session_id = kwargs.get("session_id", "")
    if session_id:
        _reset_session(session_id)


def _on_session_end(**kwargs: Any) -> None:
    """Flush any remaining data on session end (already persisted per-call)."""
    pass


# ---------------------------------------------------------------------------
# Slash command: /token
# ---------------------------------------------------------------------------


def _handle_slash(raw_args: str) -> str:
    """Handle the /token slash command."""
    argv = raw_args.strip().split()
    sub = argv[0].lower() if argv else "summary"

    if sub in ("help", "-h", "--help"):
        return _SLASH_HELP_TEXT

    if sub == "summary":
        return _summary_output()

    if sub == "history":
        limit = 50
        if len(argv) > 1:
            try:
                limit = max(1, min(int(argv[1]), 500))
            except ValueError:
                pass
        records = _get_history(limit)
        if not records:
            return "No token usage history available."
        lines = [
            _format_history_header(),
        ]
        for r in reversed(records):
            lines.append(_format_history_row(r))
        return "\n".join(lines)

    if sub == "reset":
        session_id = argv[1] if len(argv) > 1 else ""
        if session_id:
            _reset_session(session_id)
            return f"Reset token counters for session '{session_id}'."
        _reset_all()
        return "Reset all token counters and cleared persistent log."

    return (
        f"Unknown subcommand: {sub}\n\n"
        f"Usage: /token [summary|history|reset|help]"
    )


_SLASH_HELP_TEXT = """\
/token — Token consumption tracker

Subcommands:
  summary               Show current session token usage and estimated cost
  history [N]           Show last N API call records (default: 50)
  reset [session_id]    Reset counters (optionally for a specific session)
  help                  Show this help

All operations are best-effort — cost estimates depend on known model pricing.
"""


def _summary_output() -> str:
    """Build summary text for the /token slash command."""
    with _lock:
        all_sessions = {k: dict(v) for k, v in _session_totals.items()}

    if not all_sessions:
        return (
            "No token tracking data recorded yet. "
            "Token consumption is captured during active sessions."
        )

    lines = [
        "📊 Token Consumption Summary",
        "═" * 40,
    ]

    grand_total = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    for session_key, totals in sorted(all_sessions.items()):
        lines.append(f"\n  Session: {_fmt_session_label(session_key)}")
        lines.append(f"    API calls      : {totals['api_calls']}")
        lines.append(f"    Input tokens   : {_fmt_number(totals['input_tokens'])}")
        lines.append(f"    Output tokens  : {_fmt_number(totals['output_tokens'])}")
        lines.append(f"    Cache read     : {_fmt_number(totals['cache_read_tokens'])}")
        lines.append(f"    Cache write    : {_fmt_number(totals['cache_write_tokens'])}")
        lines.append(f"    Reasoning      : {_fmt_number(totals['reasoning_tokens'])}")
        lines.append(f"    Total tokens   : {_fmt_number(totals['total_tokens'])}")
        cost = totals.get("estimated_cost_usd", 0.0)
        cost_str = _fmt_cost(cost)
        lines.append(f"    Est. cost      : {cost_str}")

        grand_total["api_calls"] += totals["api_calls"]
        grand_total["input_tokens"] += totals["input_tokens"]
        grand_total["output_tokens"] += totals["output_tokens"]
        grand_total["total_tokens"] += totals["total_tokens"]
        grand_total["estimated_cost_usd"] += cost

    lines.append("\n" + "─" * 40)
    lines.append(f"  GRAND TOTAL")
    lines.append(f"    API calls      : {grand_total['api_calls']}")
    lines.append(f"    Total tokens   : {_fmt_number(grand_total['total_tokens'])}")
    lines.append(f"    Est. cost      : {_fmt_cost(grand_total['estimated_cost_usd'])}")

    # Show active session if available
    active = _get_current_session_id()
    if active and active in all_sessions:
        lines.append(f"\n  ℹ️ Active session: {active[:12]}...")

    return "\n".join(lines)


def _format_history_header() -> str:
    return (
        f"{'Time':<22} {'Model':<30} {'Input':>8} {'Output':>8} "
        f"{'Total':>8} {'Cost':<12} {'Provider':<16}"
    )


def _format_history_row(r: Dict[str, Any]) -> str:
    ts = r.get("timestamp", "")
    if len(ts) > 19:
        ts = ts[:19]  # truncate subseconds
    model = (r.get("model") or "")[:29]
    inp = _fmt_number(r.get("input_tokens", 0))
    out = _fmt_number(r.get("output_tokens", 0))
    total = _fmt_number(r.get("total_tokens", 0))
    cost = _fmt_cost(r.get("estimated_cost_usd", 0.0) or 0.0)
    prov = (r.get("provider") or "")[:15]
    return f"{ts:<22} {model:<30} {inp:>8} {out:>8} {total:>8} {cost:<12} {prov:<16}"


def _fmt_session_label(key: str) -> str:
    if len(key) > 16:
        return f"{key[:12]}..."
    return key


def _fmt_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_cost(cost: float) -> str:
    if cost is None or cost == 0.0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.6f}"
    if cost < 1.0:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _get_current_session_id() -> str:
    """Best-effort: get the current session ID from the agent."""
    try:
        import sys

        for frame in sys._current_frames().values():
            for var_name in ("agent", "self"):
                f_locals = frame.f_locals
                obj = f_locals.get(var_name)
                if obj is not None:
                    sid = getattr(obj, "session_id", None)
                    if sid:
                        return sid
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# CLI command: hermes tokens
# ---------------------------------------------------------------------------


def _setup_cli_parser(subparser) -> None:
    """Set up the `hermes tokens` argument parser."""
    subparser.add_argument(
        "action",
        nargs="?",
        default="summary",
        choices=["summary", "history", "reset", "status"],
        help="Action to perform (default: summary)",
    )
    subparser.add_argument(
        "arg",
        nargs="?",
        default="",
        help="Optional argument (e.g. session_id for reset, limit for history)",
    )


_CLI_HELP = """\
Query token consumption tracked across sessions.

Actions:
  summary          Show per-session and grand total token usage + estimated cost
  history [N]      Show last N API call records with tokens and cost
  reset [id]       Reset all counters (or for a specific session id)
  status           Alias for summary

Cost estimation uses the model pricing table in agent.usage_pricing.
Unknown models show as $0.00 with status 'unknown'.
"""


def _handle_cli(args) -> None:
    """Handler for `hermes tokens` CLI command."""
    action = args.action if hasattr(args, "action") else "summary"
    extra = args.arg if hasattr(args, "arg") else ""

    if action in ("summary", "status"):
        print(_summary_output())
        return

    if action == "history":
        limit = 50
        if extra:
            try:
                limit = max(1, min(int(extra), 500))
            except ValueError:
                pass
        records = _get_history(limit)
        if not records:
            print("No token usage history available.")
            return
        print(_format_history_header())
        for r in reversed(records):
            print(_format_history_row(r))
        return

    if action == "reset":
        if extra:
            _reset_session(extra)
            print(f"Reset token counters for session '{extra}'.")
        else:
            _reset_all()
            print("Reset all token counters and cleared persistent log.")
        return

    print(f"Unknown action: {action}")
    print(_CLI_HELP)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)

    ctx.register_command(
        "token",
        handler=_handle_slash,
        description="Query token consumption for current session.",
        args_hint="[summary|history|reset|help]",
    )

    ctx.register_cli_command(
        name="tokens",
        help="Query tracked token consumption across sessions.",
        setup_fn=_setup_cli_parser,
        handler_fn=_handle_cli,
        description="Query token usage and estimated costs from the token-consumption-tracker plugin.",
    )
