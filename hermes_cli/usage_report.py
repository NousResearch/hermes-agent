"""JSON usage report for ``hermes -z --usage-file`` and ``hermes chat -q --usage-file``.

Pipelines (Paperclip included) need a machine-readable spend record after a
non-interactive run. Oneshot and chat ``-q`` write the same shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def write_usage_file(
    path: Optional[str], result: dict, failure: Optional[str] = None
) -> None:
    """Best-effort JSON usage report.

    Written even on failure so callers can always account for spend. Never
    raises: a broken usage write must not mask the run's own outcome.
    """
    if not path:
        return
    try:
        report = {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "actual_cost_usd": result.get("actual_cost_usd"),
            "cost_status": result.get("cost_status"),
            "cost_source": result.get("cost_source"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
            "cache_write_tokens": result.get("cache_write_tokens"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "total_tokens": result.get("total_tokens"),
            "api_calls": result.get("api_calls"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "session_id": result.get("session_id"),
            "completed": result.get("completed"),
            "failed": bool(result.get("failed")) or failure is not None,
            "service_tier": result.get("service_tier"),
        }
        if failure is not None:
            report["failure"] = failure
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def result_from_agent(
    agent: Any,
    session: Optional[dict] = None,
    *,
    failed: bool = False,
    session_id: Optional[str] = None,
) -> dict:
    """Collect spend fields from a live agent, falling back to a session row."""
    session = session or {}
    sid = (
        session_id
        or (getattr(agent, "session_id", None) if agent is not None else None)
        or session.get("id")
    )
    return {
        "estimated_cost_usd": _first_number(
            getattr(agent, "session_estimated_cost_usd", None) if agent is not None else None,
            session.get("estimated_cost_usd"),
        ),
        "actual_cost_usd": _first_number(session.get("actual_cost_usd")),
        "cost_status": (
            getattr(agent, "session_cost_status", None) if agent is not None else None
        )
        or session.get("cost_status"),
        "cost_source": (
            getattr(agent, "session_cost_source", None) if agent is not None else None
        )
        or session.get("cost_source"),
        "input_tokens": _first_number(
            getattr(agent, "session_input_tokens", None) if agent is not None else None,
            session.get("input_tokens"),
        ),
        "output_tokens": _first_number(
            getattr(agent, "session_output_tokens", None) if agent is not None else None,
            session.get("output_tokens"),
        ),
        "cache_read_tokens": _first_number(
            getattr(agent, "session_cache_read_tokens", None) if agent is not None else None,
            session.get("cache_read_tokens"),
        ),
        "cache_write_tokens": _first_number(
            getattr(agent, "session_cache_write_tokens", None) if agent is not None else None,
            session.get("cache_write_tokens"),
        ),
        "reasoning_tokens": _first_number(
            getattr(agent, "session_reasoning_tokens", None) if agent is not None else None,
            session.get("reasoning_tokens"),
        ),
        "total_tokens": _first_number(
            getattr(agent, "session_total_tokens", None) if agent is not None else None,
            session.get("total_tokens"),
        ),
        "api_calls": _first_number(
            getattr(agent, "api_call_count", None) if agent is not None else None,
            session.get("api_call_count"),
        ),
        "model": (getattr(agent, "model", None) if agent is not None else None)
        or session.get("model"),
        "provider": (getattr(agent, "provider", None) if agent is not None else None)
        or session.get("provider"),
        "session_id": sid,
        "completed": True,
        "failed": failed,
        "service_tier": (
            ((getattr(agent, "request_overrides", None) or {}).get("extra_body") or {}).get(
                "service_tier"
            )
            if agent is not None
            else None
        ),
    }


def result_from_cli(cli: Any, *, failed: bool = False) -> dict:
    agent = getattr(cli, "agent", None)
    session: dict = {}
    session_id = getattr(cli, "session_id", None) or getattr(agent, "session_id", None)
    db = getattr(cli, "_session_db", None)
    if db is not None and session_id:
        try:
            session = db.get_session(session_id) or {}
        except Exception:
            session = {}
    return result_from_agent(agent, session, failed=failed, session_id=session_id)


def write_usage_file_from_cli(cli: Any, *, failure: Optional[str] = None) -> None:
    """Write ``cli.usage_file`` if the flag was set. Never raises."""
    try:
        path = getattr(cli, "usage_file", None)
        if not path:
            return
        write_usage_file(
            path,
            result_from_cli(cli, failed=failure is not None),
            failure=failure,
        )
    except Exception:
        pass
