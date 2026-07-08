"""Full Model Router turn trace persistence (opt-in via ``router.save_traces``).

When enabled, every routed turn that actually runs the classifier (a decision
cache MISS in ``RouterChatCompletions.create``) appends one JSON line to
``<hermes_home>/router-traces/<session_id>.jsonl``. The record captures the
routing decision end-to-end — the exact prompt the classifier received, its
raw output and parsed verdict (or the failure that made the turn fail open),
the chosen route, every fallback hop the acting call walked, and the acting
model's output when available — so routing quality can be audited offline:
which prompts went where, why, and what it cost.

This is a side-channel trace like ``agent/moa_trace.py``: it is NOT the
conversation ``messages`` table and never enters message history or replay.
Traces live in their own files, keyed by session id, and are safe to delete.

Cost model note: gated OFF by default. When off, the only overhead is the
``_traces_enabled_and_dir()`` config read (cheap) — no file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _traces_enabled_and_dir() -> Optional[Path]:
    """Return the trace directory if ``router.save_traces`` is on, else None.

    Reads config lazily per call (config is cheap to load and this only runs
    on a decision cache-MISS turn, i.e. once per user turn, not per tool
    iteration). ``router.trace_dir`` overrides the default
    ``<hermes_home>/router-traces/``.
    """
    try:
        from hermes_cli.config import load_config

        router_cfg = (load_config() or {}).get("router") or {}
    except Exception:  # pragma: no cover - defensive: never break a turn over tracing
        return None
    if not router_cfg.get("save_traces"):
        return None
    override = router_cfg.get("trace_dir")
    if override:
        base = Path(os.path.expandvars(os.path.expanduser(str(override))))
    else:
        base = get_hermes_home() / "router-traces"
    return base


def _sanitize_session_id(session_id: Optional[str]) -> str:
    """Make a session id safe as a filename component."""
    if not session_id:
        return "unknown-session"
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(session_id))


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
        "cache_write_tokens": getattr(usage, "cache_write_tokens", 0),
        "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
    }


def save_router_turn(
    *,
    session_id: Optional[str],
    preset_name: str,
    platform: Optional[str],
    classifier: dict[str, Any],
    route: dict[str, Any],
    fallback_events: list[dict[str, Any]],
    acting_label: str,
    acting_model: Optional[str],
    acting_provider: Optional[str],
    acting_output: Optional[str],
    acting_streamed: bool,
) -> None:
    """Append one full routed-turn record to the session's trace JSONL, if enabled.

    Best-effort: any failure is logged at debug and swallowed — tracing must
    never break a live turn. Called once per turn on a decision cache MISS.

    ``classifier`` carries the full classification side-call:
    ``{provider, model, input_messages, raw_output, verdict, failed, error,
    usage, cost_usd}`` — with ``usage`` as a CanonicalUsage (serialized here).
    ``route`` is ``{tier, provider, model, hint, short_circuited}``.
    ``fallback_events`` is a list of ``{from, to, error_class, error}`` hops.

    ``acting_output`` semantics mirror ``save_moa_turn``: captured inline on
    the non-streaming path, folded in after the fact from the caller's
    resolved assistant text on the streaming path, else the record points at
    the session store via ``output_location``.
    """
    base = _traces_enabled_and_dir()
    if base is None:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{_sanitize_session_id(session_id)}.jsonl"
        _have_output = bool(acting_output)
        if not acting_streamed:
            _output_location = "inline"
        elif _have_output:
            _output_location = "inline_from_stream"
        else:
            _output_location = "assistant_message_in_session_db"
        classifier_record = dict(classifier or {})
        if "usage" in classifier_record:
            classifier_record["usage"] = _usage_dict(classifier_record.get("usage"))
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "preset": preset_name,
            "platform": platform,
            "classifier": classifier_record,
            "route": dict(route or {}),
            "fallbacks": list(fallback_events or []),
            "acting": {
                "label": acting_label,
                "model": acting_model,
                "provider": acting_provider,
                "output": acting_output,
                "streamed": acting_streamed,
                # Where the acting output lives for this record — same
                # vocabulary as MoA traces so offline readers can share code.
                "output_location": _output_location,
            },
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - tracing must never break a turn
        logger.debug("Router trace write failed (session=%s): %s", session_id, exc)
