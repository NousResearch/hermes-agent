"""Full MoA turn trace persistence (opt-in via config ``moa.save_traces``).

When enabled, every Mixture-of-Agents turn that actually runs the reference
fan-out (a cache MISS in ``MoAChatCompletions.create``) appends one JSON line
to ``<hermes_home>/moa-traces/<session_id>.jsonl``. The record is the TRUE
FULL turn — the exact messages array each reference model received (system
prompt + advisory view, not the truncated display preview), each reference's
full output, and the exact messages array the aggregator received (including
the injected reference-context guidance block) plus its output when available
— so a run can be audited end-to-end offline: what every model saw, what every
model said, and what it cost.

This is a side-channel trace. It is NOT the conversation ``messages`` table and
never enters message history or replay — MoA references are advisory side-calls
with their own system prompt, not conversation turns, so persisting them as
message rows would corrupt role alternation / replay. Traces live in their own
files, keyed by session id, and are safe to delete.

Cost model note: gated OFF by default. When off, the only overhead is the
``_traces_enabled()`` config read (cheap) — no file I/O, no serialization.
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
    """Return the trace directory if ``moa.save_traces`` is on, else None.

    Reads config lazily per call (config is cheap to load and this only runs on
    a cache-MISS MoA turn, i.e. once per user turn, not per tool iteration).
    ``moa.trace_dir`` overrides the default ``<hermes_home>/moa-traces/``.
    """
    try:
        from hermes_cli.config import load_config

        moa_cfg = (load_config() or {}).get("moa") or {}
    except Exception:  # pragma: no cover - defensive: never break a turn over tracing
        return None
    if not moa_cfg.get("save_traces"):
        return None
    override = moa_cfg.get("trace_dir")
    if override:
        base = Path(os.path.expandvars(os.path.expanduser(str(override))))
    else:
        base = get_hermes_home() / "moa-traces"
    return base


def _sanitize_session_id(session_id: Optional[str]) -> str:
    """Make a session id safe as a filename component."""
    if not session_id:
        return "unknown-session"
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(session_id))


def _slot_trace(acct: Any, label: str) -> dict[str, Any]:
    """Render one reference's _RefAccounting into a full trace dict.

    Includes the FULL input messages the reference received and its FULL
    output — not the truncated display preview.
    """
    usage = getattr(acct, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
            "cache_write_tokens": getattr(usage, "cache_write_tokens", 0),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
        }
    return {
        "label": label,
        "model": getattr(acct, "model", None),
        "provider": getattr(acct, "provider", None),
        "temperature": getattr(acct, "temperature", None),
        "input_messages": getattr(acct, "messages", None),
        "output": getattr(acct, "output", None),
        "usage": usage_dict,
        "cost_usd": getattr(acct, "cost_usd", None),
        "cost_status": getattr(acct, "cost_status", None),
        "cost_source": getattr(acct, "cost_source", None),
    }


def save_moa_turn(
    *,
    session_id: Optional[str],
    preset_name: str,
    reference_outputs: list[tuple[str, str, Any]],
    aggregator_label: str,
    aggregator_model: Optional[str],
    aggregator_provider: Optional[str],
    aggregator_temperature: Any,
    aggregator_input_messages: Any,
    aggregator_output: Optional[str],
    aggregator_streamed: bool,
) -> None:
    """Append one full MoA turn record to the session's trace JSONL, if enabled.

    Best-effort: any failure is logged at debug and swallowed — tracing must
    never break a live turn. Called once per turn on a reference cache MISS.

    ``aggregator_output`` is the aggregator's synthesized text when it was
    captured inline (non-streaming path — the eval / quiet-mode path). When the
    aggregator streamed to a live consumer, ``aggregator_streamed`` is True and
    the output is delivered as the turn's assistant message in the session
    store instead; the trace records the full aggregator INPUT either way.
    """
    base = _traces_enabled_and_dir()
    if base is None:
        return
    try:
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{_sanitize_session_id(session_id)}.jsonl"
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "preset": preset_name,
            "references": [
                _slot_trace(acct, label)
                for label, _text, acct in reference_outputs
            ],
            "aggregator": {
                "label": aggregator_label,
                "model": aggregator_model,
                "provider": aggregator_provider,
                "temperature": aggregator_temperature,
                "input_messages": aggregator_input_messages,
                "output": aggregator_output,
                "streamed": aggregator_streamed,
                # When streamed, the aggregator's acting output is persisted as
                # the turn's assistant message in state.db (see the session
                # store); it is not duplicated here.
                "output_location": "assistant_message_in_session_db"
                if aggregator_streamed else "inline",
            },
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - tracing must never break a turn
        logger.debug("MoA trace write failed (session=%s): %s", session_id, exc)
