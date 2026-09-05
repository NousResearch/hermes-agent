"""Full MoA turn trace persistence (opt-in via config ``moa.save_traces``).

Every MoA turn that runs the reference fan-out (a cache MISS in
``MoAChatCompletions.create``) appends one JSON line to
``<hermes_home>/moa-traces/<session_id>.jsonl``: what every model saw, said, and
cost. Side-channel only: never enters the ``messages`` table, history or replay
(references are advisory side-calls whose rows would corrupt role alternation).
Off by default; when off the only overhead is the config read.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from hermes_cli.config import artifact_file_mode, secure_artifact_dir
from utils import open_private_append

logger = logging.getLogger(__name__)


def _traces_enabled_and_dir() -> Optional[Path]:
    """Trace directory if ``moa.save_traces`` is on, else None. Reads config per
    call (once per cache-MISS turn); ``moa.trace_dir`` overrides the default."""
    try:
        from hermes_cli.config import load_config
        moa_cfg = (load_config() or {}).get("moa") or {}
    except Exception:  # pragma: no cover - never break a turn over tracing
        return None
    if not moa_cfg.get("save_traces"):
        return None
    override = moa_cfg.get("trace_dir")
    if override:
        return Path(os.path.expandvars(os.path.expanduser(str(override))))
    return get_hermes_home() / "moa-traces"


def _sanitize_session_id(session_id: Optional[str]) -> str:
    if not session_id:
        return "unknown-session"
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(session_id))


_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
_ACCT_FIELDS = ("model", "provider", "temperature")
_COST_FIELDS = ("cost_usd", "cost_status", "cost_source")


def _slot_trace(acct: Any, label: str) -> dict[str, Any]:
    """One reference's _RefAccounting as a full trace dict, including the FULL
    input messages and output (not the truncated display preview)."""
    usage = getattr(acct, "usage", None)
    return {
        "label": label, **{f: getattr(acct, f, None) for f in _ACCT_FIELDS},
        "input_messages": getattr(acct, "messages", None), "output": getattr(acct, "output", None),
        "usage": {f: getattr(usage, f, 0) for f in _USAGE_FIELDS} if usage is not None else {},
        **{f: getattr(acct, f, None) for f in _COST_FIELDS},
    }


def slot_metrics(acct: Any, label: str, output: Any = None) -> dict[str, Any]:
    """``_slot_trace`` minus ``input_messages`` (the bulk of a record) for
    observability hooks. ``output`` comes from the caller because the
    privacy-redacted advisor text lives alongside the accounting, not on it."""
    trace = _slot_trace(acct, label)
    trace.pop("input_messages", None)
    if output is not None:
        trace["output"] = output
    return trace


def save_moa_turn(
    *, session_id: Optional[str], preset_name: str, reference_outputs: list[tuple[str, str, Any]],
    aggregator_label: str, aggregator_model: Optional[str], aggregator_provider: Optional[str],
    aggregator_temperature: Any, aggregator_input_messages: Any, aggregator_output: Optional[str],
    aggregator_streamed: bool,
) -> None:
    """Append one full MoA turn record to the session's trace JSONL, if enabled.

    Best-effort: failures are logged at debug and swallowed. ``aggregator_output``
    is captured inline on the non-streaming path and after the fact from the
    resolved assistant text on the streaming path; if unavailable it is None and
    ``output_location`` points at the session store.
    """
    base = _traces_enabled_and_dir()
    if base is None:
        return
    try:
        internal = base == get_hermes_home() / "moa-traces"
        secure_artifact_dir(base, tighten_existing=internal)
        path = base / f"{_sanitize_session_id(session_id)}.jsonl"
        if not aggregator_streamed:
            output_location = "inline"
        elif aggregator_output:
            output_location = "inline_from_stream"
        else:
            output_location = "assistant_message_in_session_db"
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "preset": preset_name,
            "references": [_slot_trace(acct, label) for label, _text, acct in reference_outputs],
            "aggregator": {
                "label": aggregator_label, "model": aggregator_model,
                "provider": aggregator_provider, "temperature": aggregator_temperature,
                "input_messages": aggregator_input_messages, "output": aggregator_output,
                "streamed": aggregator_streamed, "output_location": output_location,
            },
        }
        with open_private_append(path, mode=artifact_file_mode(), tighten_existing=internal) as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - tracing must never break a turn
        logger.debug("MoA trace write failed (session=%s): %s", session_id, exc)
