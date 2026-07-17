"""Edge-mode fault loop damper — blocks repeat tool calls that already failed.

Uses a stable per-call signature (tool name + canonical JSON args) plus optional
``[sig:…]`` markers embedded in the scratchpad **Faults & Blockers** section so
compaction-resistant state survives across flushes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_SIG_MARK = re.compile(r"\[sig:([^\]\s]+)\]")


def edge_fault_damper_enabled(agent) -> bool:
    """True when edge mode is on and the consecutive-failure cap is > 0.

    ``edge_max_consecutive_tool_failures: 0`` (default) fully disables the
    damper: no signature recording, no repeat blocking, and no interrupt.
    """
    if not getattr(agent, "edge_mode", False):
        return False
    try:
        return int(getattr(agent, "_edge_max_consecutive_tool_failures", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def edge_tool_signature(tool_name: str, function_args: Optional[Dict[str, Any]]) -> str:
    """Stable signature for matching duplicate failing calls."""
    name = (tool_name or "").strip().lower()
    try:
        body = json.dumps(function_args or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        body = str(function_args or {})
    digest = hashlib.sha256(f"{name}\0{body}".encode("utf-8")).hexdigest()[:24]
    return f"{name}|{digest}"


def parse_sig_markers_from_text(text: str) -> Set[str]:
    """Collect ``tool|digest`` tokens from ``[sig:…]`` markers in markdown."""
    found: Set[str] = set()
    for m in _SIG_MARK.finditer(text or ""):
        token = (m.group(1) or "").strip()
        if token and "|" in token:
            found.add(token.lower())
    return found


def resync_edge_failed_signatures(agent) -> None:
    """Repopulate ``_edge_failed_signatures`` from scratchpad markers."""
    if not getattr(agent, "edge_mode", False):
        return
    scratch = getattr(agent, "_edge_scratchpad", "") or ""
    merged: Set[str] = set()
    for x in (getattr(agent, "_edge_failed_signatures", None) or set()):
        if isinstance(x, str) and x.strip():
            merged.add(x.strip().lower())
    merged |= {s.lower() for s in parse_sig_markers_from_text(scratch)}
    agent._edge_failed_signatures = merged


def edge_precheck_tool_repeat(
    agent,
    tool_name: str,
    function_args: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return a synthetic error string if this call should be blocked."""
    if not edge_fault_damper_enabled(agent):
        return None
    sig = edge_tool_signature(tool_name, function_args)
    scratch = getattr(agent, "_edge_scratchpad", "") or ""
    blocked = set(getattr(agent, "_edge_failed_signatures", None) or set())
    blocked |= parse_sig_markers_from_text(scratch)
    blocked_norm = {b.lower() for b in blocked if isinstance(b, str)}
    if sig.lower() not in blocked_norm:
        return None
    return (
        "[Hermes edge fault damper] This tool call matches a previously failed "
        f"action (`{tool_name}` with the same arguments digest). It was blocked "
        "to prevent a deterministic repeat loop after context compaction. Alter "
        "the parameters, fix the environment (permissions, missing commands, "
        "paths), or explicitly remove the matching "
        f"``[sig:{sig}]`` line from Faults & Blockers in your scratchpad before retrying."
    )


def edge_record_tool_result_for_damper(
    agent,
    tool_name: str,
    function_args: Optional[Dict[str, Any]],
    result: str,
) -> None:
    """After a tool runs, record failures so identical calls can be blocked."""
    if not edge_fault_damper_enabled(agent):
        return
    try:
        from agent.display import _detect_tool_failure
    except Exception:
        return
    try:
        failed, _reason = _detect_tool_failure(tool_name, result or "")
    except Exception:
        return
    if not failed:
        agent._edge_consecutive_tool_failures = 0
        return

    try:
        cur = int(getattr(agent, "_edge_consecutive_tool_failures", 0) or 0) + 1
        agent._edge_consecutive_tool_failures = cur
        cap = int(getattr(agent, "_edge_max_consecutive_tool_failures", 0) or 0)
        if cur >= cap:
            agent._interrupt_requested = True
            logger.warning(
                "edge fault: consecutive tool failures %s >= cap %s — requesting interrupt",
                cur,
                cap,
            )
    except Exception:
        pass

    sig = edge_tool_signature(tool_name, function_args)
    if not hasattr(agent, "_edge_failed_signatures"):
        agent._edge_failed_signatures = set()
    if sig.lower() in {x.lower() for x in agent._edge_failed_signatures}:
        return
    agent._edge_failed_signatures.add(sig.lower())

    try:
        from agent.edge_working_memory import append_auto_fault_blocker

        preview = result if isinstance(result, str) else str(result)
        agent._edge_scratchpad = append_auto_fault_blocker(
            getattr(agent, "_edge_scratchpad", "") or "",
            tool_name,
            sig,
            preview,
        )
        try:
            from agent.edge_working_memory import persist_edge_scratchpad_now

            persist_edge_scratchpad_now(agent)
        except Exception:
            pass
    except Exception as exc:
        logger.debug("edge fault scratchpad append failed: %s", exc)
