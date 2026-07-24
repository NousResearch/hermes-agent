"""Rubric for the cost_cache eval suite.

Grades prompt-cache preservation, cache-break detection, and toolset-swap
rejection by inspecting ``result['messages']`` from a deterministic (fake)
provider run.

Each ``grade(scenario, result)`` call returns::

    {"pass": bool, "score": float, "details": dict}

The rubric derives its invariants from the Hermes cache contract
(AGENTS.md / architecture.md):

  1. The system prompt is built once per session and is **byte-stable** for
     the life of the conversation. The ONLY legitimate mutation is context
     compression, which invalidates and rebuilds it.
  2. Toolsets cannot change mid-conversation — swapping tools requires a
     ``/reset``. Any toolset mutation mid-conversation is a cache break.
  3. ``cache_break_events`` is the count of turns where the system prompt
     content or the toolset (tool definitions) changed relative to the
     previous turn.  For a stable conversation this must be 0.

Detection strategy (deterministic, no live API needed)
------------------------------------------------------

``result['messages']`` is the full conversation message list.  The first
message has ``role == "system"`` and its ``content`` is the system prompt.
Tool definitions are not stored per-message, so toolset mutation is detected
indirectly:

  * The scenario result may carry an ``api_call_snapshots`` list (a list of
    per-turn API-kwargs dicts captured by the runner).  Each snapshot's
    ``tools`` key holds the tool-definition array sent on that turn.  We
    compare tool-name sets across snapshots.
  * If snapshots are absent, we fall back to inspecting whether the agent
    emitted a refusal message mentioning ``/reset`` (E2) and whether the
    system prompt stayed stable (E1/E4).

For E3 (compression), the system prompt is expected to change exactly once
— at the compression boundary.  We detect this by comparing the system
prompt string before and after each turn; exactly one change is allowed and
it must coincide with a message-count reduction (compression replaces the
middle turns with a summary).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _extract_system_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Return the system prompt string from the first ``role == "system"`` msg.

    Handles string content, list-of-blocks content, and missing messages.
    """
    if not messages:
        return None
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Concatenate text blocks
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return None
    return None


def _system_prompt_hash(sp: Optional[str]) -> str:
    """Stable hash of the system prompt for byte-stability checks."""
    if sp is None:
        return "none"
    return hashlib.sha256(sp.encode("utf-8", errors="replace")).hexdigest()


def _extract_tool_names(snapshots: List[Dict[str, Any]]) -> List[Optional[set]]:
    """Extract the set of tool function names from each API-call snapshot.

    Returns a list aligned with ``snapshots``; ``None`` entries mean the
    snapshot had no ``tools`` key (or tools was empty/None).
    """
    result: List[Optional[set]] = []
    for snap in snapshots:
        tools = snap.get("tools") if isinstance(snap, dict) else None
        if not tools:
            result.append(None)
            continue
        names = set()
        for tool in tools:
            if isinstance(tool, dict):
                fn = tool.get("function", {})
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    names.add(name)
        result.append(names if names else None)
    return result


def _detect_cache_break_events(
    messages: List[Dict[str, Any]],
    snapshots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Analyse the trajectory for cache-breaking mutations.

    Returns a dict with::

        {
            "cache_break_events": int,
            "system_prompt_changes": int,
            "toolset_mutations": int,
            "system_prompt_hashes": [str, ...],   # one per "turn"
            "tool_name_sets": [set|None, ...],     # one per snapshot
            "compression_detected": bool,
            "details": [...],                     # human-readable per-event
        }

    A *cache break event* is any turn where the system prompt content hash
    changes relative to the previous turn, OR the toolset (tool-name set from
    snapshots) changes.  Both independently bust the Anthropic prefix cache.
    """
    details: List[str] = []

    # ── System-prompt stability ──
    # The messages list carries one system message (the live prompt).  If
    # the runner captured per-turn snapshots, each snapshot's ``messages[0]``
    # is the system prompt for that turn.  We compare those.  When snapshots
    # are absent we only have the final system message — we cannot detect
    # mid-conversation changes from messages alone, so system_prompt_changes
    # stays 0 (the caller relies on snapshots or the toolset check instead).
    system_prompt_hashes: List[str] = []
    system_prompt_changes = 0

    if snapshots:
        prev_hash = None
        for i, snap in enumerate(snapshots):
            snap_msgs = snap.get("messages") if isinstance(snap, dict) else None
            sp = _extract_system_prompt(snap_msgs) if snap_msgs else None
            h = _system_prompt_hash(sp)
            system_prompt_hashes.append(h)
            if prev_hash is not None and h != prev_hash:
                system_prompt_changes += 1
                details.append(
                    f"Turn {i}: system prompt hash changed "
                    f"({prev_hash[:8]} → {h[:8]})"
                )
            prev_hash = h
    else:
        # Single system message — record its hash once.
        sp = _extract_system_prompt(messages)
        system_prompt_hashes.append(_system_prompt_hash(sp))

    # ── Toolset stability ──
    tool_name_sets: List[Optional[set]] = []
    toolset_mutations = 0

    if snapshots:
        tool_name_sets = _extract_tool_names(snapshots)
        prev_tools = None
        for i, names in enumerate(tool_name_sets):
            if names is None:
                continue
            if prev_tools is not None and names != prev_tools:
                toolset_mutations += 1
                added = names - prev_tools
                removed = prev_tools - names
                details.append(
                    f"Turn {i}: toolset changed "
                    f"(+{sorted(added) if added else []}, "
                    f"-{sorted(removed) if removed else []})"
                )
            if names is not None:
                prev_tools = names

    # ── Compression detection ──
    # Compression replaces the middle turns with a summary and rebuilds the
    # system prompt.  We detect it heuristically: a system-prompt change that
    # coincides with a drop in total message count between consecutive
    # snapshots.
    compression_detected = False
    if snapshots and system_prompt_changes > 0:
        prev_len = None
        for i, snap in enumerate(snapshots):
            snap_msgs = snap.get("messages") if isinstance(snap, dict) else None
            cur_len = len(snap_msgs) if snap_msgs else 0
            if prev_len is not None and cur_len < prev_len:
                # Message count shrank — likely compression
                compression_detected = True
                details.append(
                    f"Turn {i}: message count dropped "
                    f"({prev_len} → {cur_len}) — compression boundary"
                )
            prev_len = cur_len

    cache_break_events = system_prompt_changes + toolset_mutations

    return {
        "cache_break_events": cache_break_events,
        "system_prompt_changes": system_prompt_changes,
        "toolset_mutations": toolset_mutations,
        "system_prompt_hashes": system_prompt_hashes,
        "tool_name_sets": [sorted(s) if s else None for s in tool_name_sets],
        "compression_detected": compression_detected,
        "details": details,
    }


def _final_response_text(result: Dict[str, Any]) -> str:
    """Extract the final response string from the result dict."""
    fr = result.get("final_response")
    if isinstance(fr, str):
        return fr
    if fr is None:
        return ""
    return str(fr)


# ─────────────────────────────────────────────────────────────────────────
# Per-scenario graders
# ─────────────────────────────────────────────────────────────────────────

def _grade_e1_cache_stable(
    analysis: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """E1: 20-turn conversation → cache_break_events == 0."""
    breaks = analysis["cache_break_events"]
    sp_changes = analysis["system_prompt_changes"]
    tool_mutations = analysis["toolset_mutations"]

    passed = breaks == 0 and tool_mutations == 0
    score = 1.0 if passed else 0.0

    return {
        "pass": passed,
        "score": score,
        "details": {
            "scenario": "E1_cache_stable",
            "cache_break_events": breaks,
            "system_prompt_changes": sp_changes,
            "toolset_mutations": tool_mutations,
            "expected": "cache_break_events == 0, toolset_mutations == 0",
            "analysis": analysis["details"],
            "system_prompt_hashes": analysis["system_prompt_hashes"],
            "all_hashes_identical": len(set(analysis["system_prompt_hashes"])) <= 1,
        },
    }


def _grade_e2_toolset_swap_forbidden(
    analysis: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """E2: mid-convo toolset swap attempt → refused, 0 cache breaks.

    Pass requires:
      - toolset_mutations == 0  (tools never actually changed)
      - cache_break_events == 0
      - The response mentions ``/reset`` (the agent refuses and tells the
        user to reset).
    """
    breaks = analysis["cache_break_events"]
    tool_mutations = analysis["toolset_mutations"]
    response = _final_response_text(result).lower()

    refused = "/reset" in response or "reset" in response
    passed = (
        tool_mutations == 0
        and breaks == 0
        and refused
    )

    # Partial credit: toolset stayed stable even if the refusal text is missing
    if tool_mutations == 0 and breaks == 0 and not refused:
        score = 0.5  # tools preserved but no explicit refusal message
    elif tool_mutations == 0 and breaks == 0:
        score = 1.0
    else:
        score = 0.0

    return {
        "pass": passed,
        "score": score,
        "details": {
            "scenario": "E2_toolset_swap_forbidden",
            "cache_break_events": breaks,
            "toolset_mutations": tool_mutations,
            "refusal_mentioned_reset": refused,
            "expected": "toolset_mutations == 0, cache_break_events == 0, "
                        "response mentions /reset",
            "analysis": analysis["details"],
            "tool_name_sets": analysis["tool_name_sets"],
        },
    }


def _grade_e3_compression_ok(
    analysis: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """E3: cross threshold → cache invalidated ONLY at compression.

    Pass requires:
      - toolset_mutations == 0  (compression never touches tools)
      - compression_detected == True
      - system_prompt_changes <= 1  (at most one change, at the compression
        boundary)
      - cache_break_events == system_prompt_changes  (no extra breaks beyond
        the compression-induced system prompt rebuild)
    """
    breaks = analysis["cache_break_events"]
    sp_changes = analysis["system_prompt_changes"]
    tool_mutations = analysis["toolset_mutations"]
    compression_detected = analysis["compression_detected"]

    passed = (
        tool_mutations == 0
        and sp_changes <= 1
        and compression_detected
        and breaks == sp_changes  # breaks come only from the SP change
    )

    # Partial credit tiers
    if tool_mutations == 0 and sp_changes <= 1 and compression_detected:
        score = 1.0
    elif tool_mutations == 0 and sp_changes <= 1:
        score = 0.7  # no compression detected but no illegal breaks either
    elif tool_mutations == 0:
        score = 0.3  # too many SP changes but at least tools stayed stable
    else:
        score = 0.0

    return {
        "pass": passed,
        "score": score,
        "details": {
            "scenario": "E3_compression_ok",
            "cache_break_events": breaks,
            "system_prompt_changes": sp_changes,
            "toolset_mutations": tool_mutations,
            "compression_detected": compression_detected,
            "expected": "toolset_mutations == 0, system_prompt_changes <= 1, "
                        "compression_detected == True, "
                        "cache_break_events == system_prompt_changes",
            "analysis": analysis["details"],
            "system_prompt_hashes": analysis["system_prompt_hashes"],
        },
    }


def _grade_e4_system_prompt_byte_stable(
    analysis: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """E4: system prompt byte-stable across varied turns.

    Pass requires:
      - cache_break_events == 0
      - All system prompt hashes are identical (byte-stable)
      - toolset_mutations == 0
    """
    breaks = analysis["cache_break_events"]
    sp_changes = analysis["system_prompt_changes"]
    tool_mutations = analysis["toolset_mutations"]
    hashes = analysis["system_prompt_hashes"]
    all_identical = len(set(hashes)) <= 1 if hashes else True

    passed = breaks == 0 and tool_mutations == 0 and all_identical

    if passed:
        score = 1.0
    elif tool_mutations == 0 and all_identical:
        score = 0.6  # no breaks detected but something else is off
    elif tool_mutations == 0:
        score = 0.3
    else:
        score = 0.0

    return {
        "pass": passed,
        "score": score,
        "details": {
            "scenario": "E4_system_prompt_byte_stable",
            "cache_break_events": breaks,
            "system_prompt_changes": sp_changes,
            "toolset_mutations": tool_mutations,
            "all_hashes_identical": all_identical,
            "unique_hashes": len(set(hashes)) if hashes else 0,
            "expected": "cache_break_events == 0, toolset_mutations == 0, "
                        "all system prompt hashes identical",
            "analysis": analysis["details"],
            "system_prompt_hashes": hashes,
        },
    }


# Scenario ID → grader dispatch
_GRADERS = {
    "E1_cache_stable": _grade_e1_cache_stable,
    "E2_toolset_swap_forbidden": _grade_e2_toolset_swap_forbidden,
    "E3_compression_ok": _grade_e3_compression_ok,
    "E4_system_prompt_byte_stable": _grade_e4_system_prompt_byte_stable,
}


# ─────────────────────────────────────────────────────────────────────────
# Public entry point (contract: rubrics/<suite>.py exports grade())
# ─────────────────────────────────────────────────────────────────────────

def grade(scenario: dict, result: dict) -> dict:
    """Grade a cost_cache scenario.

    Args:
        scenario: The scenario dict from the suite YAML (must contain ``id``).
        result:   The runner result dict with at least ``messages`` and
                  ``final_response``.  May optionally contain
                  ``api_call_snapshots`` — a list of per-turn API-kwargs dicts
                  captured by the runner for deterministic trajectory analysis.

    Returns:
        ``{"pass": bool, "score": float 0-1, "details": dict}``
    """
    scenario_id = scenario.get("id", "")
    messages = result.get("messages", []) or []
    snapshots = result.get("api_call_snapshots")

    # Run the shared trajectory analysis (deterministic — no API calls)
    analysis = _detect_cache_break_events(messages, snapshots)

    # Dispatch to the scenario-specific grader
    grader = _GRADERS.get(scenario_id)
    if grader is None:
        # Unknown scenario — fail with a clear message rather than silently pass
        return {
            "pass": False,
            "score": 0.0,
            "details": {
                "scenario": scenario_id,
                "error": f"No grader registered for scenario '{scenario_id}'. "
                         f"Known: {sorted(_GRADERS.keys())}",
            },
        }

    grade_result = grader(analysis, result)

    # Annotate with scenario-level metadata for the runner report
    grade_result["details"]["scenario_id"] = scenario_id
    grade_result["details"]["messages_count"] = len(messages)
    grade_result["details"]["has_snapshots"] = snapshots is not None
    if snapshots:
        grade_result["details"]["snapshot_count"] = len(snapshots)

    return grade_result