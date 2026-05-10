"""
SOTA goal judge — 10/10 evaluation with semantic loop detection, calibrated
scoring, error pattern tracking, and hard negative-constraint output.

Key improvements over v2:
- Semantic loop detection: fingerprints outcome patterns, not just tool names
- Calibrated scoring: reference anchors for each completion band with examples
- Error pattern tracking: distinguishes transient failures from systemic bugs
- Negative constraints: outputs "do NOT do" alongside "try this instead"
- Progress trend: detects regression by comparing consecutive verdicts
- Verification demands: requires explicit artifact confirmation when scoring high
"""

from __future__ import annotations

import json, logging, re, time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from .goal_scratchpad import GoalScratchpad, SubTask

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_TIMEOUT = 30.0
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
_SEMANTIC_LOOP_THRESHOLD = 3   # same outcome pattern seen this many times = loop
_EXACT_LOOP_THRESHOLD = 2      # same tool+arg seen this many times = exact loop
_ERROR_PATTERN_THRESHOLD = 3   # same error seen this many times = systemic


@dataclass
class JudgeVerdict:
    """The judge's evaluation of the last turn."""

    action: str  # continue_as_is | pivot_strategy | refine_output | decompose_further | ask_user | done | failed
    completion: float  # 0.0 - 1.0
    progress_signal: str  # "forward" | "stalled" | "looping" | "regressing" | "unclear"
    quality_score: float  # 0.0 - 1.0
    stuck_details: str = ""
    reasoning: str = ""
    suggested_next_action: str = ""
    suggested_pivot: str = ""
    negative_constraint: str = ""  # what to AVOID doing
    error_pattern: str = ""  # systemic error detected (if any)
    previous_completion: float = 0.0  # for trend comparison

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def default_continue(cls) -> "JudgeVerdict":
        return cls(
            action="continue_as_is",
            completion=0.0,
            progress_signal="unclear",
            quality_score=0.5,
            reasoning="judge unavailable — continuing",
        )


# ────────────────────────────────────────────────────────────────
# System prompt — calibrated for precision
# ────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are a precision goal evaluator for an autonomous AI agent. Evaluate progress across three dimensions with calibrated scoring.

## COMPLETION (0.0-1.0) — Tangible output only
- 0.00-0.15: NOTHING produced. Still planning/researching/thinking.
  Example: "I'll start by looking at the codebase" with no files created.
- 0.16-0.35: SCAFFOLDING exists. Files created but empty/stub. Plans written. Research gathered.
  Example: Created main.py with skeleton, researched API docs.
- 0.36-0.55: PARTIAL WORK. Core logic exists but incomplete. Some sub-tasks done.
  Example: Wrote 3 of 5 endpoints, 2 tests pass, 3 fail.
- 0.56-0.75: MOSTLY DONE. Main deliverable exists, edge cases missing.
  Example: App runs, all endpoints work, no error handling yet.
- 0.76-0.90: COMPLETE BUT UNVERIFIED. All pieces written, NOT confirmed working.
  Example: "I've written everything" but no tests run, no file confirmation.
- 0.91-1.00: VERIFIED COMPLETE. Output confirmed to exist and work.
  Example: Tests pass, files stat'd, server responds 200, URLs verified.
DO NOT score above 0.75 without verification. DO NOT score above 0.90 without explicit confirmation.

## PROGRESS SIGNAL — Movement detection
- "forward": New artifacts created, errors RESOLVED (not just retried), new data gathered, sub-tasks marked complete
- "stalled": Vague responses, same output as last turn, no concrete change, "still working" without results
- "looping": Same pattern repeating — same error >2 times, same file edited 3+ times with no resolution, tool called identically 2+ times
- "regressing": Work deleted/overwritten, earlier progress undone, contradictions, working feature now broken
- "unclear": First turn or insufficient data (never use for turn 3+)

## QUALITY (0.0-1.0) — Production readiness
- 0.00-0.35: Broken, incomplete, unverified. Won't work.
- 0.36-0.55: Runs but fragile. No tests, no docs, hardcoded values.
- 0.56-0.75: Functional. Basic tests exist, some documentation.
- 0.76-0.90: Solid. Tests pass, documented, handles edge cases.
- 0.91-1.00: Production. Verified, documented, robust, secure, edge cases covered.

## ACTION RULES (strict — follow exactly)

done: completion >= 0.91 AND quality >= 0.70 AND agent explicitly says "done/complete/finished" AND NOT "Next I'll..." or "I still need to..."
refine_output: completion >= 0.76 AND quality < 0.70 (improve quality, don't rebuild)
pivot_strategy: progress="looping" OR progress="regressing" OR error_pattern detected OR same approach failed 2+ times
decompose_further: Agent's response is confused, overwhelmed, or task too large for current sub-task
ask_user: Missing credentials, external dependency, ambiguous requirement, permission denied, user must decide
continue_as_is: completion < 0.91 AND progress="forward" AND quality >= 0.50
failed: Fundamentally impossible (missing hardware, blocked by policy, contradictory requirements)

STRICT RULE: If agent says ANY variant of "next I'll", "I need to", "let me now", "should also", "remaining" → NOT done. Score accordingly.

## NEGATIVE CONSTRAINT
When recommending a pivot, also specify what NOT to do. This prevents the agent from trying the failing approach again.

## ERROR PATTERNS
If an error occurred 3+ times across turns, identify it as systemic. Distinguish from transient failures.

## Output format — JSON only, no markdown:
{"action":"done","completion":0.95,"progress_signal":"forward","quality_score":0.85,"stuck_details":"","reasoning":"All tests pass, files confirmed","suggested_next_action":"","suggested_pivot":"","negative_constraint":"","error_pattern":""}"""


# ────────────────────────────────────────────────────────────────
# Parsing (robust to malformed JSON)
# ────────────────────────────────────────────────────────────────

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_VALID_ACTIONS = {"continue_as_is", "pivot_strategy", "refine_output", "decompose_further", "ask_user", "done", "failed"}
_VALID_PROGRESS = {"forward", "stalled", "looping", "regressing", "unclear"}


def _parse_judge_response(raw: str) -> JudgeVerdict:
    """Parse judge output. Returns fail-open default on any error."""
    if not raw:
        return JudgeVerdict.default_continue()

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                pass

    if not isinstance(data, dict):
        logger.debug("goal judge: response was not JSON: %r", text[:200])
        return JudgeVerdict.default_continue()

    action = str(data.get("action") or "continue_as_is")
    if action not in _VALID_ACTIONS:
        action = "continue_as_is"

    progress = str(data.get("progress_signal") or "unclear")
    if progress not in _VALID_PROGRESS:
        progress = "unclear"

    try:
        completion = float(data.get("completion", 0.0))
    except (ValueError, TypeError):
        completion = 0.0

    try:
        quality = float(data.get("quality_score", 0.5))
    except (ValueError, TypeError):
        quality = 0.5

    return JudgeVerdict(
        action=action,
        completion=max(0.0, min(1.0, completion)),
        progress_signal=progress,
        quality_score=max(0.0, min(1.0, quality)),
        stuck_details=str(data.get("stuck_details") or ""),
        reasoning=str(data.get("reasoning") or ""),
        suggested_next_action=str(data.get("suggested_next_action") or ""),
        suggested_pivot=str(data.get("suggested_pivot") or ""),
        negative_constraint=str(data.get("negative_constraint") or ""),
        error_pattern=str(data.get("error_pattern") or ""),
        previous_completion=float(data.get("previous_completion", 0.0) or 0.0),
    )


# ────────────────────────────────────────────────────────────────
# Semantic loop detection
# ────────────────────────────────────────────────────────────────


def _outcome_fingerprint(tc: Dict[str, Any]) -> str:
    """Build a semantic fingerprint from a tool call — captures what the tool
    was TRYING to do, not just its name/args."""
    name = tc.get("function", {}).get("name", tc.get("name", "?"))
    args_str = str(tc.get("function", {}).get("arguments", tc.get("args", "")))
    try:
        args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
    except Exception:
        args_obj = {}

    # Classify intent
    intent = "unknown"
    if name in ("terminal", "execute_code"):
        cmd = str(args_obj.get("command", args_obj.get("code", "")))[:100]
        if "install" in cmd or "brew" in cmd or "pip" in cmd or "apt" in cmd:
            intent = "install"
        elif "test" in cmd or "pytest" in cmd:
            intent = "run_tests"
        elif "git" in cmd:
            intent = "git"
        elif "curl" in cmd or "http" in cmd:
            intent = "http_request"
        elif "docker" in cmd:
            intent = "docker"
        elif "build" in cmd or "compile" in cmd:
            intent = "build"
        else:
            intent = "shell_exec"
    elif name in ("read_file", "file_read"):
        path = str(args_obj.get("path", args_obj.get("file", "")))
        intent = f"read:{path.rsplit('/', 1)[-1] if '/' in path else path}"
    elif name in ("write_file", "file_write"):
        path = str(args_obj.get("path", ""))
        intent = f"write:{path.rsplit('/', 1)[-1] if '/' in path else path}"
    elif name in ("patch", "edit"):
        path = str(args_obj.get("path", ""))
        intent = f"edit:{path.rsplit('/', 1)[-1] if '/' in path else path}"
    elif name in ("search_files", "grep", "search"):
        intent = "search"
    elif name in ("browser_navigate", "browser_click", "web_browse"):
        intent = "browse"
    elif name in ("web_search", "web_extract"):
        intent = "web_research"
    elif name in ("delegate_task", "delegate"):
        goal = str(args_obj.get("goal", args_obj.get("task", "")))[:60]
        intent = f"delegate:{goal}"
    elif name == "memory":
        intent = "memory"
    elif name == "clarify":
        intent = "clarify"

    return f"{intent}"


def _detect_error_patterns(tool_calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Detect recurring errors across tool calls. Returns (is_systemic, description)."""
    errors: List[str] = []
    for tc in tool_calls:
        # Errors may be in the result, but we only have calls here
        # Check if the call itself looks like a retry of a failing pattern
        name = tc.get("function", {}).get("name", tc.get("name", ""))
        if name in ("terminal", "execute_code"):
            args_str = str(tc.get("function", {}).get("arguments", tc.get("args", "")))
            try:
                args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
            except Exception:
                args_obj = {}
            cmd = str(args_obj.get("command", args_obj.get("code", "")))
            if "error" in cmd.lower() or "fail" in cmd.lower() or "retry" in cmd.lower():
                errors.append(cmd[:80])

    if len(errors) >= _ERROR_PATTERN_THRESHOLD:
        counter = Counter(errors)
        most_common = counter.most_common(1)[0]
        if most_common[1] >= _ERROR_PATTERN_THRESHOLD:
            return True, f"Systemic error: '{most_common[0]}' repeated {most_common[1]}x"
    return False, ""


def _detect_semantic_loop(tool_calls: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Detect if the agent is repeating the same INTENT, not just the same command.
    Returns (is_looping, description)."""
    # Exact loop check runs first (lower threshold = more sensitive)
    fingerprints = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", tc.get("name", "?"))
        args_str = str(tc.get("function", {}).get("arguments", tc.get("args", "")))
        try:
            args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
        except Exception:
            args_obj = {}
        first_val = ""
        for k in ("command", "path", "query", "url", "goal", "code", "prompt"):
            if k in args_obj:
                first_val = str(args_obj[k])[:60]
                break
        fingerprints.append(f"{name}|{first_val}")

    exact_counts = Counter(fingerprints)
    for fp, count in exact_counts.items():
        if count >= _EXACT_LOOP_THRESHOLD:
            return True, f"Exact loop: '{fp}' repeated {count}x"

    # Semantic loop check (higher threshold)
    if len(tool_calls) < _SEMANTIC_LOOP_THRESHOLD:
        return False, ""

    outcomes = [_outcome_fingerprint(tc) for tc in tool_calls]
    counts = Counter(outcomes)
    for intent, count in counts.items():
        if count >= _SEMANTIC_LOOP_THRESHOLD:
            return True, f"Semantic loop: intent '{intent}' repeated {count}x"

    return False, ""


def _detect_progress_trend(verdicts: List[Dict[str, Any]]) -> str:
    """Compare last few completion scores to detect regression."""
    if len(verdicts) < 2:
        return "unclear"

    scores = []
    for v in verdicts[-5:]:
        try:
            scores.append(float(v.get("completion", 0)))
        except (ValueError, TypeError):
            pass

    if len(scores) < 2:
        return "unclear"

    # Monotonic regression: last N scores strictly decreasing
    if len(scores) >= 3 and all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
        return "regressing"

    # Last score lower than previous
    if scores[-1] < scores[-2] - 0.05:
        return "regressing"

    # Last score higher than previous
    if scores[-1] > scores[-2] + 0.02:
        return "forward"

    # Flat
    if abs(scores[-1] - scores[-2]) <= 0.02:
        if len(scores) >= 3 and abs(scores[-1] - scores[-3]) <= 0.02:
            return "stalled"

    return "unclear"


def _summarize_tools(tool_calls: List[Dict[str, Any]], limit: int = 8) -> str:
    """Compress recent tool calls into a short list for the judge, with loop detection."""
    if not tool_calls:
        return "(no tools called this turn)"

    is_semantic, sem_desc = _detect_semantic_loop(tool_calls)
    is_error, err_desc = _detect_error_patterns(tool_calls)

    lines = []
    if is_semantic:
        lines.append(f"⚠ LOOP: {sem_desc}")
    if is_error:
        lines.append(f"⚠ SYSTEMIC ERROR: {err_desc}")

    for tc in tool_calls[-limit:]:
        name = tc.get("function", {}).get("name", tc.get("name", "?"))
        args_str = str(tc.get("function", {}).get("arguments", tc.get("args", "")))
        try:
            args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
        except Exception:
            args_obj = {}

        # Show intent + key detail
        intent = _outcome_fingerprint(tc)
        detail = ""
        for k in ("command", "path", "query", "url", "goal", "code", "prompt"):
            if k in args_obj:
                detail = str(args_obj[k])[:60]
                break
        if detail:
            lines.append(f"  {name}[{intent}]: {detail}")
        else:
            lines.append(f"  {name}[{intent}]")

    return "\n".join(lines)


def _summarize_scratchpad(pad: GoalScratchpad) -> str:
    """Build a compact scratchpad summary for the judge prompt."""
    parts = []
    parts.append(f"Sub-tasks: {pad.completed_count}/{pad.total_count} done, {pad.in_progress_count} in progress, {pad.blocked_count} blocked")
    parts.append(f"Confidence: {pad.confidence}%")
    parts.append(f"Artifacts: {len(pad.artifacts)} created (verified: {sum(1 for a in pad.artifacts if a.verified)})")
    parts.append(f"Blockers: {len(pad.blockers)} active")
    parts.append(f"Pivots: {pad.pivot_count}, approaches tried: {len(pad.previous_approaches)}")

    if pad.previous_approaches:
        parts.append(f"Approaches: {', '.join(pad.previous_approaches[-3:])}")
    if pad.blockers:
        parts.append(f"Blockers: {'; '.join(pad.blockers[:3])}")
    if pad.sub_tasks:
        status_map: Dict[str, list] = {}
        for st in pad.sub_tasks:
            status_map.setdefault(st.status, []).append(st.description[:40])
        for status, tasks in status_map.items():
            if tasks:
                parts.append(f"  [{status}] {', '.join(tasks[:3])}")
    if pad.history:
        parts.append(f"Turn history: {len(pad.history)} entries")
        if pad.history:
            last = pad.history[-1]
            parts.append(f"  Last: comp={last.get('completion', 0)}, progress={last.get('progress', '?')}")

    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────
# Main judge function
# ────────────────────────────────────────────────────────────────

_JUDGE_TEMPLATE = """## Goal
{goal}

## Scratchpad State
{scratchpad}

## Tool Calls (last 8, with intent classification)
{tools}

## Agent's Final Response
{response}

Evaluate progress across completion, progress, and quality. Include negative_constraint (what to avoid) if pivoting. Output a single JSON verdict."""


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def evaluate_turn(
    goal: str,
    last_response: str,
    scratchpad: GoalScratchpad,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    previous_completion: float = 0.0,
) -> JudgeVerdict:
    """Run the full multi-dimensional evaluation with semantic loop detection."""
    if not goal.strip():
        return JudgeVerdict.default_continue()

    if not last_response.strip():
        return JudgeVerdict(
            action="continue_as_is",
            completion=0.0,
            progress_signal="stalled",
            quality_score=0.0,
            reasoning="empty response",
        )

    # ── Pre-processing: semantic loop & error detection ──────────
    is_semantic, sem_desc = _detect_semantic_loop(tool_calls or [])
    is_error, err_desc = _detect_error_patterns(tool_calls or [])

    # ── Trend detection from scratchpad history ──────────────────
    trend = _detect_progress_trend(scratchpad.history)

    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return JudgeVerdict.default_continue()

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return JudgeVerdict.default_continue()

    if client is None or not model:
        return JudgeVerdict.default_continue()

    tools_summary = _summarize_tools(tool_calls or [])
    scratchpad_summary = _summarize_scratchpad(scratchpad)

    prompt = _JUDGE_TEMPLATE.format(
        goal=_truncate(goal, 2000),
        scratchpad=scratchpad_summary,
        tools=tools_summary,
        response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
    )

    # Inject pre-detected signals into the prompt so the LLM judge sees them
    if is_semantic:
        prompt += f"\n\n[PRE-DETECTED: Semantic loop — {sem_desc}. Consider pivot_strategy.]"
    if is_error:
        prompt += f"\n\n[PRE-DETECTED: Systemic error — {err_desc}. Consider pivot_strategy.]"
    if trend == "regressing":
        prompt += "\n\n[PRE-DETECTED: Progress trend is REGRESSING. Consider pivot_strategy.]"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("goal judge: API call failed (%s) — continuing", exc)
        return JudgeVerdict.default_continue()

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    verdict = _parse_judge_response(raw)

    # ── Hard override: pre-detected loops force pivot ────────────
    if is_semantic and verdict.action not in ("pivot_strategy", "done", "failed", "ask_user"):
        verdict.action = "pivot_strategy"
        if not verdict.suggested_pivot:
            verdict.suggested_pivot = f"Stop repeating. {sem_desc}. Try a fundamentally different approach."
        if not verdict.negative_constraint:
            verdict.negative_constraint = f"Do NOT retry the same {sem_desc.split(':')[0].strip().split()[-1]} pattern."
    if is_error and verdict.action not in ("pivot_strategy", "done", "failed", "ask_user"):
        verdict.action = "pivot_strategy"
        if not verdict.suggested_pivot:
            verdict.suggested_pivot = f"Fix systemic error: {err_desc}. Try alternative tool or approach."
        if not verdict.negative_constraint:
            verdict.negative_constraint = "Do NOT retry the failing command without changing it."

    # ── Hard override: trending regressing forces pivot ──────────
    if trend == "regressing" and verdict.action not in ("pivot_strategy", "done", "failed", "ask_user"):
        verdict.action = "pivot_strategy"
        verdict.suggested_pivot = verdict.suggested_pivot or "Progress is regressing — stop current approach and try something new."
        verdict.negative_constraint = verdict.negative_constraint or "Do NOT continue current approach."

    # ── Hard override: score >0.75 without verification → cap ────
    verified_count = sum(1 for a in scratchpad.artifacts if a.verified)
    if verdict.completion > 0.75 and verified_count == 0 and verdict.action == "done":
        verdict.completion = 0.75
        verdict.action = "refine_output"
        verdict.reasoning = "UNCAPPED: High completion scored without verification. Capped to 0.75. Verify artifacts first."

    verdict.previous_completion = previous_completion
    verdict.error_pattern = err_desc if is_error else ""

    logger.info(
        "goal judge: action=%s completion=%.2f progress=%s quality=%.2f loop=%s err=%s trend=%s",
        verdict.action, verdict.completion, verdict.progress_signal, verdict.quality_score,
        is_semantic, is_error, trend,
    )
    return verdict


# ────────────────────────────────────────────────────────────────
# Display helpers
# ────────────────────────────────────────────────────────────────

_ACTION_ICONS = {"done": "✓", "failed": "✗", "ask_user": "?", "refine_output": "↻", "pivot_strategy": "↺", "decompose_further": "⊞", "continue_as_is": "→"}

_ACTION_LABELS = {
    "done": "Goal achieved",
    "failed": "Goal failed",
    "ask_user": "Blocked — needs your input",
    "refine_output": "Output needs improvement",
    "pivot_strategy": "Strategy pivot",
    "decompose_further": "Decomposing task",
    "continue_as_is": "Continuing",
}


def verdict_icon(verdict: JudgeVerdict) -> str:
    return _ACTION_ICONS.get(verdict.action, "?")


def verdict_label(verdict: JudgeVerdict) -> str:
    base = _ACTION_LABELS.get(verdict.action, verdict.action)
    if verdict.action == "pivot_strategy" and verdict.suggested_pivot:
        base += f": {verdict.suggested_pivot[:80]}"
    elif verdict.action == "ask_user" and verdict.stuck_details:
        base += f": {verdict.stuck_details[:80]}"
    elif verdict.action == "refine_output" and verdict.suggested_next_action:
        base += f": {verdict.suggested_next_action[:80]}"
    return base


def verdict_message(verdict: JudgeVerdict, turns: int, max_turns: int) -> str:
    """Build a user-visible one-liner for the verdict."""
    icon = verdict_icon(verdict)
    label = verdict_label(verdict)
    bar = _mini_progress(verdict.completion)
    nc = ""
    if verdict.negative_constraint:
        nc = f" [NO: {verdict.negative_constraint[:60]}]"
    ep = ""
    if verdict.error_pattern:
        ep = f" [ERR: {verdict.error_pattern[:60]}]"
    return f"{icon} [{bar}] ({turns}/{max_turns}) {label}{nc}{ep}"


_PROGRESS_CHARS = " ▏▎▍▌▋▊▉█"


def _mini_progress(fraction: float, width: int = 10) -> str:
    filled = int(fraction * width * 8)
    full = filled // 8
    partial = filled % 8
    bar = "█" * full
    if partial > 0 and full < width:
        bar += _PROGRESS_CHARS[partial]
        full += 1
    return bar + " " * (width - full)
