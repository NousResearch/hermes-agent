"""Completion gate — two-stage verification for autonomous task completion.

Gate 1 (goal contract): All active goals must be marked met or cancelled before
declaring complete. This gate only checks contract state, not evidence quality.

Gate 2 (independent judge): A separate, temperature-0 model call reviews a
compressed execution trace and verifies that concrete evidence supports the
completion claim. Rejects vague self-assessments.

Fail-open design: judge call failures, verifyRounds cap (3), cancel_goal escape
hatch, and max_iterations outer guard all prevent infinite loops. A gate is a
gate, not a prison.

Adapted from Reina's s18 completion gate pattern:
  https://github.com/7-e1even/learn-agent/blob/main/s18_completion_gate/README.md
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERIFY_MAX_ROUNDS = 3
MAX_TRACE_CHARS = 16000


# ─── Goal System ────────────────────────────────────────────────────────────

@dataclass
class Goal:
    """A goal contract established at task start or during execution."""
    id: str
    condition: str                  # Human-readable success criterion
    status: str = "active"          # active | met | cancelled
    evidence: str = ""              # Concrete proof (test output, file:line, etc.)


@dataclass
class GoalRegistry:
    """Tracks goals for the current autonomous task."""
    goals: Dict[str, Goal] = field(default_factory=dict)
    _next_id: int = 0

    def add(self, condition: str) -> Goal:
        self._next_id += 1
        g = Goal(id=f"goal-{self._next_id}", condition=condition)
        self.goals[g.id] = g
        return g

    def mark_met(self, goal_id: str, evidence: str) -> bool:
        g = self.goals.get(goal_id)
        if g and g.status == "active":
            g.status = "met"
            g.evidence = evidence
            return True
        return False

    def cancel(self, goal_id: str, reason: str) -> bool:
        g = self.goals.get(goal_id)
        if g and g.status == "active":
            g.status = "cancelled"
            g.evidence = f"Cancelled: {reason}"
            return True
        return False

    @property
    def active_goals(self) -> List[Goal]:
        return [g for g in self.goals.values() if g.status == "active"]

    @property
    def all_resolved(self) -> bool:
        return len(self.active_goals) == 0


# ─── Compressed Trace Builder ───────────────────────────────────────────────

def _clip(text: str, max_len: int) -> str:
    """Truncate with ellipsis."""
    return text if len(text) <= max_len else text[:max_len] + "…"


def _safe_str(obj: Any) -> str:
    """Convert any value to a safe string for trace building."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return " ".join(_safe_str(x) for x in obj)
    if isinstance(obj, dict):
        text = obj.get("text", "")
        if isinstance(text, str):
            return text
    return str(obj)[:500]


def build_verify_trace(
    messages: List[Dict[str, Any]],
    goals: GoalRegistry,
    declare_args: Dict[str, Any],
    max_chars: int = MAX_TRACE_CHARS,
) -> str:
    """Build a compressed execution trace for the judge model.

    The judge doesn't need the full transcript — it needs the *shape* of
    the work done. Four sections:

      1. Message statistics (scale of work)
      2. Tool call sequence (names only — shape of activity)
      3. Recent assistant summaries (last 8, 200 chars each)
      4. Goal contracts + evidence (what was promised vs proven)
      5. Final declaration (the "I'm done" claim itself)
    """
    lines: List[str] = []

    # 1. Scale statistics
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    lines.append(
        f"Messages: {len(messages)} | "
        f"Tool results: {len(tool_msgs)} | "
        f"Assistant turns: {len(assistant_msgs)}"
    )

    # 2. Tool call sequence (names only)
    tool_names: List[str] = []
    for m in assistant_msgs:
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                tool_names.append(name)
    if tool_names:
        lines.append(f"\nTool call sequence ({len(tool_names)} calls):")
        for i, name in enumerate(tool_names):
            lines.append(f"  [{i+1}] {name}")
    else:
        lines.append("\nTool call sequence: (none)")

    # 3. Recent assistant narratives (last 8, clipped to 200 chars)
    narratives = [
        m for m in assistant_msgs
        if m.get("content") and not m.get("tool_calls")
    ][-8:]
    if narratives:
        lines.append("\nRecent assistant summaries:")
        for n in narratives:
            content = n.get("content", "")
            if isinstance(content, str):
                lines.append(f"  - {_clip(content, 200)}")
            elif isinstance(content, list):
                text_parts = [
                    _safe_str(b) for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                lines.append(f"  - {_clip(' '.join(text_parts), 200)}")

    # 4. Goal contracts + evidence
    lines.append("\nGoals:")
    for g in goals.goals.values():
        lines.append(f"  - [{g.id}] ({g.status}) {g.condition}")
        if g.evidence:
            lines.append(f"      evidence: {_clip(g.evidence, 200)}")

    # 5. Final declaration
    summary = declare_args.get("summary", "")
    status = declare_args.get("status", "complete")
    lines.append(f"\nFinal declaration (status={status}):")
    lines.append(_clip(summary, 1500))

    return _clip("\n".join(lines), max_chars)


# ─── Judge Prompts ──────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a completion verifier for an autonomous AI agent.
Review the compressed execution trace and determine if the task was genuinely completed.

CRITICAL RULES:
- "Looks good" / "should work" / self-assessment is NOT evidence
- Concrete evidence required: exit codes, pass counts, file:line references, specific outputs
- If there's no verification action (test run, health check, lint), the answer is NO
- When uncertain, return pass=false — false negatives are safer than false positives

Return ONLY valid JSON: {"pass": bool, "reason": "...", "feedback": "...", "unmet_goal_ids": [...]}"""

ECONOMY_JUDGE_SYSTEM_PROMPT = """Completion verifier. Review this compressed execution trace.
Return ONLY: {"pass": bool, "reason": "1 sentence", "feedback": "1 sentence", "unmet_goal_ids": ["goal-id", ...]}
Pass=false unless there's concrete verification output (exit codes, pass counts, file:line).
Self-assessments are NOT evidence. When uncertain, fail. Do NOT explain your reasoning."""


# ─── JSON Extraction ────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
_JSON_OBJECT_RE = re.compile(r'\{.*"pass".*\}', re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from model output that may be wrapped in markdown or prose.

    Tries: direct parse → markdown code block → regex for any JSON object.
    Fail-open: returns a passing verdict if nothing is parseable.
    """
    if not text or not isinstance(text, str):
        return {"pass": True, "reason": "empty judge output (fail-open)", "feedback": "", "unmet_goal_ids": []}

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object containing "pass"
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fail-open: if we can't parse at all, pass the task
    logger.warning("Judge output unparseable (fail-open): %.200s", text)
    return {"pass": True, "reason": "judge output unparseable (fail-open)", "feedback": "", "unmet_goal_ids": []}


# ─── Completion Gate ────────────────────────────────────────────────────────

@dataclass
class CompletionGate:
    """The two-gate completion verification system.

    Gate 1 (goal contract): All active goals must be marked met or cancelled
    before declaring complete.  This gate only checks contract state, not
    evidence quality.

    Gate 2 (independent judge): A separate model call reviews the compressed
    execution trace.  Rejects vague self-assessments, demands concrete
    evidence.  Fail-open after VERIFY_MAX_ROUNDS refusals.
    """
    goals: GoalRegistry = field(default_factory=GoalRegistry)
    verify_rounds: int = 0
    _judge_model: Optional[str] = None  # None → use agent's model
    _judge_fn: Optional[Any] = None  # Override for testing (rule-based judge)

    def declare_complete(
        self,
        agent: Any,
        messages: List[Dict[str, Any]],
        status: str = "complete",
        summary: str = "",
    ) -> Dict[str, Any]:
        """Execute both gates. Returns ``{"error": "..."}`` or ``{"output": "..."}``.

        ``{"error": ...}`` means the gate rejected — the loop continues and
        the model sees the rejection as a tool error.  ``{"output": ...}``
        means the gate passed — the loop closes with this as the final response.
        """
        declare_args = {"status": status, "summary": summary}

        # ═══ Gate 1: Goal Contract ═══
        open_goals = self.goals.active_goals
        if open_goals:
            goal_list = "\n".join(
                f"  - {g.id}: {g.condition}" for g in open_goals
            )
            return {
                "error": (
                    f"Cannot declare_complete — {len(open_goals)} goal(s) still active:\n"
                    f"{goal_list}\n\n"
                    f"Mark goals as met with mark_goal_met(goal_id, evidence) "
                    f"or cancel them with cancel_goal(goal_id, reason), then retry."
                )
            }

        # ═══ Gate 2: Independent Judge ═══
        if self.verify_rounds >= VERIFY_MAX_ROUNDS:
            # Fail-open: don't trap a completed task forever
            self.verify_rounds = 0
            return {
                "output": (
                    f"<task complete status={status}> "
                    f"(accepted after {VERIFY_MAX_ROUNDS} verify rounds — fail-open)"
                )
            }

        trace = build_verify_trace(messages, self.goals, declare_args)

        try:
            verdict = self._get_verdict(agent, trace)
        except Exception as exc:
            logger.warning("Judge call failed (fail-open): %s", exc)
            self.verify_rounds = 0
            return {
                "output": (
                    f"<task complete status={status}> "
                    f"(judge unavailable — fail-open)"
                )
            }

        if not verdict.get("pass"):
            self.verify_rounds += 1
            # Re-open goals that the judge flagged — next declare_audit_done
            # will hit Gate 1 again for these.
            for goal_id in verdict.get("unmet_goal_ids", []):
                g = self.goals.goals.get(goal_id)
                if g and g.status == "met":
                    g.status = "active"

            # Build rich error for the model to act on
            round_info = f"round {self.verify_rounds}/{VERIFY_MAX_ROUNDS}"
            if self.verify_rounds >= VERIFY_MAX_ROUNDS:
                round_info += " (last attempt before fail-open)"
            return {
                "error": (
                    f"Independent verification did NOT confirm completion "
                    f"({round_info}).\n"
                    f"Reason: {verdict.get('reason', 'unknown')}\n"
                    f"Next step: {verdict.get('feedback', 'Address the issue and retry.')}"
                )
            }

        # Both gates passed — close the loop
        self.verify_rounds = 0
        return {
            "output": (
                f"<task complete status={status}> "
                f"({verdict.get('reason', 'verified')})"
            )
        }

    def _get_verdict(self, agent: Any, trace: str) -> Dict[str, Any]:
        """Get judge verdict, using test override if set."""
        if self._judge_fn is not None:
            return self._judge_fn(trace)
        return self._call_judge(agent, trace)

    def _call_judge(self, agent: Any, trace: str) -> Dict[str, Any]:
        """Call the judge model (separate, temperature 0, JSON-only).

        The judge trace is tiny (~400 chars typically), so even the cheapest
        model can handle it. Uses the economy prompt to minimize token spend.
        """
        client = getattr(agent, "client", None)
        if client is None:
            raise RuntimeError("Agent has no client for judge call")

        model = self._judge_model or getattr(agent, "model", "")
        if not model:
            raise RuntimeError("No model configured for judge call")

        logger.debug(
            "CompletionGate judge call: model=%s trace_len=%d",
            model, len(trace),
        )

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": ECONOMY_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": trace},
            ],
        )

        content = response.choices[0].message.content
        logger.debug("Judge raw response: %.300s", content)
        return _extract_json(content)

    def reset(self):
        """Reset state for a new task."""
        self.goals = GoalRegistry()
        self.verify_rounds = 0
