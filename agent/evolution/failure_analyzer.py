"""Failure Analyzer — AEGIS Digester. 2-tier root-cause analysis (rules + LLM) with 10 failure categories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from agent.evolution.trajectory_collector import EvalCheck, EvalResult, Trajectory
from agent.evolution.task_definition import TaskDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    MISSING_TOOL = "missing_tool"
    TOOL_MISUSE = "tool_misuse"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    STRATEGY_ERROR = "strategy_error"
    EXECUTION_ERROR = "execution_error"
    PREMATURE_COMPLETION = "premature_completion"
    HALLUCINATION = "hallucination"
    TIMEOUT = "timeout"
    LOOP = "loop"
    UNKNOWN = "unknown"


CATEGORY_DESCRIPTIONS = {
    FailureCategory.MISSING_TOOL: "Agent needed a capability that has no corresponding tool",
    FailureCategory.TOOL_MISUSE: "Tool was called with wrong arguments, wrong order, or in wrong context",
    FailureCategory.INSUFFICIENT_CONTEXT: "Agent lacked necessary information (file contents, env state, user intent)",
    FailureCategory.STRATEGY_ERROR: "Wrong approach, plan, or decomposition for the task",
    FailureCategory.EXECUTION_ERROR: "Tool call failed due to environment issues (permissions, missing deps, network)",
    FailureCategory.PREMATURE_COMPLETION: "Agent declared success without actually verifying the outcome",
    FailureCategory.HALLUCINATION: "Agent fabricated information, tool results, or capabilities",
    FailureCategory.TIMEOUT: "Task exceeded time or turn limits",
    FailureCategory.LOOP: "Agent got stuck in a repetitive pattern without making progress",
    FailureCategory.UNKNOWN: "Failure cause could not be determined from available evidence",
}


# ---------------------------------------------------------------------------
# Analysis data model
# ---------------------------------------------------------------------------


@dataclass
class FailureFinding:
    """A single failure finding from trajectory analysis."""
    category: FailureCategory
    confidence: float  # 0.0 - 1.0
    description: str
    evidence: str  # Excerpt from trace supporting this finding
    implicated_tools: List[str] = field(default_factory=list)
    implicated_steps: List[int] = field(default_factory=list)
    suggested_fix_category: str = ""  # "skill", "tool", "prompt", "memory", "env"


@dataclass
class FailureAnalysis:
    """Complete failure analysis for a trajectory."""
    run_id: str = ""
    task_name: str = ""
    overall_score: float = 0.0
    findings: List[FailureFinding] = field(default_factory=list)
    failed_checks: List[Dict[str, Any]] = field(default_factory=list)
    improvement_priorities: List[str] = field(default_factory=list)
    raw_analysis: str = ""  # Full LLM response for debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "overall_score": self.overall_score,
            "findings": [
                {
                    "category": f.category.value,
                    "confidence": f.confidence,
                    "description": f.description,
                    "evidence": f.evidence,
                    "implicated_tools": f.implicated_tools,
                    "implicated_steps": f.implicated_steps,
                    "suggested_fix_category": f.suggested_fix_category,
                }
                for f in self.findings
            ],
            "failed_checks": self.failed_checks,
            "improvement_priorities": self.improvement_priorities,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureAnalysis":
        findings = []
        for f in d.get("findings", []):
            try:
                cat = FailureCategory(f["category"])
            except ValueError:
                cat = FailureCategory.UNKNOWN
            findings.append(FailureFinding(
                category=cat,
                confidence=float(f.get("confidence", 0.5)),
                description=f.get("description", ""),
                evidence=f.get("evidence", ""),
                implicated_tools=f.get("implicated_tools", []),
                implicated_steps=f.get("implicated_steps", []),
                suggested_fix_category=f.get("suggested_fix_category", ""),
            ))
        return cls(
            run_id=d.get("run_id", ""),
            task_name=d.get("task_name", ""),
            overall_score=float(d.get("overall_score", 0.0)),
            findings=findings,
            failed_checks=d.get("failed_checks", []),
            improvement_priorities=d.get("improvement_priorities", []),
            raw_analysis=d.get("raw_analysis", ""),
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """Analyzes failed trajectories to identify root causes.

    Uses a two-tier approach:
    1. Deterministic pattern matching (fast, always available)
    2. LLM-based deep analysis (slower, requires auxiliary model)
    """

    def __init__(self, llm_analyze_fn: Optional[Any] = None):
        """Initialize the analyzer.

        Args:
            llm_analyze_fn: Optional async callable for LLM analysis.
              Signature: async (prompt: str) -> str (JSON response)
        """
        self._llm_analyze_fn = llm_analyze_fn

    def analyze(
        self,
        task: TaskDefinition,
        trajectory: Trajectory,
        eval_result: EvalResult,
    ) -> FailureAnalysis:
        """Analyze a failed trajectory.

        Args:
            task: The task definition.
            trajectory: The failed agent trajectory.
            eval_result: The evaluation results (showing what failed).

        Returns:
            Structured failure analysis with root causes and fix suggestions.
        """
        # Tier 1: Deterministic pattern matching
        rule_findings = self._rule_based_analysis(trajectory, eval_result)

        # Tier 2: LLM deep analysis (if available)
        llm_findings: List[FailureFinding] = []
        if self._llm_analyze_fn and _should_deep_analyze(trajectory):
            try:
                llm_findings = self._llm_deep_analysis(task, trajectory, eval_result)
            except Exception as e:
                logger.warning("LLM failure analysis failed: %s", e)

        # Merge: LLM findings take precedence over rule findings for same category
        all_findings = _merge_findings(rule_findings, llm_findings)

        # Determine improvement priorities
        priorities = _compute_priorities(all_findings)

        # Collect failed checks
        failed_checks = [c.to_dict() for c in eval_result.checks if not c.passed]

        return FailureAnalysis(
            run_id=trajectory.run_id,
            task_name=task.name,
            overall_score=eval_result.score,
            findings=all_findings,
            failed_checks=failed_checks,
            improvement_priorities=priorities,
        )

    # -- Tier 1: Rule-based analysis ------------------------------------------

    def _rule_based_analysis(self, trajectory: Trajectory, eval_result: EvalResult) -> List[FailureFinding]:
        """Apply deterministic heuristics to identify failure patterns."""
        findings: List[FailureFinding] = []

        # Check for timeout
        if trajectory.status == "timeout":
            findings.append(FailureFinding(
                category=FailureCategory.TIMEOUT,
                confidence=0.95,
                description="Task exceeded time or turn limit",
                evidence=f"Status: {trajectory.status}, Turns: {trajectory.total_turns}",
                suggested_fix_category="strategy",
            ))

        # Check for premature completion (agent stopped but tests failed)
        if trajectory.status in ("completed", "success") and not eval_result.passed:
            findings.append(FailureFinding(
                category=FailureCategory.PREMATURE_COMPLETION,
                confidence=0.85,
                description="Agent declared completion but evaluation shows failure",
                evidence=f"Trajectory status: {trajectory.status}, Eval passed: {eval_result.passed}",
                suggested_fix_category="prompt",
            ))

        # Check for loop detection (repeated tool calls with same args)
        loop_steps = _detect_loops(trajectory)
        if loop_steps:
            findings.append(FailureFinding(
                category=FailureCategory.LOOP,
                confidence=0.80,
                description=f"Agent repeated the same tool calls {len(loop_steps)} times without progress",
                evidence=f"Repeated steps: {loop_steps}",
                implicated_steps=loop_steps,
                suggested_fix_category="prompt",
            ))

        # Check for execution errors
        error_tools = _extract_error_tools(trajectory)
        if error_tools:
            most_common = max(set(error_tools), key=error_tools.count)
            findings.append(FailureFinding(
                category=FailureCategory.EXECUTION_ERROR,
                confidence=0.75,
                description=f"Tool '{most_common}' failed {error_tools.count(most_common)} time(s)",
                evidence=f"Tool errors: {error_tools}",
                implicated_tools=list(set(error_tools)),
                suggested_fix_category="tool",
            ))

        # Check for missing tools (agent tried commands that don't map to tools)
        missing_tool_indicators = _detect_missing_tool_indicators(trajectory)
        if missing_tool_indicators:
            findings.append(FailureFinding(
                category=FailureCategory.MISSING_TOOL,
                confidence=0.60,
                description="Agent may have needed a tool that doesn't exist",
                evidence=f"Indicators: {missing_tool_indicators}",
                suggested_fix_category="tool",
            ))

        return findings

    # -- Tier 2: LLM deep analysis --------------------------------------------

    def _llm_deep_analysis(
        self,
        task: TaskDefinition,
        trajectory: Trajectory,
        eval_result: EvalResult,
    ) -> List[FailureFinding]:
        """Use an LLM to deeply analyze failure causes."""
        if not self._llm_analyze_fn:
            return []

        prompt = _build_analysis_prompt(task, trajectory, eval_result)
        try:
            response = self._llm_analyze_fn(prompt)
        except Exception as e:
            logger.debug("LLM deep analysis failed: %s", e)
            return []

        findings = _parse_llm_analysis(response)
        return findings


# ---------------------------------------------------------------------------
# Analysis prompt builder
# ---------------------------------------------------------------------------


def _build_analysis_prompt(
    task: TaskDefinition,
    trajectory: Trajectory,
    eval_result: EvalResult,
) -> str:
    """Build the prompt for LLM-based failure analysis."""
    # Summarize the trajectory
    step_summaries = []
    for step in trajectory.steps[-20:]:  # Last 20 steps
        icon = {"model_call": "🧠", "tool_execution": "🔧", "thinking": "💭"}.get(step.type, "❓")
        status_icon = "✅" if step.status == "success" else "❌"
        step_summaries.append(f"  {icon} Step {step.step} [{step.type}] {status_icon}: {step.summary[:150]}")

    failed_checks_text = "\n".join(
        f"  - {c['type']}: {'PASS' if c.get('passed') else 'FAIL'} — {c.get('detail', '')[:200]}"
        for c in [ck.to_dict() for ck in eval_result.checks]
    )

    errors_text = "\n".join(
        f"  - Step {e['step']}: {e.get('tool', 'unknown')} — {e.get('message', '')[:200]}"
        for e in trajectory.errors[-10:]
    )

    categories_text = "\n".join(
        f"  - {cat.value}: {desc}" for cat, desc in CATEGORY_DESCRIPTIONS.items()
    )

    return f"""You are a failure analysis agent for an AI agent system. Analyze the trajectory below to identify WHY the agent failed.

TASK: {task.name}
DESCRIPTION: {task.description}
DOMAIN: {task.domain}
COMPLEXITY: {task.complexity}/14

EVALUATION RESULT (Score: {eval_result.score:.2f}, Passed: {eval_result.passed}):
{failed_checks_text}

EXECUTION SUMMARY:
- Turns: {trajectory.total_turns}
- Tool calls: {trajectory.total_tool_calls}
- Tokens: {trajectory.total_tokens}
- Status: {trajectory.status}

LAST STEPS:
{chr(10).join(step_summaries) if step_summaries else '  (no steps recorded)'}

ERRORS:
{errors_text if errors_text else '  (no errors recorded)'}

FAILURE CATEGORIES:
{categories_text}

Analyze the failure and respond with ONLY valid JSON:

{{
  "findings": [
    {{
      "category": "<one of the categories above>",
      "confidence": 0.0-1.0,
      "description": "<what went wrong, be specific>",
      "evidence": "<quote or reference from the trace>",
      "implicated_tools": ["<tool names>"],
      "implicated_steps": [<step numbers>],
      "suggested_fix_category": "<skill|tool|prompt|memory|env>"
    }}
  ],
  "improvement_priorities": ["<ordered list of what to fix first>"],
  "root_cause_summary": "<one paragraph summarizing the root cause>"
}}

RULES:
- Be specific. Reference exact tool names, step numbers, and error messages.
- Each finding must cite evidence from the trace.
- Order findings by confidence (highest first).
- Suggest fix categories that are actionable (skill, tool, prompt, memory, env).
- If the agent hallucinated or fabricated information, call it out explicitly.
"""


# ---------------------------------------------------------------------------
# Deterministic heuristics
# ---------------------------------------------------------------------------


def _detect_loops(trajectory: Trajectory, window: int = 3, max_repeats: int = 3) -> List[int]:
    """Detect repetitive tool call patterns (loop detection)."""
    tool_sequence = [
        s.extra.get("tool", "") for s in trajectory.steps
        if s.type == "tool_execution"
    ]
    if len(tool_sequence) < max_repeats * window:
        return []

    loop_steps: List[int] = []
    for i in range(len(tool_sequence) - window * max_repeats + 1):
        pattern = tool_sequence[i:i + window]
        # Count how many times this pattern repeats consecutively
        count = 1
        j = i + window
        while j + window <= len(tool_sequence) and tool_sequence[j:j + window] == pattern:
            count += 1
            j += window
        if count >= max_repeats:
            # Find the actual step numbers
            for step in trajectory.steps:
                if step.type == "tool_execution" and step.extra.get("tool") in pattern:
                    loop_steps.append(step.step)
            break
    return loop_steps


def _extract_error_tools(trajectory: Trajectory) -> List[str]:
    """Extract tool names from recorded errors."""
    return [e.get("tool", "unknown") for e in trajectory.errors if e.get("tool")]


def _detect_missing_tool_indicators(trajectory: Trajectory) -> List[str]:
    """Detect signs that the agent needed a non-existent tool."""
    indicators = []
    for step in trajectory.steps:
        if step.type == "model_call":
            summary_lower = step.summary.lower()
            for phrase in ["need a tool", "don't have a tool", "no tool to", "couldn't find a way",
                          "no way to", "would need", "would require a"]:
                if phrase in summary_lower:
                    indicators.append(f"Step {step.step}: '{phrase}' in model thinking")
    for error in trajectory.errors:
        msg = error.get("message", "").lower()
        if "not found" in msg or "no such" in msg or "command not found" in msg:
            indicators.append(f"Step {error.get('step')}: '{error.get('message', '')[:100]}'")
    return indicators


def _should_deep_analyze(trajectory: Trajectory) -> bool:
    """Determine if deep LLM analysis is warranted."""
    # Only deep-analyze if there are enough steps to make it worthwhile
    if trajectory.total_turns < 2:
        return False
    # Skip if it's clearly a timeout (fast path)
    if trajectory.status == "timeout" and trajectory.total_tool_calls <= 1:
        return False
    return True


def _merge_findings(
    rule_findings: List[FailureFinding],
    llm_findings: List[FailureFinding],
) -> List[FailureFinding]:
    """Merge rule-based and LLM findings, with LLM taking precedence."""
    seen_categories: set = set()
    merged: List[FailureFinding] = []

    # LLM findings first (higher quality when available)
    for f in llm_findings:
        if f.category not in seen_categories:
            merged.append(f)
            seen_categories.add(f.category)

    # Add rule findings for categories LLM didn't cover
    for f in rule_findings:
        if f.category not in seen_categories:
            merged.append(f)
            seen_categories.add(f.category)

    # Sort by confidence
    merged.sort(key=lambda f: f.confidence, reverse=True)
    return merged


def _compute_priorities(findings: List[FailureFinding]) -> List[str]:
    """Determine the ordered list of improvement priorities."""
    # Weight: higher confidence + actionable fix categories first
    scored = []
    for f in findings:
        score = f.confidence
        if f.suggested_fix_category in ("tool", "prompt"):
            score += 0.1  # Code/prompt fixes are high-impact
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [f"{f.category.value} → {f.suggested_fix_category} ({f.confidence:.0%})" for _, f in scored]


def _parse_llm_analysis(response: str) -> List[FailureFinding]:
    """Parse the LLM's JSON analysis response."""
    # Try direct JSON parse
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM analysis JSON from code block")
                return []
        else:
            logger.warning("Failed to parse LLM analysis response as JSON")
            return []

    findings = []
    for f in data.get("findings", []):
        try:
            cat = FailureCategory(f.get("category", "unknown"))
        except ValueError:
            cat = FailureCategory.UNKNOWN
        findings.append(FailureFinding(
            category=cat,
            confidence=float(f.get("confidence", 0.5)),
            description=f.get("description", ""),
            evidence=f.get("evidence", ""),
            implicated_tools=f.get("implicated_tools", []),
            implicated_steps=f.get("implicated_steps", []),
            suggested_fix_category=f.get("suggested_fix_category", ""),
        ))
    return findings
