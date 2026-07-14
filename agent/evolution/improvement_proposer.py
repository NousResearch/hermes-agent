"""Improvement Proposer — AEGIS Evolver. Generates skill/tool/prompt fixes from failure analysis."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.task_definition import ImprovementActionType, TaskDefinition
from agent.evolution.failure_analyzer import FailureAnalysis, FailureCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Improvement proposal data model
# ---------------------------------------------------------------------------


@dataclass
class ImprovementProposal:
    """A concrete, reviewable improvement proposal."""

    action_type: ImprovementActionType
    target: str  # What is being changed (skill name, tool name, prompt section, etc.)
    description: str  # Human-readable description of the change
    rationale: str  # Why this change should fix the failure

    # The actual change content
    content: str = ""  # For skills: SKILL.md content; for tools: Python code; for prompts: new text

    # For code modifications: the specific edit
    file_path: str = ""       # Path to the file being modified
    old_string: str = ""      # For patch operations
    new_string: str = ""

    # Metadata
    confidence: float = 0.5  # Estimated probability this fix works
    failure_categories_addressed: List[str] = field(default_factory=list)
    estimated_token_cost: int = 0

    # Safety
    is_destructive: bool = False  # Could break existing functionality
    requires_approval: bool = True
    rollback_instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "description": self.description,
            "rationale": self.rationale,
            "content": self.content,
            "file_path": self.file_path,
            "old_string": self.old_string,
            "new_string": self.new_string,
            "confidence": self.confidence,
            "failure_categories_addressed": self.failure_categories_addressed,
            "estimated_token_cost": self.estimated_token_cost,
            "is_destructive": self.is_destructive,
            "requires_approval": self.requires_approval,
            "rollback_instructions": self.rollback_instructions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImprovementProposal":
        try:
            action_type = ImprovementActionType(d["action_type"])
        except ValueError:
            action_type = ImprovementActionType.SKILL_CREATE
        return cls(
            action_type=action_type,
            target=d.get("target", ""),
            description=d.get("description", ""),
            rationale=d.get("rationale", ""),
            content=d.get("content", ""),
            file_path=d.get("file_path", ""),
            old_string=d.get("old_string", ""),
            new_string=d.get("new_string", ""),
            confidence=float(d.get("confidence", 0.5)),
            failure_categories_addressed=d.get("failure_categories_addressed", []),
            estimated_token_cost=int(d.get("estimated_token_cost", 0)),
            is_destructive=bool(d.get("is_destructive", False)),
            requires_approval=bool(d.get("requires_approval", True)),
            rollback_instructions=d.get("rollback_instructions", ""),
        )


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class ImprovementProposer:
    """Generates improvement proposals based on failure analysis.

    Uses an LLM (auxiliary model) to propose concrete, actionable fixes.
    Can also generate rule-based proposals for common failure patterns.
    """

    def __init__(self, llm_propose_fn: Optional[Any] = None):
        """Initialize the proposer.

        Args:
            llm_propose_fn: Optional async callable for LLM-based proposal generation.
              Signature: async (prompt: str) -> str (JSON response with proposals)
        """
        self._llm_propose_fn = llm_propose_fn

    def propose(
        self,
        task: TaskDefinition,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]] = None,
        existing_tools: Optional[List[str]] = None,
    ) -> List[ImprovementProposal]:
        """Generate improvement proposals for a failed task.

        Args:
            task: The task that failed.
            analysis: The failure analysis from FailureAnalyzer.
            existing_skills: Names of existing skills (to avoid duplicates).
            existing_tools: Names of existing tools (to avoid duplicates).

        Returns:
            Ordered list of improvement proposals (highest confidence first).
        """
        proposals: List[ImprovementProposal] = []

        # Tier 1: Rule-based proposals for common patterns
        rule_proposals = self._rule_based_proposals(analysis, existing_skills, existing_tools)
        proposals.extend(rule_proposals)

        # Tier 2: LLM-generated proposals (if available)
        if self._llm_propose_fn and analysis.findings:
            try:
                llm_proposals = self._llm_generate_proposals(task, analysis, existing_skills, existing_tools)
                proposals.extend(llm_proposals)
            except Exception as e:
                logger.warning("LLM proposal generation failed: %s", e)

        # Deduplicate by target
        proposals = _deduplicate_proposals(proposals)

        # Sort by confidence (highest first)
        proposals.sort(key=lambda p: p.confidence, reverse=True)

        return proposals

    # -- Rule-based proposals ------------------------------------------------

    def _rule_based_proposals(
        self,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]] = None,
        existing_tools: Optional[List[str]] = None,
    ) -> List[ImprovementProposal]:
        """Generate proposals from deterministic rules with REAL content."""
        proposals: List[ImprovementProposal] = []
        existing_skills = existing_skills or []
        existing_tools = existing_tools or []

        for finding in analysis.findings:
            if finding.category == FailureCategory.MISSING_TOOL:
                tool_desc = finding.description[:200]
                skill_name = f"workaround-{analysis.task_name}"[:64]
                if skill_name not in existing_skills:
                    proposals.append(ImprovementProposal(
                        action_type=ImprovementActionType.SKILL_CREATE,
                        target=skill_name,
                        description=f"Workarounds for: {tool_desc}"[:60],
                        rationale=f"Agent needed capability that doesn't exist: {finding.description}",
                        content=_generate_workaround_skill(analysis.task_name, finding),
                        failure_categories_addressed=[finding.category.value],
                        confidence=0.5,
                        requires_approval=False,
                        rollback_instructions=f"Delete ~/.hermes/skills/{skill_name}/",
                    ))

            elif finding.category == FailureCategory.PREMATURE_COMPLETION:
                skill_name = "verify-before-complete"
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.SKILL_CREATE,
                    target=skill_name,
                    description="Verify task completion before declaring done",
                    rationale="Agent declared success without verification. Need explicit verification step.",
                    content=_generate_verification_skill(analysis.task_name, analysis.failed_checks),
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.7,
                    requires_approval=False,
                    rollback_instructions=f"Delete ~/.hermes/skills/{skill_name}/",
                ))

            elif finding.category == FailureCategory.LOOP:
                implicated = finding.implicated_tools or ["unknown"]
                skill_name = "detect-and-break-loops"
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.SKILL_CREATE,
                    target=skill_name,
                    description="Detect and break repetitive tool-call loops",
                    rationale=f"Agent stuck in loop with tools: {', '.join(implicated)}",
                    content=_generate_loop_detection_skill(implicated, finding),
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.65,
                    requires_approval=False,
                    rollback_instructions=f"Delete ~/.hermes/skills/{skill_name}/",
                ))

            elif finding.category == FailureCategory.EXECUTION_ERROR:
                tool = finding.implicated_tools[0] if finding.implicated_tools else "terminal"
                skill_name = f"troubleshoot-{tool}"[:64]
                if skill_name not in existing_skills:
                    proposals.append(ImprovementProposal(
                        action_type=ImprovementActionType.SKILL_CREATE,
                        target=skill_name,
                        description=f"Troubleshoot common {tool} errors",
                        rationale=f"Tool execution errors with {tool}: {finding.description[:100]}",
                        content=_generate_troubleshooting_skill(tool, finding),
                        failure_categories_addressed=[finding.category.value],
                        confidence=0.55,
                        requires_approval=False,
                        rollback_instructions=f"Delete ~/.hermes/skills/{skill_name}/",
                    ))

            elif finding.category == FailureCategory.TIMEOUT:
                skill_name = f"time-efficient-{analysis.task_name}"[:64]
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.SKILL_CREATE,
                    target=skill_name,
                    description=f"Time-efficient approach for {analysis.task_name}",
                    rationale=f"Task timed out — need more efficient strategy: {finding.description}",
                    content=_generate_time_efficient_skill(analysis.task_name, finding),
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.6,
                    requires_approval=False,
                    rollback_instructions=f"Delete ~/.hermes/skills/{skill_name}/",
                ))

        return proposals

    # -- LLM-based proposals -------------------------------------------------

    def _llm_generate_proposals(
        self,
        task: TaskDefinition,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]],
        existing_tools: Optional[List[str]],
    ) -> List[ImprovementProposal]:
        """Use an LLM to generate improvement proposals."""
        if not self._llm_propose_fn:
            return []

        import asyncio

        prompt = _build_proposal_prompt(task, analysis, existing_skills or [], existing_tools or [])
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._llm_propose_fn(prompt))
                    response = future.result(timeout=60)
            else:
                response = loop.run_until_complete(self._llm_propose_fn(prompt))
        except Exception as e:
            logger.debug("LLM proposal generation skipped: %s", e)
            return []

        return _parse_llm_proposals(response)


# ---------------------------------------------------------------------------
# Proposal prompt
# ---------------------------------------------------------------------------


def _build_proposal_prompt(
    task: TaskDefinition,
    analysis: FailureAnalysis,
    existing_skills: List[str],
    existing_tools: List[str],
) -> str:
    """Build the prompt for LLM-based proposal generation."""
    findings_text = "\n".join(
        f"  [{i+1}] {f.category.value} (confidence: {f.confidence:.0%})\n"
        f"      Description: {f.description}\n"
        f"      Evidence: {f.evidence[:200]}\n"
        f"      Suggested fix category: {f.suggested_fix_category}"
        for i, f in enumerate(analysis.findings)
    )

    failed_checks_text = "\n".join(
        f"  - {c.get('type', 'unknown')}: {c.get('detail', '')[:200]}"
        for c in analysis.failed_checks
    )

    existing_skills_text = "\n".join(f"  - {s}" for s in existing_skills[:30]) if existing_skills else "  (none)"
    existing_tools_text = "\n".join(f"  - {t}" for t in existing_tools[:50]) if existing_tools else "  (none)"

    return f"""You are an improvement agent for an AI agent system. Generate concrete, actionable proposals to fix the failures described below.

FAILED TASK: {task.name}
DESCRIPTION: {task.description}
DOMAIN: {task.domain}

FAILURE ANALYSIS:
{findings_text}

FAILED EVALUATION CHECKS:
{failed_checks_text}

EXISTING SKILLS (don't create duplicates):
{existing_skills_text}

EXISTING TOOLS (don't create duplicates):
{existing_tools_text}

IMPROVEMENT PRIORITIES: {', '.join(analysis.improvement_priorities)}

Generate improvement proposals. Respond with ONLY valid JSON:

{{
  "proposals": [
    {{
      "action_type": "skill_create|skill_patch|tool_create|tool_modify|prompt_modify|memory_update|strategy_change",
      "target": "<name of skill, tool, prompt section, etc.>",
      "description": "<one-line description of the change>",
      "rationale": "<why this addresses the root cause>",
      "content": "<the full SKILL.md content, Python code, or prompt text>",
      "file_path": "<if modifying an existing file, the path>",
      "old_string": "<if patching, the exact text to replace>",
      "new_string": "<if patching, the replacement text>",
      "confidence": 0.0-1.0,
      "failure_categories_addressed": ["<category names>"],
      "is_destructive": true/false,
      "rollback_instructions": "<how to undo this change>"
    }}
  ]
}}

RULES:
- Be concrete. Generate actual SKILL.md content or Python code, not just descriptions.
- For skill_create: include FULL SKILL.md with frontmatter (name, description ≤60 chars, version 0.1.0).
- For tool_create: include complete Python function with type hints and docstring.
- For skill_patch: specify exact old_string and new_string for targeted replacement.
- Don't propose tools/skills that already exist (check the lists above).
- Don't propose changes that would break existing functionality.
- Prioritize the highest-confidence failure categories first.
"""
    return prompt


def _parse_llm_proposals(response: str) -> List[ImprovementProposal]:
    """Parse LLM-generated proposals from JSON response."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM proposals")
                return []
        else:
            return []

    proposals = []
    for p in data.get("proposals", []):
        try:
            action_type = ImprovementActionType(p.get("action_type", "skill_create"))
        except ValueError:
            action_type = ImprovementActionType.SKILL_CREATE

        proposals.append(ImprovementProposal(
            action_type=action_type,
            target=p.get("target", ""),
            description=p.get("description", ""),
            rationale=p.get("rationale", ""),
            content=p.get("content", ""),
            file_path=p.get("file_path", ""),
            old_string=p.get("old_string", ""),
            new_string=p.get("new_string", ""),
            confidence=float(p.get("confidence", 0.5)),
            failure_categories_addressed=p.get("failure_categories_addressed", []),
            is_destructive=bool(p.get("is_destructive", False)),
            requires_approval=action_type.value not in ("skill_create", "skill_patch"),
            rollback_instructions=p.get("rollback_instructions", ""),
        ))
    return proposals


def _deduplicate_proposals(proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
    """Remove duplicate proposals (same action_type + target)."""
    seen: set = set()
    unique = []
    for p in proposals:
        key = (p.action_type.value, p.target)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Real content generators — no templates, no placeholders
# ---------------------------------------------------------------------------


def _skill_frontmatter(name: str, desc: str, tags: str = "Verification") -> str:
    return f"---\nname: {name}\ndescription: {desc}\nversion: 0.1.0\nauthor: Hermes\nmetadata:\n  hermes:\n    tags: [{tags}]\n---"

def _generate_verification_skill(task_name: str, failed_checks: List[Dict[str, Any]]) -> str:
    checks_desc = "\n".join(
        f"  - {c.get('type', 'unknown')}: {c.get('detail', 'verification failed')[:100]}"
        for c in failed_checks[:5]
    )
    return _skill_frontmatter("verify-before-complete", "Always verify task completion before declaring done.", "Verification, Quality, Safety") + f"""

# Verify Before Complete

Prevents premature task completion. When the agent believes a task is done,
it MUST run every verification step before declaring success.

## When to Use

- After applying a code fix
- After creating or modifying files
- After running deployment commands
- Whenever a task has defined success criteria
- The agent is about to say "Done" or "Fixed"

## How to Run

1. Identify the task's success criteria. If the task "{task_name}" was defined
   with specific checks, those checks ARE the verification.
2. Run every check before declaring completion:
   - Use the `terminal` tool to execute test commands
   - Use `read_file` to verify file contents match expectations
   - Use `search_files` or `grep` via terminal to pattern-match outputs

## Procedure

1. Complete the main work (patch, write, deploy)
2. Read any success criteria defined for this task
3. Execute EACH criterion using the appropriate tool
4. Record results — if ANY criterion fails, the task is NOT done
5. Only declare completion when ALL criteria pass
6. If any criterion fails, analyze the failure and fix it before retrying

## Failed Checks That Triggered This Skill

{checks_desc}

## Pitfalls

- Never assume a command succeeded without checking its exit code
- Never assume a file was created without verifying it exists
- "I applied the patch" is NOT the same as "the bug is fixed"
- Verification must happen BEFORE declaring completion, not after

## Verification

Run: `test -f ~/.hermes/skills/verify-before-complete/SKILL.md && echo "skill installed"`
"""


def _generate_workaround_skill(task_name: str, finding: Any) -> str:
    return _skill_frontmatter(f"workaround-{task_name[:48]}", f"Workarounds for missing capabilities in {task_name}.", "Workaround, Troubleshooting") + f"""

# Workarounds for {task_name}

When the agent lacks a specific tool or capability, this skill documents
alternative approaches using existing tools.

## Problem

{finding.description}

## When to Use

- The agent needs a capability that has no corresponding tool
- A command is not available in the current environment
- An API or service is inaccessible

## How to Run

1. Identify the missing capability precisely
2. Check if any existing tool can achieve the same result:
   - `terminal` tool for shell commands, package installs, and scripts
   - `write_file` to create helper scripts that fill capability gaps
   - `web_search` to find alternative approaches
   - `read_file` to check if the capability already exists elsewhere
3. If no existing tool works, use `terminal` to install needed packages
   or write a Python script via `write_file` that implements the missing capability
4. Document the workaround in this skill for future reference

## Common Workarounds

- Missing CLI tool: install via package manager (`apt`, `pip`, `npm`)
- Missing API access: use `web_search` + `web_extract` as alternative data source
- Missing file parser: write a Python script using stdlib, invoke via `terminal`
- Missing database client: use `terminal` with `curl` against REST APIs

## Verification

Run: `test -f ~/.hermes/skills/workaround-{task_name[:48]}/SKILL.md && echo "skill installed"`
"""


def _generate_loop_detection_skill(implicated_tools: List[str], finding: Any) -> str:
    tools_list = ", ".join(implicated_tools[:5])
    return _skill_frontmatter("detect-and-break-loops", "Detect and escape repetitive tool-call patterns.", "Safety, Loop-Detection, Efficiency") + f"""

# Detect and Break Loops

When the agent calls the same tools with the same arguments repeatedly
without making progress, it's stuck in a loop. This skill teaches how
to detect and escape.

## When to Use

- The same tool is called 3+ times with similar arguments
- Progress has stalled despite multiple attempts
- The agent is cycling between the same few tools ({tools_list})
- Turn count is high but no forward progress

## How to Run

1. After every 3 tool calls, pause and evaluate:
   - Did the last action change anything?
   - Am I closer to the goal than before?
   - Have I tried this exact approach already?
2. If stuck in a loop:
   - STOP the current approach immediately
   - Switch to a DIFFERENT strategy
   - If debugging, try a completely different angle
   - If searching, broaden or narrow the search
   - If patching, revert and try a different fix

## Procedure

1. Detect: count consecutive calls to the same tool with similar arguments
2. Acknowledge: explicitly state "I am stuck in a loop with {tools_list}"
3. Pivot: choose a strategy you have NOT tried yet
4. Verify: after the pivot, confirm forward progress was made

## Failure Evidence

{finding.description}

## Pitfalls

- Loops waste tokens and time — detecting early saves both
- The same tool with DIFFERENT arguments is not necessarily a loop
- A loop can span multiple tools (A→B→A→B pattern)

## Verification

Run: `test -f ~/.hermes/skills/detect-and-break-loops/SKILL.md && echo "skill installed"`
"""


def _generate_troubleshooting_skill(tool_name: str, finding: Any) -> str:
    return _skill_frontmatter(f"troubleshoot-{tool_name[:48]}", f"Diagnose and fix common {tool_name} errors.", f"Troubleshooting, {tool_name}") + f"""

# Troubleshoot {tool_name}

Common failure modes for the `{tool_name}` tool and how to resolve them.

## When to Use

- `{tool_name}` returns an error or non-zero exit code
- Tool output is unexpected or empty
- The tool appears to hang or timeout

## Error Pattern

{finding.description}

## How to Run

1. Read the exact error message — don't guess
2. Check common causes:
   - Permission denied: use `terminal` to check file/directory permissions
   - Command not found: use `terminal` to install missing dependencies
   - Network timeout: retry with `web_search` or check connectivity
   - Invalid arguments: re-read the tool schema, verify argument types
3. Apply the fix and retry the tool
4. If the same error persists after 2 attempts, try a DIFFERENT approach

## Procedure

1. Capture the exact error message from the tool result
2. Diagnose using the common causes checklist above
3. Apply the most likely fix
4. Retry the tool call
5. If still failing, escalate: use `write_file` to create a debug script

## Pitfalls

- Don't retry the same failing call more than twice without changing something
- Check file paths for typos — the most common error
- Environment variables may differ between `terminal` and other tools

## Verification

Run: `test -f ~/.hermes/skills/troubleshoot-{tool_name[:48]}/SKILL.md && echo "skill installed"`
"""


def _generate_time_efficient_skill(task_name: str, finding: Any) -> str:
    return _skill_frontmatter(f"time-efficient-{task_name[:46]}", f"Time-efficient approach for completing {task_name}.", "Efficiency, Time-Management") + f"""

# Time-Efficient {task_name}

When tasks timeout, the strategy must change. This skill teaches the agent
to complete "{task_name}" within time constraints by prioritizing the most
impactful actions first.

## When to Use

- Task "{task_name}" exceeded its time or turn limit
- Approaching the turn budget with work remaining
- Complex task that requires many steps

## Failure Evidence

{finding.description}

## How to Run

1. Before starting, estimate the number of steps needed
2. Prioritize: most impactful actions first, nice-to-haves last
3. After each major action, check remaining turns/time
4. If approaching the limit, deliver a partial result with clear notes
   on what remains, rather than timing out with nothing

## Procedure

1. Read the task definition — identify the CRITICAL criteria vs optional ones
2. Plan: allocate turns to each criterion based on complexity
3. Execute in priority order, checking progress every 2-3 turns
4. If time is running out, deliver what you have with a summary of gaps
5. Next attempt: skip steps that already succeeded, focus on remaining gaps

## Pitfalls

- Don't spend 80% of turns on the first step — budget your turns
- Reading files is cheap — doing it upfront prevents later rework
- Running verification early catches problems before time runs out

## Verification

Run: `test -f ~/.hermes/skills/time-efficient-{task_name[:46]}/SKILL.md && echo "skill installed"`
"""
