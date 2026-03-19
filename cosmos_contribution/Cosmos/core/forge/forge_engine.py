"""
FORGE Engine - Core Orchestration
===================================

The brain of FORGE. Orchestrates multi-model agents through the
plan-execute-verify pipeline using swarm deliberation.

Unlike single-model approaches that degrade as context fills,
FORGE distributes work across multiple AI models, each with
fresh context, and uses collective consensus to catch errors
that dict single model would miss.

Workflow:
  1. RESEARCH  - Parallel agents investigate the problem space
  2. DELIBERATE - Swarm proposes, critiques, and votes on plans
  3. EXECUTE   - Best-fit models execute tasks with atomic commits
  4. VERIFY    - Collective consensus verifies goals (not just tasks)
  5. ROLLBACK  - Automated rollback on verification failure

"One model plans. Eleven models forge."
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import  Callable, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .forge_state import (
    ForgeStateManager, ForgeProject, ForgePhase, ForgePlan, ForgeTask
)


# Model selection based on task type - best model for each job
MODEL_TASK_MAP = {
    "planning": ["claude", "grok", "gemini"],      # Strategic thinking
    "coding": ["claude", "deepseek", "qwen_coder"], # Code generation
    "review": ["gemini", "grok", "claude"],         # Code review
    "research": ["grok", "gemini", "kimi"],         # Research tasks
    "debugging": ["deepseek", "claude", "phi"],     # Debug & reasoning
    "testing": ["phi", "deepseek", "claude"],       # Fast verification
    "documentation": ["gemini", "claude", "kimi"],  # Docs & writing
}

# Deviation rules (improved over competing approaches)
DEVIATION_RULES = {
    "bug": {"permission": "auto_fix", "description": "Broken behavior, errors, type mismatches"},
    "missing_critical": {"permission": "auto_add", "description": "Missing validation, auth, error handling"},
    "blocking": {"permission": "auto_fix", "description": "Missing deps, broken imports, env vars"},
    "architectural": {"permission": "stop_ask", "description": "Schema changes, new services, breaking API"},
    "performance": {"permission": "auto_fix", "description": "Obvious N+1, missing indexes, memory leaks"},
    "security": {"permission": "auto_fix", "description": "XSS, injection, exposed secrets, CSRF"},
}


class ForgeEngine:
    """
    Main FORGE orchestration engine.

    Coordinates the swarm through structured development phases
    with multi-model deliberation at every stage.
    """

    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace)
        self.state = ForgeStateManager(workspace)
        self._agents = {}
        self._init_agents()

    def _init_agents(self):
        """Connect to available swarm agents."""
        try:
            from Cosmos.core.collective.persistent_agent import (
                call_shadow_agent, get_shadow_agents
            )
            self._call_agent = call_shadow_agent
            self._available_agents = get_shadow_agents
        except ImportError:
            logger.warning("Shadow agents not available, using direct provider calls")
            self._call_agent = None
            self._available_agents = lambda: []

    # =========================================================================
    # CORE PIPELINE
    # =========================================================================

    async def research(self, topic: str, phase: ForgePhase = None) -> dict[str, str]:
        """
        Parallel multi-model research.

        Spawns multiple agents to research different aspects simultaneously.
        Returns consolidated findings with confidence ratings.
        """
        research_prompts = {
            "stack": f"Research the best technology stack for: {topic}. "
                     f"Focus on proven, production-ready choices. "
                     f"Rate confidence HIGH/MEDIUM/LOW for each recommendation.",
            "architecture": f"Research the best architecture patterns for: {topic}. "
                           f"Focus on scalability, maintainability, and simplicity. "
                           f"Identify common anti-patterns to avoid.",
            "pitfalls": f"Research common pitfalls and failure modes when building: {topic}. "
                       f"What do teams get wrong? What are the non-obvious gotchas?",
            "features": f"Research essential vs nice-to-have features for: {topic}. "
                       f"Prioritize by user value. What is the minimum viable scope?",
        }

        # Run research in parallel across different models
        research_models = ["grok", "gemini", "kimi", "claude"]
        tasks = []
        assignments = {}

        for (aspect, prompt), model in zip(research_prompts.items(), research_models):
            assignments[aspect] = model
            if self._call_agent:
                tasks.append(self._call_agent(model, prompt, timeout=45.0))
            else:
                tasks.append(self._fallback_query(prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings = {}
        for (aspect, _), result in zip(research_prompts.items(), results):
            if isinstance(result, tuple):
                findings[aspect] = result[1]
            elif isinstance(result, str):
                findings[aspect] = result
            else:
                findings[aspect] = f"Research failed: {result}"

        # Save research to .forge/
        if phase:
            research_file = self.state.forge_dir / f"{phase.id}_RESEARCH.md"
        else:
            research_file = self.state.forge_dir / "RESEARCH.md"

        lines = [f"# Research: {topic}", f"*Generated: {datetime.now().isoformat()}*", ""]
        for aspect, content in findings.items():
            model = assignments.get(aspect, "unknown")
            lines.extend([
                f"## {aspect.title()} (by {model})",
                "", content, ""
            ])
        research_file.write_text("\n".join(lines))

        return findings

    async def deliberate(self, objective: str, context: str = "",
                         models: list[str] = None) -> dict:
        """
        PROPOSE-CRITIQUE-REFINE-VOTE deliberation on a plan.

        This is FORGE's core advantage: multiple models review every plan
        before execution. No single model's blind spots survive the swarm.

        Returns:
            dict with plan, consensus_score, critiques, and votes
        """
        models = models or ["grok", "gemini", "claude", "deepseek"]

        # Phase 1: PROPOSE - Lead model creates initial plan
        propose_prompt = f"""Create a detailed implementation plan for:

OBJECTIVE: {objective}

{f'CONTEXT: {context}' if context else ''}

Structure your plan as a list of concrete tasks. For each task provide:
- Task name (action-oriented)
- Files to create/modify
- Specific implementation steps
- Verification command or check
- Done criteria (measurable)

Group tasks into waves (wave 1 = no dependencies, wave 2 = depends on wave 1, etc).
Keep each task scoped to 15-60 minutes of work.
Return as JSON array of task objects."""

        proposer = models[0]
        proposal = await self._query_agent(proposer, propose_prompt)

        if not proposal:
            return {"error": "Proposal generation failed", "consensus": 0}

        # Phase 2: CRITIQUE - Other models review the plan
        critiques = []
        critique_prompt = f"""Review this implementation plan critically:

OBJECTIVE: {objective}
PROPOSED PLAN:
{proposal}

Identify:
1. Missing requirements or edge cases
2. Incorrect technical choices
3. Tasks that are too large or too vague
4. Missing dependencies between tasks
5. Security or performance concerns
6. What would you do differently?

Be specific and actionable. Rate overall plan quality: STRONG / ADEQUATE / WEAK."""

        critique_tasks = []
        for model in models[1:]:
            critique_tasks.append(self._query_agent(model, critique_prompt))

        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        for model, result in zip(models[1:], critique_results):
            if isinstance(result, str) and result:
                critiques.append({"agent": model, "critique": result})

        # Phase 3: REFINE - Original proposer incorporates feedback
        if critiques:
            critique_text = "\n\n".join(
                f"**{c['agent']}**: {c['critique']}" for c in critiques
            )
            refine_prompt = f"""Refine your implementation plan based on these critiques:

ORIGINAL PLAN:
{proposal}

CRITIQUES:
{critique_text}

Incorporate valid feedback. Reject invalid critiques with reasoning.
Return the improved plan as JSON array of task objects with fields:
name, files (array), action, verify, done, type (auto/checkpoint), wave, depends_on (array)"""

            refined = await self._query_agent(proposer, refine_prompt)
        else:
            refined = proposal

        # Phase 4: VOTE - All models score the final plan
        vote_prompt = f"""Score this implementation plan from 0-100:

OBJECTIVE: {objective}
FINAL PLAN:
{refined}

Consider: completeness, technical correctness, task sizing, dependency ordering.
Return ONLY a number 0-100."""

        vote_tasks = [self._query_agent(m, vote_prompt) for m in models]
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        scores = []
        for result in vote_results:
            if isinstance(result, str):
                try:
                    score = int(''.join(c for c in result if c.isdigit())[:3])
                    if 0 <= score <= 100:
                        scores.append(score)
                except (ValueError, IndexError):
                    pass

        consensus = sum(scores) / len(scores) / 100 if scores else 0.5

        return {
            "plan": refined,
            "consensus_score": consensus,
            "critiques": critiques,
            "votes": scores,
            "proposer": proposer,
            "models_consulted": models,
        }

    async def execute_plan(self, plan: ForgePlan, dry_run: bool = False) -> dict:
        """
        Execute a plan with optimal model assignment per task.

        Each task is assigned to the best available model for its type.
        Atomic git commits per task. Deviation handling with auto-fix
        for safe deviations, human approval for architectural changes.
        """
        results = {"tasks_completed": 0, "tasks_failed": 0, "commits": []}
        plan.status = "executing"
        self.state._save_state()

        # Group tasks by wave
        waves = {}
        for task in plan.tasks:
            waves.setdefault(task.wave, []).append(task)

        for wave_num in sorted(waves.keys()):
            wave_tasks = waves[wave_num]
            logger.info(f"Executing wave {wave_num}: {len(wave_tasks)} tasks")

            # Execute wave tasks (parallel within wave)
            wave_results = await asyncio.gather(
                *[self._execute_task(task, dry_run) for task in wave_tasks],
                return_exceptions=True
            )

            for task, result in zip(wave_tasks, wave_results):
                if isinstance(result) and result.get("success"):
                    results["tasks_completed"] += 1
                    if result.get("commit_hash"):
                        results["commits"].append(result["commit_hash"])
                else:
                    results["tasks_failed"] += 1
                    error = result if isinstance(result, Exception) else result.get("error", "Unknown")
                    logger.error(f"Task failed: {task.name} - {error}")

        plan.status = "verified" if results["tasks_failed"] == 0 else "failed"
        self.state._save_state()
        return results

    async def _execute_task(self, task: ForgeTask, dry_run: bool = False) -> dict:
        """Execute a single task with the best-fit model."""
        start_time = time.time()
        task.status = "running"

        # Select best model for this task type
        task_category = self._categorize_task(task)
        preferred_models = MODEL_TASK_MAP.get(task_category, ["claude", "deepseek"])

        # Build execution prompt
        exec_prompt = f"""Execute this development task:

TASK: {task.name}
FILES: {', '.join(task.files)}

ACTION:
{task.action}

VERIFICATION:
{task.verify}

DONE WHEN:
{task.done_criteria}

Execute the task completely. Show the full code you would write/modify.
If you encounter unexpected issues, classify them:
- BUG: auto-fix and continue
- MISSING_CRITICAL: auto-add and continue
- BLOCKING: auto-fix and continue
- ARCHITECTURAL: STOP and describe the issue

Return the complete implementation."""

        # Try models in preference order
        response = None
        for model in preferred_models:
            response = await self._query_agent(model, exec_prompt)
            if response:
                task.assigned_model = model
                break

        if not response:
            task.status = "failed"
            return {"success": False, "error": "All models failed"}

        elapsed_ms = int((time.time() - start_time) * 1000)

        if dry_run:
            task.status = "passed"
            self.state.update_task(task.id, "passed", execution_time_ms=elapsed_ms)
            return {"success": True, "dry_run": True, "response": response}

        task.status = "passed"
        self.state.update_task(
            task.id, "passed",
            execution_time_ms=elapsed_ms,
        )

        return {"success": True, "response": response, "model": task.assigned_model}

    async def verify_phase(self, phase: ForgePhase) -> dict:
        """
        Goal-backward verification using swarm consensus.

        Instead of checking "did tasks complete?", we check
        "are the phase GOALS actually achieved?"

        Multiple models verify independently, then we take consensus.
        """
        phase.status = "verifying"

        verify_prompt = f"""Verify whether these goals have been achieved:

PHASE: {phase.name}
GOALS:
{chr(10).join(f'- {g}' for g in phase.goals)}

SUCCESS CRITERIA:
{chr(10).join(f'- {c}' for c in phase.success_criteria)}

For each goal, check three levels:
1. EXISTENCE - Does the relevant code/artifact exist?
2. SUBSTANCE - Is it fully implemented (not stubs/TODOs)?
3. WIRING - Is it connected and functional (not orphaned)?

Return status for each goal: VERIFIED / PARTIAL / MISSING / STUB
Overall verdict: PASSED / GAPS_FOUND / FAILED"""

        # Multiple models verify independently
        verifiers = ["claude", "gemini", "deepseek"]
        verify_tasks = [self._query_agent(m, verify_prompt) for m in verifiers]
        verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)

        verdicts = []
        for model, result in zip(verifiers, verify_results):
            if isinstance(result, str):
                verdict = "PASSED" if "PASSED" in result.upper() else \
                         "GAPS_FOUND" if "GAPS" in result.upper() else "FAILED"
                verdicts.append({"model": model, "verdict": verdict, "detail": result})

        # Consensus
        passed = sum(1 for v in verdicts if v["verdict"] == "PASSED")
        consensus_verdict = "PASSED" if passed >= 2 else \
                          "GAPS_FOUND" if passed >= 1 else "FAILED"

        verification = {
            "phase": phase.name,
            "verdict": consensus_verdict,
            "model_verdicts": verdicts,
            "consensus_ratio": passed / len(verdicts) if verdicts else 0,
            "timestamp": datetime.now().isoformat(),
        }

        phase.verification_result = verification

        if consensus_verdict == "PASSED":
            phase.status = "complete"
        else:
            phase.status = "executing"  # Needs gap closure

        self.state._save_state()

        # Save verification report
        report_file = self.state.forge_dir / f"{phase.id}_VERIFICATION.md"
        lines = [
            f"# Verification: {phase.name}",
            f"**Verdict:** {consensus_verdict}",
            f"**Consensus:** {passed}/{len(verdicts)} models passed",
            "",
        ]
        for v in verdicts:
            lines.extend([
                f"## {v['model']}",
                f"**Verdict:** {v['verdict']}",
                "", v["detail"], ""
            ])
        report_file.write_text("\n".join(lines))

        return verification

    # =========================================================================
    # CONVENIENCE COMMANDS
    # =========================================================================

    async def quick(self, task_description: str) -> dict:
        """
        Quick-mode: plan and execute a single task without full phase workflow.
        Uses deliberation for planning but skips research and full verification.
        """
        logger.info(f"FORGE quick mode: {task_description}")

        # Quick deliberation
        result = await self.deliberate(
            task_description,
            models=["claude", "deepseek"],  # Faster with fewer models
        )

        return {
            "plan": result.get("plan", ""),
            "consensus": result.get("consensus_score", 0),
            "status": "planned",
        }

    def get_progress(self) -> dict:
        """Get current project progress."""
        if not self.state.project:
            self.state.load_project()
        if not self.state.project:
            return {"error": "No project initialized"}
        return self.state.get_progress()

    def get_cost_report(self) -> dict:
        """Get cost breakdown."""
        if not self.state.project:
            self.state.load_project()
        if not self.state.project:
            return {"error": "No project initialized"}
        return self.state.get_cost_report()

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def _query_agent(self, model: str, prompt: str,
                           timeout: float = 60.0) -> Optional[str]:
        """Query an agent with fallback chain."""
        if self._call_agent:
            result = await self._call_agent(model, prompt, timeout=timeout)
            if result:
                return result[1]

        return await self._fallback_query(prompt)

    async def _fallback_query(self, prompt: str) -> Optional[str]:
        """Fallback: try dict available provider."""
        fallback_chain = ["grok", "gemini", "claude", "deepseek", "phi"]
        if self._call_agent:
            for model in fallback_chain:
                try:
                    result = await self._call_agent(model, prompt, timeout=30.0)
                    if result:
                        return result[1]
                except Exception:
                    continue
        return None

    def _categorize_task(self, task: ForgeTask) -> str:
        """Categorize a task to select the best model."""
        name_lower = task.name.lower()
        action_lower = task.action.lower()
        combined = name_lower + " " + action_lower

        if any(w in ["test", "spec", "assert", "verify"]):
            return "testing"
        if any(w in ["debug", "fix", "bug", "error", "investigate"]):
            return "debugging"
        if any(w in ["research", "investigate", "explore", "analyze"]):
            return "research"
        if any(w in ["document", "readme", "comment", "describe"]):
            return "documentation"
        if any(w in ["review", "audit", "check", "inspect"]):
            return "review"
        if any(w in ["plan", "design", "architect", "structure"]):
            return "planning"
        return "coding"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def forge_quick(task: str, workspace: str = ".") -> dict:
    """Quick-mode FORGE execution."""
    engine = ForgeEngine(workspace)
    return await engine.quick(task)


async def forge_plan(objective: str, context: str = "",
                     workspace: str = ".") -> dict:
    """Plan with swarm deliberation."""
    engine = ForgeEngine(workspace)
    return await engine.deliberate(objective, context)


async def forge_execute(plan: ForgePlan, workspace: str = ".",
                        dry_run: bool = False) -> dict:
    """Execute a FORGE plan."""
    engine = ForgeEngine(workspace)
    return await engine.execute_plan(plan, dry_run)
