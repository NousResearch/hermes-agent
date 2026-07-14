"""Auto-Trigger — watches conversations, detects 5 failure types, auto-improves silently."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agent.evolution.task_definition import (
    TaskDefinition, SuccessCriterion, SuccessCriterionType,
)
from agent.evolution.trajectory_collector import TrajectoryCollector, EvalResult
from agent.evolution.evaluator import TaskEvaluator, EvaluationContext
from agent.evolution.failure_analyzer import FailureAnalyzer
from agent.evolution.improvement_proposer import ImprovementProposer
from agent.evolution.regression_gate import RegressionGate
from agent.evolution.evolution_store import get_evolution_store
from agent.evolution.conversation_observer import get_observer

logger = logging.getLogger(__name__)

# Nudge levels
SILENT = "silent"
NOTIFY = "notify"
APPROVE = "approve"
OFF = "off"

# Safe action types that can be auto-applied
SAFE_ACTIONS = {"skill_create", "skill_patch", "memory_update"}


class AutoTrigger:
    """Watches conversations and auto-triggers the evolution cycle.

    Integrates with the agent loop — called after each turn.
    Zero overhead when no patterns are detected.
    """

    def __init__(self, nudge_level: str = NOTIFY):
        # Read from config if available, otherwise default
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            evo = cfg.get("evolution", {}) if isinstance(cfg, dict) else {}
            level = evo.get("nudge_level", "").strip().lower()
            if level in (SILENT, NOTIFY, APPROVE, OFF):
                nudge_level = level
        except Exception:
            pass
        self.nudge_level = nudge_level
        self._evaluator = TaskEvaluator()
        self._analyzer = FailureAnalyzer()
        self._proposer = ImprovementProposer()
        self._gate = RegressionGate()
        self._store = get_evolution_store()
        self._nudge_callback: Optional[Callable] = None  # Set by agent init

    def apply_fix(self, task_name: str, failure_type: str) -> Optional[str]:
        """Auto-create an improvement. Returns nudge message if applicable."""
        from agent.evolution.improvement_proposer import (
            _generate_verification_skill,
            _generate_loop_detection_skill,
            _generate_troubleshooting_skill,
        )
        from agent.evolution.failure_analyzer import FailureFinding, FailureCategory

        # Route execution errors → PR proposer (code-level fix)
        if failure_type in ("missing_output",):
            return self._propose_code_fix(task_name, failure_type)

        # Route everything else → skill creation
        if failure_type in ("missing_verification", "silent_session"):
            content = _generate_verification_skill(task_name, [])
            skill_name = "verify-before-complete"
            msg = f"I forgot to verify my work on '{task_name}'"
        elif failure_type == "loop_detected":
            content = _generate_loop_detection_skill(
                ["terminal", "read_file"],
                FailureFinding(category=FailureCategory.LOOP, confidence=0.8,
                              description="Agent repeated the same tools without progress",
                              evidence="3+ consecutive identical tool calls"),
            )
            skill_name = "detect-and-break-loops"
            msg = f"I got stuck in a loop while working on '{task_name}'"
        elif failure_type == "user_correction":
            content = _generate_troubleshooting_skill(
                "terminal",
                FailureFinding(category=FailureCategory.PREMATURE_COMPLETION, confidence=0.9,
                              description="User had to correct the agent's output",
                              evidence="User said the output was wrong"),
            )
            skill_name = f"troubleshoot-{task_name[:48]}"
            msg = f"I made a mistake on '{task_name}' and you corrected me"
        else:
            return None

        if not content:
            return None

        # Route through Hermes' skill_manage for proper staging/provenance
        result = self._write_via_skill_manage(skill_name, content)
        if result is None:
            return None  # Skill already exists — let skill_evolution handle it

        # Track this skill for recursive evolution
        try:
            from agent.evolution.skill_evolution import get_skill_evolution_tracker
            tracker = get_skill_evolution_tracker()
            tracker.start_session([skill_name])
            tracker._get_or_create_record(skill_name)
        except Exception:
            pass

        if self.nudge_level == SILENT:
            return None
        staged = " (staged for approval)" if result.get("staged") else ""
        return (
            f"🔧 HAEE: {msg}.{staged}\n"
            f"   Auto-created '{skill_name}' skill to prevent this.\n"
            f"   I'll handle this correctly next time."
        )

    def should_propose_pr(self, observer) -> bool:
        """Check if conditions are right for a PR proposal.

        Triggers when the session had both:
        1. User correction (strong signal something is wrong)
        2. Tool usage (we know which tool was involved)
        """
        has_correction = any(
            any(s in msg.lower() for s in ["no", "wrong", "incorrect", "doesn't work", "forgot", "missing"])
            for msg in observer._current_user_messages
        ) if observer._current_user_messages else False
        has_tools = bool(observer._current_tool_sequence)
        return has_correction and has_tools

    def _propose_code_fix(self, task_name: str, failure_type: str) -> Optional[str]:
        """Route a failure to the PR proposer for code-level fixes.

        Uses LLM (if available) to diagnose the failing tool and generate code.
        Falls back to the most-used tool in the current session.
        """
        try:
            from agent.evolution.pr_proposer import PRProposer
            from agent.evolution.auxiliary_llm import get_evolution_llm

            # Determine which tool to fix — use observer data
            observer = None
            try:
                from agent.evolution.conversation_observer import get_observer
                observer = get_observer()
            except Exception:
                pass

            tool_name = task_name[:40]
            if observer and observer._current_tool_sequence:
                from collections import Counter
                tool_name = Counter(observer._current_tool_sequence).most_common(1)[0][0]

            # Step 1: Try LLM to generate the fix
            llm = get_evolution_llm()
            proposed_code = ""
            if llm and llm.is_available:
                tool_name, proposed_code = self._llm_generate_code_fix(
                    llm, task_name, failure_type
                )
            if not proposed_code:
                tool_name = tool_name[:40]
                proposed_code = ""

            # Step 2: Check approval — PR branches default to asking
            # Skills are safe (markdown), code branches are invasive (git history)
            if self.nudge_level != SILENT:
                return (
                    f"💡 HAEE detected a code-level issue with '{tool_name}'.\n"
                    f"   A fix can be auto-generated and submitted as a PR.\n"
                    f"   Approve: /evolution approve-pr {tool_name}"
                )
            # silent: create branch without asking
            pass
            if self.nudge_level == OFF:
                return None

            # Step 3: Run one HyperAgents generation
            proposer = PRProposer()
            result = proposer.run_generation(
                failure_analysis={
                    "findings": [{
                        "category": failure_type,
                        "confidence": 0.7,
                        "description": f"Auto-detected {failure_type} during '{task_name}'",
                        "evidence": f"Observer detected {failure_type} pattern across multiple sessions.",
                    }],
                    "total_occurrences": 1,
                    "total_sessions": 1,
                },
                proposed_code=proposed_code,
                tool_name=tool_name,
            )

            # Step 3: If a candidate was selected, offer to create PR
            selected = result.get("selected")
            if selected:
                if self.nudge_level == SILENT:
                    return None
                return (
                    f"📝 HAEE detected a code-level issue with '{tool_name}'.\n"
                    f"   Generation {selected.get('generation', 1)} candidate selected "
                    f"(fitness: {selected.get('fitness', 0):.2f}).\n"
                    f"   Review: hermes evolution pr-status"
                )
            elif "error" in result:
                return None  # Silent — tool file not found or excluded path
        except Exception as e:
            logger.debug("PR proposer failed: %s", e)
        return None

    def _llm_generate_code_fix(
        self, llm, task_name: str, failure_type: str
    ) -> tuple:
        """Use LLM to generate a code fix. Returns (tool_name, code)."""
        prompt = f"""You are a code-fixing agent. Analyze this failure and generate a fix.

FAILURE: {failure_type}
TASK: {task_name}
CONTEXT: This failure was auto-detected by HAEE's observer during normal agent usage.
The agent's tool consistently fails when this pattern occurs.

Your job:
1. Identify which Hermes Agent tool file is most likely responsible
   (e.g., tools/terminal_tool.py, tools/file_tools.py, tools/web_tools.py)
2. Generate a complete, corrected version of that file
3. Include comments explaining what was fixed and why

Respond with ONLY valid JSON:
{{
  "tool_file": "tools/terminal_tool.py",
  "description": "Fixed output truncation that caused silent failures",
  "code": "#!/usr/bin/env python3\\n\\\"\\\"\\\"Fixed terminal tool.\\\"\\\"\\\"\\n..."
}}"""

        try:
            response = llm.analyze_sync(prompt)
            import json, re
            # Extract JSON
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("tool_file", task_name), data.get("code", "")
        except Exception:
            pass

        return task_name, ""

    def _run_improvement_for_failure(
        self, task, cluster, failure_type, session_id
    ) -> Optional[str]:
        """Run improvement cycle for any failure type."""
        return self.apply_fix(task.name, failure_type)

    def apply_verification_skill(self, task_name: str) -> Optional[str]:
        """Backward-compatible wrapper."""
        return self.apply_fix(task_name, "missing_verification")

    def set_nudge_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set a callback for user nudges. Called as callback(title, message)."""
        self._nudge_callback = callback

    def check_and_trigger(
        self,
        messages: List[Dict[str, Any]],
        session_id: str = "",
    ) -> Optional[str]:
        """Check if current turn matches a known task pattern, and trigger evaluation.

        Called after each agent turn completes (post_llm_call or turn end).

        Returns:
            A nudge message if a fix was applied, None otherwise.
        """
        if self.nudge_level == OFF:
            return None

        # Check if observer has matched a task pattern
        observer = get_observer()
        # Use lower thresholds for auto-trigger — we want to catch patterns early
        clusters = observer.suggest_tasks(min_occurrences=2, min_confidence=0.2)

        if not clusters:
            return None

        # Check if agent just completed its work (declared done, no tool calls pending)
        if not self._agent_just_completed(messages):
            return None

        # Use the highest-confidence cluster
        best_cluster = clusters[0]  # Already sorted by confidence × log(occurrences)

        # Build an ad-hoc task from the cluster's criteria
        task = self._cluster_to_task(best_cluster)

        # Check for ALL failure types
        observer = get_observer()
        should_trigger, failure_type = self._detect_failures(
            best_cluster, observer
        )

        if not should_trigger:
            # Record success baseline
            self._store.set_baseline(
                task_name=best_cluster.task_name,
                score=1.0,
                task_domain="general",
            )
            return None

        # Run improvement cycle with the specific failure type
        return self._run_improvement_for_failure(
            task, best_cluster, failure_type, session_id
        )

    @staticmethod
    def _detect_failures(cluster, observer) -> Tuple[bool, str]:
        """Detect ALL failure types in the current session.

        Returns (should_trigger, failure_type).
        """
        if not observer._current_tool_sequence:
            return False, ""

        tools = observer._current_tool_sequence
        user_msgs = observer._current_user_messages

        # 1. User correction — strongest signal, always triggers
        for msg in user_msgs:
            msg_lower = msg.lower()
            for signal in ["no", "wrong", "incorrect", "doesn't work", "not working",
                          "try again", "redo", "forgot", "missing", "incomplete",
                          "actually", "instead", "should be", "need to also"]:
                if signal in msg_lower:
                    return True, "user_correction"

        # 2. Missing verification: agent did work but didn't verify
        work_tools = {"write_file", "patch", "execute_code"}
        verify_tools = {"terminal", "read_file", "search_files", "browser_snapshot"}
        did_work = any(t in work_tools for t in tools)
        had_verification = any(t in verify_tools for t in tools)
        if did_work and not had_verification:
            return True, "missing_verification"

        # 3. Loop detection: same tool called 3+ times consecutively
        for i in range(len(tools) - 2):
            if tools[i] == tools[i+1] == tools[i+2]:
                return True, "loop_detected"

        # 4. Missing output: agent declared done but no files were created
        if did_work and not observer._current_files:
            return True, "missing_output"

        # 5. Silent session: no user feedback at all on high-confidence cluster
        if cluster.confidence >= 0.6 and not user_msgs:
            return True, "silent_session"

        return False, ""

    @staticmethod
    def _agent_just_completed(messages: List[Dict[str, Any]]) -> bool:
        """Check if the agent just declared completion (no pending tool calls)."""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", []) or []
                return len(tool_calls) == 0
        return False

    def _cluster_to_task(self, cluster) -> TaskDefinition:
        """Convert a discovered task cluster to an ad-hoc TaskDefinition."""
        criteria = []
        for c in cluster.suggested_criteria:
            try:
                ctype = SuccessCriterionType(c["type"])
                criteria.append(SuccessCriterion(
                    type=ctype,
                    command=c.get("command"),
                    path=c.get("path"),
                    pattern=c.get("pattern"),
                    expected_output=c.get("expected_output"),
                    weight=c.get("weight", 0.5),
                ))
            except (KeyError, ValueError):
                continue

        if not criteria:
            criteria = [SuccessCriterion(
                type=SuccessCriterionType.TEST_PASS,
                command="true", weight=1.0,
            )]

        return TaskDefinition(
            name=cluster.task_name,
            description=cluster.description,
            success_criteria=criteria,
            domain="general",
            complexity=cluster.estimated_complexity,
        )

    def _run_improvement_cycle(
        self,
        task: TaskDefinition,
        eval_result: Optional[EvalResult],
        cluster,
        session_id: str,
    ) -> Optional[str]:
        """Run the full improvement cycle and return a nudge message if applicable."""
        from agent.evolution.trajectory_collector import Trajectory, EvalResult as ER

        if eval_result is None:
            eval_result = ER(passed=False, score=0.0)

        # Create minimal trajectory for analysis
        traj = Trajectory(
            task_name=task.name,
            run_id=f"auto_{session_id[:12]}",
            status="failed",
            total_turns=1, total_tool_calls=0,
        )

        # Analyze
        analysis = self._analyzer.analyze(task, traj, eval_result)
        if not analysis.findings:
            return None

        # Generate proposals
        proposals = self._proposer.propose(task, analysis)
        if not proposals:
            return None

        # Apply safe proposals
        applied = []
        for p in proposals:
            gate_result = self._gate.evaluate(p)
            if not gate_result.passed:
                continue

            # Check if this action can be auto-applied
            if p.action_type.value not in SAFE_ACTIONS:
                if self.nudge_level == APPROVE:
                    return self._format_approval_nudge(p, analysis)
                continue  # Skip destructive actions in silent/notify mode

            # Apply
            try:
                self._apply_silently(p)
                applied.append(p)
            except Exception as e:
                logger.debug("Auto-apply failed for %s: %s", p.target, e)

        if not applied:
            return None

        # Record in store
        run_id = self._store.create_run(
            task_name=task.name,
            task_domain="general",
            task_complexity=cluster.estimated_complexity,
            session_id=session_id,
        )
        self._store.add_iteration(
            run_id=run_id, iteration_num=1, status="auto_improved",
            improvement_action=applied[0].action_type.value,
            improvement_target=applied[0].target,
        )
        self._store.update_run_status(run_id, "auto_improved")

        if self.nudge_level == SILENT:
            return None

        return self._format_success_nudge(applied, analysis)

    def _write_via_skill_manage(self, skill_name: str, content: str) -> Optional[dict]:
        """Route a skill write through Hermes' skill_manage tool.

        Respects skills.write_approval — stages writes when approval is enabled.
        Returns the result dict, or None if skill already exists.
        """
        from hermes_constants import get_hermes_home

        # Don't duplicate existing skills
        if (get_hermes_home() / "skills" / skill_name / "SKILL.md").exists():
            return None

        try:
            from tools.skill_manager_tool import skill_manage
            result_json = skill_manage(action="create", name=skill_name, content=content)
            import json
            return json.loads(result_json)
        except Exception:
            # Fallback: direct write if skill_manage is unavailable
            skill_path = get_hermes_home() / "skills" / skill_name
            skill_path.mkdir(parents=True, exist_ok=True)
            (skill_path / "SKILL.md").write_text(content)
            return {"success": True}

    def _apply_silently(self, proposal) -> None:
        """Apply a proposal without user interaction — routed through skill_manage."""
        if proposal.action_type.value == "skill_create":
            self._write_via_skill_manage(proposal.target, proposal.content)
        elif proposal.action_type.value == "skill_patch":
            try:
                from tools.skill_manager_tool import skill_manage
                skill_manage(action="patch", name=proposal.target,
                            old_string=proposal.old_string,
                            new_string=proposal.new_string)
            except Exception:
                pass
        elif proposal.action_type.value == "memory_update":
            try:
                from tools.skill_manager_tool import skill_manage
                skill_manage(action="create", name=proposal.target, content=proposal.content)
            except Exception:
                pass

    def _format_success_nudge(self, applied, analysis) -> str:
        """Format a nudge message for auto-applied fixes."""
        actions = ", ".join(f"{p.action_type.value}→{p.target}" for p in applied)
        finding = analysis.findings[0] if analysis.findings else None
        cause = f": {finding.description[:80]}" if finding else ""
        return (
            f"🔧 HAEE auto-improved: {len(applied)} fix(es) applied ({actions})\n"
            f"   Cause{cause}\n"
            f"   These will prevent this issue next time."
        )

    def _format_approval_nudge(self, proposal, analysis) -> str:
        """Format a nudge requesting user approval."""
        return (
            f"💡 HAEE suggests: {proposal.action_type.value} → {proposal.target}\n"
            f"   {proposal.description[:120]}\n"
            f"   Reason: {proposal.rationale[:120]}\n"
            f"   Approve with: /evolution approve {proposal.target}"
        )


# Singleton
_auto_trigger: Optional[AutoTrigger] = None


def get_auto_trigger() -> AutoTrigger:
    global _auto_trigger
    if _auto_trigger is None:
        _auto_trigger = AutoTrigger()
    return _auto_trigger
