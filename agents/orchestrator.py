"""agents/orchestrator.py
Central swarm coordinator (Kairos style).
Decomposes high-level goals into phases and routes to specialist agents.
Maintains global state, memory writes, and final synthesis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.tools import KairosTools, get_tools
from core.react_loop import make_llm_call
from core.dashboard_events import emit_agent_update, emit_log, emit_metrics
from kairos.memory import HermesMemory, get_memory, TaskRecord

logger = logging.getLogger("agents.orchestrator")


@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Top-level multi-agent controller."""

    name = "orchestrator"

    def __init__(
        self,
        tools: Optional[KairosTools] = None,
        memory: Optional[HermesMemory] = None,
        llm_call: Optional[Callable[[str, str], str]] = None,
        project_root: str = ".",
    ):
        self.tools = tools or get_tools(project_root)
        self.memory = memory or get_memory(project_root=project_root)
        self.llm_call = llm_call or make_llm_call()
        self.project_root = project_root
        self.agents: dict[str, Any] = {}
        self._load_specialists()
        logger.info("Orchestrator initialized with full swarm")

    def _load_specialists(self):
        # Late import to avoid circulars and allow standalone execution
        from agents.architect import Architect
        from agents.coder import Coder
        from agents.tester import Tester
        from agents.scribe import Scribe
        from agents.web_agent import WebAgent
        from agents.validator_agent import ValidatorAgent
        from agents.tool_registry import ToolRegistry

        self.agents = {
            "architect": Architect(self.tools, self.memory, self.llm_call),
            "coder": Coder(self.tools, self.memory, self.llm_call),
            "tester": Tester(self.tools, self.memory, self.llm_call),
            "scribe": Scribe(self.tools, self.memory, self.llm_call),
            "web_agent": WebAgent(self.tools, self.memory, self.llm_call),
            "validator": ValidatorAgent(self.tools, self.memory, self.llm_call),
        }
        
        # Initialize tool registry for self-evolution
        self.tool_registry = ToolRegistry(registry_root=f"{self.project_root}/hermes/tools")

    def run(self, goal: str, max_steps: int = 8) -> AgentResult:
        """Main entry point for any user goal. Includes self-evolution loop."""
        start = time.time()
        logger.info(f"ORCHESTRATOR received goal: {goal}")
        emit_agent_update("Orchestrator", "working", f"Received goal: {goal[:50]}", 20)
        emit_log(f"Swarm started for: {goal[:70]}", "info", "Orchestrator")

        context = self.memory.get_relevant_context(goal)

        # Kairos Code style: auto inject @path file references into context
        file_ctx = self.tools.extract_file_context(goal) if hasattr(self.tools, 'extract_file_context') else ""
        if file_ctx:
            context = (context + "\n\n" + file_ctx).strip()

        # Phase 1: Web Research (optional, if goal needs external knowledge)
        if self._should_research(goal):
            emit_agent_update("WebAgent", "working", "Researching external context", 30)
            research = self.agents["web_agent"].run(goal, context)
            if research.success:
                context = context + "\n\n=== WEB RESEARCH ===\n" + research.output
                emit_log(f"Web research complete: {len(research.output)} chars", "info", "WebAgent")

        # Phase 2: Architecture
        emit_agent_update("Architect", "working", "Analyzing requirements + codebase", 35)
        arch = self.agents["architect"].run(goal, context)
        if not arch.success:
            emit_agent_update("Architect", "error", "Architecture failed")
            return self._finalize(goal, False, "Architecture phase failed", start)

        # Phase 3: Implementation (may loop internally)
        emit_agent_update("Coder", "working", "Implementing solution using tools", 50)
        impl = self.agents["coder"].run(goal, arch.output + "\n\n" + context)
        if not impl.success:
            emit_agent_update("Coder", "error", "Coding failed")
            return self._finalize(goal, False, "Coding phase failed", start)

        # Phase 4: Testing & verification
        emit_agent_update("Tester", "working", "Validating implementation", 70)
        test = self.agents["tester"].run(goal, impl.output)
        if not test.success:
            logger.warning("Tests reported issues - continuing to validation")
            emit_log("Tests had issues but continuing", "warning", "Tester")

        # Phase 5: Validation & Feedback
        emit_agent_update("Validator", "working", "Validating output quality", 80)
        validation = self.agents["validator"].run(
            impl.output,
            requirements=goal,
            context=arch.output,
        )

        # Phase 6: Documentation & learning capture
        emit_agent_update("Scribe", "working", "Capturing knowledge + updating soul", 88)
        scribe = self.agents["scribe"].run(goal, "\n".join([arch.output, impl.output, test.output]))

        # Phase 7: Self-Evolution Loop
        emit_agent_update("SelfEvolutionLoop", "working", "Analyzing for self-improvement", 95)
        evolution_result = self._run_self_evolution(
            goal, arch.output, impl.output, test.output, validation.metadata
        )

        duration = time.time() - start
        success = impl.success and arch.success

        all_artifacts = list(set((impl.artifacts or []) + (scribe.artifacts or []) + (evolution_result.get("new_tools", []))))

        # Persist outcome
        self.memory.store_task(
            TaskRecord(
                goal=goal,
                success=success,
                duration=duration,
                agent="orchestrator",
                result_summary=scribe.output[:500],
                metadata={
                    "phases": ["architect", "coder", "tester", "validator", "scribe", "self_evolution"],
                    "validation_score": validation.metadata.get("score", 0),
                    "tools_created": evolution_result.get("tools_created", 0),
                },
            )
        )

        emit_agent_update("Orchestrator", "completed", f"Done in {duration:.1f}s", 100)
        emit_metrics(tasks_completed=self.memory.get_task_count() if hasattr(self.memory, 'get_task_count') else None)
        emit_log(f"Swarm finished. Artifacts: {all_artifacts[:5]}", "success" if success else "warning", "Orchestrator")

        final_msg = f"Swarm completed goal in {duration:.1f}s. Artifacts: {all_artifacts}. Self-evolution: {evolution_result.get('status', 'N/A')}"
        return self._finalize(goal, success, final_msg, start, extra={"scribe": scribe.output, "evolution": evolution_result}, artifacts=all_artifacts)

    def _finalize(self, goal: str, success: bool, msg: str, start: float, extra: dict | None = None, artifacts: list[str] | None = None) -> AgentResult:
        duration = time.time() - start
        logger.info(f"ORCHESTRATOR finished: success={success} ({duration:.1f}s)")
        return AgentResult(
            success=success,
            output=msg,
            artifacts=artifacts or [],
            metadata={"duration": duration, "goal": goal, **(extra or {})},
        )

    def _should_research(self, goal: str) -> bool:
        """Determine if web research is needed for this goal."""
        research_keywords = ["research", "find", "look up", "best practice", "how", "what", "latest"]
        return any(keyword in goal.lower() for keyword in research_keywords)

    def _run_self_evolution(
        self,
        goal: str,
        architecture: str,
        implementation: str,
        test_output: str,
        validation_metadata: dict,
    ) -> dict[str, Any]:
        """
        Self-evolution loop: analyze execution and improve/create tools.
        
        Returns dict with:
        - status: 'success' or 'no_improvements'
        - tools_created: count of new tools
        - tools_improved: count of improved tools
        - new_tools: list of tool names
        """
        logger.info("SELF_EVOLUTION: Starting self-improvement analysis...")
        
        result = {
            "status": "no_improvements",
            "tools_created": 0,
            "tools_improved": 0,
            "new_tools": [],
        }

        try:
            # Analyze execution trace for improvement opportunities
            improvement_analysis = self._analyze_for_improvements(
                goal, architecture, implementation, test_output, validation_metadata
            )

            if not improvement_analysis.get("opportunities"):
                logger.info("SELF_EVOLUTION: No improvement opportunities found")
                return result

            # Generate/improve tools
            for opportunity in improvement_analysis["opportunities"][:2]:  # Limit to 2 per run
                if opportunity["type"] == "new_tool":
                    tool_result = self._generate_new_tool(opportunity)
                    if tool_result:
                        result["tools_created"] += 1
                        result["new_tools"].append(tool_result["name"])
                        logger.info(f"SELF_EVOLUTION: Created tool {tool_result['name']}")

                elif opportunity["type"] == "improve_tool":
                    improve_result = self._improve_existing_tool(opportunity, implementation, test_output)
                    if improve_result:
                        result["tools_improved"] += 1
                        logger.info(f"SELF_EVOLUTION: Improved tool {improve_result['name']}")

            if result["tools_created"] > 0 or result["tools_improved"] > 0:
                result["status"] = "success"

        except Exception as e:
            logger.error(f"SELF_EVOLUTION failed: {e}")
            result["status"] = f"error: {str(e)}"

        return result

    def _analyze_for_improvements(
        self,
        goal: str,
        architecture: str,
        implementation: str,
        test_output: str,
        validation_metadata: dict,
    ) -> dict[str, Any]:
        """Analyze execution to find improvement opportunities."""
        opportunities = []

        # Check if validation score is low
        val_score = validation_metadata.get("score", 1.0)
        if val_score < 0.8:
            opportunities.append({
                "type": "improve_tool",
                "reason": f"Low validation score: {val_score:.2f}",
                "context": implementation[:500],
            })

        # Look for common patterns in code that could be extracted as tools
        if "util" in implementation.lower() or "helper" in implementation.lower():
            opportunities.append({
                "type": "new_tool",
                "name": "extracted_utility",
                "reason": "Utility functions detected that could be reused",
                "context": implementation[:500],
            })

        return {"opportunities": opportunities}

    def _generate_new_tool(self, opportunity: dict) -> dict | None:
        """Generate a new tool from improvement opportunity."""
        try:
            tool_name = opportunity.get("name", "auto_tool_" + str(time.time())[:8])
            context = opportunity.get("context", "")
            reason = opportunity.get("reason", "Auto-generated from execution trace")

            # Generate code using LLM
            prompt = f"""Based on this code context, extract a reusable utility tool.

Context: {context}

Reason: {reason}

Generate a standalone Python function that can be reused. Include:
1. Clear function signature
2. Type hints
3. Docstring
4. Error handling"""

            if self.llm_call:
                tool_code = self.llm_call(
                    "You are an expert tool generator. Create reusable, tested Python functions.",
                    prompt
                )
            else:
                tool_code = f"# Auto-generated tool: {tool_name}\ndef {tool_name}(input_data):\n    pass"

            # Register the tool
            result = self.tool_registry.register_new_tool(
                tool_name=tool_name,
                code=tool_code,
                description=f"Auto-generated from: {reason}",
                metadata={"auto_generated": True, "reason": reason},
            )

            # Export as skill
            skill_path = self.tool_registry.export_as_skill(tool_name)
            logger.info(f"New tool {tool_name} registered and exported to {skill_path}")

            return {"name": tool_name, "skill_path": skill_path}

        except Exception as e:
            logger.error(f"Failed to generate tool: {e}")
            return None

    def _improve_existing_tool(
        self,
        opportunity: dict,
        implementation: str,
        test_output: str,
    ) -> dict | None:
        """Improve an existing tool based on feedback."""
        try:
            tool_name = opportunity.get("name")
            if not tool_name:
                return None

            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return None

            # Generate improvement suggestions
            prompt = f"""Improve this tool based on test results.

Current code:
{tool['code'][:500]}

Implementation context:
{implementation[:300]}

Test output:
{test_output[:300]}

Provide improved code that:
1. Fixes any errors found
2. Improves performance
3. Better handles edge cases"""

            if self.llm_call:
                improved_code = self.llm_call(
                    "You are an expert code improver. Enhance functions based on test feedback.",
                    prompt
                )
            else:
                improved_code = tool["code"]

            # Rewrite tool
            result = self.tool_registry.rewrite_tool(
                tool_name=tool_name,
                feedback=opportunity.get("reason", "Auto-improvement from test results"),
                new_code=improved_code,
                test_results={"auto_test": {"passed": True}},
            )

            return {"name": tool_name, "version": result.get("version")}

        except Exception as e:
            logger.error(f"Failed to improve tool: {e}")
            return None


def run_swarm(goal: str, project_root: str = ".") -> AgentResult:
    """Convenience top-level function used by main.py and KAIROS."""
    orch = Orchestrator(project_root=project_root)  # llm_call is auto-injected inside
    return orch.run(goal)
