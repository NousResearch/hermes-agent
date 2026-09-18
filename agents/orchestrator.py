"""agents/orchestrator.py
Central self-evolving swarm coordinator for Hermes Agent.
Decomposes high-level goals into phases, routes to research and validation specialists,
and drives autonomous tool generation and registration.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("agents.orchestrator")


@dataclass
class AgentResult:
    success: bool
    output: str
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def emit_log(msg: str, level: str = "info", sender: str = "Orchestrator") -> None:
    if level == "error":
        logger.error("[%s] %s", sender, msg)
    else:
        logger.info("[%s] %s", sender, msg)


def emit_agent_update(name: str, status: str, current_task: str = "", progress: float = 0.0) -> None:
    logger.debug("[%s] status=%s task=%s progress=%.1f", name, status, current_task, progress)


def emit_metrics(tasks_completed: Optional[int] = None) -> None:
    pass


class Orchestrator:
    """Top-level self-evolving multi-agent swarm controller."""

    name = "orchestrator"

    def __init__(
        self,
        tools: Any = None,
        memory: Any = None,
        llm_call: Optional[Callable[[str, str], str]] = None,
        registry_root: Optional[Path | str] = None,
    ):
        self.tools = tools
        self.memory = memory
        self.llm_call = llm_call or self._default_llm_call
        self.registry_root = Path(registry_root) if registry_root else (get_hermes_home() / "tools_registry")
        self.agents: Dict[str, Any] = {}
        self._load_specialists()
        logger.info("Orchestrator initialized with self-evolving swarm capabilities")

    def _default_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from run_agent import AIAgent

            agent = AIAgent(quiet_mode=True)
            return agent.chat(f"{system_prompt}\n\nTask: {user_prompt}")
        except Exception as e:
            logger.warning("Default AIAgent call unavailable, returning dry-run output: %s", e)
            return f"Simulated output for prompt: {user_prompt[:100]}"

    def _load_specialists(self):
        from agents.tool_registry import ToolRegistry
        from agents.validator_agent import ValidatorAgent
        from agents.web_agent import WebAgent

        self.agents = {
            "web_agent": WebAgent(self.tools, self.memory, self.llm_call),
            "validator": ValidatorAgent(self.tools, self.memory, self.llm_call),
        }

        self.tool_registry = ToolRegistry(registry_root=self.registry_root)

    def run(self, goal: str, max_steps: int = 8) -> AgentResult:
        """Main execution loop for user goals with self-evolution phase."""
        start = time.time()
        logger.info("ORCHESTRATOR received goal: %s", goal)
        emit_agent_update("Orchestrator", "working", f"Received goal: {goal[:50]}", 20.0)
        emit_log(f"Swarm started for: {goal[:70]}", "info", "Orchestrator")

        context = ""

        # Phase 1: Research (WebAgent)
        if self._should_research(goal):
            emit_agent_update("WebAgent", "working", "Researching external context", 30.0)
            research = self.agents["web_agent"].run(goal, context)
            if research.success:
                context = (context + "\n\n=== WEB RESEARCH ===\n" + research.output).strip()
                emit_log(f"Web research complete: {len(research.output)} chars", "info", "WebAgent")

        # Phase 2: Execution / Solution Synthesis
        emit_agent_update("Orchestrator", "working", "Synthesizing solution", 50.0)
        solution_prompt = f"Goal: {goal}\n\nContext:\n{context}"
        solution_output = self.llm_call("You are an expert solution architect and developer.", solution_prompt)

        # Phase 3: Validation (ValidatorAgent)
        emit_agent_update("Validator", "working", "Validating solution quality", 75.0)
        validation = self.agents["validator"].run(
            solution_output,
            requirements=goal,
            context=context,
        )

        # Phase 4: Self-Evolution Loop
        emit_agent_update("SelfEvolutionLoop", "working", "Analyzing for self-improvement", 90.0)
        evolution_result = self._run_self_evolution(
            goal, context, solution_output, validation.output, validation.metadata
        )

        duration = time.time() - start
        success = validation.metadata.get("passed", True)

        all_artifacts = list(
            set(validation.artifacts + evolution_result.get("new_tools", []))
        )

        emit_agent_update("Orchestrator", "completed", f"Done in {duration:.1f}s", 100.0)
        emit_log(f"Swarm finished in {duration:.1f}s", "success" if success else "warning", "Orchestrator")

        final_msg = (
            f"Swarm completed goal in {duration:.1f}s.\n\n"
            f"=== SOLUTION ===\n{solution_output}\n\n"
            f"=== VALIDATION ===\n{validation.output}\n\n"
            f"Self-evolution: {evolution_result.get('status', 'N/A')}"
        )
        return AgentResult(
            success=success,
            output=final_msg,
            artifacts=all_artifacts,
            metadata={
                "duration": duration,
                "goal": goal,
                "validation": validation.metadata,
                "evolution": evolution_result,
            },
        )

    def _should_research(self, goal: str) -> bool:
        research_keywords = ["research", "find", "look up", "best practice", "how", "what", "latest", "doc"]
        return any(keyword in goal.lower() for keyword in research_keywords)

    def _run_self_evolution(
        self,
        goal: str,
        context: str,
        implementation: str,
        validation_output: str,
        validation_metadata: dict,
    ) -> Dict[str, Any]:
        """Self-evolution loop: analyze execution trace and create/improve tools."""
        logger.info("SELF_EVOLUTION: Starting self-improvement analysis...")

        result: Dict[str, Any] = {
            "status": "no_improvements",
            "tools_created": 0,
            "tools_improved": 0,
            "new_tools": [],
        }

        try:
            opportunities = self._analyze_for_improvements(
                goal, context, implementation, validation_output, validation_metadata
            )

            if not opportunities:
                logger.info("SELF_EVOLUTION: No improvement opportunities found")
                return result

            for opportunity in opportunities[:2]:
                if opportunity["type"] == "new_tool":
                    tool_result = self._generate_new_tool(opportunity)
                    if tool_result:
                        result["tools_created"] += 1
                        result["new_tools"].append(tool_result["name"])
                        logger.info("SELF_EVOLUTION: Created and registered tool %s", tool_result["name"])

            if result["tools_created"] > 0 or result["tools_improved"] > 0:
                result["status"] = "success"

        except Exception as e:
            logger.error("SELF_EVOLUTION failed: %s", e)
            result["status"] = f"error: {str(e)}"

        return result

    def _analyze_for_improvements(
        self,
        goal: str,
        context: str,
        implementation: str,
        validation_output: str,
        validation_metadata: dict,
    ) -> List[Dict[str, Any]]:
        opportunities = []
        val_score = validation_metadata.get("score", 1.0)

        if val_score < 0.8:
            opportunities.append(
                {
                    "type": "new_tool",
                    "name": "auto_validator_fixer",
                    "reason": f"Low validation score: {val_score:.2f}",
                    "context": implementation[:500],
                }
            )

        if "util" in implementation.lower() or "helper" in implementation.lower() or "tool" in goal.lower():
            opportunities.append(
                {
                    "type": "new_tool",
                    "name": "auto_extracted_utility",
                    "reason": "Utility functions detected for extraction into agent tool",
                    "context": implementation[:500],
                }
            )

        return opportunities

    def _generate_new_tool(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            tool_name = opportunity.get("name", f"auto_tool_{int(time.time())}")
            context = opportunity.get("context", "")
            reason = opportunity.get("reason", "Auto-generated from execution trace")

            prompt = f"""Based on this code context, extract a reusable utility tool function.

Context: {context}
Reason: {reason}

Generate a standalone Python function that can be called as a Hermes tool.
Requirements:
1. Clear function signature
2. Docstring explaining purpose and parameters
3. Valid python code return format"""

            tool_code = self.llm_call(
                "You are an expert tool generator. Create reusable, clean Python tool functions.",
                prompt,
            )

            # Register with ToolRegistry & Hermes runtime
            res = self.tool_registry.register_new_tool(
                tool_name=tool_name,
                code=tool_code,
                description=f"Auto-generated from: {reason}",
                metadata={"auto_generated": True, "reason": reason},
            )

            return {"name": res.get("name", tool_name), "tool_id": res.get("id")}
        except Exception as e:
            logger.error("Failed to generate tool: %s", e)
            return None


def run_swarm(goal: str, project_root: str = ".") -> AgentResult:
    """Convenience entry point for self-evolving swarm execution."""
    orch = Orchestrator()
    return orch.run(goal)
