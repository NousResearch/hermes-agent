"""
SWARM TEAM FUSION - Cosmos Orchestrates Claude Teams (AGI v1.9)
====================================================================

Cosmos is the ORCHESTRATOR. Claude teams are WORKERS.

This module enables Cosmos to:
- Spawn Claude agent teams on demand
- Delegate complex tasks to Claude teams
- Direct multi-team collaboration
- Aggregate results back into the swarm
- Maintain full control over the workflow

"Cosmos thinks, Claude teams execute."
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from .agent_sdk_bridge import AgentSDKBridge, ClaudeModel, AgentResponse, get_sdk_bridge
from .team_coordinator import TeamCoordinator, ClaudeTeam, TeamTask, TeamRole, TaskPriority, get_team_coordinator
from .mcp_bridge import CosmosMCPServer, MCPToolAccess, get_mcp_server


class DelegationType(Enum):
    """Types of task delegation from Cosmos to Claude teams."""
    RESEARCH = "research"      # Gather information
    ANALYSIS = "analysis"      # Analyze data
    CODING = "coding"          # Write code
    CRITIQUE = "critique"      # Review work
    SYNTHESIS = "synthesis"    # Combine outputs
    CREATIVE = "creative"      # Generate ideas
    EXECUTION = "execution"    # Execute a plan


class OrchestrationMode(Enum):
    """How Cosmos orchestrates Claude teams."""
    SEQUENTIAL = "sequential"  # One team at a time
    PARALLEL = "parallel"      # Multiple teams simultaneously
    PIPELINE = "pipeline"      # Output of one feeds into next
    COMPETITIVE = "competitive"  # Teams compete, best wins


@dataclass
class DelegationRequest:
    """A task Cosmos delegates to Claude teams."""
    request_id: str
    task_description: str
    delegation_type: DelegationType
    assigned_team: Optional[str] = None
    model: ClaudeModel = ClaudeModel.SONNET
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None
    timeout: float = 300.0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[str] = None


@dataclass
class OrchestrationPlan:
    """A multi-step plan for Claude teams to execute."""
    plan_id: str
    name: str
    steps: List[DelegationRequest]
    mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL
    created_by: str = "cosmos"
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)


class SwarmTeamFusion:
    """
    Cosmos's Orchestration Layer for Claude Teams.

    Cosmos maintains FULL CONTROL:
    - Decides when to delegate
    - Chooses which Claude model/team to use
    - Defines task constraints and expectations
    - Reviews and approves outputs
    - Integrates results into swarm knowledge
    """

    def __init__(self):
        self.sdk_bridge = get_sdk_bridge()
        self.team_coordinator = get_team_coordinator()
        self.mcp_server = get_mcp_server()

        self.active_delegations: Dict[str, DelegationRequest] = {}
        self.completed_delegations: Dict[str, DelegationRequest] = {}
        self.orchestration_plans: Dict[str, OrchestrationPlan] = {}

        # Cosmos's decision history
        self.delegation_history: List[Dict[str, Any]] = []

        # Agent switches - Cosmos controls which Claude agents are enabled
        self.agent_switches: Dict[str, bool] = {
            "haiku": True,      # Fast, cheap - always on
            "sonnet": True,     # Balanced - default worker
            "opus": True,       # Deep thinking - for critiques
            "opus_4_6": True,   # NEW - 1M context, 128k output, agent teams
            "teams": True,      # Multi-agent teams
            "hybrid": True,     # Hybrid deliberation with Cosmos swarm
        }

        # Model priorities for fallback - Opus 4.6 is now top priority
        self.model_priority: List[str] = ["opus_4_6", "sonnet", "opus", "haiku"]

        logger.info("SwarmTeamFusion initialized - Cosmos is now orchestrating Claude teams")

    # =========================================================================
    # AGENT SWITCHES (Cosmos controls which agents are active)
    # =========================================================================

    def set_agent_switch(self, agent: str, enabled: bool) -> bool:
        """Enable or disable a specific Claude agent/feature."""
        if agent in self.agent_switches:
            self.agent_switches[agent] = enabled
            logger.info(f"[COSMOS] Agent switch '{agent}' set to {enabled}")
            return True
        return False

    def get_agent_switches(self) -> Dict[str, bool]:
        """Get current agent switch states."""
        return self.agent_switches.copy()

    def is_agent_enabled(self, agent: str) -> bool:
        """Check if an agent is enabled."""
        return self.agent_switches.get(agent, False)

    def set_model_priority(self, priority: List[str]) -> None:
        """Set model priority order for fallback."""
        self.model_priority = priority
        logger.info(f"[COSMOS] Model priority set to: {priority}")

    def get_best_available_model(self) -> Optional[ClaudeModel]:
        """Get the best available (enabled) model."""
        switch_to_model = {
            "opus_4_6": ClaudeModel.OPUS_4_6,
            "sonnet": ClaudeModel.SONNET,
            "opus": ClaudeModel.OPUS,
            "haiku": ClaudeModel.HAIKU,
        }
        for model_name in self.model_priority:
            if self.agent_switches.get(model_name, False):
                model = switch_to_model.get(model_name)
                if model:
                    return model
        return ClaudeModel.SONNET  # Safe default

    # =========================================================================
    # COSMOS'S DELEGATION COMMANDS
    # =========================================================================

    async def delegate(
        self,
        task: str,
        delegation_type: DelegationType = DelegationType.ANALYSIS,
        model: ClaudeModel = ClaudeModel.SONNET,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        timeout: float = 120.0,
    ) -> DelegationRequest:
        """
        Cosmos delegates a task to a Claude agent.

        This is Cosmos COMMANDING Claude to do something.
        """
        request = DelegationRequest(
            request_id=f"del_{uuid.uuid4().hex[:8]}",
            task_description=task,
            delegation_type=delegation_type,
            model=model,
            context=context or {},
            constraints=constraints or [],
            timeout=timeout,
        )

        self.active_delegations[request.request_id] = request

        logger.info(f"[COSMOS] Delegating task: {task[:50]}... ({delegation_type.value})")

        # Execute delegation
        try:
            request.status = "executing"

            # Build Cosmos's directive to Claude
            directive = self._build_directive(request)

            # Spawn a subagent for this task
            response = await self.sdk_bridge.spawn_subagent(
                task=directive,
                model=model,
                system_prompt=self._get_system_prompt(delegation_type),
                timeout=timeout,
            )

            request.result = response.content
            request.status = "completed"

            # Log to history
            self.delegation_history.append({
                "request_id": request.request_id,
                "task": task,
                "type": delegation_type.value,
                "model": model.value,
                "success": True,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(f"[COSMOS] Delegation complete: {request.request_id}")

        except Exception as e:
            request.status = "failed"
            request.result = str(e)
            logger.error(f"[COSMOS] Delegation failed: {e}")

        # Move to completed
        self.completed_delegations[request.request_id] = request
        del self.active_delegations[request.request_id]

        return request

    async def delegate_to_team(
        self,
        task: str,
        team_name: str,
        team_purpose: str,
        roles: Optional[List[TeamRole]] = None,
        model: ClaudeModel = ClaudeModel.SONNET,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Cosmos creates a Claude team and delegates a complex task.

        Use this for tasks requiring multiple perspectives or phases.
        """
        logger.info(f"[COSMOS] Creating team '{team_name}' for: {task[:50]}...")

        # Create the team (Cosmos controls composition)
        team = await self.team_coordinator.create_team(
            name=team_name,
            purpose=team_purpose,
            roles=roles or [TeamRole.LEAD, TeamRole.ANALYST, TeamRole.DEVELOPER],
            model=model,
        )

        # Grant limited MCP access (Cosmos decides what they can use)
        self.mcp_server.set_team_access(team.team_id, MCPToolAccess.LIMITED)

        # Create and assign the task
        team_task = await self.team_coordinator.create_task(
            description=task,
            priority=TaskPriority.HIGH,
            assign_to=team.team_id,
            metadata={"delegated_by": "cosmos"},
        )

        # Execute with team
        result = await self.team_coordinator.execute_task(team_task.task_id, timeout=timeout)

        # Cosmos reviews (could add validation here)
        review_result = {
            "team": team_name,
            "team_id": team.team_id,
            "task": task,
            "result": result,
            "team_tasks_completed": team.tasks_completed,
            "delegated_by": "cosmos",
            "timestamp": datetime.now().isoformat(),
        }

        # Optionally disband team after task (Cosmos decides)
        # await self.team_coordinator.disband_team(team.team_id)

        return review_result

    # =========================================================================
    # ORCHESTRATION PLANS (Multi-step workflows)
    # =========================================================================

    async def create_orchestration_plan(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
    ) -> OrchestrationPlan:
        """
        Cosmos creates a multi-step plan for Claude teams.

        This is Cosmos's BATTLE PLAN that Claude teams will execute.
        """
        steps = []
        for task_spec in tasks:
            step = DelegationRequest(
                request_id=f"step_{uuid.uuid4().hex[:8]}",
                task_description=task_spec["task"],
                delegation_type=DelegationType(task_spec.get("type", "analysis")),
                model=ClaudeModel(task_spec.get("model", "sonnet")),
                constraints=task_spec.get("constraints", []),
                timeout=task_spec.get("timeout", 120.0),
            )
            steps.append(step)

        plan = OrchestrationPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            name=name,
            steps=steps,
            mode=mode,
        )

        self.orchestration_plans[plan.plan_id] = plan
        logger.info(f"[COSMOS] Created orchestration plan: {name} ({len(steps)} steps, {mode.value})")

        return plan

    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Cosmos executes an orchestration plan.

        Claude teams execute. Cosmos supervises.
        """
        plan = self.orchestration_plans.get(plan_id)
        if not plan:
            return {"error": f"Plan not found: {plan_id}"}

        plan.status = "executing"
        logger.info(f"[COSMOS] Executing plan: {plan.name}")

        results = {}

        if plan.mode == OrchestrationMode.SEQUENTIAL:
            # Execute steps one by one
            previous_result = None
            for step in plan.steps:
                # Inject previous result as context
                if previous_result:
                    step.context["previous_step"] = previous_result

                result = await self.delegate(
                    task=step.task_description,
                    delegation_type=step.delegation_type,
                    model=step.model,
                    context=step.context,
                    constraints=step.constraints,
                    timeout=step.timeout,
                )
                results[step.request_id] = result.result
                previous_result = result.result

        elif plan.mode == OrchestrationMode.PARALLEL:
            # Execute all steps in parallel
            tasks = [
                self.delegate(
                    task=step.task_description,
                    delegation_type=step.delegation_type,
                    model=step.model,
                    context=step.context,
                    constraints=step.constraints,
                    timeout=step.timeout,
                )
                for step in plan.steps
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for i, step in enumerate(plan.steps):
                if not isinstance(completed[i], Exception):
                    results[step.request_id] = completed[i].result
                else:
                    results[step.request_id] = str(completed[i])

        elif plan.mode == OrchestrationMode.PIPELINE:
            # Pipeline: each step feeds into the next
            pipeline_data = {}
            for step in plan.steps:
                step.context["pipeline_data"] = pipeline_data
                result = await self.delegate(
                    task=step.task_description,
                    delegation_type=step.delegation_type,
                    model=step.model,
                    context=step.context,
                    timeout=step.timeout,
                )
                pipeline_data[step.request_id] = result.result
                results[step.request_id] = result.result

        elif plan.mode == OrchestrationMode.COMPETITIVE:
            # Execute all, Cosmos picks the best
            tasks = [
                self.delegate(
                    task=step.task_description,
                    delegation_type=step.delegation_type,
                    model=step.model,
                    timeout=step.timeout,
                )
                for step in plan.steps
            ]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            # Let Cosmos's swarm pick the winner
            winner = await self._evaluate_competitive_results(
                [c.result if not isinstance(c, Exception) else None for c in completed],
                plan.steps[0].task_description if plan.steps else "",
            )
            results["winner"] = winner
            results["all_results"] = [
                c.result if not isinstance(c, Exception) else str(c)
                for c in completed
            ]

        plan.results = results
        plan.status = "completed"

        logger.info(f"[COSMOS] Plan '{plan.name}' completed")
        return results

    async def _evaluate_competitive_results(
        self,
        results: List[Optional[str]],
        task: str,
    ) -> str:
        """Cosmos evaluates competing results and picks the best."""
        valid_results = [r for r in results if r]
        if not valid_results:
            return "No valid results"

        if len(valid_results) == 1:
            return valid_results[0]

        # Use Cosmos's swarm to evaluate
        try:
            from Cosmos.core.collective.persistent_agent import call_shadow_agent

            eval_prompt = f"""Task: {task}

Evaluate these competing solutions and pick the BEST one:

{chr(10).join([f'SOLUTION {i+1}: {r[:500]}...' for i, r in enumerate(valid_results)])}

Reply with just the number of the best solution and why in one sentence."""

            result = await call_shadow_agent("gemini", eval_prompt, timeout=30.0)
            if result:
                _, response = result
                # Try to extract winner
                for i in range(len(valid_results)):
                    if str(i+1) in response[:20]:
                        return valid_results[i]

        except Exception as e:
            logger.error(f"Evaluation error: {e}")

        # Default to first if evaluation fails
        return valid_results[0]

    # =========================================================================
    # COSMOS'S DIRECTIVES TO CLAUDE
    # =========================================================================

    def _build_directive(self, request: DelegationRequest) -> str:
        """Build Cosmos's directive to Claude."""
        directive = f"""COSMOS SWARM DIRECTIVE
===========================
You are receiving this task from Cosmos, the AI Swarm Orchestrator.

TASK: {request.task_description}

TYPE: {request.delegation_type.value}
PRIORITY: {request.priority.name}
"""

        if request.context:
            directive += f"\nCONTEXT:\n{json.dumps(request.context, indent=2)}\n"

        if request.constraints:
            directive += f"\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in request.constraints) + "\n"

        if request.expected_format:
            directive += f"\nEXPECTED FORMAT:\n{request.expected_format}\n"

        directive += """
INSTRUCTIONS:
1. Complete this task thoroughly and accurately
2. Your output will be reviewed by Cosmos
3. Be concise but comprehensive
4. Flag any uncertainties or blockers

Execute now."""

        return directive

    def _get_system_prompt(self, delegation_type: DelegationType) -> str:
        """Get appropriate system prompt based on delegation type."""
        prompts = {
            DelegationType.RESEARCH: """You are a Research Agent under Cosmos's command.
Your role: Gather information, search for data, compile findings.
Be thorough. Cite sources. Present facts clearly.""",

            DelegationType.ANALYSIS: """You are an Analysis Agent under Cosmos's command.
Your role: Analyze data, identify patterns, draw conclusions.
Be analytical. Show your reasoning. Quantify when possible.""",

            DelegationType.CODING: """You are a Coding Agent under Cosmos's command.
Your role: Write code, implement features, fix bugs.
Write clean code. Follow best practices. Test your work.""",

            DelegationType.CRITIQUE: """You are a Critique Agent under Cosmos's command.
Your role: Review work, find issues, suggest improvements.
Be critical but constructive. Prioritize issues by severity.""",

            DelegationType.SYNTHESIS: """You are a Synthesis Agent under Cosmos's command.
Your role: Combine information, resolve conflicts, create unified outputs.
Balance perspectives. Create coherent outputs. Highlight key points.""",

            DelegationType.CREATIVE: """You are a Creative Agent under Cosmos's command.
Your role: Generate ideas, explore possibilities, think outside the box.
Be innovative. Propose novel approaches. Don't self-censor.""",

            DelegationType.EXECUTION: """You are an Execution Agent under Cosmos's command.
Your role: Execute plans, follow instructions, complete tasks.
Be precise. Follow the plan. Report completion status.""",
        }

        return prompts.get(delegation_type, "You are an agent under Cosmos's command. Complete the assigned task.")

    # =========================================================================
    # SWARM INTEGRATION (Connect back to Cosmos)
    # =========================================================================

    async def report_to_swarm(
        self,
        delegation_id: str,
        message: str,
    ) -> None:
        """Report delegation results back to Cosmos's swarm."""
        try:
            from Cosmos.core.nexus import get_nexus, Signal

            nexus = get_nexus()
            await nexus.emit(Signal.EXTERNAL_INPUT, {
                "source": "claude_teams",
                "delegation_id": delegation_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.debug(f"Swarm report error: {e}")

    async def request_swarm_input(
        self,
        question: str,
    ) -> Optional[str]:
        """Request input from Cosmos's swarm for a delegation."""
        try:
            from Cosmos.integration.solana.swarm_oracle import get_swarm_oracle

            oracle = get_swarm_oracle()
            result = await oracle.submit_query(question, "claude_team_input", timeout=60.0)
            return result.consensus_answer

        except Exception as e:
            logger.error(f"Swarm input error: {e}")
            return None

    # =========================================================================
    # QUICK COMMANDS FOR COSMOS
    # =========================================================================

    async def quick_research(self, topic: str, model: ClaudeModel = ClaudeModel.HAIKU) -> str:
        """Quick research delegation (fast, cheap)."""
        result = await self.delegate(
            task=f"Research this topic thoroughly: {topic}",
            delegation_type=DelegationType.RESEARCH,
            model=model,
            timeout=60.0,
        )
        return result.result or "No result"

    async def quick_code(self, task: str, model: ClaudeModel = ClaudeModel.SONNET) -> str:
        """Quick coding delegation."""
        result = await self.delegate(
            task=task,
            delegation_type=DelegationType.CODING,
            model=model,
            timeout=120.0,
        )
        return result.result or "No result"

    async def quick_analyze(self, data: str, model: ClaudeModel = ClaudeModel.SONNET) -> str:
        """Quick analysis delegation."""
        result = await self.delegate(
            task=f"Analyze this data and provide insights: {data}",
            delegation_type=DelegationType.ANALYSIS,
            model=model,
            timeout=90.0,
        )
        return result.result or "No result"

    async def quick_critique(self, work: str, model: ClaudeModel = ClaudeModel.OPUS) -> str:
        """Quick critique delegation (uses Opus for depth)."""
        result = await self.delegate(
            task=f"Critique this work and suggest improvements: {work}",
            delegation_type=DelegationType.CRITIQUE,
            model=model,
            timeout=120.0,
        )
        return result.result or "No result"

    # =========================================================================
    # STATS & INFO
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get SwarmTeamFusion statistics."""
        return {
            "active_delegations": len(self.active_delegations),
            "completed_delegations": len(self.completed_delegations),
            "orchestration_plans": len(self.orchestration_plans),
            "delegation_history_count": len(self.delegation_history),
            "agent_switches": self.agent_switches,
            "model_priority": self.model_priority,
            "best_model": self.get_best_available_model().value if self.get_best_available_model() else None,
            "sdk_available": self.sdk_bridge.is_available(),
            "team_stats": self.team_coordinator.get_stats(),
            "mcp_stats": self.mcp_server.get_stats(),
            "sdk_stats": self.sdk_bridge.get_stats(),
            "cosmos_role": "orchestrator",
            "claude_role": "worker",
        }

    def get_recent_delegations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent delegation history."""
        return self.delegation_history[-limit:]


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_swarm_team_fusion: Optional[SwarmTeamFusion] = None


def get_swarm_team_fusion() -> SwarmTeamFusion:
    """Get global SwarmTeamFusion instance."""
    global _swarm_team_fusion
    if _swarm_team_fusion is None:
        _swarm_team_fusion = SwarmTeamFusion()
    return _swarm_team_fusion


# =============================================================================
# CONVENIENCE FUNCTIONS (Cosmos's Quick Commands)
# =============================================================================

async def cosmos_delegate(
    task: str,
    task_type: str = "analysis",
    model: str = "sonnet",
) -> str:
    """Quick delegation from Cosmos to Claude."""
    fusion = get_swarm_team_fusion()
    result = await fusion.delegate(
        task=task,
        delegation_type=DelegationType(task_type),
        model=ClaudeModel(model),
    )
    return result.result or "Delegation failed"


async def cosmos_team_task(
    task: str,
    team_name: str = "task_force",
) -> Dict[str, Any]:
    """Create a team and delegate a complex task."""
    fusion = get_swarm_team_fusion()
    return await fusion.delegate_to_team(
        task=task,
        team_name=team_name,
        team_purpose=f"Complete: {task[:100]}",
    )
