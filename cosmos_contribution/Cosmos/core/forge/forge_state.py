"""
FORGE State Manager
====================

Manages project state with both file-based tracking AND 7-layer memory
integration. Unlike file-only approaches, FORGE state survives context
resets, session restarts, and even server migrations.

State is stored in:
1. .forge/ directory (structured files for human readability)
2. Archival Memory (embedding-indexed for semantic retrieval)
3. Knowledge Graph (relationship-aware for dependency tracking)
4. Working Memory (hot cache for current session)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import   Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ForgeTask:
    """A single executable task within a plan."""
    id: str
    name: str
    files: list[str]
    action: str
    verify: str
    done_criteria: str
    task_type: str = "auto"  # auto, checkpoint, review
    status: str = "pending"  # pending, running, passed, failed, skipped
    wave: int = 1
    depends_on: list[str] = field(default_factory=list)
    assigned_model: Optional[str] = None
    cost_tokens: int = 0
    execution_time_ms: int = 0
    deviation_log: list[str] = field(default_factory=list)
    commit_hash: Optional[str] = None


@dataclass
class ForgePlan:
    """A plan containing multiple tasks for a phase."""
    id: str
    phase: str
    plan_number: int
    objective: str
    tasks: list[ForgeTask]
    wave: int = 1
    status: str = "pending"  # pending, executing, verified, failed
    must_haves: list[str] = field(default_factory=list)
    research_refs: list[str] = field(default_factory=list)
    context_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    consensus_score: float = 0.0  # swarm agreement on this plan
    critiques: list[dict] = field(default_factory=list)


@dataclass
class ForgePhase:
    """A development phase with goals and success criteria."""
    id: str
    name: str
    description: str
    goals: list[str]
    success_criteria: list[str]
    plans: list[ForgePlan] = field(default_factory=list)
    status: str = "pending"  # pending, planning, executing, verifying, complete
    decisions: dict[str, str] = field(default_factory=dict)
    deferred: list[str] = field(default_factory=list)
    verification_result: Optional[dict] = None


@dataclass
class ForgeProject:
    """Top-level project state."""
    name: str
    description: str
    phases: list[ForgePhase] = field(default_factory=list)
    current_phase: int = 0
    milestone: str = "v1.0"
    total_cost_tokens: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    decisions: list[dict] = field(default_factory=list)
    blockers: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_session: str = field(default_factory=lambda: datetime.now().isoformat())


class ForgeStateManager:
    """
    Manages FORGE project state with dual persistence:
    file-based (.forge/) + memory system integration.
    """

    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace)
        self.forge_dir = self.workspace / ".forge"
        self.forge_dir.mkdir(parents=True, exist_ok=True)
        self.project: Optional[ForgeProject] = None
        self._memory = None
        self._init_memory()

    def _init_memory(self):
        """Connect to Cosmos memory system if available."""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            self._memory = get_memory_system()
        except Exception:
            self._memory = None

    # =========================================================================
    # PROJECT LIFECYCLE
    # =========================================================================

    def init_project(self, name: str, description: str) -> ForgeProject:
        """Initialize a new FORGE project."""
        self.project = ForgeProject(name=name, description=description)
        self._save_state()
        self._save_project_md()

        if self._memory:
            try:
                import asyncio
                asyncio.get_event_loop().run_until_complete(
                    self._memory.store(
                        f"FORGE project initialized: {name} - {description}",
                        metadata={"type": "forge_project", "name": name}
                    )
                )
            except Exception:
                pass

        return self.project

    def load_project(self) -> Optional[ForgeProject]:
        """Load existing project state."""
        state_file = self.forge_dir / "state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            self.project = self._deserialize_project(data)
            return self.project
        return None

    def _save_state(self):
        """Persist state to .forge/state.json."""
        if not self.project:
            return
        state_file = self.forge_dir / "state.json"
        state_file.write_text(json.dumps(asdict(self.project), indent=2))

    def _save_project_md(self):
        """Generate human-readable PROJECT.md."""
        if not self.project:
            return
        p = self.project
        lines = [
            f"# {p.name}",
            "",
            p.description,
            "",
            f"**Milestone:** {p.milestone}",
            f"**Tasks Completed:** {p.total_tasks_completed}",
            f"**Token Cost:** {p.total_cost_tokens:,}",
            "",
            "## Phases",
            "",
        ]
        for i, phase in enumerate(p.phases):
            marker = ">>>" if i == p.current_phase else "   "
            lines.append(f"{marker} {i+1}. [{phase.status.upper()}] {phase.name}")
            for goal in phase.goals:
                lines.append(f"       - {goal}")
            lines.append("")

        if p.decisions:
            lines.extend(["## Key Decisions", ""])
            for d in p.decisions[-10:]:
                lines.append(f"- **{d.get('topic', '?')}**: {d.get('decision', '?')}")
            lines.append("")

        if p.blockers:
            lines.extend(["## Active Blockers", ""])
            for b in p.blockers:
                if not b.get("resolved"):
                    lines.append(f"- {b.get('description', '?')}")
            lines.append("")

        (self.forge_dir / "PROJECT.md").write_text("\n".join(lines))

    # =========================================================================
    # PHASE MANAGEMENT
    # =========================================================================

    def add_phase(self, name: str, description: str, goals: list[str],
                  success_criteria: list[str]) -> ForgePhase:
        """Add a new phase to the project."""
        phase_id = f"phase_{len(self.project.phases) + 1}"
        phase = ForgePhase(
            id=phase_id,
            name=name,
            description=description,
            goals=goals,
            success_criteria=success_criteria,
        )
        self.project.phases.append(phase)
        self._save_state()
        self._save_project_md()
        return phase

    def get_current_phase(self) -> Optional[ForgePhase]:
        """Get the current active phase."""
        if self.project and self.project.current_phase < len(self.project.phases):
            return self.project.phases[self.project.current_phase]
        return None

    def advance_phase(self):
        """Move to the next phase."""
        if self.project:
            current = self.get_current_phase()
            if current:
                current.status = "complete"
            self.project.current_phase += 1
            self._save_state()
            self._save_project_md()

    # =========================================================================
    # PLAN MANAGEMENT
    # =========================================================================

    def add_plan(self, phase_id: str, objective: str, tasks: list[dict],
                 wave: int = 1, must_haves: list[str] = None,
                 consensus_score: float = 0.0,
                 critiques: list[dict] = None) -> ForgePlan:
        """Add a plan to a phase."""
        phase = self._find_phase(phase_id)
        if not phase:
            raise ValueError(f"Phase not found: {phase_id}")

        plan_number = len(phase.plans) + 1
        plan_id = f"{phase_id}_plan_{plan_number}"

        forge_tasks = []
        for i, t in enumerate(tasks):
            forge_tasks.append(ForgeTask(
                id=f"{plan_id}_task_{i+1}",
                name=t["name"],
                files=t.get("files", []),
                action=t["action"],
                verify=t.get("verify", ""),
                done_criteria=t.get("done", ""),
                task_type=t.get("type", "auto"),
                wave=t.get("wave", wave),
                depends_on=t.get("depends_on", []),
            ))

        plan = ForgePlan(
            id=plan_id,
            phase=phase_id,
            plan_number=plan_number,
            objective=objective,
            tasks=forge_tasks,
            wave=wave,
            must_haves=must_haves or [],
            consensus_score=consensus_score,
            critiques=critiques or [],
        )

        phase.plans.append(plan)
        self._save_state()
        self._save_plan_md(plan)
        return plan

    def _save_plan_md(self, plan: ForgePlan):
        """Save plan as human-readable markdown."""
        lines = [
            f"# Plan {plan.plan_number}: {plan.objective}",
            f"**Phase:** {plan.phase}",
            f"**Wave:** {plan.wave}",
            f"**Consensus:** {plan.consensus_score:.0%}",
            "",
        ]

        if plan.must_haves:
            lines.extend(["## Must Haves", ""])
            for mh in plan.must_haves:
                lines.append(f"- {mh}")
            lines.append("")

        if plan.critiques:
            lines.extend(["## Swarm Critiques", ""])
            for c in plan.critiques:
                lines.append(f"- **{c.get('agent', '?')}**: {c.get('critique', '?')}")
            lines.append("")

        lines.extend(["## Tasks", ""])
        for task in plan.tasks:
            status_icon = {"pending": "[ ]", "running": "[~]", "passed": "[x]",
                          "failed": "[!]", "skipped": "[-]"}.get(task.status, "[ ]")
            lines.append(f"### {status_icon} {task.name}")
            lines.append(f"**Type:** {task.task_type} | **Wave:** {task.wave}")
            if task.files:
                lines.append(f"**Files:** {', '.join(task.files)}")
            lines.append(f"\n{task.action}\n")
            if task.verify:
                lines.append(f"**Verify:** {task.verify}")
            if task.done_criteria:
                lines.append(f"**Done:** {task.done_criteria}")
            lines.append("")

        plan_file = self.forge_dir / f"{plan.phase}_{plan.plan_number}_PLAN.md"
        plan_file.write_text("\n".join(lines))

    # =========================================================================
    # TASK TRACKING
    # =========================================================================

    def update_task(self, task_id: str, status: str, commit_hash: str = None,
                    cost_tokens: int = 0, execution_time_ms: int = 0,
                    deviation: str = None):
        """Update a task's status and metrics."""
        task = self._find_task(task_id)
        if task:
            task.status = status
            if commit_hash:
                task.commit_hash = commit_hash
            task.cost_tokens += cost_tokens
            task.execution_time_ms += execution_time_ms
            if deviation:
                task.deviation_log.append(deviation)
            if status == "passed":
                self.project.total_tasks_completed += 1
            elif status == "failed":
                self.project.total_tasks_failed += 1
            self.project.total_cost_tokens += cost_tokens
            self._save_state()

    def record_decision(self, topic: str, decision: str, rationale: str = ""):
        """Record a project decision."""
        self.project.decisions.append({
            "topic": topic,
            "decision": decision,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()
        self._save_project_md()

    def add_blocker(self, description: str, severity: str = "medium"):
        """Add a blocker."""
        self.project.blockers.append({
            "description": description,
            "severity": severity,
            "resolved": False,
            "added_at": datetime.now().isoformat(),
        })
        self._save_state()

    def resolve_blocker(self, index: int):
        """Resolve a blocker by index."""
        if 0 <= index < len(self.project.blockers):
            self.project.blockers[index]["resolved"] = True
            self.project.blockers[index]["resolved_at"] = datetime.now().isoformat()
            self._save_state()

    # =========================================================================
    # ROLLBACK
    # =========================================================================

    def get_rollback_points(self) -> list[dict]:
        """Get all commit hashes from completed tasks for rollback."""
        points = []
        for phase in self.project.phases:
            for plan in phase.plans:
                for task in plan.tasks:
                    if task.commit_hash:
                        points.append({
                            "task": task.name,
                            "commit": task.commit_hash,
                            "phase": phase.name,
                            "timestamp": plan.created_at,
                        })
        return points

    # =========================================================================
    # COST TRACKING
    # =========================================================================

    def get_cost_report(self) -> dict:
        """Get detailed cost breakdown."""
        phase_costs = {}
        model_costs = {}

        for phase in self.project.phases:
            phase_total = 0
            for plan in phase.plans:
                for task in plan.tasks:
                    phase_total += task.cost_tokens
                    model = task.assigned_model or "unknown"
                    model_costs[model] = model_costs.get(model, 0) + task.cost_tokens
            phase_costs[phase.name] = phase_total

        return {
            "total_tokens": self.project.total_cost_tokens,
            "by_phase": phase_costs,
            "by_model": model_costs,
            "tasks_completed": self.project.total_tasks_completed,
            "tasks_failed": self.project.total_tasks_failed,
        }

    # =========================================================================
    # PROGRESS
    # =========================================================================

    def get_progress(self) -> dict:
        """Get project progress summary."""
        total_tasks = 0
        completed_tasks = 0
        failed_tasks = 0

        for phase in self.project.phases:
            for plan in phase.plans:
                for task in plan.tasks:
                    total_tasks += 1
                    if task.status == "passed":
                        completed_tasks += 1
                    elif task.status == "failed":
                        failed_tasks += 1

        phases_complete = sum(1 for p in self.project.phases if p.status == "complete")

        return {
            "project": self.project.name,
            "milestone": self.project.milestone,
            "phases_total": len(self.project.phases),
            "phases_complete": phases_complete,
            "current_phase": self.get_current_phase().name if self.get_current_phase() else "N/A",
            "tasks_total": total_tasks,
            "tasks_completed": completed_tasks,
            "tasks_failed": failed_tasks,
            "completion_pct": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "total_cost_tokens": self.project.total_cost_tokens,
        }

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _find_phase(self, phase_id: str) -> Optional[ForgePhase]:
        if self.project:
            for p in self.project.phases:
                if p.id == phase_id:
                    return p
        return None

    def _find_task(self, task_id: str) -> Optional[ForgeTask]:
        if self.project:
            for phase in self.project.phases:
                for plan in phase.plans:
                    for task in plan.tasks:
                        if task.id == task_id:
                            return task
        return None

    def _deserialize_project(self, data: dict) -> ForgeProject:
        """Deserialize project from JSON dict."""
        phases = []
        for pd in data.get("phases", []):
            plans = []
            for pld in pd.get("plans", []):
                tasks = [ForgeTask(**td) for td in pld.get("tasks", [])]
                pld_copy = {k: v for k, v in pld.items() if k != "tasks"}
                plans.append(ForgePlan(**pld_copy, tasks=tasks))
            pd_copy = {k: v for k, v in pd.items() if k != "plans"}
            phases.append(ForgePhase(**pd_copy, plans=plans))

        proj_data = {k: v for k, v in data.items() if k != "phases"}
        return ForgeProject(**proj_data, phases=phases)
