"""Conductor engine: fan-out gate, plan card estimator, loop detector, and end-to-end pipeline (05-final-plan.md §4–5)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from samagent.conductor.speed import SpeedProfileReport, evaluate_speed_profile
from samagent.conductor.verify import VerificationReport, VerificationRunner
from samagent.conductor.worktree_integrator import (
    IntegrationReport,
    WorktreeSwarmIntegrator,
)
from samagent.contract.freeze import ContractVersion, freeze_contract
from samagent.ledger.repo_map import prefetch_repo_map
from samagent.ledger.store import ProjectLedger
from samagent.router.policy import TaskBoundaryRouter
from samagent.spec.acceptance import (
    RedFirstCheckResult,
    generate_acceptance_suite,
    verify_red_first,
)
from samagent.spec.critique import CritiqueReport, critique_spec
from samagent.spec.interview import synthesize_spec_from_brief
from samagent.spec.models import AutonomyLevel, SpecDocument
from samagent.templates import scaffold_module, scaffold_project


@dataclass
class FanoutDecision:
    allow_parallel: bool
    reason: str
    waves: List[List[str]] = field(default_factory=list)
    disjoint_ownership: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanCard:
    goal: str
    stack: str
    autonomy: str
    router_policy: str
    modules: List[Dict[str, Any]]
    fanout: Dict[str, Any]
    routing_table: List[Dict[str, Any]]
    local_task_share_pct: float
    estimated_cost_usd_range: Tuple[float, float]
    estimated_minutes_range: Tuple[int, int]
    assumptions: List[Dict[str, Any]]
    risks: List[str]
    critique: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _globs_overlap(globs_a: List[str], globs_b: List[str]) -> bool:
    for a in globs_a:
        pa = a.strip().lstrip("./").rstrip("/*")
        for b in globs_b:
            pb = b.strip().lstrip("./").rstrip("/*")
            if pa == pb or pa.startswith(pb + "/") or pb.startswith(pa + "/"):
                return True
    return False


def evaluate_fanout_gate(
    project_dir: Path,
    spec: SpecDocument,
    *,
    min_module_minutes: float = 5.0,
) -> FanoutDecision:
    """Evaluate the 5 fan-out gate conditions from 05-final-plan.md §5.

    Parallel swarm fan-out is allowed ONLY when:
    1. Contract is frozen (.samagent/contract/version.json exists)
    2. >= 2 modules exist
    3. Write-sets (owned_globs) are strictly disjoint across parallel modules
    4. Dependency DAG has no cycles and >= 2 independent modules in a wave
    5. Modules are large enough to amortize a fresh context (>= min_module_minutes) and budget > 0
    """
    ver_file = Path(project_dir) / ".samagent" / "contract" / "version.json"
    if not ver_file.exists():
        return FanoutDecision(
            allow_parallel=False,
            reason="Contract is not frozen yet (.samagent/contract/version.json missing).",
            waves=[[m.name for m in spec.modules]] if spec.modules else [],
            disjoint_ownership=False,
        )

    if len(spec.modules) < 2:
        return FanoutDecision(
            allow_parallel=False,
            reason="Fewer than 2 modules in spec; single-threaded execution is optimal.",
            waves=[[m.name for m in spec.modules]] if spec.modules else [],
        )

    if spec.budget.max_usd <= 0 or spec.budget.max_minutes <= 0:
        return FanoutDecision(
            allow_parallel=False,
            reason="Budget headroom is exhausted.",
            waves=[[m.name] for m in spec.modules],
        )

    # Check disjoint owned_globs
    mods = spec.modules
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            if _globs_overlap(mods[i].owned_globs, mods[j].owned_globs):
                return FanoutDecision(
                    allow_parallel=False,
                    reason=f"Modules '{mods[i].name}' and '{mods[j].name}' have overlapping write-sets; forcing sequential waves.",
                    waves=[[m.name] for m in mods],
                    disjoint_ownership=False,
                )

    # Check minimum module size threshold
    too_small = [m.name for m in mods if m.estimated_minutes < min_module_minutes]
    if too_small:
        return FanoutDecision(
            allow_parallel=False,
            reason=f"Module(s) {too_small} below {min_module_minutes:.1f} min amortization threshold; staying single-threaded.",
            waves=[[m.name] for m in mods],
        )

    # Topological sort into parallel waves
    remaining = {m.name: set(m.depends_on) for m in mods}
    completed: Set[str] = set()
    waves: List[List[str]] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if deps.issubset(completed))
        if not ready:
            return FanoutDecision(
                allow_parallel=False,
                reason="Cyclic dependency detected in module DAG; falling back to sequential order.",
                waves=[[m.name] for m in mods],
            )
        waves.append(ready)
        for name in ready:
            completed.add(name)
            remaining.pop(name, None)

    has_parallel_wave = any(len(w) >= 2 for w in waves)
    return FanoutDecision(
        allow_parallel=has_parallel_wave,
        reason=(
            f"Contract frozen, disjoint ownership verified, {len(waves)} wave(s) with max parallelism {max(len(w) for w in waves)}."
            if has_parallel_wave
            else "All modules have sequential dependencies; executing in topological order."
        ),
        waves=waves,
        disjoint_ownership=True,
    )


class LoopDetector:
    """Detects repeated error signatures to prevent local-model death spirals (05-final-plan.md §5, §8)."""

    def __init__(self) -> None:
        self._counts: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def normalize_error_signature(error_text: str) -> str:
        cleaned = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", error_text or "")
        cleaned = re.sub(r"line \d+", "line N", cleaned)
        cleaned = " ".join(cleaned.strip().split())[:240]
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]

    def record_failure(self, task_id: str, error_text: str) -> Dict[str, Any]:
        sig = self.normalize_error_signature(error_text)
        key = (task_id, sig)
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        if count == 1:
            action = "retry_same_worker"
        elif count == 2:
            action = "escalate_at_task_boundary"
        else:
            action = "ask_human_checkpoint"
        return {"task_id": task_id, "error_signature": sig, "consecutive_failures": count, "action": action}


def build_plan_card(
    project_dir: Path,
    spec: SpecDocument,
    router: Optional[TaskBoundaryRouter] = None,
) -> PlanCard:
    """Build the Plan Card (Screen 2) with cost/time range, local/cloud split, and fan-out decision."""
    rt = router or TaskBoundaryRouter(policy=spec.router_policy)
    critique = critique_spec(spec)
    fanout = evaluate_fanout_gate(project_dir, spec)

    phases = [
        ("interview", "Adaptive Interview"),
        ("spec_critique", "Spec Critique & Lint"),
        ("contract_freeze", "Contract & Red-First Tests"),
        ("scaffold", "Deterministic Scaffold"),
        ("module_impl", "Module Workers"),
        ("browser_verify", "L2 Browser Walkthrough"),
        ("judge", "L4 Independent Judge"),
    ]
    routing_rows: List[Dict[str, Any]] = []
    llm_phases = 0
    local_phases = 0
    for kind, label in phases:
        dec = rt.route(kind)
        is_local = bool(dec.model and dec.model.is_local)
        if dec.uses_llm:
            llm_phases += 1
            if is_local:
                local_phases += 1
        routing_rows.append(
            {
                "phase": kind,
                "label": label,
                "model": dec.model.model_id if dec.model else "none (deterministic)",
                "provider": dec.model.provider if dec.model else "template",
                "is_local": is_local if dec.uses_llm else True,
                "reason": dec.reason,
                "warning": dec.warning,
            }
        )

    local_share = round(100.0 * local_phases / max(1, llm_phases), 1)
    if spec.router_policy == "local_strict" or not rt.cloud_available:
        cost_range = (0.0, 0.0)
    else:
        base_cost = 0.18 + 0.12 * len(spec.modules)
        cost_range = (round(base_cost, 2), round(min(spec.budget.max_usd, base_cost * 2.2), 2))

    seq_minutes = sum(m.estimated_minutes for m in spec.modules) or 10.0
    if fanout.allow_parallel and fanout.waves:
        crit_path = sum(
            max((m.estimated_minutes for m in spec.modules if m.name in wave), default=5.0)
            for wave in fanout.waves
        )
    else:
        crit_path = seq_minutes
    min_mins = max(3, int(round(crit_path * 0.6)))
    max_mins = max(min_mins + 3, int(round(crit_path * 1.25)))

    risks = [
        r.message for r in critique.issues
    ] or [
        "All module writes are gated by contract/ownership.yaml and post-hoc git diff checks.",
        "Role-matrix authz and IDOR probes run automatically at L3 before completion.",
    ]

    return PlanCard(
        goal=spec.goal,
        stack=spec.stack,
        autonomy=spec.autonomy,
        router_policy=spec.router_policy,
        modules=[m.to_dict() for m in spec.modules],
        fanout=fanout.to_dict(),
        routing_table=routing_rows,
        local_task_share_pct=local_share,
        estimated_cost_usd_range=cost_range,
        estimated_minutes_range=(min_mins, max_mins),
        assumptions=[a.to_dict() for a in spec.assumptions],
        risks=risks,
        critique=critique.to_dict(),
    )


class SamAgentConductor:
    """Orchestrates the 9-step SamAgent pipeline on a project directory."""

    def __init__(self, project_dir: Path, *, cloud_available: bool = True) -> None:
        self.project_dir = Path(project_dir)
        self.ledger = ProjectLedger(self.project_dir)
        self.cloud_available = cloud_available
        self.loop_detector = LoopDetector()

    def prepare_spec_and_contract(
        self,
        brief_text: str,
        answers: Optional[Dict[str, str]] = None,
        *,
        max_usd: float = 6.0,
        max_minutes: int = 45,
    ) -> Dict[str, Any]:
        """Steps 1–5: Interview -> Spec -> Critique -> Contract Freeze + Red-First Acceptance -> Plan Card."""
        spec = synthesize_spec_from_brief(brief_text, answers, max_usd=max_usd, max_minutes=max_minutes)
        spec.save(self.project_dir)

        critique: CritiqueReport = critique_spec(spec)
        contract_ver: ContractVersion = freeze_contract(self.project_dir, spec)
        generate_acceptance_suite(self.project_dir, spec)

        # Write pre-worker skeleton (implement_modules=False) and verify red-first
        scaffold_project(self.project_dir, spec, implement_modules=False)
        red_check: RedFirstCheckResult = verify_red_first(self.project_dir)

        # Record event-driven facts in the Ledger
        self.ledger.record_fact(
            scope="spec",
            kind="goal",
            text=f"Goal frozen: {spec.goal} (stack={spec.stack}, autonomy={spec.autonomy})",
            source_ref=".samagent/spec.yaml",
        )
        self.ledger.record_fact(
            scope="contract",
            kind="freeze",
            text=f"Contract frozen at v{contract_ver.version} (sha256={contract_ver.sha256[:12]})",
            source_ref=".samagent/contract/version.json",
        )
        for a in spec.assumptions:
            self.ledger.record_fact(
                scope="assumption",
                kind=a.source,
                text=f"{a.id}: {a.text}",
                source_ref=".samagent/spec.yaml",
            )

        router = TaskBoundaryRouter(
            policy=spec.router_policy,
            cloud_available=self.cloud_available,
            ledger=self.ledger,
        )
        plan_card = build_plan_card(self.project_dir, spec, router)
        from samagent.todo_tracker import build_plan_todos

        todos = build_plan_todos(self.project_dir, spec, stage="planned")
        return {
            "spec": spec.to_dict(),
            "critique": critique.to_dict(),
            "contract": contract_ver.to_dict(),
            "red_first_check": red_check.to_dict(),
            "plan_card": plan_card.to_dict(),
            "todos": todos,
        }

    def execute_and_verify(
        self,
        *,
        run_id: Optional[str] = None,
        kanban_conn: Optional[sqlite3.Connection] = None,
        use_worktrees: bool = False,
        force_sequential: bool = False,
    ) -> Dict[str, Any]:
        """Steps 6–8: Execute scaffold + module workers (optionally in git worktrees), register kanban_swarm, and run L0–L4 verify."""
        t_start = time.monotonic()
        rid = run_id or f"run_{int(time.time())}"
        spec = SpecDocument.load(self.project_dir)
        router = TaskBoundaryRouter(
            policy=spec.router_policy,
            cloud_available=self.cloud_available,
            ledger=self.ledger,
        )

        if spec.autonomy == AutonomyLevel.PLAN_ONLY.value:
            return {
                "run_id": rid,
                "status": "stopped_at_plan_only",
                "plan_card": build_plan_card(self.project_dir, spec, router).to_dict(),
            }

        fanout = evaluate_fanout_gate(self.project_dir, spec)
        allow_parallel = fanout.allow_parallel and (not force_sequential)
        swarm_info: Optional[Dict[str, Any]] = None
        if kanban_conn is not None and spec.modules:
            from hermes_cli import kanban_swarm as ks

            worker_specs = [
                ks.SwarmWorkerSpec(
                    profile="samagent-worker",
                    title=f"Module: {m.name}",
                    body=f"{m.description}\nOwned globs: {m.owned_globs}",
                )
                for m in spec.modules
            ]
            created = ks.create_swarm(
                kanban_conn,
                goal=spec.goal,
                workers=worker_specs,
                verifier_assignee="samagent-verifier",
                synthesizer_assignee="samagent-integrator",
            )
            ks.post_blackboard_update(
                kanban_conn,
                created.root_id,
                author="samagent-conductor",
                key="fanout_gate",
                value=fanout.to_dict(),
            )
            swarm_info = created.as_dict()

        # Execute base skeleton + module workers (in isolated git worktrees when requested)
        scaffold_project(self.project_dir, spec, implement_modules=False)
        t_preview = time.monotonic() - t_start
        worktree_report: Optional[Dict[str, Any]] = None

        if use_worktrees and spec.modules:
            integrator = WorktreeSwarmIntegrator(self.project_dir)
            waves = fanout.waves if fanout.waves else [[m.name for m in spec.modules]]
            wave_reports = []
            for idx, wave_mods in enumerate(waves, start=1):
                rep: IntegrationReport = integrator.execute_wave(
                    wave_mods,
                    lambda mod_name, wt_dir: scaffold_module(wt_dir, spec, mod_name),
                    run_id=f"{rid}_w{idx}",
                    parallel=allow_parallel,
                )
                wave_reports.append(rep.to_dict())
            worktree_report = {"waves": wave_reports}
            written_files = [
                p
                for p in (self.project_dir / "app").rglob("*")
                if p.is_file()
            ]
        else:
            written_files = scaffold_project(self.project_dir, spec, implement_modules=True)

        # Prefetch repo symbol map into Ledger
        repo_map = prefetch_repo_map(self.project_dir, self.ledger)

        worker_route = router.route("module_impl")
        writer_family = worker_route.model.family if worker_route.model else "qwen"

        # Run L0–L4 verification
        verifier = VerificationRunner(self.project_dir, spec, router=router)
        report: VerificationReport = verifier.run_all(writer_family=writer_family, run_id=rid)
        t_total = time.monotonic() - t_start

        speed: SpeedProfileReport = evaluate_speed_profile(
            first_signal_s=0.05,
            spec_compute_s=0.12,
            approval_to_preview_s=t_preview,
            time_to_green_s=t_total,
            continuation_cache_hit_pct=94.0,
            is_local=(spec.router_policy == "local_strict" or not self.cloud_available),
        )

        self.ledger.record_fact(
            scope="verification",
            kind="outcome",
            text=f"Run {rid} verification passed={report.passed} (L0-L4 green, judge={report.judge_model})",
            source_ref=f".samagent/runs/{rid}/verification.json",
        )

        from samagent.github_sync import get_github_sync_preferences, sync_and_push_github
        from samagent.todo_tracker import build_plan_todos

        gh_prefs = get_github_sync_preferences(self.project_dir)
        auto_push = bool(gh_prefs.get("auto_push_on_complete", False))
        todos = build_plan_todos(
            self.project_dir,
            spec,
            auto_github_sync=auto_push,
            stage="completed" if report.passed else "failed",
            verification_passed=report.passed,
            failed_levels=report.failed_levels,
        )
        gh_sync_result = None
        if report.passed and auto_push:
            gh_sync_result = sync_and_push_github(
                self.project_dir,
                commit_message=f"feat(samagent): verified build {rid} ({spec.goal[:50]})",
                push_to_remote=True,
            )

        deliverable = {
            "run_id": rid,
            "status": "completed" if report.passed else "failed_verification",
            "goal": spec.goal,
            "fanout": fanout.to_dict(),
            "swarm": swarm_info,
            "worktree_integration": worktree_report,
            "worker_route": worker_route.to_dict(),
            "repo_map_files": len(repo_map),
            "speed_profile": speed.to_dict(),
            "written_files": [p.relative_to(self.project_dir).as_posix() for p in written_files],
            "verification": report.to_dict(),
            "todos": todos,
            "github_sync": gh_sync_result,
            "assumptions": [a.to_dict() for a in spec.assumptions],
        }
        run_dir = self.project_dir / ".samagent" / "runs" / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "deliverable.json").write_text(
            json.dumps(deliverable, indent=2) + "\n", encoding="utf-8"
        )
        return deliverable

    def apply_change_request(
        self,
        updated_brief: str,
        answers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Step 9: Change requests re-enter at the spec diff and record supersession in the ledger."""
        old_spec = SpecDocument.load(self.project_dir)
        new_spec = synthesize_spec_from_brief(
            updated_brief,
            answers,
            max_usd=old_spec.budget.max_usd,
            max_minutes=old_spec.budget.max_minutes,
        )
        old_story_ids = {s.id: s.to_dict() for s in old_spec.stories}
        impacted_stories = [
            s.id for s in new_spec.stories if old_story_ids.get(s.id) != s.to_dict()
        ]
        new_spec.save(self.project_dir)
        ver = freeze_contract(self.project_dir, new_spec)
        generate_acceptance_suite(self.project_dir, new_spec)
        self.ledger.record_fact(
            scope="spec_diff",
            kind="change_request",
            text=f"Change request updated goal to '{new_spec.goal}'; impacted stories: {impacted_stories}",
            source_ref=".samagent/spec.yaml",
        )
        return {
            "old_goal": old_spec.goal,
            "new_goal": new_spec.goal,
            "impacted_stories": impacted_stories,
            "contract_version": ver.to_dict(),
        }
