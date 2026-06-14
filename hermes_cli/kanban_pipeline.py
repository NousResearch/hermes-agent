"""Kanban Pipeline — the 4-profile orchestration pattern as a reusable helper.

The pipeline pattern is the durable, board-persisted shape of the
``coder + reviewer + researcher + analyst`` workflow validated on
2026-06-13 (see ``hermes-multi-profile-orchestration`` skill). It writes
a small task graph into the existing kanban kernel:

    planning root (completed immediately, stays as the shared blackboard)
        ├─ coder      (parallel)
        ├─ reviewer   (parallel)
        ├─ researcher (parallel)
        └─ analyst    (parallel)

This is deliberately simpler than ``kanban_swarm`` (no verifier, no
synthesizer): the 4-profile pattern's verifier is the reviewer and the
synthesizer is the analyst, baked in by role, so the topology collapses
to one level of fan-out.

Presets are pure data so new patterns (e.g. ``2-profile {coder, reviewer}``,
``6-profile {coder, reviewer, researcher, analyst, security, perf}``)
can be added without code changes — just register the preset in
``PRESETS`` and the CLI auto-discovers it.

Why a separate module (not folded into ``kanban_swarm.py``): the
swarm module's contract is "parallel workers → verifier → synthesizer",
which fits the LLM-judge pattern. The 4-profile pipeline has no
separate verifier; the role IS the verification. Mixing the two
shapes would force every swarm to declare a verifier that sometimes
overlaps with a worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb


# ----- presets -----------------------------------------------------------

@dataclass(frozen=True)
class PipelineRoleSpec:
    """One role in a pipeline preset."""

    profile: str
    """Profile name to assign (becomes the card's ``assignee``)."""

    title_template: str
    """Title template for the child card. ``{goal}`` is substituted; the
    resulting title is what the dispatcher and the worker see."""

    body_template: str
    """Body template for the child card. ``{goal}`` is substituted."""

    model: Optional[str] = None
    """Per-task model override (becomes the card's ``model_override``).
    Mirrors ``hermes kanban create --model``."""

    skills: list[str] = field(default_factory=list)
    """Skills to force-load into the worker (passed via ``--skills``)."""


# 4-profile preset — the validated pattern.
#
# Each role's body embeds a directive shape and an output file path. The
# body is the worker's contract: read it, do the work, write the artifact.
# We deliberately keep the body templates in code (not in the caller's
# CLI args) so the same preset produces the same body across every
# invocation, and so a "swap the demo" change is a one-line edit here.
FOUR_PROFILE_PRESET: dict[str, PipelineRoleSpec] = {
    "coder": PipelineRoleSpec(
        profile="coder",
        title_template="Coder: {goal} (kimi-k2.7-code)",
        body_template=(
            "EXECUTE. No clarifying questions. Use the test-driven-development skill.\n\n"
            "Goal: {goal}\n\n"
            "Process (TDD per the test-driven-development skill):\n"
            "1. Write a stub that raises NotImplementedError, plus at least 5 failing\n"
            "   assert tests so `python <file>` exits non-zero on a red run.\n"
            "2. Run the file; confirm it exits non-zero.\n"
            "3. Replace the stub with a real implementation.\n"
            "4. Run the file again; confirm it exits 0.\n"
            "5. Return a 3-line summary: complexity, exit code, any deviations.\n"
        ),
        model="kimi-k2.7-code",
        skills=["test-driven-development"],
    ),
    "reviewer": PipelineRoleSpec(
        profile="reviewer",
        title_template="Reviewer: code review for {goal} (glm-5.1)",
        body_template=(
            "EXECUTE. No questions. No interactive back-and-forth.\n\n"
            "Goal: {goal}\n\n"
            "Read the coder's artifact and write a numbered list of issues, each with\n"
            "severity (blocker / major / minor / nit) and a one-line fix suggestion.\n"
            "At least 3 items, under 400 words. If no issues, write a one-paragraph\n"
            "'approved' justification explaining what you checked.\n\n"
            "Do not modify the coder's source. Return a 2-line summary: form chosen\n"
            "(numbered / approved) and word count.\n"
        ),
        model="glm-5.1",
        skills=["requesting-code-review"],
    ),
    "researcher": PipelineRoleSpec(
        profile="researcher",
        title_template="Researcher: brief for {goal} (deepseek-v4-pro)",
        body_template=(
            "EXECUTE. No clarifying questions. Do the research and write the file.\n\n"
            "Goal: {goal}\n\n"
            "Use the web tool. Produce a brief in exactly 5 markdown bullets.\n"
            "Each bullet <= 30 words. At least 3 bullets must include a URL\n"
            "citation. No preamble, no conclusion.\n\n"
            "Return a 2-line summary: bullet count and URL count.\n"
        ),
        model="deepseek-v4-pro",
        skills=[],
    ),
    "analyst": PipelineRoleSpec(
        profile="analyst",
        title_template="Analyst: benchmark for {goal} (deepseek-v4-flash)",
        body_template=(
            "EXECUTE. No questions.\n\n"
            "Goal: {goal}\n\n"
            "After the coder's artifact exists:\n"
            "1. Write a benchmark that imports the artifact, runs both the artifact's\n"
            "   implementation and a baseline (e.g. heapq.nlargest, Counter.most_common),\n"
            "   on (n=1_000,k=10), (n=100_000,k=10), (n=100_000,k=1_000). 5 reps per case.\n"
            "   Report median ms in a markdown table.\n"
            "2. Run the benchmark and confirm the table file exists.\n\n"
            "Return a 3-line summary: impl strategy, n=100k,k=1000 medians, winner.\n"
        ),
        model="deepseek-v4-flash",
        skills=[],
    ),
}


PRESETS: dict[str, dict[str, PipelineRoleSpec]] = {
    "4-profile": FOUR_PROFILE_PRESET,
}


# ----- outputs -----------------------------------------------------------

@dataclass(frozen=True)
class PipelineCreated:
    """IDs produced by :func:`create_pipeline`."""

    root_id: str
    worker_ids: dict[str, str]
    """``{role_name: task_id}`` — one per role in the preset."""

    preset: str
    goal: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "worker_ids": dict(self.worker_ids),
            "preset": self.preset,
            "goal": self.goal,
        }


# ----- helpers -----------------------------------------------------------

def _require_text(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _resolve_goal(goal: str) -> str:
    """Normalize the goal string. The kanban task title and each child's
    body embed ``{goal}``; collapse internal newlines so the title stays
    a single readable line."""
    g = _require_text(goal, "goal")
    return " ".join(line.strip() for line in g.splitlines() if line.strip()) or g


def _apply_overrides(
    role_name: str,
    spec: PipelineRoleSpec,
    overrides: Optional[dict[str, dict[str, Any]]],
) -> PipelineRoleSpec:
    """Apply per-role overrides from the caller. Supported keys: ``model``,
    ``skills`` (list), ``title_template``, ``body_template``. The
    ``profile`` field is intentionally NOT overridable — the preset fixes
    which profile handles which role, and re-pointing a role to a
    different profile is what would make the preset a different preset.
    """
    if not overrides:
        return spec
    role_over = overrides.get(role_name) or {}
    if not role_over:
        return spec
    return PipelineRoleSpec(
        profile=spec.profile,
        title_template=role_over.get("title_template", spec.title_template),
        body_template=role_over.get("body_template", spec.body_template),
        model=role_over.get("model", spec.model),
        skills=list(role_over.get("skills", spec.skills)),
    )


def _pipeline_blackboard(root_id: str, goal: str, preset: str, role_specs: dict[str, PipelineRoleSpec]) -> str:
    """Structured comment payload recorded on the root so a future reader
    can reconstruct the topology without re-running the helper. Symmetric
    with :func:`hermes_cli.kanban_swarm.latest_blackboard`."""
    return json.dumps({
        "kind": "kanban_pipeline_v1",
        "preset": preset,
        "root_id": root_id,
        "goal": goal,
        "roles": {
            r: {
                "profile": s.profile,
                "model": s.model,
                "skills": list(s.skills),
            }
            for r, s in role_specs.items()
        },
    }, ensure_ascii=False)


def create_pipeline(
    conn: sqlite3.Connection,
    *,
    goal: str,
    preset: str = "4-profile",
    overrides: Optional[dict[str, dict[str, Any]]] = None,
    root_title: Optional[str] = None,
    tenant: Optional[str] = None,
    created_by: str = "pipeline-orchestrator",
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
) -> PipelineCreated:
    """Create a durable Kanban pipeline graph for the given goal.

    The graph is immediately dispatchable: the planning root is marked
    ``done`` with topology metadata, the 4 children start in ``ready``
    (the parent being ``done`` unblocks them), and the dispatcher picks
    them up in parallel subject to
    ``delegation.max_concurrent_children``.

    Returns:
      :class:`PipelineCreated` with the root id and the per-role
      ``{role: task_id}`` map. The CLI surfaces this as JSON.

    Raises:
      ValueError: unknown preset, empty goal, or no roles in the preset.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; available: {sorted(PRESETS.keys())}"
        )
    goal_text = _resolve_goal(goal)
    base_specs = PRESETS[preset]
    if not base_specs:
        raise ValueError(f"preset {preset!r} has no roles")

    final_specs: dict[str, PipelineRoleSpec] = {
        role: _apply_overrides(role, spec, overrides)
        for role, spec in base_specs.items()
    }

    # 1. Create the planning root. Marked ``done`` immediately so children
    # can promote; it remains the shared blackboard for the duration of
    # the pipeline.
    root_title_final = root_title or f"Pipeline[{preset}]: {goal_text[:80]}"
    root = kb.create_task(
        conn,
        title=root_title_final,
        body=(
            "Kanban Pipeline v1 planning/root card. This card is completed "
            "immediately so the 4 child roles can start while it remains "
            "the shared blackboard and audit anchor.\n\n"
            f"Goal:\n{goal_text}\n\n"
            f"Preset: {preset}\n"
            f"Roles: {', '.join(final_specs.keys())}\n"
        ),
        assignee=created_by,
        created_by=created_by,
        tenant=tenant,
        priority=priority,
        idempotency_key=idempotency_key,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        skills=["kanban-orchestrator"],
    )

    # 2. Build the 4 child cards in stable role order, each with a
    # ``parents=[root]`` edge so it cannot start until the root is
    # released (it already is, so it lands in ``ready``).
    worker_ids: dict[str, str] = {}
    for role_name in ("coder", "reviewer", "researcher", "analyst"):
        if role_name not in final_specs:
            continue
        spec = final_specs[role_name]
        title = spec.title_template.format(goal=goal_text)
        body = spec.body_template.format(goal=goal_text)
        tid = kb.create_task(
            conn,
            title=title,
            body=body,
            assignee=spec.profile,
            created_by=created_by,
            parents=[root],
            tenant=tenant,
            priority=priority,
            workspace_kind=workspace_kind,
            workspace_path=workspace_path,
            skills=spec.skills or None,
            model_override=spec.model,
        )
        worker_ids[role_name] = tid

    # 3. Mark the root done with topology metadata so a later reader can
    # reconstruct the graph from a single task id.
    kb.complete_task(
        conn,
        root,
        summary=f"Pipeline[{preset}] topology planned; {len(worker_ids)} roles ready.",
        metadata={
            "kind": "kanban_pipeline_v1",
            "preset": preset,
            "goal": goal_text,
            "role_count": len(worker_ids),
            "worker_ids": dict(worker_ids),
        },
    )

    # 4. Record a structured blackboard comment for human readers. Kept
    # separate from the metadata blob so the kanban UI's comment thread
    # is the canonical human-readable log.
    try:
        kb.add_comment(
            conn,
            root,
            created_by,
            "[pipeline:blackboard] " + _pipeline_blackboard(root, goal_text, preset, final_specs),
        )
    except AttributeError:
        # Older kanban_db without add_comment — best-effort, the
        # topology is already in the task metadata.
        pass

    return PipelineCreated(
        root_id=root,
        worker_ids=worker_ids,
        preset=preset,
        goal=goal_text,
    )
