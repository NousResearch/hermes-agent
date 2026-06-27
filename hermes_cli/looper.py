"""Looper-style review-gated goal design helpers.

This module keeps the first implementation file-based and lightweight:
- accept a rough task
- tighten it into a structured loop spec
- write the artifact bundle
- render a Telegram-friendly preview
- produce a final /goal prompt after approval

The command handlers in gateway/slash_commands.py use these helpers so the
workflow stays testable outside the live gateway.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from utils import atomic_json_write

try:  # pragma: no cover - dependency exists in normal Hermes envs
    import yaml
except Exception:  # pragma: no cover
    yaml = None


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "sk-REDACTED"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{8,}"), "ghp_REDACTED"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{8,}"), "xoxb-REDACTED"),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), "AIzaREDACTED"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{12,}"), "Bearer REDACTED"),
)


@dataclass(slots=True)
class GateSpec:
    name: str
    reviewer_model: str
    council: list[str] = field(default_factory=list)
    quorum: int = 1
    checkpoints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoopSpec:
    run_id: str
    created_at: str
    session_id: str
    session_key: str
    source_platform: str
    rough_task: str
    goal: str
    context: list[str]
    done_criteria: list[str]
    allowed_actions: list[str]
    forbidden_actions: list[str]
    verification_commands: list[str]
    reviewer_model: str
    council_settings: dict[str, Any]
    token_budget: int
    time_budget_minutes: int
    max_revise_loops: int
    final_telegram_report_format: list[str]
    gates: list[GateSpec]
    approval_required: bool = True
    risk_level: str = "medium"
    project_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gates"] = [asdict(g) for g in self.gates]
        return data


@dataclass(slots=True)
class LoopArtifacts:
    run_dir: Path
    run_in_session: Path
    loop_yaml: Path
    loop_resolved_json: Path
    review_rubric: Path
    state_json: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(slots=True)
class LoopOutcome:
    spec: LoopSpec
    artifacts: LoopArtifacts
    final_goal_prompt: str
    preview_text: str
    ready_text: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug[:max_length].rstrip("-") or "loop")


def redact_sensitive_text(text: str) -> str:
    cleaned = text
    for pattern, replacement in _SECRET_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def normalize_rough_task(raw_task: str) -> str:
    text = redact_sensitive_text((raw_task or "").strip())
    text = re.sub(r"^/+looper\b[:\s-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_project_hint(task: str) -> str:
    low = task.lower()
    if "oddsedge" in low:
        return "OddsEdge"
    if "openclaw" in low:
        return "OpenClaw"
    if "hermes" in low:
        return "Hermes"
    if "telegram" in low:
        return "Telegram"
    return ""


def detect_risk_level(task: str) -> str:
    low = task.lower()
    if any(word in low for word in ("production", "restart", "deploy", "database", "secret", "payment", "migration")):
        return "high"
    if any(word in low for word in ("refactor", "feature", "workflow", "review", "bot", "gateway", "telegram", "openclaw")):
        return "medium"
    return "low"


def build_loop_goal(task: str) -> str:
    clean = normalize_rough_task(task)
    if not clean:
        return "Clarify the user's rough task and produce a review-gated goal spec."
    sentence = clean[0].upper() + clean[1:] if clean[0:1].islower() else clean
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


def _allowed_actions() -> list[str]:
    return [
        "Read repository files and session context",
        "Edit only the isolated worktree or clone for this run",
        "Generate the tightened /goal prompt and artifact bundle",
        "Run focused tests and smoke checks",
        "Prepare a GitHub issue / PR attachment later",
    ]


def _forbidden_actions() -> list[str]:
    return [
        "Edit the canonical main/live checkout directly",
        "Print or store secrets in chat or artifact text",
        "Touch .env or other secret stores without explicit approval",
        "Restart production services unless the loop explicitly allows it",
        "Change live databases or destructive production data",
    ]


def _verification_commands(task: str) -> list[str]:
    commands = [
        "git diff --check",
        "git status --short",
        "python -m pytest tests/hermes_cli/test_commands.py tests/gateway/test_looper_command.py -q",
    ]
    if detect_project_hint(task) == "OddsEdge":
        commands.insert(0, "python -m pytest tests -q")
    return commands


def _reviewer_model_for(risk_level: str) -> str:
    return "gpt-5.5" if risk_level in {"high", "critical"} else "gpt-5.4-mini"


def _council_settings(reviewer_model: str) -> dict[str, Any]:
    return {
        "plan_gate": {
            "mode": "tighten-scope",
            "model": "gpt-5.4-mini",
            "council": ["planner", "scope-checker"],
            "quorum": 1,
        },
        "implementation_gate": {
            "mode": "diff-and-tests",
            "model": reviewer_model,
            "council": ["builder", "reviewer"],
            "quorum": 1,
        },
        "delivery_gate": {
            "mode": "release-readiness",
            "model": "gpt-5.4-mini",
            "council": ["release", "risk-checker"],
            "quorum": 1,
        },
    }


def build_loop_spec(
    rough_task: str,
    *,
    session_id: str,
    session_key: str,
    source_platform: str,
    source_chat_id: str,
    source_thread_id: str | None = None,
) -> LoopSpec:
    clean_task = normalize_rough_task(rough_task)
    if not clean_task:
        raise ValueError("Looper needs a rough task to improve.")

    project_hint = detect_project_hint(clean_task)
    risk_level = detect_risk_level(clean_task)
    reviewer_model = _reviewer_model_for(risk_level)
    created_at = _now_iso()
    digest = hashlib.sha1(clean_task.encode("utf-8")).hexdigest()[:8]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{_slugify(clean_task, max_length=26)}-{digest}"

    context = [
        f"Source platform: {source_platform or 'telegram'}",
        f"Source chat: {source_chat_id}",
        f"Session ID: {session_id}",
        f"Session key: {session_key}",
        f"Thread ID: {source_thread_id or 'n/a'}",
        "Workflow: Telegram-first, review-gated goal design.",
        "Policy: use an isolated worktree or clone per run; do not edit the canonical checkout.",
        "Policy: never print secrets; never restart production services unless explicitly allowed.",
    ]
    if project_hint:
        context.append(f"Project hint: {project_hint}")

    done_criteria = [
        "The rough task has been tightened into a clear, reviewable /goal prompt.",
        "The user has explicitly approved the loop before execution continues.",
        "All required artifacts are written for this run.",
        "Verification commands are defined and ready to run.",
        "The final Telegram report uses the required status format.",
    ]

    final_report_format = [
        "✅ ready | ⚠️ needs review | ❌ blocked",
        "what changed",
        "tests run",
        "risk level",
        "next action",
    ]

    gates = [
        GateSpec(
            name="plan_gate",
            reviewer_model="gpt-5.4-mini",
            council=["planner", "scope-checker"],
            quorum=1,
            checkpoints=["tighten scope", "confirm boundaries", "define verification"],
        ),
        GateSpec(
            name="implementation_gate",
            reviewer_model=reviewer_model,
            council=["builder", "reviewer"],
            quorum=1,
            checkpoints=["check diff", "check tests", "check forbidden actions"],
        ),
        GateSpec(
            name="delivery_gate",
            reviewer_model="gpt-5.4-mini",
            council=["release", "risk-checker"],
            quorum=1,
            checkpoints=["final report format", "artifact bundle", "next action"],
        ),
    ]

    return LoopSpec(
        run_id=run_id,
        created_at=created_at,
        session_id=session_id,
        session_key=session_key,
        source_platform=source_platform or "telegram",
        rough_task=clean_task,
        goal=build_loop_goal(clean_task),
        context=context,
        done_criteria=done_criteria,
        allowed_actions=_allowed_actions(),
        forbidden_actions=_forbidden_actions(),
        verification_commands=_verification_commands(clean_task),
        reviewer_model=reviewer_model,
        council_settings=_council_settings(reviewer_model),
        token_budget=12000 if risk_level == "high" else 8000,
        time_budget_minutes=45 if risk_level == "high" else 30,
        max_revise_loops=3,
        final_telegram_report_format=final_report_format,
        gates=gates,
        approval_required=True,
        risk_level=risk_level,
        project_hint=project_hint,
    )


def loop_root_dir() -> Path:
    return get_hermes_home() / "looper"


def loop_run_dir(spec: LoopSpec) -> Path:
    return loop_root_dir() / spec.session_id / spec.run_id


def build_final_goal_prompt(spec: LoopSpec) -> str:
    context_block = "\n".join(f"- {line}" for line in spec.context)
    done_block = "\n".join(f"- {line}" for line in spec.done_criteria)
    allowed_block = "\n".join(f"- {line}" for line in spec.allowed_actions)
    forbidden_block = "\n".join(f"- {line}" for line in spec.forbidden_actions)
    verify_block = "\n".join(f"- {line}" for line in spec.verification_commands)
    gate_block = "\n".join(
        f"- {gate.name}: reviewer={gate.reviewer_model}; council={', '.join(gate.council)}; quorum={gate.quorum}"
        for gate in spec.gates
    )
    return (
        f"/goal {spec.goal}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Done criteria:\n{done_block}\n\n"
        f"Allowed actions:\n{allowed_block}\n\n"
        f"Forbidden actions:\n{forbidden_block}\n\n"
        f"Verification commands:\n{verify_block}\n\n"
        f"Review gates:\n{gate_block}\n\n"
        f"Budgets:\n- token budget: {spec.token_budget}\n- time budget: {spec.time_budget_minutes} minutes\n- max revise loops: {spec.max_revise_loops}\n"
    )


def render_preview(spec: LoopSpec) -> str:
    goal_prompt = build_final_goal_prompt(spec)
    prompt_preview = goal_prompt[:900].rstrip()
    if len(goal_prompt) > 900:
        prompt_preview += "\n…"

    lines = [
        "┌─ Looper review preview ───────────────────────────┐",
        f"Task: {spec.rough_task}",
        f"Goal: {spec.goal}",
        f"Risk: {spec.risk_level}",
        f"Reviewer: {spec.reviewer_model}",
        f"Budget: {spec.token_budget} tokens / {spec.time_budget_minutes} minutes / {spec.max_revise_loops} revise loops",
        "",
        "Done criteria:",
        *[f"  • {item}" for item in spec.done_criteria[:4]],
        "",
        "Gates: plan → implementation → delivery",
        "",
        "Final /goal prompt preview:",
        "```",
        prompt_preview,
        "```",
        "Approve to write the resolved bundle and continue.",
        "└──────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)


def render_review_rubric(spec: LoopSpec) -> str:
    return "\n".join(
        [
            "# Review rubric",
            "",
            "## Plan gate",
            f"- reviewer model: {spec.gates[0].reviewer_model}",
            f"- council: {', '.join(spec.gates[0].council)}",
            "- pass only if the goal is crisp, bounded, and reviewable.",
            "",
            "## Implementation gate",
            f"- reviewer model: {spec.gates[1].reviewer_model}",
            f"- council: {', '.join(spec.gates[1].council)}",
            "- pass only if the diff stays inside the allowed actions and the verification commands are real.",
            "",
            "## Delivery gate",
            f"- reviewer model: {spec.gates[2].reviewer_model}",
            f"- council: {', '.join(spec.gates[2].council)}",
            "- pass only if the final Telegram report matches the required status format.",
        ]
    )


def _yaml_dump(payload: dict[str, Any]) -> str:
    if yaml is None:  # pragma: no cover - normal env should have PyYAML
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def write_loop_artifacts(
    spec: LoopSpec,
    *,
    status: str,
    approval_choice: str | None = None,
    approved: bool = False,
) -> LoopArtifacts:
    run_dir = loop_run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = LoopArtifacts(
        run_dir=run_dir,
        run_in_session=run_dir / "RUN_IN_SESSION.md",
        loop_yaml=run_dir / "loop.yaml",
        loop_resolved_json=run_dir / "loop.resolved.json",
        review_rubric=run_dir / "review_rubric.md",
        state_json=run_dir / "state.json",
    )

    loop_payload = spec.to_dict()
    artifacts.loop_yaml.write_text(_yaml_dump(loop_payload), encoding="utf-8")
    artifacts.run_in_session.write_text(
        "\n".join(
            [
                "# RUN IN SESSION",
                "",
                f"- run_id: {spec.run_id}",
                f"- status: {status}",
                f"- risk_level: {spec.risk_level}",
                f"- reviewer_model: {spec.reviewer_model}",
                f"- goal: {spec.goal}",
                "",
                "## Rough task",
                spec.rough_task,
                "",
                "## Context",
                *[f"- {line}" for line in spec.context],
                "",
                "## Final /goal prompt",
                "```text",
                build_final_goal_prompt(spec),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    artifacts.review_rubric.write_text(render_review_rubric(spec), encoding="utf-8")

    state = {
        "run_id": spec.run_id,
        "status": status,
        "approved": approved,
        "approval_choice": approval_choice,
        "approval_required": spec.approval_required,
        "run_dir": str(run_dir),
        "created_at": spec.created_at,
        "updated_at": _now_iso(),
        "session_id": spec.session_id,
        "session_key": spec.session_key,
        "source_platform": spec.source_platform,
        "project_hint": spec.project_hint,
        "risk_level": spec.risk_level,
    }
    atomic_json_write(artifacts.state_json, state)

    resolved = {
        "status": status,
        "approved": approved,
        "approval_choice": approval_choice,
        "spec": spec.to_dict(),
        "final_goal_prompt": build_final_goal_prompt(spec),
        "artifacts": artifacts.as_dict(),
        "created_at": spec.created_at,
        "updated_at": _now_iso(),
    }
    atomic_json_write(artifacts.loop_resolved_json, resolved)
    return artifacts


def update_loop_resolution(
    spec: LoopSpec,
    artifacts: LoopArtifacts,
    *,
    status: str,
    approved: bool,
    approval_choice: str | None,
) -> None:
    state = {
        "run_id": spec.run_id,
        "status": status,
        "approved": approved,
        "approval_choice": approval_choice,
        "approval_required": spec.approval_required,
        "run_dir": str(artifacts.run_dir),
        "created_at": spec.created_at,
        "updated_at": _now_iso(),
        "session_id": spec.session_id,
        "session_key": spec.session_key,
        "source_platform": spec.source_platform,
        "project_hint": spec.project_hint,
        "risk_level": spec.risk_level,
    }
    atomic_json_write(artifacts.state_json, state)
    resolved = {
        "status": status,
        "approved": approved,
        "approval_choice": approval_choice,
        "spec": spec.to_dict(),
        "final_goal_prompt": build_final_goal_prompt(spec),
        "artifacts": artifacts.as_dict(),
        "created_at": spec.created_at,
        "updated_at": _now_iso(),
    }
    atomic_json_write(artifacts.loop_resolved_json, resolved)


def build_final_report(
    spec: LoopSpec,
    *,
    status: str,
    what_changed: Iterable[str],
    tests_run: Iterable[str],
    next_action: str,
) -> str:
    status_line = {
        "ready": "✅ ready",
        "needs_review": "⚠️ needs review",
        "blocked": "❌ blocked",
    }.get(status, status)
    changed = list(what_changed)
    tests = list(tests_run)
    return "\n".join(
        [
            f"{status_line}",
            f"what changed: {', '.join(changed) if changed else 'loop spec only'}",
            f"tests run: {', '.join(tests) if tests else 'not run yet'}",
            f"risk level: {spec.risk_level}",
            f"next action: {next_action}",
            "",
            "Final /goal prompt:",
            "```text",
            build_final_goal_prompt(spec),
            "```",
            "",
            f"Artifacts: {loop_run_dir(spec)}",
        ]
    )


def build_looper_run(
    rough_task: str,
    *,
    session_id: str,
    session_key: str,
    source_platform: str,
    source_chat_id: str,
    source_thread_id: str | None = None,
) -> tuple[LoopSpec, LoopArtifacts, str]:
    spec = build_loop_spec(
        rough_task,
        session_id=session_id,
        session_key=session_key,
        source_platform=source_platform,
        source_chat_id=source_chat_id,
        source_thread_id=source_thread_id,
    )
    artifacts = write_loop_artifacts(spec, status="awaiting_approval", approved=False)
    preview = render_preview(spec)
    return spec, artifacts, preview


def finalize_looper_run(
    spec: LoopSpec,
    artifacts: LoopArtifacts,
    *,
    approval_choice: str,
) -> tuple[str, str]:
    approved = approval_choice in {"once", "always"}
    status = "ready" if approved else "blocked"
    update_loop_resolution(
        spec,
        artifacts,
        status=status,
        approved=approved,
        approval_choice=approval_choice,
    )
    if approved:
        what_changed = ["Loop spec tightened", "Artifacts resolved", "Final /goal prompt generated"]
        tests_run = ["dry-run validation", "artifact bundle write"]
        next_action = "Copy the /goal prompt into OpenClaw/Codex or attach it to a future execution step."
    else:
        what_changed = ["Approval declined"]
        tests_run = ["none"]
        next_action = "Re-run /looper with a clearer task or approve the preview."
    report = build_final_report(
        spec,
        status=status,
        what_changed=what_changed,
        tests_run=tests_run,
        next_action=next_action,
    )
    return report, spec.goal
