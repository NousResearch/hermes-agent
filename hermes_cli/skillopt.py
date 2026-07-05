"""Manual SkillOpt CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from agent.skillopt_state import (
    append_skillopt_rejection,
    load_skillopt_proposal,
    mark_skillopt_adopted,
    stage_skillopt_proposal,
    update_skillopt_evaluation,
)
from tools.skill_usage import skillopt_candidate_report


def _find_skill_path(skill_name: str) -> Path:
    from agent.skill_utils import is_excluded_skill_path, parse_frontmatter

    base = get_hermes_home() / "skills"
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        frontmatter, _body = parse_frontmatter(text)
        parsed_name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
        if parsed_name == skill_name or skill_md.parent.name == skill_name:
            return skill_md
    raise FileNotFoundError(f"skill not found: {skill_name}")


def _run_dir_for_id(run_id: str) -> Path:
    from agent.skillopt_state import _validate_skill_name

    try:
        safe_id = _validate_skill_name(str(run_id))
    except ValueError as exc:
        raise ValueError("invalid SkillOpt run_id") from exc
    runs_root = (get_hermes_home() / "skillopt" / "runs").resolve()
    path = (runs_root / safe_id).resolve()
    if path == runs_root or runs_root not in path.parents:
        raise ValueError("invalid SkillOpt run_id")
    if not path.exists():
        raise FileNotFoundError(f"SkillOpt run not found: {run_id}")
    return path


def _ensure_skill_path_is_adoptable(path: Path) -> Path:
    skills_root = (get_hermes_home() / "skills").resolve()
    resolved = path.expanduser().resolve()
    if resolved.name != "SKILL.md" or skills_root not in resolved.parents:
        raise RuntimeError("proposal current_skill_path is outside Hermes skills directory")
    return resolved


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_status(args) -> int:
    rows = skillopt_candidate_report(limit=getattr(args, "limit", None))
    skill = getattr(args, "skill", None)
    if skill:
        rows = [r for r in rows if r.get("name") == skill]
    if getattr(args, "json", False):
        _print_json(rows)
        return 0
    if not rows:
        print("No SkillOpt candidates found.")
        return 0
    print("SkillOpt candidates:")
    for row in rows:
        reasons = ", ".join(row.get("reasons") or [])
        print(f"- {row['name']}: score={row['skillopt_score']} reasons={reasons}")
    return 0


def _cmd_distill(args) -> int:
    from agent.skillopt_harvest import distill_trace_to_skill

    trace_path = Path(getattr(args, "trace")).expanduser()
    distilled = distill_trace_to_skill(json.loads(trace_path.read_text(encoding="utf-8")))
    out = getattr(args, "out", None)
    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(distilled["skill_markdown"], encoding="utf-8")
        print(f"Wrote distilled skill draft: {out_path}")
    else:
        print(distilled["skill_markdown"])
    return 0


def _cmd_propose(args) -> int:
    skill_name = getattr(args, "skill")
    skill_path = _find_skill_path(skill_name)
    candidate_path = Path(getattr(args, "candidate")).expanduser()
    candidate = candidate_path.read_text(encoding="utf-8")
    staged = stage_skillopt_proposal(
        skill_name=skill_name,
        current_skill_path=skill_path,
        candidate_skill=candidate,
        source={"from_session": getattr(args, "from_session", None), "candidate_path": str(candidate_path)},
        rationale=getattr(args, "rationale", "") or "",
    )
    print(f"Staged SkillOpt proposal: {staged.run_id}")
    print(f"Run directory: {staged.run_dir}")
    print("Live skill was not modified; run `hermes skillopt evaluate` before `adopt`.")
    return 0


def _cmd_reject(args) -> int:
    run_dir = _run_dir_for_id(getattr(args, "run_id"))
    append_skillopt_rejection(run_dir, reason=getattr(args, "reason", ""), reviewer=getattr(args, "reviewer", "cli"))
    print(f"Rejected SkillOpt proposal: {getattr(args, 'run_id')}")
    return 0


def _cmd_adopt(args) -> int:
    run_dir = _run_dir_for_id(getattr(args, "run_id"))
    loaded = load_skillopt_proposal(run_dir)
    scores = loaded.proposal.get("scores") if isinstance(loaded.proposal, dict) else {}
    if not isinstance(scores, dict):
        scores = {}
    if loaded.status != "evaluated":
        raise RuntimeError("SkillOpt proposal must be evaluated successfully before adopt")
    if (
        not scores.get("heldout_ready")
        or int(scores.get("total") or 0) <= 0
        or int(scores.get("failed") or 0) != 0
        or float(scores.get("score") or 0.0) <= 0.0
    ):
        raise RuntimeError("SkillOpt proposal must have passing evaluation scores before adopt")
    current_path = _ensure_skill_path_is_adoptable(loaded.current_skill_path)
    current_text = current_path.read_text(encoding="utf-8")
    import hashlib

    current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
    if current_hash != loaded.current_sha256:
        raise RuntimeError("live skill hash changed since proposal was staged; refusing adopt")
    current_path.write_text(loaded.candidate_skill, encoding="utf-8")
    mark_skillopt_adopted(run_dir)
    print(f"Adopted SkillOpt proposal: {loaded.run_id}")
    return 0


def _cmd_show(args) -> int:
    loaded = load_skillopt_proposal(_run_dir_for_id(getattr(args, "run_id")))
    if getattr(args, "json", False):
        _print_json(loaded.proposal)
    else:
        print(f"Run: {loaded.run_id}")
        print(f"Skill: {loaded.skill_name}")
        print(f"Status: {loaded.status}")
        print(f"Candidate: {loaded.run_dir / 'candidate.SKILL.md'}")
    return 0


def _cmd_evaluate(args) -> int:
    run_dir = _run_dir_for_id(getattr(args, "run_id"))
    from agent.skillopt_scoring import score_verification_evidence

    scores = score_verification_evidence(
        root=getattr(args, "root"),
        session_id=getattr(args, "session_id", None),
        min_events=getattr(args, "min_events", 2),
    )
    loaded = update_skillopt_evaluation(run_dir, scores)
    print(f"Evaluated SkillOpt proposal: {loaded.run_id}")
    print(f"Status: {loaded.status}")
    print(f"Score: {scores.get('score')} ({scores.get('passed')}/{scores.get('total')} passed)")
    return 0 if loaded.status == "evaluated" else 1


def cmd_skillopt(args) -> int:
    command = getattr(args, "skillopt_command", None) or "status"
    if command == "status":
        return _cmd_status(args)
    if command == "distill":
        return _cmd_distill(args)
    if command == "propose":
        return _cmd_propose(args)
    if command == "reject":
        return _cmd_reject(args)
    if command == "adopt":
        return _cmd_adopt(args)
    if command == "show":
        return _cmd_show(args)
    if command == "evaluate":
        return _cmd_evaluate(args)
    raise SystemExit(f"unknown skillopt command: {command}")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser("skillopt", help="Stage and inspect SkillOpt skill-improvement proposals")
    sub = parser.add_subparsers(dest="skillopt_command")
    status = sub.add_parser("status", help="List ranked SkillOpt candidate skills")
    status.add_argument("skill", nargs="?", default=None)
    status.add_argument("--limit", type=int, default=None)
    status.add_argument("--json", action="store_true")
    distilled = sub.add_parser("distill", help="Distill a JSON trace into a staged candidate skill draft")
    distilled.add_argument("trace", help="Path to a JSON trace with skill_name and messages")
    distilled.add_argument("--out", default=None, help="Write candidate SKILL.md to this path")
    propose = sub.add_parser("propose", help="Stage a candidate SKILL.md without mutating the live skill")
    propose.add_argument("skill")
    propose.add_argument("--candidate", required=True, help="Path to candidate SKILL.md content")
    propose.add_argument("--rationale", default="")
    propose.add_argument("--from-session", default=None)
    show = sub.add_parser("show", help="Show a staged proposal")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true")
    reject = sub.add_parser("reject", help="Reject a staged proposal")
    reject.add_argument("run_id")
    reject.add_argument("--reason", default="")
    reject.add_argument("--reviewer", default="cli")
    adopt = sub.add_parser("adopt", help="Adopt an evaluated proposal if scores pass and the live skill hash still matches")
    adopt.add_argument("run_id")
    evaluate = sub.add_parser("evaluate", help="Evaluate a staged proposal against verification evidence")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--root", required=True, help="Workspace root whose verification evidence should be scored")
    evaluate.add_argument("--session-id", default=None)
    evaluate.add_argument("--min-events", type=int, default=2)
    parser.set_defaults(func=cmd_skillopt)
