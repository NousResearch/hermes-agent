"""Durable staged artifacts for SkillOpt-style skill proposals.

This module stores reviewable candidate skill updates under
``$HERMES_HOME/skillopt/runs/<run_id>/``. It never mutates the live skill; adopt
or reject decisions stay explicit and reversible in higher-level code.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class SkillOptProposal:
    """Loaded or newly staged SkillOpt proposal."""

    run_id: str
    run_dir: Path
    skill_name: str
    current_skill_path: Path
    current_sha256: str
    candidate_sha256: str
    candidate_skill: str
    status: str
    proposal: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_skill_name(skill_name: str) -> str:
    value = str(skill_name or "").strip()
    if not _SKILL_NAME_RE.fullmatch(value) or ".." in value or "/" in value or "\\" in value:
        raise ValueError("invalid skill_name")
    return value


def _runs_dir() -> Path:
    return get_hermes_home() / "skillopt" / "runs"


def _run_id(skill_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{skill_name}-{stamp}-{time.time_ns() % 1_000_000:06d}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
        Path(tmp).replace(path)
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        finally:
            raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _proposal_from_payload(run_dir: Path, payload: dict[str, Any], candidate_skill: str) -> SkillOptProposal:
    return SkillOptProposal(
        run_id=str(payload["run_id"]),
        run_dir=run_dir,
        skill_name=str(payload["skill_name"]),
        current_skill_path=Path(str(payload["current_skill_path"])),
        current_sha256=str(payload["current_sha256"]),
        candidate_sha256=str(payload["candidate_sha256"]),
        candidate_skill=candidate_skill,
        status=str(payload.get("status", "staged")),
        proposal=payload,
    )


def stage_skillopt_proposal(
    *,
    skill_name: str,
    current_skill_path: str | Path,
    candidate_skill: str,
    edits: list[dict[str, Any]] | None = None,
    scores: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    rationale: str = "",
    run_id: str | None = None,
) -> SkillOptProposal:
    """Create reviewable staged artifacts for a candidate skill update.

    The live ``current_skill_path`` is read for hashing only. It is never
    modified. The candidate document is stored as ``candidate.SKILL.md`` and the
    proposal metadata is stored as JSON for later scoring/adopt/reject flows.
    """

    safe_name = _validate_skill_name(skill_name)
    skill_path = Path(current_skill_path).expanduser()
    current_skill = skill_path.read_text(encoding="utf-8")
    candidate = str(candidate_skill)
    rid = _validate_skill_name(run_id) if run_id else _run_id(safe_name)
    run_dir = _runs_dir() / rid
    run_dir.mkdir(parents=True, exist_ok=False)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "skill_name": safe_name,
        "status": "staged",
        "created_at": _now_iso(),
        "current_skill_path": str(skill_path),
        "current_sha256": _sha256(current_skill),
        "candidate_sha256": _sha256(candidate),
        "edits": list(edits or []),
        "scores": dict(scores or {}),
        "source": dict(source or {}),
        "rationale": rationale,
    }
    _atomic_write(run_dir / "candidate.SKILL.md", candidate)
    _write_json(run_dir / "proposal.json", payload)
    _atomic_write(
        run_dir / "meta.md",
        "# SkillOpt Proposal\n\n"
        f"- Run: `{rid}`\n"
        f"- Skill: `{safe_name}`\n"
        f"- Status: staged\n"
        f"- Created: {payload['created_at']}\n\n"
        f"## Rationale\n\n{rationale or '(none)'}\n",
    )
    _atomic_write(run_dir / "rejected.jsonl", "")
    return _proposal_from_payload(run_dir, payload, candidate)


def load_skillopt_proposal(run_dir: str | Path) -> SkillOptProposal:
    """Load a staged proposal and fail closed if candidate content was changed."""

    path = Path(run_dir)
    payload = json.loads((path / "proposal.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported SkillOpt proposal schema")
    candidate = (path / "candidate.SKILL.md").read_text(encoding="utf-8")
    if _sha256(candidate) != payload.get("candidate_sha256"):
        raise ValueError("candidate hash mismatch")
    return _proposal_from_payload(path, payload, candidate)


def append_skillopt_rejection(run_dir: str | Path, *, reason: str, reviewer: str = "") -> None:
    """Append a rejection record and mark the proposal rejected."""

    path = Path(run_dir)
    loaded = load_skillopt_proposal(path)
    record = {
        "rejected_at": _now_iso(),
        "reason": str(reason),
        "reviewer": str(reviewer),
    }
    with (path / "rejected.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    payload = dict(loaded.proposal)
    payload["status"] = "rejected"
    payload["rejected_at"] = record["rejected_at"]
    payload["rejection_reason"] = record["reason"]
    _write_json(path / "proposal.json", payload)


def mark_skillopt_adopted(run_dir: str | Path) -> None:
    """Mark a proposal adopted after the caller has updated the live skill."""
    path = Path(run_dir)
    loaded = load_skillopt_proposal(path)
    payload = dict(loaded.proposal)
    payload["status"] = "adopted"
    payload["adopted_at"] = _now_iso()
    _write_json(path / "proposal.json", payload)


def update_skillopt_evaluation(run_dir: str | Path, scores: dict[str, Any]) -> SkillOptProposal:
    """Persist evaluation scores and gate status for a staged proposal."""
    path = Path(run_dir)
    loaded = load_skillopt_proposal(path)
    payload = dict(loaded.proposal)
    payload["scores"] = dict(scores)
    payload["evaluated_at"] = _now_iso()
    if not scores.get("heldout_ready"):
        payload["status"] = "needs_evidence"
    elif float(scores.get("score") or 0.0) <= 0.0 or int(scores.get("failed") or 0) > 0:
        payload["status"] = "failed_evaluation"
    else:
        payload["status"] = "evaluated"
    _write_json(path / "proposal.json", payload)
    return load_skillopt_proposal(path)
