"""Tests for staged SkillOpt artifact storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _with_home(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_stage_skillopt_proposal_writes_reviewable_artifacts_without_mutating_skill(tmp_path):
    home = tmp_path / ".hermes"
    skill_dir = home / "skills" / "software-development" / "demo-skill"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("---\nname: demo-skill\n---\n\n# Demo\n", encoding="utf-8")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal

        return stage_skillopt_proposal(
            skill_name="demo-skill",
            current_skill_path=skill_path,
            candidate_skill="# Demo\n\nImproved procedure.\n",
            edits=[{"op": "append", "content": "Improved procedure."}],
            scores={"current": 0.5, "candidate": 0.75},
            source={"session_id": "s1", "task_id": "t1"},
            rationale="Add missing verification step.",
        )

    proposal = _with_home(home, run)

    assert skill_path.read_text(encoding="utf-8") == "---\nname: demo-skill\n---\n\n# Demo\n"
    assert proposal.run_id.startswith("demo-skill-")
    assert proposal.run_dir.is_dir()
    assert (proposal.run_dir / "candidate.SKILL.md").read_text(encoding="utf-8") == "# Demo\n\nImproved procedure.\n"
    payload = json.loads((proposal.run_dir / "proposal.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["skill_name"] == "demo-skill"
    assert payload["status"] == "staged"
    assert payload["scores"] == {"current": 0.5, "candidate": 0.75}
    assert payload["source"] == {"session_id": "s1", "task_id": "t1"}
    assert payload["current_sha256"]
    assert payload["candidate_sha256"]
    assert "Add missing verification" in (proposal.run_dir / "meta.md").read_text(encoding="utf-8")


def test_stage_skillopt_proposal_rejects_path_traversal_skill_names(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("safe", encoding="utf-8")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal

        with pytest.raises(ValueError, match="invalid skill_name"):
            stage_skillopt_proposal(
                skill_name="../bad",
                current_skill_path=skill_path,
                candidate_skill="candidate",
            )

    _with_home(home, run)


def test_load_skillopt_proposal_validates_hashes_and_detects_tampering(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("current", encoding="utf-8")

    def run():
        from agent.skillopt_state import load_skillopt_proposal, stage_skillopt_proposal

        staged = stage_skillopt_proposal(
            skill_name="demo",
            current_skill_path=skill_path,
            candidate_skill="candidate",
        )
        loaded = load_skillopt_proposal(staged.run_dir)
        assert loaded.skill_name == "demo"
        assert loaded.candidate_skill == "candidate"
        (staged.run_dir / "candidate.SKILL.md").write_text("tampered", encoding="utf-8")
        with pytest.raises(ValueError, match="candidate hash mismatch"):
            load_skillopt_proposal(staged.run_dir)

    _with_home(home, run)


def test_append_skillopt_rejection_records_jsonl(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("current", encoding="utf-8")

    def run():
        from agent.skillopt_state import append_skillopt_rejection, stage_skillopt_proposal

        staged = stage_skillopt_proposal(
            skill_name="demo",
            current_skill_path=skill_path,
            candidate_skill="candidate",
        )
        append_skillopt_rejection(staged.run_dir, reason="score regressed", reviewer="unit-test")
        lines = (staged.run_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["reason"] == "score regressed"
        assert record["reviewer"] == "unit-test"
        proposal = json.loads((staged.run_dir / "proposal.json").read_text(encoding="utf-8"))
        assert proposal["status"] == "rejected"

    _with_home(home, run)
