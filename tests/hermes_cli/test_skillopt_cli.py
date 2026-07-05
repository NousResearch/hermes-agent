"""Tests for manual SkillOpt CLI helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_skill(home: Path, name: str, text: str = "# Demo\n") -> Path:
    path = home / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\n---\n\n{text}", encoding="utf-8")
    return path


def _with_home(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_skillopt_status_prints_ranked_candidates(tmp_path, capsys):
    home = tmp_path / ".hermes"
    _write_skill(home, "demo")
    (home / "skills" / ".usage.json").write_text(
        '{"demo":{"created_by":"agent","use_count":3,"patch_count":1}}',
        encoding="utf-8",
    )

    def run():
        from hermes_cli.skillopt import cmd_skillopt

        return cmd_skillopt(argparse.Namespace(skillopt_command="status", skill=None, limit=5, json=False))

    assert _with_home(home, run) == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "patched-often" in out


def test_skillopt_propose_stages_candidate_without_mutating_live_skill(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")
    candidate = tmp_path / "candidate.md"
    candidate.write_text("# Better Demo\n", encoding="utf-8")

    def run():
        from hermes_cli.skillopt import cmd_skillopt

        return cmd_skillopt(
            argparse.Namespace(
                skillopt_command="propose",
                skill="demo",
                candidate=str(candidate),
                rationale="unit test",
                from_session=None,
            )
        )

    assert _with_home(home, run) == 0
    assert "# Demo" in skill_path.read_text(encoding="utf-8")
    staged = list((home / "skillopt" / "runs").glob("demo-*"))
    assert len(staged) == 1
    assert (staged[0] / "candidate.SKILL.md").read_text(encoding="utf-8") == "# Better Demo\n"


def test_skillopt_reject_marks_proposal_rejected(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal, load_skillopt_proposal
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        rc = cmd_skillopt(argparse.Namespace(skillopt_command="reject", run_id=staged.run_id, reason="no", reviewer="test"))
        loaded = load_skillopt_proposal(staged.run_dir)
        return rc, loaded.status

    assert _with_home(home, run) == (0, "rejected")


def test_skillopt_adopt_requires_evaluated_status_then_updates_live_skill(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")

    def run():
        from agent.skillopt_state import load_skillopt_proposal, stage_skillopt_proposal, update_skillopt_evaluation
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        update_skillopt_evaluation(staged.run_dir, {"heldout_ready": True, "score": 1.0, "passed": 2, "failed": 0, "total": 2})
        rc = cmd_skillopt(argparse.Namespace(skillopt_command="adopt", run_id=staged.run_id))
        loaded = load_skillopt_proposal(staged.run_dir)
        return rc, loaded.status, skill_path.read_text(encoding="utf-8")

    assert _with_home(home, run) == (0, "adopted", "# Candidate\n")


def test_skillopt_adopt_refuses_unevaluated_proposal(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        with pytest.raises(RuntimeError, match="must be evaluated"):
            cmd_skillopt(argparse.Namespace(skillopt_command="adopt", run_id=staged.run_id))
        return skill_path.read_text(encoding="utf-8")

    assert _with_home(home, run).endswith("# Demo\n")


def test_skillopt_adopt_refuses_evaluated_status_with_bad_scores(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        proposal_path = staged.run_dir / "proposal.json"
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        payload["status"] = "evaluated"
        payload["scores"] = {"heldout_ready": False, "score": 0.0, "passed": 0, "failed": 1, "total": 1}
        proposal_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="passing evaluation scores"):
            cmd_skillopt(argparse.Namespace(skillopt_command="adopt", run_id=staged.run_id))
        return skill_path.read_text(encoding="utf-8")

    assert _with_home(home, run).endswith("# Demo\n")


def test_skillopt_evaluate_records_scores_and_rejects_low_evidence(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")

    def run():
        from agent.skillopt_state import load_skillopt_proposal, stage_skillopt_proposal
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        rc = cmd_skillopt(
            argparse.Namespace(
                skillopt_command="evaluate",
                run_id=staged.run_id,
                root=str(tmp_path),
                session_id="s1",
                min_events=2,
            )
        )
        loaded = load_skillopt_proposal(staged.run_dir)
        return rc, loaded.status, loaded.proposal["scores"]

    rc, status, scores = _with_home(home, run)
    assert rc == 1
    assert status == "needs_evidence"
    assert scores["heldout_ready"] is False


def test_skillopt_run_id_rejects_traversal_and_absolute_paths(tmp_path):
    home = tmp_path / ".hermes"

    def run():
        from hermes_cli.skillopt import cmd_skillopt

        with pytest.raises(ValueError, match="invalid SkillOpt run_id"):
            cmd_skillopt(argparse.Namespace(skillopt_command="show", run_id="../evil", json=False))
        with pytest.raises(ValueError, match="invalid SkillOpt run_id"):
            cmd_skillopt(argparse.Namespace(skillopt_command="show", run_id=str(tmp_path / "evil"), json=False))

    home.mkdir(parents=True)
    _with_home(home, run)


def test_skillopt_adopt_refuses_proposal_target_outside_skills_dir(tmp_path):
    home = tmp_path / ".hermes"
    skill_path = _write_skill(home, "demo", "# Demo\n")
    outside = tmp_path / "outside.md"
    outside.write_text("outside current", encoding="utf-8")

    def run():
        from agent.skillopt_state import stage_skillopt_proposal
        from hermes_cli.skillopt import cmd_skillopt

        staged = stage_skillopt_proposal(skill_name="demo", current_skill_path=skill_path, candidate_skill="# Candidate\n")
        proposal_path = staged.run_dir / "proposal.json"
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        import hashlib

        payload["current_skill_path"] = str(outside)
        payload["current_sha256"] = hashlib.sha256(outside.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        payload["status"] = "evaluated"
        payload["scores"] = {"heldout_ready": True, "score": 1.0, "passed": 2, "failed": 0, "total": 2}
        proposal_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="outside Hermes skills directory"):
            cmd_skillopt(argparse.Namespace(skillopt_command="adopt", run_id=staged.run_id))
        return outside.read_text(encoding="utf-8")

    assert _with_home(home, run) == "outside current"
