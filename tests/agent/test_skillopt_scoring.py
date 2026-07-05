"""Tests for SkillOpt verification-evidence scoring."""

from __future__ import annotations

from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_score_verification_evidence_computes_pass_rate_from_ledger(tmp_path):
    home = tmp_path / ".hermes"
    root = tmp_path / "repo"
    root.mkdir()
    token = set_hermes_home_override(home)
    try:
        from agent.verification_evidence import _connect
        from agent.skillopt_scoring import score_verification_evidence

        with _connect() as conn:
            conn.execute(
                "INSERT INTO verification_events(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, exit_code, output_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("2026-01-01T00:00:00+00:00", "s1", str(root), str(root.resolve()), "pytest ok", "pytest", "test", "targeted", "passed", 0, "ok"),
            )
            conn.execute(
                "INSERT INTO verification_events(created_at, session_id, cwd, root, command, canonical_command, kind, scope, status, exit_code, output_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("2026-01-01T00:01:00+00:00", "s1", str(root), str(root.resolve()), "pytest bad", "pytest", "test", "targeted", "failed", 1, "bad"),
            )
            conn.commit()
        score = score_verification_evidence(root=root, session_id="s1")
    finally:
        reset_hermes_home_override(token)

    assert score["total"] == 2
    assert score["passed"] == 1
    assert score["failed"] == 1
    assert score["score"] == 0.5
    assert score["heldout_ready"] is True


def test_score_verification_evidence_rejects_empty_or_too_small_sets(tmp_path):
    home = tmp_path / ".hermes"
    root = tmp_path / "repo"
    root.mkdir()
    token = set_hermes_home_override(home)
    try:
        from agent.skillopt_scoring import score_verification_evidence

        score = score_verification_evidence(root=root, session_id="missing", min_events=1)
    finally:
        reset_hermes_home_override(token)

    assert score["score"] == 0.0
    assert score["total"] == 0
    assert score["heldout_ready"] is False
