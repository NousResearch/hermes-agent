from __future__ import annotations

from pathlib import Path

from hermes_cli.commands import resolve_command
from plugins.memory import load_memory_provider


def _seed_candidate(tmp_path: Path):
    provider = load_memory_provider("layered")
    provider.initialize(session_id="skill-candidate-cli", hermes_home=str(tmp_path), platform="cli")
    for _ in range(3):
        provider.on_session_end([
            {"role": "user", "content": "Please implement a bugfix with tests."},
            {"role": "assistant", "content": "I wrote failing tests first, fixed the bug, and all tests passed."},
        ])
    return provider


def test_skill_candidates_command_registered():
    cmd = resolve_command("skill-candidates")
    assert cmd is not None
    assert cmd.name == "skill-candidates"


def test_handle_skill_candidates_list_prints_candidate_summary(monkeypatch, tmp_path, capsys):
    from hermes_cli.skill_candidates import handle_skill_candidates_slash

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    handle_skill_candidates_slash("/skill-candidates list")

    output = capsys.readouterr().out
    assert "write-failing-tests-first-then-verify-tests-pass" in output
    assert "pending" in output


def test_handle_skill_candidates_inspect_prints_detail(monkeypatch, tmp_path, capsys):
    from hermes_cli.skill_candidates import handle_skill_candidates_slash

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    handle_skill_candidates_slash("/skill-candidates inspect write-failing-tests-first-then-verify-tests-pass")

    output = capsys.readouterr().out
    assert "write-failing-tests-first-then-verify-tests-pass" in output
    assert "skill_draft_path" in output
    assert "publish_ready_dir" in output


def test_handle_skill_candidates_approve_installs_skill(monkeypatch, tmp_path, capsys):
    from hermes_cli.skill_candidates import handle_skill_candidates_slash

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_candidate(tmp_path)

    handle_skill_candidates_slash("/skill-candidates approve write-failing-tests-first-then-verify-tests-pass")

    output = capsys.readouterr().out
    assert "approved" in output.lower() or "installed" in output.lower()
    assert (tmp_path / "skills" / "write-failing-tests-first-then-verify-tests-pass" / "SKILL.md").exists()


def test_handle_skill_candidates_reject_updates_status(monkeypatch, tmp_path, capsys):
    from hermes_cli.skill_candidates import handle_skill_candidates_slash

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _seed_candidate(tmp_path)

    handle_skill_candidates_slash("/skill-candidates reject write-failing-tests-first-then-verify-tests-pass manual_reject")

    output = capsys.readouterr().out
    assert "rejected" in output.lower()
    details = provider.inspect_skill_candidate("write-failing-tests-first-then-verify-tests-pass")
    assert details["review_status"] == "rejected"
    assert details["review_gate_reason"] == "manual_reject"
