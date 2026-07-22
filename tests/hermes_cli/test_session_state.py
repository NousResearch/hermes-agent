from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import session_state as ss


@pytest.fixture(autouse=True)
def isolated_project_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep project-state tests away from the user's real ~/.hermes tree."""
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    project_state = hermes_home / "project_state"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(ss, "PROJECT_STATE_DIR", project_state)
    return project_state


def _assert_under(path: str | Path, root: Path) -> None:
    Path(path).expanduser().resolve().relative_to(root.expanduser().resolve())


def _write_package(root: Path) -> tuple[Path, Path]:
    design = root / "hermes_runtime_platform_v1_8_execution_completion_design_readonly"
    checkpoint = root / "hermes_runtime_platform_v1_8_execution_completion_design_readonly_checkpoint"
    design.mkdir()
    checkpoint.mkdir()

    primary = design / "HRP_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY.md"
    primary.write_text("HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY_PASS\n", encoding="utf-8")
    validation = design / "validation_report.json"
    validation.write_text(json.dumps({"validations": {"schema_json_valid": True}, "checks": [{"name": "ok", "passed": True}]}), encoding="utf-8")
    manifest = design / "manifest.json"
    manifest.write_text(json.dumps({
        "package": "HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY",
        "version": "1.8",
        "result": "PASS",
        "result_token": "HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY_PASS",
        "sources_consumed_unchanged": [str(root / "baseline")],
    }), encoding="utf-8")
    hashes = design / "verified_hashes.txt"
    lines = []
    for path in [primary, validation, manifest]:
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    hashes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    canonical = checkpoint / "canonical_artifacts.tsv"
    canonical.write_text("package\trelative_path\tsha256\nHRP_v1_8\tmanifest.json\tabc\n", encoding="utf-8")
    (checkpoint / "next_step_recommendation.md").write_text("Recommended: HRP v1.9 — TEST DESIGN READONLY\n", encoding="utf-8")
    (checkpoint / "manifest.json").write_text(json.dumps({
        "checkpoint": "HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY_CHECKPOINT",
        "version": "1.8",
        "result": "PASS",
        "result_token": "HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY_CHECKPOINT_PASS",
        "source_dir": str(design),
        "source_result_token": "HERMES_RUNTIME_PLATFORM_V1_8_EXECUTION_COMPLETION_DESIGN_READONLY_PASS",
    }), encoding="utf-8")
    return design, checkpoint


def test_checkpoint_generates_machine_readable_session_state(tmp_path: Path):
    design, checkpoint = _write_package(tmp_path)

    state_path = ss.generate_session_state(checkpoint_dir=checkpoint, project_root=tmp_path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state) == ss._REQUIRED_STATE_FIELDS
    assert state["current_version"] == "1.8"
    assert state["current_stage"] == "EXECUTION_COMPLETION_DESIGN_READONLY"
    assert state["latest_design_directory"] == str(design.resolve())
    assert state["latest_checkpoint_directory"] == str(checkpoint.resolve())
    assert state["latest_manifest"] == str((design / "manifest.json").resolve())
    assert state["latest_verified_hashes"] == str((design / "verified_hashes.txt").resolve())
    assert state["latest_validation_report"] == str((design / "validation_report.json").resolve())
    assert state["canonical_artifacts"] == str((checkpoint / "canonical_artifacts.tsv").resolve())
    assert state["next_recommended_prompt"] == "HRP v1.9 — TEST DESIGN READONLY"
    assert all(result.passed for result in ss.validate_state(state))


def test_status_and_resume_validate_only_canonical_state(tmp_path: Path, capsys, isolated_project_state: Path):
    _design, checkpoint = _write_package(tmp_path)
    ss.generate_session_state(checkpoint_dir=checkpoint, project_root=tmp_path)

    status_rc = ss.command_status(SimpleNamespace(path=str(tmp_path)))
    resume_rc = ss.command_resume(SimpleNamespace(path=str(tmp_path), json=False))
    output = capsys.readouterr().out
    located = ss.load_session_state(tmp_path)

    assert status_rc == 0
    assert resume_rc == 0
    assert "Current Version: 1.8" in output
    assert "Hash validation: PASS" in output
    assert "Resume validation: PASS" in output
    _assert_under(located.path, isolated_project_state)
    for state_file in located.state["state_files"].values():
        _assert_under(state_file, isolated_project_state)


def test_resume_blocks_on_hash_mismatch(tmp_path: Path, capsys):
    design, checkpoint = _write_package(tmp_path)
    state_path = ss.generate_session_state(checkpoint_dir=checkpoint, project_root=tmp_path)
    (design / "manifest.json").write_text("{}", encoding="utf-8")

    rc = ss.command_resume(SimpleNamespace(path=str(state_path.parent), json=False))
    err = capsys.readouterr().err

    assert rc == 1
    assert "BLOCK" in err


def test_next_dry_run_runs_checkpoint_status_resume_without_chat(tmp_path: Path, capsys):
    _design, checkpoint = _write_package(tmp_path)

    rc = ss.command_next(SimpleNamespace(
        checkpoint_dir=str(checkpoint),
        project_root=str(tmp_path),
        output=None,
        dry_run=True,
    ))
    output = capsys.readouterr().out

    assert rc == 0
    assert "Checkpoint session state: PASS" in output
    assert "Hash validation: PASS" in output
    assert "Resume validation: PASS" in output
    assert "Next execution dry-run: HRP v1.9 — TEST DESIGN READONLY" in output


def test_gc_archives_only_temporary_metadata(tmp_path: Path, capsys):
    _design, checkpoint = _write_package(tmp_path)
    ss.generate_session_state(checkpoint_dir=checkpoint, project_root=tmp_path)
    home = tmp_path / "home"
    sessions = home / "sessions"
    sessions.mkdir(parents=True)
    temp = sessions / "chat.tmp"
    temp.write_text("temporary", encoding="utf-8")
    canonical = sessions / "SESSION_STATE.json"
    canonical.write_text("{}", encoding="utf-8")

    rc = ss.command_gc(SimpleNamespace(hermes_home=str(home), dry_run=False))
    output = capsys.readouterr().out

    assert rc == 0
    assert "Canonical state preserved: PASS" in output
    assert not temp.exists()
    assert canonical.exists()
    assert list((home / "session_state_gc_archive").rglob("chat.tmp"))
