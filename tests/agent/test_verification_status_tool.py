"""``verification_status``: the ledger's verdict as the resumed agent sees it."""

import json
import shutil
import subprocess

import pytest

from tools import verification_status_tool as vst
from tools.verification_status_tool import verification_status_tool


def _patch_status(monkeypatch, payload):
    from agent import verification_evidence

    monkeypatch.setattr(verification_evidence, "verification_status", lambda **_kw: payload)


def test_passed_evidence_names_the_command_and_when_it_ran(monkeypatch):
    _patch_status(monkeypatch, {
        "status": "passed",
        "root": "/repo",
        "evidence": {
            "id": 12, "command": "pytest -q tests/agent", "kind": "test", "scope": "full",
            "exit_code": 0, "status": "passed", "created_at": "2026-11-09T18:02:11+00:00",
            "output_summary": "21 passed",
        },
        "changed_paths": [],
    })

    out = verification_status_tool({"cwd": "/repo"}, session_id="s1")

    assert out.startswith("status: passed")
    assert "evidence #12" in out
    assert "command: pytest -q tests/agent" in out
    assert "exit: 0" in out
    assert "2026-11-09T18:02:11+00:00" in out
    assert "passed: the last recorded verification run passed" in out


def test_stale_evidence_tells_the_model_to_re_run(monkeypatch):
    _patch_status(monkeypatch, {
        "status": "stale",
        "root": "/repo",
        "evidence": {"id": 3, "command": "pytest", "kind": "test", "scope": "targeted",
                     "exit_code": 0, "status": "passed", "created_at": "2026-11-01T09:00:00+00:00"},
        "changed_paths": ["agent/context_compressor.py", "tools/todo_tool.py"],
    })

    out = verification_status_tool({"cwd": "/repo"})

    assert "status: stale" in out
    assert "changed since: agent/context_compressor.py, tools/todo_tool.py" in out
    assert "predates a later edit" in out and "Re-run" in out


def test_missing_evidence_is_named_unverified(monkeypatch):
    _patch_status(monkeypatch, {"status": "unverified", "root": "/repo", "evidence": None, "changed_paths": []})

    out = verification_status_tool({"cwd": "/repo"})

    assert "status: unverified" in out
    assert "no passing verification is recorded" in out


def test_non_project_says_not_applicable(monkeypatch):
    _patch_status(monkeypatch, {"status": "not_applicable", "evidence": None})

    out = verification_status_tool({"cwd": "/notes"})

    assert "status: not_applicable" in out
    assert "no code project detected" in out


def test_disabled_names_the_switch(monkeypatch):
    _patch_status(monkeypatch, {"status": "disabled", "evidence": None})

    out = verification_status_tool({"cwd": "/repo"})

    assert "status: disabled" in out
    assert "verify_on_stop" in out


def test_runaway_output_is_bounded(monkeypatch):
    _patch_status(monkeypatch, {
        "status": "passed", "root": "/repo",
        "evidence": {"id": 1, "command": "pytest", "kind": "test", "scope": "full", "exit_code": 0,
                     "status": "passed", "created_at": "2026-11-09T18:02:11+00:00",
                     "output_summary": "x" * 5000},
    })

    out = verification_status_tool({"cwd": "/repo"})

    assert "[truncated]" in out
    assert len(out) < 900


def test_availability_follows_the_switch(monkeypatch):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    assert vst.verification_status_available() is True
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    assert vst.verification_status_available() is False


def test_ledger_failure_returns_an_error_string_instead_of_raising(monkeypatch):
    from agent import verification_evidence

    def _boom(**_kw):
        raise RuntimeError("ledger locked")

    monkeypatch.setattr(verification_evidence, "verification_status", _boom)

    payload = json.loads(vst._handler({"cwd": "/repo"}))

    assert "ledger locked" in payload["error"]


@pytest.mark.skipif(not shutil.which("git"), reason="git not installed")
def test_end_to_end_against_the_real_ledger(tmp_path, monkeypatch):
    """A recorded pass reads as passed; any later edit turns the same row into stale."""
    from agent import verification_evidence as ve

    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)
    run("init")
    run("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "--allow-empty", "-m", "one")

    assert ve.record_verify_run(root=repo, session_id="s1", ok=True, output="3 passed") is not None

    passed = verification_status_tool({"cwd": str(repo)}, session_id="s1")
    assert "status: passed" in passed, passed
    assert "output: 3 passed" in passed

    ve.mark_workspace_edited(session_id="s1", cwd=repo, paths=["agent/x.py"])

    stale = verification_status_tool({"cwd": str(repo)}, session_id="s1")
    assert "status: stale" in stale, stale
    assert "agent/x.py" in stale
