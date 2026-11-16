"""With verify-on-stop off (the default) the evidence ledger must be fully inert: no
recording, no staleness tracking, and no ``verification_evidence.db`` ever created.
The ledger's only consumer is the stop guard; an unconsumed ledger is disk churn."""

from pathlib import Path

import pytest

from agent.verification_evidence import (
    mark_workspace_edited,
    record_terminal_result,
    record_verify_run,
    verification_status,
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_VERIFY_ON_STOP", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text('{"scripts": {"test": "vitest"}}', encoding="utf-8")
    return root


def test_disabled_guard_never_touches_the_ledger(project, tmp_path):
    db = tmp_path / ".hermes" / "verification_evidence.db"

    assert record_terminal_result(command="npm test", cwd=project, session_id="s1", exit_code=0) is None
    assert mark_workspace_edited(session_id="s1", cwd=project, paths=[str(project / "app.ts")]) is None
    assert record_verify_run(root=project, session_id="s1", ok=True) is None
    assert verification_status(session_id="s1", cwd=project)["status"] == "disabled"
    assert not db.exists()


def test_enabled_guard_records_into_the_ledger(project, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    assert record_terminal_result(command="npm test", cwd=project, session_id="s1", exit_code=0)
    assert verification_status(session_id="s1", cwd=project)["status"] == "passed"
    assert Path(tmp_path / ".hermes" / "verification_evidence.db").exists()
