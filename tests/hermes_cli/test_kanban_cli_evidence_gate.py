from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


pytestmark = pytest.mark.real_completion_evidence_gate


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task_id_from_create(output: str) -> str:
    import re

    match = re.search(r"(t_[a-f0-9]+)", output)
    assert match, output
    return match.group(1)


def test_kanban_complete_cli_rejects_missing_evidence(kanban_home):
    tid = _task_id_from_create(kc.run_slash("create 'needs proof' --assignee alice"))

    out = kc.run_slash(f"complete {tid} --summary 'done without proof'")

    assert "completion blocked" in out
    assert "evidence path" in out
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"


def test_kanban_complete_cli_accepts_evidence_flag(kanban_home):
    tid = _task_id_from_create(kc.run_slash("create 'has proof' --assignee alice"))
    evidence = kanban_home / "cli-evidence.md"
    evidence.write_text("cli evidence\n", encoding="utf-8")

    out = kc.run_slash(
        f"complete {tid} --summary 'done with proof' --evidence {evidence}"
    )

    assert f"Completed {tid}" in out
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.metadata is not None
        assert run.metadata["evidence_refs"] == [str(evidence)]


def test_kanban_complete_cli_rejects_bulk_evidence_reuse(kanban_home):
    a = _task_id_from_create(kc.run_slash("create 'a' --assignee alice"))
    b = _task_id_from_create(kc.run_slash("create 'b' --assignee alice"))
    evidence = kanban_home / "bulk-evidence.md"
    evidence.write_text("ambiguous\n", encoding="utf-8")

    out = kc.run_slash(f"complete {a} {b} --evidence {evidence}")

    assert "--summary / --metadata / --evidence are per-task" in out
    with kb.connect() as conn:
        for tid in (a, b):
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.status == "ready"
