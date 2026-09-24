"""Completion must not attach an unchanged, already-uploaded deliverable twice."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


@pytest.mark.parametrize("handoff", ["complete", "review"])
def test_completion_reuses_unchanged_explicit_upload(board, handoff):
    task = kb.create_task(board, title="deliver report")
    workspace = kbw.resolve_workspace(kb.get_task(board, task))
    kbw.set_workspace_path(board, task, workspace)
    report = workspace / "report.html"
    report.write_bytes(b"<p>report</p>")
    attachment_id = kb.store_attachment_bytes(
        board, task, report.name, report.read_bytes(),
        content_type="text/html", uploaded_by="agent",
    )
    original = kb.get_attachment(board, attachment_id)
    if handoff == "review":
        kb.claim_task(board, task)
        run_id = kb.get_task(board, task).current_run_id
        assert kb.request_review(board, task, summary="review report",
                                 metadata={"artifacts": [str(report)]}, expected_run_id=run_id)
        assert workspace.exists()
        assert kb.complete_task(board, task, result="approved", metadata={"artifacts": [str(report)]})
    else:
        assert kb.complete_task(board, task, result="delivered", metadata={"artifacts": [str(report)]})
    attachments = kb.list_attachments(board, task)
    assert [a.id for a in attachments] == [attachment_id]
    assert attachments[0].content_type == "text/html"
    assert Path(original.stored_path).read_bytes() == b"<p>report</p>"
    assert not workspace.exists()
    assert kb.latest_run(board, task).metadata["artifacts"] == [original.stored_path]
    completed = [e for e in kb.list_events(board, task) if e.kind == "completed"][-1]
    assert completed.payload["artifacts"] == [original.stored_path]
    assert len([e for e in kb.list_events(board, task) if e.kind == "attached"]) == 1
    assert list(Path(original.stored_path).parent.iterdir()) == [Path(original.stored_path)]


@pytest.mark.parametrize("case", ["changed", "missing", "other_task", "other_name", "empty", "unreadable"])
def test_completion_preserves_nonmatching_artifact(board, case, monkeypatch):
    task = kb.create_task(board, title="deliver report")
    workspace = kbw.resolve_workspace(kb.get_task(board, task))
    kbw.set_workspace_path(board, task, workspace)
    report = workspace / "report.txt"
    report.write_bytes(b"new")
    old = None
    if case != "empty":
        owner = kb.create_task(board, title="other card") if case == "other_task" else task
        aid = kb.store_attachment_bytes(
            board, owner, "other.txt" if case == "other_name" else report.name,
            b"old" if case == "changed" else b"new", content_type="text/plain", uploaded_by="agent",
        )
        old = kb.get_attachment(board, aid)
        if case == "missing":
            Path(old.stored_path).unlink()
        if case == "unreadable":
            real_open = Path.open
            def open_file(path, *args, **kwargs):
                if path == Path(old.stored_path):
                    raise PermissionError("unreadable old upload")
                return real_open(path, *args, **kwargs)
            monkeypatch.setattr(Path, "open", open_file)
    assert kb.complete_task(board, task, result="done", metadata={"artifacts": [str(report)]})
    artifacts = kb.latest_run(board, task).metadata["artifacts"]
    assert len(artifacts) == 1
    assert Path(artifacts[0]).read_bytes() == b"new"
    if old is not None:
        assert artifacts != [old.stored_path]
        assert kb.get_attachment(board, old.id) == old


def test_failed_review_keeps_reused_attachment_and_removes_new_copy(board, monkeypatch):
    task = kb.create_task(board, title="review rollback")
    workspace = kbw.resolve_workspace(kb.get_task(board, task))
    kbw.set_workspace_path(board, task, workspace)
    old = workspace / "old.txt"
    old.write_bytes(b"old")
    new = workspace / "new.txt"
    new.write_bytes(b"new")
    aid = kb.store_attachment_bytes(board, task, old.name, old.read_bytes(), uploaded_by="agent")
    stored = kb.get_attachment(board, aid)
    kb.claim_task(board, task)
    run_id = kb.get_task(board, task).current_run_id
    def fail(*args, **kwargs):
        raise RuntimeError("handoff failed")
    with monkeypatch.context() as patcher:
        patcher.setattr(kb, "_end_or_synthesize_run", fail)
        with pytest.raises(RuntimeError, match="handoff failed"):
            kb.request_review(board, task, summary="ready", expected_run_id=run_id,
                              metadata={"artifacts": [str(old), str(new)]})
    assert kb.list_attachments(board, task) == [stored]
    assert list(Path(stored.stored_path).parent.iterdir()) == [Path(stored.stored_path)]
    assert Path(stored.stored_path).read_bytes() == b"old"
    assert workspace.exists()

