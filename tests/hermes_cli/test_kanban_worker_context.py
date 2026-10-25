"""Worker-context approval boundary regressions for #137584."""

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _task_with_context(monkeypatch, tmp_path) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="approval-aware task")
        return kb.build_worker_context(conn, task_id)


def test_worker_context_surfaces_single_query_denials(monkeypatch, tmp_path):
    """A brief author sees the capabilities a default unattended worker cannot use."""
    from tools import approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "deny")

    context = _task_with_context(monkeypatch, tmp_path)

    assert "## Unattended approval boundary" in context
    assert "no user can approve requests" in context
    assert "`approvals.single_query_mode: deny`" in context
    assert "dangerous terminal commands" in context
    assert "`execute_code`" in context


def test_worker_context_distinguishes_trusted_single_query_policy(monkeypatch, tmp_path):
    """CONTROL: an explicitly trusted worker is not described as denied."""
    from tools import approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: "approve")

    context = _task_with_context(monkeypatch, tmp_path)

    assert "`approvals.single_query_mode: approve`" in context
    assert "trusts this worker" in context
    assert "Do not accept a brief" not in context
