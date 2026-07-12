from argparse import Namespace

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import _cmd_complete


def test_complete_command_reports_unsatisfied_gate(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Review", created_by="test")
        kb.require_completion_gate(conn, task_id, gate="pass")

    exit_code = _cmd_complete(Namespace(
        task_ids=[task_id],
        result=None,
        summary=None,
        metadata=None,
    ))

    assert exit_code == 1
    error = capsys.readouterr().err
    assert f"cannot complete {task_id}" in error
    assert "requires metadata" in error
