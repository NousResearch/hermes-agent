"""hermes workflow — list stored graphs and start a run."""

from argparse import Namespace

from hermes_cli.subcommands.workflow import workflow_command
from workflow.store import save_documents


def _args(**kwargs):
    defaults = {"workflow_command": None, "name": None, "payload": ""}
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_list_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    workflow_command(_args(workflow_command="list"))
    assert "No workflows stored." in capsys.readouterr().out


def test_list_and_run_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents(
        [
            {
                "id": "wf-1",
                "name": "Ship it",
                "scenario": {
                    "steps": [{"id": "go", "kind": "trigger", "config": {"title": "Play"}}],
                    "edges": [],
                },
            }
        ],
        "wf-1",
    )
    workflow_command(_args(workflow_command="list"))
    listed = capsys.readouterr().out
    assert "wf-1" in listed
    assert "Ship it" in listed

    workflow_command(_args(workflow_command="run", name="Ship it"))
    started = capsys.readouterr().out
    assert "wf-1" in started

    workflow_command(_args(workflow_command="status", name="Ship it"))
    status = capsys.readouterr().out
    assert "wf-1" in status


def test_run_unknown(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    workflow_command(_args(workflow_command="run", name="missing"))
    assert "No workflow" in capsys.readouterr().out
