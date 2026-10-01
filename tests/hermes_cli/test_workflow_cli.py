"""hermes workflow — the CLI publishes; the reactor runs. An approval is answered from anywhere."""

import re
from argparse import Namespace

from hermes_cli.subcommands.workflow import run_slash, workflow_command
from workflow import reactor
from workflow.store import list_runs, save_documents


def _args(**kwargs):
    defaults = {"workflow_command": None, "name": None, "payload": "", "code": None}
    defaults.update(kwargs)
    return Namespace(**defaults)


def _agent(goal, _context, _payload, _config):
    return {"ok": True, "summary": goal, "verdict": "PASS", "output": {}}


def test_list_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    workflow_command(_args(workflow_command="list"))
    assert "No workflows stored." in capsys.readouterr().out


def test_run_is_queued_until_the_reactor_takes_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents([{
        "id": "wf-1", "name": "Ship it",
        "scenario": {"steps": [{"id": "go", "kind": "trigger", "config": {"title": "Play"}}], "edges": []},
    }], "wf-1")
    workflow_command(_args(workflow_command="list"))
    assert "Ship it" in capsys.readouterr().out

    workflow_command(_args(workflow_command="run", name="Ship it"))
    started = capsys.readouterr().out
    assert "queued" in started and "desktop app or `hermes gateway`" in started
    assert list_runs() == []

    assert reactor.drain_once(background=False, execute_fn=_agent) == 1
    workflow_command(_args(workflow_command="status", name="Ship it"))
    assert "succeeded" in capsys.readouterr().out


def test_an_approval_is_answered_by_its_code_from_a_chat(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents([{
        "id": "rel", "name": "Release",
        "scenario": {
            "steps": [
                {"id": "ask", "kind": "human", "config": {"goal": "Tag the release?"}},
                {"id": "tag", "kind": "agent", "config": {"goal": "tag"}},
            ],
            "edges": [{"id": "1", "source": "ask", "target": "tag"}],
        },
    }], "rel")
    workflow_command(_args(workflow_command="run", name="rel"))
    reactor.drain_once(background=False, execute_fn=_agent)
    capsys.readouterr()

    workflow_command(_args(workflow_command="status"))
    code = re.search(r"\(code ([0-9a-f]+)\)", capsys.readouterr().out).group(1)

    reply = run_slash(f"approve {code}", by="telegram:brooklyn")
    assert "approved" in reply and "Tag the release?" in reply
    reactor.drain_once(background=False, execute_fn=_agent)
    (run,) = list_runs()
    assert run["status"] == "succeeded" and "tag" in run["ran"]
    assert "Nothing is waiting" in run_slash(f"deny {code}")


def test_run_unknown(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    workflow_command(_args(workflow_command="run", name="missing"))
    assert "No workflow" in capsys.readouterr().out
