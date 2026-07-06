"""Tests for the workflow CLI surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import workflows as wc


@pytest.fixture
def workflow_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    workflow_parser = wc.build_parser(sub)
    workflow_parser.set_defaults(func=wc.workflow_command)
    return parser


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    args = _parser().parse_args(["workflow", *argv])
    rc = wc.workflow_command(args)
    captured = capsys.readouterr()
    return int(rc or 0), captured.out, captured.err


def _write_workflow(path: Path, *, bad_target: bool = False) -> Path:
    target = "missing" if bad_target else "done"
    path.write_text(
        f"""
id: code-change-review
name: Code Change Review
version: 1
enabled: true
triggers:
  - type: manual
    id: manual
nodes:
  start:
    type: pass
    output:
      ok: true
  done:
    type: pass
    output:
      finished: true
edges:
  - from: start
    to: {target}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_validate_reports_bad_edge_target(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "bad.yaml", bad_target=True)

    rc, out, err = _run(["validate", str(spec_path)], capsys)

    assert rc == 1
    assert out == ""
    assert "Error:" in err
    assert "unknown edge target: missing" in err
    assert "Traceback" not in err


def test_deploy_then_list_and_show_json(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")

    rc, out, err = _run(["deploy", str(spec_path)], capsys)
    assert rc == 0
    assert "Deployed workflow code-change-review v1" in out
    assert err == ""

    rc, out, err = _run(["list", "--json"], capsys)
    assert rc == 0
    assert err == ""
    rows = json.loads(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "code-change-review"
    assert row["workflow_id"] == "code-change-review"
    assert row["name"] == "Code Change Review"
    assert row["version"] == 1
    assert row["enabled"] is True
    assert row["checksum"]
    assert row["created_at"]

    rc, out, err = _run(["show", "code-change-review", "--json"], capsys)
    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["workflow_id"] == "code-change-review"
    assert payload["spec"]["id"] == "code-change-review"
    assert payload["spec"]["edges"] == [{"from": "start", "to": "done"}]


def test_run_input_creates_execution_and_lists_it(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"reviewer": "bot"}), encoding="utf-8")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0

    rc, out, err = _run(
        ["run", "code-change-review", "--input", str(input_path), "--json"],
        capsys,
    )

    assert rc == 0
    assert err == ""
    run_payload = json.loads(out)
    execution_id = run_payload["execution_id"]
    assert execution_id.startswith("wfexec_")
    assert run_payload == {
        "execution_id": execution_id,
        "status": "queued",
        "version": 1,
        "workflow_id": "code-change-review",
    }

    rc, out, err = _run(
        ["executions", "list", "--workflow", "code-change-review", "--json"],
        capsys,
    )
    assert rc == 0
    assert err == ""
    rows = json.loads(out)
    assert [row["execution_id"] for row in rows] == [execution_id]
    assert rows[0]["input"] == {"reviewer": "bot"}

    rc, out, err = _run(["executions", "show", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    show_payload = json.loads(out)
    assert show_payload["execution_id"] == execution_id
    assert show_payload["context"] == {"input": {"reviewer": "bot"}, "node": {}}


def test_tick_json_advances_cheap_workflow_to_success(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, _err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 0
    execution_id = json.loads(out)["execution_id"]

    rc, out, err = _run(["tick", "--limit", "10", "--json"], capsys)

    assert rc == 0
    assert err == ""
    assert json.loads(out) == {"processed": 1}

    rc, out, err = _run(["executions", "show", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["status"] == "succeeded"
    assert payload["context"]["node"]["done"]["output"] == {"finished": True}


def test_cancel_queued_execution(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, _err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 0
    execution_id = json.loads(out)["execution_id"]

    rc, out, err = _run(["executions", "cancel", execution_id], capsys)

    assert rc == 0
    assert err == ""
    assert f"Cancelled execution {execution_id}" in out

    rc, out, err = _run(["executions", "show", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    assert json.loads(out)["status"] == "cancelled"
