"""Tests for the workflow CLI surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from hermes_cli import workflows as wc
from hermes_cli import workflows_db as wfdb


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


def test_validate_rejects_unsupported_send_message(workflow_home, tmp_path, capsys):
    spec_path = tmp_path / "unsupported.yaml"
    spec_path.write_text(
        """
id: unsupported_send_message_demo
name: Unsupported Send Message Demo
version: 1
triggers:
  - type: manual
nodes:
  start:
    type: send_message
    output:
      text: hi
edges: []
""".strip(),
        encoding="utf-8",
    )

    rc, out, err = _run(["validate", str(spec_path)], capsys)

    assert rc == 1
    assert out == ""
    assert "unsupported node type: send_message on node start" in err
    assert "Traceback" not in err


def test_main_workflow_validate_bad_edge_exits_without_traceback(
    workflow_home,
    tmp_path,
    monkeypatch,
    capsys,
):
    spec_path = _write_workflow(tmp_path / "bad.yaml", bad_target=True)

    from hermes_cli import config as config_mod
    from hermes_cli import main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "workflow", "validate", str(spec_path)])
    monkeypatch.setattr(main_mod, "_set_process_title", lambda: None)
    monkeypatch.setattr(main_mod, "_cleanup_quarantined_exes", lambda: None)
    monkeypatch.setattr(main_mod, "_recover_from_interrupted_install", lambda: None)
    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "unknown edge target: missing" in captured.err
    assert "Traceback" not in captured.err


def test_run_input_help_describes_json_file_path(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _parser().parse_args(["workflow", "run", "--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "JSON file path containing an object" in captured.out


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


def test_deploy_show_preserves_agent_provider_model_fields(
    workflow_home, tmp_path, capsys
):
    # Ensure "reviewer" profile is available for the profile validation check
    (workflow_home / "profiles" / "reviewer").mkdir(parents=True)

    spec_path = tmp_path / "routed.yaml"
    spec_path.write_text(
        """
id: routed_review
name: Routed Review
version: 1
triggers:
  - type: manual
    id: manual
nodes:
  review:
    type: agent_task
    profile: reviewer
    provider: openai-codex
    model: gpt-5.5
    prompt: 'Return JSON only: {"ok": true}'
    result_contract:
      ok: boolean
edges: []
""".lstrip(),
        encoding="utf-8",
    )

    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, err = _run(["show", "routed_review", "--json"], capsys)

    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    node = payload["spec"]["nodes"]["review"]
    assert node["provider"] == "openai-codex"
    assert node["model"] == "gpt-5.5"


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
    # `run` ticks once inline (same as the dashboard Run button), so a
    # cheap all-pass workflow completes immediately.
    assert run_payload == {
        "execution_id": execution_id,
        "status": "succeeded",
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
    assert show_payload["context"]["input"] == {"reviewer": "bot"}
    assert show_payload["context"]["node"]["done"]["output"] == {"finished": True}


def test_execution_drilldown_json_redacts_secret_like_values(workflow_home, tmp_path, capsys):
    spec_path = tmp_path / "secret-workflow.yaml"
    spec_path.write_text(
        """
id: secret-cli
name: Secret CLI
version: 1
triggers:
  - type: manual
    id: manual
nodes:
  start:
    type: pass
    output:
      api_key: "${ input.api_key }"
      nested:
        token: "${ input.token }"
""".lstrip(),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"api_key": "sk-test-secret-1234567890", "token": "tok-secret-abc", "safe": "visible"}),
        encoding="utf-8",
    )
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, err = _run(["run", "secret-cli", "--input", str(input_path), "--json"], capsys)
    assert rc == 0
    assert err == ""
    execution_id = json.loads(out)["execution_id"]

    outputs = []
    for args in (
        ["executions", "list", "--workflow", "secret-cli", "--json"],
        ["executions", "show", execution_id, "--json"],
        ["executions", "node-runs", execution_id, "--json"],
        ["executions", "events", execution_id, "--json"],
    ):
        rc, out, err = _run(args, capsys)
        assert rc == 0
        assert err == ""
        outputs.append(out)

    combined = "\n".join(outputs)
    assert "sk-test-secret-1234567890" not in combined
    assert "tok-secret-abc" not in combined
    assert "[REDACTED]" in combined

    listed = json.loads(outputs[0])[0]
    assert listed["input"]["api_key"] == "[REDACTED]"
    shown = json.loads(outputs[1])
    assert shown["context"]["input"]["token"] == "[REDACTED]"
    node_run = json.loads(outputs[2])["node_runs"][0]
    assert node_run["output"]["api_key"] == "[REDACTED]"


def test_run_rejects_missing_required_input_before_creating_execution(
    workflow_home,
    tmp_path,
    capsys,
):
    spec_path = tmp_path / "required.yaml"
    spec_path.write_text(
        """
id: required-run
name: Required Run
version: 1
triggers:
  - type: manual
    id: manual
    input_schema:
      brief:
        kind: long_text
        required: true
nodes:
  start:
    type: pass
""".lstrip(),
        encoding="utf-8",
    )
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0

    rc, out, err = _run(["run", "required-run", "--json"], capsys)

    assert rc == 1
    assert out == ""
    assert "brief is required" in err
    with wfdb.connect() as conn:
        assert conn.execute("SELECT count(*) FROM workflow_executions").fetchone()[0] == 0


def _start_queued_execution(workflow_id: str, input_data: dict | None = None) -> str:
    """Seed a queued execution without the run command's inline tick."""
    wfdb.init_db()
    with wfdb.connect() as conn:
        return wfdb.start_execution(
            conn,
            workflow_id,
            input_data=input_data or {},
            trigger_type="manual",
        )


def test_tick_json_advances_cheap_workflow_to_success(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    execution_id = _start_queued_execution("code-change-review")

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
    execution_id = _start_queued_execution("code-change-review")

    rc, out, err = _run(["executions", "cancel", execution_id], capsys)

    assert rc == 0
    assert err == ""
    assert f"Cancelled execution {execution_id}" in out

    rc, out, err = _run(["executions", "show", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    assert json.loads(out)["status"] == "cancelled"


def test_cancel_does_not_overwrite_terminal_status_race(
    workflow_home,
    tmp_path,
    capsys,
    monkeypatch,
):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, _err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 0
    execution_id = json.loads(out)["execution_id"]
    original_get_execution = wfdb.get_execution
    first_read = True

    def get_execution_then_complete(conn, requested_id):
        nonlocal first_read
        execution = original_get_execution(conn, requested_id)
        if requested_id == execution_id and first_read:
            first_read = False
            conn.execute(
                """
                UPDATE workflow_executions
                   SET status = 'succeeded', updated_at = ?
                 WHERE execution_id = ?
                """,
                (execution.updated_at + 1, execution_id),
            )
        return execution

    monkeypatch.setattr(wfdb, "get_execution", get_execution_then_complete)

    rc, out, err = _run(["executions", "cancel", execution_id], capsys)

    assert rc == 0
    assert err == ""
    assert f"Execution {execution_id} already succeeded." in out
    with wfdb.connect() as conn:
        assert original_get_execution(conn, execution_id).status == "succeeded"


def test_deploy_bump_flag_redeploys_changed_spec_as_next_version(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0

    changed = spec_path.read_text(encoding="utf-8").replace(
        "name: Code Change Review", "name: Code Change Review v2"
    )
    spec_path.write_text(changed, encoding="utf-8")

    rc, _out, err = _run(["deploy", str(spec_path)], capsys)
    assert rc == 1
    assert "different checksum" in err

    rc, out, err = _run(["deploy", str(spec_path), "--bump"], capsys)
    assert rc == 0
    assert err == ""
    assert "Deployed workflow code-change-review v2 (bumped from v1)" in out


def test_enable_disable_toggle_blocks_and_allows_runs(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0

    rc, out, err = _run(["disable", "code-change-review"], capsys)
    assert rc == 0
    assert err == ""
    assert "now disabled" in out

    rc, _out, err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 1
    assert "is disabled" in err

    rc, out, err = _run(["enable", "code-change-review"], capsys)
    assert rc == 0
    assert "now enabled" in out

    rc, out, err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 0
    assert json.loads(out)["status"] == "succeeded"


def test_executions_node_runs_and_events_drilldowns(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    rc, out, _err = _run(["run", "code-change-review", "--json"], capsys)
    assert rc == 0
    execution_id = json.loads(out)["execution_id"]

    rc, out, err = _run(["executions", "events", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    events = json.loads(out)["events"]
    kinds = [event["kind"] for event in events]
    assert "execution_started" in kinds
    assert "execution_succeeded" in kinds

    rc, out, err = _run(["executions", "node-runs", execution_id, "--json"], capsys)
    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["execution_id"] == execution_id
    assert isinstance(payload["node_runs"], list)

    rc, _out, err = _run(["executions", "events", "missing", "--json"], capsys)
    assert rc == 1
    assert "workflow execution not found" in err


def test_executions_list_newest_first_with_limit(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    wfdb.init_db()
    with wfdb.connect() as conn:
        ids = [
            wfdb.start_execution(
                conn, "code-change-review", input_data={}, trigger_type="manual", now=100 + i
            )
            for i in range(3)
        ]

    rc, out, err = _run(["executions", "list", "--limit", "2", "--json"], capsys)
    assert rc == 0
    assert err == ""
    rows = json.loads(out)
    assert [row["execution_id"] for row in rows] == list(reversed(ids))[:2]


def test_status_reports_dispatcher_and_counts(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0
    _start_queued_execution("code-change-review")

    rc, out, err = _run(["status", "--json"], capsys)

    assert rc == 0
    assert err == ""
    payload = json.loads(out)
    assert isinstance(payload["dispatcher"]["dispatch_in_gateway"], bool)
    assert payload["definitions"] == 1
    assert payload["executions_by_status"] == {"queued": 1}


def test_run_slash_help_status_and_errors(workflow_home, tmp_path, capsys):
    output = wc.run_slash("")
    assert output.startswith("/workflow — workflow graph engine")
    assert "/workflow executions events" in output

    output = wc.run_slash("status")
    assert "Definitions: 0" in output

    output = wc.run_slash("list")
    assert "(no workflows deployed)" in output

    output = wc.run_slash("bogus-verb")
    assert output.startswith("⚠ /workflow usage error")

    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert "Deployed workflow" in wc.run_slash(f"deploy {spec_path}")
    assert "code-change-review v1 enabled" in wc.run_slash("list")


def test_validate_rejects_missing_profile(workflow_home, tmp_path, capsys):
    spec_path = tmp_path / "ghost_profile.yaml"
    spec_path.write_text(
        """
id: ghost-profile
name: Ghost Profile
version: 1
triggers:
  - type: manual
    id: manual
nodes:
  start:
    type: pass
    output:
      ok: true
  review:
    type: agent_task
    profile: nonexistent_ghost
    prompt: "Review this."
edges:
  - from: start
    to: review
""".strip(),
        encoding="utf-8",
    )

    rc, out, err = _run(["validate", str(spec_path)], capsys)

    assert rc == 1
    assert out == ""
    assert "workflow_profile_not_found" in err
    assert "nonexistent_ghost" in err


def test_deploy_rejects_missing_profile(workflow_home, tmp_path, capsys):
    spec_path = tmp_path / "ghost_profile.yaml"
    spec_path.write_text(
        """
id: ghost-deploy
name: Ghost Deploy
version: 1
triggers:
  - type: manual
    id: manual
nodes:
  task:
    type: agent_task
    profile: nonexistent_ghost
    prompt: "Do work."
""".strip(),
        encoding="utf-8",
    )

    rc, out, err = _run(["deploy", str(spec_path)], capsys)

    assert rc == 1
    assert "workflow_profile_not_found" in err


def test_cli_deploy_never_mutates_existing_version(workflow_home, tmp_path, capsys):
    spec_path = _write_workflow(tmp_path / "workflow.yaml")
    assert _run(["deploy", str(spec_path)], capsys)[0] == 0

    changed = tmp_path / "changed.yaml"
    changed.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "name: Code Change Review", "name: Code Change Review CHANGED"
        ),
        encoding="utf-8",
    )

    rc, out, err = _run(["deploy", str(changed)], capsys)

    assert rc == 1
    assert "different checksum" in err

    rc, out, err = _run(["show", "code-change-review", "--json"], capsys)
    assert rc == 0
    payload = json.loads(out)
    assert payload["name"] == "Code Change Review"
