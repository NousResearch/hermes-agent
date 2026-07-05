from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.workflows_spec import WorkflowSpec


def _switch_spec() -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "demo", "name": "Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "start": {"type": "pass", "output": {"score": "${ input.score }"}},
            "route": {"type": "switch", "cases": [
                {"name": "high", "when": {"op": "gte", "left": {"path": "$.node.start.output.score"}, "right": 0.8}}
            ]},
            "high": {"type": "pass", "output": {"bucket": "high"}},
            "low": {"type": "pass", "output": {"bucket": "low"}},
        },
        "edges": [
            {"from": "start", "to": "route"},
            {"from": "route.high", "to": "high"},
            {"from": "route.default", "to": "low"},
        ],
    })


def test_tick_runs_queued_pass_switch_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _switch_spec(), created_by="test")
        exec_id = wfdb.start_execution(conn, "demo", input_data={"score": 0.9}, trigger_type="manual")

    assert workflows_dispatcher.tick(limit=1) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        events = [row["kind"] for row in conn.execute(
            "SELECT kind FROM workflow_events WHERE execution_id = ? ORDER BY id",
            (exec_id,),
        )]
    assert execution.status == "succeeded"
    assert execution.context["node"]["high"]["output"] == {"bucket": "high"}
    assert "execution_started" in events
    assert "node_succeeded" in events
    assert "execution_succeeded" in events


def test_tick_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _switch_spec(), created_by="test")
        first = wfdb.start_execution(conn, "demo", input_data={"score": 0.9}, trigger_type="manual")
        second = wfdb.start_execution(conn, "demo", input_data={"score": 0.1}, trigger_type="manual")

    assert workflows_dispatcher.tick(limit=1) == 1

    with wfdb.connect() as conn:
        statuses = {
            exec_id: wfdb.get_execution(conn, exec_id).status
            for exec_id in (first, second)
        }
    assert sorted(statuses.values()) == ["queued", "succeeded"]
