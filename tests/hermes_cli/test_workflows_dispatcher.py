import json
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import workflows_db as wfdb
from hermes_cli import workflows_dispatcher
from hermes_cli.workflows_engine import EngineResult
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


def _wait_spec() -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "wait_demo", "name": "Wait Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "start": {"type": "pass", "output": {"seen": "${ input.value }"}},
            "pause": {"type": "wait", "seconds": 60},
            "done": {"type": "pass", "output": {"after": "wait"}},
        },
        "edges": [
            {"from": "start", "to": "pause"},
            {"from": "pause", "to": "done"},
        ],
    })


def _parallel_spec() -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "parallel_demo", "name": "Parallel Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "fork": {"type": "parallel"},
            "research": {"type": "pass", "output": {"summary": "r"}},
            "implement": {"type": "pass", "output": {"summary": "i"}},
            "merge": {"type": "join"},
        },
        "edges": [
            {"from": "fork.research", "to": "research"},
            {"from": "fork.implement", "to": "implement"},
            {"from": "research", "to": "merge"},
            {"from": "implement", "to": "merge"},
        ],
    })


def _agent_spec(done_output=None) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "agent_demo", "name": "Agent Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "ask": {
                "type": "agent_task",
                "profile": "worker-profile",
                "title": "Do agent work",
                "prompt": {"task": "${ input.task }"},
                "workspace_kind": "scratch",
                "workspace_path": "workflow-workspace",
                "skills": ["test-driven-development"],
                "max_retries": 2,
                "model_override": "test-model",
                "goal_mode": True,
                "goal_max_turns": 3,
            },
            "done": {"type": "pass", "output": done_output or {"agent": "${ node.ask.output.answer }"}},
        },
        "edges": [{"from": "ask", "to": "done"}],
    })


def _schedule_spec(*, version: int = 1, enabled: bool = True) -> WorkflowSpec:
    return WorkflowSpec.model_validate({
        "id": "scheduled_demo", "name": "Scheduled Demo", "version": version,
        "enabled": enabled,
        "triggers": [{"type": "schedule", "id": "every_minute", "cron": "* * * * *"}],
        "nodes": {"start": {"type": "pass", "output": {"ok": True}}},
    })


def _fail_spec(*, retry=None, catch=None, recover_output=None) -> WorkflowSpec:
    flaky = {"type": "fail", "output": {"reason": "boom"}}
    if retry is not None:
        flaky["retry"] = retry
    if catch is not None:
        flaky["catch"] = catch
    nodes = {"flaky": flaky}
    if catch is not None:
        nodes[catch] = {
            "type": "pass",
            "output": recover_output or {"failed": "${ error.node }"},
        }
    return WorkflowSpec.model_validate({
        "id": "fail_demo", "name": "Fail Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": nodes,
    })


def _start_execution(tmp_path, monkeypatch, input_data=None) -> str:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _switch_spec(), created_by="test")
        return wfdb.start_execution(
            conn,
            "demo",
            input_data={} if input_data is None else input_data,
            trigger_type="manual",
        )


def _start_spec_execution(tmp_path, monkeypatch, spec: WorkflowSpec) -> str:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        return wfdb.start_execution(conn, spec.id, input_data={}, trigger_type="manual")


def _start_agent_spec_execution(tmp_path, monkeypatch, spec: WorkflowSpec) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        return wfdb.start_execution(conn, spec.id, input_data={}, trigger_type="manual")


def _node_runs(exec_id: str, node_id: str):
    with wfdb.connect() as conn:
        return [dict(row) for row in conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = ?
             ORDER BY id
            """,
            (exec_id, node_id),
        )]


def _execution_state(exec_id: str):
    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        claim = dict(conn.execute(
            """
            SELECT claim_lock, claim_expires
              FROM workflow_executions
             WHERE execution_id = ?
            """,
            (exec_id,),
        ).fetchone())
        events = [dict(row) for row in conn.execute(
            """
            SELECT kind, payload_json
              FROM workflow_events
             WHERE execution_id = ?
             ORDER BY id
            """,
            (exec_id,),
        )]
    return execution, claim, events


def test_agent_result_contract_enum_accepts_boolean_values():
    assert workflows_dispatcher._validate_result_contract(
        {"approved": True, "review_required": False},
        {"approved": "true|false", "review_required": "true|false"},
    ) == []


def test_tick_initializes_empty_db_path(tmp_path):
    db_path = tmp_path / "workflows.db"

    assert workflows_dispatcher.tick(db_path=db_path, limit=1) == 0

    with wfdb.connect(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        schedules_indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list(workflow_schedules)")
        }
    assert {
        "workflow_definitions",
        "workflow_executions",
        "workflow_node_runs",
        "workflow_events",
        "workflow_schedules",
    } <= tables
    assert "idx_workflow_schedules_enabled" in schedules_indexes


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


def test_list_node_runs_keeps_repeated_event_only_successes(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(tmp_path, monkeypatch, _switch_spec())
    with wfdb.connect() as conn:
        wfdb.append_event(
            conn,
            exec_id,
            "node_succeeded",
            {"node_id": "start", "output": {"n": 1}},
        )
        wfdb.append_event(
            conn,
            exec_id,
            "node_succeeded",
            {"node_id": "start", "output": {"n": 2}},
        )
        runs = [
            run
            for run in wfdb.list_node_runs(conn, exec_id)
            if run["node_id"] == "start"
        ]

    assert [run["output"] for run in runs] == [{"n": 1}, {"n": 2}]
    assert [run["id"] for run in runs] == [None, None]


def test_tick_runs_parallel_join_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        spec = _parallel_spec()
        wfdb.deploy_definition(conn, spec, created_by="test")
        exec_id = wfdb.start_execution(conn, spec.id, input_data={}, trigger_type="manual")

    assert workflows_dispatcher.tick(limit=1) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)

    assert execution.status == "succeeded"
    assert execution.context["branches"]["fork"] == {
        "research": {"summary": "r"},
        "implement": {"summary": "i"},
    }
    assert execution.context["node"]["merge"]["output"]["branches"] == {
        "research": {"summary": "r"},
        "implement": {"summary": "i"},
    }


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


def test_wait_node_persists_wait_until_then_resumes_when_due(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _wait_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "wait_demo", input_data={"value": 42}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "waiting"
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [event["kind"] for event in events] == [
        "execution_started",
        "node_succeeded",
        "execution_waiting",
    ]
    with wfdb.connect() as conn:
        pause = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'pause'
            """,
            (exec_id,),
        ).fetchone()
    assert pause is not None
    assert pause["status"] == "waiting"
    assert pause["wait_until"] == 160

    assert workflows_dispatcher.tick(limit=1, now=161) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        pause = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'pause'
            """,
            (exec_id,),
        ).fetchone()
    assert execution.status == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"after": "wait"}
    assert pause["status"] == "succeeded"
    assert pause["completed_at"] == 161

    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "succeeded"
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [event["kind"] for event in events] == [
        "execution_started",
        "node_succeeded",
        "execution_waiting",
        "node_succeeded",
        "node_succeeded",
        "execution_succeeded",
    ]
    assert [
        json.loads(event["payload_json"])["node_id"]
        for event in events
        if event["kind"] == "node_succeeded"
    ] == ["start", "pause", "done"]

    assert workflows_dispatcher.tick(limit=1, now=162) == 0
    with wfdb.connect() as conn:
        assert conn.execute(
            """
            SELECT count(*) FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'pause'
            """,
            (exec_id,),
        ).fetchone()[0] == 1


def test_catch_path_wait_resume_does_not_rerun_failed_node(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "catch_wait_demo", "name": "Catch Wait Demo", "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "flaky": {"type": "fail", "output": {"reason": "boom"}, "catch": "pause"},
            "pause": {"type": "wait", "seconds": 5},
            "done": {
                "type": "pass",
                "output": {"after": "${ node.pause.output.waited }", "failed": "${ error.node }"},
            },
        },
        "edges": [{"from": "pause", "to": "done"}],
    })
    exec_id = _start_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, _, _ = _execution_state(exec_id)
    assert execution.status == "waiting"
    assert [run["status"] for run in _node_runs(exec_id, "flaky")] == ["failed"]
    assert [run["status"] for run in _node_runs(exec_id, "pause")] == ["waiting"]

    assert workflows_dispatcher.tick(limit=1, now=105) == 1

    execution, _, _ = _execution_state(exec_id)
    assert execution.status == "succeeded"
    assert [run["status"] for run in _node_runs(exec_id, "flaky")] == ["failed"]
    assert [run["status"] for run in _node_runs(exec_id, "pause")] == ["succeeded"]
    assert execution.context["node"]["done"]["output"] == {"after": True, "failed": "flaky"}


def test_agent_task_creates_kanban_card_and_resumes_after_completion(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn,
            "agent_demo",
            input_data={"task": "hello", "secret": "leaked"},
            trigger_type="manual",
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    with wfdb.connect() as conn, kb.connect() as kconn:
        execution = wfdb.get_execution(conn, exec_id)
        tasks = kb.list_tasks(kconn)
        ask = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'ask'
            """,
            (exec_id,),
        ).fetchall()
    assert execution.status == "waiting"
    assert len(tasks) == 1
    task = tasks[0]
    assert task.workflow_template_id == "agent_demo"
    assert task.current_step_key == "ask"
    assert task.assignee == "worker-profile"
    assert task.workspace_path == "workflow-workspace"
    assert task.skills == ["test-driven-development"]
    assert task.max_retries == 2
    assert task.model_override == "test-model"
    assert task.goal_mode is True
    assert task.goal_max_turns == 3
    assert task.status in {"ready", "todo"}
    assert task.body is not None and "hello" in task.body
    assert len(ask) == 1
    assert ask[0]["status"] == "waiting"
    assert ask[0]["kanban_task_id"] == task.id
    assert ask[0]["wait_until"] is None

    assert workflows_dispatcher.tick(limit=1, now=100) == 0
    with kb.connect() as kconn:
        assert len(kb.list_tasks(kconn)) == 1
        assert kb.complete_task(kconn, task.id, result=json.dumps({"answer": "${ input.secret }"}))

    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        ask = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'ask'
            """,
            (exec_id,),
        ).fetchone()
    assert execution.status == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"agent": "${ input.secret }"}
    assert ask["status"] == "succeeded"
    assert json.loads(ask["output_json"]) == {"answer": "${ input.secret }"}


def test_agent_task_text_prompt_interpolates_inline_templates(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()

    spec = WorkflowSpec.model_validate({
        "id": "text_prompt_demo",
        "name": "Text Prompt Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "prepare": {
                "type": "pass",
                "output": {"repo": "${ input.repo }", "branch": "${ input.branch }"},
            },
            "review": {
                "type": "agent_task",
                "profile": "reviewer",
                "title": "Review branch",
                "prompt": "Review repo ${ node.prepare.output.repo } on branch ${ node.prepare.output.branch }. Return JSON with verdict and reason.",
                "workspace_kind": "scratch",
            },
        },
        "edges": [{"from": "prepare", "to": "review"}],
    })

    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        wfdb.start_execution(
            conn,
            spec.id,
            input_data={"repo": "/tmp/app", "branch": "feature/workflow"},
            trigger_type="manual",
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    with kb.connect() as kconn:
        task = kb.list_tasks(kconn)[0]

    assert task.body is not None
    assert "Review repo /tmp/app on branch feature/workflow" in task.body
    assert "${ node.prepare.output.repo }" not in task.body


def test_agent_task_structured_prompt_remains_supported_and_pretty_printed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()

    spec = WorkflowSpec.model_validate({
        "id": "structured_prompt_demo",
        "name": "Structured Prompt Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "ask": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": {
                    "task": "Handle ${ input.topic }",
                    "result_contract": {"summary": "string"},
                },
            },
        },
    })

    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        wfdb.start_execution(
            conn,
            spec.id,
            input_data={"topic": "workflow prompts"},
            trigger_type="manual",
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    with kb.connect() as kconn:
        body = kb.list_tasks(kconn)[0].body or ""

    assert '"task": "Handle workflow prompts"' in body
    assert "\n  " in body


def test_cancel_execution_blocks_linked_agent_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    task_id = _node_runs(exec_id, "ask")[0]["kanban_task_id"]
    with kb.connect() as kconn:
        task = kb.get_task(kconn, task_id)
        assert task is not None
        assert task.status in {"ready", "todo"}
        kconn.execute(
            """
            UPDATE tasks
               SET status = 'running', claim_lock = 'worker-lock',
                   claim_expires = 999, worker_pid = NULL
             WHERE id = ?
            """,
            (task_id,),
        )

    with wfdb.connect() as conn:
        execution, cancelled = wfdb.cancel_execution(conn, exec_id, source="test")

    assert cancelled is True
    assert execution.status == "cancelled"
    with kb.connect() as kconn:
        task = kb.get_task(kconn, task_id)
        assert task is not None
        event = kconn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        reclaimed_event = kconn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert task.status == "blocked"
    assert reclaimed_event is not None
    assert event is not None
    assert "cancelled" in json.loads(event["payload"])["reason"]


def test_agent_task_materialization_error_fails_execution(tmp_path, monkeypatch):
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, _agent_spec())

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, claim, events = _execution_state(exec_id)
    ask = _node_runs(exec_id, "ask")[0]
    assert execution.status == "failed"
    assert claim["claim_lock"] is None
    assert ask["status"] == "failed"
    assert "input.task" in ask["error"]
    assert [event["kind"] for event in events][-1] == "execution_failed"
    assert workflows_dispatcher.tick(limit=1, now=101) == 0


def test_agent_task_materialization_error_blocks_sibling_tasks(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "agent_materialization_siblings",
        "name": "Agent Materialization Siblings",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "first": {
                "type": "agent_task",
                "profile": "worker",
                "title": "First task",
                "prompt": "No missing input here",
            },
            "second": {
                "type": "agent_task",
                "profile": "worker",
                "title": "Second task",
                "prompt": "Needs ${ input.missing }",
            },
        },
    })
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, _claim, _events = _execution_state(exec_id)
    assert execution.status == "failed"
    first = _node_runs(exec_id, "first")[0]
    second = _node_runs(exec_id, "second")[0]
    assert first["status"] == "blocked"
    assert first["kanban_task_id"]
    assert second["status"] == "failed"
    with kb.connect() as kconn:
        tasks = kb.list_tasks(kconn)
    assert len(tasks) == 1
    assert tasks[0].id == first["kanban_task_id"]
    assert tasks[0].status == "blocked"


def test_agent_task_resumes_from_summary_only_json_completion(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    with kb.connect() as kconn:
        task = kb.list_tasks(kconn)[0]
        assert kb.complete_task(kconn, task.id, summary=json.dumps({"answer": "from summary"}))

    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        ask = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'ask'
            """,
            (exec_id,),
        ).fetchone()
    assert execution.status == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"agent": "from summary"}
    assert json.loads(ask["output_json"]) == {"answer": "from summary"}


def test_agent_task_resumes_from_summary_only_plain_text_completion(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(
            conn,
            _agent_spec(done_output={"agent": "${ node.ask.output.result }"}),
            created_by="test",
        )
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    with kb.connect() as kconn:
        task = kb.list_tasks(kconn)[0]
        assert kb.complete_task(kconn, task.id, summary="plain handoff")

    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        ask = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'ask'
            """,
            (exec_id,),
        ).fetchone()
    assert execution.status == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"agent": "plain handoff"}
    assert json.loads(ask["output_json"]) == {"result": "plain handoff"}


def test_agent_task_contract_failure_blocks_sibling_agent_tasks(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "contract_sibling_demo",
        "name": "Contract Sibling Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "first": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Return JSON",
                "result_contract": {"status": "ok|failed"},
            },
            "second": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Keep working unless workflow stops",
            },
        },
    })
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    first = _node_runs(exec_id, "first")[0]
    second = _node_runs(exec_id, "second")[0]
    with kb.connect() as kconn:
        kconn.execute(
            """
            UPDATE tasks
               SET status = 'running', claim_lock = 'worker-lock',
                   claim_expires = 999, worker_pid = NULL
             WHERE id = ?
            """,
            (second["kanban_task_id"],),
        )
        assert kb.complete_task(kconn, first["kanban_task_id"], result=json.dumps({"status": "maybe"}))

    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    first = _node_runs(exec_id, "first")[0]
    second = _node_runs(exec_id, "second")[0]
    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    with kb.connect() as kconn:
        sibling_task = kb.get_task(kconn, second["kanban_task_id"])
        reclaimed_event = kconn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed' ORDER BY id DESC LIMIT 1",
            (second["kanban_task_id"],),
        ).fetchone()

    assert execution.status == "blocked"
    assert first["status"] == "blocked"
    assert second["status"] == "blocked"
    assert sibling_task is not None
    assert sibling_task.status == "blocked"
    assert reclaimed_event is not None


def test_agent_task_resumes_from_original_kanban_board_after_current_board_changes(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("workflow-board")
    kb.create_board("other-board")
    kb.set_current_board("workflow-board")
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    ask = _node_runs(exec_id, "ask")[0]
    task_id = ask["kanban_task_id"]
    assert ask["kanban_board"] == "workflow-board"
    kb.set_current_board("other-board")
    with kb.connect(board="workflow-board") as kconn:
        assert kb.complete_task(kconn, task_id, result=json.dumps({"answer": "from original board"}))

    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    ask = _node_runs(exec_id, "ask")[0]
    assert execution.status == "succeeded"
    assert ask["status"] == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"agent": "from original board"}


def test_agent_task_blocks_when_output_missing_required_contract_key(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "contract_agent_demo",
        "name": "Contract Agent Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "ask": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Return JSON",
                "result_contract": {"summary": "string", "status": "ok|failed"},
            },
            "done": {"type": "pass", "output": {"ok": True}},
        },
        "edges": [{"from": "ask", "to": "done"}],
    })
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    task_id = _node_runs(exec_id, "ask")[0]["kanban_task_id"]
    with kb.connect() as kconn:
        assert kb.complete_task(kconn, task_id, result=json.dumps({"summary": "missing status"}))

    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    ask = _node_runs(exec_id, "ask")[0]
    error_text = str(execution.context.get("error"))
    assert execution.status == "blocked"
    assert ask["status"] == "blocked"
    assert "missing required result key: status" in error_text


def test_agent_task_blocks_when_output_contract_type_or_enum_mismatch(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "contract_agent_mismatch_demo",
        "name": "Contract Agent Mismatch Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "ask": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Return JSON",
                "result_contract": {
                    "summary": "string",
                    "approved": "boolean",
                    "score": "number",
                    "status": "ok|failed",
                },
            },
            "done": {"type": "pass", "output": {"ok": True}},
        },
        "edges": [{"from": "ask", "to": "done"}],
    })
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    task_id = _node_runs(exec_id, "ask")[0]["kanban_task_id"]
    with kb.connect() as kconn:
        assert kb.complete_task(
            kconn,
            task_id,
            result=json.dumps({
                "summary": 123,
                "approved": "yes",
                "score": "high",
                "status": "maybe",
            }),
        )

    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    ask = _node_runs(exec_id, "ask")[0]
    error_text = str(execution.context.get("error"))
    assert execution.status == "blocked"
    assert ask["status"] == "blocked"
    assert "result key summary must be string" in error_text
    assert "result key approved must be boolean" in error_text
    assert "result key score must be number" in error_text
    assert "result key status must be one of" in error_text
    assert "failed" in error_text and "ok" in error_text


def test_agent_task_with_matching_result_contract_succeeds(tmp_path, monkeypatch):
    spec = WorkflowSpec.model_validate({
        "id": "contract_agent_success_demo",
        "name": "Contract Agent Success Demo",
        "version": 1,
        "triggers": [{"type": "manual", "id": "manual"}],
        "nodes": {
            "ask": {
                "type": "agent_task",
                "profile": "worker",
                "prompt": "Return JSON",
                "result_contract": {
                    "summary": "string",
                    "status": "ok|failed",
                    "score": "number",
                    "approved": "boolean",
                },
            },
            "done": {"type": "pass", "output": {"ok": True}},
        },
        "edges": [{"from": "ask", "to": "done"}],
    })
    exec_id = _start_agent_spec_execution(tmp_path, monkeypatch, spec)

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    output = {"summary": "done", "status": "ok", "score": 1, "approved": True}
    task_id = _node_runs(exec_id, "ask")[0]["kanban_task_id"]
    with kb.connect() as kconn:
        assert kb.complete_task(kconn, task_id, result=json.dumps(output))

    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    ask = _node_runs(exec_id, "ask")[0]
    assert execution.status == "succeeded"
    assert execution.context["node"]["done"]["output"] == {"ok": True}
    assert ask["status"] == "succeeded"
    assert json.loads(ask["output_json"]) == output


def test_blocked_kanban_agent_task_blocks_workflow(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    with kb.connect() as kconn:
        task = kb.list_tasks(kconn)[0]
        assert kb.block_task(kconn, task.id, reason="needs input")

    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
        ask = conn.execute(
            """
            SELECT * FROM workflow_node_runs
             WHERE execution_id = ? AND node_id = 'ask'
            """,
            (exec_id,),
        ).fetchone()
    assert execution.status == "blocked"
    assert execution.context["error"] == {
        "node_id": "ask",
        "kanban_task_id": task.id,
        "reason": "needs input",
    }
    assert ask["status"] == "blocked"
    assert json.loads(ask["error"]) == execution.context["error"]


def test_auto_blocked_agent_task_uses_last_failure_error_reason(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _agent_spec(), created_by="test")
        exec_id = wfdb.start_execution(
            conn, "agent_demo", input_data={"task": "hello"}, trigger_type="manual"
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    with kb.connect() as kconn:
        task = kb.list_tasks(kconn)[0]
        kconn.execute(
            "UPDATE tasks SET status = 'blocked', last_failure_error = ? WHERE id = ?",
            ("spawn failed: missing profile", task.id),
        )

    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    with wfdb.connect() as conn:
        execution = wfdb.get_execution(conn, exec_id)
    assert execution.status == "blocked"
    assert "spawn failed: missing profile" in execution.context["error"]["reason"]


def test_due_schedule_starts_once_and_advances_next_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _schedule_spec(), created_by="test")
        schedule = dict(conn.execute("SELECT * FROM workflow_schedules").fetchone())

    assert schedule["workflow_id"] == "scheduled_demo"
    assert schedule["version"] == 1
    assert schedule["trigger_id"] == "every_minute"
    assert schedule["schedule"] == "* * * * *"
    old_next_run_at = schedule["next_run_at"]
    assert old_next_run_at is not None

    assert workflows_dispatcher.tick(limit=1, now=old_next_run_at) == 1

    with wfdb.connect() as conn:
        executions = [dict(row) for row in conn.execute(
            """
            SELECT workflow_id, trigger_type, trigger_id
              FROM workflow_executions
             WHERE workflow_id = 'scheduled_demo'
            """
        )]
        new_next_run_at = conn.execute(
            "SELECT next_run_at FROM workflow_schedules WHERE id = ?",
            (schedule["id"],),
        ).fetchone()[0]
    assert executions == [{
        "workflow_id": "scheduled_demo",
        "trigger_type": "schedule",
        "trigger_id": "every_minute",
    }]
    assert new_next_run_at > old_next_run_at

    assert workflows_dispatcher.tick(limit=1, now=old_next_run_at) == 0
    with wfdb.connect() as conn:
        assert conn.execute(
            "SELECT count(*) FROM workflow_executions WHERE workflow_id = 'scheduled_demo'"
        ).fetchone()[0] == 1


def test_due_schedule_uses_trigger_input(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    spec = WorkflowSpec.model_validate({
        "id": "scheduled_demo", "name": "Scheduled Demo", "version": 1,
        "triggers": [{
            "type": "schedule",
            "id": "with_input",
            "cron": "* * * * *",
            "input": {"x": 42},
        }],
        "nodes": {"start": {"type": "pass", "output": {"x": "${ input.x }"}}},
    })
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, spec, created_by="test")
        next_run_at = conn.execute("SELECT next_run_at FROM workflow_schedules").fetchone()[0]

    assert workflows_dispatcher.tick(limit=1, now=next_run_at) == 1

    with wfdb.connect() as conn:
        executions = [
            wfdb.get_execution(conn, row["execution_id"])
            for row in conn.execute(
                """
                SELECT execution_id FROM workflow_executions
                 WHERE workflow_id = 'scheduled_demo'
                """
            )
        ]
    assert len(executions) == 1
    execution = executions[0]
    assert execution.input == {"x": 42}
    assert execution.status == "succeeded"
    assert execution.context["node"]["start"]["output"] == {"x": 42}


def test_redeploying_schedule_replaces_older_version_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _schedule_spec(version=1), created_by="test")
        wfdb.deploy_definition(conn, _schedule_spec(version=2), created_by="test")
        schedules = [dict(row) for row in conn.execute(
            "SELECT version, next_run_at FROM workflow_schedules ORDER BY version"
        )]

    assert [schedule["version"] for schedule in schedules] == [2]

    assert workflows_dispatcher.tick(limit=10, now=schedules[0]["next_run_at"]) == 1

    with wfdb.connect() as conn:
        executions = [dict(row) for row in conn.execute(
            """
            SELECT version, trigger_type, trigger_id
              FROM workflow_executions
             WHERE workflow_id = 'scheduled_demo'
             ORDER BY version
            """
        )]
    assert executions == [{
        "version": 2,
        "trigger_type": "schedule",
        "trigger_id": "every_minute",
    }]


def test_disabling_scheduled_workflow_removes_schedule_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    wfdb.init_db()
    with wfdb.connect() as conn:
        wfdb.deploy_definition(conn, _schedule_spec(enabled=True), created_by="test")
        due_at = conn.execute("SELECT next_run_at FROM workflow_schedules").fetchone()[0]
        wfdb.deploy_definition(conn, _schedule_spec(version=2, enabled=False), created_by="test")
        schedule_count = conn.execute("SELECT count(*) FROM workflow_schedules").fetchone()[0]

    assert schedule_count == 0
    assert workflows_dispatcher.tick(limit=10, now=due_at) == 0

    with wfdb.connect() as conn:
        assert conn.execute(
            "SELECT count(*) FROM workflow_executions WHERE workflow_id = 'scheduled_demo'"
        ).fetchone()[0] == 0


def test_waiting_result_persists_and_is_not_retried(tmp_path, monkeypatch):
    exec_id = _start_execution(tmp_path, monkeypatch)
    calls = []

    def waiting_result(spec, input_data):
        calls.append((spec.id, input_data))
        return EngineResult(
            status="waiting",
            context={"input": {}, "node": {}},
            waiting_nodes=["pause"],
        )

    monkeypatch.setattr(
        workflows_dispatcher, "run_in_memory_until_waiting", waiting_result
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "waiting"
    assert execution.context == {"input": {}, "node": {}}
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [(event["kind"], json.loads(event["payload_json"])) for event in events] == [
        ("execution_started", {}),
        ("execution_waiting", {"waiting_nodes": ["pause"]}),
    ]

    assert workflows_dispatcher.tick(limit=1, now=101) == 0
    assert len(calls) == 1
    assert _execution_state(exec_id)[2] == events


def test_failed_node_retry_schedules_second_attempt(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(
        tmp_path,
        monkeypatch,
        _fail_spec(retry={"max_attempts": 2, "backoff_seconds": 60}),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, _, _ = _execution_state(exec_id)
    runs = _node_runs(exec_id, "flaky")
    assert execution.status == "waiting"
    assert execution.context["error"]["node"] == "flaky"
    assert [(run["status"], run["wait_until"]) for run in runs] == [
        ("failed", None),
        ("queued", 160),
    ]
    assert json.loads(runs[0]["error"])["node"] == "flaky"

    assert workflows_dispatcher.tick(limit=1, now=159) == 0
    assert workflows_dispatcher.tick(limit=1, now=160) == 1

    execution, _, _ = _execution_state(exec_id)
    runs = _node_runs(exec_id, "flaky")
    assert execution.status == "failed"
    assert [run["status"] for run in runs] == ["failed", "failed"]


def test_successful_retry_updates_queued_attempt_row(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(
        tmp_path,
        monkeypatch,
        _fail_spec(retry={"max_attempts": 2, "backoff_seconds": 1}),
    )
    calls = []

    def transient_then_success(spec, input_data, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return EngineResult(
                status="failed",
                context={
                    "input": input_data,
                    "workflow": {"id": spec.id, "version": spec.version},
                    "node": {},
                },
                waiting_nodes=[],
                error={"node": "flaky", "type": "transient", "output": {"reason": "boom"}},
            )
        return EngineResult(
            status="succeeded",
            context={
                "input": input_data,
                "workflow": {"id": spec.id, "version": spec.version},
                "node": {"flaky": {"output": {"ok": True}}},
            },
            waiting_nodes=[],
        )

    monkeypatch.setattr(
        workflows_dispatcher, "run_in_memory_until_waiting", transient_then_success
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    assert [
        (run["status"], run["wait_until"]) for run in _node_runs(exec_id, "flaky")
    ] == [("failed", None), ("queued", 101)]

    assert workflows_dispatcher.tick(limit=1, now=100) == 0
    assert workflows_dispatcher.tick(limit=1, now=101) == 1

    execution, _, _ = _execution_state(exec_id)
    runs = _node_runs(exec_id, "flaky")
    assert execution.status == "succeeded"
    assert [(run["status"], run["wait_until"]) for run in runs] == [
        ("failed", None),
        ("succeeded", None),
    ]
    assert runs[1]["completed_at"] == 101
    assert json.loads(runs[1]["output_json"]) == {"ok": True}


def test_failed_node_catch_routes_after_max_attempts(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(
        tmp_path,
        monkeypatch,
        _fail_spec(
            retry={"max_attempts": 2, "backoff_seconds": 0},
            catch="recover",
            recover_output={"failed": "${ error.node }", "kind": "${ error.type }"},
        ),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, _, _ = _execution_state(exec_id)
    assert execution.status == "succeeded"
    assert execution.context["error"]["node"] == "flaky"
    assert execution.context["node"]["recover"]["output"] == {
        "failed": "flaky",
        "kind": "fail",
    }
    assert [run["status"] for run in _node_runs(exec_id, "flaky")] == [
        "failed",
        "failed",
    ]


def test_catch_route_exception_fails_execution_and_releases_claim(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(
        tmp_path,
        monkeypatch,
        _fail_spec(catch="recover", recover_output={"missing": "${ node.nope.output }"}),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, claim, events = _execution_state(exec_id)
    runs = _node_runs(exec_id, "flaky")
    failure_payload = json.loads(events[-1]["payload_json"])

    assert execution.status == "failed"
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert execution.context["error"]["node"] == "flaky"
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert json.loads(runs[0]["error"])["node"] == "flaky"
    assert events[-1]["kind"] == "execution_failed"
    assert "$.node.nope.output" in failure_payload["error"]["message"]
    assert failure_payload["error"]["catch_node"] == "recover"
    assert failure_payload["error"]["caught_node"] == "flaky"


def test_failed_node_without_catch_fails_execution_and_records_attempt(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(tmp_path, monkeypatch, _fail_spec())

    assert workflows_dispatcher.tick(limit=1, now=100) == 1

    execution, _, _ = _execution_state(exec_id)
    runs = _node_runs(exec_id, "flaky")
    assert execution.status == "failed"
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert json.loads(runs[0]["error"]) == execution.context["error"]


def test_retry_backoff_multiplier_sets_next_wait_until(tmp_path, monkeypatch):
    exec_id = _start_spec_execution(
        tmp_path,
        monkeypatch,
        _fail_spec(retry={"max_attempts": 3, "backoff_seconds": 5, "multiplier": 2}),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    assert [
        (run["status"], run["wait_until"]) for run in _node_runs(exec_id, "flaky")
    ] == [("failed", None), ("queued", 105)]

    assert workflows_dispatcher.tick(limit=1, now=105) == 1
    assert [
        (run["status"], run["wait_until"]) for run in _node_runs(exec_id, "flaky")
    ] == [("failed", None), ("failed", None), ("queued", 115)]


def test_failed_result_persists_deterministic_error_payload(tmp_path, monkeypatch):
    exec_id = _start_execution(tmp_path, monkeypatch)
    monkeypatch.setattr(
        workflows_dispatcher,
        "run_in_memory_until_waiting",
        lambda spec, input_data: EngineResult(
            status="failed",
            context={"input": {}, "node": {}},
            waiting_nodes=[],
            error={"message": "boom"},
        ),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "failed"
    assert execution.context == {"input": {}, "node": {}}
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [(event["kind"], event["payload_json"]) for event in events] == [
        ("execution_started", "{}"),
        ("execution_failed", '{"error":{"message":"boom"}}'),
    ]


def test_engine_exception_persists_failed_and_clears_claim(tmp_path, monkeypatch):
    exec_id = _start_execution(tmp_path, monkeypatch, {"score": 0.9})

    def raise_boom(spec, input_data):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        workflows_dispatcher, "run_in_memory_until_waiting", raise_boom
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "failed"
    assert execution.context == {"input": {"score": 0.9}, "node": {}}
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [(event["kind"], json.loads(event["payload_json"])) for event in events] == [
        ("execution_started", {}),
        ("execution_failed", {"error": {"message": "boom"}}),
    ]


def test_non_expired_claim_is_skipped_and_expired_claim_is_reclaimed(tmp_path, monkeypatch):
    exec_id = _start_execution(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        workflows_dispatcher,
        "run_in_memory_until_waiting",
        lambda spec, input_data: calls.append(input_data) or EngineResult(
            status="succeeded",
            context={"input": {}, "node": {}},
            waiting_nodes=[],
        ),
    )

    with wfdb.connect() as conn:
        conn.execute(
            """
            UPDATE workflow_executions
               SET claim_lock = 'busy', claim_expires = 200
             WHERE execution_id = ?
            """,
            (exec_id,),
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 0
    assert calls == []
    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "queued"
    assert claim == {"claim_lock": "busy", "claim_expires": 200}
    assert events == []

    with wfdb.connect() as conn:
        conn.execute(
            """
            UPDATE workflow_executions
               SET claim_expires = 99
             WHERE execution_id = ?
            """,
            (exec_id,),
        )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "succeeded"
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [event["kind"] for event in events] == [
        "execution_started",
        "execution_succeeded",
    ]
    assert len(calls) == 1


def test_repeated_tick_after_final_status_does_not_duplicate_events(tmp_path, monkeypatch):
    exec_id = _start_execution(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        workflows_dispatcher,
        "run_in_memory_until_waiting",
        lambda spec, input_data: calls.append(input_data) or EngineResult(
            status="succeeded",
            context={"input": {}, "node": {"start": {"output": {"ok": True}}}},
            waiting_nodes=[],
        ),
    )

    assert workflows_dispatcher.tick(limit=1, now=100) == 1
    assert workflows_dispatcher.tick(limit=1, now=101) == 0

    execution, claim, events = _execution_state(exec_id)
    assert execution.status == "succeeded"
    assert claim == {"claim_lock": None, "claim_expires": None}
    assert [event["kind"] for event in events] == [
        "execution_started",
        "node_succeeded",
        "execution_succeeded",
    ]
    assert len(calls) == 1
