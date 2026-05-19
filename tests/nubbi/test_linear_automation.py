import os


class FakeLinearClient:
    def __init__(self):
        self.comments = []
        self.transitions = []
        self.issues = [
            {
                "id": "lin-1",
                "identifier": "MEM-77",
                "title": "Set up Nubbi automation",
                "description": "Monitor the Nubbi Command Center project.",
                "url": "https://linear.app/nubbi/issue/MEM-77",
                "state": {"name": "Todo"},
                "project": {"name": "Nubbi Command Center"},
                "blocked": False,
            }
        ]

    def list_project_issues(self, project_name, state_names):
        assert project_name == "Nubbi Command Center"
        assert state_names == ("Todo", "Backlog")
        return self.issues

    def add_comment(self, issue_id, body):
        self.comments.append((issue_id, body))

    def transition_issue(self, issue_id, state_name):
        self.transitions.append((issue_id, state_name))
        for issue in self.issues:
            if issue["id"] == issue_id:
                issue["state"] = {"name": state_name}


class SyncLinearClient:
    def __init__(self, issue):
        self.issue = issue
        self.requested_states = []
        self.transitions = []

    def list_project_issues(self, project_name, state_names):
        self.requested_states.append(tuple(state_names))
        return [self.issue]

    def add_comment(self, issue_id, body):
        raise AssertionError("existing mirrored issues should not get a start comment")

    def transition_issue(self, issue_id, state_name):
        self.transitions.append((issue_id, state_name))


def test_tick_imports_actionable_issue_once_and_marks_linear_in_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LINEAR_API_KEY", "lin-test")

    from hermes_cli import kanban_db
    from hermes_cli.nubbi_linear_automation import tick

    client = FakeLinearClient()
    config = {
        "enabled": True,
        "project_name": "Nubbi Command Center",
        "source_states": ["Todo", "Backlog"],
        "start_state": "In Progress",
        "review_state": "In Review",
        "done_state": "Done",
        "kanban_board": "default",
        "assignee": "codex",
        "workspace_kind": "dir",
        "workspace_path": os.getcwd(),
    }

    first = tick(config=config, linear_client=client)
    second = tick(config=config, linear_client=client)

    conn = kanban_db.connect(board="default")
    try:
        tasks = kanban_db.list_tasks(conn)
    finally:
        conn.close()

    assert first.created == 1
    assert first.skipped_duplicates == 0
    assert second.created == 0
    assert second.skipped_duplicates == 1
    assert len(tasks) == 1
    assert tasks[0].title == "MEM-77: Set up Nubbi automation"
    assert tasks[0].assignee == "codex"
    assert "https://linear.app/nubbi/issue/MEM-77" in (tasks[0].body or "")
    assert client.comments == [
        (
            "lin-1",
            "Nubbi automation started work by creating Hermes Kanban task "
            f"{tasks[0].id}.",
        )
    ]
    assert client.transitions == [("lin-1", "In Progress")]


def test_tick_syncs_existing_review_and_done_tasks_to_linear_terminal_states(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LINEAR_API_KEY", "lin-test")

    from hermes_cli import kanban_db
    from hermes_cli.nubbi_linear_automation import tick

    config = {
        "enabled": True,
        "project_name": "Nubbi Command Center",
        "source_states": ["Todo", "Backlog"],
        "active_states": ["In Progress", "In Review"],
        "start_state": "In Progress",
        "review_state": "In Review",
        "done_state": "Done",
        "kanban_board": "default",
        "assignee": "codex",
    }
    issue = {
        "id": "lin-2",
        "identifier": "MEM-78",
        "title": "Review sync",
        "description": "",
        "url": "https://linear.app/nubbi/issue/MEM-78",
        "state": {"name": "In Progress"},
        "project": {"name": "Nubbi Command Center"},
        "blocked": False,
    }

    conn = kanban_db.connect(board="default")
    try:
        task_id = kanban_db.create_task(
            conn,
            title="MEM-78: Review sync",
            body="existing task",
            assignee="codex",
            created_by="test",
            idempotency_key="linear:lin-2",
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        conn.commit()
    finally:
        conn.close()

    client = SyncLinearClient(issue)
    review = tick(config=config, linear_client=client)

    conn = kanban_db.connect(board="default")
    try:
        assert kanban_db.claim_review_task(conn, task_id, claimer="reviewer")
        assert kanban_db.complete_task(conn, task_id, result="done")
    finally:
        conn.close()
    issue["state"] = {"name": "In Review"}
    done = tick(config=config, linear_client=client)

    assert client.requested_states == [
        ("Todo", "Backlog", "In Progress", "In Review"),
        ("Todo", "Backlog", "In Progress", "In Review"),
    ]
    assert review.synced == 1
    assert done.synced == 1
    assert client.transitions == [("lin-2", "In Review"), ("lin-2", "Done")]


def test_default_config_exposes_disabled_nubbi_linear_automation():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["nubbi"]["linear"]

    assert cfg["enabled"] is False
    assert cfg["project_name"] == "Nubbi Command Center"
    assert cfg["source_states"] == ["Todo", "Backlog"]
    assert cfg["active_states"] == ["In Progress", "In Review"]
    assert cfg["poll_interval_seconds"] >= 60


def test_tick_from_root_config_uses_nested_nubbi_linear_config():
    from hermes_cli.nubbi_linear_automation import NubbiTickResult, tick_from_root_config

    calls = []

    def fake_tick(*, config, linear_client=None):
        calls.append(config)
        return NubbiTickResult(created=2)

    result = tick_from_root_config(
        {"nubbi": {"linear": {"enabled": True, "project_name": "Nubbi Command Center"}}},
        tick_fn=fake_tick,
    )

    assert result.created == 2
    assert calls == [{"enabled": True, "project_name": "Nubbi Command Center"}]
