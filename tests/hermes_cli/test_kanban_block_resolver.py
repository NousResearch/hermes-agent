from hermes_cli import kanban_block_resolver as resolver
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return kbc.connect()


def test_resolver_defers_to_creator_wake_and_human_gates(tmp_path, monkeypatch):
    conn = _board(tmp_path, monkeypatch)
    try:
        creator_task = kb.create_task(conn, title="creator-owned", assignee="worker")
        kbn.add_notify_sub(
            conn, task_id=creator_task, platform="telegram", chat_id="creator-chat",
            delivery_mode="notify+wake",
        )
        assert kb.block_task(conn, creator_task, reason="missing context", kind="transient")

        human_task = kb.create_task(conn, title="approval", assignee="worker")
        assert kb.block_task(conn, human_task, reason="approve production deploy", kind="needs_input")

        monkeypatch.setattr(
            resolver, "_call_aux",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resolver must not run")),
        )
        outcome = resolver.resolve_one(conn)

        assert not outcome.attempted
        assert kb.get_task(conn, creator_task).status == "blocked"
        assert kb.get_task(conn, human_task).status == "blocked"
    finally:
        conn.close()


def test_resolver_adds_context_retries_once_and_preserves_attempt_budget(tmp_path, monkeypatch):
    conn = _board(tmp_path, monkeypatch)
    calls = []
    try:
        task_id = kb.create_task(conn, title="flaky fetch", body="Fetch the report", assignee="worker")
        assert kb.block_task(conn, task_id, reason="temporary upstream timeout", kind="transient")

        def fake_call(*args, **kwargs):
            calls.append(kwargs["user"])
            return ('{"action":"retry","context":"Retry now; the timeout is transient.",'
                    '"rationale":"A bounded retry is safe."}', "")

        monkeypatch.setattr(resolver, "_call_aux", fake_call)
        outcome = resolver.resolve_one(conn, max_attempts=1)

        assert outcome.action == "retry"
        assert kb.get_task(conn, task_id).status == "ready"
        assert "temporary upstream timeout" in calls[0]
        comments = kb.list_comments(conn, task_id)
        assert comments[-1].author == "blocked-resolver"
        assert "Retry now" in comments[-1].body

        # A different block kind avoids the DB's same-cause recurrence breaker;
        # the resolver's own task-level budget must still prevent a second call.
        assert kb.block_task(conn, task_id, reason="still unclear", kind=None)
        second = resolver.resolve_one(conn, max_attempts=1)
        assert not second.attempted
        assert len(calls) == 1
        assert kb.get_task(conn, task_id).status == "blocked"
    finally:
        conn.close()
