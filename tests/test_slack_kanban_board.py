from __future__ import annotations

from gateway.platforms.slack_kanban_board import (
    BoardFilters,
    CREATE_TASK_STATUSES,
    EDITABLE_DETAIL_STATUSES,
    MANUAL_MOVE_STATUSES,
    action_value,
    approve_task_and_continue,
    apply_task_action,
    build_board_blocks,
    build_board_help_text,
    build_board_text,
    create_task_for_status,
    dependency_options,
    move_action_value,
    move_task_status,
    parse_add_action_value,
    parse_action_value,
    parse_board_args,
    parse_board_command,
    parse_move_action_value,
    project_options,
    request_task_changes,
    task_approval_context,
    task_detail_blocks,
    task_detail_text,
    task_edit_values,
    update_task_fields,
)
from hermes_cli import kanban_db as kb


def test_parse_board_args_filters():
    filters = parse_board_args(
        "--board launch --status ready --assignee writer --tenant acme --limit 5 --archived"
    )

    assert filters.board == "launch"
    assert filters.status == "ready"
    assert filters.assignee == "writer"
    assert filters.tenant == "acme"
    assert filters.limit == 5
    assert filters.include_archived is True
    approval = parse_board_args("--status blocked --approval-required --page 2 --limit 10")
    assert approval.status == "blocked"
    assert approval.approval_only is True
    assert approval.page == 1
    assert approval.limit == 10
    assert parse_board_args("--limit 12").limit == 10


def test_parse_board_command_short_options_and_actions():
    command = parse_board_command("-p youtube -s ready -a -l 9 -t --summary")
    assert command.filters.tenant == "youtube"
    assert command.filters.status == "ready"
    assert command.filters.approval_only is True
    assert command.filters.limit == 9
    assert command.render == "text"
    assert command.text_detail == "summary"

    new_command = parse_board_command('-n "AI news scrape" -p youtube -s todo')
    assert new_command.action == "new"
    assert new_command.title == "AI news scrape"
    assert new_command.filters.tenant == "youtube"
    assert new_command.filters.status == "todo"

    edit_command = parse_board_command("-e t_abc123")
    assert edit_command.action == "edit"
    assert edit_command.task_id == "t_abc123"

    delete_command = parse_board_command("--delete t_abc123")
    assert delete_command.action == "delete"
    assert delete_command.task_id == "t_abc123"


def test_parse_board_command_natural_request():
    report = parse_board_command("youtube 프로젝트 ready 텍스트로 보여줘")
    assert report.render == "text"
    assert report.filters.tenant == "youtube"
    assert report.filters.status == "ready"

    new_task = parse_board_command("bright data 조사 추가")
    assert new_task.action == "new"
    assert "bright data" in (new_task.title or "")

    detail = parse_board_command("t_abc123 상세 보기")
    assert detail.action == "detail"
    assert detail.task_id == "t_abc123"

    help_command = parse_board_command("--help")
    assert help_command.action == "help"
    assert help_command.render == "text"
    assert "/board -t --summary" in build_board_help_text()


def test_action_value_round_trip():
    filters = BoardFilters(board="launch", status="blocked", tenant="acme")
    value = action_value("unblock", "t_abc123", filters)

    action, task_id, parsed = parse_action_value(value)

    assert action == "unblock"
    assert task_id == "t_abc123"
    assert parsed.board == "launch"
    assert parsed.status == "blocked"
    assert parsed.tenant == "acme"

    move_value = move_action_value("t_abc123", "done", filters)
    task_id, target_status, move_filters = parse_move_action_value(move_value)

    assert len(move_value) < 151
    assert task_id == "t_abc123"
    assert target_status == "done"
    assert move_filters.board == "launch"

    long_filters = BoardFilters(
        board="b" * 80,
        tenant="tenant" * 20,
        assignee="assignee" * 20,
        status="ready",
    )
    long_move_value = move_action_value("t_abc123", "done", long_filters)
    assert len(long_move_value) < 151


def test_build_board_blocks_and_apply_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        ready = kb.create_task(
            conn,
            title="Build Slack board",
            body="Render the Kanban cards with Block Kit.",
            assignee="manager",
            tenant="test",
            created_by="U123EXAMPLE",
        )
        kb.create_task(
            conn,
            title="Unassigned card",
            tenant="test",
        )
        blocked = kb.create_task(
            conn,
            title="Blocked card",
            assignee="manager",
            tenant="test",
        )
        assert kb.block_task(conn, blocked, reason="Need input")
        conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, status, started_at, ended_at, outcome, summary, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ready, "manager", "crashed", 1, 2, "crashed", "", ""),
        )

    fallback, blocks = build_board_blocks(BoardFilters(tenant="test", limit=3))
    text_report = build_board_text(BoardFilters(tenant="test", query="Slack", limit=3))

    assert "Hermes Kanban Board" in fallback
    assert "Build Slack board" in text_report
    assert "Unassigned card" not in text_report
    assert "project: test" in text_report
    assert len(blocks) <= 50
    assert sum(1 for block in blocks if block.get("type") == "divider") >= 2
    assert any(
        element.get("action_id") == "hermes_board_task_add"
        for block in blocks
        if block.get("type") == "actions"
        for element in block.get("elements", [])
    )
    assert not any(
        block.get("accessory", {}).get("action_id") == "hermes_board_task_add"
        for block in blocks
    )
    assert any(
        block.get("type") == "carousel"
        for block in blocks
    )
    assert any(
        block.get("type") == "section"
        and ">*Todo* `0`" in block.get("text", {}).get("text", "")
        for block in blocks
    )
    assert not any(
        block.get("type") == "alert"
        for block in blocks
    )
    assert not any(
        ":ballot_box_with_check:" in block.get("text", {}).get("text", "")
        for block in blocks
        if isinstance(block.get("text"), dict)
    )
    carousel_cards = [
        card
        for block in blocks
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    ]
    assert carousel_cards
    assert len(carousel_cards) <= 10
    assert all(card.get("type") == "card" for card in carousel_cards)
    assert any(
        "Assignee `manager`" in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert any(
        "Project `test`" in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert any(
        "Priority `Normal`" in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert any(
        "Assignee `Default profile`" in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert not any(
        " · " in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert not any(
        "1h" in card.get("body", {}).get("text", "")
        for card in carousel_cards
    )
    assert any(
        any(action.get("action_id") == "hermes_board_task_move_open" for action in card.get("actions", []))
        for card in carousel_cards
    )
    assert all(len(card.get("actions", [])) <= 2 for card in carousel_cards)
    assert not any(
        any(action.get("action_id") == "hermes_board_task_done" for action in card.get("actions", []))
        for card in carousel_cards
    )
    assert not any(
        any(action.get("action_id") == "hermes_board_task_delete" for action in card.get("actions", []))
        for card in carousel_cards
    )
    assert "Marked" in apply_task_action("done", ready, BoardFilters(tenant="test"))
    assert "Unblocked" in apply_task_action("unblock", blocked, BoardFilters(tenant="test"))
    assert "Deleted" in apply_task_action("delete", ready, BoardFilters(tenant="test"))
    assert "Already deleted" in apply_task_action("delete", ready, BoardFilters(tenant="test"))
    detail = task_detail_text(ready, BoardFilters(tenant="test"))
    assert "Build Slack board" in detail
    assert "project: `test`" in detail
    detail_blocks = task_detail_blocks(ready, BoardFilters(tenant="test"))
    assert detail_blocks[0]["type"] == "alert"
    assert detail_blocks[0]["level"] in {"default", "info"}
    assert any(
        block.get("type") == "card"
        and "Build Slack board" in block.get("title", {}).get("text", "")
        for block in detail_blocks
    )
    summary_card = next(block for block in detail_blocks if block.get("type") == "card")
    assert "Project `test`" in summary_card.get("subtitle", {}).get("text", "")
    assert "Priority `Normal`" in summary_card.get("subtitle", {}).get("text", "")
    assert "Render the Kanban cards" in summary_card.get("body", {}).get("text", "")
    assert "Assignee" not in summary_card.get("body", {}).get("text", "")
    assert "Created by" not in summary_card.get("body", {}).get("text", "")
    assert any(block.get("fields") for block in detail_blocks)
    assert any(
        "Project" in field.get("text", "")
        for block in detail_blocks
        for field in block.get("fields", [])
    )
    assert any(
        "Normal" in field.get("text", "")
        for block in detail_blocks
        for field in block.get("fields", [])
    )
    assert any(
        "<@U123EXAMPLE>" in field.get("text", "")
        for block in detail_blocks
        for field in block.get("fields", [])
    )
    assert any(
        block.get("type") == "alert"
        and "Description" in block.get("text", {}).get("text", "")
        for block in detail_blocks
    )
    assert not any(
        "Completed from Slack /board" in block.get("text", {}).get("text", "")
        for block in detail_blocks
        if isinstance(block.get("text"), dict)
    )
    assert not any(
        "Recent Runs" in block.get("text", {}).get("text", "")
        for block in detail_blocks
        if isinstance(block.get("text"), dict)
    )
    status, parsed_filters = parse_add_action_value(None)
    assert status == "todo"
    assert parsed_filters.board is None


def test_create_task_for_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))

    task_id, filters = create_task_for_status(
        status="todo",
        title="Draft new task",
        body="Created from Slack modal.",
        assignee="Manager",
        tenant="acme",
        priority=2,
        filters=BoardFilters(tenant="acme"),
        created_by="test",
    )

    assert filters.tenant == "acme"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "todo"
    assert task.assignee == "manager"
    assert task.tenant == "acme"
    assert task.priority == 2


def test_update_task_fields_from_detail_modal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Original",
            body="Old body",
            assignee="manager",
            tenant="acme",
        )
        assert kb.block_task(conn, task_id, reason="Needs edit")

    values = task_edit_values(task_id, BoardFilters(tenant="acme"))
    assert values is not None
    assert values["title"] == "Original"
    assert values["status"] == "blocked"
    assert "blocked" in EDITABLE_DETAIL_STATUSES

    notice, filters = update_task_fields(
        task_id,
        title="Updated title",
        body="Updated description",
        assignee="Default",
        tenant="ops",
        filters=BoardFilters(tenant="acme"),
    )

    assert "Updated" in notice
    assert filters.tenant == "ops"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "Updated title"
        assert task.body == "Updated description"
        assert task.assignee == "default"
        assert task.tenant == "ops"
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert events is not None
    assert events["kind"] == "updated"
    assert "slack_board" in events["payload"]


def test_update_task_fields_rejects_done_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Complete me")
        assert kb.complete_task(conn, task_id, result="done")

    notice, _filters = update_task_fields(
        task_id,
        title="Should not change",
        filters=BoardFilters(),
    )

    assert "Cannot edit" in notice
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.title == "Complete me"


def test_detail_blocks_include_debugging_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Debug task",
            body="Needs investigation",
            assignee="manager",
            tenant="ops",
        )
        with kb.write_txn(conn):
            conn.execute(
                """
                INSERT INTO task_runs (
                    task_id, profile, status, started_at, ended_at, outcome, summary, metadata, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    "manager",
                    "failed",
                    10,
                    70,
                    "failed",
                    "Collected partial output",
                    '{"step":"fetch"}',
                    "Traceback: provider auth failed",
                ),
            )
            kb._append_event(conn, task_id, "worker_failed", {"error": "provider auth failed"})
    log_path = kb.worker_log_path(task_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stdout line\nstderr provider auth failed\n", encoding="utf-8")

    blocks = task_detail_blocks(task_id, BoardFilters(tenant="ops"))
    rendered = "\n".join(
        block.get("text", {}).get("text", "")
        for block in blocks
        if isinstance(block.get("text"), dict)
    )

    assert "Run History" in rendered
    assert "Collected partial output" in rendered
    assert "Traceback: provider auth failed" in rendered
    assert "Recent Events" in rendered
    assert "worker_failed" in rendered
    assert "Worker Log Tail" in rendered
    assert "stderr provider auth failed" in rendered


def test_approval_context_actions_and_detail_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Send approval email",
            body="Draft an email and wait for approval before sending.",
            assignee="default",
            tenant="ops",
        )
        assert kb.block_task(conn, task_id, reason="이메일 발송 전 승인 필요")
        kb.add_comment(conn, task_id, "default", "수신자: user@example.com\n제목: Draft\n본문: Hello")

    context = task_approval_context(task_id, BoardFilters(tenant="ops"))
    assert context is not None
    assert "승인" in context["reason"]
    assert "수신자" in context["draft"]

    fallback, board_blocks = build_board_blocks(BoardFilters(tenant="ops"))
    assert "Hermes Kanban Board" in fallback
    board_text = "\n".join(
        card.get("body", {}).get("text", "")
        for block in board_blocks
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    )
    assert "Approval `Required`" in board_text

    detail_blocks = task_detail_blocks(task_id, BoardFilters(tenant="ops"))
    rendered = "\n".join(
        block.get("text", {}).get("text", "")
        for block in detail_blocks
        if isinstance(block.get("text"), dict)
    )
    assert "Approval Required" in rendered
    assert "Draft / Pending Output" in rendered
    assert "user@example.com" in rendered

    change_notice, filters = request_task_changes(
        task_id,
        BoardFilters(tenant="ops"),
        requested_by="U12345678",
        feedback="제목을 더 짧게 바꿔주세요.",
    )
    assert "Requested changes" in change_notice
    assert filters.tenant == "ops"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert task is not None
    assert task.status == "blocked"
    assert "제목을 더 짧게" in comments[-1].body
    assert events[-1].kind == "approval_changes_requested"

    approve_notice, filters = approve_task_and_continue(
        task_id,
        BoardFilters(tenant="ops", status="blocked"),
        approved_by="U12345678",
    )
    assert "Approved" in approve_notice
    assert filters.status == "ready"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert task is not None
    assert task.status == "ready"
    assert any(event.kind == "approved" for event in events)
    assert any(event.kind == "unblocked" for event in events)


def test_status_pagination_and_approval_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        for idx in range(12):
            kb.create_task(conn, title=f"Ready task {idx}", tenant="ops")
        approval_id = kb.create_task(
            conn,
            title="Approval task",
            body="Send only after approval.",
            tenant="ops",
        )
        assert kb.block_task(conn, approval_id, reason="승인 필요")
        kb.add_comment(conn, approval_id, "default", "초안입니다.")

    fallback, blocks = build_board_blocks(BoardFilters(tenant="ops"))
    assert "Hermes Kanban Board" in fallback
    cards = [
        card
        for block in blocks
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    ]
    assert len(cards) <= 5 * 6
    assert any(
        element.get("action_id") == "hermes_board_filter_approval"
        for block in blocks
        if block.get("type") == "actions"
        for element in block.get("elements", [])
    )
    assert all(
        len(option.get("value", "")) < 151
        for block in blocks
        if block.get("type") == "actions"
        for element in block.get("elements", [])
        for option in element.get("options", [])
    )

    fallback, ready_page_1 = build_board_blocks(BoardFilters(status="ready", tenant="ops", limit=10))
    ready_cards_1 = [
        card
        for block in ready_page_1
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    ]
    assert len(ready_cards_1) == 10
    assert any(
        element.get("action_id") == "hermes_board_page"
        and element.get("text", {}).get("text") == "Next"
        for block in ready_page_1
        if block.get("type") == "actions"
        for element in block.get("elements", [])
    )

    _fallback, ready_page_2 = build_board_blocks(BoardFilters(status="ready", tenant="ops", limit=10, page=1))
    ready_cards_2 = [
        card
        for block in ready_page_2
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    ]
    assert len(ready_cards_2) == 2
    assert any(
        element.get("action_id") == "hermes_board_page"
        and element.get("text", {}).get("text") == "Previous"
        for block in ready_page_2
        if block.get("type") == "actions"
        for element in block.get("elements", [])
    )

    _fallback, approval_blocks = build_board_blocks(BoardFilters(tenant="ops", approval_only=True))
    approval_text = "\n".join(
        card.get("title", {}).get("text", "") + "\n" + card.get("body", {}).get("text", "")
        for block in approval_blocks
        if block.get("type") == "carousel"
        for card in block.get("elements", [])
    )
    assert "Approval task" in approval_text
    assert "Ready task" not in approval_text
    assert "Approval `Required`" in approval_text


def test_project_options(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        kb.create_task(conn, title="Content task", tenant="acme")
        kb.create_task(conn, title="Finance task", tenant="finance")

    options = project_options(BoardFilters(tenant="acme"))
    values = [option.get("value") for option in options]

    assert values[0] == "__none__"
    assert "acme" in values
    assert "finance" in values
    assert all(len(value) <= 150 for value in values)


def test_create_task_with_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="Parent research", tenant="acme")

    child, filters = create_task_for_status(
        status="ready",
        title="Child writeup",
        tenant="acme",
        parents=[parent],
        filters=BoardFilters(tenant="acme"),
        created_by="test",
    )

    assert filters.tenant == "acme"
    with kb.connect() as conn:
        task = kb.get_task(conn, child)
        assert task is not None
        assert task.status == "todo"
        assert kb.parent_ids(conn, child) == [parent]

    options = dependency_options(BoardFilters(tenant="acme"))
    assert any(option.get("value") == parent for option in options)
    assert all(len(option.get("value", "")) < 151 for option in options)


def test_create_task_status_options_exclude_running_and_blocked():
    assert "running" not in CREATE_TASK_STATUSES
    assert "blocked" not in CREATE_TASK_STATUSES
    assert "done" not in CREATE_TASK_STATUSES
    assert CREATE_TASK_STATUSES == ("triage", "todo", "ready")


def test_move_task_status(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Move me",
            assignee="manager",
            tenant="test",
        )

    notice = move_task_status(task_id, "blocked", BoardFilters(tenant="test"))

    assert "Moved" in notice
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "blocked"
    assert "already" in move_task_status(task_id, "blocked", BoardFilters(tenant="test"))
    assert "running" not in MANUAL_MOVE_STATUSES

    notice = move_task_status(task_id, "running", BoardFilters(tenant="test"))

    assert "Queued" in notice
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "ready"


def test_manual_move_to_todo_survives_board_render(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Stay in todo",
            tenant="test",
        )

    assert "Moved" in move_task_status(task_id, "todo", BoardFilters(tenant="test"))

    fallback, blocks = build_board_blocks(BoardFilters(tenant="test"))

    assert "Hermes Kanban Board" in fallback
    rendered = "\n".join(
        block.get("text", {}).get("text", "")
        for block in blocks
        if isinstance(block.get("text"), dict)
    )
    assert "*Todo* `1`" in rendered
    assert "*Ready* `0`" in rendered
