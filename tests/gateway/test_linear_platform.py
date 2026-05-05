"""Tests for the Linear gateway platform adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType, ProcessingOutcome


@pytest.fixture(autouse=True)
def _isolate_kanban_env(monkeypatch):
    """Keep Linear/Kanban adapter tests off the caller's live board.

    Kanban workers run with HERMES_KANBAN_BOARD/HERMES_KANBAN_DB pinned to
    the task board. These tests set their own temporary HERMES_KANBAN_HOME,
    so inherited board/db pins must not leak through kanban_db's resolution
    chain and make assertions read the live board.
    """
    for name in (
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_linear_platform_env_config(monkeypatch):
    monkeypatch.setenv("LINEAR_ENABLED", "true")
    monkeypatch.setenv("SYMPHONY_LINEAR_API_KEY", "lin_test")
    monkeypatch.setenv("LINEAR_TEAM_KEY", "DEC")
    monkeypatch.setenv("LINEAR_PROJECT_SLUG", "hermes-linear-interface")
    monkeypatch.setenv("LINEAR_START_STATES", "Todo")
    monkeypatch.setenv("LINEAR_WAITING_STATES", "Blocked")
    monkeypatch.setenv("LINEAR_WORKING_STATE", "Working")
    monkeypatch.setenv("LINEAR_BLOCKED_STATE", "Blocked")
    monkeypatch.setenv("LINEAR_TERMINAL_STATES", "Done, Canceled, Duplicate")
    monkeypatch.setenv("LINEAR_INBOX_STATES", "Inbox, Chat")
    monkeypatch.setenv("LINEAR_REVIEW_STATE", "Ready for Review")
    monkeypatch.setenv("LINEAR_POLL_INTERVAL_SECONDS", "7")

    config = load_gateway_config()

    assert Platform.LINEAR in config.platforms
    linear = config.platforms[Platform.LINEAR]
    assert linear.enabled is True
    assert linear.api_key == "lin_test"
    assert linear.extra["team_key"] == "DEC"
    assert linear.extra["project_slug"] == "hermes-linear-interface"
    assert linear.extra["start_states"] == ["Todo"]
    assert linear.extra["waiting_states"] == ["Blocked"]
    assert linear.extra["working_state"] == "Working"
    assert linear.extra["blocked_state"] == "Blocked"
    assert linear.extra["terminal_states"] == ["Done", "Canceled", "Duplicate"]
    assert linear.extra["inbox_states"] == ["Inbox", "Chat"]
    assert linear.extra["review_state"] == "Ready for Review"
    assert linear.extra["poll_interval_seconds"] == 7


def test_linear_platform_prefers_symphony_token_over_generic_linear_key(monkeypatch):
    monkeypatch.setenv("LINEAR_ENABLED", "true")
    monkeypatch.setenv("LINEAR_API_KEY", "lin_ab99")
    monkeypatch.setenv("SYMPHONY_LINEAR_API_KEY", "lin_december")

    config = load_gateway_config()

    assert config.platforms[Platform.LINEAR].api_key == "lin_december"



def test_linear_platform_prefers_symphony_token_over_hermes_linear_key(monkeypatch):
    monkeypatch.setenv("LINEAR_ENABLED", "true")
    monkeypatch.setenv("HERMES_LINEAR_API_KEY", "lin_generic_hermes")
    monkeypatch.setenv("SYMPHONY_LINEAR_API_KEY", "lin_december")

    config = load_gateway_config()

    assert config.platforms[Platform.LINEAR].api_key == "lin_december"



def test_linear_adapter_prefers_symphony_token_over_hermes_linear_key(monkeypatch):
    monkeypatch.setenv("HERMES_LINEAR_API_KEY", "lin_generic_hermes")
    monkeypatch.setenv("SYMPHONY_LINEAR_API_KEY", "lin_december")

    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig())

    assert adapter.api_key == "lin_december"



def test_linear_platform_does_not_use_generic_linear_api_key(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_ab99")
    monkeypatch.delenv("LINEAR_ENABLED", raising=False)
    monkeypatch.delenv("SYMPHONY_LINEAR_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LINEAR_API_KEY", raising=False)

    config = load_gateway_config()

    assert Platform.LINEAR not in config.platforms



def test_linear_adapter_does_not_fall_back_to_generic_linear_api_key(monkeypatch):
    monkeypatch.setenv("LINEAR_API_KEY", "lin_ab99")
    monkeypatch.delenv("SYMPHONY_LINEAR_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_LINEAR_API_KEY", raising=False)

    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig())

    assert adapter.api_key == ""



def test_linear_events_do_not_expose_configured_api_key():
    from gateway.platforms.linear import LinearAdapter

    secret = "lin_secret_should_not_reach_worker_prompt"
    adapter = LinearAdapter(PlatformConfig(api_key=secret))
    issue = {
        "id": "issue-id",
        "identifier": "DEC-7",
        "title": "Implement Linear adapter",
        "description": "Do the work without leaking configured credentials.",
        "state": {"name": "In Progress"},
        "creator": {"id": "user-id", "name": "Anton", "email": "anton@example.com"},
    }
    comment = {
        "id": "comment-id",
        "body": "Comment body",
        "user": {"id": "user-id", "name": "Anton", "email": "anton@example.com"},
    }

    issue_event = adapter._event_from_issue(issue)
    comment_event = adapter._event_from_comment(issue, comment)

    assert secret not in issue_event.text
    assert secret not in comment_event.text
    assert secret not in str(issue_event.source.to_dict())
    assert secret not in str(comment_event.source.to_dict())



def test_linear_adapter_turns_comment_into_message_event_with_issue_context():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    issue = {
        "id": "issue-id",
        "identifier": "DEC-6",
        "title": "Define Hermes Linear gateway contract",
        "url": "https://linear.app/december/issue/DEC-6/example",
        "state": {"name": "In Progress"},
        "team": {"key": "DEC", "name": "December"},
        "project": {"id": "project-id", "name": "Hermes Linear Interface", "slugId": "bf612e7cb9be"},
        "labels": {"nodes": [{"name": "hermes"}, {"name": "gateway"}]},
    }
    comment = {
        "id": "comment-id",
        "body": "Use this issue as the gateway thread.",
        "createdAt": "2026-04-28T15:00:00.000Z",
        "user": {"id": "user-id", "name": "Anton", "email": "anton@example.com"},
    }

    event = adapter._event_from_comment(issue, comment)

    assert event.message_id == "comment-id"
    assert event.message_type is MessageType.TEXT
    assert "Linear issue DEC-6: Define Hermes Linear gateway contract" in event.text
    assert "Team: DEC" in event.text
    assert "Project: Hermes Linear Interface (bf612e7cb9be)" in event.text
    assert "Labels: hermes, gateway" in event.text
    assert "Comment from Anton:" in event.text
    assert "Use this issue as the gateway thread." in event.text
    assert event.source.platform is Platform.LINEAR
    assert event.source.chat_id == "issue-id"
    assert event.source.thread_id == "DEC-6"
    assert event.source.chat_name == "DEC-6: Define Hermes Linear gateway contract"
    assert event.source.user_id == "user-id"
    assert event.source.user_name == "Anton"
    assert event.raw_message["issue"]["identifier"] == "DEC-6"


def test_linear_adapter_turns_issue_into_initial_message_event():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    issue = {
        "id": "issue-id",
        "identifier": "DEC-7",
        "title": "Implement Linear adapter for Hermes sessions",
        "description": "Build native Hermes platform adapter.",
        "url": "https://linear.app/december/issue/DEC-7/example",
        "state": {"name": "Todo"},
        "labels": {"nodes": [{"name": "hermes"}, {"name": "backend"}]},
        "creator": {"id": "user-id", "name": "Anton", "email": "anton@example.com"},
    }

    event = adapter._event_from_issue(issue)

    assert event.message_id == "issue:issue-id"
    assert event.source.chat_id == "issue-id"
    assert event.source.thread_id == "DEC-7"
    assert "Linear issue DEC-7" in event.text
    assert "Implement Linear adapter for Hermes sessions" in event.text
    assert "Build native Hermes platform adapter." in event.text
    assert "Labels: hermes, backend" in event.text


@pytest.mark.asyncio
async def test_linear_client_omits_project_id_variable_when_not_filtering_by_id():
    from gateway.platforms.linear import LinearGraphQLClient

    class CaptureClient(LinearGraphQLClient):
        def __init__(self):
            super().__init__("lin_test")
            self.query = ""
            self.variables = {}

        async def graphql(self, query, variables=None):
            self.query = query
            self.variables = variables or {}
            return {"issues": {"nodes": []}}

    client = CaptureClient()

    await client.fetch_candidate_issues(
        team_key="DEC",
        project_id="",
        project_name="Hermes Linear Interface",
        state_names=["In Progress"],
    )

    assert "$projectId" not in client.query
    assert "projectId" not in client.variables


@pytest.mark.asyncio
async def test_linear_send_posts_comment_to_issue():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}

    result = await adapter.send("issue-id", "Hermes response")

    assert result.success is True
    assert result.message_id == "created-comment"
    adapter._client.comment_create.assert_awaited_once_with("issue-id", "Hermes response")


@pytest.mark.asyncio
async def test_linear_send_replies_under_triggering_comment_when_reply_to_is_set():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-reply"}

    result = await adapter.send("issue-id", "Hermes response", reply_to="comment-id")

    assert result.success is True
    assert result.message_id == "created-reply"
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "Hermes response",
        parent_id="comment-id",
    )


@pytest.mark.asyncio
async def test_linear_send_does_not_use_synthetic_issue_message_id_as_parent():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}

    result = await adapter.send("issue-id", "Hermes response", reply_to="issue:issue-id")

    assert result.success is True
    adapter._client.comment_create.assert_awaited_once_with("issue-id", "Hermes response")


@pytest.mark.asyncio
async def test_linear_client_comment_create_can_set_parent_comment():
    from gateway.platforms.linear import LinearGraphQLClient

    class CaptureClient(LinearGraphQLClient):
        def __init__(self):
            super().__init__("lin_test")
            self.query = ""
            self.variables = {}

        async def graphql(self, query, variables=None):
            self.query = query
            self.variables = variables or {}
            return {"commentCreate": {"success": True, "comment": {"id": "reply-id"}}}

    client = CaptureClient()

    comment = await client.comment_create(
        "issue-id",
        "Hermes response",
        parent_id="comment-id",
    )

    assert comment == {"id": "reply-id"}
    assert "$parentId: String" in client.query
    assert "parentId: $parentId" in client.query
    assert client.variables == {
        "issueId": "issue-id",
        "body": "Hermes response",
        "parentId": "comment-id",
    }


@pytest.mark.asyncio
async def test_linear_poll_ignores_backlog_and_starts_issue_body_from_todo_only():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [
        {
            "id": "backlog-issue",
            "identifier": "DEC-20",
            "title": "Dormant backlog task",
            "state": {"name": "Backlog"},
            "comments": {"nodes": [
                {"id": "backlog-comment", "body": "do this", "user": {"name": "Anton"}}
            ]},
        },
        {
            "id": "todo-issue",
            "identifier": "DEC-21",
            "title": "Ready task",
            "description": "Start when Todo.",
            "state": {"name": "Todo"},
            "comments": {"nodes": []},
        },
    ]
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    adapter._client.fetch_candidate_issues.assert_awaited_once()
    _, kwargs = adapter._client.fetch_candidate_issues.await_args
    assert kwargs["state_names"] == ["Todo", "Blocked", "Ready for Review", "Working"]
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].source.thread_id == "DEC-21"


@pytest.mark.asyncio
async def test_linear_processing_start_moves_todo_to_working():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-22",
        "title": "Executable task",
        "state": {"name": "Todo"},
        "team": {"key": "DEC", "name": "December"},
    }
    event = adapter._event_from_issue(issue)

    await adapter.on_processing_start(event)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Working",
        team_key="DEC",
        team_id="",
    )


@pytest.mark.asyncio
async def test_linear_processing_start_moves_blocked_reply_to_working():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-23",
        "title": "Blocked task",
        "state": {"name": "Blocked"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    comment = {
        "id": "comment-id",
        "body": "Here is the answer.",
        "user": {"id": "user-id", "name": "Anton"},
    }
    event = adapter._event_from_comment(issue, comment)

    await adapter.on_processing_start(event)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Working",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_processing_start_moves_review_comment_to_working():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-24",
        "title": "Review follow-up",
        "state": {"name": "Ready for Review"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    comment = {
        "id": "comment-id",
        "body": "Change this part and try again.",
        "user": {"id": "user-id", "name": "Anton"},
    }
    event = adapter._event_from_comment(issue, comment)

    await adapter.on_processing_start(event)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Working",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_poll_processes_ready_for_review_comments_without_restarting_issue_body():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [
        {
            "id": "review-issue",
            "identifier": "DEC-25",
            "title": "Completed task with follow-up",
            "description": "Original task body.",
            "state": {"name": "Ready for Review"},
            "comments": {"nodes": [
                {
                    "id": "review-comment",
                    "body": "Tighten the copy and rerun tests.",
                    "user": {"id": "user-id", "name": "Anton"},
                }
            ]},
        }
    ]
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    _, kwargs = adapter._client.fetch_candidate_issues.await_args
    assert kwargs["state_names"] == ["Todo", "Blocked", "Ready for Review", "Working"]
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_id == "review-comment"
    assert event.source.thread_id == "DEC-25"
    assert "Comment from Anton:" in event.text
    assert "Tighten the copy and rerun tests." in event.text


@pytest.mark.asyncio
async def test_linear_send_without_state_directive_does_not_move_issue_to_review():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}

    result = await adapter.send(
        "issue-id",
        "Here is the actual Hermes response.",
        metadata={"team_key": "DEC"},
    )

    assert result.success is True
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "Here is the actual Hermes response.",
    )
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_send_drops_gateway_system_notice_comments():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()

    result = await adapter.send(
        "issue-id",
        "⏳ Still working... (9 min elapsed — iteration 20/100, running: browser_navigate)",
        metadata={"team_key": "DEC", "hermes_system_notice": True},
    )

    assert result.success is True
    assert result.message_id is None
    adapter._client.comment_create.assert_not_awaited()
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_send_keeps_actual_agent_interim_messages():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "agent-interim"}

    result = await adapter.send(
        "issue-id",
        "I found the failing path. Patching it now.",
        metadata={"team_key": "DEC"},
    )

    assert result.success is True
    assert result.message_id == "agent-interim"
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "I found the failing path. Patching it now.",
    )
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_processing_complete_moves_successful_issue_to_ready_for_review():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-26",
        "title": "Complete task",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    event = adapter._event_from_issue(issue)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Ready for Review",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_processing_complete_moves_cancelled_issue_to_blocked():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-27",
        "title": "Interrupted task",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    event = adapter._event_from_issue(issue)

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Blocked",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_processing_complete_moves_failed_issue_to_blocked():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-28",
        "title": "Failed task",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    event = adapter._event_from_issue(issue)

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Blocked",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_processing_complete_does_not_override_blocked_directive():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}
    issue = {
        "id": "issue-id",
        "identifier": "DEC-29",
        "title": "Needs Anton",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
    }
    event = adapter._event_from_issue(issue)

    await adapter.send(
        "issue-id",
        "I need your decision before proceeding.\n\n<!-- linear-state: blocked -->",
        metadata={"team_key": "DEC", "team_id": "team-id"},
    )
    adapter._client.set_issue_state_by_name.reset_mock()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_reconcile_orphaned_working_issue_moves_resume_pending_to_blocked():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-30",
        "title": "Orphaned task",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
        "comments": {"nodes": []},
    }
    adapter._client.fetch_candidate_issues.return_value = [issue]
    session_key = adapter._session_key_for_issue(issue)
    adapter.set_session_store(SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(resume_pending=True, suspended=False)},
    ))

    await adapter._reconcile_orphaned_working_issues()

    adapter._client.fetch_candidate_issues.assert_awaited_once_with(
        team_key="",
        project_id="",
        project_name="",
        state_names=["Working"],
    )
    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Blocked",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_reconcile_leaves_non_orphaned_working_issue_alone():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-31",
        "title": "Live task",
        "state": {"name": "Working"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
        "comments": {"nodes": []},
    }
    adapter._client.fetch_candidate_issues.return_value = [issue]
    session_key = adapter._session_key_for_issue(issue)
    adapter.set_session_store(SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={session_key: SimpleNamespace(resume_pending=False, suspended=False)},
    ))

    await adapter._reconcile_orphaned_working_issues()

    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_send_moves_issue_to_blocked_and_strips_marker():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}

    result = await adapter.send(
        "issue-id",
        "I need your decision before proceeding.\n\n<!-- linear-state: blocked -->",
        metadata={"team_key": "DEC", "team_id": "team-id"},
    )

    assert result.success is True
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "I need your decision before proceeding.",
    )
    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Blocked",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_client_uses_linear_schema_string_types_for_issue_state_update():
    from gateway.platforms.linear import LinearGraphQLClient

    class CaptureClient(LinearGraphQLClient):
        def __init__(self):
            super().__init__("lin_test")
            self.query = ""
            self.variables = {}

        async def graphql(self, query, variables=None):
            self.query = query
            self.variables = variables or {}
            return {"issueUpdate": {"success": True, "issue": {"id": "issue-id"}}}

    client = CaptureClient()

    await client.issue_update_state("issue-id", "state-id")

    assert "$issueId: String!" in client.query
    assert "$stateId: String" in client.query
    assert "$issueId: ID!" not in client.query
    assert "$stateId: ID!" not in client.query
    assert client.variables == {"issueId": "issue-id", "stateId": "state-id"}


@pytest.mark.asyncio
async def test_linear_kanban_mode_creates_task_for_todo_issue_and_ignores_assignee(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={
        "execution_mode": "kanban",
        "kanban_default_assignee": "default",
    }))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [{
        "id": "issue-id",
        "identifier": "DEC-44",
        "title": "Build the bridge",
        "description": "Create real Kanban-backed Linear execution.",
        "url": "https://linear.app/december/issue/DEC-44/build-the-bridge",
        "state": {"name": "Todo"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
        "assignee": {"id": "anton-id", "name": "Anton"},
        "comments": {"nodes": []},
    }]
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    adapter.handle_message.assert_not_awaited()
    conn = kb.connect()
    tasks = kb.list_tasks(conn, include_archived=True)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "DEC-44: Build the bridge"
    assert task.assignee == "default"
    assert task.status == "ready"
    assert "Linear issue DEC-44" in (task.body or "")
    subs = kb.list_notify_subs(conn, task.id)
    assert subs and subs[0]["platform"] == "linear"
    assert subs[0]["chat_id"] == "issue-id"
    assert subs[0]["thread_id"] == "DEC-44"
    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Working",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_kanban_comment_on_blocked_task_adds_comment_and_unblocks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={"execution_mode": "kanban"}))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-45",
        "title": "Needs answer",
        "description": "Blocked until Anton replies.",
        "state": {"name": "Blocked"},
        "team": {"id": "team-id", "key": "DEC"},
        "comments": {"nodes": []},
    }
    task_id = adapter._ensure_kanban_task_for_issue(issue)
    conn = kb.connect()
    assert kb.block_task(conn, task_id, reason="need Anton") is True
    comment = {
        "id": "comment-id",
        "body": "Use option B and continue.",
        "user": {"id": "anton-id", "name": "Anton", "email": "anton@example.com"},
    }

    await adapter._handle_kanban_comment(issue, comment)

    task = kb.get_task(conn, task_id)
    assert task.status == "ready"
    comments = kb.list_comments(conn, task_id)
    assert comments[-1].author == "Anton"
    assert "Use option B and continue." in comments[-1].body
    assert "Linear comment: comment-id" in comments[-1].body


@pytest.mark.asyncio
async def test_linear_kanban_comment_after_done_creates_followup_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={
        "execution_mode": "kanban",
        "kanban_default_assignee": "default",
    }))
    adapter._client = AsyncMock()
    issue = {
        "id": "issue-id",
        "identifier": "DEC-46",
        "title": "Review loop",
        "description": "Initial task.",
        "state": {"name": "Ready for Review"},
        "team": {"id": "team-id", "key": "DEC"},
        "comments": {"nodes": []},
    }
    task_id = adapter._ensure_kanban_task_for_issue(issue)
    conn = kb.connect()
    assert kb.complete_task(conn, task_id, summary="first pass done") is True
    comment = {
        "id": "review-comment",
        "body": "Tighten the copy and rerun QA.",
        "user": {"id": "anton-id", "name": "Anton", "email": "anton@example.com"},
    }

    followup_id = await adapter._handle_kanban_comment(issue, comment)

    tasks = kb.list_tasks(conn, include_archived=True)
    assert len(tasks) == 2
    followup = kb.get_task(conn, followup_id)
    assert followup.title == "DEC-46 follow-up: Tighten the copy and rerun QA."
    assert followup.assignee == "default"
    assert followup.status == "ready"
    assert adapter._kanban_task_by_issue_id["issue-id"] == followup_id
    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Working",
        team_key="DEC",
        team_id="team-id",
    )


@pytest.mark.asyncio
async def test_linear_status_issue_routes_comments_to_persistent_chat_not_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={
        "execution_mode": "kanban",
        "status_issue_identifier": "DEC-STATUS",
    }))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [{
        "id": "status-issue-id",
        "identifier": "DEC-STATUS",
        "title": "Hermes status",
        "description": "Persistent status/chat issue.",
        "state": {"name": "Working"},
        "comments": {"nodes": [{
            "id": "status-comment-id",
            "body": "What is blocked right now?",
            "user": {"id": "anton-id", "name": "Anton", "email": "anton@example.com"},
        }]},
    }]
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.thread_id == "DEC-STATUS"
    assert "Persistent Linear status/chat issue" in event.text
    assert "What is blocked right now?" in event.text
    assert kb.list_tasks(kb.connect(), include_archived=True) == []
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_inbox_issue_routes_to_direct_chat_not_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={
        "execution_mode": "kanban",
        "inbox_states": ["Inbox"],
    }))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [{
        "id": "inbox-issue-id",
        "identifier": "DEC-47",
        "title": "Quick question",
        "description": "Can I treat this issue as a chat?",
        "state": {"name": "Inbox"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
        "comments": {"nodes": []},
    }]
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.thread_id == "DEC-47"
    assert "Linear inbox/chat issue" in event.text
    assert "Can I treat this issue as a chat?" in event.text
    assert kb.list_tasks(kb.connect(), include_archived=True) == []
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_inbox_comment_routes_to_direct_chat_not_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from gateway.platforms.linear import LinearAdapter
    from hermes_cli import kanban_db as kb

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test", extra={
        "execution_mode": "kanban",
        "inbox_states": ["Inbox"],
    }))
    adapter._client = AsyncMock()
    adapter._client.fetch_candidate_issues.return_value = [{
        "id": "inbox-issue-id",
        "identifier": "DEC-48",
        "title": "Quick thread",
        "description": "",
        "state": {"name": "Inbox"},
        "team": {"id": "team-id", "key": "DEC", "name": "December"},
        "comments": {"nodes": [{
            "id": "inbox-comment-id",
            "body": "Answer this without creating work.",
            "user": {"id": "anton-id", "name": "Anton", "email": "anton@example.com"},
        }]},
    }]
    adapter._seen_issue_ids.add("inbox-issue-id")
    adapter.handle_message = AsyncMock()

    await adapter._poll_once()

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.message_id == "inbox-comment-id"
    assert "Linear inbox/chat issue" in event.text
    assert "Answer this without creating work." in event.text
    assert kb.list_tasks(kb.connect(), include_archived=True) == []
    adapter._client.set_issue_state_by_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_linear_send_maps_kanban_terminal_events_to_linear_states():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(api_key="lin_test"))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}
    adapter._issue_context_by_id["issue-id"] = {"team_key": "DEC", "team_id": "team-id"}

    done = await adapter.send(
        "issue-id",
        "✔ @default Kanban abc123 done — DEC-1: Done",
        metadata={"kanban_event_kind": "completed"},
    )
    blocked = await adapter.send(
        "issue-id",
        "⏸ @default Kanban def456 blocked: need input",
        metadata={"kanban_event_kind": "blocked"},
    )

    assert done.success is True
    assert done.message_id is None
    assert done.raw_response["reason"] == "kanban_terminal_comment"
    assert blocked.success is True
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "⏸ @default Kanban def456 blocked: need input",
    )
    assert adapter._client.set_issue_state_by_name.await_args_list[0].kwargs == {
        "issue_id": "issue-id",
        "state_name": "Ready for Review",
        "team_key": "DEC",
        "team_id": "team-id",
    }
    assert adapter._client.set_issue_state_by_name.await_args_list[1].kwargs == {
        "issue_id": "issue-id",
        "state_name": "Blocked",
        "team_key": "DEC",
        "team_id": "team-id",
    }


@pytest.mark.asyncio
async def test_linear_can_opt_into_completed_kanban_terminal_comments():
    from gateway.platforms.linear import LinearAdapter

    adapter = LinearAdapter(PlatformConfig(
        api_key="lin_test",
        extra={"kanban_terminal_comment_kinds": ["completed", "blocked"]},
    ))
    adapter._client = AsyncMock()
    adapter._client.comment_create.return_value = {"id": "created-comment"}
    adapter._issue_context_by_id["issue-id"] = {"team_key": "DEC", "team_id": "team-id"}

    done = await adapter.send(
        "issue-id",
        "✔ @default Kanban abc123 done — DEC-1: Done",
        metadata={"kanban_event_kind": "completed"},
    )

    assert done.success is True
    assert done.message_id == "created-comment"
    adapter._client.comment_create.assert_awaited_once_with(
        "issue-id",
        "✔ @default Kanban abc123 done — DEC-1: Done",
    )
    adapter._client.set_issue_state_by_name.assert_awaited_once_with(
        issue_id="issue-id",
        state_name="Ready for Review",
        team_key="DEC",
        team_id="team-id",
    )
