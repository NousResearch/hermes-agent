import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slack_intent import (
    SlackAsyncPolicy,
    SlackIntentKind,
    build_kanban_create_text,
    classify_slack_intent,
    policy_from_config,
)


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="xoxb-test")}
    )
    runner.adapters = {}
    runner._background_tasks = set()
    runner._slack_async_handoffs = {}
    runner._check_slash_access = lambda _source, _command: None
    return runner


def _make_event(text="research this deeply", message_id="1700000000.000100"):
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="group",
        user_id="U123",
        thread_id="1700000000.000000",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


def test_slack_intent_classifier_separates_quick_long_project_and_forced_prefixes():
    policy = SlackAsyncPolicy()

    quick = classify_slack_intent("quick take: why is CI red?", policy)
    assert quick.kind == SlackIntentKind.QUICK

    long = classify_slack_intent("please do a deep dive on these logs", policy)
    assert long.kind == SlackIntentKind.LONG_AD_HOC

    short_research = classify_slack_intent("research postgres locks", policy)
    assert short_research.kind == SlackIntentKind.QUICK

    project_question = classify_slack_intent("why did tests fail in this repo?", policy)
    assert project_question.kind == SlackIntentKind.QUICK

    project = classify_slack_intent("fix this bug and open a PR", policy)
    assert project.kind == SlackIntentKind.PROJECT_CODE

    forced = classify_slack_intent("fg: implement is being used hypothetically", policy)
    assert forced.kind == SlackIntentKind.QUICK
    assert forced.text == "implement is being used hypothetically"
    assert forced.forced is True


def test_policy_from_config_reads_slack_async_routing_defaults_and_overrides():
    policy = policy_from_config(
        {
            "slack": {
                "async_routing": {
                    "foreground_max_iterations": 4,
                    "project_routing": "background",
                    "project_assignee": "worker-a",
                    "project_skills": ["github-pr-workflow"],
                }
            }
        }
    )

    assert policy.enabled is True
    assert policy.foreground_max_iterations == 4
    assert policy.project_routing == "background"
    assert policy.project_assignee == "worker-a"
    assert policy.project_skills == ("github-pr-workflow",)


@pytest.mark.asyncio
async def test_long_ad_hoc_slack_intent_acknowledges_and_schedules_background(monkeypatch):
    runner = _make_runner()
    event = _make_event("please research this thoroughly in the background")
    runner._run_background_task = AsyncMock()
    monkeypatch.setattr(
        GatewayRunner,
        "_slack_async_policy_from_config",
        staticmethod(lambda: SlackAsyncPolicy()),
    )

    response, decision, _policy = await runner._maybe_route_slack_async_intent(event)

    assert "report back here" in response
    assert "background" in response
    assert "Task ID:" in response
    assert decision.kind == SlackIntentKind.LONG_AD_HOC
    assert len(runner._background_tasks) == 1
    await asyncio.gather(*runner._background_tasks)
    runner._run_background_task.assert_awaited_once()
    kwargs = runner._run_background_task.await_args.kwargs
    assert kwargs["event_message_id"] == event.message_id


@pytest.mark.asyncio
async def test_project_intent_can_use_background_mode_without_foreground_block(monkeypatch):
    runner = _make_runner()
    event = _make_event("fix this bug and open a PR")
    runner._run_background_task = AsyncMock()
    policy = SlackAsyncPolicy(project_routing="background")
    monkeypatch.setattr(
        GatewayRunner,
        "_slack_async_policy_from_config",
        staticmethod(lambda: policy),
    )

    response, decision, _policy = await runner._maybe_route_slack_async_intent(event)

    assert "Queued for project work" in response
    assert "background" in response
    assert decision.kind == SlackIntentKind.PROJECT_CODE
    await asyncio.gather(*runner._background_tasks)
    runner._run_background_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_kanban_project_mode_does_not_silently_fallback(monkeypatch):
    runner = _make_runner()
    event = _make_event("fix this bug and open a PR")
    runner._run_background_task = AsyncMock()
    policy = SlackAsyncPolicy(project_routing="kanban", project_assignee="worker-a")
    monkeypatch.setattr(
        GatewayRunner,
        "_slack_async_policy_from_config",
        staticmethod(lambda: policy),
    )
    monkeypatch.setattr(runner, "_slack_project_kanban_available", lambda _policy: False)

    response, decision, _policy = await runner._maybe_route_slack_async_intent(event)

    assert "kanban worker routing is not available" in response
    assert "did not start a background fallback" in response
    assert decision.kind == SlackIntentKind.PROJECT_CODE
    assert runner._background_tasks == set()
    runner._run_background_task.assert_not_called()


def test_build_kanban_create_text_preserves_thread_identity_and_idempotency():
    policy = SlackAsyncPolicy(
        project_assignee="codex-worker",
        project_workspace="scratch",
        project_goal=True,
        project_skills=("github-pr-workflow",),
    )
    decision = classify_slack_intent("fix this bug and open a PR", policy)

    text = build_kanban_create_text(
        decision=decision,
        policy=policy,
        chat_id="C123",
        thread_id="1700000000.000000",
        message_id="1700000000.000100",
        user_id="U123",
    )

    assert text.startswith("/kanban create")
    assert "--assignee codex-worker" in text
    assert "--created-by slack:U123" in text
    assert "--idempotency-key slack:" in text
    assert "--skill github-pr-workflow" in text


@pytest.mark.asyncio
async def test_budget_exceeded_slack_quick_turn_hands_off_to_background(monkeypatch):
    runner = _make_runner()
    event = _make_event("quickly compare these two options")
    runner._run_background_task = AsyncMock()
    policy = SlackAsyncPolicy(foreground_max_iterations=2, async_on_budget_exceeded=True)

    result = {"api_calls": 2, "final_response": "maximum iterations reached", "failed": True}
    assert runner._slack_budget_exceeded(result, policy) is True

    response = await runner._handoff_slack_budget_exceeded(event, result, policy)

    assert "foreground budget" in response
    assert "Task ID:" in response
    await asyncio.gather(*runner._background_tasks)
    runner._run_background_task.assert_awaited_once()
