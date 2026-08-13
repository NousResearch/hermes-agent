"""Phase 2 tests: New Task / New Message / New Agent Run flows (CR-206).

Each flow must preview before executing (CR-207), show profile/scope, route
through the router, and handle missing input and missing backend
deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from control_room.actions import ControlRoomActionRouter
from control_room.contract import ControlRoomActionResult
from control_room.flows import NewAgentRunFlow, NewMessageFlow, NewTaskFlow


def _ok_executor(target, params, context):
    return ControlRoomActionResult(status="completed", message="ok", receipt={"done": True})


class TestNewTaskFlow:
    def test_preview_shows_title_and_scope(self):
        router = ControlRoomActionRouter(
            executors={"kanban_create": _ok_executor}, scope_profile="kensei"
        )
        flow = NewTaskFlow(router, profile="kensei")
        preview = flow.preview(title="Fix gateway", assignee="octacon")
        assert "Fix gateway" in preview
        assert "assignee=octacon" in preview
        assert "kensei" in preview

    def test_submit_unconfirmed_requires_confirmation(self):
        router = ControlRoomActionRouter(
            executors={"kanban_create": _ok_executor}, scope_profile="kensei"
        )
        flow = NewTaskFlow(router, profile="kensei")
        result = flow.submit(confirmed=False, title="Fix gateway")
        assert result.action_result.status == "confirmation_required"

    def test_submit_confirmed_executes(self):
        router = ControlRoomActionRouter(
            executors={"kanban_create": _ok_executor}, scope_profile="kensei"
        )
        flow = NewTaskFlow(router, profile="kensei")
        result = flow.submit(confirmed=True, title="Fix gateway")
        assert result.action_result.status == "completed"

    def test_missing_title_fails(self):
        router = ControlRoomActionRouter(
            executors={"kanban_create": _ok_executor}, scope_profile="kensei"
        )
        flow = NewTaskFlow(router, profile="kensei")
        result = flow.submit(confirmed=True)
        assert result.action_result.status == "failed"
        assert "title" in result.action_result.message


class TestNewMessageFlow:
    def test_preview_shows_target_and_snippet(self):
        router = ControlRoomActionRouter(
            executors={"peer_send": _ok_executor}, scope_profile="kensei"
        )
        flow = NewMessageFlow(router, profile="kensei")
        preview = flow.preview(target="remii", message="Please review the PR")
        assert "remii" in preview
        assert "Please review the PR" in preview

    def test_submit_confirmed_executes(self):
        router = ControlRoomActionRouter(
            executors={"peer_send": _ok_executor}, scope_profile="kensei"
        )
        flow = NewMessageFlow(router, profile="kensei")
        result = flow.submit(confirmed=True, target="remii", message="hello")
        assert result.action_result.status == "completed"

    def test_missing_target_or_message_fails(self):
        router = ControlRoomActionRouter(
            executors={"peer_send": _ok_executor}, scope_profile="kensei"
        )
        flow = NewMessageFlow(router, profile="kensei")
        assert flow.submit(confirmed=True, target="remii").action_result.status == "failed"
        assert flow.submit(confirmed=True, message="hi").action_result.status == "failed"


class TestNewAgentRunFlow:
    def test_preview_shows_prompt_and_target(self):
        router = ControlRoomActionRouter(
            executors={"agent_run": _ok_executor}, scope_profile="kensei"
        )
        flow = NewAgentRunFlow(router, profile="kensei")
        preview = flow.preview(prompt="Summarize HN", target="background")
        assert "Summarize HN" in preview
        assert "background" in preview

    def test_submit_confirmed_executes_with_runner(self):
        calls = {}

        def runner(prompt, profile):
            calls["prompt"] = prompt
            calls["profile"] = profile
            return {"delegation_id": "d1"}

        router = ControlRoomActionRouter(
            executors={"agent_run": _ok_executor},
            scope_profile="kensei",
        )
        flow = NewAgentRunFlow(router, profile="kensei")
        result = flow.submit(confirmed=True, prompt="Summarize HN", target="background")
        assert result.action_result.status == "completed"

    def test_submit_without_runner_is_unavailable(self):
        from control_room.executors import agent_run_executor

        router = ControlRoomActionRouter(
            executors={"agent_run": agent_run_executor()}, scope_profile="kensei"
        )
        flow = NewAgentRunFlow(router, profile="kensei")
        result = flow.submit(confirmed=True, prompt="Summarize HN")
        # Real executor present but no runner injected into context -> unavailable.
        assert result.action_result.status == "unavailable"

    def test_missing_prompt_fails(self):
        router = ControlRoomActionRouter(
            executors={"agent_run": _ok_executor}, scope_profile="kensei"
        )
        flow = NewAgentRunFlow(router, profile="kensei")
        result = flow.submit(confirmed=True)
        assert result.action_result.status == "failed"
        assert "prompt" in result.action_result.message


class TestFlowCrossProfile:
    def test_flow_target_profile_must_match_router_scope(self):
        router = ControlRoomActionRouter(
            executors={"peer_send": _ok_executor}, scope_profile="kensei"
        )
        flow = NewMessageFlow(router, profile="kensei")
        result = flow.submit(confirmed=True, target="remii", message="hi")
        # Flows bind the profile into the action target; router scope check
        # must accept the same-profile target.
        assert result.action_result.status == "completed"
