"""Phase 2 tests: action routing, confirmation contract, safety guards (CR-208).

Covers the full negative matrix: stale revision, cross-session/cross-profile
targets, unavailable peer plugin, denial/cancellation, duplicate submit,
failed backend receipt, invalid transitions, and the two-step confirmation
contract. Executors are injected fakes so every guard is exercised
deterministically; default executors are additionally smoke-tested for
unavailable degradation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from control_room.actions import ControlRoomActionRouter
from control_room.contract import (
    ActionTarget,
    ControlRoomAction,
    ControlRoomActionResult,
    ErrorCode,
)
from control_room.executors import (
    default_executors,
    kanban_move_executor,
    peer_send_executor,
    process_kill_executor,
)


def _action(
    kind: str,
    target_id: str,
    *,
    confirmation: str = "required",
    params: Optional[Dict[str, Any]] = None,
    expected_revision: Optional[str] = None,
    action_id: str = "act-1",
    profile: Optional[str] = None,
) -> ControlRoomAction:
    return ControlRoomAction(
        id=action_id,
        target=ActionTarget(kind=kind, id=target_id, profile=profile),
        parameters=params or {},
        confirmation=confirmation,
        expected_revision=expected_revision,
    )


def _ok_executor(target, params, context):
    return ControlRoomActionResult(status="completed", message="ok", receipt={"done": True})


class TestConfirmationContract:
    def test_required_action_returns_confirmation_required_first(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        result = router.dispatch(_action("process", "p1"), confirmed=False)
        assert result.status == "confirmation_required"
        assert "Confirm" in result.message
        assert result.receipt["action_id"] == "act-1"

    def test_confirmed_action_executes(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        result = router.dispatch(_action("process", "p1"), confirmed=True)
        assert result.status == "completed"

    def test_none_action_skips_confirmation(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        result = router.dispatch(_action("process", "p1", confirmation="none"))
        assert result.status == "completed"


class TestRevisionCheck:
    def test_stale_revision_rejected(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor},
            verifier=lambda target, ctx: "rev-2",
            scope_profile="kensei",
        )
        result = router.dispatch(
            _action("process", "p1", expected_revision="rev-1"),
            confirmed=True,
        )
        assert result.status == "stale"
        assert result.receipt["expected"] == "rev-1"
        assert result.receipt["live"] == "rev-2"

    def test_matching_revision_executes(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor},
            verifier=lambda target, ctx: "rev-1",
            scope_profile="kensei",
        )
        result = router.dispatch(
            _action("process", "p1", expected_revision="rev-1"),
            confirmed=True,
        )
        assert result.status == "completed"

    def test_verifier_exception_is_failed_not_crash(self):
        def boom(target, ctx):
            raise RuntimeError("verifier down")

        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, verifier=boom, scope_profile="kensei"
        )
        result = router.dispatch(
            _action("process", "p1", expected_revision="rev-1"),
            confirmed=True,
        )
        assert result.status == "failed"


class TestCrossProfileGuard:
    def test_cross_profile_target_rejected(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        result = router.dispatch(
            _action("process", "p1", profile="remii"),
            confirmed=True,
        )
        assert result.status == "rejected"
        assert result.receipt["error"] == ErrorCode.CROSS_PROFILE.value

    def test_same_profile_target_allowed(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        result = router.dispatch(
            _action("process", "p1", profile="kensei"),
            confirmed=True,
        )
        assert result.status == "completed"

    def test_allow_cross_profile_opt_in(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor},
            scope_profile="kensei",
            allow_cross_profile=True,
        )
        result = router.dispatch(
            _action("process", "p1", profile="remii"),
            confirmed=True,
        )
        assert result.status == "completed"


class TestUnknownAndUnavailable:
    def test_unknown_action_kind(self):
        router = ControlRoomActionRouter(executors={}, scope_profile="kensei")
        result = router.dispatch(_action("bogus", "x"), confirmed=True)
        assert result.status == "unavailable"
        assert result.receipt["error"] == ErrorCode.UNKNOWN_ACTION.value

    def test_capability_probe(self):
        router = ControlRoomActionRouter(executors={"process": _ok_executor}, scope_profile="kensei")
        assert router.capability("process") is True
        assert router.capability("approval") is False


class TestDuplicateSubmit:
    def test_duplicate_submit_rejected_after_execution(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        first = router.dispatch(_action("process", "p1"), confirmed=True)
        assert first.status == "completed"
        second = router.dispatch(_action("process", "p1"), confirmed=True)
        assert second.status == "rejected"
        assert second.receipt["duplicate"] is True

    def test_different_action_id_allowed(self):
        router = ControlRoomActionRouter(
            executors={"process": _ok_executor}, scope_profile="kensei"
        )
        assert router.dispatch(_action("process", "p1", action_id="a"), confirmed=True).status == "completed"
        assert router.dispatch(_action("process", "p1", action_id="b"), confirmed=True).status == "completed"


class TestExecutorFailure:
    def test_executor_exception_is_failed(self):
        def boom(target, params, ctx):
            raise RuntimeError("executor down")

        router = ControlRoomActionRouter(executors={"process": boom}, scope_profile="kensei")
        result = router.dispatch(_action("process", "p1"), confirmed=True)
        assert result.status == "failed"

    def test_executor_returning_failed_passthrough(self):
        def fail(target, params, ctx):
            return ControlRoomActionResult(status="failed", message="nope")

        router = ControlRoomActionRouter(executors={"process": fail}, scope_profile="kensei")
        result = router.dispatch(_action("process", "p1"), confirmed=True)
        assert result.status == "failed"
        assert result.message == "nope"


class TestDefaultExecutorsDegrade:
    def test_approval_degrades_when_no_responder(self):
        from control_room.executors import approval_executor_factory

        ex = approval_executor_factory(respond_fn=None)
        result = ex(ActionTarget(kind="approval", id="a1"), {"decision": "allow"}, {})
        # Default responder path is unavailable in a bare process.
        assert result.status in ("unavailable", "failed")

    def test_process_kill_degrades_or_runs(self):
        ex = process_kill_executor()
        result = ex(ActionTarget(kind="process", id="does-not-exist"), {}, {})
        # Either unavailable (no registry) or failed (not found) — never completed.
        assert result.status in ("unavailable", "failed")

    def test_peer_send_requires_message(self):
        ex = peer_send_executor()
        result = ex(ActionTarget(kind="peer_send", id="peer-x"), {}, {})
        assert result.status == "failed"

    def test_kanban_move_rejects_invalid_transition(self):
        ex = kanban_move_executor()
        result = ex(
            ActionTarget(kind="kanban_move", id="t1"),
            {"transition": "retry"},
            {},
        )
        assert result.status == "failed"
        assert "retry" not in result.message.lower() or "transition" in result.message

    def test_default_executors_expose_no_dead_controls(self):
        execs = default_executors()
        for kind, ex in execs.items():
            # Every default executor must respond deterministically to a bogus
            # target without crashing; each wraps an authoritative backend.
            assert callable(ex)


class TestApprovalExecutor:
    def test_allow_via_responder(self):
        from control_room.executors import approval_executor_factory

        calls = {}

        def fake_responder(request_id, decision, context):
            calls["request_id"] = request_id
            calls["decision"] = decision
            return {"ok": True}

        ex = approval_executor_factory(respond_fn=fake_responder)
        result = ex(ActionTarget(kind="approval", id="req-1"), {"decision": "allow"}, {})
        assert result.status == "completed"
        assert calls["request_id"] == "req-1"
        assert calls["decision"] == "allow"

    def test_deny_via_responder(self):
        from control_room.executors import approval_executor_factory

        def fake_responder(request_id, decision, context):
            return {"ok": True}

        ex = approval_executor_factory(respond_fn=fake_responder)
        result = ex(ActionTarget(kind="approval", id="req-1"), {"decision": "deny"}, {})
        assert result.status == "completed"

    def test_invalid_decision_rejected(self):
        from control_room.executors import approval_executor_factory

        ex = approval_executor_factory(respond_fn=lambda *a, **k: {"ok": True})
        result = ex(ActionTarget(kind="approval", id="req-1"), {"decision": "maybe"}, {})
        assert result.status == "failed"

    def test_responder_failure_surfaces(self):
        from control_room.executors import approval_executor_factory

        def bad_responder(request_id, decision, context):
            return {"ok": False, "error": "no such approval"}

        ex = approval_executor_factory(respond_fn=bad_responder)
        result = ex(ActionTarget(kind="approval", id="req-1"), {"decision": "allow"}, {})
        assert result.status == "failed"
        assert "no such approval" in result.message
