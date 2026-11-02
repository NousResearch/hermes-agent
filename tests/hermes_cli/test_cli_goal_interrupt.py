"""Tests for CLI goal-continuation interrupt handling.

Covers:
- Ctrl+C during a /goal turn auto-pauses the goal (no more continuations).
- Empty/whitespace-only responses skip the judge (no phantom continuations).
- Clean response without interrupt still drives the judge + enqueues.

These tests exercise ``_maybe_continue_goal_after_turn`` directly on a
minimal ``HermesCLI`` stub (pattern used elsewhere in tests/cli).
"""

from __future__ import annotations

import queue
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal module's DB cache so it re-resolves HERMES_HOME each test.
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()

def _make_cli_with_goal(session_id: str, goal_text: str = "build a thing"):
    """Build a minimal HermesCLI stub with an active goal wired in."""
    from cli import HermesCLI
    from hermes_cli.goals import GoalManager

    cli = HermesCLI.__new__(HermesCLI)
    # State the hook + helpers touch directly.
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = []
    # `_get_goal_manager()` reads `self.session_id` directly, not
    # `self.agent.session_id`. Match the production lookup.
    cli.session_id = session_id
    cli.agent = MagicMock()
    cli.agent.session_id = session_id

    mgr = GoalManager(session_id=session_id, default_max_turns=5)
    mgr.set(goal_text)
    cli._goal_manager = mgr
    return cli, mgr

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

class TestInterruptAutoPause:

    def test_interrupted_turn_is_resumable(self, hermes_home):
        """After auto-pause from Ctrl+C, /goal resume puts it back to active."""
        sid = f"sid-resume-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli.conversation_history = [
            {"role": "assistant", "content": "partial"},
        ]
        with patch("hermes_cli.goals.judge_goal"):
            cli._maybe_continue_goal_after_turn()
        assert mgr.state.status == "paused"

        mgr.resume()
        assert mgr.state.status == "active"

class TestHealthyTurnStillRuns:
    def test_clean_response_enqueues_continuation_when_judge_says_continue(
        self, hermes_home,
    ):
        """Sanity check: the hook still works in the happy path."""
        sid = f"sid-healthy-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "did some work, more to do"},
        ]

        # Force the judge to say "continue" without touching the network.
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "needs more steps", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        # Continuation prompt must be queued.
        assert not cli._pending_input.empty()
        queued = cli._pending_input.get_nowait()
        assert "Continuing toward your standing goal" in queued
        assert mgr.state.status == "active"

    def test_clean_response_marks_done_when_judge_says_done(self, hermes_home):
        sid = f"sid-done-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        from hermes_cli.goals import GoalContract
        mgr.set_contract(GoalContract(verification="The artifact is built and its check passes."))
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "tool", "name": "terminal", "content": '{"exit_code": 0, "output": "artifact built; check passed"}'},
            {"role": "assistant", "content": "all finished, here's the result"},
        ]

        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("done", "goal satisfied", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "done"


class TestGoalEvidence:
    refusal = (
        "Since I cannot directly restart services without proper instructions from you about how "
        "to proceed (the manual intervention is required for service restarts), the next step "
        "would be to document the exact failures for your review and implementation."
    )

    def _run(self, cli, verdict, reason="needs instructions"):
        with patch("hermes_cli.goals.judge_goal", return_value=(verdict, reason, False, None, False)):
            cli._maybe_continue_goal_after_turn()

    def test_original_unsupported_service_refusal_continues(self, hermes_home):
        goal = ("do all the next steps also keep improving all local models all memory i added "
                "lite llm ... many regenerative stuff ... alot of setting and underestanding "
                "of things and meanings are wrong as well")
        cli, mgr = _make_cli_with_goal("original-refusal", goal)
        mgr.state.max_turns = 20
        cli.conversation_history = [{"role": "user", "content": goal},
                                    {"role": "assistant", "content": self.refusal}]
        self._run(cli, "blocked", "Service restarts require user instructions (manual intervention required).")
        assert mgr.state.status == "active"
        assert mgr.state.turns_used == 1
        prompt = cli._pending_input.get_nowait()
        assert "Try an available tool" in prompt
        assert "do not bypass real permission" in prompt

    def test_actual_permission_denial_still_pauses(self, hermes_home):
        cli, mgr = _make_cli_with_goal("real-denial", "Repair and restart the service")
        cli.conversation_history = [
            {"role": "user", "content": mgr.state.goal},
            {"role": "tool", "name": "terminal", "content": '{"exit_code": 1, "output": "Operation not permitted"}'},
            {"role": "assistant", "content": self.refusal},
        ]
        self._run(cli, "blocked", "The service manager returned Operation not permitted.")
        assert mgr.state.status == "paused"
        assert cli._pending_input.empty()

    @pytest.mark.parametrize("goal,response", [
        ("Restart the chosen service", "I need instructions about which service you want restarted."),
        ("Repair the service; do not restart it", refusal),
        ("Repair the service; ask before restarting", "A security policy requires approval before restarting."),
        ("Repair the service", "I need approval for a paid service before proceeding."),
        ("Repair the service", "I need instructions: please specify the service name."),
        ("Audit the service and report its status", refusal),
    ])
    def test_real_decisions_and_boundaries_still_pause(self, hermes_home, goal, response):
        cli, mgr = _make_cli_with_goal("real-boundary", goal)
        cli.conversation_history = [{"role": "user", "content": goal},
                                    {"role": "assistant", "content": response}]
        self._run(cli, "blocked")
        assert mgr.state.status == "paused"
        assert cli._pending_input.empty()

    def test_report_only_operational_done_is_rejected(self, hermes_home):
        cli, mgr = _make_cli_with_goal("report-only", "Implement the service repairs and verify them")
        cli.conversation_history = [{"role": "assistant", "content": "Done. Here is my repair plan for you to implement."}]
        self._run(cli, "done", "all completed")
        assert mgr.state.status == "active"
        assert "Perform the requested work" in cli._pending_input.get_nowait()

    @pytest.mark.parametrize("name,content", [
        ("terminal", '{"exit_code":0,"output":"README.md  src  tests"}'),
        ("search", '{"success":true,"results":["service restart guide"]}'),
        ("dispatch", '{"ok":true,"task_id":"repair-queued"}'),
    ])
    def test_diagnostics_and_dispatch_are_not_operational_proof(self, hermes_home, name, content):
        cli, mgr = _make_cli_with_goal("diagnostic-only", "Repair the service and verify it is healthy")
        cli.conversation_history = [{"role": "tool", "name": name, "content": content},
                                    {"role": "assistant", "content": "Done. Repair brief dispatched."}]
        self._run(cli, "done")
        assert mgr.state.status == "active"
        prompt = cli._pending_input.get_nowait()
        assert "GoalManager('diagnostic-only')" in prompt
        assert "m.set_contract(c)" in prompt

    @pytest.mark.parametrize("content", [
        '{"ok":false}', '{"success":false}', '{"status":"failure"}',
        'Traceback (most recent call last):\nRuntimeError: broken',
    ])
    def test_explicit_tool_failure_cannot_satisfy_verification(self, hermes_home, content):
        from hermes_cli.goals import GoalContract

        cli, mgr = _make_cli_with_goal("false-result", "Repair the service")
        mgr.set_contract(GoalContract(verification="The destination health check returns healthy."))
        cli.conversation_history = [{"role": "tool", "content": content},
                                    {"role": "assistant", "content": "Done."}]
        self._run(cli, "done")
        assert mgr.state.status == "active"

    def test_agent_can_fill_missing_verification_through_native_api(self, hermes_home):
        from hermes_cli.goals import GoalManager

        cli, mgr = _make_cli_with_goal("fill-verification", "Repair the service")
        writer = GoalManager(mgr.session_id)
        contract = writer.state.contract
        contract.verification = "The destination health check prints healthy with exit 0."
        writer.set_contract(contract)
        cli.conversation_history = [
            {"role": "tool", "name": "terminal", "content": '{"exit_code":0,"output":"healthy"}'},
            {"role": "assistant", "content": "The destination health check passed."},
        ]
        self._run(cli, "done", "destination health verified")
        assert mgr.state.contract.verification == contract.verification
        assert mgr.state.status == "done"

    @pytest.mark.parametrize("goal,expected", [
        ("write a script, run it and repair the service", True),
        ("write a poem about a service restart", False),
        ("Please draft a report about repair options", False),
        ("draft the plan, then implement the repairs", True),
    ])
    def test_mixed_deliverables_keep_operational_requirements(self, goal, expected):
        from hermes_cli.goals import _operational_goal
        assert _operational_goal(goal) is expected

    def test_failed_tool_is_not_completion_evidence(self, hermes_home):
        cli, mgr = _make_cli_with_goal("failed-tool", "Repair the service")
        cli.conversation_history = [
            {"role": "tool", "content": '{"exit_code": 7, "output": "connection failed"}'},
            {"role": "assistant", "content": "Done."},
        ]
        self._run(cli, "done")
        assert mgr.state.status == "active"
        assert not cli._pending_input.empty()

    def test_previous_turn_tools_cannot_prove_current_completion(self, hermes_home):
        cli, mgr = _make_cli_with_goal("old-evidence", "Repair the second service")
        cli.conversation_history = [
            {"role": "tool", "content": '{"exit_code": 0, "output": "first service healthy"}'},
            {"role": "user", "content": "Repair the second service"},
            {"role": "assistant", "content": "All done."},
        ]
        self._run(cli, "done")
        assert mgr.state.status == "active"

    def test_passing_deterministic_gate_allows_verified_completion(self, hermes_home):
        import shlex
        import sys

        cli, mgr = _make_cli_with_goal("verified-gate", "Repair the service")
        mgr.add_gate(f'{shlex.quote(sys.executable)} -c \'print("destination healthy")\'')
        cli.conversation_history = [{"role": "assistant", "content": "Repairs complete; verify destination health."}]
        self._run(cli, "done", "destination check passed")
        assert mgr.state.status == "done"
        assert mgr.state.gates[0].last_exit_code == 0
        assert "destination healthy" in mgr.state.gates[0].last_output_tail
        assert cli._pending_input.empty()

    def test_text_deliverable_does_not_require_tools(self, hermes_home):
        cli, mgr = _make_cli_with_goal("text-only", "Write a greeting")
        cli.conversation_history = [{"role": "assistant", "content": "Hello, Fares."}]
        self._run(cli, "done", "greeting delivered")
        assert mgr.state.status == "done"

    def test_corrected_refusal_still_obeys_turn_budget(self, hermes_home):
        cli, mgr = _make_cli_with_goal("refusal-budget", "Repair the service")
        mgr.state.max_turns = 1
        cli.conversation_history = [{"role": "assistant", "content": self.refusal}]
        self._run(cli, "blocked")
        assert mgr.state.status == "paused"
        assert "turn budget exhausted" in mgr.state.paused_reason
        assert cli._pending_input.empty()

    def test_judge_receives_actual_tool_result_not_just_prose(self, hermes_home):
        from hermes_cli.goals import judge_goal

        messages = [
            {"role": "user", "content": "Repair the service"},
            {"role": "assistant", "tool_calls": [{"id": "check-1", "function": {"name": "terminal"}}]},
            {"role": "tool", "tool_call_id": "check-1", "content": '{"exit_code": 0, "output": "destination healthy"}'},
        ]
        with patch("agent.auxiliary_client.call_llm", return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"verdict":"done","reason":"checked"}'))],
        )) as call:
            verdict, *_ = judge_goal("Repair the service", "Done", turn_messages=messages)
        assert verdict == "done"
        prompt = call.call_args.kwargs["messages"][1]["content"]
        assert "Actual tool results" in prompt
        assert "terminal:" in prompt
        assert "destination healthy" in prompt

    def test_gateway_reads_persisted_evidence_for_streamed_turn(self, hermes_home):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from gateway.run_goals import GatewayGoalsMixin
        from hermes_cli import goals

        mgr = goals.GoalManager("gateway-evidence")
        mgr.set("Repair the service")
        messages = [{"role": "tool", "content": '{"exit_code": 0, "output": "healthy"}'}]
        db = MagicMock()
        db.get_messages.return_value = messages

        async def execute(fn):
            return fn()

        runner = SimpleNamespace(
            _post_turn_manager=AsyncMock(return_value=mgr),
            _run_in_executor_with_context=execute,
        )
        with patch.object(mgr, "evaluate_after_turn", return_value={}) as evaluate, patch.object(
            goals, "_get_session_db", return_value=db,
        ):
            asyncio.run(GatewayGoalsMixin._post_turn_goal_continuation(
                runner, session_entry=SimpleNamespace(session_id=mgr.session_id),
                source=None, final_response="The streamed repair is complete.",
            ))
        assert evaluate.call_args.kwargs["turn_messages"] == messages
        db.get_messages.assert_called_once_with(mgr.session_id, limit=200, latest=True)

    def test_desktop_passes_actual_turn_results_to_goal_manager(self, hermes_home):
        from tui_gateway import prompt_turn

        mgr = MagicMock()
        mgr.evaluate_after_turn.return_value = {"should_continue": True, "continuation_prompt": "do the repair"}
        messages = [{"role": "tool", "content": '{"ok":false}'}]
        session = {"session_key": "desktop-goal", "history": []}
        with patch.object(prompt_turn, "_active_goal_manager", return_value=mgr), patch.object(
            prompt_turn, "_plan_goal_compression_recovery", return_value=(None, None),
        ), patch.object(prompt_turn, "_emit", create=True):
            result = prompt_turn._goal_followup_after_turn(
                "desktop", session, {"messages": messages}, "complete", "Done",
            )
        assert result == "do the repair"
        assert mgr.evaluate_after_turn.call_args.kwargs["turn_messages"] == messages
