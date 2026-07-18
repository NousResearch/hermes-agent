"""Regression / reproduction for #60543.

`/steer` has three delivery paths.  Two of them wrap the user's message in the
`[OUT-OF-BAND USER MESSAGE]` marker so the model trusts it as a genuine user
instruction rather than treating it as tool output / prompt injection:

  * mid-batch drain  -> agent_runtime_helpers.apply_pending_steer_to_tool_results
  * pre-API drain    -> conversation_loop.py (calls format_steer_marker)

The THIRD path — the "leftover" path, taken when a steer lands *after* the
final assistant turn (no more tool batches to drain into) — is handed back as
raw ``result["pending_steer"]`` by turn_finalizer. Delivery consumers must wrap
it before queuing the next turn:

  * CLI          -> agent_runtime_helpers.queue_cli_leftover_steer
  * gateway      -> resolve_gateway_leftover_steer (+ discard_pending_slash_command)
  * Ink TUI      -> tui_gateway.server._deliver_leftover_steer

These tests exercise the production delivery seams (helpers actually called by
cli.py / gateway/run.py / tui_gateway/server.py), not mirrored copies of the
call-site statements.
"""
from __future__ import annotations

import inspect
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import (
    discard_pending_slash_command,
    format_leftover_steer_for_delivery,
    queue_cli_leftover_steer,
    resolve_gateway_leftover_steer,
)
from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN, format_steer_marker
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent

STEER_TEXT = "actually, stop and summarize what you have so far"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _bare_agent() -> AIAgent:
    """AIAgent without __init__ + manual steer state — matches test_steer.py."""
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    return agent


class _FinalizeAgent:
    """Minimal agent surface that finalize_turn reads from — mirrors the stub
    in tests/agent/test_turn_finalizer_final_response_persistence.py, but with
    a pending steer that lands after the final assistant turn (issue #60543).
    """

    def __init__(self, steer_text: str):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._steer_slot = steer_text

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_a, **_kw):
        pass

    def _safe_print(self, *_a, **_kw):
        pass

    def _save_trajectory(self, *_a, **_kw):
        pass

    def _cleanup_task_resources(self, *_a, **_kw):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        # A steer that arrived after the final assistant turn — the leftover
        # scenario. Drains once (like the real lock-guarded slot).
        text, self._steer_slot = self._steer_slot, None
        return text

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kw):
        pass


def _run_finalize(agent) -> dict:
    return finalize_turn(
        agent,
        final_response="Here is what I found.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": "on it",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "ok"},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="done",
    )


@pytest.fixture(autouse=True)
def _no_plugin_hooks(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])


def test_sibling_drain_path_marks_the_steer():
    """Baseline: the mid-batch path DOES wrap the steer in the OOB marker."""
    agent = _bare_agent()
    agent.steer(STEER_TEXT)
    messages = [{"role": "tool", "content": "output", "tool_call_id": "c1"}]
    agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
    assert STEER_MARKER_OPEN in messages[-1]["content"]
    assert STEER_TEXT in messages[-1]["content"]


def test_finalize_turn_hands_back_raw_unmarked_steer():
    """finalize_turn stashes the RAW steer (marking is the delivery layer's
    job). This is the value delivery seams receive."""
    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)
    assert result.get("pending_steer") == STEER_TEXT
    assert STEER_MARKER_OPEN not in result["pending_steer"]


def test_cli_delivery_seam_queues_marked_steer():
    """CLI seam: queue_cli_leftover_steer (called by cli.py) must mark the leftover."""
    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)
    pending_input: queue.Queue[str] = queue.Queue()

    delivered = queue_cli_leftover_steer(pending_input, result)

    assert delivered is not None
    assert delivered == pending_input.get_nowait()
    assert STEER_TEXT in delivered
    assert STEER_MARKER_OPEN in delivered, (
        "leftover /steer reached the model as raw command text, not as an "
        "out-of-band user message (#60543)"
    )
    assert not delivered.startswith("\n")


def test_delivery_helper_matches_marker_contract():
    """The leftover-delivery helper must emit the same open/close marker the
    system prompt tells the model to trust (see STEER_CHANNEL_NOTE)."""
    out = format_leftover_steer_for_delivery(STEER_TEXT)
    assert STEER_MARKER_OPEN in out
    assert STEER_MARKER_CLOSE in out
    assert STEER_TEXT in out
    assert out == format_steer_marker(STEER_TEXT).strip()


def test_gateway_delivery_seam_returns_marked_steer():
    """Gateway seam: resolve_gateway_leftover_steer (called by gateway/run.py)."""
    agent = _FinalizeAgent(STEER_TEXT)
    result = _run_finalize(agent)

    pending = resolve_gateway_leftover_steer(result)

    assert pending is not None
    assert STEER_TEXT in pending
    assert STEER_MARKER_OPEN in pending, (
        "gateway delivered leftover /steer as raw command text, not as an "
        "out-of-band user message (#60543)"
    )


def test_gateway_marked_steer_survives_slash_command_net():
    """Marked leftover must survive discard_pending_slash_command (gateway net).

    Exercises the real safety-net helper with a /stop … leftover — the bug
    where raw pending_steer beginning with '/' was silently dropped.
    """
    agent = _FinalizeAgent("/stop the search and summarize")
    result = _run_finalize(agent)

    pending = resolve_gateway_leftover_steer(result)
    assert pending is not None
    pending = discard_pending_slash_command(pending)

    assert pending is not None, "marked steer was wrongly discarded by the slash-command net"
    assert STEER_MARKER_OPEN in pending
    assert "/stop the search and summarize" in pending


def test_gateway_slash_net_still_drops_raw_commands():
    """Safety net must still discard unmarked slash-command pending text."""
    assert discard_pending_slash_command("/stop the search") is None
    assert discard_pending_slash_command("keep going") == "keep going"


def test_gateway_leftover_yields_to_existing_pending():
    """Queued gateway messages retain priority over leftover /steer."""
    result = {"pending_steer": STEER_TEXT}
    assert resolve_gateway_leftover_steer(result, pending="user already queued") is None
    assert resolve_gateway_leftover_steer(result, pending_event=object()) is None


def test_tui_delivery_seam_dispatches_marked_steer(monkeypatch):
    """TUI seam: _deliver_leftover_steer (called by _run_prompt_submit)."""
    from tui_gateway import server

    submitted: list[str] = []

    def _capture_submit(rid, sid, session, text):
        submitted.append(text)
        with session["history_lock"]:
            session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", _capture_submit)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    session = {
        "running": False,
        "history_lock": threading.Lock(),
    }
    result = {"pending_steer": STEER_TEXT}

    assert server._deliver_leftover_steer("r1", "sid", session, result) is True
    assert len(submitted) == 1
    assert STEER_MARKER_OPEN in submitted[0]
    assert STEER_TEXT in submitted[0]


def test_tui_delivery_seam_skips_when_queued_prompt_would_win():
    """_deliver_leftover_steer must not fire when the session is already running
    (e.g. after _drain_queued_prompt claimed the next turn)."""
    from tui_gateway import server

    session = {
        "running": True,
        "history_lock": threading.Lock(),
    }
    assert (
        server._deliver_leftover_steer(
            "r1", "sid", session, {"pending_steer": STEER_TEXT}
        )
        is False
    )


def test_production_call_sites_wire_delivery_seams():
    """Call-site wiring check: consumers must invoke the shared seams.

    Fails if cli.py / gateway/run.py / tui_gateway revert to raw pending_steer
    passthrough while these unit tests still pass against the helpers alone.
    """
    cli_src = (REPO_ROOT / "cli.py").read_text(encoding="utf-8")
    assert "queue_cli_leftover_steer" in cli_src

    gateway_src = (REPO_ROOT / "gateway" / "run.py").read_text(encoding="utf-8")
    assert "resolve_gateway_leftover_steer" in gateway_src
    assert "discard_pending_slash_command" in gateway_src

    from tui_gateway import server

    submit_src = inspect.getsource(server._run_prompt_submit)
    assert "_drain_queued_prompt" in submit_src
    assert "_deliver_leftover_steer" in submit_src
    assert submit_src.index("_drain_queued_prompt") < submit_src.index(
        "_deliver_leftover_steer"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
