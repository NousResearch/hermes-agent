"""Tests for the end-of-turn completion explainer (#34452).

When a turn ends abnormally after tools (empty content after retries, a
partial/truncated stream, exhausted retries, or an iteration/budget limit)
the user should get a single user-visible explanation of why the reply
stopped instead of a blank or fragmentary response box.  Normal short
replies (e.g. ``Done.``) must stay quiet.

These tests exercise:
  1. ``_format_turn_completion_explanation`` — the pure reason→message map.
  2. ``_turn_completion_explainer_enabled`` — the env/config seam.
  3. An end-to-end ``run_conversation`` turn that exhausts empty-response
     retries and verifies the explanation reaches ``final_response``.

All assertions work under the mocked OpenAI SDK used elsewhere in this
suite (we patch ``run_agent.OpenAI`` and drive ``agent.client``), so they
pass identically in CI and locally.
"""

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# --------------------------------------------------------------------------
# Fixtures (mirrors tests/run_agent/test_tool_call_guardrail_runtime.py)
# --------------------------------------------------------------------------
def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    # No fallback chain so empty responses exhaust deterministically.
    agent._fallback_chain = []
    return agent


# --------------------------------------------------------------------------
# 1. Pure formatter
# --------------------------------------------------------------------------
def test_explanation_quiet_for_normal_text_response():
    """A healthy text_response exit must NOT produce any explanation."""
    out = AIAgent._format_turn_completion_explanation(
        "text_response(finish_reason=stop)"
    )
    assert out == ""


def test_explanation_quiet_for_empty_reason():
    assert AIAgent._format_turn_completion_explanation("") == ""
    assert AIAgent._format_turn_completion_explanation("unknown") == ""
    # guardrail_halt surfaces its own message; explainer stays out of the way.
    assert AIAgent._format_turn_completion_explanation("guardrail_halt") == ""






def test_explanation_for_max_iterations_reached_prefix_match():
    """``max_iterations_reached(...)`` carries a parenthetical suffix."""
    out = AIAgent._format_turn_completion_explanation(
        "max_iterations_reached(10/10)"
    )
    assert "iteration" in out.lower()






# --------------------------------------------------------------------------
# 1b. Cause-aware session-persistence wording
# --------------------------------------------------------------------------
def test_explanation_persistence_locked_cause_says_busy_not_disk():
    """Write-lock contention must NOT be misdiagnosed as a disk problem."""
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "locked"
    )
    lower = out.lower()
    assert "busy" in lower
    assert "saved" in lower
    assert "send it again" in lower
    assert "disk" not in lower
    assert "permission" not in lower


def test_explanation_persistence_compression_cause_is_specific():
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "compression"
    )
    lower = out.lower()
    assert "compression" in lower
    assert "database" not in lower
    assert "disk" not in lower


def test_explanation_persistence_turn_lease_cause_is_specific():
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "turn_lease"
    )
    lower = out.lower()
    assert "took over" in lower
    assert "not saved" in lower
    assert "disk" not in lower
    assert "compression" not in lower
    assert "hermes doctor" not in lower


def test_explanation_persistence_disk_cause_keeps_disk_wording():
    out = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "disk"
    )
    lower = out.lower()
    assert "disk" in lower
    assert "free some space" in lower or "disk space" in lower


def test_explanation_persistence_unknown_cause_is_neutral():
    """None/'unknown' cause must not claim disk-full — point at diagnostics."""
    for cause in (None, "unknown"):
        out = AIAgent._format_turn_completion_explanation(
            "session_persistence_failed", cause
        )
        lower = out.lower()
        assert out.strip() != ""
        assert "disk space" not in lower
        assert "full disk" not in lower
        assert "hermes doctor" in lower
        assert "again" in lower


def test_explanation_persistence_one_arg_backward_compat():
    """Existing one-arg callers must keep working (optional second param)."""
    out = AIAgent._format_turn_completion_explanation("session_persistence_failed")
    assert out.strip() != ""
    assert "session storage" in out.lower()


def test_explanation_cause_ignored_for_other_reasons():
    """The cause parameter must not perturb non-persistence reasons."""
    assert (
        AIAgent._format_turn_completion_explanation(
            "text_response(finish_reason=stop)", "locked"
        )
        == ""
    )
    out = AIAgent._format_turn_completion_explanation(
        "max_iterations_reached(10/10)", "locked"
    )
    assert "iteration" in out.lower()


# --------------------------------------------------------------------------
# 1c. classify_persistence_error — the pure cause classifier
# --------------------------------------------------------------------------
def test_classify_persistence_error_categories():
    import sqlite3

    from hermes_state import classify_persistence_error

    assert classify_persistence_error(
        sqlite3.OperationalError("database is locked")
    ) == "locked"
    assert classify_persistence_error("SQLITE_BUSY: busy") == "locked"
    assert classify_persistence_error(
        sqlite3.OperationalError("database or disk is full")
    ) == "disk"
    assert classify_persistence_error("attempt to write a readonly database") == "disk"
    assert classify_persistence_error("read-only file system") == "disk"
    assert classify_persistence_error("no space left on device") == "disk"
    assert classify_persistence_error("disk I/O error") == "disk"
    assert classify_persistence_error("something else entirely") == "unknown"
    assert classify_persistence_error(None) == "unknown"
    assert classify_persistence_error("") == "unknown"


def test_classify_persistence_error_reuses_disk_full_markers():
    """The disk bucket delegates to hermes_state.is_disk_full_error, so
    every marker that helper recognizes (ENOSPC, 'not enough space', ...)
    must classify as 'disk' — the two classifiers can never drift apart."""
    import errno

    from hermes_state import classify_persistence_error

    assert classify_persistence_error("ENOSPC writing state.db") == "disk"
    assert classify_persistence_error(
        "There is not enough space on the disk"
    ) == "disk"
    assert classify_persistence_error(
        OSError(errno.ENOSPC, "No space left on device")
    ) == "disk"


def test_classify_persistence_error_compression_busy_is_distinct():
    """A live compression lease refusing the write is contention, not
    storage damage — but its message contains neither 'locked' nor 'busy',
    so it must classify by exception type (and by phrase for RPC-wrapped
    strings). This is the exact failure mode of issue #81227."""
    from hermes_state import (
        CompressionSessionBusyError,
        SessionCompressionInProgressError,
    )
    from hermes_state import classify_persistence_error

    assert classify_persistence_error(
        SessionCompressionInProgressError(
            "Session 'abc' is being compressed by another writer"
        )
    ) == "compression"
    assert classify_persistence_error(
        CompressionSessionBusyError("Compression lease lost before publication: abc")
    ) == "compression"
    # RPC-wrapped string forms (exception type lost in transit).
    assert classify_persistence_error(
        "Session 'abc' is being compressed by another writer"
    ) == "compression"
    assert classify_persistence_error(
        "Compression lease lost before publication: abc"
    ) == "compression"


def test_classify_persistence_error_turn_lease_lost_is_distinct():
    from hermes_state import SessionTurnLeaseLostError, classify_persistence_error

    assert classify_persistence_error(
        SessionTurnLeaseLostError(
            "Session turn lease lost; refusing transcript write for 'abc'"
        )
    ) == "turn_lease"
    assert classify_persistence_error(
        "Session turn lease lost; refusing transcript write for 'abc'"
    ) == "turn_lease"


def test_persistence_error_causes_tuple_matches_classifier():
    """PERSISTENCE_ERROR_CAUSES must cover every value the classifier can
    return (consumers like cron suppression iterate it)."""
    from hermes_state import PERSISTENCE_ERROR_CAUSES, classify_persistence_error

    probes = (
        "database is locked",
        "Session 'abc' is being compressed by another writer",
        "Session turn lease lost; refusing transcript write for 'abc'",
        "database or disk is full",
        "something else entirely",
        None,
    )
    for probe in probes:
        assert classify_persistence_error(probe) in PERSISTENCE_ERROR_CAUSES


# --------------------------------------------------------------------------
# 2. Enable/disable seam
# --------------------------------------------------------------------------
def test_explainer_enabled_by_default():
    agent = _make_agent()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_TURN_COMPLETION_EXPLAINER", None)
        with patch("hermes_cli.config.load_config", return_value={}):
            assert agent._turn_completion_explainer_enabled() is True


def test_explainer_disabled_via_env():
    agent = _make_agent()
    with patch.dict(
        os.environ, {"HERMES_TURN_COMPLETION_EXPLAINER": "0"}, clear=False
    ):
        assert agent._turn_completion_explainer_enabled() is False


def test_explainer_config_read_once_then_cached():
    """Measured-work pin: the config lookup happens once per agent.

    The explainer gate runs at the end of every turn, so a fresh
    ``load_config()`` per call is wasted work (measured ~0.9 ms/call on a
    warm mtime-cache on this host; per-turn config reads were killed
    repo-wide in #74211, and this seam was missed).  The config read must
    be cached after the first call; the env-var override must still win on
    every call, cached or not.
    """
    agent = _make_agent()
    calls = {"n": 0}

    def counting_load():
        calls["n"] += 1
        return {"display": {"turn_completion_explainer": True}}

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_TURN_COMPLETION_EXPLAINER", None)
        with patch("hermes_cli.config.load_config", counting_load):
            # First call reads config and caches the result.
            assert agent._turn_completion_explainer_enabled() is True
            assert calls["n"] == 1
            # Subsequent calls must not re-read config.
            assert agent._turn_completion_explainer_enabled() is True
            assert agent._turn_completion_explainer_enabled() is True
            assert calls["n"] == 1
            # Env override stays authoritative even after the cache is warm.
            with patch.dict(
                os.environ, {"HERMES_TURN_COMPLETION_EXPLAINER": "0"}, clear=False
            ):
                assert agent._turn_completion_explainer_enabled() is False
            assert calls["n"] == 1  # env path never touches config




# --------------------------------------------------------------------------
# 3. End-to-end: empty-response exhaustion surfaces the explanation
# --------------------------------------------------------------------------
def test_run_conversation_empty_exhausted_surfaces_explanation():
    """Four empty responses in a row should exhaust retries and the final
    response should be the actionable explanation, not a bare '(empty)'."""
    agent = _make_agent(max_iterations=10)
    # 4 empty responses: retries 1..3 then the terminal on the 4th.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop") for _ in range(8)
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "empty_response_exhausted"
    # The user must NOT be left with a bare sentinel; the explanation wins.
    assert result["final_response"] != "(empty)"
    assert result["final_response"].strip() != ""
    assert "No reply:" in result["final_response"]


def test_run_conversation_partial_stream_recovery_surfaces_explanation():
    """A long recovered partial stream still needs the visible footer.

    Without this, the gateway marks the turn as previewed and suppresses
    the final send, leaving messaging users with a fragment and no reason.
    """
    agent = _make_agent(max_iterations=10)
    empty_stub = _mock_response(content=None, finish_reason="stop")
    recovered = (
        "I inspected the running gateway and found that the current turn "
        "stopped after the provider stream timed out."
    )

    def _fake_api_call(_api_kwargs):
        agent._current_streamed_assistant_text = recovered
        return empty_stub

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do something")

    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert result["final_response"].startswith(recovered)
    assert "No reply:" in result["final_response"]
    assert result["response_previewed"] is False


