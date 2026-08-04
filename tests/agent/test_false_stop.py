"""False-stop detection: gating, detector signatures, budget, transcript hygiene.

When the model produces finish_reason=stop with text indicating intent to
continue (a colon-preamble that lost its tool_calls, or a narrated
continuation) right after a tool round, the conversation loop injects a
bounded synthetic nudge instead of silently ending the turn (#42503).

This file verifies:

  - Gating (env / config / surface-aware "auto") matches the verify-on-stop
    precedence contract.
  - Signature A (short colon-preamble after a tool round) triggers only in a
    tool round and only under the strict <120 length bound.
  - Signature B (CJK-ending narrated continuation) triggers only with an
    intent marker and is suppressed by done/wait/question markers.
  - The nudge budget is bounded at 2 per turn.
  - The synthetic nudge is registered as ephemeral scaffolding: DB flush and
    JSON log drop only the nudge and keep the assistant candidate.
"""

import json
import sys
from unittest.mock import MagicMock

import pytest

from agent.false_stop import (
    build_false_stop_nudge,
    false_stop_detection_enabled,
)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_false_stop_env(monkeypatch):
    """Clear every env signal false_stop_detection_enabled consults."""
    for var in (
        "HERMES_FALSE_STOP_DETECTION",
        "HERMES_PLATFORM",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_env_can_force_on(clear_false_stop_env):
    clear_false_stop_env.setenv("HERMES_FALSE_STOP_DETECTION", "1")
    clear_false_stop_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert false_stop_detection_enabled({"agent": {}}) is True


def test_env_can_force_off(clear_false_stop_env):
    clear_false_stop_env.setenv("HERMES_FALSE_STOP_DETECTION", "0")
    clear_false_stop_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert false_stop_detection_enabled({"agent": {}}) is False


@pytest.mark.parametrize("source", ["cli", "tui", "desktop", "codex", "local"])
def test_auto_on_for_interactive_surfaces(clear_false_stop_env, source):
    clear_false_stop_env.setenv("HERMES_SESSION_SOURCE", source)
    assert false_stop_detection_enabled({"agent": {"false_stop_detection": "auto"}}) is True


@pytest.mark.parametrize("platform", ["telegram", "discord", "slack", "whatsapp", "signal"])
def test_auto_off_for_messaging_surfaces(clear_false_stop_env, platform):
    clear_false_stop_env.setenv("HERMES_SESSION_PLATFORM", platform)
    assert false_stop_detection_enabled({"agent": {"false_stop_detection": "auto"}}) is False


def test_config_true_forces_on(clear_false_stop_env):
    clear_false_stop_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert false_stop_detection_enabled({"agent": {"false_stop_detection": True}}) is True


def test_config_false_forces_off(clear_false_stop_env):
    clear_false_stop_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert false_stop_detection_enabled({"agent": {"false_stop_detection": False}}) is False


def test_auto_default_path_through_load_config(tmp_path, clear_false_stop_env):
    # E2E: the production caller passes no config, so false_stop_detection_enabled
    # resolves through load_config() + DEFAULT_CONFIG, whose default is the
    # surface-aware "auto" sentinel.
    clear_false_stop_env.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli.config import load_config

    merged = load_config()
    assert merged["agent"]["false_stop_detection"] == "auto"

    # Interactive surface resolves ON through the real loader.
    clear_false_stop_env.setenv("HERMES_SESSION_SOURCE", "cli")
    assert false_stop_detection_enabled() is True

    # A messaging platform resolves OFF.
    clear_false_stop_env.setenv("HERMES_SESSION_PLATFORM", "telegram")
    assert false_stop_detection_enabled() is False


# ---------------------------------------------------------------------------
# Detector fixtures
# ---------------------------------------------------------------------------


def _tool_round_messages():
    """Messages where a recent assistant message carried tool_calls."""
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "reading files", "tool_calls": [{"id": "tc1", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
    ]


def _no_tool_round_messages():
    """Messages with no prior assistant tool_calls."""
    return [
        {"role": "user", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# Signature A: short colon-preamble (lost tool-call narration)
# ---------------------------------------------------------------------------


def test_signature_a_colon_preamble_in_tool_round():
    """Short colon-preamble after a tool round triggers a nudge."""
    content = "接下来读取文件："  # ends with fullwidth colon, < 120 chars
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is not None
    assert "tool call" in nudge.lower()


def test_signature_a_not_in_tool_round():
    """Colon-preamble without prior tool round does NOT trigger."""
    content = "清单如下："  # standalone answer, no tool round
    nudge = build_false_stop_nudge(content, _no_tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_a_ascii_colon():
    """ASCII colon also triggers Signature A."""
    content = "next, reading the file:"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is not None


@pytest.mark.parametrize("length,should_trigger", [(119, True), (120, False), (121, False)])
def test_signature_a_length_boundary(length, should_trigger):
    """Content of exactly 120 chars does NOT trigger (uses <, not <=)."""
    content = "a" * (length - 1) + "："
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert (nudge is not None) == should_trigger


# ---------------------------------------------------------------------------
# Signature B: narrated continuation with intent markers
# ---------------------------------------------------------------------------


def test_signature_b_cjk_positive():
    """CJK intent marker + tool round + <=200 chars + CJK ending triggers."""
    content = "继续读取剩余的文件。"  # intent marker "继续", ends with "。", < 200 chars
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is not None


def test_signature_b_english_positive():
    """English intent marker with CJK ending triggers."""
    content = "let me continue reading the rest。"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is not None


@pytest.mark.parametrize("length,should_trigger", [(199, True), (200, True), (201, False)])
def test_signature_b_length_boundary(length, should_trigger):
    """Content of exactly 200 chars DOES trigger (uses <=). 201 does not."""
    # Build content with intent marker + CJK ending
    prefix = "继续"
    suffix = "。"
    middle = "x" * (length - len(prefix) - len(suffix))
    content = prefix + middle + suffix
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert (nudge is not None) == should_trigger


def test_signature_b_done_marker_exclusion():
    """Done markers suppress the nudge even with intent markers."""
    content = "已完成下一步操作。"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_b_wait_marker_exclusion():
    """Wait markers suppress the nudge."""
    content = "继续等待后台任务完成。"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_b_question_marker_exclusion():
    """Question markers suppress the nudge."""
    content = "你决定下一步如何处理。"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_b_not_in_tool_round():
    """Signature B without prior tool round does NOT trigger."""
    content = "继续读取剩余的文件。"
    nudge = build_false_stop_nudge(content, _no_tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_b_no_intent_marker():
    """Content with CJK ending but no intent marker does NOT trigger."""
    content = "天气很好今天。"
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is None


def test_signature_b_english_period_not_cjk_ending():
    """Content ending with ASCII '.' does NOT trigger Signature B."""
    content = "let me continue reading the rest."
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is None


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_exhausted_no_nudge():
    """When attempts >= 2 (max), no nudge is returned."""
    content = "接下来读取文件："
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=2)
    assert nudge is None


def test_budget_one_nudge_remaining():
    """At attempt 1 (of max 2), one more nudge is allowed."""
    content = "接下来读取文件："
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=1)
    assert nudge is not None


def test_budget_zero_nudges_used():
    """At attempt 0, nudges are available."""
    content = "接下来读取文件："
    nudge = build_false_stop_nudge(content, _tool_round_messages(), attempts=0)
    assert nudge is not None


def test_empty_content_no_nudge():
    """Empty content never triggers."""
    assert build_false_stop_nudge("", _tool_round_messages(), attempts=0) is None
    assert build_false_stop_nudge("   ", _tool_round_messages(), attempts=0) is None
    assert build_false_stop_nudge(None, _tool_round_messages(), attempts=0) is None


# ---------------------------------------------------------------------------
# Transcript hygiene: the synthetic nudge is ephemeral scaffolding; the
# assistant candidate persists. Mirrors test_verification_stop_caching.py.
# ---------------------------------------------------------------------------


def _fresh_run_agent(hermes_home):
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    import run_agent  # noqa: F401
    return sys.modules["run_agent"]


def test_false_stop_flag_registered_as_ephemeral(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)

    assert "_false_stop_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS

    # The nudge message IS scaffolding (carries the synthetic flag).
    assert ra._is_ephemeral_scaffolding(
        {"role": "user", "content": "issue the tool call", "_false_stop_synthetic": True}
    )
    # Real messages (including the assistant candidate) are NOT.
    assert not ra._is_ephemeral_scaffolding({"role": "user", "content": "hi"})
    assert not ra._is_ephemeral_scaffolding(
        {"role": "assistant", "content": "premature stop"}
    )


def _make_agent(ra, session_id, tmp_path):
    agent = ra.AIAgent(
        session_id=session_id,
        api_key="test-key",
        base_url="http://127.0.0.1:8000/v1",
        provider="openai-compat",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path / "logs"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    return agent


def test_db_flush_drops_only_nudge_keeps_candidate(tmp_path, monkeypatch):
    """The assistant candidate is NOT flagged synthetic, so it persists.
    Only the nudge (flagged synthetic) is dropped from the DB flush."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)
    agent = _make_agent(ra, "sess_db", tmp_path)

    messages = [
        {"role": "user", "content": "do the task"},
        # Assistant candidate — NOT flagged synthetic, persists.
        {"role": "assistant", "content": "premature stop"},
        # Nudge — flagged synthetic, gets dropped.
        {"role": "user", "content": "issue the tool call now", "_false_stop_synthetic": True},
        {"role": "assistant", "content": "actual tool call result"},
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    persisted = [
        msg.get("content")
        for _args, kwargs in agent._session_db.append_messages_batch.call_args_list
        for msg in kwargs["messages"]
    ]
    assert "do the task" in persisted
    assert "actual tool call result" in persisted
    # The assistant candidate persists — it is real content.
    assert "premature stop" in persisted
    # Only the nudge is dropped.
    assert "issue the tool call now" not in persisted


def test_json_log_drops_only_nudge_keeps_candidate(tmp_path, monkeypatch):
    """The assistant candidate is NOT flagged synthetic, so it persists in the
    JSON log. Only the nudge (flagged synthetic) is dropped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)
    agent = _make_agent(ra, "sess_json", tmp_path)

    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "premature stop"},
        {"role": "user", "content": "issue the tool call now", "_false_stop_synthetic": True},
        {"role": "assistant", "content": "actual tool call result"},
    ]

    agent._save_session_log(messages)

    log_file = agent.logs_dir / "session_sess_json.json"
    assert log_file.exists()
    data = json.loads(log_file.read_text(encoding="utf-8"))
    contents = [m.get("content") for m in data["messages"]]
    assert "premature stop" in contents
    assert "actual tool call result" in contents
    assert "do the task" in contents
    assert "issue the tool call now" not in contents
    assert all(not m.get("_false_stop_synthetic") for m in data["messages"])
