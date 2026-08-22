"""Tests for opt-in tool-iteration budget signpost (agent.tool_loop_budget_warning).

Clones the run-budget wrap-up pattern: one-shot SYSTEM NOTICE appended to the
newest role:\"tool\" message when a *finite* iteration cap is nearly exhausted.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.iteration_budget import IterationBudget


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path, monkeypatch, config_body: str = "", **overrides):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, config_body)

    from run_agent import AIAgent

    kwargs = dict(
        model="gpt-5.5",
        provider="openai",
        api_key="sk-dummy",
        base_url="https://api.openai.com/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
        max_iterations=10,
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── normalization ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        (False, False),
        (True, True),
        (0, False),
        ("abc", False),
        (3, 3),
        ("5", 5),
    ],
)
def test_normalize_tool_loop_budget_warning(raw, expected):
    from agent.agent_init import _normalize_tool_loop_budget_warning

    assert _normalize_tool_loop_budget_warning(raw) == expected


# ── constructor / config plumbing ─────────────────────────────────────────


def test_warning_off_by_default(monkeypatch, tmp_path):
    agent = _make_agent(tmp_path, monkeypatch)
    assert agent.tool_loop_budget_warning is False
    assert agent._tool_loop_budget_wrapup_injected is False


def test_config_key_sets_warning_true(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body="agent:\n  tool_loop_budget_warning: true\n",
    )
    assert agent.tool_loop_budget_warning is True


def test_config_key_sets_warning_int(monkeypatch, tmp_path):
    agent = _make_agent(
        tmp_path,
        monkeypatch,
        config_body="agent:\n  tool_loop_budget_warning: 2\n",
    )
    assert agent.tool_loop_budget_warning == 2


# ── wrap-up injection ─────────────────────────────────────────────────────


class _StubAgent:
    def __init__(self, warning=False, max_total=10, used=0):
        self.tool_loop_budget_warning = warning
        self._tool_loop_budget_wrapup_injected = False
        self.iteration_budget = IterationBudget(max_total)
        self.iteration_budget._used = used
        # Production prepare uses agent.max_iterations for the fresh budget.
        self.max_iterations = max_total


def _tool_messages():
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result one"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "result two"},
    ]


def test_off_never_mutates_messages():
    from agent.conversation_loop import _maybe_inject_tool_loop_budget_wrapup

    agent = _StubAgent(warning=False, max_total=10, used=9)
    messages = _tool_messages()
    before = copy.deepcopy(messages)
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False
    assert messages == before


def test_unlimited_never_injects_even_if_warning_true():
    from agent.conversation_loop import _maybe_inject_tool_loop_budget_wrapup

    agent = _StubAgent(warning=True, max_total=sys.maxsize, used=sys.maxsize - 1)
    messages = _tool_messages()
    before = copy.deepcopy(messages)
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False
    assert messages == before
    assert agent._tool_loop_budget_wrapup_injected is False


def test_threshold_true_mode_injects_once_on_newest_tool():
    from agent.conversation_loop import (
        TOOL_LOOP_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_tool_loop_budget_wrapup,
    )

    # max=10, used=8 => 80% — should fire
    agent = _StubAgent(warning=True, max_total=10, used=8)
    messages = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is True
    assert agent._tool_loop_budget_wrapup_injected is True
    assert TOOL_LOOP_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]
    assert messages[-1]["content"].startswith("result two")
    assert messages[2]["content"] == "result one"
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]

    snapshot = copy.deepcopy(messages)
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False
    assert messages == snapshot
    assert messages[-1]["content"].count(TOOL_LOOP_BUDGET_WRAPUP_NOTICE) == 1


def test_threshold_true_mode_not_before_80_percent():
    from agent.conversation_loop import _maybe_inject_tool_loop_budget_wrapup

    agent = _StubAgent(warning=True, max_total=10, used=7)  # 70%
    messages = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False
    assert agent._tool_loop_budget_wrapup_injected is False


def test_threshold_remaining_int_mode():
    from agent.conversation_loop import (
        TOOL_LOOP_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_tool_loop_budget_wrapup,
    )

    # max=10, used=8 => remaining=2; warning=2 => fire
    agent = _StubAgent(warning=2, max_total=10, used=8)
    messages = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is True
    assert TOOL_LOOP_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]

    agent2 = _StubAgent(warning=2, max_total=10, used=7)  # remaining=3
    messages2 = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent2, messages2) is False


def test_retries_when_no_tool_message_yet():
    from agent.conversation_loop import (
        TOOL_LOOP_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_tool_loop_budget_wrapup,
    )

    agent = _StubAgent(warning=True, max_total=5, used=4)
    messages = [{"role": "user", "content": "do the task"}]
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False
    assert agent._tool_loop_budget_wrapup_injected is False

    messages += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
    ]
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is True
    assert TOOL_LOOP_BUDGET_WRAPUP_NOTICE in messages[-1]["content"]


def test_multimodal_tool_content():
    from agent.conversation_loop import (
        TOOL_LOOP_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_tool_loop_budget_wrapup,
    )

    agent = _StubAgent(warning=True, max_total=5, used=4)
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [{"type": "text", "text": "block"}],
        },
    ]
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is True
    blocks = messages[-1]["content"]
    assert blocks[0] == {"type": "text", "text": "block"}
    assert blocks[-1] == {"type": "text", "text": TOOL_LOOP_BUDGET_WRAPUP_NOTICE}


def test_latch_reset_each_turn():
    """After turn_context prepare resets the latch, a fresh turn can re-append."""
    from agent.conversation_loop import (
        TOOL_LOOP_BUDGET_WRAPUP_NOTICE,
        _maybe_inject_tool_loop_budget_wrapup,
    )

    agent = _StubAgent(warning=True, max_total=10, used=8)
    messages = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is True
    assert agent._tool_loop_budget_wrapup_injected is True
    # Still latched: no second inject on the same turn.
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages) is False

    # Mirror turn_context.build_turn_context prepare (production path):
    # fresh IterationBudget + one-shot latch reset each turn.
    agent.iteration_budget = IterationBudget(agent.max_iterations)
    agent.iteration_budget._used = 8
    agent._tool_loop_budget_wrapup_injected = False

    assert agent._tool_loop_budget_wrapup_injected is False
    messages2 = _tool_messages()
    assert _maybe_inject_tool_loop_budget_wrapup(agent, messages2) is True
    assert agent._tool_loop_budget_wrapup_injected is True
    assert TOOL_LOOP_BUDGET_WRAPUP_NOTICE in messages2[-1]["content"]
    assert messages2[-1]["content"].count(TOOL_LOOP_BUDGET_WRAPUP_NOTICE) == 1


# ── integration-lite ───────────────────────────────────────────────────────


def _mock_tool_call(name="web_search", arguments="{}", call_id="c1"):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, type="function", function=fn)


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def test_integration_notice_on_messages_before_final_call(monkeypatch, tmp_path):
    """With warning on and max_iterations=5, tool-call loop should see the
    notice on the newest tool content before the final text completion."""
    from agent.conversation_loop import TOOL_LOOP_BUDGET_WRAPUP_NOTICE
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, "agent:\n  tool_loop_budget_warning: true\n")

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=5,
            platform="cli",
        )

    agent.client = MagicMock()
    agent.tool_loop_budget_warning = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    agent.api_mode = "chat_completions"

    tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
    tool_resp = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
    text_resp = _mock_response(content="done from state", finish_reason="stop")

    responses = [tool_resp, tool_resp, tool_resp, tool_resp, text_resp]
    call_idx = {"i": 0}
    captured = []

    def _create(*args, **kwargs):
        msgs = kwargs.get("messages")
        if msgs is not None:
            captured.append(copy.deepcopy(msgs))
        i = call_idx["i"]
        call_idx["i"] += 1
        if i >= len(responses):
            return text_resp
        return responses[i]

    agent.client.chat.completions.create = _create

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do multi-step work")

    found = False
    for msgs in captured:
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") != "tool":
                continue
            content = m.get("content") or ""
            if isinstance(content, str) and TOOL_LOOP_BUDGET_WRAPUP_NOTICE in content:
                found = True
                break
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and TOOL_LOOP_BUDGET_WRAPUP_NOTICE in (
                        b.get("text") or ""
                    ):
                        found = True
                        break
        if found:
            break
    assert found, (
        f"Expected tool-loop budget notice in some create() messages; "
        f"captured {len(captured)} calls"
    )

    # Second turn: fresh budget (per-turn reset) can run on prior history.
    history = result.get("messages") or []
    captured.clear()
    call_idx["i"] = 0
    responses[:] = [text_resp]
    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result2 = agent.run_conversation(
            "continue",
            conversation_history=history if isinstance(history, list) else None,
        )
    assert result2 is not None
    assert call_idx["i"] >= 1
