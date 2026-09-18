"""Regression tests: overflow recovery re-renders an inline tool surface deferred.

#6465 reported a session carrying 53 MCP tools plus 17 built-in tools whose provider rejected
every request: the tool schemas did not fit the window, and compaction cannot shrink them (it
rewrites messages only). Every compression pass re-sent the same floor, so the turn ended at
"cannot compress further" with a tool surface that never got to run.

The fix re-renders such a surface through the progressive-disclosure bridge
(`tools/tool_search.py`) when the schemas alone are what overflows, and retries the turn. The
schemas leave the request, the session keeps every tool — the model loads one with
`tool_describe` and invokes it with `tool_call` — and nothing is changed when the transcript is
what does not fit (compaction owns that case).

Both tests drive a real ``AIAgent.run_conversation`` against a mocked client.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.model_metadata import _estimate_tools_tokens_rough, estimate_request_tokens_rough
from tools.registry import registry
from tools.tool_search import BRIDGE_TOOL_NAMES

# The window a small-window fallback (or a local 32K model) provides.
_WINDOW_TOKENS = 32_000
_CORE_TOOL_NAMES = ("read_file", "terminal", "web_search")
# 60 MCP servers' worth of definitions at ~850 tokens each: the payload class the report carried
# (53 MCP tools), and several times the 32K window on its own.
_MCP_TOOL_COUNT = 60
_MCP_DESCRIPTION_CHARS = 3_400
_MCP_PREFIX = "mcp__"
_MCP_SERVER = "overflow-probe"


def _tool_def(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _core_tool_defs() -> list:
    return [_tool_def(name, f"{name} tool") for name in _CORE_TOOL_NAMES]


def _mcp_tool_defs() -> list:
    filler = "MCP tool description. " * (_MCP_DESCRIPTION_CHARS // 23)
    return [_tool_def(f"{_MCP_PREFIX}{_MCP_SERVER}__tool_{i}", filler) for i in range(_MCP_TOOL_COUNT)]


def _register_mcp_tools() -> None:
    """Real registrations: deferral eligibility is resolved from the registry's toolset
    (``mcp-<server>``), exactly as ``tools/mcp_tool.py`` registers a discovered server."""
    for i in range(_MCP_TOOL_COUNT):
        name = f"{_MCP_PREFIX}{_MCP_SERVER}__tool_{i}"
        registry.register(
            name=name,
            toolset=f"mcp-{_MCP_SERVER}",
            schema=_tool_def(name, "MCP tool")["function"],
            handler=lambda args, **kwargs: "{}",
        )


def _tool_names(tools) -> list:
    return [(t.get("function") or {}).get("name") or t.get("name") or "" for t in tools or []]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], model="test/model")
    resp.usage = None
    return resp


def _context_overflow_error() -> Exception:
    """A 400 the classifier routes to ``context_overflow``; its window figure is higher than this
    test's window, so the provider limit is not adopted and the configured window stands."""
    err = Exception(
        "Error code: 400 - {'error': {'message': "
        "\"This endpoint's maximum context length is 128000 tokens. "
        "However, you requested about 200000 tokens. "
        "Please reduce the length of the messages.\"}}"
    )
    err.status_code = 400
    return err


@pytest.fixture()
def agent():
    _register_mcp_tools()
    defs = _core_tool_defs() + _mcp_tool_defs()
    with (
        patch("model_tools.get_tool_definitions", return_value=defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = True
    a.save_trajectories = False
    a.context_compressor.context_length = _WINDOW_TOKENS
    return a


def _input_budget(agent) -> int:
    """The window minus the reserved output space — the budget the deferral gate must respect."""
    return _WINDOW_TOKENS - (agent.max_tokens or 0)


def _run(agent, history):
    """Run one turn with compaction stubbed to a no-op (compaction is not the mechanism under
    test, and a real pass would call the auxiliary model)."""
    with (
        patch.object(agent, "_compress_context") as compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        compress.return_value = (history, "system prompt")
        result = agent.run_conversation("hello", conversation_history=history)
    return result, compress


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


def test_defers_inline_schemas_when_they_alone_do_not_fit(agent):
    """The tool floor is what overflows: the schemas move behind the bridge and the turn runs."""
    calls = agent.client.chat.completions.create
    calls.side_effect = [_context_overflow_error(), _mock_response(content="Recovered")]

    result, _compress = _run(agent, _prefill())

    assert calls.call_count >= 2, "the rejected request must be retried against the same route"
    first, last = calls.call_args_list[0].kwargs, calls.call_args_list[-1].kwargs

    # Premise, measured on the actual payloads: the inline schemas alone overflow the window, and
    # the request fits once they are behind the bridge.
    budget = _input_budget(agent)
    sent_full = estimate_request_tokens_rough(
        first["messages"], system_prompt=agent._cached_system_prompt, tools=agent.tools
    )
    assert sent_full > budget
    assert sent_full - _estimate_tools_tokens_rough(agent.tools) + _estimate_tools_tokens_rough(
        last["tools"]
    ) < budget

    # The rejected request carried the whole surface; the retry keeps the built-in tools and
    # swaps the MCP schemas for the bridge, so every deferred tool stays reachable.
    assert [n for n in _tool_names(first["tools"]) if n.startswith(_MCP_PREFIX)]
    retried = _tool_names(last["tools"])
    assert not [n for n in retried if n.startswith(_MCP_PREFIX)]
    assert set(_CORE_TOOL_NAMES) <= set(retried)
    assert BRIDGE_TOOL_NAMES <= set(retried)

    # agent.tools is untouched: the deferral is a request-level surface, not a toolset mutation —
    # the next turn starts from the configured surface again.
    assert [n for n in _tool_names(agent.tools) if n.startswith(_MCP_PREFIX)]

    assert result.get("failed") is not True
    assert result["final_response"] == "Recovered"


def test_does_not_defer_when_the_transcript_is_what_does_not_fit(agent):
    """Messages over the budget with the schemas deferred: compaction stays the only lever."""
    calls = agent.client.chat.completions.create
    calls.side_effect = [_context_overflow_error(), _mock_response(content="Recovered")]

    # ~50K tokens of transcript: deferring the schemas cannot bring this under a 32K window.
    history = [
        {"role": "user", "content": "x" * 200_000},
        {"role": "assistant", "content": "noted"},
    ]
    result, _compress = _run(agent, history)

    from agent.tool_surface_overflow import deferred_tool_surface

    assert deferred_tool_surface(agent) is None, "the transcript, not the schemas, is what overflows"
    for call in calls.call_args_list:
        sent = _tool_names(call.kwargs["tools"])
        assert [n for n in sent if n.startswith(_MCP_PREFIX)], (
            "a request the deferred surface cannot rescue must keep the configured surface"
        )
        assert not BRIDGE_TOOL_NAMES & set(sent)
    assert result.get("failed") is True
