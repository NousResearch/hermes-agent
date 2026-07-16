"""Regression: outer-loop failures must not leak raw HTML pages to users.

#3069 added ``_clean_error_message`` / ``_summarize_api_error`` and
b892ee2bc applied the summarizer to the non-retryable result path (covered
by ``test_nonretryable_error_html_summary.py``).  Two outer-loop paths in
``run_conversation`` still interpolated the raw exception string:

    * the synthetic ``role="tool"`` results injected for unanswered
      tool_call_ids after a tool-execution error, and
    * the "I apologize, but I encountered repeated errors" final response
      emitted near max_iterations.

Both strings reach user-visible surfaces — chat platforms deliver the final
response verbatim, and the synthetic tool results persist into session
history — so a provider/proxy HTML error page (e.g. a Cloudflare challenge)
must be collapsed before it lands in either.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


# A representative Cloudflare "managed challenge" body (same shape as the
# fixture in test_nonretryable_error_html_summary.py).  Padded so
# length-based assertions are meaningful.
_CLOUDFLARE_CHALLENGE_HTML = (
    "<!DOCTYPE html>\n<html>\n  <head>\n"
    '    <meta http-equiv="refresh" content="360"></head>\n'
    "  <body>\n    <div class=\"data\"><noscript>"
    "Enable JavaScript and cookies to continue</noscript>"
    "<script>(function(){window._cf_chl_opt = {cRay: 'a0ca002c4f91769c',"
    "cZone: 'chatgpt.com', cType: 'managed', "
    + ("md: '" + "x" * 4000 + "',")
    + "};})();</script></div>\n  </body>\n</html>\n"
)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent() -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            provider="openai",
            api_mode="chat_completions",
            model="gpt-5.5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="", finish_reason="tool_calls", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_outer_loop_html_error_sanitized_in_tool_results_and_final_response():
    """An HTML page raised during tool execution must be collapsed in both
    the injected tool results and the near-max-iterations final response."""
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.return_value = _mock_response(
        tool_calls=[tool_call]
    )
    # max_iterations=2 makes the first failed call already satisfy
    # ``api_call_count >= max_iterations - 1``, so one exception exercises
    # BOTH raw paths: the tool-result fill and the final response.
    agent.max_iterations = 2

    executed = {"count": 0}

    def _raise_html(*args, **kwargs):
        executed["count"] += 1
        raise Exception(_CLOUDFLARE_CHALLENGE_HTML)

    with (
        patch.object(agent, "_execute_tool_calls", side_effect=_raise_html),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search something please")

    # Guard against a vacuous pass: the HTML exception must actually have
    # fired inside the loop.
    assert agent.client.chat.completions.create.called
    assert executed["count"] == 1

    # Surface 1: the near-max-iterations final response.
    final = result.get("final_response") or ""
    assert "I apologize, but I encountered repeated errors:" in final
    assert "<html" not in final.lower()
    assert "<!doctype" not in final.lower()
    assert "_cf_chl_opt" not in final
    # _clean_error_message collapses HTML pages to this one-liner.
    assert "Service temporarily unavailable" in final
    assert len(final) < 500

    # Surface 2: the synthetic tool results injected for the unanswered
    # tool_call_id persist into history — they must be equally clean.
    messages = result.get("messages") or []
    injected = [
        m
        for m in messages
        if isinstance(m, dict)
        and m.get("role") == "tool"
        and str(m.get("content", "")).startswith("Error executing tool:")
    ]
    assert injected, "outer handler should fill the unanswered tool result"
    for m in injected:
        content = str(m.get("content", ""))
        assert "<html" not in content.lower()
        assert "<!doctype" not in content.lower()
        assert "_cf_chl_opt" not in content
        assert "Service temporarily unavailable" in content
        assert len(content) < 500
