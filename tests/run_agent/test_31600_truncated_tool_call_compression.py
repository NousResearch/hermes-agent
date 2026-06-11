"""Regression test for #31600 — truncated-tool-call compression escape hatch.

Before the fix, when a high-context model (e.g. 1M-token Gemini) produced
a tool call with truncated JSON arguments repeatedly, the agent
returned ``error: "Response truncated due to output length limit"`` and
bailed.  The background review agent would then retry the same turn next
cycle with even more context, looping forever — the compression
threshold sits far above the point at which the model's structured
output starts truncating, so the existing compression triggers
(``context length`` error from the API) never fire.

After the fix, once the truncated-tool-call *retry* budget is exhausted
(``main`` retries the API call 3× via the ``truncated_tool_call_retries
< 3`` gate, boosting ``max_tokens`` each time), the next truncated
response triggers ``_compress_context`` before giving up.  Compression
shrinks the prompt below the threshold, the next API call gets clean
output, and the deadlock is broken.

Counter semantics (see ``agent/conversation_loop.py`` truncation handler,
gate at the ``truncated_tool_call_retries < 3`` check):
``truncated_tool_call_retries`` starts at 0 and is only reset after a
successful tool execution.  Each truncated tool-call response while
``retries < 3`` increments the counter and re-runs the same API call
(``continue``).  So truncations #1/#2/#3 consume the 3-retry budget
(counter 0→1→2→3) and truncation #4 (counter already 3, ``3 < 3`` is
False) falls through to the compression escape hatch.  The tests
therefore feed FOUR consecutive truncated responses before the
recovered/clean one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    """Short-circuit the 2s time.sleep between compression retries."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _truncated_tool_call_response(tool_name: str = "fact_store"):
    """A non-streaming response carrying a tool_call whose JSON arguments
    are truncated mid-string.  Mirrors what chat_completion_helpers
    produces when ``has_truncated_tool_args`` flips ``finish_reason`` to
    ``"length"`` for unrepairable mid-stream cuts (#31600 log line).
    """
    tc = SimpleNamespace(
        id="tc_truncated",
        type="function",
        function=SimpleNamespace(
            name=tool_name,
            # No closing quote, no closing brace — the exact shape of the
            # GLM/Gemini truncation in the issue report.
            arguments='{"action": "add", "category": "user_pref", "content": "Dieter',
        ),
    )
    msg = SimpleNamespace(
        content=None,
        tool_calls=[tc],
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="length")
    resp = SimpleNamespace(choices=[choice], model="gemini-3.5-flash")
    resp.usage = SimpleNamespace(prompt_tokens=60_000, completion_tokens=128, total_tokens=60_128)
    return resp


def _clean_response(text: str = "Done after compression"):
    msg = SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="gemini-3.5-flash")
    resp.usage = SimpleNamespace(prompt_tokens=10_000, completion_tokens=20, total_tokens=10_020)
    return resp


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("fact_store")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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
        a.tool_delay = 0
        a.save_trajectories = False
        # Force a large context so the per-percent compression threshold sits
        # well above 60K tokens — exactly the regime the issue describes.
        a.compression_enabled = True
        a.context_compressor.context_length = 1_000_000
        a.context_compressor.threshold_tokens = 500_000
        return a


class TestTruncatedToolCallTriggersCompression:
    """When repeated truncation looks like the deadlock #31600 describes,
    the agent must attempt compression instead of bailing."""

    def test_truncated_tool_call_triggers_compress_context_after_retries(self, agent):
        """Truncated tool-call responses past the retry budget → compress + retry.

        ``main`` retries the API call 3× on truncated tool-call JSON
        (``truncated_tool_call_retries < 3`` gate, counter 0→1→2→3 on
        truncations #1/#2/#3).  Truncation #4 (counter already 3) is the
        one that, previously a hard return, must now call
        ``_compress_context`` and restart the API call against the
        compressed messages.  We feed FOUR truncations to exhaust the
        budget and reach the escape hatch — see module docstring for the
        full counter trace (tied to the ``< 3`` gate).
        """
        # Four consecutive truncations exhaust main's 3-retry budget; the
        # 4th reaches the compression escape hatch.  The 5th (recovered)
        # response is what the post-compression retry receives.
        truncated = [_truncated_tool_call_response() for _ in range(4)]
        recovered = _clean_response("All good after compression")
        agent.client.chat.completions.create.side_effect = [
            *truncated, recovered,
        ]

        prefill = [
            {"role": "user", "content": "remember my workstation name"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression shrinks 3 messages down to 1.
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation(
                "remember my workstation name is GOD",
                conversation_history=prefill,
            )

        # The whole point of the fix: compression must run at least once.
        assert mock_compress.call_count >= 1, (
            "Repeated truncated-tool-call responses must trigger "
            "_compress_context, not return 'Response truncated' immediately."
        )
        # And the loop must reach the clean post-compression response.
        assert result["completed"] is True
        assert result["final_response"] == "All good after compression"

    def test_compression_disabled_falls_back_to_old_behavior(self, agent):
        """When compression is disabled, exhausting the retry budget still
        returns a partial error — we must not regress this path or break
        agents running with ``compression_enabled = False``.

        ``main`` retries 3× (truncations #1/#2/#3), then truncation #4
        reaches the escape hatch.  With compression disabled the
        ``compression_enabled`` guard is False, so the code skips
        ``_compress_context`` and takes the give-up / "Response truncated"
        branch.  See module docstring for the counter trace.
        """
        agent.compression_enabled = False

        # Four consecutive truncations exhaust main's 3-retry budget and
        # reach the give-up branch.  Provide a couple of spare entries so a
        # regressed code path that loops one more time can't trip
        # StopIteration and mask the real assertion (we assert the give-up,
        # not an exception).
        truncated = [_truncated_tool_call_response() for _ in range(6)]
        agent.client.chat.completions.create.side_effect = truncated

        prefill = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        # With compression disabled, the old bail-out behavior must hold.
        mock_compress.assert_not_called()
        assert result.get("completed") is False
        assert result.get("partial") is True
        assert "truncated" in (result.get("error") or "").lower()
