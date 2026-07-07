"""Regression test: MoA reference models must not receive tool definitions.

Reference models are advisory only and should never attempt to call tools.
The system prompt (_REFERENCE_SYSTEM_PROMPT) instructs them not to, but
the fix also explicitly passes tools=None to prevent providers from
sending tool definitions in the first place.

Issue: #60272
"""

from types import SimpleNamespace

import pytest


def _response(content="advice", tool_calls=None):
    """Fake LLM response for testing."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


@pytest.fixture
def captured_reference_calls(monkeypatch):
    """Capture all call_llm invocations made by _run_reference."""
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response()

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    return calls


def test_reference_models_receive_no_tools(captured_reference_calls, monkeypatch):
    """MoA reference model calls must pass tools=None to prevent tool calls."""
    from agent import moa_loop

    # Mock _slot_runtime to return a valid runtime (non-anthropic to avoid cache control)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "openai",
            "model": "gpt-5.5",
            "base_url": "",
            "api_mode": "openai_chat",
        },
    )

    # Mock _extract_text to return content from response
    monkeypatch.setattr(moa_loop, "_extract_text", lambda r: r.choices[0].message.content)

    # Call _run_reference with a minimal slot and message
    slot = {"provider": "openai", "model": "gpt-5.5"}
    ref_messages = [{"role": "user", "content": "what should I do?"}]

    label, text, usage = moa_loop._run_reference(slot, ref_messages)

    # Verify call_llm was invoked exactly once
    assert len(captured_reference_calls) == 1, "Expected exactly one call_llm invocation"

    # Verify tools=None was passed
    call_kwargs = captured_reference_calls[0]
    assert "tools" in call_kwargs, "tools parameter must be present"
    assert call_kwargs["tools"] is None, "tools must be explicitly None to prevent tool calls"

    # Verify the call had the expected task marker
    assert call_kwargs.get("task") == "moa_reference", "Task marker must be moa_reference"

    # Verify system prompt was prepended
    assert call_kwargs["messages"][0]["role"] == "system"
    system_content = call_kwargs["messages"][0]["content"]
    # For non-Anthropic routes, content is a string
    if isinstance(system_content, str):
        assert "Mixture of Agents" in system_content
        assert "NOT the acting agent" in system_content
    else:
        # For Anthropic routes, content is a list of blocks
        assert any("Mixture of Agents" in block.get("text", "") for block in system_content if isinstance(block, dict))
        assert any("NOT the acting agent" in block.get("text", "") for block in system_content if isinstance(block, dict))
