"""Regression tests for the May 2026 xAI OAuth (SuperGrok / X Premium) bugs.

Three distinct failure modes the user community hit during rollout:

1. ``RuntimeError("Expected to have received `response.created` before
   `error`")`` on multi-turn xAI OAuth conversations.  The OpenAI SDK's
   Responses streaming state machine collapses an upstream ``error`` SSE
   frame into a generic stream-ordering error.  ``_run_codex_stream``
   now treats this the same way it already treats the missing
   ``response.completed`` postlude — fall back to a non-stream
   ``responses.create(stream=True)`` which surfaces the real provider
   error.  Also closes #8133 (``response.in_progress`` prelude on custom
   relays) and #14634 (``codex.rate_limits`` prelude on codex-lb).

2. The HTTP 403 entitlement error xAI returns when an OAuth token lacks
   SuperGrok / X Premium ("You have either run out of available
   resources or do not have an active Grok subscription") used to read
   as a confusing wall of JSON.  ``_summarize_api_error`` now appends a
   one-line hint pointing the user at https://grok.com and ``/model``.

3. Multi-turn replay of ``codex_reasoning_items`` (with
   ``encrypted_content``) is now suppressed for ``is_xai_responses=True``
   in ``_chat_messages_to_responses_input``.  xAI's OAuth/SuperGrok
   surface rejects replayed encrypted reasoning items; Grok still
   reasons natively each turn, so coherence rides on visible message
   text.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix A: prelude error fallback
# ---------------------------------------------------------------------------


def _make_codex_agent():
    """Build a minimal AIAgent wired for codex_responses streaming tests."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.x.ai/v1",
        model="grok-4.3",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "xai-oauth"
    agent._interrupt_requested = False
    return agent


@pytest.mark.parametrize(
    "prelude_event_type",
    [
        "error",                  # xAI OAuth multi-turn
        "codex.rate_limits",      # codex-lb relays (#14634)
        "response.in_progress",   # custom Responses relays (#8133)
    ],
)
def test_codex_stream_prelude_error_falls_back_to_create_stream(prelude_event_type):
    """The SDK's prelude RuntimeError must trigger the non-stream fallback.

    When the first SSE event isn't ``response.created``, openai-python
    raises RuntimeError before our event loop sees anything.  We must
    detect that, retry once, then fall back to ``create(stream=True)``
    which surfaces the real provider error or a real response.
    """
    agent = _make_codex_agent()

    prelude_error = RuntimeError(
        f"Expected to have received `response.created` before `{prelude_event_type}`"
    )

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = prelude_error

    fallback_response = SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="fallback ok")],
        )],
        status="completed",
    )

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response
    mock_fallback.assert_called_once_with({}, client=mock_client)


def test_codex_stream_prelude_error_retries_once_before_fallback():
    """The retry path must fire one extra stream attempt before falling back."""
    agent = _make_codex_agent()

    call_count = {"n": 0}

    def stream_side_effect(**kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "Expected to have received `response.created` before `error`"
        )

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = stream_side_effect

    fallback_response = SimpleNamespace(output=[], status="completed")
    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        agent._run_codex_stream({}, client=mock_client)

    # max_stream_retries=1 → one retry + final attempt → 2 stream calls,
    # THEN the fallback path runs.
    assert call_count["n"] == 2
    mock_fallback.assert_called_once()


def test_codex_stream_unrelated_runtimeerror_still_raises():
    """RuntimeErrors that aren't prelude/postlude shape must propagate."""
    agent = _make_codex_agent()

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = RuntimeError("something else broke")

    with patch.object(agent, "_run_codex_create_stream_fallback") as mock_fallback:
        with pytest.raises(RuntimeError, match="something else broke"):
            agent._run_codex_stream({}, client=mock_client)

    mock_fallback.assert_not_called()


def test_codex_stream_postlude_error_still_falls_back():
    """Existing ``response.completed`` fallback must not regress."""
    agent = _make_codex_agent()

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = RuntimeError(
        "Didn't receive a `response.completed` event."
    )

    fallback_response = SimpleNamespace(output=[], status="completed")
    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response
    mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Fix B: friendly entitlement message
# ---------------------------------------------------------------------------


def test_summarize_api_error_decorates_xai_entitlement_403():
    """xAI's OAuth 403 must end with the subscribe-or-switch hint."""
    from run_agent import AIAgent

    error = RuntimeError(
        "HTTP 403: Error code: 403 - {'code': 'The caller does not have permission "
        "to execute the specified operation', 'error': 'You have either run out of "
        "available resources or do not have an active Grok subscription. Manage "
        "subscriptions at https://grok.com'}"
    )
    summary = AIAgent._summarize_api_error(error)
    assert "do not have an active Grok subscription" in summary
    assert "SuperGrok" in summary
    assert "/model" in summary
    assert "https://grok.com" in summary


def test_summarize_api_error_decorates_xai_body_message():
    """SDK-style error with structured body must also get the hint."""
    from run_agent import AIAgent

    class _XaiErr(Exception):
        status_code = 403
        body = {
            "error": {
                "message": (
                    "You have either run out of available resources or do "
                    "not have an active Grok subscription. Manage at "
                    "https://grok.com"
                )
            }
        }

    summary = AIAgent._summarize_api_error(_XaiErr("403"))
    assert "HTTP 403" in summary
    assert "SuperGrok / X Premium" in summary


def test_summarize_api_error_idempotent_for_entitlement_hint():
    """Decorating twice must not double up the hint."""
    from run_agent import AIAgent

    raw = "HTTP 403: do not have an active Grok subscription"
    once = AIAgent._decorate_xai_entitlement_error(raw)
    twice = AIAgent._decorate_xai_entitlement_error(once)
    assert once == twice


def test_summarize_api_error_passes_through_unrelated_errors():
    """Non-xAI / non-entitlement errors must not be touched."""
    from run_agent import AIAgent

    error = RuntimeError("HTTP 500: upstream is sad")
    summary = AIAgent._summarize_api_error(error)
    assert "SuperGrok" not in summary
    assert "grok.com" not in summary
    assert "upstream is sad" in summary


# ---------------------------------------------------------------------------
# Fix C: reasoning replay gating for xai-oauth
# ---------------------------------------------------------------------------


def _assistant_msg_with_encrypted_reasoning(text="hi from grok", encrypted="enc_blob"):
    return {
        "role": "assistant",
        "content": text,
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "id": "rs_xai_001",
                "encrypted_content": encrypted,
                "summary": [],
            }
        ],
    }


def test_codex_reasoning_replay_default_includes_encrypted_content():
    """Native Codex backend (default) must still replay encrypted reasoning."""
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        _assistant_msg_with_encrypted_reasoning(),
        {"role": "user", "content": "what's your name?"},
    ]

    items = _chat_messages_to_responses_input(msgs)
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["encrypted_content"] == "enc_blob"


def test_codex_reasoning_replay_stripped_for_xai_oauth():
    """xAI OAuth surface must NOT receive replayed encrypted reasoning."""
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        _assistant_msg_with_encrypted_reasoning(),
        {"role": "user", "content": "what's your name?"},
    ]

    items = _chat_messages_to_responses_input(msgs, is_xai_responses=True)
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert reasoning == []

    # The assistant's visible text must still survive — coherence across
    # turns rides on the message text alone.
    assistant_items = [
        it for it in items
        if it.get("role") == "assistant" or it.get("type") == "message"
    ]
    assert assistant_items, "assistant message must still be present"


def test_codex_transport_xai_request_omits_encrypted_content_include():
    """Verify the xAI ``include`` array no longer requests encrypted reasoning."""
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hi"},
        ],
        tools=None,
        instructions="you are a helpful assistant",
        reasoning_config={"enabled": True, "effort": "medium"},
        is_xai_responses=True,
    )
    # Without this gate, xAI would echo back encrypted_content blobs we'd
    # then store in codex_reasoning_items and replay next turn — which is
    # exactly the multi-turn failure mode we're closing.
    assert kwargs["include"] == []


def test_codex_transport_xai_strips_replayed_reasoning_in_input():
    """End-to-end: build_kwargs on xai-oauth must strip prior reasoning."""
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model="grok-4.3",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _assistant_msg_with_encrypted_reasoning(text="hi from grok"),
            {"role": "user", "content": "what's your name?"},
        ],
        tools=None,
        instructions="sys",
        reasoning_config={"enabled": True, "effort": "medium"},
        is_xai_responses=True,
    )
    input_items = kwargs["input"]
    reasoning_items = [it for it in input_items if it.get("type") == "reasoning"]
    assert reasoning_items == []


def test_codex_transport_native_codex_still_replays_reasoning_in_input():
    """Regression guard: openai-codex must keep the existing replay path."""
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model="gpt-5-codex",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            _assistant_msg_with_encrypted_reasoning(text="hi from codex"),
            {"role": "user", "content": "next"},
        ],
        tools=None,
        instructions="sys",
        reasoning_config={"enabled": True, "effort": "medium"},
        is_xai_responses=False,
    )
    input_items = kwargs["input"]
    reasoning_items = [it for it in input_items if it.get("type") == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"] == "enc_blob"
    # Native Codex still asks for encrypted_content back.
    assert "reasoning.encrypted_content" in kwargs.get("include", [])
