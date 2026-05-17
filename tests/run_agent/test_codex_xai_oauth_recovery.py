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
# Fix B: surface xAI's entitlement body verbatim (no editorializing)
#
# The original PR #26644 appended a hint that led with "X Premium+ does NOT
# include xAI API access — only standalone SuperGrok subscribers can use this
# provider."  xAI announced on 2026-05-16 that X Premium subs now work in
# Hermes (https://x.ai/news/grok-hermes), making that hint actively wrong:
# a Premium+ user hitting a real entitlement issue (no Grok sub, wrong tier,
# exhausted quota) would be misdirected to switch subscriptions when their
# Premium sub is in fact valid.  We now surface xAI's own body text verbatim
# (which already says "Manage subscriptions at https://grok.com/?_s=usage")
# and leave the diagnosis to xAI's wording.
# ---------------------------------------------------------------------------


def test_summarize_api_error_surfaces_xai_entitlement_body_verbatim():
    """xAI's OAuth 403 body must surface as-is, with no Hermes-side hint."""
    from run_agent import AIAgent

    error = RuntimeError(
        "HTTP 403: Error code: 403 - {'code': 'The caller does not have permission "
        "to execute the specified operation', 'error': 'You have either run out of "
        "available resources or do not have an active Grok subscription. Manage "
        "subscriptions at https://grok.com'}"
    )
    summary = AIAgent._summarize_api_error(error)
    # xAI's own body text must reach the user — they need it to diagnose.
    assert "do not have an active Grok subscription" in summary
    # No stale claim that X Premium is incompatible with Hermes.
    assert "X Premium+ does NOT include" not in summary
    assert "standalone SuperGrok subscribers" not in summary


def test_summarize_api_error_xai_body_message_unwrapped():
    """SDK-style error with structured body surfaces the message cleanly."""
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
    assert "do not have an active Grok subscription" in summary
    # No editorializing on top of xAI's own wording.
    assert "X Premium+ does NOT include" not in summary


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


# ---------------------------------------------------------------------------
# Fix D: entitlement 403 must NOT trigger credential-pool refresh loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # The exact wire text RaidenTyler and Don Piedro captured.
        "You have either run out of available resources or do not have an "
        "active Grok subscription. Manage at https://grok.com",
        # Permission-style variant from the same 403 body.
        "The caller does not have permission to execute the specified "
        "operation for grok-4.3",
    ],
)
def test_is_entitlement_failure_matches_real_xai_bodies(message):
    from run_agent import AIAgent

    assert AIAgent._is_entitlement_failure(
        {"message": message, "reason": "permission_denied"},
        403,
    )


def test_is_entitlement_failure_false_for_status_other_than_401_403():
    """200/429/500 must never be classified as entitlement, even if body matches."""
    from run_agent import AIAgent

    body = {
        "message": "do not have an active Grok subscription",
    }
    assert not AIAgent._is_entitlement_failure(body, 500)
    assert not AIAgent._is_entitlement_failure(body, 429)
    assert not AIAgent._is_entitlement_failure(body, 200)


def test_is_entitlement_failure_false_for_unrelated_auth_errors():
    """A real auth failure (expired token, wrong key) must keep refreshing."""
    from run_agent import AIAgent

    # Generic Anthropic-style auth failure
    assert not AIAgent._is_entitlement_failure(
        {"message": "Invalid API key", "reason": "authentication_error"},
        401,
    )
    # OAuth token expired
    assert not AIAgent._is_entitlement_failure(
        {"message": "Token has expired", "reason": "unauthorized"},
        401,
    )
    # Empty context
    assert not AIAgent._is_entitlement_failure({}, 401)
    assert not AIAgent._is_entitlement_failure(None, 401)


def test_recover_with_credential_pool_skips_refresh_on_entitlement_403():
    """The recovery path must NOT call pool.try_refresh_current() on entitlement 403.

    Before the fix, an unsubscribed xAI OAuth account would burn the agent
    loop indefinitely: refresh → 403 → refresh → 403, infinitely.  With
    the entitlement guard, recovery returns False so the error surfaces
    normally with the friendly hint from _summarize_api_error.
    """
    from run_agent import AIAgent
    from agent.error_classifier import FailoverReason

    agent = _make_codex_agent()

    # Wire a fake credential pool that records refresh attempts.
    refresh_calls = {"n": 0}

    class _FakePool:
        def try_refresh_current(self):
            refresh_calls["n"] += 1
            return MagicMock(id="should_not_be_called")

        def mark_exhausted_and_rotate(self, **_kwargs):
            return None

        def has_available(self):
            return False

    agent._credential_pool = _FakePool()

    error_context = {
        "reason": "The caller does not have permission to execute the specified operation",
        "message": "You have either run out of available resources or do not have an "
                   "active Grok subscription. Manage at https://grok.com",
    }

    recovered, _retried_429 = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context=error_context,
    )

    assert recovered is False, "Entitlement 403 must surface, not silently recover"
    assert refresh_calls["n"] == 0, "try_refresh_current must NOT be called on entitlement 403"


def test_recover_with_credential_pool_still_refreshes_genuine_auth_failure():
    """Regression guard: legitimate auth errors must still trigger refresh."""
    from run_agent import AIAgent
    from agent.error_classifier import FailoverReason

    agent = _make_codex_agent()

    refresh_calls = {"n": 0}

    class _FakePool:
        def try_refresh_current(self):
            refresh_calls["n"] += 1
            # Return a fake refreshed entry — semantically "refresh worked"
            entry = MagicMock()
            entry.id = "entry_refreshed"
            return entry

        def mark_exhausted_and_rotate(self, **_kwargs):
            return None

        def has_available(self):
            return False

    agent._credential_pool = _FakePool()
    # _swap_credential is called by the recovery path — stub it out
    agent._swap_credential = MagicMock()

    error_context = {
        "reason": "authentication_error",
        "message": "Invalid API key",
    }

    recovered, _retried_429 = agent._recover_with_credential_pool(
        status_code=401,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context=error_context,
    )

    assert recovered is True, "Genuine auth failure must still recover via refresh"
    assert refresh_calls["n"] == 1


# ---------------------------------------------------------------------------
# Fix E: grok-4.3 context length must be 1M, not 256K
# ---------------------------------------------------------------------------


def test_grok_4_3_context_length_is_1m():
    """grok-4.3 ships with 1M context per docs.x.ai/developers/models/grok-4.3.

    Hermes' substring-match fallback used to return 256k (from the
    "grok-4" catch-all) which under-reported the model's real capacity.
    """
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

    # The entry exists with the expected value.
    assert DEFAULT_CONTEXT_LENGTHS["grok-4.3"] == 1_000_000

    # And longest-first substring matching resolves grok-4.3 and
    # grok-4.3-latest to the new value, NOT the grok-4 catch-all.
    for slug in ("grok-4.3", "grok-4.3-latest"):
        matched_key = max(
            (k for k in DEFAULT_CONTEXT_LENGTHS if k in slug.lower()),
            key=len,
        )
        assert matched_key == "grok-4.3", (
            f"Expected longest-first match to land on grok-4.3 for {slug}, "
            f"got {matched_key}"
        )
        assert DEFAULT_CONTEXT_LENGTHS[matched_key] == 1_000_000


def test_grok_4_still_resolves_to_256k():
    """Regression guard: grok-4 (non-.3) must still resolve to 256k."""
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

    for slug in ("grok-4", "grok-4-0709"):
        matched_key = max(
            (k for k in DEFAULT_CONTEXT_LENGTHS if k in slug.lower()),
            key=len,
        )
        # grok-4-0709 contains "grok-4" but not "grok-4.3"; matched key
        # must be "grok-4" (or a more specific variant family if one is
        # ever added).  The 256k contract must hold.
        assert DEFAULT_CONTEXT_LENGTHS[matched_key] == 256_000
