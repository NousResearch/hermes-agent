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
   ``encrypted_content``) was briefly suppressed for ``is_xai_responses``
   in PR #26644 on the theory that xAI's OAuth/SuperGrok surface
   rejected replayed encrypted reasoning items.  That suppression was
   reverted shortly after: xAI confirmed they explicitly want Hermes to
   thread encrypted reasoning back across turns, and the original
   multi-turn failure mode was actually the prelude-SSE issue closed by
   Fix A above.  The remaining tests here lock in that xAI receives
   replayed reasoning AND that we ask xAI to echo it back in the
   ``include`` array.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fix A: prelude error surfacing via wire `error` events
#
# With the migration to ``responses.create(stream=True)`` raw event iteration,
# the SDK's high-level state-machine RuntimeError no longer mediates between
# the wire and us — we read the wire directly.  When the chatgpt.com Codex
# backend (or xAI, codex-lb, custom relays) emits a ``type=error`` frame as
# its first event, our consumer raises ``_StreamErrorEvent`` straight from
# the wire payload, which carries the real provider message in ``.body`` /
# ``.message`` shape for ``_summarize_api_error`` to consume.  This is
# strictly better than the old "SDK raises RuntimeError → we retry → fall
# back to a second non-stream call" two-phase dance, because the error
# surfaces on the first event instead of after one wasted round trip.
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
    "provider_message",
    [
        "You do not have an active Grok subscription",
        "rate limit exceeded",
        "model not available",
    ],
)
def test_codex_stream_wire_error_event_surfaces_stream_error_event(provider_message):
    """A wire ``type=error`` SSE frame raises ``_StreamErrorEvent`` with the
    provider's real message in the body."""
    from run_agent import _StreamErrorEvent

    agent = _make_codex_agent()

    class _ErrorCreateStream:
        def __iter__(self_inner):
            yield SimpleNamespace(type="error", message=provider_message, code="forbidden")

        def close(self_inner):
            pass

    mock_client = MagicMock()
    mock_client.responses.create.return_value = _ErrorCreateStream()

    with pytest.raises(_StreamErrorEvent) as excinfo:
        agent._run_codex_stream({}, client=mock_client)

    assert provider_message in str(excinfo.value)
    assert excinfo.value.body["error"]["message"] == provider_message


def test_codex_stream_retries_remote_protocol_error_once():
    """Transport errors (``httpx.RemoteProtocolError``) trigger a single retry.

    Previously this was on the ``responses.stream(...)`` helper; now it's on
    ``responses.create(stream=True)`` itself.  The user-facing behavior is the
    same: one retry, then re-raise if the second attempt also fails.
    """
    import httpx

    agent = _make_codex_agent()
    call_count = {"n": 0}

    def create_side_effect(**kwargs):
        call_count["n"] += 1
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = create_side_effect

    with pytest.raises(httpx.RemoteProtocolError):
        agent._run_codex_stream({}, client=mock_client)

    # max_stream_retries=1 → one retry + final attempt → 2 create calls total.
    assert call_count["n"] == 2


def test_codex_stream_unrelated_runtimeerror_still_raises():
    """RuntimeErrors that aren't transport errors must propagate.

    With the event-driven path there's no separate fallback function to
    short-circuit into; any RuntimeError from ``responses.create()`` or the
    consumer surfaces directly.
    """
    agent = _make_codex_agent()

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = RuntimeError("something else broke")

    with pytest.raises(RuntimeError, match="something else broke"):
        agent._run_codex_stream({}, client=mock_client)


def test_codex_stream_truncated_no_terminal_event_raises():
    """Streams that end without a terminal event AND no items raise.

    Preserves the "Codex Responses stream did not emit a terminal response"
    signal callers use to distinguish "stream truncated mid-flight" from
    "stream completed with empty body".  Previously surfaced by the SDK's
    ``RuntimeError("Didn't receive a `response.completed` event.")``; now
    surfaced directly by the event consumer.
    """
    agent = _make_codex_agent()

    class _EmptyStream:
        def __iter__(self_inner):
            return iter(())

        def close(self_inner):
            pass

    mock_client = MagicMock()
    mock_client.responses.create.return_value = _EmptyStream()

    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        agent._run_codex_stream({}, client=mock_client)


# ---------------------------------------------------------------------------
# Fix B: friendly entitlement message
# ---------------------------------------------------------------------------


def test_summarize_api_error_decorates_xai_entitlement_403():
    """xAI's OAuth 403 must surface the X Premium+ gotcha + neutral causes.

    Wording deliberately leads with the X Premium+ gotcha because that's
    the #1 confusing case: people see Grok in their X app, assume it
    works here too, and hit this 403 with no idea API access is a
    separate SKU.  Other causes (no subscription, wrong tier, exhausted
    quota) follow.
    """
    from run_agent import AIAgent

    error = RuntimeError(
        "HTTP 403: Error code: 403 - {'code': 'The caller does not have permission "
        "to execute the specified operation', 'error': 'You have either run out of "
        "available resources or do not have an active Grok subscription. Manage "
        "subscriptions at https://grok.com'}"
    )
    summary = AIAgent._summarize_api_error(error)
    # The original xAI text must survive — it's still useful diagnostic info.
    assert "do not have an active Grok subscription" in summary
    # The hint MUST lead with the X Premium+ gotcha (most likely cause
    # for users who think they're subscribed).
    assert "X Premium+ does NOT include" in summary
    assert "standalone SuperGrok subscribers" in summary
    # Other causes still listed.
    assert "no Grok subscription" in summary
    assert "tier doesn't include this model" in summary
    assert "quota is exhausted" in summary
    # The hint must point at the usage page where the user can verify.
    assert "https://grok.com/?_s=usage" in summary
    # Switching providers is still a valid escape hatch.
    assert "/model" in summary


def test_summarize_api_error_does_not_accuse_subscribers():
    """Hint must not confidently say the user has no subscription.

    Don Piedro reported his subscription is active. The hint must not
    contradict him — leading with the X Premium+ gotcha gives subscribers
    a plausible reason ("oh, I'm on Premium+ not pure SuperGrok") instead
    of accusing them of lying about having a subscription.
    """
    from run_agent import AIAgent

    error = RuntimeError(
        "HTTP 403: do not have an active Grok subscription"
    )
    summary = AIAgent._summarize_api_error(error)
    # MUST NOT contain language that flatly assumes the user is unsubscribed.
    assert "lacks SuperGrok" not in summary
    assert "you are not subscribed" not in summary.lower()
    # MUST lead with the most-likely-but-non-accusatory cause.
    assert "X Premium+ does NOT include" in summary


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
    assert "X Premium+ does NOT include" in summary


def test_summarize_api_error_idempotent_for_entitlement_hint():
    """Decorating twice must not double up the hint."""
    from run_agent import AIAgent

    raw = "HTTP 403: do not have an active Grok subscription"
    once = AIAgent._decorate_xai_entitlement_error(raw)
    twice = AIAgent._decorate_xai_entitlement_error(once)
    assert once == twice
    # Sanity: the hint did fire on the first pass.
    assert "X Premium+ does NOT include" in once


def test_summarize_api_error_passes_through_unrelated_errors():
    """Non-xAI / non-entitlement errors must not be touched."""
    from run_agent import AIAgent

    error = RuntimeError("HTTP 500: upstream is sad")
    summary = AIAgent._summarize_api_error(error)
    assert "SuperGrok" not in summary
    assert "grok.com" not in summary
    assert "upstream is sad" in summary


# ---------------------------------------------------------------------------
# Fix D: _StreamErrorEvent xAI entitlement classified as auth, not retryable
#
# run_codex_create_stream_fallback raises _StreamErrorEvent (status_code=None)
# when the Responses stream emits a ``type=error`` SSE frame.  Before this
# fix, classify_api_error had no match for "grok subscription" in its pattern
# lists, so it returned FailoverReason.unknown (retryable=True) — burning
# max_retries before the agent stopped.  _is_entitlement_failure was never
# called because it only runs when FailoverReason.auth is returned.
# ---------------------------------------------------------------------------


def test_classify_api_error_stream_event_grok_subscription_is_auth():
    """_StreamErrorEvent with xAI subscription message classifies as auth/non-retryable.

    The SSE error path has status_code=None, so _classify_by_status is
    skipped.  The explicit pattern added at step 1 must fire first and
    return auth/non-retryable so _is_entitlement_failure can stop the loop.
    """
    from run_agent import _StreamErrorEvent
    from agent.error_classifier import classify_api_error, FailoverReason

    err = _StreamErrorEvent(
        "You have either run out of available resources or do not have an "
        "active Grok subscription. Manage subscriptions at https://grok.com",
        code="The caller does not have permission to execute the specified operation",
    )
    result = classify_api_error(err, provider="xai-oauth", model="grok-4.3")
    assert result.reason == FailoverReason.auth
    assert result.retryable is False
    assert result.should_fallback is True


def test_classify_api_error_stream_event_resources_exhausted_grok_is_auth():
    """'out of available resources' + 'grok' variant also classifies as auth."""
    from run_agent import _StreamErrorEvent
    from agent.error_classifier import classify_api_error, FailoverReason

    err = _StreamErrorEvent(
        "You have run out of available resources for Grok.",
    )
    result = classify_api_error(err, provider="xai-oauth", model="grok-4.3")
    assert result.reason == FailoverReason.auth
    assert result.retryable is False


def test_classify_api_error_stream_event_unrelated_not_reclassified():
    """An unrelated _StreamErrorEvent must not be caught by the xAI guard."""
    from run_agent import _StreamErrorEvent
    from agent.error_classifier import classify_api_error, FailoverReason

    err = _StreamErrorEvent("Internal server error — try again later")
    result = classify_api_error(err, provider="xai-oauth", model="grok-4.3")
    assert result.reason != FailoverReason.auth


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


def test_codex_reasoning_replay_includes_encrypted_content_for_xai():
    """xAI must receive replayed encrypted reasoning items (May 2026 reversal).

    Earlier we stripped these on the theory that the OAuth/SuperGrok
    surface rejected them.  xAI subsequently confirmed they explicitly
    want Hermes to thread encrypted reasoning back across turns for
    cross-turn coherence — that's the whole point of the partnership
    integration.
    """
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        _assistant_msg_with_encrypted_reasoning(),
        {"role": "user", "content": "what's your name?"},
    ]

    items = _chat_messages_to_responses_input(msgs, is_xai_responses=True)
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert len(reasoning) == 1, (
        "xAI must receive replayed reasoning items — see docstring for the "
        "May 2026 reversal of the earlier suppression gate."
    )
    assert reasoning[0]["encrypted_content"] == "enc_blob"

    # And the assistant's visible text must still be present alongside it.
    assistant_items = [
        it for it in items
        if it.get("role") == "assistant" or it.get("type") == "message"
    ]
    assert assistant_items, "assistant message must still be present"


def test_codex_transport_xai_request_includes_encrypted_content():
    """xAI ``include`` array must request ``reasoning.encrypted_content``.

    This is the request-side half of the May 2026 reversal: we ask xAI
    to echo back encrypted reasoning so the next turn can replay it.
    """
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
    assert kwargs["include"] == ["reasoning.encrypted_content"]


def test_codex_transport_xai_replays_reasoning_in_input():
    """End-to-end: build_kwargs on xAI must replay prior encrypted reasoning."""
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
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["encrypted_content"] == "enc_blob"


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


def test_recover_with_credential_pool_skips_refresh_on_bare_403_for_xai_oauth():
    """A bare HTTP 403 from ``xai-oauth`` (no keyword match) must NOT loop refresh.

    Regression for #26847 — xAI's backend has been seen to 403 standard
    SuperGrok subscribers with a terser body that doesn't contain any of
    the existing entitlement keywords ("do not have an active Grok
    subscription", etc.). Before the defense-in-depth guard, the recovery
    path would happily mint a fresh token, get a fresh 403, and spin.
    """
    from run_agent import AIAgent
    from agent.error_classifier import FailoverReason

    agent = _make_codex_agent()
    assert agent.provider == "xai-oauth"

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
        "reason": "forbidden",
        "message": "Forbidden",
    }
    assert not AIAgent._is_entitlement_failure(error_context, 403), (
        "Pre-condition: bare 'Forbidden' body must NOT match the keyword "
        "heuristic — otherwise this test isn't covering the defense-in-depth path."
    )

    recovered, _retried_429 = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context=error_context,
    )

    assert recovered is False, "Bare 403 on xai-oauth must surface, not refresh-loop"
    assert refresh_calls["n"] == 0, "try_refresh_current must NOT be called on xai-oauth 403"


def test_recover_with_credential_pool_still_refreshes_genuine_auth_failure():
    """Regression guard: legitimate auth errors must still trigger refresh."""
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
# Fix D-bis: bad-credentials 403 must NOT be classified as entitlement (#29344)
#
# xAI returns the same permission-denied ``code`` text for two distinct
# conditions: unsubscribed account vs. stale OAuth access token.  The
# ``error`` field's ``[WKE=unauthenticated:...]`` suffix (and the
# accompanying "OAuth2 access token could not be validated" phrasing) is
# xAI's authoritative disambiguator — when present, the body is an auth
# failure, not entitlement, and the credential-pool refresh path must
# run.  Pre-fix, long-running TUI sessions stuck on a stale token
# surfaced as a non-retryable client error; the workaround was to exit
# and reopen the TUI so the startup-resolve path refreshed.
# ---------------------------------------------------------------------------


def test_is_entitlement_failure_false_for_bad_credentials_wke_suffix():
    """403 with ``[WKE=unauthenticated:bad-credentials]`` is auth, not entitlement.

    Verbatim shape from the #29344 reporter — the ``code`` text matches
    the entitlement permission-denied heuristic, but the ``error`` field
    carries xAI's explicit "this is a credential validation failure"
    signal.  Classifier must honor it.
    """
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "code": "The caller does not have permission to execute the specified operation",
            "error": "The OAuth2 access token could not be validated. [WKE=unauthenticated:bad-credentials]",
        },
        403,
    )


def test_is_entitlement_failure_false_for_wke_suffix_in_normalized_shape():
    """The same body after ``_extract_api_error_context`` normalisation.

    Real runtime paths feed the classifier through
    ``_extract_api_error_context``, which converts the raw body to
    ``{message, reason, reset_at}``.  The disambiguator must fire in
    BOTH the raw-body shape (test above) and the normalised shape so
    the fix actually reaches the production call site at
    ``_recover_with_credential_pool``.
    """
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "reason": "The caller does not have permission to execute the specified operation",
            "message": "The OAuth2 access token could not be validated. [WKE=unauthenticated:bad-credentials]",
        },
        403,
    )


@pytest.mark.parametrize("wke_variant", [
    # The headline variant — what xAI returns today.
    "[WKE=unauthenticated:bad-credentials]",
    # Forward-compat: xAI documents the WKE prefix as a stable shape,
    # the suffix after the colon is the "reason code" and could grow
    # new values.  Anything under ``unauthenticated:`` must route to
    # the refresh path.
    "[WKE=unauthenticated:expired-token]",
    "[WKE=unauthenticated:revoked]",
    "[WKE=unauthenticated:some-future-reason]",
])
def test_is_entitlement_failure_false_for_any_wke_unauthenticated_variant(wke_variant):
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "code": "The caller does not have permission to execute the specified operation",
            "error": f"Token rejected. {wke_variant}",
        },
        403,
    )


def test_is_entitlement_failure_false_via_oauth2_validation_phrase_alone():
    """Second disambiguator: the "OAuth2 access token could not be
    validated" phrase by itself (no WKE suffix) must also route to
    refresh.  This is a belt-and-braces guard against xAI dropping or
    reformatting the WKE suffix in a future API revision without
    changing the human-readable error text."""
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "code": "The caller does not have permission to execute the specified operation",
            "error": "The OAuth2 access token could not be validated.",
        },
        403,
    )


def test_is_entitlement_failure_wke_signal_overrides_entitlement_keywords():
    """Defensive: if a future xAI body somehow carries BOTH the WKE
    suffix AND entitlement language, the WKE signal wins.  Auth is
    recoverable; entitlement isn't.  If the refreshed token still
    can't access the resource, the next 403 (without WKE) lands on
    the entitlement path correctly."""
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "code": "The caller does not have permission to execute the specified operation",
            "error": (
                "do not have an active Grok subscription. "
                "[WKE=unauthenticated:bad-credentials]"
            ),
        },
        403,
    )


def test_is_entitlement_failure_case_insensitive_wke_match():
    """Substring match is case-insensitive — the classifier lowercases
    everything before matching, so a future xAI build that uppercases
    the prefix wouldn't reintroduce the misclassification."""
    from run_agent import AIAgent

    assert not AIAgent._is_entitlement_failure(
        {
            "code": "The caller does not have permission to execute the specified operation",
            "error": "[wke=Unauthenticated:Bad-Credentials]",
        },
        403,
    )


def test_recover_with_credential_pool_refreshes_on_xai_bad_credentials_403():
    """End-to-end #29344: a bad-credentials 403 from xai-oauth MUST
    call ``try_refresh_current()`` so the long-running TUI session
    recovers without an exit/reopen cycle.

    Mirrors the scaffolding of
    ``test_recover_with_credential_pool_still_refreshes_genuine_auth_failure``
    but with the exact 403 body shape xAI ships for stale tokens —
    the very body that pre-fix tripped the entitlement classifier
    and short-circuited the refresh path.
    """
    from agent.error_classifier import FailoverReason

    agent = _make_codex_agent()

    refresh_calls = {"n": 0}

    class _FakePool:
        def try_refresh_current(self):
            refresh_calls["n"] += 1
            entry = MagicMock()
            entry.id = "entry_refreshed_after_stale"
            return entry

        def mark_exhausted_and_rotate(self, **_kwargs):
            return None

        def has_available(self):
            return False

    agent._credential_pool = _FakePool()
    agent._swap_credential = MagicMock()

    # Normalised shape that ``_extract_api_error_context`` would
    # produce for the reporter's wire-level body.
    error_context = {
        "reason": (
            "The caller does not have permission to execute the specified operation"
        ),
        "message": (
            "The OAuth2 access token could not be validated. "
            "[WKE=unauthenticated:bad-credentials]"
        ),
    }

    recovered, _retried_429 = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context=error_context,
    )

    assert recovered is True, (
        "Stale OAuth token (bad-credentials 403) must trigger refresh — "
        "pre-fix this returned False because the entitlement classifier "
        "over-matched on the permission-denied code text"
    )
    assert refresh_calls["n"] == 1, "try_refresh_current must run exactly once"
    agent._swap_credential.assert_called_once()


def test_recover_with_credential_pool_still_blocks_real_entitlement():
    """Companion regression guard for the #29344 fix: the original
    #26847 protection — entitlement 403 must NOT refresh — must
    survive the new disambiguator.  A real unsubscribed-account body
    has no WKE suffix and no OAuth2-validation phrase, so the
    classifier still classifies it as entitlement and short-circuits."""
    from agent.error_classifier import FailoverReason

    agent = _make_codex_agent()

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

    # Pure entitlement body — no WKE suffix, no OAuth2 phrase.
    error_context = {
        "reason": (
            "The caller does not have permission to execute the specified operation"
        ),
        "message": (
            "You have either run out of available resources or do not have an "
            "active Grok subscription. Manage at https://grok.com"
        ),
    }

    recovered, _retried_429 = agent._recover_with_credential_pool(
        status_code=403,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context=error_context,
    )

    assert recovered is False, "Entitlement 403 must surface, not refresh"
    assert refresh_calls["n"] == 0


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


def test_grok_composer_context_length_is_262k():
    """grok-composer-2.5-fast is OAuth-only and missing from /v1/models.

    Without a specific entry it fell through to the generic ``grok`` 131k
    catch-all, under-reporting ~262k enforced on /v1/responses.
    """
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

    assert DEFAULT_CONTEXT_LENGTHS["grok-composer"] == 262_144
    slug = "grok-composer-2.5-fast"
    matched_key = max(
        (k for k in DEFAULT_CONTEXT_LENGTHS if k in slug.lower()),
        key=len,
    )
    assert matched_key == "grok-composer", (
        f"Expected longest-first match on grok-composer for {slug}, got {matched_key}"
    )
    assert DEFAULT_CONTEXT_LENGTHS[matched_key] == 262_144


# ---------------------------------------------------------------------------
# Cross-issuer reasoning replay guard
#
# When a session switches model providers mid-conversation (e.g. user runs
# /model gpt-5.5 after several turns on grok-4.3), the persisted reasoning
# items carry encrypted_content that only the issuing endpoint can decrypt.
# Replaying them against the new endpoint deterministically returns HTTP 400
# invalid_encrypted_content and breaks every subsequent turn. The cross-issuer
# guard stamps each reasoning item with its issuer on normalize and drops
# foreign-issuer items on replay.
# ---------------------------------------------------------------------------


def _stamped_assistant_msg(issuer_kind, *, text="hi", encrypted="enc_blob", rs_id="rs_001"):
    return {
        "role": "assistant",
        "content": text,
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "id": rs_id,
                "encrypted_content": encrypted,
                "summary": [],
                "_issuer_kind": issuer_kind,
            }
        ],
    }


def test_cross_issuer_reasoning_is_dropped_on_replay():
    """Reasoning minted by one Responses endpoint must not be replayed to
    another. This is the regression for the chatgpt-backend vs xAI-OAuth
    swap that returned invalid_encrypted_content on every turn after the
    user changed model mid-session.
    """
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        _stamped_assistant_msg("xai_responses", encrypted="grok_blob"),
        {"role": "user", "content": "next"},
    ]

    # Calling against codex_backend — the grok-issued blob must be dropped.
    items = _chat_messages_to_responses_input(
        msgs, current_issuer_kind="codex_backend"
    )
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert reasoning == [], (
        "Reasoning items stamped with a foreign _issuer_kind must be dropped "
        "before the API rejects the whole request with invalid_encrypted_content."
    )


def test_same_issuer_reasoning_is_still_replayed():
    """Same-endpoint reasoning replay is the documented happy path (May 2026
    reversal). The cross-issuer guard must not regress it.
    """
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        _stamped_assistant_msg("xai_responses", encrypted="grok_blob"),
        {"role": "user", "content": "next"},
    ]

    items = _chat_messages_to_responses_input(
        msgs, current_issuer_kind="xai_responses"
    )
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["encrypted_content"] == "grok_blob"
    # The internal stamp must not leak to the API payload.
    assert "_issuer_kind" not in reasoning[0]


def test_unstamped_reasoning_is_replayed_for_backwards_compat():
    """Reasoning items persisted before this patch don't carry _issuer_kind.
    They must still be replayed (legacy-compatible behaviour).
    """
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "codex_reasoning_items": [
                {
                    "type": "reasoning",
                    "id": "rs_legacy",
                    "encrypted_content": "legacy_blob",
                    "summary": [],
                }
            ],
        },
        {"role": "user", "content": "next"},
    ]

    items = _chat_messages_to_responses_input(
        msgs, current_issuer_kind="codex_backend"
    )
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["encrypted_content"] == "legacy_blob"


def test_normalize_codex_response_stamps_issuer_on_reasoning():
    """Reasoning captured from a response must be stamped with the issuer so
    a later replay against a different endpoint can drop it.
    """
    from types import SimpleNamespace

    from agent.codex_responses_adapter import _normalize_codex_response

    reasoning_item = SimpleNamespace(
        type="reasoning",
        id="rs_new",
        encrypted_content="fresh_blob",
        summary=[],
    )
    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="ok")],
        id="msg_1",
    )
    response = SimpleNamespace(output=[reasoning_item, message_item], status="completed")

    msg, _ = _normalize_codex_response(response, issuer_kind="xai_responses")
    assert msg.codex_reasoning_items and len(msg.codex_reasoning_items) == 1
    assert msg.codex_reasoning_items[0]["_issuer_kind"] == "xai_responses"
    assert msg.codex_reasoning_items[0]["encrypted_content"] == "fresh_blob"


def test_transport_round_trip_drops_foreign_reasoning():
    """Full transport flow: build_kwargs against codex_backend after grok turns
    must produce an `input` array that contains zero foreign reasoning items.
    """
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    messages = [
        {"role": "system", "content": "you are hermes"},
        {"role": "user", "content": "hi"},
        _stamped_assistant_msg("xai_responses", encrypted="grok_blob"),
        {"role": "user", "content": "엑스다임 프로젝트 파악, 스킬로 정리."},
    ]

    kwargs = transport.build_kwargs(
        model="gpt-5.5",
        messages=messages,
        tools=None,
        is_codex_backend=True,
        is_xai_responses=False,
        is_github_responses=False,
        base_url="https://chatgpt.com/backend-api/codex",
        instructions="you are hermes",
    )

    reasoning = [it for it in kwargs["input"] if it.get("type") == "reasoning"]
    assert reasoning == [], (
        "Cross-issuer reasoning leaked through build_kwargs — this is the "
        "exact regression that broke session 40de1ae0 on 2026-05-25 01:09."
    )
