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


# ---------------------------------------------------------------------------
# Nested error envelope on ``type=error`` SSE frames (opencode#36130 port)
#
# The Responses spec carries error details at the top level of the frame,
# but the official OpenAI SDK and several OpenAI-compatible proxies wrap
# them in an HTTP-style nested envelope:
#   {"type": "error", "error": {"code": ..., "message": ..., "param": ...}}
# Before the fix, _raise_stream_error only read top-level fields, so these
# frames collapsed to the generic "stream emitted error event" placeholder
# and the error classifier never saw the provider's real code/message.
# ---------------------------------------------------------------------------




def test_codex_stream_wire_error_event_nested_envelope_attr_style():
    """Details nested under ``error`` (SDK attr-object shape) are surfaced."""
    from run_agent import _StreamErrorEvent

    agent = _make_codex_agent()

    class _ErrorCreateStream:
        def __iter__(self_inner):
            yield SimpleNamespace(
                type="error",
                message=None,
                code=None,
                param=None,
                error=SimpleNamespace(
                    type="rate_limit_error",
                    code="rate_limit_exceeded",
                    message="Slow down",
                    param=None,
                ),
            )

        def close(self_inner):
            pass

    mock_client = MagicMock()
    mock_client.responses.create.return_value = _ErrorCreateStream()

    with pytest.raises(_StreamErrorEvent) as excinfo:
        agent._run_codex_stream({}, client=mock_client)

    assert "Slow down" in str(excinfo.value)
    assert excinfo.value.code == "rate_limit_exceeded"












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
        def try_refresh_matching(self, api_key_hint=None):
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


def test_recover_with_credential_pool_rotates_on_xai_spending_limit_403():
    """xAI's explicit spending-limit 403 must rotate, not hit the entitlement guard."""
    from agent.error_classifier import FailoverReason, classify_api_error

    agent = _make_codex_agent()
    next_entry = MagicMock(id="healthy-account")
    refresh_calls = {"n": 0}

    class _SpendingLimitError(Exception):
        status_code = 403
        body = {
            "code": "personal-team-blocked:spending-limit",
            "error": (
                "You have run out of credits or need a Grok subscription. "
                "Add credits at Grok or upgrade at Grok."
            ),
        }

    class _FakePool:
        provider = "xai-oauth"

        def try_refresh_matching(self, api_key_hint=None):
            refresh_calls["n"] += 1
            return MagicMock(id="should_not_be_called")

        def mark_exhausted_and_rotate(
            self,
            *,
            status_code,
            error_context=None,
            api_key_hint=None,
        ):
            assert status_code == 403
            assert api_key_hint == "test-key"
            assert error_context == {
                "reason": "personal-team-blocked:spending-limit",
                "message": (
                    "You have run out of credits or need a Grok subscription. "
                    "Add credits at Grok or upgrade at Grok."
                ),
            }
            return next_entry

    error = _SpendingLimitError("Error code: 403")
    classified = classify_api_error(error, provider="xai-oauth", model="grok-4.5")
    error_context = agent._extract_api_error_context(error)
    setattr(agent, "_credential_pool", _FakePool())
    agent._swap_credential = MagicMock()

    recovered, retried_429 = agent._recover_with_credential_pool(
        status_code=error.status_code,
        has_retried_429=False,
        classified_reason=classified.reason,
        error_context=error_context,
    )

    assert classified.reason == FailoverReason.billing
    assert recovered is True
    assert retried_429 is False
    assert refresh_calls["n"] == 0
    agent._swap_credential.assert_called_once_with(next_entry)






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
