"""Pool-level contract for the relay 403: the healthy credential survives.

The classifier verdict is only half the fix — what the operator sees is the credential pool. This
drives REAL classifier output into the REAL recovery helper against a REAL ``CredentialPool`` and
asserts the sole ``opencode-go`` credential is not benched for the relay's ``code=server_error``
403, while a genuine auth 403 still benches it (the behavior the pool exists for).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool
from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason, classify_api_error

PROVIDER = "opencode-go"
MODEL = "deepseek-v4.1-flash"
BASE = "https://opencode.ai/zen/go/v1"

# Envelopes transcribed from what the reporters on #117869 saw the relay send, so the verdict is
# pinned to the real wrapper instead of a paraphrase of it.
RELAY_ENVELOPES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "opencode_go_403_relay_envelopes.json").read_text()
)["envelopes"]

RELAY_BODY = {"error": {"message": "Upstream request failed: Upstream response was not valid JSON",
                        "type": "server_error", "code": "server_error"}}
AUTH_BODY = {"error": {"message": "Invalid API key provided", "code": "invalid_api_key"}}


class _MockAPIError(Exception):
    """An OpenAI-SDK-shaped error carrying status + body."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


class _Agent:
    """Session stand-in: real pool + real recovery helper, no client build."""

    _fallback_activated = False
    _fallback_index = 0

    def __init__(self, pool, provider=PROVIDER, model=MODEL, base_url=BASE):
        self._credential_pool = pool
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self._primary_runtime = {"provider": provider, "model": model, "base_url": base_url}
        entry = pool.entries()[0]
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def _sole_entry_pool(provider=PROVIDER, base_url=BASE):
    entry = PooledCredential.from_dict(provider, {
        "id": "relay403", "label": f"{provider}-sub", "auth_type": "api_key", "priority": 0,
        "access_token": "***", "base_url": base_url, "source": "manual",
    })
    return CredentialPool(provider=provider, entries=[entry])


def _verdict_and_context(body):
    """The two halves of the real call site: the classifier verdict and the raw body context."""
    error = _MockAPIError("Forbidden", status_code=403, body=body)
    return (
        classify_api_error(error, provider=PROVIDER, model=MODEL, base_url=BASE),
        extract_api_error_context(error),
    )


def _recover(agent, verdict, context, status_code=403):
    return recover_with_credential_pool(
        agent, status_code=status_code, has_retried_429=False,
        classified_reason=verdict.reason, error_context=context, billing_unverified=False,
    )


def test_relay_403_leaves_the_sole_credential_available():
    """The field-observed relay 403 must not bench the only opencode-go credential."""
    pool = _sole_entry_pool()
    agent = _Agent(pool)
    verdict, context = _verdict_and_context(RELAY_BODY)
    assert verdict.reason == FailoverReason.overloaded  # classifier half of the fix
    assert context["reason"] == "server_error"  # what the relay body itself said

    recovered, _ = _recover(agent, verdict, context)

    entry = pool.entries()[0]
    assert recovered is False, "the overloaded verdict must skip credential recovery"
    assert entry.last_status is None, "the entry was never written to"
    assert entry.last_error_code is None
    assert entry.last_error_reason is None, "the body's 'server_error' never became a pool verdict"
    assert entry.failure_reason is None
    assert pool.has_available() is True, "the credential is still selectable"


def test_auth_403_still_benches_the_credential():
    """Control: a real permission failure keeps the auth recovery path this PR leaves alone.

    Also pins the two fields apart: ``last_error_reason`` carries the body's word while
    ``failure_reason`` carries the classifier's verdict — the mismatch the field trace showed.
    """
    pool = _sole_entry_pool()
    agent = _Agent(pool)
    verdict, context = _verdict_and_context(AUTH_BODY)
    assert verdict.reason == FailoverReason.auth
    assert context["reason"] == "invalid_api_key"

    _recover(agent, verdict, context)

    entry = pool.entries()[0]
    assert entry.last_status == STATUS_EXHAUSTED
    assert entry.last_error_code == 403
    assert entry.last_error_reason == "invalid_api_key"
    assert entry.failure_reason == "auth"


@pytest.mark.parametrize("envelope", RELAY_ENVELOPES, ids=[e["id"] for e in RELAY_ENVELOPES])
def test_captured_relay_envelope_leaves_the_sole_credential_available(envelope):
    """A body transcribed from the field report, not a paraphrase of it: same invariant.

    The captures carry wrapper prose the synthetic body above does not — ``Error from provider
    (Console Go):`` and a bracketed ``[server_error]`` marker inside the message — so a pattern
    added ahead of the provider-scoped code check that matched either one would fail here. Every
    input below is read from the envelope, so the context that classifies and the pool that gets
    mutated cannot drift apart.
    """
    provider, model, base_url = envelope["provider"], envelope["model"], envelope["base_url"]
    pool = _sole_entry_pool(provider, base_url)
    agent = _Agent(pool, provider, model, base_url)
    error = _MockAPIError("Forbidden", status_code=envelope["status_code"], body=envelope["body"])
    verdict = classify_api_error(error, provider=provider, model=model, base_url=base_url)
    assert verdict.reason == FailoverReason.overloaded
    assert verdict.retryable is True
    assert verdict.should_rotate_credential is False

    recovered, _ = _recover(agent, verdict, extract_api_error_context(error), envelope["status_code"])

    entry = pool.entries()[0]
    assert recovered is False, "the overloaded verdict must skip credential recovery"
    assert entry.last_status is None
    assert entry.last_error_code is None
    assert entry.last_error_reason is None
    assert entry.failure_reason is None
    assert pool.has_available(model=model) is True
