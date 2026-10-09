"""Pool-level contract for the OpenCode relay 403 (#117869): the healthy credential survives.

The relay answers its own upstream failure with HTTP 403 + ``error.code=server_error``. The verdict
is only half the fix; what the operator sees is the credential pool. This drives the REAL
classifier into the REAL recovery helper against a REAL ``CredentialPool``, on bodies transcribed
from the reporters' logs, and builds the error the way the OpenAI SDK does (``body`` is the inner
``error`` object), so the captured envelope, the verdict and the pool mutation cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest

from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason, classify_api_error

RELAY_ENVELOPES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "opencode_go_403_relay_envelopes.json").read_text()
)["envelopes"]


def _sdk_error(envelope) -> openai.PermissionDeniedError:
    """The exception ``openai`` raises for this response: ``body`` is the inner error object."""
    request = httpx.Request("POST", f"{envelope['base_url']}/chat/completions")
    response = httpx.Response(envelope["status_code"], json=envelope["body"], request=request)
    return openai.PermissionDeniedError(
        f"Error code: {envelope['status_code']} - {envelope['body']}", response=response,
        body=envelope["body"]["error"],
    )


class _Agent:
    """Session stand-in: real pool + real recovery helper, no client build."""

    _fallback_activated = False
    _fallback_index = 0

    def __init__(self, pool, provider, model, base_url):
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


@pytest.mark.parametrize("envelope", RELAY_ENVELOPES, ids=[e["id"] for e in RELAY_ENVELOPES])
def test_captured_relay_envelope_leaves_the_sole_credential_available(envelope):
    provider, model, base_url = envelope["provider"], envelope["model"], envelope["base_url"]
    entry = PooledCredential.from_dict(provider, {
        "id": "relay403", "label": f"{provider}-sub", "auth_type": "api_key", "priority": 0,
        "access_token": "***", "base_url": base_url, "source": "manual",
    })
    pool = CredentialPool(provider=provider, entries=[entry])
    error = _sdk_error(envelope)

    verdict = classify_api_error(error, provider=provider, model=model, base_url=base_url)
    assert (verdict.reason, verdict.retryable, verdict.should_rotate_credential) == (
        FailoverReason.overloaded, True, False)

    recovered, _ = recover_with_credential_pool(
        _Agent(pool, provider, model, base_url), status_code=envelope["status_code"], has_retried_429=False,
        classified_reason=verdict.reason, error_context=extract_api_error_context(error),
        billing_unverified=False,
    )

    entry = pool.entries()[0]
    assert recovered is False, "the overloaded verdict must skip credential recovery"
    assert (entry.last_status, entry.last_error_code, entry.last_error_reason, entry.failure_reason) == (
        None, None, None, None), "the entry was never written to"
    assert pool.has_available(model=model) is True, "the credential is still selectable"
