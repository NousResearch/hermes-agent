# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""Mid-run credential rotation never rebinds a session to a same-provider entry for another endpoint.

A mixed ``openai`` pool (public api.openai.com key + Azure resource key) is legitimately shared with
an Azure-bound child (#68237). ``_swap_credential`` adopts the entry's base_url, so rotating the
Azure session onto the public entry on a 429 would send its traffic — and the public key — to the
wrong host. Rotation must be vetoed exactly like a rotation that yields nothing.
"""

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential

_AZURE = "https://res.cognitiveservices.azure.com/openai/v1"
_PUBLIC = "https://api.openai.com/v1"


def _entry(eid, url):
    return PooledCredential(provider="openai", id=eid, label=eid, auth_type="api_key", priority=0,
                            source=f"env:{eid}", access_token=f"key-{eid}", base_url=url)


class _AzureChild:
    """Session bound to the Azure entry of a mixed pool; real pool + real recovery helper."""

    provider = "openai"
    model = "gpt-5.4"
    base_url = _AZURE
    _fallback_activated = False

    def __init__(self, pool):
        self._credential_pool = pool
        self._credential_pool_entry_id = "az"
        self.api_key = "key-az"

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self.base_url = entry.base_url
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def test_rotation_on_mixed_pool_never_rebinds_to_an_entry_for_another_endpoint():
    pool = CredentialPool("openai", [_entry("az", _AZURE), _entry("pub", _PUBLIC)])
    agent = _AzureChild(pool)

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True, error_context={"message": "Rate limit"},
    )

    assert recovered is False
    assert (agent.api_key, agent.base_url, agent._credential_pool_entry_id) == ("key-az", _AZURE, "az")
    # The failed Azure entry is still benched; only the swap onto the wrong host was refused.
    assert next(e for e in pool.entries() if e.id == "az").last_status == "exhausted"
