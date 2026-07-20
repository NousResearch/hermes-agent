"""Live Friendli smoke test — exercises the Hermes runtime, not a raw SDK client.

Opt-in only:
    HERMES_LIVE_TESTS=1 FRIENDLI_API_KEY=... \\
        pytest tests/run_agent/test_friendli_live.py -q

Unlike a bare OpenAI() client pointed at the endpoint, this drives Hermes'
own provider resolution — ``resolve_provider_client('friendli')`` — so it
verifies the auth/config/base-URL/aux-model wiring that the bundled
provider actually ships, then makes a real call through that client.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"
FRIENDLI_API_KEY = os.environ.get("FRIENDLI_API_KEY", "")

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="live-only: set HERMES_LIVE_TESTS=1"),
    pytest.mark.skipif(not FRIENDLI_API_KEY, reason="FRIENDLI_API_KEY not configured"),
    pytest.mark.integration,
]


def _resolve_runtime_client(provider="friendli"):
    """Build the Friendli client the way the Hermes runtime does."""
    from agent.auxiliary_client import resolve_provider_client

    client, model = resolve_provider_client(provider)
    assert client is not None, "Hermes failed to build a Friendli client"
    return client, model


def test_hermes_wires_friendli_client():
    """The runtime resolves a Friendli client pointed at the right endpoint."""
    client, model = _resolve_runtime_client()
    assert "api.friendli.ai" in str(client.base_url)
    assert model


def test_friendli_basic_chat_through_runtime():
    """A single-turn completion via the Hermes-resolved client returns text."""
    client, model = _resolve_runtime_client()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly the word 'pong' and nothing else."}],
        timeout=60,
    )

    content = response.choices[0].message.content
    assert content and "pong" in content.lower()


def test_friendli_alias_resolves_through_runtime():
    """The 'friendliai' alias resolves to the same Friendli client via the runtime."""
    client, _ = _resolve_runtime_client("friendliai")
    assert "api.friendli.ai" in str(client.base_url)
