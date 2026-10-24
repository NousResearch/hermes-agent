"""A ``fallback_providers`` entry's ``base_url`` must reach the client (#121359).

``try_activate_fallback`` forwards the entry's ``base_url`` to
``resolve_provider_client(explicit_base_url=...)``.  Every registry API-key provider honours
that override, but the two providers with their own resolver branches — ``anthropic`` and
``openrouter`` — dropped it and built the client on the vendor's canonical host, sending the
fallback turn (and the key) somewhere the user never configured.

These assert the RELATIONSHIP: given an explicit endpoint, the resolved client's base URL is
that endpoint. No live request is made; all hosts here are loopback/``.invalid``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
                "ANTHROPIC_BASE_URL", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


_RELAY_ANTHROPIC = "http://127.0.0.1:9002/anthropic"
_RELAY_OPENROUTER = "http://127.0.0.1:9002/openrouter/v1"


def _client_base_url(client) -> str:
    for chain in (("base_url",), ("_real_client", "base_url"), ("_client", "base_url")):
        obj = client
        try:
            for attr in chain:
                obj = getattr(obj, attr)
            return str(obj)
        except AttributeError:
            continue
    return ""


def test_anthropic_fallback_entry_base_url_is_the_resolved_endpoint():
    """A fallback entry pointing anthropic at an Anthropic-protocol relay must be honoured."""
    from agent.auxiliary_client import resolve_provider_client

    fake_anthropic = MagicMock(name="anthropic_sdk_client")
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=fake_anthropic) as mock_build:
        client, _model = resolve_provider_client(
            "anthropic", model="claude-haiku-4-5-20251001", raw_codex=True,
            explicit_base_url=_RELAY_ANTHROPIC, explicit_api_key="sk-test-not-a-real-key")

    assert client is not None
    assert mock_build.call_args[0][1] == _RELAY_ANTHROPIC
    assert client.base_url == _RELAY_ANTHROPIC


def test_anthropic_fallback_entry_refuses_a_non_anthropic_endpoint():
    """An explicit target mismatch is REFUSED — never silently demoted to the canonical host.

    Continuing with `base_url` reset to api.anthropic.com would send the caller's explicit
    credential to a host the caller did not ask for. No client is produced and the SDK
    builder is never reached.
    """
    from agent.auxiliary_client import resolve_provider_client

    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as mock_build:
        client, _model = resolve_provider_client(
            "anthropic", model="claude-haiku-4-5-20251001", raw_codex=True,
            explicit_base_url="http://127.0.0.1:9002/openai/v1", explicit_api_key="sk-test-not-a-real-key")

    assert client is None, "an incompatible explicit endpoint must not yield a client"
    assert mock_build.call_count == 0, "no canonical client may be constructed on refusal"


def test_openrouter_fallback_entry_base_url_is_the_resolved_endpoint():
    """A fallback entry pointing openrouter at a relay must not resolve to openrouter.ai."""
    from agent.auxiliary_client import resolve_provider_client

    client, _model = resolve_provider_client(
        "openrouter", model="some/model", raw_codex=True,
        explicit_base_url=_RELAY_OPENROUTER, explicit_api_key="sk-test-not-a-real-key")

    assert client is not None
    assert _client_base_url(client).rstrip("/") == _RELAY_OPENROUTER


def test_openrouter_without_an_entry_base_url_keeps_the_configured_default():
    """No override: resolution is unchanged (no relay invented)."""
    from agent.auxiliary_client import OPENROUTER_BASE_URL, resolve_provider_client

    client, _model = resolve_provider_client(
        "openrouter", model="some/model", raw_codex=True, explicit_api_key="sk-test-not-a-real-key")

    assert client is not None
    assert _client_base_url(client).rstrip("/") == OPENROUTER_BASE_URL.rstrip("/")
