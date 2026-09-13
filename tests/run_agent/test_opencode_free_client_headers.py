"""Regression guard: opencode-free client keyless header handling.

OpenCode's free tier at ``https://opencode.ai/zen/v1`` is served ANONYMOUSLY:
requests with no recognizable Authorization bearer succeed, while any bearer
the relay doesn't recognize — placeholders included — is rejected with 401
"Invalid API key" (verified live 2026-08-21).

The client therefore must ship an EMPTY ``Authorization`` default header for
every opencode-free build, which overrides the OpenAI SDK's always-injected
``Authorization: Bearer <api_key>`` so no credential-shaped value ever reaches
the wire.

The free tier is additionally gated on the request identifying as the OpenCode
client rather than as Hermes (Hermes' canonical attribution headers draw HTTP
429 ``FreeUsageLimitError``, the OpenCode fingerprint gets 200), so this client
also carries that fingerprint — see the wire-level guard in
``test_opencode_zen_free_wire_headers.py`` (#106495).
"""
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import create_openai_client

ZEN_V1 = "https://opencode.ai/zen/v1"


class _FakeAgent:
    def __init__(self, api_key):
        self.provider = "opencode-free"
        self.base_url = ZEN_V1
        self.api_key = api_key
        self.model = "x-preview-f-free"
        self.api_mode = "chat_completions"

    def _client_log_context(self):
        return {}

    def _build_keepalive_http_client(self, base_url, verify=True):
        return None


def _zen_call_headers(mock_openai):
    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == ZEN_V1
    ]
    assert matching, "OpenAI was never constructed with the zen base_url"
    return dict(matching[-1].kwargs.get("default_headers") or {})


@patch("agent.process_bootstrap.OpenAI")
def test_opencode_free_blanks_authorization_header(mock_openai):
    """Whatever api_key value reaches the client build (placeholder, stale
    key, empty), the Authorization default header must be blanked so the
    SDK's Bearer never hits the wire."""
    mock_openai.return_value = MagicMock()
    for key in ("opencode-zen-free-keyless", "no-key-required", "", "sk-stale"):
        mock_openai.reset_mock()
        create_openai_client(
            _FakeAgent(api_key=key),
            {"api_key": key, "base_url": ZEN_V1},
            reason="test",
            shared=False,
        )
        headers = _zen_call_headers(mock_openai)
        assert headers.get("Authorization") == "", (
            f"opencode-free with api_key={key!r} must blank Authorization; "
            f"got {headers!r}"
        )


@patch("agent.process_bootstrap.OpenAI")
def test_opencode_free_sends_opencode_fingerprint(mock_openai):
    """Keyless requests identify as the OpenCode client, not as Hermes: the relay's free
    tier 429s (FreeUsageLimitError) the canonical Hermes attribution headers (#106495)."""
    mock_openai.return_value = MagicMock()
    create_openai_client(
        _FakeAgent(api_key="opencode-zen-free-keyless"),
        {"api_key": "opencode-zen-free-keyless", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )
    headers = _zen_call_headers(mock_openai)
    assert headers.get("X-Title") == "opencode"
    assert headers.get("User-Agent") == "opencode/0.20.5"
    assert headers.get("HTTP-Referer") == "https://opencode.ai/"
    assert headers.get("X-Session-ID")
    assert not str(headers.get("User-Agent", "")).startswith("HermesAgent/")


@patch("agent.process_bootstrap.OpenAI")
def test_zen_free_model_selected_under_keyed_provider_gets_the_same_fingerprint(mock_openai):
    """A free slug picked under ``opencode-zen`` heals to the same keyless runtime, so the
    client build must key the header set off the keyless placeholder, not the provider name."""
    mock_openai.return_value = MagicMock()
    agent = _FakeAgent(api_key="opencode-zen-free-keyless")
    agent.provider = "opencode-zen"
    create_openai_client(
        agent,
        {"api_key": "opencode-zen-free-keyless", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )
    headers = _zen_call_headers(mock_openai)
    assert headers.get("User-Agent") == "opencode/0.20.5"
    assert headers.get("X-Session-ID")


@patch("agent.process_bootstrap.OpenAI")
def test_other_providers_unaffected(mock_openai):
    """The opencode-free header policy must not leak to other providers."""
    mock_openai.return_value = MagicMock()
    agent = _FakeAgent(api_key="sk-real")
    agent.provider = "opencode-zen"
    create_openai_client(
        agent,
        {"api_key": "sk-real", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )
    headers = _zen_call_headers(mock_openai)
    assert "Authorization" not in headers, (
        "opencode-zen (keyed) must not have its Authorization header blanked"
    )
