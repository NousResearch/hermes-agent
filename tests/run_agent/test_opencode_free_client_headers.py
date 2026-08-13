"""Regression guard: opencode-free client auth + User-Agent handling.

The OpenCode free tier at ``https://opencode.ai/zen/v1`` changed its policy:
it now (a) requires a real account API key (the old literal "free" / keyless
requests return HTTP 401 AuthError) and (b) throttles third-party clients by
User-Agent, returning HTTP 429 ``FreeUsageLimitError`` unless the request
identifies as the opencode client (``User-Agent: opencode/...``).

Before this fix hermes stripped the Authorization header unconditionally for
``opencode-free`` and sent its own ``hermes-cli/...`` User-Agent, so every
request was rejected. The client must now keep the Authorization header AND
send an opencode User-Agent when a real key is configured, while preserving
the keyless auth-strip path for setups with no key at all.
"""
from unittest.mock import MagicMock, patch

import httpx

from agent.agent_runtime_helpers import create_openai_client

ZEN_V1 = "https://opencode.ai/zen/v1"


def _hook_names(http_client):
    """Return the names of request event hooks on an httpx client (if any)."""
    if not isinstance(http_client, httpx.Client):
        return []
    names = []
    for hooks in http_client.event_hooks.values():
        names.extend(getattr(h, "__name__", "") for h in hooks)
    return names


class _FakeAgent:
    def __init__(self, api_key):
        self.provider = "opencode-free"
        self.base_url = ZEN_V1
        self.api_key = api_key
        self.model = "deepseek-v4-flash-free"
        self.api_mode = "chat_completions"

    def _client_log_context(self):
        return {}

    def _build_keepalive_http_client(self, base_url, verify=True):
        return None


@patch("run_agent.OpenAI")
def test_opencode_free_with_key_keeps_auth_and_sets_opencode_ua(mock_openai):
    """A configured key must be sent AND the request must present as opencode.

    (a) no auth-stripping wrapper is installed, and (b) ``default_headers``
    carries ``User-Agent: opencode/latest`` so the free tier does not 429.
    """
    mock_openai.return_value = MagicMock()
    create_openai_client(
        _FakeAgent(api_key="sk-real-account-key"),
        {"api_key": "sk-real-account-key", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )

    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == ZEN_V1
    ]
    assert matching, "OpenAI was never constructed with the zen base_url"

    call = matching[-1].kwargs
    headers = dict(call.get("default_headers") or {})
    assert headers.get("User-Agent") == "opencode/latest", (
        "opencode-free with a key must send the opencode User-Agent so the "
        f"free tier does not 429; got {headers!r}"
    )
    assert "_strip_auth" not in _hook_names(call.get("http_client")), (
        "a configured key must NOT be stripped; the free tier now requires "
        "a real account API key (401 otherwise)"
    )


@patch("run_agent.OpenAI")
def test_opencode_free_without_key_preserves_keyless_strip_path(mock_openai):
    """No key configured → the historical keyless path stays intact: the httpx
    client is wrapped with an event hook that strips the Authorization header
    (the SDK always injects an empty ``Bearer `` header), and the opencode
    User-Agent is still applied so the free tier does not 429 by UA."""
    mock_openai.return_value = MagicMock()
    create_openai_client(
        _FakeAgent(api_key=""),
        {"api_key": "", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )

    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == ZEN_V1
    ]
    assert matching, "OpenAI was never constructed with the zen base_url"

    call = matching[-1].kwargs
    assert "_strip_auth" in _hook_names(call.get("http_client")), (
        "keyless opencode-free must keep the auth-stripping wrapper so the "
        "empty Bearer header never hits the wire"
    )
    headers = dict(call.get("default_headers") or {})
    assert headers.get("User-Agent") == "opencode/latest", (
        "keyless opencode-free must still send the opencode User-Agent or "
        "the free tier 429s the request"
    )


@patch("run_agent.OpenAI")
def test_opencode_free_placeholder_key_is_treated_as_keyless(mock_openai):
    """Hermes's keyless placeholder (``no-key-required``) must NOT be sent as a
    Bearer token — it is not a usable account key.  When a placeholder reaches
    the client (e.g. the auxiliary resolver's no-auth fallback), the request
    must be routed down the auth-stripping path instead of 401ing with the
    placeholder as the key."""
    mock_openai.return_value = MagicMock()
    create_openai_client(
        _FakeAgent(api_key="no-key-required"),
        {"api_key": "no-key-required", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )

    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == ZEN_V1
    ]
    assert matching, "OpenAI was never constructed with the zen base_url"

    call = matching[-1].kwargs
    assert "_strip_auth" in _hook_names(call.get("http_client")), (
        "a placeholder key must be stripped, not sent as the Authorization "
        "header (server 401s on 'no-key-required')"
    )
    headers = dict(call.get("default_headers") or {})
    assert headers.get("User-Agent") == "opencode/latest"
