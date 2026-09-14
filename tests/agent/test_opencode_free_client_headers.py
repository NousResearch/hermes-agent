"""Regression guard: opencode-free client keyless header handling.

OpenCode's free tier at ``https://opencode.ai/zen/v1`` is served ANONYMOUSLY:
requests with no recognizable Authorization bearer succeed, while any bearer
the relay doesn't recognize — placeholders included — is rejected with 401
"Invalid API key" (verified live 2026-08-21).

The client therefore must ship an EMPTY ``Authorization`` default header for
every opencode-free build, which overrides the OpenAI SDK's always-injected
``Authorization: Bearer <api_key>`` so no credential-shaped value ever
reaches the wire.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import create_openai_client
from agent.auxiliary_client import _create_openai_client, _to_async_client

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
def test_opencode_free_sends_hermes_attribution(mock_openai):
    """Keyless requests still identify as Hermes (attribution headers match
    the opencode zen/go profiles)."""
    mock_openai.return_value = MagicMock()
    create_openai_client(
        _FakeAgent(api_key="opencode-zen-free-keyless"),
        {"api_key": "opencode-zen-free-keyless", "base_url": ZEN_V1},
        reason="test",
        shared=False,
    )
    headers = _zen_call_headers(mock_openai)
    assert headers.get("X-Title") == "Hermes Agent"
    assert str(headers.get("User-Agent", "")).startswith("HermesAgent/")


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


@patch("agent.process_bootstrap.OpenAI")
def test_free_aliases_blank_authorization_with_stale_key(mock_openai):
    """agent.provider stores the raw alias (free / opencode_free). A stale
    key must still blank Authorization — same contract as opencode-free."""
    mock_openai.return_value = MagicMock()
    for provider in ("free", "opencode_free"):
        mock_openai.reset_mock()
        agent = _FakeAgent(api_key="sk-stale")
        agent.provider = provider
        create_openai_client(
            agent,
            {"api_key": "sk-stale", "base_url": ZEN_V1},
            reason="test",
            shared=False,
        )
        headers = _zen_call_headers(mock_openai)
        assert headers.get("Authorization") == "", (
            f"{provider} with a stale key must blank Authorization; got {headers!r}"
        )


@patch("agent.process_bootstrap.OpenAI")
def test_aliased_zen_placeholder_blanks_authorization(mock_openai):
    """ALIASES maps opencode-zen → opencode. A healed *-free session still
    carries the keyless placeholder; Authorization must be blanked even
    though agent.provider is no longer opencode-free (#93890)."""
    mock_openai.return_value = MagicMock()
    for provider in ("opencode", "opencode-zen"):
        mock_openai.reset_mock()
        agent = _FakeAgent(api_key="opencode-zen-free-keyless")
        agent.provider = provider
        create_openai_client(
            agent,
            {
                "api_key": "opencode-zen-free-keyless",
                "base_url": ZEN_V1,
            },
            reason="test",
            shared=False,
        )
        headers = _zen_call_headers(mock_openai)
        assert headers.get("Authorization") == "", (
            f"{provider} with the keyless placeholder must blank "
            f"Authorization; got {headers!r}"
        )


@patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={})
@patch("openai.AsyncOpenAI")
def test_to_async_client_blanks_keyless_authorization(mock_async, _http):
    """Aux vision/title/compression rebuilds AsyncOpenAI from (api_key, base_url)
    and must not re-emit Bearer <placeholder> (#93890)."""
    mock_async.return_value = MagicMock()
    sync = SimpleNamespace(
        api_key="opencode-zen-free-keyless",
        base_url=ZEN_V1,
    )
    _to_async_client(sync, "mimo-v2.5-free", is_vision=True)
    headers = dict(mock_async.call_args.kwargs.get("default_headers") or {})
    assert headers.get("Authorization") == "", (
        f"async wrap of keyless placeholder must blank Authorization; got {headers!r}"
    )


@patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={})
@patch("openai.AsyncOpenAI")
def test_to_async_client_keyed_zen_does_not_blank_authorization(mock_async, _http):
    """A real Zen key must not pick up Authorization: \"\" on the async wrap."""
    mock_async.return_value = MagicMock()
    sync = SimpleNamespace(
        api_key="sk-real-opencode",
        base_url=ZEN_V1,
    )
    _to_async_client(sync, "claude-sonnet-4-5", is_vision=True)
    headers = dict(mock_async.call_args.kwargs.get("default_headers") or {})
    assert "Authorization" not in headers, (
        f"keyed Zen must not insert Authorization: \"\"; got {headers!r}"
    )


@patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={})
@patch("openai.AsyncOpenAI")
def test_to_async_client_opencode_free_provider_blanks_stale_key(mock_async, _http):
    """Dedicated opencode-free (and public aliases) is keyless even when the
    leaf key is not the placeholder — the async wrap must honour the tag."""
    mock_async.return_value = MagicMock()
    for provider in ("opencode-free", "free", "opencode_free"):
        mock_async.reset_mock()
        sync = SimpleNamespace(
            api_key="sk-stale",
            base_url=ZEN_V1,
            _hermes_aux_effective_provider=provider,
        )
        _to_async_client(sync, "x-preview-f-free", is_vision=True)
        headers = dict(mock_async.call_args.kwargs.get("default_headers") or {})
        assert headers.get("Authorization") == "", (
            f"async wrap of {provider} + stale key must blank Authorization; "
            f"got {headers!r}"
        )


@patch("agent.auxiliary_client._openai_http_client_kwargs", return_value={})
@patch("agent.auxiliary_client.OpenAI")
def test_aux_create_openai_client_blanks_placeholder(mock_openai, _http):
    """Aux sync construction uses the same keyless predicate as the primary
    rebuild paths, not a private string compare."""
    mock_openai.return_value = MagicMock()
    _create_openai_client(api_key="opencode-zen-free-keyless", base_url=ZEN_V1)
    headers = dict(mock_openai.call_args.kwargs.get("default_headers") or {})
    assert headers.get("Authorization") == "", (
        f"aux _create_openai_client must blank placeholder Authorization; "
        f"got {headers!r}"
    )
