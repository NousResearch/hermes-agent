"""Tests for per-provider ``custom_providers[].extra_headers`` on the auxiliary client.

Companion to ``tests/agent/test_auxiliary_user_default_headers.py`` (which
covers the global ``model.default_headers`` override). The main agent applies
``custom_providers[].extra_headers`` via
``hermes_cli.config.apply_custom_provider_extra_headers_to_client_kwargs``
(see ``run_agent.py`` / ``agent/agent_init.py``), but the auxiliary client
(title generation, context compression, vision routing, web_extract) built a
separate OpenAI client and never received the matching per-provider headers —
so a host-scoped override such as ``User-Agent: curl/8.7.1`` reached the main
turn but NOT the same turn's title-generation call, which 502ed behind a
gateway/WAF that rejected the OpenAI SDK's identifying headers.

These tests assert the auxiliary OpenAI-wire client-construction routes
inherit the matching per-provider ``extra_headers`` and that provider-specific
headers override pre-existing defaults (mirroring the main agent's precedence:
SDK/profile defaults < ``model.default_headers`` < ``custom_providers[].extra_headers``).
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so load_config() reads our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


class TestAuxClientReceivesProviderExtraHeaders:
    """resolve_provider_client must pass the matching per-provider extra_headers
    to the OpenAI client for a named custom provider."""

    def test_named_custom_provider_extra_headers_reach_aux_client(self, tmp_path):
        """The production reproduction: a named custom provider behind a
        gateway/WAF declares ``extra_headers`` to override the OpenAI SDK's
        User-Agent. The auxiliary client (title generation path) must inherit
        them, not just the main agent."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "fastai",
                    "base_url": "https://fastai.example.com/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("fastai", "test-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"

    def test_provider_headers_override_pre_existing_defaults(self, tmp_path):
        """Provider-specific ``extra_headers`` win over pre-existing defaults
        already in ``default_headers`` (here: a global ``model.default_headers``
        User-Agent). The provider value is the most specific config level and
        must survive, while unrelated defaults are preserved."""
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "sdk-default", "X-Keep": "keep-me"},
            },
            "custom_providers": [
                {
                    "name": "fastai",
                    "base_url": "https://fastai.example.com/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("fastai", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        # Provider-specific value overrides the pre-existing default.
        assert headers.get("User-Agent") == "curl/8.7.1"
        # An unrelated pre-existing default is preserved (not wiped).
        assert headers.get("X-Keep") == "keep-me"

    def test_unmatched_base_url_gets_no_provider_headers(self, tmp_path):
        """Matching, not blanket application: an ``extra_headers`` entry whose
        base_url does NOT match the resolved endpoint must not contribute its
        headers (a different custom provider's credentials must never leak)."""
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "acme",
                    "base_url": "https://acme.example.com/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1"},
                },
                {
                    "name": "fastai",
                    "base_url": "https://fastai.example.com/v1",
                    "api_key": "k",
                },
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("fastai", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        # acme's headers must NOT leak onto a fastai client.
        assert "User-Agent" not in headers


class TestAuxAsyncClientReceivesProviderExtraHeaders:
    """The async OpenAI-wire route (_to_async_client) must also inherit the
    matching per-provider extra_headers — it builds its own AsyncOpenAI client
    and is a distinct construction path from _create_openai_client."""

    def test_to_async_client_applies_matching_provider_headers(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {
                    "name": "fastai",
                    "base_url": "https://fastai.example.com/v1",
                    "api_key": "k",
                    "extra_headers": {"User-Agent": "curl/8.7.1", "X-Tag": "aux"},
                },
            ],
        })
        # A plain OpenAI-shaped sync client pointed at the custom provider's
        # base_url — _to_async_client reads .api_key / .base_url off it.
        sync_stub = SimpleNamespace(
            api_key="k", base_url="https://fastai.example.com/v1"
        )
        with patch("openai.AsyncOpenAI") as mock_async:
            mock_async.return_value = MagicMock()
            from agent.auxiliary_client import _to_async_client
            async_client, model = _to_async_client(sync_stub, "test-model")

        assert mock_async.called
        headers = mock_async.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"
        assert headers.get("X-Tag") == "aux"


class TestSharedBaseUrlNamedProvidersKeepIdentity:
    """Two named custom providers may legitimately share one ``base_url`` while
    declaring distinct ``extra_headers`` (tenant routing / per-tenant auth).
    Resolving a named provider must apply ONLY that provider's headers — the
    generic base-URL matcher must never overwrite them with the first
    URL-matching entry's headers (wrong-tenant / wrong-credential routing).
    """

    _SHARED_CONFIG = {
        "model": {"default": "test-model"},
        "custom_providers": [
            {
                "name": "tenant-a",
                "base_url": "https://shared.example.com/v1",
                "api_key": "k",
                "extra_headers": {"X-Tenant": "a"},
            },
            {
                "name": "tenant-b",
                "base_url": "https://shared.example.com/v1",
                "api_key": "k",
                "extra_headers": {"X-Tenant": "b"},
            },
        ],
    }

    def test_named_provider_sends_only_its_own_headers_sync(self, tmp_path):
        """Resolving ``tenant-b`` on the sync OpenAI-wire route must carry
        ``X-Tenant=b`` and never ``tenant-a``'s ``X-Tenant=a``."""
        _write_config(tmp_path, self._SHARED_CONFIG)
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("tenant-b", "test-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Tenant") == "b"
        # The first URL-matching entry's header must NOT leak through.
        assert headers.get("X-Tenant") != "a"

    def test_named_provider_sends_only_its_own_headers_async(self, tmp_path):
        """The async conversion route (_to_async_client) must equally preserve
        the selected named provider's identity when two providers share a
        base_url. _to_async_client re-derives headers from base_url, so without
        identity propagation it silently reverts to the first URL match."""
        _write_config(tmp_path, self._SHARED_CONFIG)
        # resolve_provider_client(async_mode=True) builds a sync OpenAI client
        # then converts it via _to_async_client. Stub the sync OpenAI call so
        # _to_async_client reads real .api_key / .base_url attributes off it.
        with patch("agent.auxiliary_client.OpenAI") as mock_openai, \
                patch("openai.AsyncOpenAI") as mock_async:
            mock_openai.return_value = SimpleNamespace(
                api_key="k", base_url="https://shared.example.com/v1"
            )
            mock_async.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client(
                "tenant-b", "test-model", async_mode=True
            )

        assert mock_async.called
        headers = mock_async.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("X-Tenant") == "b"
        assert headers.get("X-Tenant") != "a"


class TestSharedBaseUrlHeaderlessNamedProviderSuppressesUrlMatch:
    """A named provider selected by identity must suppress generic base-URL
    matching EVEN WHEN it declares no ``extra_headers`` of its own.

    tenant-a declares credential/tenant headers; tenant-b is headerless but
    shares tenant-a's base_url (a legit setup: same gateway, different auth
    strategy). Resolving tenant-b must NOT inherit tenant-a's headers via the
    generic first-URL-match fallback — that would send tenant-a's credentials
    on tenant-b's request. Selecting a named provider by identity is
    authoritative for headers (including "no headers"); only routes with no
    named-provider identity fall back to generic URL matching.
    """

    _SHARED_CONFIG = {
        "model": {"default": "test-model"},
        "custom_providers": [
            {
                "name": "tenant-a",
                "base_url": "https://shared.example.com/v1",
                "api_key": "k",
                "extra_headers": {
                    "X-Tenant": "a",
                    "Authorization": "Bearer a-secret-token",
                },
            },
            {
                "name": "tenant-b",
                "base_url": "https://shared.example.com/v1",
                "api_key": "k",
            },
        ],
    }

    def test_headerless_named_provider_gets_no_other_tenants_headers_sync(self, tmp_path):
        """Resolving headerless ``tenant-b`` (sync) must carry NEITHER
        ``tenant-a``'s ``X-Tenant`` nor its ``Authorization``."""
        _write_config(tmp_path, self._SHARED_CONFIG)
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("tenant-b", "test-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "X-Tenant" not in headers
        assert "Authorization" not in headers

    def test_headerless_named_provider_gets_no_other_tenants_headers_async(self, tmp_path):
        """Resolving headerless ``tenant-b`` (async conversion route) must
        equally suppress tenant-a's headers — _to_async_client re-derives
        headers from base_url and would otherwise revert to the first URL
        match."""
        _write_config(tmp_path, self._SHARED_CONFIG)
        with patch("agent.auxiliary_client.OpenAI") as mock_openai, \
                patch("openai.AsyncOpenAI") as mock_async:
            mock_openai.return_value = SimpleNamespace(
                api_key="k", base_url="https://shared.example.com/v1"
            )
            mock_async.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client(
                "tenant-b", "test-model", async_mode=True
            )

        assert mock_async.called
        headers = mock_async.call_args.kwargs.get("default_headers", {}) or {}
        assert "X-Tenant" not in headers
        assert "Authorization" not in headers
