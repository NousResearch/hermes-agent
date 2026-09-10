"""Tests for the Microsoft Entra ID adapter (agent/azure_identity_adapter.py).

Covers:
  - Scope resolution per Azure host shape
  - Display masking for callable + string + None inputs
  - Cache-fingerprint stability under callable refresh
  - is_token_provider truthiness on callables vs strings
  - EntraIdentityConfig serialization round-trip
  - Token provider construction with mocked azure-identity
  - Credential cache reuse + reset
  - has_azure_identity_credentials timeout / failure paths
  - describe_active_credential structural reporting
  - Lazy-install error path when azure-identity absent + lazy installs
    disabled

We mock azure.identity at the import boundary rather than hitting any
real Azure endpoint. Tests must remain hermetic per AGENTS.md.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

# Ensure we always import a fresh adapter module — credential caches in
# the adapter persist across tests otherwise, polluting assertions
# about cache invalidation.
@pytest.fixture(autouse=True)
def _reset_adapter_cache():
    from agent.azure_identity_adapter import reset_credential_cache
    reset_credential_cache()
    yield
    reset_credential_cache()


# ---------------------------------------------------------------------------
# Scope constant
# ---------------------------------------------------------------------------


class TestEntraScopeConstant:
    """Pin the Microsoft-documented Foundry inference scope.

    Microsoft's official samples for both ``*.openai.azure.com`` and
    ``*.services.ai.azure.com`` use ``https://ai.azure.com/.default``.
    The older ``cognitiveservices.azure.com/.default`` is the
    control-plane scope and is rejected for inference by newer
    Azure OpenAI / Foundry resources.

    Users with sovereign-cloud or unusual-tenant requirements pass the
    scope explicitly via ``model.entra.scope`` in ``config.yaml``.

    Refs:
      * https://learn.microsoft.com/azure/ai-foundry/openai/how-to/managed-identity
      * https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id
    """

    def test_default_scope_matches_microsoft_documentation(self):
        from agent.azure_identity_adapter import SCOPE_AI_AZURE_DEFAULT
        assert SCOPE_AI_AZURE_DEFAULT == "https://ai.azure.com/.default"


# ---------------------------------------------------------------------------
# Cache fingerprint + http-bearer helpers
# ---------------------------------------------------------------------------


class TestMaterializeBearerForHttp:
    """The only helper that mints a real bearer JWT — must call the
    callable exactly once and never fall through to display masking."""

    def test_callable_is_invoked_and_returns_token(self):
        from agent.azure_identity_adapter import materialize_bearer_for_http

        invoked = {"count": 0}

        def provider():
            invoked["count"] += 1
            return "fresh-jwt"

        assert materialize_bearer_for_http(provider) == "fresh-jwt"
        assert invoked["count"] == 1



    def test_empty_string_raises(self):
        from agent.azure_identity_adapter import materialize_bearer_for_http
        with pytest.raises(ValueError):
            materialize_bearer_for_http("")
        with pytest.raises(ValueError):
            materialize_bearer_for_http(None)


# ---------------------------------------------------------------------------
# build_bearer_http_client — the Anthropic-on-Foundry bridge
# ---------------------------------------------------------------------------


class TestBuildBearerHttpClient:
    """``build_bearer_http_client`` returns an ``httpx.Client`` whose
    request event hook mints a fresh JWT per outbound request. This is
    how Entra ID auth reaches the Anthropic SDK (which does not accept
    callable ``auth_token``)."""


    def test_hook_overrides_authorization_header(self):
        import httpx
        from agent.azure_identity_adapter import build_bearer_http_client

        minted_tokens = []

        def provider():
            minted_tokens.append(f"jwt-{len(minted_tokens) + 1}")
            return minted_tokens[-1]

        client = build_bearer_http_client(provider)
        try:
            hook = client.event_hooks["request"][0]
            # Build a request with conflicting pre-set headers and verify
            # the hook strips them and installs the fresh bearer.
            req = httpx.Request(
                "POST", "https://example.com/v1/messages",
                headers={
                    "Authorization": "Bearer stale-token",
                    "api-key": "static-key",
                    "x-api-key": "static-key",
                },
                json={"hello": "world"},
            )
            hook(req)
            assert req.headers["Authorization"] == "Bearer jwt-1"
            # The static-key headers must be stripped — sending both
            # auth values would be ambiguous on Azure.
            assert "api-key" not in req.headers
            assert "x-api-key" not in req.headers

            # Second invocation mints a fresh token.
            req2 = httpx.Request("GET", "https://example.com/v1/models")
            hook(req2)
            assert req2.headers["Authorization"] == "Bearer jwt-2"
            assert len(minted_tokens) == 2
        finally:
            client.close()

    def test_hook_strips_auth_headers_and_warns_when_token_provider_fails(self, caplog):
        """When the token provider fails (chain exhausted, IMDS down, az
        login expired), the hook must:
          1. Log at WARNING level so the misconfiguration is visible at
             default log level (not buried at DEBUG).
          2. Strip any pre-set Authorization headers — including the
             placeholder ``entra-id-bearer-via-http-hook`` sentinel that
             :func:`_build_anthropic_client_with_bearer_hook` sets on the
             Anthropic SDK constructor. This produces a clean
             "missing auth" 401 from Azure rather than a sentinel-bearing
             401 that's harder to diagnose AND avoids leaking the
             sentinel string into upstream access logs.
        """
        import logging
        import httpx
        from agent.azure_identity_adapter import build_bearer_http_client

        def bad_provider():
            return ""  # empty token → materialize_bearer_for_http raises

        client = build_bearer_http_client(bad_provider)
        try:
            hook = client.event_hooks["request"][0]
            req = httpx.Request(
                "POST", "https://example.com/v1/messages",
                headers={
                    "Authorization": "Bearer entra-id-bearer-via-http-hook",
                    "api-key": "leaked-placeholder",
                },
            )
            with caplog.at_level(logging.WARNING, logger="agent.azure_identity_adapter"):
                hook(req)  # Must not raise.
            # Pre-set auth headers stripped — no sentinel makes it to Azure.
            assert "Authorization" not in req.headers
            assert "api-key" not in req.headers
            # WARNING was logged so the user sees the misconfiguration.
            assert any(
                rec.levelno == logging.WARNING and "Entra ID token provider" in rec.message
                for rec in caplog.records
            )
        finally:
            client.close()




class TestIsTokenProvider:
    def test_callable_is_token_provider(self):
        from agent.azure_identity_adapter import is_token_provider
        assert is_token_provider(lambda: "x") is True

    def test_string_is_not_token_provider(self):
        from agent.azure_identity_adapter import is_token_provider
        assert is_token_provider("static-key") is False
        # ``str`` instances are technically callable in some edge cases
        # — confirm they're never classified as token providers.
        assert is_token_provider("") is False


# ---------------------------------------------------------------------------
# EntraIdentityConfig
# ---------------------------------------------------------------------------


class TestEntraIdentityConfig:
    """The serializable config that crosses multiprocessing boundaries —
    must round-trip through dict cleanly and never lose fields."""

    def test_to_dict_round_trip(self):
        from agent.azure_identity_adapter import EntraIdentityConfig
        cfg = EntraIdentityConfig(
            scope="https://ai.azure.com/.default",
            exclude_interactive_browser=False,
        )
        rebuilt = EntraIdentityConfig.from_dict(cfg.to_dict())
        assert rebuilt == cfg





    def test_dataclass_is_frozen(self):
        # Frozen dataclasses are hashable / safe to pass through caches.
        from agent.azure_identity_adapter import EntraIdentityConfig
        cfg = EntraIdentityConfig()
        with pytest.raises((AttributeError, Exception)):
            setattr(cfg, "scope", "mutated")


# ---------------------------------------------------------------------------
# Credential / token provider construction
# ---------------------------------------------------------------------------


class _FakeAzureIdentity:
    """Stand-in for the ``azure.identity`` module.

    Captures kwargs passed to ``DefaultAzureCredential`` so tests can
    assert how config flows into the SDK.
    """

    def __init__(self):
        self.last_credential_kwargs = None
        self.last_scope = None
        self.credential_count = 0

    def DefaultAzureCredential(self, **kwargs):  # noqa: N802 — match SDK
        self.last_credential_kwargs = kwargs
        self.credential_count += 1
        return SimpleNamespace(
            get_token=lambda scope: SimpleNamespace(token="fake-jwt", expires_on=9999999999),
            kwargs=kwargs,
        )

    def get_bearer_token_provider(self, credential, scope):
        self.last_scope = scope
        # Return a callable that mints a token when invoked.
        return lambda: f"jwt-for-{scope}"


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Install a fake azure.identity into sys.modules and stub the
    adapter's `_require_azure_identity` so all tests use the fake."""
    fake = _FakeAzureIdentity()

    fake_module = SimpleNamespace(
        DefaultAzureCredential=fake.DefaultAzureCredential,
        get_bearer_token_provider=fake.get_bearer_token_provider,
    )
    monkeypatch.setitem(sys.modules, "azure", SimpleNamespace(identity=fake_module))
    monkeypatch.setitem(sys.modules, "azure.identity", fake_module)

    # The adapter's `_require_azure_identity` does its own import, so
    # patch that too to make sure tests never hit the real package's
    # singleton state.
    from agent import azure_identity_adapter as _adapter
    monkeypatch.setattr(_adapter, "_require_azure_identity", lambda: fake_module)

    return fake


class TestBuildCredential:
    def test_default_kwargs_are_minimal(self, fake_azure_identity):
        """SDK default for ``exclude_interactive_browser_credential`` is
        True; we only pass it when the user opts IN to interactive
        browser auth. Tenant / authority / service principal config
        flow through the standard ``AZURE_*`` env vars (read by
        azure-identity directly), not Hermes config kwargs."""
        from agent.azure_identity_adapter import EntraIdentityConfig, build_credential
        cred = build_credential(EntraIdentityConfig())
        kwargs = fake_azure_identity.last_credential_kwargs
        # Default config should produce empty kwargs — SDK uses its own
        # defaults plus env-var-driven settings.
        assert kwargs == {}
        assert cred is not None


    def test_credential_is_cached_per_config(self, fake_azure_identity):
        from agent.azure_identity_adapter import EntraIdentityConfig, build_credential
        cfg = EntraIdentityConfig(scope="s1")
        c1 = build_credential(cfg)
        c2 = build_credential(cfg)
        assert c1 is c2
        assert fake_azure_identity.credential_count == 1

    def test_distinct_configs_get_distinct_credentials(self, fake_azure_identity):
        from agent.azure_identity_adapter import EntraIdentityConfig, build_credential
        c1 = build_credential(EntraIdentityConfig(scope="s1"))
        c2 = build_credential(EntraIdentityConfig(scope="s2"))
        assert c1 is not c2
        assert fake_azure_identity.credential_count == 2



class TestBuildTokenProvider:
    def test_returns_callable_for_scope(self, fake_azure_identity):
        from agent.azure_identity_adapter import build_token_provider
        provider = build_token_provider(scope="https://ai.azure.com/.default")
        assert callable(provider)
        assert provider() == "jwt-for-https://ai.azure.com/.default"
        assert fake_azure_identity.last_scope == "https://ai.azure.com/.default"



    def test_config_object_wins_over_kwargs(self, fake_azure_identity):
        from agent.azure_identity_adapter import (
            EntraIdentityConfig,
            build_token_provider,
        )
        cfg = EntraIdentityConfig(scope="cfg-scope")
        build_token_provider(scope="ignored", config=cfg)
        assert fake_azure_identity.last_scope == "cfg-scope"
        assert fake_azure_identity.last_credential_kwargs == {}


# ---------------------------------------------------------------------------
# Lazy-install / missing-package surface
# ---------------------------------------------------------------------------


class TestRequireAzureIdentityMissing:
    def test_clear_error_when_lazy_install_disabled(self, monkeypatch):
        """When azure-identity isn't importable AND lazy installs are
        off, the adapter must raise ImportError with an actionable
        message, not propagate FeatureUnavailable."""
        from agent import azure_identity_adapter as _adapter

        # Force the import path to fail.
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__
        def _fake_import(name, *args, **kwargs):
            if name == "azure.identity" or name.startswith("azure.identity."):
                raise ImportError("simulated missing azure-identity")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fake_import)

        # Simulate lazy installs disabled.
        from tools.lazy_deps import FeatureUnavailable

        def _fake_ensure(*args, **kwargs):
            raise FeatureUnavailable(
                "provider.azure_identity",
                ("azure-identity==1.25.3",),
                "lazy installs disabled (test simulation)",
            )

        # The adapter calls ``ensure`` from ``tools.lazy_deps``; intercept
        # it by patching the actual symbol path.
        monkeypatch.setattr("tools.lazy_deps.ensure", _fake_ensure)

        with pytest.raises(ImportError) as exc_info:
            _adapter._require_azure_identity()
        msg = str(exc_info.value)
        assert "azure-identity" in msg
        assert "Foundry" in msg or "foundry" in msg.lower()


# ---------------------------------------------------------------------------
# has_azure_identity_credentials probe (timeout-bounded)
# ---------------------------------------------------------------------------


class TestHasAzureIdentityCredentials:

    def test_lazy_install_triggered_when_package_missing(self, monkeypatch):
        """With allow_install=True (default), the probe must trigger the
        lazy-install path before bailing — otherwise the wizard's
        ``preflight`` would silently fail for fresh installs that haven't
        run ``pip install azure-identity`` yet."""
        from agent import azure_identity_adapter as _adapter

        installed = {"called": False}

        def _fake_install():
            installed["called"] = True
            # After install, pretend the package is now importable.
            monkeypatch.setattr(_adapter, "has_azure_identity_installed", lambda: True)
            return SimpleNamespace(
                DefaultAzureCredential=lambda **kw: SimpleNamespace(
                    kwargs=kw,
                    get_token=lambda scope: SimpleNamespace(token="post-install-jwt", expires_on=0),
                ),
                get_bearer_token_provider=lambda c, s: lambda: "x",
            )

        monkeypatch.setattr(_adapter, "has_azure_identity_installed", lambda: False)
        monkeypatch.setattr(_adapter, "_require_azure_identity", _fake_install)

        # Provide a credential factory so the probe proceeds after install.
        monkeypatch.setattr(
            _adapter, "build_credential",
            lambda config: SimpleNamespace(
                get_token=lambda scope: SimpleNamespace(token="probe-jwt", expires_on=0),
            ),
        )

        result = _adapter.has_azure_identity_credentials(
            "https://x/.default", timeout_seconds=0.5,
        )
        assert installed["called"] is True, (
            "has_azure_identity_credentials must trigger lazy install "
            "before bailing"
        )
        assert result is True



    def test_returns_false_on_timeout(self, monkeypatch):
        """Slow IMDS / network must time out, not hang the caller."""
        import threading
        from agent import azure_identity_adapter as _adapter

        slow_release = threading.Event()

        def _slow_credential(_config):
            class _Cred:
                def get_token(self, scope):
                    # Block forever from the test's perspective; the
                    # adapter must give up via its thread-bounded probe.
                    slow_release.wait(timeout=10)
                    return SimpleNamespace(token="never-returned", expires_on=0)
            return _Cred()

        monkeypatch.setattr(_adapter, "build_credential", _slow_credential)
        monkeypatch.setattr(_adapter, "has_azure_identity_installed", lambda: True)
        try:
            assert _adapter.has_azure_identity_credentials(
                "https://x/.default", timeout_seconds=0.1
            ) is False
        finally:
            slow_release.set()


# ---------------------------------------------------------------------------
# describe_active_credential — used by hermes doctor + hermes auth
# ---------------------------------------------------------------------------


class TestDescribeActiveCredential:

    def test_reports_install_failure(self, monkeypatch):
        """When lazy install is allowed but fails (e.g. lazy installs
        disabled), the diagnostic surfaces the failure as the error."""
        from agent import azure_identity_adapter as _adapter
        monkeypatch.setattr(_adapter, "has_azure_identity_installed", lambda: False)

        def _fail_install():
            raise ImportError("simulated: lazy installs disabled")

        monkeypatch.setattr(_adapter, "_require_azure_identity", _fail_install)
        info = _adapter.describe_active_credential(
            scope="https://x/.default", allow_install=True,
        )
        assert info["ok"] is False
        assert "lazy installs disabled" in info["error"]
        assert "lazy" in info["hint"].lower()

    def test_reports_env_sources_for_managed_identity(self, fake_azure_identity, monkeypatch):
        from agent.azure_identity_adapter import describe_active_credential
        monkeypatch.setenv("IDENTITY_ENDPOINT", "http://169.254.169.254")
        info = describe_active_credential(scope="https://x/.default", timeout_seconds=0.5)
        assert info["ok"] is True
        sources = info.get("env_sources") or []
        assert any("ManagedIdentity" in s for s in sources)



