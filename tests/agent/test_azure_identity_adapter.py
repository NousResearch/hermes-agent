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
from types import SimpleNamespace

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




# ---------------------------------------------------------------------------
# Cache fingerprint + http-bearer helpers
# ---------------------------------------------------------------------------




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
        self.scoped_calls = []

    def DefaultAzureCredential(self, **kwargs):  # noqa: N802 — match SDK
        self.last_credential_kwargs = kwargs
        self.credential_count += 1
        return SimpleNamespace(
            get_token=lambda scope: SimpleNamespace(token="fake-jwt", expires_on=9999999999),
            kwargs=kwargs,
        )

    def ClientSecretCredential(self, tenant_id, client_id, client_secret):  # noqa: N802
        self.scoped_calls.append(("client_secret", tenant_id, client_id, client_secret))
        return SimpleNamespace(kind="client_secret", tenant_id=tenant_id, client_id=client_id)

    def WorkloadIdentityCredential(self, **kwargs):  # noqa: N802
        self.scoped_calls.append(("workload_identity", kwargs))
        return SimpleNamespace(kind="workload_identity", kwargs=kwargs)

    def ManagedIdentityCredential(self, **kwargs):  # noqa: N802
        self.scoped_calls.append(("managed_identity", kwargs))
        return SimpleNamespace(kind="managed_identity", kwargs=kwargs)

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
        ClientSecretCredential=fake.ClientSecretCredential,
        WorkloadIdentityCredential=fake.WorkloadIdentityCredential,
        ManagedIdentityCredential=fake.ManagedIdentityCredential,
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


class TestScopedCredential:
    """A served multiplex profile never mints the launch profile's ambient chain (#116313)."""

    def test_two_homes_multiplex_refuses_ambient_chain_and_keeps_standalone(
        self, fake_azure_identity, tmp_path, monkeypatch,
    ):
        """A -> B -> A over two real homes: A (own AZURE_* in .env) builds its ClientSecretCredential,
        cred-less B is refused instead of getting DefaultAzureCredential (which reads A's AZURE_* from
        the process env), A again is unaffected; the probe thread runs under the caller's scope so the
        doctor path surfaces the same refusal. Control: a standalone run keeps the ambient chain."""
        from agent import secret_scope
        from agent.azure_identity_adapter import EntraIdentityConfig, _probe_token, build_credential
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home_a, home_b = tmp_path / "home-A", tmp_path / "home-B"
        for home in (home_a, home_b):
            home.mkdir()
        (home_a / ".env").write_text("AZURE_TENANT_ID=tenant-A\nAZURE_CLIENT_ID=client-A\nAZURE_CLIENT_SECRET=secret-A\n")
        (home_b / ".env").write_text("")
        # The launch profile's .env is in the process env, exactly what DefaultAzureCredential reads.
        monkeypatch.setenv("AZURE_TENANT_ID", "tenant-A")
        monkeypatch.setenv("AZURE_CLIENT_ID", "client-A")
        monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-A")
        config = EntraIdentityConfig()

        def in_scope(home, fn):
            h_tok = set_hermes_home_override(str(home))
            s_tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
            try:
                return fn()
            finally:
                secret_scope.reset_secret_scope(s_tok)
                reset_hermes_home_override(h_tok)

        # Control: standalone (no multiplex, no override) keeps today's ambient chain.
        assert build_credential(config).kwargs is not None
        assert fake_azure_identity.credential_count == 1

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        assert in_scope(home_a, lambda: build_credential(config)).client_id == "client-A"
        with pytest.raises(RuntimeError, match="refused for this profile"):
            in_scope(home_b, lambda: build_credential(config))
        probe = in_scope(home_b, lambda: _probe_token(config, 5.0))
        assert "refused for this profile" in probe["error"]
        assert in_scope(home_a, lambda: build_credential(config)).client_id == "client-A"
        # Absence on the wrong side: B never reached the ambient chain.
        assert fake_azure_identity.credential_count == 1


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
        from pm import InstallError as FeatureUnavailable

        def _fake_ensure(*args, **kwargs):
            raise FeatureUnavailable(
                "azure-identity",
                "lazy installs disabled (test simulation)",
            )

        # The adapter calls ``ensure_import`` from ``pm``; intercept
        # it by patching the actual symbol path.
        monkeypatch.setattr("pm.ensure_import", _fake_ensure)

        with pytest.raises(ImportError) as exc_info:
            _adapter._require_azure_identity()
        assert "azure-identity" in str(exc_info.value)


# ---------------------------------------------------------------------------
# has_azure_identity_credentials probe (timeout-bounded)
# ---------------------------------------------------------------------------


class TestHasAzureIdentityCredentials:

    def test_lazy_install_triggered_when_package_missing(self, monkeypatch):
        """With allow_install=True (default), the probe must trigger the
        lazy-install path before bailing — otherwise the wizard's
        ``preflight`` would silently fail for fresh installs that haven't
        enabled the Azure identity extra yet."""
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
        assert "hermes pm install --extra azure-identity" in info["hint"]

    def test_reports_env_sources_for_managed_identity(self, fake_azure_identity, monkeypatch):
        from agent.azure_identity_adapter import describe_active_credential
        monkeypatch.setenv("IDENTITY_ENDPOINT", "http://169.254.169.254")
        info = describe_active_credential(scope="https://x/.default", timeout_seconds=0.5)
        assert info["ok"] is True
        sources = info.get("env_sources") or []
        assert any("ManagedIdentity" in s for s in sources)



