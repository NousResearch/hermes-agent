"""Tests for auxiliary-client routing of the ``vertex`` provider.

Covers the plugin-catalog fallback in
``agent.auxiliary_client.resolve_provider_client``:

  ``PROVIDER_REGISTRY`` in :mod:`hermes_cli.auth` auto-extends from the
  provider-plugin catalog, but the extension's filter (``auth_type !=
  "api_key" or not env_vars``) intentionally excludes non-api_key providers
  (vertex, aws_sdk, oauth_*). Without a fallback, the resolver
  short-circuits at the registry lookup with "unknown provider" and the
  existing ``elif pconfig.auth_type == "vertex":`` handler below is dead
  code. Every auxiliary task (vision, compression, curator,
  session_search) silently breaks on ``provider: vertex`` deployments.

  The fallback consults ``providers.get_provider_profile`` directly and
  synthesizes a minimal pconfig so the downstream ``auth_type`` dispatch
  can run against the plugin-catalog profile.

All tests mock the credential seams (``has_vertex_credentials`` +
``get_vertex_config``) so they run hermetically without live GCP
dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Plugin-catalog fallback reaches the vertex handler
# ---------------------------------------------------------------------------


class TestPluginCatalogFallback:
    """The whole point of the fix — ``provider="vertex"`` must reach the
    vertex handler even though vertex is not in ``PROVIDER_REGISTRY``."""

    def test_vertex_provider_reaches_dispatch_branch(self):
        """Without the fallback this call returns (None, None) silently.
        With the fallback it reaches the Gemini branch and constructs a
        real OpenAI-compat client."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            client, model = resolve_provider_client(
                "vertex", "google/gemini-3.1-pro-preview",
            )
        assert client is not None, (
            "Regression: vertex fell off the registry lookup and never "
            "reached the auth_type == 'vertex' handler."
        )
        assert model == "google/gemini-3.1-pro-preview"

    def test_unknown_provider_still_returns_none(self):
        """The fallback must not swallow genuinely-unknown providers —
        a typo like ``verttex`` should still bail with (None, None) so
        callers can fall back to their auto chain."""
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("nonexistent-fake-provider")
        assert client is None
        assert model is None

    def test_vertex_alias_reaches_dispatch(self):
        """The vertex provider profile registers aliases (``google-vertex``,
        ``vertex-ai``, ``gcp-vertex``). ``get_provider_profile`` resolves
        them, so the fallback should reach the same handler for aliases."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            for alias in ("google-vertex", "vertex-ai", "gcp-vertex"):
                client, _ = resolve_provider_client(
                    alias, "google/gemini-3.1-pro-preview",
                )
                assert client is not None, (
                    f"Alias {alias!r} did not reach the vertex dispatch "
                    "branch. The fallback must resolve aliases via "
                    "get_provider_profile, not just canonical names."
                )


# ---------------------------------------------------------------------------
# Vertex + ``google/`` slug or empty model → OpenAI-compat aggregator
# ---------------------------------------------------------------------------


class TestVertexGeminiDispatch:
    """The pre-existing vertex handler serves Gemini via the OpenAI-compat
    endpoint with an OAuth2 bearer token. Once the plugin-catalog fallback
    makes it reachable, these tests pin its behaviour."""

    def test_google_prefix_builds_openai_client(self):
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            client, model = resolve_provider_client(
                "vertex", "google/gemini-3.1-pro-preview",
            )

        assert isinstance(client, OpenAI)
        assert model == "google/gemini-3.1-pro-preview"
        assert client.api_key == "mocked-token"
        assert "aiplatform.googleapis.com" in str(client.base_url)

    def test_no_model_falls_through_to_gemini_default(self):
        """No caller-supplied model → the default aux Gemini slug picks
        up. ``resolve_vision_provider_client``'s auto branch relies on
        this to stand up a client on machines where ``auxiliary.vision``
        isn't configured."""
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            client, model = resolve_provider_client("vertex")

        assert isinstance(client, OpenAI)
        assert model.startswith("google/")

    def test_bare_gemini_slug_falls_to_gemini_handler(self):
        """Bare ``gemini-*`` (no ``google/`` prefix) still resolves to
        the OpenAI-compat aggregator — the vertex handler doesn't
        rewrite the model. Vertex's Gemini endpoint requires the
        ``google/`` prefix and will 404 the bare form, which is the
        intended loud-fail behaviour."""
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            client, model = resolve_provider_client(
                "vertex", "gemini-3.1-pro-preview",
            )

        assert isinstance(client, OpenAI)
        assert "claude" not in (model or "").lower()

    def test_missing_gcp_credentials_returns_none(self):
        from agent.auxiliary_client import resolve_provider_client

        with patch("agent.vertex_adapter.has_vertex_credentials",
                   return_value=False):
            client, model = resolve_provider_client(
                "vertex", "google/gemini-3.1-pro-preview",
            )
        assert client is None
        assert model is None

    def test_missing_oauth_token_returns_none(self):
        """Credentials configured but token mint fails at call time."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=(None, None)),
        ):
            client, model = resolve_provider_client(
                "vertex", "google/gemini-3.1-pro-preview",
            )
        assert client is None
        assert model is None

    def test_async_mode_wraps_in_async_openai(self):
        from agent.auxiliary_client import resolve_provider_client
        from openai import AsyncOpenAI

        with (
            patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            patch("agent.vertex_adapter.get_vertex_config",
                  return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
        ):
            client, _ = resolve_provider_client(
                "vertex", "google/gemini-3.1-pro-preview", async_mode=True,
            )

        assert isinstance(client, AsyncOpenAI)


# ---------------------------------------------------------------------------
# Historical regression — the bug this fix closes
# ---------------------------------------------------------------------------


class TestHistoricalRegression:
    """Pin the exact silent-break so a future refactor of the
    ``PROVIDER_REGISTRY`` auto-extension in ``hermes_cli/auth.py`` cannot
    reintroduce it."""

    def test_vertex_not_in_hardcoded_registry_still_works(self):
        """The bug was: ``PROVIDER_REGISTRY.get("vertex")`` returns None,
        so ``elif pconfig.auth_type == "vertex":`` was dead. This test
        pins the invariant even if someone later re-declares vertex in
        ``PROVIDER_REGISTRY`` (belt + braces)."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        from agent.auxiliary_client import resolve_provider_client

        # Simulate the historical state — vertex explicitly absent from
        # the registry. Even in this state, resolve_provider_client must
        # succeed via the plugin-catalog fallback.
        original_vertex = PROVIDER_REGISTRY.pop("vertex", None)
        try:
            with (
                patch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
                patch("agent.vertex_adapter.get_vertex_config",
                      return_value=("mocked-token", "https://aiplatform.googleapis.com/x")),
            ):
                client, model = resolve_provider_client(
                    "vertex", "google/gemini-3.1-pro-preview",
                )
            assert client is not None, (
                "This is exactly the historical bug — vertex silently "
                "resolves to (None, None) because the plugin-catalog "
                "fallback was removed or the filter widened."
            )
            assert model == "google/gemini-3.1-pro-preview"
        finally:
            if original_vertex is not None:
                PROVIDER_REGISTRY["vertex"] = original_vertex

# ---------------------------------------------------------------------------
# _resolve_auto — vertex reaches the vertex handler through the full chain
# ---------------------------------------------------------------------------


class TestResolveAutoVertex:
    """The plugin-catalog fallback must be reachable through the full
    ``_resolve_auto`` chain that every auxiliary task uses in practice —
    not just the direct ``resolve_provider_client`` entry point that the
    other tests hit.

    Prior to the fix, an aux task on a vertex-only deployment:
      1. ``_resolve_auto`` reads ``main.provider == "vertex"``,
         ``main.model == "google/gemini-3-pro-preview"``.
      2. Calls ``resolve_provider_client("vertex", ...)``.
      3. Registry lookup returns None → the elif-chain never fires.
      4. Step 1 returns ``(None, None)``.
      5. Fallback chain (Step 2: OpenRouter → Nous → custom → Codex →
         API-key providers) runs. On a vertex-only fleet none of these
         have credentials.
      6. Chain terminates in ``RuntimeError: No LLM provider configured
         for task=<task> provider=auto. Run: hermes setup``.

    After the fix, Step 1 succeeds and the aux task runs on the same
    Gemini model the operator picked for chat."""

    def test_vertex_main_provider_reaches_aux_client(self):
        from unittest.mock import MagicMock, patch as mpatch

        from agent.auxiliary_client import _resolve_auto

        with (
            mpatch("agent.auxiliary_client._read_main_provider",
                   return_value="vertex"),
            mpatch("agent.auxiliary_client._read_main_model",
                   return_value="google/gemini-3-pro-preview"),
            mpatch("agent.vertex_adapter.has_vertex_credentials", return_value=True),
            mpatch("agent.vertex_adapter.get_vertex_config",
                   return_value=("mocked-token",
                                 "https://aiplatform.googleapis.com/x")),
        ):
            client, model = _resolve_auto()

        assert client is not None, (
            "Regression: _resolve_auto Step 1 (main provider + main model) "
            "returned no client for provider=vertex, meaning "
            "resolve_provider_client('vertex', ...) fell through to "
            "(None, None) — the exact silent-break the plugin-catalog "
            "fallback fixes."
        )
        assert model == "google/gemini-3-pro-preview"


# ---------------------------------------------------------------------------
# _refresh_provider_credentials — vertex branch clears the module cache
# ---------------------------------------------------------------------------


class TestRefreshProviderCredentialsVertex:
    """The cached-Vertex-client stale-token problem @teknium1 flagged.

    Vertex mints OAuth2 access tokens via google-auth and caches the
    Credentials object in ``vertex_adapter._creds_cache``. That object
    auto-refreshes on read when < 5min from expiry — but the auxiliary
    layer caches OpenAI clients with the token baked in as ``api_key``,
    so the cached client keeps the stale token even after Credentials
    refresh and 401s until the aux-client cache is evicted.

    ``_refresh_provider_credentials`` is invoked from the auth-retry
    paths (``_call_fallback_candidate_*``, sync + async main-agent
    retry). Without a ``vertex`` branch, aux tasks on long-lived
    sessions could not recover from a stale token and had to wait for
    process restart. This test pins the branch's contract.
    """

    def test_refresh_clears_module_cache_and_evicts_aux_clients(self):
        from unittest.mock import patch as mpatch

        from agent.auxiliary_client import _refresh_provider_credentials
        from agent import vertex_adapter

        # Prime the module cache with a stale entry so we can observe the
        # clear() call.
        vertex_adapter._creds_cache["__adc__"] = (object(), "stale-project")

        with (
            mpatch("agent.vertex_adapter.get_vertex_config",
                   return_value=("fresh-token",
                                 "https://aiplatform.googleapis.com/x")),
            mpatch("agent.auxiliary_client._evict_cached_clients") as evict,
        ):
            ok = _refresh_provider_credentials("vertex")

        assert ok is True
        assert "__adc__" not in vertex_adapter._creds_cache, (
            "Cache entry must be cleared so the next get_vertex_config() "
            "call re-mints from scratch — the whole point of the branch."
        )
        evict.assert_called_once_with("vertex")

    def test_refresh_returns_false_when_token_mint_fails(self):
        """Cache clear happens, but if the fresh mint fails (revoked
        creds, network blip), the refresh returns False so the caller
        can bail cleanly rather than serve a stale response."""
        from unittest.mock import patch as mpatch

        from agent.auxiliary_client import _refresh_provider_credentials

        with (
            mpatch("agent.vertex_adapter.get_vertex_config",
                   return_value=(None, None)),
            mpatch("agent.auxiliary_client._evict_cached_clients") as evict,
        ):
            ok = _refresh_provider_credentials("vertex")

        assert ok is False
        evict.assert_not_called()

    def test_refresh_bails_gracefully_if_vertex_adapter_missing(self):
        """The google-auth / vertex_adapter import can fail on a
        minimal install. Refresh must return False rather than raise."""
        from unittest.mock import patch as mpatch

        from agent.auxiliary_client import _refresh_provider_credentials

        # Simulate ImportError by nulling the adapter in sys.modules.
        import sys
        original = sys.modules.pop("agent.vertex_adapter", None)
        sys.modules["agent.vertex_adapter"] = None  # type: ignore[assignment]
        try:
            with mpatch("agent.auxiliary_client._evict_cached_clients") as evict:
                ok = _refresh_provider_credentials("vertex")
            assert ok is False
            evict.assert_not_called()
        finally:
            if original is not None:
                sys.modules["agent.vertex_adapter"] = original
            else:
                sys.modules.pop("agent.vertex_adapter", None)


# ---------------------------------------------------------------------------
# _auth_refresh_provider_for_route — global + regional Vertex hosts
# ---------------------------------------------------------------------------


class TestAuthRefreshProviderRouteVertex:
    """When an auto-routed aux call selects a concrete Vertex client, the
    refresh helper needs to infer ``provider="vertex"`` from the client's
    base URL so a 401 retry can force a fresh token. The subtlety: Vertex
    uses TWO host shapes — a bare ``aiplatform.googleapis.com`` (global
    location) and ``{region}-aiplatform.googleapis.com`` (regional
    locations, e.g. ``us-central1-aiplatform...``). The regional form is
    NOT a subdomain of the bare form (no dot between region and
    ``aiplatform``), so ``base_url_host_matches`` alone doesn't catch it.
    """

    def test_global_vertex_host_returns_vertex(self):
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route(
            "auto",
            "https://aiplatform.googleapis.com/v1beta1/projects/p/"
            "locations/global/endpoints/openapi",
        ) == "vertex"

    def test_regional_vertex_host_returns_vertex(self):
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route(
            "auto",
            "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/"
            "locations/us-central1/endpoints/openapi",
        ) == "vertex"

    def test_look_alike_host_does_not_match(self):
        """A malicious or misconfigured host like ``fake-aiplatform.googleapis.com.evil.com``
        must NOT be classified as vertex."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route(
            "auto",
            "https://fake-aiplatform.googleapis.com.evil.com/v1",
        ) != "vertex"

    def test_resolved_provider_wins_over_url_inference(self):
        """When resolved_provider is already concrete, URL inference is
        skipped — the caller knows better than the URL."""
        from agent.auxiliary_client import _auth_refresh_provider_for_route

        assert _auth_refresh_provider_for_route(
            "openrouter",
            "https://aiplatform.googleapis.com/v1",
        ) == "openrouter"
