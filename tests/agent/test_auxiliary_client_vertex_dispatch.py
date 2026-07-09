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
