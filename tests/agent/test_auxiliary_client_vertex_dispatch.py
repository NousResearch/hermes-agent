"""Tests for auxiliary-client routing of the ``vertex`` provider.

Covers the two-part fix in ``agent.auxiliary_client.resolve_provider_client``:

  Part A — plugin-catalog fallback for non-api_key providers.
    ``PROVIDER_REGISTRY`` in :mod:`hermes_cli.auth` auto-extends from the
    provider-plugin catalog, but the extension's filter (``auth_type !=
    "api_key" or not env_vars``) intentionally excludes vertex / bedrock /
    OAuth providers. Without the fallback, ``resolve_provider_client("vertex",
    ...)`` short-circuits at the registry lookup with "unknown provider" and
    the existing ``elif pconfig.auth_type == "vertex":`` handler below is
    dead code. Every auxiliary task (vision, compression, curator,
    session_search) silently breaks on ``provider: vertex`` deployments.

  Part B — Anthropic-vs-Gemini dispatch inside the vertex branch.
    Vertex Model Garden hosts both Google Gemini (OpenAI-compat aggregator)
    and Anthropic Claude (native Messages at
    ``publishers/anthropic/models/*:rawPredict``). One provider name, model
    prefix picks the wire protocol. Mirrors
    :func:`hermes_cli.runtime_provider.resolve_runtime_provider`'s main-agent
    dispatch so auxiliary calls behave identically.

All tests mock the two credential seams (``has_*_credentials`` +
``get_*_config`` / ``_resolve_google_credentials``) and the SDK factories
so they run hermetically without live GCP / Anthropic dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_google_credentials():
    """Return a stand-in for the ``(Credentials, project_id)`` tuple that
    :func:`agent.anthropic_vertex_adapter._resolve_google_credentials`
    returns on a real ADC-configured host."""
    creds = MagicMock(name="google_credentials")
    creds.token = "mocked-oauth-token"
    return creds, "test-project-42"


def _mock_anthropic_sdk():
    """Return a stand-in for the ``anthropic`` module with an
    ``AnthropicVertex`` class. Instances are MagicMocks so downstream
    ``real_client.messages.create(...)`` calls don't hit the wire."""
    sdk = MagicMock(name="anthropic_sdk")
    sdk.AnthropicVertex = MagicMock(
        return_value=MagicMock(name="AnthropicVertex_instance"),
    )
    return sdk


# ---------------------------------------------------------------------------
# Part A — plugin-catalog fallback reaches the vertex handler
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
            # Every alias resolves to the same "vertex" auth_type, so the
            # fallback shim reaches the Gemini branch.
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
# Part B — Anthropic-on-Vertex dispatch
# ---------------------------------------------------------------------------


class TestVertexAnthropicDispatch:
    """Vertex + ``anthropic/`` model → AnthropicVertex SDK path."""

    def _patched_anthropic_success(self):
        """Return the context manager stack for a happy AnthropicVertex
        path — credentials present, SDK importable, project resolved."""
        return (
            patch(
                "agent.anthropic_vertex_adapter.has_anthropic_vertex_credentials",
                return_value=True,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=_mock_google_credentials(),
            ),
            patch(
                "agent.anthropic_vertex_adapter._get_anthropic_sdk",
                return_value=_mock_anthropic_sdk(),
            ),
        )

    def test_anthropic_prefix_builds_anthropic_client(self):
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, model = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8", is_vision=True,
            )
        assert isinstance(client, AnthropicAuxiliaryClient), (
            "Expected AnthropicAuxiliaryClient wrapping the AnthropicVertex "
            f"SDK, got {type(client).__name__}."
        )

    def test_anthropic_model_prefix_stripped_in_stored_model(self):
        """The AnthropicVertex SDK expects the bare model id
        (``claude-opus-4-8``). ``_normalize_resolved_model`` should strip
        the ``anthropic/`` prefix before we hand it to the wrapper."""
        from agent.auxiliary_client import resolve_provider_client

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            _, model = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert model == "claude-opus-4-8"

    def test_anthropic_client_carries_vertex_placeholder_api_key(self):
        """``AnthropicAuxiliaryClient`` demands a non-empty ``api_key`` so
        downstream code that checks ``bool(client.api_key)`` treats
        Anthropic-on-Vertex as authenticated. The AnthropicVertex SDK
        mints its own OAuth tokens; the ``vertex-adc`` placeholder is the
        agreed sentinel (matches runtime_provider.py + agent_init.py)."""
        from agent.auxiliary_client import resolve_provider_client

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, _ = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert client.api_key == "vertex-adc"

    def test_anthropic_client_base_url_reports_vertex_endpoint(self):
        """The base_url on the wrapper is display-only (for logs and
        billing attribution). It must reflect the actual Vertex publisher
        endpoint shape so ``agent.usage_pricing``'s ``aiplatform.
        googleapis.com`` heuristic and log lines quoting the base URL
        both work."""
        from agent.auxiliary_client import resolve_provider_client

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, _ = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert "aiplatform.googleapis.com" in client.base_url
        assert "publishers/anthropic" in client.base_url
        assert "test-project-42" in client.base_url

    def test_anthropic_missing_gcp_credentials_returns_none(self):
        """No ADC / service-account JSON / vertex.project_id — return
        (None, None) so callers can fall through to their auto chain,
        rather than raising."""
        from agent.auxiliary_client import resolve_provider_client

        with patch(
            "agent.anthropic_vertex_adapter.has_anthropic_vertex_credentials",
            return_value=False,
        ):
            client, model = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert client is None
        assert model is None

    def test_anthropic_missing_project_id_returns_none(self):
        """Credentials present but project resolution fails (e.g. ADC
        with no embedded project + no ``vertex.project_id`` in config)."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.anthropic_vertex_adapter.has_anthropic_vertex_credentials",
                return_value=True,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=(MagicMock(), None),
            ),
        ):
            client, model = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert client is None
        assert model is None

    def test_anthropic_sdk_missing_returns_none(self):
        """anthropic package not installed (or too old to have
        ``AnthropicVertex``) — return (None, None), warn, don't raise."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.anthropic_vertex_adapter.has_anthropic_vertex_credentials",
                return_value=True,
            ),
            patch(
                "agent.anthropic_vertex_adapter._resolve_google_credentials",
                return_value=_mock_google_credentials(),
            ),
            patch(
                "agent.anthropic_vertex_adapter._get_anthropic_sdk",
                return_value=None,
            ),
        ):
            client, model = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8",
            )
        assert client is None
        assert model is None

    def test_uppercase_anthropic_prefix_still_dispatches_to_anthropic(self):
        """``is_anthropic_vertex_model`` is case-insensitive per its
        docstring — protect that contract at the auxiliary path."""
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, _ = resolve_provider_client(
                "vertex", "ANTHROPIC/claude-opus-4-8",
            )
        assert isinstance(client, AnthropicAuxiliaryClient)

    def test_bare_claude_slug_dispatches_to_anthropic_on_aux_path(self):
        """``agent_init.py::normalize_model_for_provider`` strips the
        ``anthropic/`` prefix from the runtime main model for
        provider=vertex. ``set_runtime_main`` then stores the BARE form
        (``claude-opus-4-8``), and every auxiliary read via
        ``_read_main_model()`` sees that bare form.

        The strict classifier ``is_anthropic_vertex_model`` intentionally
        rejects bare ``claude-*`` so main-agent config typos surface as a
        loud Vertex 404. The auxiliary vertex handler must widen
        detection to also match bare ``claude-*`` — otherwise the
        auxiliary path silently misroutes Claude calls to Vertex's
        OpenAI-compat Gemini endpoint and 400s with "Malformed publisher
        model" while the SAME session works fine on the main-agent path.
        Direct probes that pass ``anthropic/claude-...`` with the prefix
        intact would work; the gateway/runtime path (which sees the
        already-stripped bare form) would 400."""
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, model = resolve_provider_client(
                "vertex", "claude-opus-4-8", is_vision=True,
            )
        assert isinstance(client, AnthropicAuxiliaryClient), (
            "Bare 'claude-opus-4-8' must dispatch to AnthropicVertex on "
            "the auxiliary path — the runtime main model is stored bare "
            "after agent_init normalization, and any Claude-on-Vertex "
            "aux call reads that bare form."
        )
        assert model == "claude-opus-4-8"

    def test_bare_claude_case_insensitive(self):
        """Uppercase / mixed-case bare Claude slug also dispatches."""
        from agent.auxiliary_client import (
            AnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, _ = resolve_provider_client(
                "vertex", "Claude-Opus-4-8",
            )
        assert isinstance(client, AnthropicAuxiliaryClient)

    def test_async_mode_wraps_in_async_client(self):
        """``async_mode=True`` must return the async wrapper so async
        callers (compression, session_search) don't need to switch
        client types based on provider."""
        from agent.auxiliary_client import (
            AsyncAnthropicAuxiliaryClient,
            resolve_provider_client,
        )

        p1, p2, p3 = self._patched_anthropic_success()
        with p1, p2, p3:
            client, _ = resolve_provider_client(
                "vertex", "anthropic/claude-opus-4-8", async_mode=True,
            )
        assert isinstance(client, AsyncAnthropicAuxiliaryClient)


# ---------------------------------------------------------------------------
# Part B — Gemini-on-Vertex dispatch (regression on the existing path)
# ---------------------------------------------------------------------------


class TestVertexGeminiDispatch:
    """Vertex + ``google/`` or empty model → OpenAI-compat aggregator."""

    def test_google_prefix_builds_openai_client(self):
        """The pre-fix behaviour on the ``google/`` slug — protect it
        against accidental regression when the Anthropic dispatch was
        added on top."""
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

    def test_bare_gemini_slug_still_falls_to_gemini_aggregator(self):
        """Bare ``gemini-*`` (no ``google/`` prefix) still resolves to
        the OpenAI-compat aggregator — the Anthropic widening only
        matches ``claude-*``. Vertex's Gemini endpoint requires the
        ``google/`` prefix and will 404 the bare form, which is the
        intended loud-fail behaviour for the Gemini path."""
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
