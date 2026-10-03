"""Auxiliary-client routing of Claude on the shared ``vertex`` provider.

Vertex Model Garden hosts both Google Gemini (OpenAI-compat aggregator) and
Anthropic Claude (native Messages at ``publishers/anthropic/models/*:rawPredict``).
One provider name, and the model name picks the wire protocol inside
``_build_vertex_client``, mirroring the main-agent dispatch in
:func:`hermes_cli.runtime_provider.resolve_runtime_provider` so auxiliary calls
behave identically.

All tests mock the credential seams (``has_*_credentials`` + ``get_*_config``)
and the SDK factories, so they run hermetically without live GCP access.
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
    return creds, "my-gcp-project"


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
# Anthropic-on-Vertex dispatch
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
        assert "my-gcp-project" in client.base_url

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
        The gateway path is the one that hits it: agent init strips the
        ``anthropic/`` prefix before recording runtime state, so auxiliary
        tasks see the bare name."""
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
# Gemini-on-Vertex dispatch (regression on the existing path)
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
