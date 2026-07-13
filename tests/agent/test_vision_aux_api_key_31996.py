"""Regression tests for #31996 sub-issue 2 — auxiliary.vision.api_key plumbing.

Issue #31996 reports that setting ``auxiliary.vision.api_key`` in
``config.yaml`` does not actually authenticate the OpenRouter (or
similar) vision client — instead the auxiliary client falls back to
``os.getenv("OPENROUTER_API_KEY")`` and, when that env var is empty,
marks the provider unhealthy for 60s.

Root cause: ``_resolve_strict_vision_backend`` discarded the
``resolved_api_key`` value and called ``_try_openrouter(model=…)`` with
no ``explicit_api_key``, so the env-var lookup was the only auth source.

The fix (a) adds an ``api_key`` keyword argument to
``_resolve_strict_vision_backend`` and routes it through to
``_try_openrouter`` / ``_try_anthropic`` / ``resolve_provider_client``,
and (b) updates the explicit-provider branch of
``resolve_vision_provider_client`` to pass ``resolved_api_key`` down.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_strict_vision_backend forwards api_key to every backend
# ---------------------------------------------------------------------------


class TestResolveStrictVisionBackendForwardsApiKey:
    """The strict-vision dispatcher now plumbs ``api_key`` to each provider."""

    def test_signature_accepts_keyword_api_key(self):
        """``api_key`` must be a keyword-only parameter so callers self-document."""
        from agent.auxiliary_client import _resolve_strict_vision_backend
        sig = inspect.signature(_resolve_strict_vision_backend)
        assert "api_key" in sig.parameters
        param = sig.parameters["api_key"]
        # Keyword-only: no positional shadowing of older 2-arg call sites.
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    def test_openrouter_branch_passes_api_key_to_try_openrouter(self):
        from agent.auxiliary_client import _resolve_strict_vision_backend

        with patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(MagicMock(), "vision-model"),
        ) as mock_try:
            _resolve_strict_vision_backend(
                "openrouter", model="some-model", api_key="sk-or-v1-config"
            )

        mock_try.assert_called_once()
        kwargs = mock_try.call_args.kwargs
        assert kwargs.get("explicit_api_key") == "sk-or-v1-config"
        assert kwargs.get("model") == "some-model"

    def test_anthropic_branch_passes_api_key_to_try_anthropic(self):
        from agent.auxiliary_client import _resolve_strict_vision_backend

        with patch(
            "agent.auxiliary_client._try_anthropic",
            return_value=(MagicMock(), "claude-3"),
        ) as mock_try:
            _resolve_strict_vision_backend("anthropic", api_key="sk-ant-config")

        mock_try.assert_called_once()
        kwargs = mock_try.call_args.kwargs
        assert kwargs.get("explicit_api_key") == "sk-ant-config"

    def test_copilot_branch_passes_api_key_to_resolve_provider_client(self):
        from agent.auxiliary_client import _resolve_strict_vision_backend

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(), "gpt-4o"),
        ) as mock_rpc:
            _resolve_strict_vision_backend(
                "copilot", model="gpt-4o", api_key="ghu_token"
            )

        mock_rpc.assert_called_once()
        kwargs = mock_rpc.call_args.kwargs
        assert kwargs.get("explicit_api_key") == "ghu_token"
        assert kwargs.get("is_vision") is True

    def test_codex_branch_passes_api_key_to_resolve_provider_client(self):
        from agent.auxiliary_client import _resolve_strict_vision_backend

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(MagicMock(), "gpt-5"),
        ) as mock_rpc:
            _resolve_strict_vision_backend(
                "openai-codex", model="gpt-5", api_key="sk-config"
            )

        mock_rpc.assert_called_once()
        kwargs = mock_rpc.call_args.kwargs
        assert kwargs.get("explicit_api_key") == "sk-config"
        assert kwargs.get("is_vision") is True

    def test_empty_or_whitespace_api_key_normalized_to_none(self):
        """Whitespace-only keys must NOT shadow env-var lookup downstream.

        ``str(task_config.get("api_key", "")).strip() or None`` is the
        contract upstream — pin it inside ``_resolve_strict_vision_backend``
        so a forgotten ``.strip()`` at any future call site doesn't pass
        ``"   "`` through as if it were a real credential.
        """
        from agent.auxiliary_client import _resolve_strict_vision_backend

        with patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(MagicMock(), "model"),
        ) as mock_try:
            _resolve_strict_vision_backend("openrouter", api_key="   ")
            _resolve_strict_vision_backend("openrouter", api_key="")
            _resolve_strict_vision_backend("openrouter", api_key=None)

        for call in mock_try.call_args_list:
            assert call.kwargs.get("explicit_api_key") is None


class TestResolveVisionProviderClientPlumbsApiKey:
    """The wrapper that callers actually use plumbs the resolved key down."""

    def test_explicit_provider_branch_passes_resolved_api_key(self):
        """``provider: openrouter`` + ``api_key: …`` reaches _try_openrouter.

        End-to-end check for the bug as reported: configure
        ``auxiliary.vision.provider: openrouter`` AND
        ``auxiliary.vision.api_key: sk-or-v1-…`` in config.yaml, then
        call ``resolve_vision_provider_client()`` (the entry point used
        by ``vision_analyze``).  Before the fix, _try_openrouter saw an
        empty env var and marked the provider unhealthy.  After the
        fix, the configured key reaches the SDK constructor.
        """
        from agent.auxiliary_client import resolve_vision_provider_client

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("openrouter", "vision-model", None, "sk-or-v1-cfg", None),
        ), patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(MagicMock(), "vision-model"),
        ) as mock_try:
            resolve_vision_provider_client()

        # Sanity: _try_openrouter was reached and got the configured key.
        mock_try.assert_called_once()
        assert mock_try.call_args.kwargs.get("explicit_api_key") == "sk-or-v1-cfg"

    def test_auto_branch_does_not_force_misleading_api_key(self):
        """Auto-resolution must NOT inject a stale config api_key into env path.

        When ``auxiliary.vision.provider`` is unset/auto,
        ``_resolve_task_provider_model`` returns api_key=None — pin
        that contract so the explicit-provider branch's plumbing
        doesn't accidentally leak a ghost key into the auto-detection
        chain.  Otherwise users with a stale config api_key would see
        the wrong credential get tried for unrelated providers.
        """
        from agent.auxiliary_client import resolve_vision_provider_client

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ), patch(
            "agent.auxiliary_client._read_main_provider", return_value=""
        ), patch(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            return_value=(None, None),
        ) as mock_strict:
            resolve_vision_provider_client()

        # Every fallthrough call in auto mode passes api_key=None.
        for call in mock_strict.call_args_list:
            assert call.kwargs.get("api_key") is None or "api_key" not in call.kwargs


# ---------------------------------------------------------------------------
# Source-level guards — the wiring stays in place under refactors
# ---------------------------------------------------------------------------


class TestSourceGuards31996:
    """Make accidental removal of the api_key plumbing loud at code review."""

    def test_resolve_strict_vision_backend_references_31996(self):
        from agent.auxiliary_client import _resolve_strict_vision_backend

        src = inspect.getsource(_resolve_strict_vision_backend)
        assert "31996" in src
        # Must forward to _try_openrouter with explicit_api_key (not a
        # bare positional arg that future callers could accidentally
        # drop).
        assert "_try_openrouter(model=model, explicit_api_key=" in src

    def test_resolve_vision_provider_client_passes_resolved_api_key(self):
        from agent.auxiliary_client import resolve_vision_provider_client

        src = inspect.getsource(resolve_vision_provider_client)
        # The fix's explicit hint — "31996" must show up in the fixed
        # branch so a refactor that drops the api_key kwarg can't skip
        # the issue context.
        assert "31996" in src
        # The buggy v0.5.1 shape is gone.  Pin that we no longer call
        # ``_resolve_strict_vision_backend(requested, resolved_model)``
        # without the keyword key.
        assert "_resolve_strict_vision_backend(\n            requested, resolved_model\n        )" not in src
