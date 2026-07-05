"""Regression test for issue #58639.

When MoA (Mixture-of-Agents) is the active main provider, auxiliary tasks
configured with ``provider: auto`` (the default) fail with HTTP 400 because
the MoA preset name (e.g. ``"default"``) is sent as the model ID instead of
the real aggregator model.

Root cause: ``resolve_provider_client()`` pre-fills ``model`` from
``_read_main_model()`` before the ``auto`` branch runs.  When MoA is main,
``_read_main_model()`` returns the preset name — a truthy but invalid model
ID.  The ``auto`` branch then does ``final_model = model or resolved``, which
picks the pre-filled preset name over ``_resolve_auto()``'s correctly resolved
aggregator model (the fix from PR #53827).

Fix: ``final_model = resolved or model`` — ``_resolve_auto()`` already handles
MoA→aggregator resolution; its result takes precedence.  ``model`` remains as
a fallback for the rare case where ``_resolve_auto`` returns a valid client
with no model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestAutoBranchMoaPresetOverride:
    """resolve_provider_client("auto") must not let a pre-filled MoA preset
    name override _resolve_auto()'s resolved aggregator model."""

    def test_moa_preset_does_not_override_resolved_aggregator(self):
        """MoA main → _read_main_model() returns preset "default".

        _resolve_auto() resolves MoA→aggregator and returns the real model
        (e.g. "maas-glm-5.2-aliyun").  The auto branch must use that resolved
        model, not the pre-filled preset name "default".

        Before fix: final_model = model or resolved  → "default" (preset)
        After fix:  final_model = resolved or model  → "maas-glm-5.2-aliyun"
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="default",  # MoA preset name, not a real model ID
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",  # no catalog default for "auto"
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, "maas-glm-5.2-aliyun"),
            ) as mock_resolve_auto,
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("auto")

        assert client is mock_client
        # The resolved aggregator model must win over the pre-filled preset name.
        assert model == "maas-glm-5.2-aliyun"
        assert model != "default", (
            "Preset name leaked as final model — the auto branch is "
            "overriding _resolve_auto() with the pre-filled model."
        )
        mock_resolve_auto.assert_called_once()

    def test_resolved_none_falls_back_to_model(self):
        """When _resolve_auto() returns a client but no model (resolved=None),
        the pre-filled model is used as a fallback — no regression for
        providers that don't report a model.
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="fallback-model",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, None),
            ),
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("auto")

        assert client is mock_client
        # resolved was None → fall back to the pre-filled model.
        assert model == "fallback-model"

    def test_non_moa_resolved_still_preferred_over_prefill(self):
        """Non-MoA scenario: _resolve_auto() resolves to the main provider's
        real model.  resolved should still take precedence over a pre-filled
        model — this is the general behavior change and should not regress
        existing non-MoA users.
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="claude-opus-4-6",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, "claude-opus-4-6"),
            ),
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("auto")

        assert client is mock_client
        assert model == "claude-opus-4-6"

    def test_openrouter_format_model_dropped_for_non_or_provider(self):
        """The existing OpenRouter-format drop guard must still work.

        When the pre-filled model is in OpenRouter format (contains "/") and
        _resolve_auto() lands on a non-OpenRouter provider (resolved has no
        "/"), the pre-filled model is dropped.  After the drop, resolved
        should be used (it's now the only truthy value).
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="google/gemini-3-flash-preview",  # OR format
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, "local-llama-3.3"),  # non-OR, no "/"
            ),
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("auto")

        assert client is mock_client
        # The OR-format model was dropped; resolved (local model) is used.
        assert model == "local-llama-3.3"
        assert "/" not in model

    def test_explicit_caller_model_not_overridden_by_resolved(self):
        """When the caller passes an explicit model (e.g. from
        auxiliary.<task>.model via _resolve_task_provider_model), it must NOT
        be overridden by _resolve_auto()'s resolved (main) model.

        This is the case AmirF194 raised on PR #58665: a user on OpenRouter
        who sets a cheaper auxiliary.compression.model but leaves the per-task
        provider as 'auto' should get their configured cheap model, not the
        expensive main model.

        Before the _model_was_prefilled guard, `resolved or model` would
        silently return the main model (e.g. claude-opus) instead of the
        caller's explicit gpt-4o-mini.
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="anthropic/claude-opus-4-6",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, "anthropic/claude-opus-4-6"),
            ),
        ):
            from agent.auxiliary_client import resolve_provider_client

            # Caller explicitly passes a cheaper model (per-task config override)
            client, model = resolve_provider_client(
                "auto", "openai/gpt-4o-mini",
            )

        assert client is mock_client
        # Explicit caller model wins — resolved (main model) does NOT override it.
        assert model == "openai/gpt-4o-mini"
        assert model != "anthropic/claude-opus-4-6"

    def test_explicit_plain_slug_model_not_overridden_by_resolved(self):
        """Same as above but with plain slug models (no '/').

        The OpenRouter-format drop guard only fires when model has '/' and
        resolved doesn't.  Plain slugs like 'gpt-4o-mini' bypass that guard,
        so the _model_was_prefilled check is the only thing protecting the
        caller's explicit choice.
        """
        mock_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="claude-opus-4-6",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",
            ),
            patch(
                "agent.auxiliary_client._resolve_auto",
                return_value=(mock_client, "claude-opus-4-6"),
            ),
        ):
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("auto", "gpt-4o-mini")

        assert client is mock_client
        assert model == "gpt-4o-mini"
