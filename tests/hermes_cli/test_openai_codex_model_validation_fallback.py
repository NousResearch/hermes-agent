"""Regression tests for OpenAI Codex model validation when the listing lags behind
actually usable backend model IDs.

The bug originally reported in #16172: `/model` and `switch_model()` rejected
`gpt-5.3-codex-spark` because the curated listing omitted it, even though direct
runtime calls succeeded. PR #19729 fixed this by soft-accepting unknown-but-
plausible Codex slugs with a warning, and this test pins the soft-accept
behavior so it doesn't regress.

Note: gpt-5.3-codex-spark itself is now in the curated catalog (PR #22991),
so the real-world Spark request takes the `recognized=True` fast path. This
test still uses Spark as the example slug but explicitly mocks
``provider_model_ids`` to omit it, exercising the soft-accept path generically
for any future entitlement-gated Codex slug that ships before Hermes catalogs
it.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from hermes_cli.models_validate import validate_requested_model


def test_openai_codex_unknown_but_plausible_model_is_accepted_with_warning():
    """If the Codex listing is incomplete, `/model` should soft-accept the model
    with a warning instead of hard-rejecting it.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = validate_requested_model("gpt-5.3-codex-spark", "openai-codex")

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    assert "gpt-5.3-codex-spark" in result["message"]
    assert "OpenAI Codex model listing" in result["message"]
    assert "Similar models" in result["message"]
    assert "gpt-5.3-codex" in result["message"]


def test_switch_model_allows_openai_codex_model_missing_from_listing():
    """switch_model() should succeed for Codex models that the runtime accepts
    even when the listing has not caught up yet.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = switch_model(
            "gpt-5.3-codex-spark",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            user_providers=None,
        )

    assert result.success is True
    assert result.new_model == "gpt-5.3-codex-spark"
    assert result.target_provider == "openai-codex"
    assert result.warning_message
    assert "OpenAI Codex model listing" in result.warning_message


def test_settings_only_provider_block_keeps_builtin_validation():
    """A ``providers.<built-in-slug>`` block that carries settings only (no
    base_url/url/api of its own) must not reroute model validation through the
    custom-endpoint branch (#120020). The Codex backend exposes no ``/models``
    listing, so the custom branch hard-rejects any model while the built-in
    catalog path soft-accepts it.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = switch_model(
            "gpt-5.3-codex-spark",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            user_providers={"openai-codex": {"request_timeout_seconds": 3600}},
            custom_providers=[],
        )

    assert result.success is True
    assert result.target_provider == "openai-codex"


def test_provider_block_with_own_endpoint_still_validates_as_custom():
    """A ``providers.<slug>`` entry that declares an endpoint of its own is the
    user's own endpoint and must keep the custom-endpoint validation branch
    (#120020 — the fix must not swallow real custom endpoints).
    """
    with patch(
        "hermes_cli.models_validate.validate_requested_model",
        return_value={"accepted": True, "persist": True, "recognized": True, "message": "ok"},
    ) as mock_validate:
        switch_model(
            "my-model",
            current_provider="myprov",
            current_model="my-model",
            current_base_url="",
            current_api_key="",
            explicit_provider="myprov",
            user_providers={"myprov": {"base_url": "https://example.invalid/v1"}},
            custom_providers=[],
        )

    assert mock_validate.called
    args, kwargs = mock_validate.call_args
    validate_as = args[1] if len(args) > 1 else kwargs.get("validate_as")
    assert validate_as == "custom:myprov"
