"""Tests for alias-aware catalog matching in model validation (#134925).

`_match_in_catalog` folds both sides through the picker's alias table before
declaring a requested name "not named in this endpoint's model listing", so a
known wire-id/public-slug pair (`k3` served as `kimi-k3`) validates cleanly
instead of warning on every switch. Case-only differences stay unmatched in
case-sensitive branches, and genuinely unknown names still soft-accept.
"""

from unittest.mock import patch

from hermes_cli.models_validate import validate_requested_model


def _probe(models):
    return {
        "models": list(models),
        "probed_url": "https://api.example.com/v1/models",
        "resolved_base_url": "https://api.example.com/v1",
        "suggested_base_url": None,
        "used_fallback": False,
    }


class TestAliasAwareValidation:
    def test_slug_against_wire_id_listing_validates_cleanly(self):
        # kimi-coding style custom provider: the endpoint lists the bare wire id.
        with patch("hermes_cli.models.probe_api_models", return_value=_probe(["k3"])):
            result = validate_requested_model(
                "kimi-k3",
                "custom:kimi-coding",
                api_key="k",
                base_url="https://api.example.com/v1",
                api_mode="anthropic_messages",
            )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_wire_id_against_slug_listing_validates_cleanly(self):
        # Reverse direction: the endpoint lists the public slug, the user asks for the wire id.
        with patch(
            "hermes_cli.models.probe_api_models", return_value=_probe(["kimi-k3"])
        ):
            result = validate_requested_model(
                "k3",
                "custom:kimi-coding",
                api_key="k",
                base_url="https://api.example.com/v1",
                api_mode="anthropic_messages",
            )
        assert result["recognized"] is True
        assert result["message"] is None

    def test_case_only_difference_is_not_aliased(self):
        # The alias fold must not silently turn into a case fold: case-sensitive
        # branches keep rejecting case-only near misses to the soft-accept note.
        with patch(
            "hermes_cli.models.probe_api_models", return_value=_probe(["llama3"])
        ):
            result = validate_requested_model(
                "LLAMA3",
                "custom:local-gateway",
                api_key="k",
                base_url="https://api.example.com/v1",
                api_mode="chat_completions",
            )
        assert result["accepted"] is True
        assert result["recognized"] is False
        assert result["message"] is not None

    def test_unknown_name_still_soft_accepts_with_note(self):
        with patch("hermes_cli.models.probe_api_models", return_value=_probe(["k3"])):
            result = validate_requested_model(
                "gpt-9-turbo",
                "custom:kimi-coding",
                api_key="k",
                base_url="https://api.example.com/v1",
                api_mode="anthropic_messages",
            )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is False
        assert "not found in this custom endpoint's model listing" in result["message"]
