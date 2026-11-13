"""Malformed /models pricing aliases must not discard endpoint metadata."""

from agent.model_metadata import _extract_pricing, _parse_models_payload


class TestEndpointMetadataParsingRobustness:
    def test_non_scalar_pricing_alias_is_skipped_not_raised(self):
        assert _extract_pricing({"modalities": {"input": ["text"], "output": ["text"]}}) == {}
        assert _extract_pricing({"input": ["a", "b"]}) == {}
        assert _extract_pricing({"request": {"nested": 1}}) == {}

    def test_boolean_pricing_alias_is_skipped_not_priced(self):
        # bool subclasses int, but JSON booleans are not prices.
        assert _extract_pricing({"input": True}) == {}
        assert _extract_pricing({"request": False}) == {}
        assert _extract_pricing({"input": True, "unit": "per_token"}) == {}

    def test_scalar_pricing_aliases_still_parse(self):
        assert _extract_pricing({"prompt": "0.000001"}) == {"prompt": "0.000001"}
        assert _extract_pricing({"input": 1.5, "unit": "per_token"}) == {"prompt": 1.5}
        assert _extract_pricing({"input": 1.5}) == {"prompt": "1.5e-06"}

    def test_model_entry_survives_non_scalar_pricing_alias(self):
        parsed = _parse_models_payload({
            "data": [{
                "id": "some/model",
                "modalities": {"input": ["text"], "output": ["text"]},
                "context_window": 1_000_000,
            }],
        })
        entry = parsed["some/model"]
        assert entry["context_length"] == 1_000_000
        assert "pricing" not in entry
