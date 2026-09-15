"""Shape-tolerant pricing extraction from endpoint ``/models`` metadata.

Contract under test (PR 3): both observed pricing conventions normalize to
dollars-per-MILLION, with the completion alias and ``global`` nesting handled,
container-level unit inference, and the round-1 review fixes:

- OpenRouter-style ($/token): ``{"prompt": "0.00000075", "completion": "0.00000375"}``
- Per-million style ($/M): ``{"global": {"prompt": 0.75, "completions": 3.75,
  "input_cache_read": 0.075}, "regional_increase_percent": 0.1}``
- Nested-dict guard: only known nest keys flatten (Flash round-1 BLOCKER 1).
- Root-over-nested precedence (both reviewers SHOULD-FIX 2).
- Container-level unit inference: a sub-cent $/M cache rate inside a per-million
  container must not be misread as $/token (Flash round-1 SHOULD-FIX 3).
"""

from decimal import Decimal

from agent.pricing_shape import (
    extract_pricing_fields,
    unwrap_pricing_container,
)


class TestOpenRouterShape:
    def test_dollar_per_token_strings_scaled_to_per_million(self):
        fields = extract_pricing_fields({"prompt": "0.00000075", "completion": "0.00000375"})
        assert fields["prompt"] == Decimal("0.75")
        assert fields["completion"] == Decimal("3.75")

    def test_expensive_model_token_price_below_ceiling(self):
        """$8/M = 0.000008/token < 0.01 ceiling: still per-token."""
        fields = extract_pricing_fields({"prompt": "0.00001", "completion": "0.0003"})
        assert fields["prompt"] == Decimal("10.00")
        assert fields["completion"] == Decimal("300.0")


class TestPerMillionShape:
    def test_global_nesting_with_completions_alias(self):
        """The exact observed shape: $/M values, completions alias, global nest."""
        fields = extract_pricing_fields({
            "global": {"completions": 3.75, "prompt": 0.75, "input_cache_read": 0.075},
            "regional_increase_percent": 0.1,
        })
        assert fields["prompt"] == Decimal("0.75")
        assert fields["completion"] == Decimal("3.75")
        assert fields["cache_read"] == Decimal("0.075")

    def test_sub_cent_cache_rate_inside_per_million_container(self):
        """Flash round-1 SHOULD-FIX 3: a $0.005/M cache-read inside a per-million
        container must normalize as $/M ($0.005/M), NOT as $0.005/token ($5,000/M).
        Container-level inference is what makes this work."""
        fields = extract_pricing_fields({
            "global": {"prompt": 0.75, "completions": 3.75, "input_cache_read": 0.005},
        })
        assert fields["prompt"] == Decimal("0.75")
        assert fields["completion"] == Decimal("3.75")
        assert fields["cache_read"] == Decimal("0.005")

    def test_expensive_model_per_million(self):
        """$300/M output (frontier tier)."""
        fields = extract_pricing_fields({"prompt": 30.0, "completions": 300.0})
        assert fields["prompt"] == Decimal("30.0")
        assert fields["completion"] == Decimal("300.0")

    def test_cache_write_alias(self):
        fields = extract_pricing_fields({"global": {"input_cache_write": 0.041667}})
        assert fields["cache_write"] == Decimal("0.041667")


class TestNestedContainerGuard:
    def test_unrelated_nested_dicts_do_not_bleed(self):
        """Flash round-1 BLOCKER 1: {architecture: {output: 4096}} must NOT put a
        phantom `output` (= completion alias) price into the fields."""
        fields = extract_pricing_fields({
            "prompt": "0.00000075", "completion": "0.00000375",
            "architecture": {"output": 4096, "input": 8192},
            "limits": {"request": 60},
        })
        assert fields["completion"] == Decimal("3.75")
        assert fields["request"] is None

    def test_root_keys_win_over_nested(self):
        """Flash round-1 SHOULD-FIX 2: flat root keys take precedence over
        unwrapped nested values, regardless of dict insertion order."""
        nested_first = {"global": {"prompt": 1.0}, "prompt": "0.00000075"}
        flat_first = {"prompt": "0.00000075", "global": {"prompt": 1.0}}
        assert extract_pricing_fields(nested_first)["prompt"] == Decimal("0.75")
        assert extract_pricing_fields(flat_first)["prompt"] == Decimal("0.75")

    def test_regional_increase_sibling_ignored(self):
        fields = extract_pricing_fields({
            "global": {"prompt": 0.75, "completions": 3.75},
            "regional_increase_percent": 0.1,
        })
        assert fields["prompt"] == Decimal("0.75")
        assert all(fields[k] != Decimal("0.1") for k in fields)


class TestDegenerateInput:
    def test_missing_pricing_returns_all_none(self):
        fields = extract_pricing_fields({})
        assert all(v is None for v in fields.values())

    def test_non_dict_container(self):
        assert extract_pricing_fields(None) == {
            "prompt": None, "completion": None, "cache_read": None,
            "cache_write": None, "request": None,
        }

    def test_malformed_values_skipped(self):
        fields = extract_pricing_fields({"prompt": "not-a-number", "completion": True})
        assert fields["prompt"] is None
        assert fields["completion"] is None  # bool rejected

    def test_flat_container_unwrapped_passthrough(self):
        flat = {"prompt": "0.00000075"}
        assert unwrap_pricing_container(flat) == flat


def test_unwrap_rejects_non_dict_nest_values():
    """A nest key holding a non-dict (e.g. global: 5) must not crash."""
    assert unwrap_pricing_container({"global": 5}) == {}


class TestIntegrationPricingEntry:
    """Integration: _pricing_entry_from_metadata on the two real shapes."""

    def _entry(self, pricing):
        from agent.usage_pricing import _pricing_entry_from_metadata
        return _pricing_entry_from_metadata(
            {"model-x": {"pricing": pricing}}, "model-x",
            source_url="https://g.example/v1/models", pricing_version="test",
        )

    def test_per_million_shape_entry(self):
        entry = self._entry({
            "global": {"completions": 3.75, "prompt": 0.75, "input_cache_read": 0.075},
            "regional_increase_percent": 0.1,
        })
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0.75")
        assert entry.output_cost_per_million == Decimal("3.75")
        assert entry.cache_read_cost_per_million == Decimal("0.075")
        assert entry.source == "provider_models_api"

    def test_openrouter_shape_entry_unchanged(self):
        entry = self._entry({"prompt": "0.00000075", "completion": "0.00000375"})
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0.75")
        assert entry.output_cost_per_million == Decimal("3.75")

    def test_per_million_shape_survives_the_endpoint_probe_path(self):
        """The nested shape must still price BOTH rates after the `/models` probe
        has filtered the pricing dict.

        `fetch_endpoint_model_metadata` builds its cache through
        `_parse_models_payload` -> `_extract_pricing`, which keeps only the keys
        whose spelling is in its alias map. A key it drops is unpriced no matter
        what this module's alias chains support, so the contract is asserted
        against that real path rather than a hand-built metadata dict.
        """
        from agent.model_metadata import _parse_models_payload
        from agent.usage_pricing import _pricing_entry_from_metadata

        metadata = _parse_models_payload({
            "data": [{
                "id": "model-x",
                "pricing": {
                    "global": {"prompt": 0.75, "completions": 3.75, "input_cache_read": 0.075},
                    "regional_increase_percent": 0.1,
                },
            }],
        })
        entry = _pricing_entry_from_metadata(
            metadata, "model-x",
            source_url="https://g.example/v1/models", pricing_version="test",
        )
        assert entry is not None
        assert entry.input_cost_per_million == Decimal("0.75")
        assert entry.output_cost_per_million == Decimal("3.75")
        assert entry.cache_read_cost_per_million == Decimal("0.075")
