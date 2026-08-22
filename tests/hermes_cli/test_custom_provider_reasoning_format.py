"""Unit tests for ``get_custom_provider_reasoning_format``.

The helper resolves ``providers.<name>.reasoning_format`` by normalized
route identity (mirroring ``get_custom_provider_context_length``) and
fails closed: anything that isn't exactly one of the three accepted
dialects returns ``None``, which callers treat as the historical
``top_level`` default — a typo must never silently change the wire format.
"""

from __future__ import annotations

import pytest

from hermes_cli.config import get_custom_provider_reasoning_format


BASE_URL = "http://127.0.0.1:8317/v1"


def _entry(**overrides):
    entry = {"name": "cpa", "base_url": BASE_URL}
    entry.update(overrides)
    return entry


class TestValueResolution:
    @pytest.mark.parametrize("fmt", ["top_level", "reasoning_object", "none"])
    def test_accepted_formats_round_trip(self, fmt):
        result = get_custom_provider_reasoning_format(
            BASE_URL, custom_providers=[_entry(reasoning_format=fmt)]
        )
        assert result == fmt

    def test_case_and_whitespace_normalized(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL,
            custom_providers=[_entry(reasoning_format="  Reasoning_Object ")],
        )
        assert result == "reasoning_object"

    @pytest.mark.parametrize(
        "bad", ["object", "topLevel", "reasoning-object", "", "  ", "off"]
    )
    def test_unknown_values_fail_closed(self, bad):
        """Typos return None so the caller keeps the historical default."""
        result = get_custom_provider_reasoning_format(
            BASE_URL, custom_providers=[_entry(reasoning_format=bad)]
        )
        assert result is None

    @pytest.mark.parametrize("bad", [None, 5, True, ["top_level"], {"v": 1}])
    def test_non_string_values_fail_closed(self, bad):
        result = get_custom_provider_reasoning_format(
            BASE_URL, custom_providers=[_entry(reasoning_format=bad)]
        )
        assert result is None

    def test_missing_key_returns_none(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL, custom_providers=[_entry()]
        )
        assert result is None

    def test_invalid_entry_does_not_shadow_later_valid_one(self):
        """Same-route duplicate entries: scan continues past an invalid value."""
        result = get_custom_provider_reasoning_format(
            BASE_URL,
            custom_providers=[
                _entry(reasoning_format="bogus"),
                _entry(reasoning_format="reasoning_object"),
            ],
        )
        assert result == "reasoning_object"


class TestRouteMatching:
    def test_trailing_slash_on_entry_still_matches(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL,
            custom_providers=[
                _entry(base_url=BASE_URL + "/", reasoning_format="none")
            ],
        )
        assert result == "none"

    def test_trailing_slash_on_query_still_matches(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL + "/",
            custom_providers=[_entry(reasoning_format="none")],
        )
        assert result == "none"

    def test_different_route_does_not_match(self):
        result = get_custom_provider_reasoning_format(
            "http://127.0.0.1:20128/v1",
            custom_providers=[_entry(reasoning_format="reasoning_object")],
        )
        assert result is None

    def test_only_matching_entry_is_consulted(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL,
            custom_providers=[
                _entry(
                    base_url="http://127.0.0.1:20128/v1",
                    reasoning_format="none",
                ),
                _entry(reasoning_format="reasoning_object"),
            ],
        )
        assert result == "reasoning_object"


class TestDegenerateInputs:
    def test_empty_base_url_returns_none(self):
        result = get_custom_provider_reasoning_format(
            "", custom_providers=[_entry(reasoning_format="none")]
        )
        assert result is None

    def test_non_list_custom_providers_returns_none(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL, custom_providers={"cpa": _entry()}
        )
        assert result is None

    def test_non_dict_entries_are_skipped(self):
        result = get_custom_provider_reasoning_format(
            BASE_URL,
            custom_providers=["junk", None, _entry(reasoning_format="none")],
        )
        assert result == "none"


class TestConfigPath:
    """End-to-end through get_compatible_custom_providers + the normalizer.

    Pins that ``_normalize_custom_provider_entry`` passes ``reasoning_format``
    through — without the passthrough, a ``providers:`` dict entry would
    lose the key before this helper ever sees it.
    """

    def test_providers_dict_entry_resolves(self):
        config = {
            "providers": {
                "cpa": {"api": BASE_URL, "reasoning_format": "reasoning_object"}
            }
        }
        result = get_custom_provider_reasoning_format(BASE_URL, config=config)
        assert result == "reasoning_object"

    def test_legacy_custom_providers_list_resolves(self):
        config = {
            "custom_providers": [
                {"name": "cpa", "base_url": BASE_URL, "reasoning_format": "none"}
            ]
        }
        result = get_custom_provider_reasoning_format(BASE_URL, config=config)
        assert result == "none"
