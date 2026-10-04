"""A local probe-down answer is the **largest catalog match at or below the generic default**.

When every endpoint-specific probe fails, `_resolve_custom_endpoint_context_length` falls back to
`DEFAULT_CONTEXT_LENGTHS` — a table of **cloud API** limits keyed on the model *name*. For a local
server that value describes nothing: its window is whatever it was launched with (`--ctx-size`,
`max_model_len`), not what a vendor sells over HTTP. Three things have to hold at once:

- **Not the inflated hit.** `qwen3.8-flash` = 1,000,000 against a 131,072-token local model computes
  a trigger past the hard limit, and the session dies at the endpoint instead of compacting.
- **Not the default either, when a family entry already has the answer.** `_longest_key_match`
  consults only the most specific key, so discarding that hit alone falls through to 256,000 with
  `qwen` = 131,072 — one row away — never read. Hence the *largest match at or below* the default,
  over every matching key rather than the longest one.
- **A small specific entry does not win by being specific.** `grok-2-vision` = 8,192 beside
  `grok-2` = 131,072 is the mirror of the first bug: the cloud value describes a hosted vision API,
  not a window a local box was launched with.

Catalog data is seeded synthetically — nested keys included — so no assertion depends on real
catalog entries or on any entry's name surviving a rename.
"""

import pytest

import agent.model_metadata as mm
from agent.model_metadata import (
    DEFAULT_FALLBACK_CONTEXT,
    _resolve_custom_endpoint_context_length,
)

LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
# Contains both synthetic keys below, so `_catalog_key_matches` sees the specific entry and its
# shorter family at the same time — the shape that makes `_longest_key_match` alone insufficient.
MODEL = "Fake-Local-Model-Serve"


@pytest.fixture
def probes_down(monkeypatch):
    """Every endpoint-specific probe reports 'I could not determine the window'."""
    monkeypatch.setattr(mm, "_resolve_endpoint_context_length", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_probe_local_context_length", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_query_ollama_api_show", lambda *a, **k: None)
    monkeypatch.setattr(mm, "get_cached_context_length", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(mm, "_warn_context_length_fallback", lambda model, base_url: warned.append((model, base_url)))
    return warned


def _seed(monkeypatch, values: dict) -> None:
    for key, value in values.items():
        monkeypatch.setitem(mm.DEFAULT_CONTEXT_LENGTHS, key, value)


class TestLocalEndpointCatalogFallback:
    @pytest.mark.parametrize(
        "catalog,expected",
        [
            pytest.param({"fake-local-model": 1_000_000}, DEFAULT_FALLBACK_CONTEXT,
                         id="inflated-hit-only-falls-back-to-default"),
            pytest.param({"fake-local-model": 1_000_000, "fake-local": 131_072}, 131_072,
                         id="family-entry-holds-the-answer-beside-the-inflated-hit"),
            pytest.param({"fake-local-model": 131_072}, 131_072,
                         id="at-default-is-kept"),
            pytest.param({"fake-local-model": 8_192, "fake-local": 131_072}, 131_072,
                         id="small-specific-entry-loses-to-the-larger-family"),
        ],
    )
    def test_local_result_is_the_largest_match_at_or_below_default(
        self, probes_down, monkeypatch, catalog, expected
    ):
        """The invariant: a local answer is the largest matching value not exceeding the default."""
        _seed(monkeypatch, catalog)

        ctx = _resolve_custom_endpoint_context_length(MODEL, LOCAL_BASE_URL, "", "custom")

        assert ctx == expected, (
            f"local endpoint answered {ctx:,}; expected {expected:,} — a local answer is the "
            f"largest catalog match at or below {DEFAULT_FALLBACK_CONTEXT:,}"
        )

    def test_no_usable_match_is_visible_as_a_guess(self, probes_down, monkeypatch):
        """Falling through to the default must not swallow the warning — it is the only sign of a guess."""
        _seed(monkeypatch, {"fake-local-model": 1_000_000})

        ctx = _resolve_custom_endpoint_context_length(MODEL, LOCAL_BASE_URL, "", "custom")

        assert ctx == DEFAULT_FALLBACK_CONTEXT
        assert probes_down, (
            "no usable match was found, yet _warn_context_length_fallback never ran, "
            "so the user was never told the window was a guess"
        )
