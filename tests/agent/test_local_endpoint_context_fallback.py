"""A local endpoint's probe-down answer must never exceed `min(catalog, DEFAULT_FALLBACK_CONTEXT)`.

When every endpoint-specific probe fails, `_resolve_custom_endpoint_context_length` falls back to
`DEFAULT_CONTEXT_LENGTHS` — a table of **cloud API** limits keyed on the model *name*. For a local
server that value describes nothing: its window is whatever it was launched with (`--ctx-size`,
`max_model_len`), not what a vendor sells over HTTP.

Both directions of that have to hold, and declining the catalog outright breaks one of them:

- **Above the default** — declining a 1,000,000-token catalog hit on a 131,072-token local model is
  the point of the change. The compression trigger derives from this window, so an inflated value
  puts compaction past the real limit and the session dies at the endpoint instead of compacting
  before it.
- **At or below the default** — declining a 131,072-token hit would *raise* the guess to 256,000 and
  move the 0.8 trigger from 104,857 to 204,800. That is the same costly direction, and it covers 33
  of the 124 catalog entries (the `llama`/`qwen`/`gemma-3`/`deepseek`/`nemotron` catch-alls), so
  refusing the catalog wholesale trades one wrong guess for another.

One rule covers both: keep the catalog value when it is the lower guess, decline it when it is not.

Catalog data is seeded synthetically so these assertions never depend on real catalog entries.
"""

import pytest

import agent.model_metadata as mm
from agent.model_metadata import (
    DEFAULT_FALLBACK_CONTEXT,
    _resolve_custom_endpoint_context_length,
)

LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"


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


def _fake_catalog(monkeypatch, value: int) -> str:
    """Seed a synthetic catalog entry so no assertion depends on real catalog data."""
    monkeypatch.setitem(mm.DEFAULT_CONTEXT_LENGTHS, "fake-local-model", value)
    return "Fake-Local-Model"


class TestLocalEndpointCatalogFallback:
    @pytest.mark.parametrize(
        "catalog",
        [1_000_000, 131_072],
        ids=["catalog-above-default", "catalog-at-or-below-default"],
    )
    def test_local_result_never_exceeds_min_catalog_and_default(self, probes_down, monkeypatch, catalog):
        """The invariant: a local probe-down answer is exactly `min(catalog, default)`."""
        model = _fake_catalog(monkeypatch, catalog)

        ctx = _resolve_custom_endpoint_context_length(model, LOCAL_BASE_URL, "", "custom")

        assert ctx == min(catalog, DEFAULT_FALLBACK_CONTEXT), (
            f"local endpoint answered {ctx:,}; a local answer must never exceed "
            f"min(catalog={catalog:,}, default={DEFAULT_FALLBACK_CONTEXT:,})"
        )

    def test_declined_local_guess_is_visible(self, probes_down, monkeypatch):
        """Declining the catalog must not swallow the warning — it is the only sign of a guess."""
        model = _fake_catalog(monkeypatch, 1_000_000)

        ctx = _resolve_custom_endpoint_context_length(model, LOCAL_BASE_URL, "", "custom")

        assert ctx == DEFAULT_FALLBACK_CONTEXT
        assert probes_down, (
            "the catalog branch returned before _warn_context_length_fallback, "
            "so the user was never told the window was a guess"
        )
