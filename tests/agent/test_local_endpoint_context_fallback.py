"""Local endpoints must not inherit a cloud vendor's window from the hardcoded catalog.

When every endpoint-specific probe fails, `_resolve_custom_endpoint_context_length` falls
back to `DEFAULT_CONTEXT_LENGTHS` — a table of *cloud API* limits keyed on the model name.
For a local server that value describes nothing: a local window is whatever the server was
launched with (`--ctx-size`, `max_model_len`), not what the vendor sells over HTTP.

Two things go wrong together on that path:

1. The probe-down log line promises ``DEFAULT_FALLBACK_CONTEXT`` and the function then
   returns the catalog number instead, so the log and the return value disagree.
2. The catalog branch ``return``s before ``_warn_context_length_fallback``, so the one
   warning that exists to say "we guessed" never fires for the case that guesses loudest.

The resulting window is trusted by the compression trigger, so a 1,000,000-token catalog
hit on a 131,072-token local model computes an 800,000-token compaction trigger that the
session can never reach — compression silently never runs.
"""

from unittest.mock import patch

import pytest

import agent.model_metadata as mm
from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _resolve_custom_endpoint_context_length,
)

# A model id that substring-matches the catalog entry below (the real-world shape: a local
# MLX/Llama.cpp server serving a model whose vendor name maps to a 1M cloud API limit).
LOCAL_MODEL = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw"
LOCAL_BASE_URL = "http://192.168.0.16:6068"
REMOTE_BASE_URL = "https://my-gateway.example.com/v1"


def _catalog_hit_for(model: str) -> int:
    """The catalog value this model id would resolve to, so a test can state the delta."""
    hit = mm._longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
    assert hit, f"test model {model!r} must match the hardcoded catalog"
    return hit[1]


@pytest.fixture
def probes_down(monkeypatch):
    """Every endpoint-specific probe reports 'I could not determine the window'."""
    monkeypatch.setattr(mm, "_resolve_endpoint_context_length", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_probe_local_context_length", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_query_ollama_api_show", lambda *a, **k: None)
    monkeypatch.setattr(mm, "get_cached_context_length", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(mm, "_warn_context_length_fallback", lambda model, base_url: seen.append((model, base_url)))
    return seen


class TestLocalEndpointCatalogFallback:
    def test_local_endpoint_does_not_inherit_cloud_catalog_window(self, probes_down):
        """A local server whose probes failed must not report a cloud vendor's window."""
        catalog = _catalog_hit_for(LOCAL_MODEL)
        assert mm.is_local_endpoint(LOCAL_BASE_URL)

        ctx = _resolve_custom_endpoint_context_length(LOCAL_MODEL, LOCAL_BASE_URL, "", "custom")

        assert ctx != catalog, (
            f"local endpoint inherited the cloud catalog window {catalog:,}; "
            f"its own window is whatever the server was launched with"
        )
        assert ctx == DEFAULT_FALLBACK_CONTEXT

    def test_local_endpoint_guess_still_warns(self, probes_down):
        """The 'we guessed' warning must fire — it is what tells the user to pin the window."""
        _resolve_custom_endpoint_context_length(LOCAL_MODEL, LOCAL_BASE_URL, "", "custom")

        assert probes_down, (
            "the catalog branch returned before _warn_context_length_fallback, "
            "so the user was never told the window was a guess"
        )

    def test_catalog_still_serves_proxied_remote_gateway(self, probes_down):
        """Remote/proxied endpoints keep the catalog: that is the case the branch exists for
        (a proxied Anthropic gateway fails the probes but its model name is still real)."""
        assert not mm.is_local_endpoint(REMOTE_BASE_URL)

        ctx = _resolve_custom_endpoint_context_length(LOCAL_MODEL, REMOTE_BASE_URL, "", "custom")

        assert ctx == _catalog_hit_for(LOCAL_MODEL)

    def test_local_probe_down_log_matches_return_value(self, probes_down, caplog):
        """The probe-down log promises DEFAULT_FALLBACK_CONTEXT; the return must agree."""
        import logging

        with caplog.at_level(logging.INFO, logger="agent.model_metadata"):
            ctx = _resolve_custom_endpoint_context_length(LOCAL_MODEL, LOCAL_BASE_URL, "", "custom")

        assert ctx == DEFAULT_FALLBACK_CONTEXT
        for record in caplog.records:
            if "probe-down" in record.getMessage():
                assert f"{DEFAULT_FALLBACK_CONTEXT:,}" in record.getMessage()
                break
        else:
            pytest.fail("expected a probe-down log line")
