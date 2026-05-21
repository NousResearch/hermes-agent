"""Regression coverage for #29802 — the TUI context display showed
exactly half of the value the user passed via ``llama-server -c``.

Root cause. llama.cpp PR ggml-org/llama.cpp#10704 ("server: various
fixes", Dec 2024) moved slot members including ``n_ctx`` from
``slot_params`` to ``server_slot``. Since that refactor,
``/v1/props.default_generation_settings.n_ctx`` is the **per-slot**
budget — i.e. ``-c`` divided by ``--parallel``. Modern llama-server
defaults ``--parallel -1`` (auto) which picks ``total_slots >= 2`` on
common single-GPU setups, halving (or more) the reported value.

Without the fix in ``fetch_endpoint_model_metadata`` we wrote that
per-slot value verbatim into the context-length cache and every
downstream consumer (TUI indicator, max-context guard, compression
trigger) under-reported the context window.

Pinned here:

* ``TestLlamaCppPropsMultipliesBySlots`` — wire-level: parallel=2,
  parallel=4, and the auto-1 no-op case all yield the correct total.
* ``TestLlamaCppPropsCacheTargeting`` — robust cache-key resolution
  via ``model_alias`` (legacy) → ``model_path`` basename (modern)
  → lone-cache-entry fallback (one model per llama-server is the
  normal case). Without these fallbacks a missing alias silently
  dropped the props correction.
* ``TestLlamaCppPropsDegrades`` — props endpoint 404, malformed
  payload, network exception all leave the original ``/v1/models``
  context_length in place; we never make the situation worse.
* ``TestLlamaCppNonLlamacppUntouched`` — non-llama.cpp endpoints
  (LM Studio, vLLM, plain custom OpenAI-compat) don't get props
  multiplication.
* ``TestSourceGuardrail`` — static asserts so a future refactor
  can't quietly drop the multiplier or the cache-targeting fallbacks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(ok: bool = True, payload=None, status_code: int = 200):
    r = MagicMock()
    r.ok = ok
    r.status_code = status_code
    r.raise_for_status.return_value = None
    r.json.return_value = payload if payload is not None else {}
    r.text = ""
    return r


def _clear_caches():
    from agent.model_metadata import (
        _endpoint_model_metadata_cache,
        _endpoint_model_metadata_cache_time,
    )
    _endpoint_model_metadata_cache.clear()
    _endpoint_model_metadata_cache_time.clear()


def _make_models_payload(model_id: str, owned_by: str = "llamacpp",
                         baseline_ctx: int | None = None):
    entry = {"id": model_id, "owned_by": owned_by}
    if baseline_ctx is not None:
        entry["context_length"] = baseline_ctx
    return {"data": [entry]}


def _wire_requests(monkeypatch, *, models_payload, props_payload,
                   props_ok: bool = True, props_raises: bool = False):
    """Wire ``requests.get`` so that ``/v1/models`` returns
    ``models_payload`` and ``/v1/props`` returns ``props_payload``.

    Set ``props_raises=True`` to make the props call raise an
    exception instead of returning a response.
    """
    def fake_get(url, **kw):
        if url.endswith("/v1/models") or url.endswith("/models"):
            return _resp(True, models_payload)
        if url.endswith("/v1/props") or url.endswith("/props"):
            if props_raises:
                raise ConnectionError("simulated props failure")
            return _resp(props_ok, props_payload, status_code=200 if props_ok else 500)
        return _resp(False, status_code=404)

    monkeypatch.setattr("agent.model_metadata.requests.get", fake_get)


# ---------------------------------------------------------------------------
# Wire-level: n_ctx * total_slots
# ---------------------------------------------------------------------------


class TestLlamaCppPropsMultipliesBySlots:
    """The exact-half bug: when ``total_slots > 1`` the per-slot ``n_ctx``
    must be multiplied so the cached context length equals what ``-c``
    was set to."""

    def test_reporter_repro_parallel_2_yields_full_minus_c(self, monkeypatch):
        """Reporter's exact scenario: ``-c 131072`` with default
        ``--parallel -1`` settling on 2 slots reported 65.5K — the
        fix must report 131072."""
        _clear_caches()
        models = _make_models_payload("llama-3.1-8b-instruct",
                                      baseline_ctx=65536)
        props = {
            "default_generation_settings": {"n_ctx": 65536},
            "total_slots": 2,
            "model_path": "/opt/models/llama-3.1-8b-instruct.q4_k_m.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        ctx = result["llama-3.1-8b-instruct"]["context_length"]
        assert ctx == 131_072, (
            f"Expected 131072 (65536 * 2 slots = -c 131072), got {ctx}. "
            "Regression: #29802 — TUI displayed half the configured -c."
        )

    def test_parallel_4_multiplies_by_4(self, monkeypatch):
        _clear_caches()
        models = _make_models_payload("model-x", baseline_ctx=8192)
        props = {
            "default_generation_settings": {"n_ctx": 8192},
            "total_slots": 4,
            "model_path": "/models/model-x.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["model-x"]["context_length"] == 32_768

    def test_parallel_1_is_noop_modern_value_passes_through(self, monkeypatch):
        """When the user runs ``--parallel 1`` the per-slot ``n_ctx``
        already equals total ``-c`` — multiplication must be a no-op,
        not 1×1=1 confused arithmetic."""
        _clear_caches()
        models = _make_models_payload("model-y", baseline_ctx=16384)
        props = {
            "default_generation_settings": {"n_ctx": 131_072},
            "total_slots": 1,
            "model_path": "/models/model-y.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["model-y"]["context_length"] == 131_072

    def test_missing_total_slots_assumes_1(self, monkeypatch):
        """Older / minimal llama.cpp builds omit ``total_slots``. The
        fix must treat that as ``total_slots = 1`` (the documented
        default before the auto-parallel change) so the value passes
        through unchanged — never accidentally zero or doubled."""
        _clear_caches()
        models = _make_models_payload("model-z", baseline_ctx=8192)
        props = {
            "default_generation_settings": {"n_ctx": 65_536},
            "model_path": "/models/model-z.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["model-z"]["context_length"] == 65_536

    def test_zero_total_slots_does_not_zero_out(self, monkeypatch):
        """Defensive: a buggy ``total_slots: 0`` payload must NOT
        produce a 0-token context (which would crash the agent on
        startup); the multiplication is gated on ``> 1``."""
        _clear_caches()
        models = _make_models_payload("model-q", baseline_ctx=16_000)
        props = {
            "default_generation_settings": {"n_ctx": 16_384},
            "total_slots": 0,
            "model_path": "/models/model-q.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        # n_ctx passes through (16384) — no zero-multiplication
        assert result["model-q"]["context_length"] == 16_384

    def test_non_int_total_slots_passes_n_ctx_through(self, monkeypatch):
        """Defensive: a malformed ``total_slots`` (string, float,
        null) must not corrupt the n_ctx with bad arithmetic."""
        _clear_caches()
        models = _make_models_payload("model-w", baseline_ctx=8192)
        props = {
            "default_generation_settings": {"n_ctx": 32_768},
            "total_slots": "garbage",
            "model_path": "/models/model-w.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["model-w"]["context_length"] == 32_768


# ---------------------------------------------------------------------------
# Cache-key targeting — the fallback chain
# ---------------------------------------------------------------------------


class TestLlamaCppPropsCacheTargeting:
    """The original code only updated the cache when ``props.model_alias``
    happened to match a cache key. On modern llama.cpp (which uses
    ``model_path`` instead) the props correction silently dropped on
    every release that omitted the legacy alias field — masking the
    half-context bug as "we just never override the /v1/models value"
    in some setups and "we override it but with the wrong number" in
    others. The fix layers three fallbacks."""

    def test_legacy_model_alias_matches_cache(self, monkeypatch):
        """The original happy path: cache has an entry keyed by the
        model_alias, and the props correction lands on it."""
        _clear_caches()
        models = _make_models_payload("legacy-alias", baseline_ctx=8192)
        props = {
            "default_generation_settings": {"n_ctx": 32_768},
            "total_slots": 2,
            "model_alias": "legacy-alias",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["legacy-alias"]["context_length"] == 65_536

    def test_modern_model_path_basename_matches_cache(self, monkeypatch):
        """When ``model_alias`` is absent but ``model_path`` is set
        (the modern schema), the basename of model_path drives the
        cache-key match."""
        _clear_caches()
        # llama.cpp commonly registers the model under the gguf file
        # basename in /v1/models.
        models = _make_models_payload("phi-3-mini-q4.gguf",
                                      baseline_ctx=2048)
        props = {
            "default_generation_settings": {"n_ctx": 2048},
            "total_slots": 2,
            "model_path": "/srv/models/phi-3-mini-q4.gguf",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["phi-3-mini-q4.gguf"]["context_length"] == 4096

    def test_lone_entry_fallback_when_no_alias_or_path_match(
        self, monkeypatch
    ):
        """When neither ``model_alias`` nor ``model_path`` basename
        matches any cache key (e.g. the model is listed in /v1/models
        under a totally different id), but the cache has exactly one
        model entry, that lone entry gets the props correction —
        unambiguously, since llama-server serves one model per
        process."""
        _clear_caches()
        models = _make_models_payload("arbitrary-id-from-models-endpoint",
                                      baseline_ctx=8192)
        props = {
            "default_generation_settings": {"n_ctx": 16_384},
            "total_slots": 4,
        }  # no alias, no model_path
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)

        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        ctx = result["arbitrary-id-from-models-endpoint"]["context_length"]
        assert ctx == 65_536, (
            "Lone-cache-entry fallback dropped — modern llama.cpp builds "
            "that omit both model_alias and model_path silently keep the "
            "wrong per-slot value."
        )


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestLlamaCppPropsDegrades:
    """When ``/v1/props`` is unreachable / malformed, the props
    correction must be silently skipped — never raise, never corrupt
    the cache."""

    def test_props_404_leaves_baseline_intact(self, monkeypatch):
        _clear_caches()
        models = _make_models_payload("m", baseline_ctx=65_536)

        def fake_get(url, **kw):
            if url.endswith("/v1/models") or url.endswith("/models"):
                return _resp(True, models)
            return _resp(False, status_code=404)

        monkeypatch.setattr("agent.model_metadata.requests.get", fake_get)
        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        # Baseline from /v1/models is preserved.
        assert result["m"]["context_length"] == 65_536

    def test_props_raises_does_not_propagate(self, monkeypatch):
        _clear_caches()
        models = _make_models_payload("m", baseline_ctx=8192)
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload={}, props_raises=True)
        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        # No crash; baseline preserved.
        assert result["m"]["context_length"] == 8192

    def test_props_payload_missing_n_ctx_is_skipped(self, monkeypatch):
        _clear_caches()
        models = _make_models_payload("m", baseline_ctx=8192)
        props = {
            "default_generation_settings": {},  # no n_ctx
            "total_slots": 2,
            "model_alias": "m",
        }
        _wire_requests(monkeypatch, models_payload=models,
                       props_payload=props)
        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="llamacpp"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8080/v1", force_refresh=True
            )
        assert result["m"]["context_length"] == 8192


# ---------------------------------------------------------------------------
# Non-llama.cpp endpoints aren't affected
# ---------------------------------------------------------------------------


class TestLlamaCppNonLlamacppUntouched:
    """The /props code path is gated on ``owned_by == "llamacpp"``.
    Plain OpenAI-compatible / vLLM / etc. servers must not even try
    /v1/props — and definitely must not get any multiplication."""

    def test_vllm_owned_by_path_skips_props(self, monkeypatch):
        _clear_caches()
        models_payload = {
            "data": [{
                "id": "qwen2.5-coder",
                "owned_by": "vllm",
                "max_model_len": 65_536,
            }]
        }
        # Wire props to a sentinel that, if read, would clobber the
        # answer to the wrong value (16). If the test still passes
        # with 65_536, the props branch was correctly skipped.
        props_sentinel = {
            "default_generation_settings": {"n_ctx": 16},
            "total_slots": 2,
        }
        _wire_requests(monkeypatch, models_payload=models_payload,
                       props_payload=props_sentinel)
        with patch("agent.model_metadata.is_local_endpoint", return_value=True), \
             patch("agent.model_metadata.detect_local_server_type",
                   return_value="vllm"):
            from agent.model_metadata import fetch_endpoint_model_metadata
            result = fetch_endpoint_model_metadata(
                "http://localhost:8000/v1", force_refresh=True
            )
        assert result["qwen2.5-coder"]["context_length"] == 65_536


# ---------------------------------------------------------------------------
# Source guardrail — prevent silent regression
# ---------------------------------------------------------------------------


class TestSourceGuardrail:
    @pytest.fixture
    def source(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2]
                / "agent" / "model_metadata.py").read_text(encoding="utf-8")

    def test_multiplies_n_ctx_by_total_slots(self, source):
        assert "n_ctx * total_slots" in source, (
            "The #29802 fix must multiply ``n_ctx`` by ``total_slots`` "
            "in the llama.cpp /props branch."
        )

    def test_multiplier_guarded_on_total_slots_greater_than_one(self, source):
        assert "total_slots > 1" in source, (
            "The multiplier must be gated on ``total_slots > 1`` so "
            "single-slot servers and malformed payloads pass n_ctx "
            "through unchanged."
        )

    def test_model_path_basename_fallback_present(self, source):
        assert "model_path" in source and "basename" in source, (
            "Modern llama.cpp /v1/props uses ``model_path`` instead of "
            "``model_alias``; the cache-key fallback must use its basename."
        )

    def test_lone_cache_entry_fallback_present(self, source):
        assert "len(cache) == 1" in source, (
            "The lone-cache-entry fallback is the last line of defence "
            "when llama.cpp omits both ``model_alias`` and ``model_path`` "
            "— without it the props correction silently drops on minimal "
            "build configurations."
        )

    def test_issue_number_referenced_in_comment(self, source):
        assert "#29802" in source
