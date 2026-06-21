"""Tests for semantic recall (agent/recall.py).

Covers the three backends, the persistent store, the injection formatter,
and the RecallService lifecycle. All tests must pass with zero new
dependencies installed — they only exercise the NoopBackend, the in-memory
shapes, and the sqlite-backed store.

NumpyBackend is exercised at the boundary: we monkey-patch a fake
embedder into the instance so we don't need sentence-transformers
installed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


# Make sure the agent package is importable when pytest runs from the
# repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


VEC_DIM = 384


# ──────────────────────────── NoopBackend ────────────────────────────


def test_noop_backend_returns_zero_vector():
    from agent.recall import NoopBackend
    b = NoopBackend()
    v = b.embed("anything at all")
    assert v.shape == (VEC_DIM,)
    assert np.allclose(v, 0.0)
    assert b.name == "noop"


def test_noop_backend_top_k_always_empty():
    from agent.recall import NoopBackend
    b = NoopBackend()
    items = [("a", np.ones(VEC_DIM)), ("b", np.zeros(VEC_DIM))]
    assert b.top_k(np.ones(VEC_DIM), items, k=5) == []


# ──────────────────────────── NumpyBackend (no model) ───────────────


def test_numpy_backend_uses_lazy_loading(monkeypatch):
    """When sentence_transformers isn't installed, NumpyBackend.embed
    returns zeros and healthy() returns False. Must never raise."""
    from agent.recall import NumpyBackend
    b = NumpyBackend(model_name="nonexistent-model")
    # Don't actually call _ensure_model — simulate it failing by
    # forcing _model=None and _healthy=False
    b._model = None
    b._healthy = False
    v = b.embed("hello")
    assert np.allclose(v, 0.0)
    assert b.healthy() is False


def test_numpy_backend_top_k_with_fake_embedder():
    """With a manually-injected fake embedder, top_k returns the
    expected ranking."""
    from agent.recall import NumpyBackend
    b = NumpyBackend()

    def fake_embed(text):
        if "alpha" in text:
            return np.array([1.0, 0.0] + [0.0] * (VEC_DIM - 2),
                            dtype=np.float32)
        if "beta" in text:
            return np.array([0.0, 1.0] + [0.0] * (VEC_DIM - 2),
                            dtype=np.float32)
        if "gamma" in text:
            return np.array([0.9, 0.1] + [0.0] * (VEC_DIM - 2),
                            dtype=np.float32)
        return np.zeros(VEC_DIM, dtype=np.float32)

    b._model = object()  # truthy so _ensure_model is a no-op
    b._healthy = True
    b.embed = fake_embed  # type: ignore[assignment]

    query = np.array([1.0, 0.0] + [0.0] * (VEC_DIM - 2), dtype=np.float32)
    items = [
        ("a", fake_embed("alpha")),
        ("b", fake_embed("beta")),
        ("c", fake_embed("gamma")),
    ]
    ranked = b.top_k(query, items, k=2)
    assert [k for k, _ in ranked] == ["a", "c"]
    # scores should be roughly 1.0 (alpha) and ~0.99 (gamma ≈ alpha)
    assert ranked[0][1] > ranked[1][1] > 0.9


# ──────────────────────────── recall_backend factory ─────────────────


def test_recall_backend_factory_noop_default(monkeypatch):
    monkeypatch.delenv("HERMES_RECALL_BACKEND", raising=False)
    from agent.recall import recall_backend, NoopBackend
    assert isinstance(recall_backend(), NoopBackend)


def test_recall_backend_factory_noop_explicit(monkeypatch):
    monkeypatch.setenv("HERMES_RECALL_BACKEND", "noop")
    from agent.recall import recall_backend, NoopBackend
    assert isinstance(recall_backend(), NoopBackend)


def test_recall_backend_factory_unknown_falls_back_to_noop(monkeypatch):
    monkeypatch.setenv("HERMES_RECALL_BACKEND", "does-not-exist")
    from agent.recall import recall_backend, NoopBackend
    assert isinstance(recall_backend(), NoopBackend)


# ──────────────────────────── RecallStore ────────────────────────────


def test_recall_store_roundtrip(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        s.append(turn_id="t1", role="user", content="how do I deploy?",
                 vec=np.ones(VEC_DIM, dtype=np.float32))
        s.append(turn_id="t2", role="assistant",
                 content="use vercel promote",
                 vec=np.zeros(VEC_DIM, dtype=np.float32))
        items = s.recent_embeddings()
        assert len(items) == 2
        # DESC order: t2 first, t1 second
        assert items[0][0] == "t2"
        assert items[1][0] == "t1"
        assert items[0][1] == "assistant"
    finally:
        s.close()


def test_recall_store_window_eviction(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db", max_rows=3)
    try:
        for i in range(5):
            v = np.full(VEC_DIM, float(i), dtype=np.float32)
            s.append(turn_id=f"t{i}", role="user",
                     content=str(i), vec=v)
        assert s.count() == 3
        items = s.recent_embeddings()
        ids = [t for t, *_ in items]
        # Most recent 3 kept: t2, t3, t4 (t0, t1 evicted)
        assert ids == ["t4", "t3", "t2"]
    finally:
        s.close()


def test_recall_store_clear(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        s.append(turn_id="t1", role="user", content="x",
                 vec=np.ones(VEC_DIM, dtype=np.float32))
        assert s.count() == 1
        s.clear()
        assert s.count() == 0
    finally:
        s.close()


def test_recall_store_validates_vec_shape(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        with pytest.raises(ValueError):
            s.append(turn_id="t1", role="user", content="x",
                     vec=np.ones(100, dtype=np.float32))
    finally:
        s.close()


# ──────────────────────────── formatter ────────────────────────────


def test_format_recall_block_empty_returns_empty_string():
    from agent.recall import format_recall_block
    assert format_recall_block([]) == ""


def test_format_recall_block_wrapped_in_xml_tags():
    from agent.recall import format_recall_block, RecallHit
    hits = [RecallHit(turn_id="t1", role="user",
                      content="hello", score=0.8, ts=0)]
    block = format_recall_block(hits)
    assert block.startswith("<recalled_context>")
    assert block.endswith("</recalled_context>")


def test_format_recall_block_truncates_to_char_cap():
    from agent.recall import format_recall_block, RecallHit
    hits = [
        RecallHit(turn_id=f"t{i}", role="user",
                  content="x" * 1000, score=0.9 - i * 0.01, ts=0)
        for i in range(20)
    ]
    block = format_recall_block(hits, max_tokens=200, max_chars=2000)
    assert "more turns truncated" in block
    assert block.endswith("</recalled_context>")


def test_format_recall_block_orders_hits_as_given():
    from agent.recall import format_recall_block, RecallHit
    hits = [
        RecallHit(turn_id="t1", role="user", content="first", score=0.9, ts=0),
        RecallHit(turn_id="t2", role="assistant", content="second", score=0.5, ts=0),
    ]
    block = format_recall_block(hits)
    assert block.index("first") < block.index("second")


# ──────────────────────────── RecallService ────────────────────────────


def test_recall_service_disabled_is_noop(tmp_path):
    from agent.recall import RecallService
    s = RecallService(enabled=False)
    assert s.ephemeral_block("anything") == ""
    # record_turn on disabled service should not raise
    s.record_turn("user", "hello")


def test_recall_service_disabled_no_store_built(tmp_path):
    from agent.recall import build_recall_service
    s = build_recall_service(profile_dir=tmp_path, config={"enabled": False})
    assert s.enabled is False
    assert s.store is None
    assert s.ephemeral_block("hello") == ""


def test_recall_service_enabled_with_noop_backend(tmp_path):
    from agent.recall import build_recall_service
    s = build_recall_service(profile_dir=tmp_path,
                             config={"enabled": True, "backend": "noop"})
    assert s.enabled is True
    assert s.store is not None
    # Even with enabled=True, a NoopBackend produces no hits.
    assert s.ephemeral_block("hello") == ""


def test_recall_service_end_to_end_with_numpy_fake(tmp_path):
    """Inject a fake backend to verify the full record → recall loop."""
    from agent.recall import (
        build_recall_service, NumpyBackend, RecallHit,
    )

    service = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "top_k": 2,
                "max_turns": 10, "max_tokens": 200},
    )
    assert service.enabled

    # Replace backend with one that uses a deterministic fake embedder.
    fake = NumpyBackend()
    fake._model = object()
    fake._healthy = True

    def fake_embed(text):
        # encode "alpha" as [1,0,...], "beta" as [0,1,...]
        if "alpha" in text:
            v = np.zeros(VEC_DIM, dtype=np.float32)
            v[0] = 1.0
            return v
        if "beta" in text:
            v = np.zeros(VEC_DIM, dtype=np.float32)
            v[1] = 1.0
            return v
        return np.zeros(VEC_DIM, dtype=np.float32)
    fake.embed = fake_embed  # type: ignore[assignment]
    service.backend = fake

    # Record two turns
    service.record_turn("user", "Tell me about alpha please")
    service.record_turn("assistant", "Alpha is the first letter")
    assert service.store is not None and service.store.count() == 2

    # Query for alpha — should return the user turn first.
    block = service.ephemeral_block("I have a question about alpha")
    assert "<recalled_context>" in block
    assert "alpha" in block.lower()
    # The assistant turn should also be present, but ranked lower.
    assert "first letter" in block


# ──────────────────────────── health summary ────────────────────────────


def test_recall_health_summary_shape(tmp_path):
    from agent.recall import build_recall_service, recall_health_summary
    s = build_recall_service(profile_dir=tmp_path,
                             config={"enabled": False})
    h = recall_health_summary(s)
    assert h == {
        "enabled": False,
        "backend": "noop",
        "backend_healthy": True,
        "embeddings_stored": 0,
        "max_rows": 0,
        "top_k": 5,
        "max_tokens": 1500,
    }


# ──────────────────────────── env var / config isolation ────────────────


def test_config_overrides_env_var(monkeypatch, tmp_path):
    """Config's backend choice wins over the env var fallback. The env
    var is only consulted when config doesn't specify a backend."""
    from agent.recall import build_recall_service, NoopBackend
    monkeypatch.setenv("HERMES_RECALL_BACKEND", "noop")
    s = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy"},
    )
    # Config says numpy → factory creates a NumpyBackend (which will
    # silently fall back to zeros at embed time since sentence-transformers
    # isn't installed — but it's still a NumpyBackend instance).
    from agent.recall import NumpyBackend
    assert isinstance(s.backend, NumpyBackend)


def test_env_var_used_when_config_omits_backend(monkeypatch, tmp_path):
    from agent.recall import build_recall_service, NoopBackend
    monkeypatch.setenv("HERMES_RECALL_BACKEND", "noop")
    # No 'backend' key in config — factory should consult env.
    s = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True},
    )
    assert isinstance(s.backend, NoopBackend)
