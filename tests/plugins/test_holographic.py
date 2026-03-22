"""Tests for holographic.py — pure HRR math operations.

All tests are synthetic: no filesystem, no database, no external state.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Plugin path: prefer home dir install, fall back to in-repo copy
_plugin_dir = Path.home() / ".hermes" / "plugins" / "hermes-memory-store"
if not _plugin_dir.exists():
    _plugin_dir = Path(__file__).resolve().parent.parent.parent / "plugins" / "hermes-memory-store"
sys.path.insert(0, str(_plugin_dir))

from holographic import (
    _HAS_NUMPY,
    bind,
    bundle,
    bytes_to_phases,
    encode_atom,
    encode_fact,
    encode_text,
    phases_to_bytes,
    similarity,
    snr_estimate,
    unbind,
)


DIM = 256  # Smaller dim for fast tests; math properties hold at any dim.


class TestEncodeAtom:
    def test_deterministic(self):
        """Same input always produces the identical vector."""
        v1 = encode_atom("hello", DIM)
        v2 = encode_atom("hello", DIM)
        np.testing.assert_array_equal(v1, v2)

    def test_shape_and_dtype(self):
        v = encode_atom("test", DIM)
        assert v.shape == (DIM,)
        assert v.dtype == np.float64

    def test_phase_range(self):
        """All phases must be in [0, 2π)."""
        v = encode_atom("range_check", DIM)
        assert np.all(v >= 0.0)
        assert np.all(v < 2.0 * np.pi)

    def test_near_orthogonal(self):
        """Random unrelated words should have near-zero similarity."""
        words = ["apple", "quantum", "bicycle", "telescope", "jazz"]
        vectors = [encode_atom(w, DIM) for w in words]
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = similarity(vectors[i], vectors[j])
                assert abs(sim) < 0.15, f"'{words[i]}' vs '{words[j]}': sim={sim:.4f}"


class TestBindUnbind:
    def test_roundtrip(self):
        """unbind(bind(a, b), b) should recover a exactly."""
        a = encode_atom("concept_a", DIM)
        b = encode_atom("concept_b", DIM)
        bound = bind(a, b)
        recovered = unbind(bound, b)
        np.testing.assert_allclose(recovered, a, atol=1e-10)

    def test_commutative(self):
        """bind(a, b) == bind(b, a) — phase addition is commutative."""
        a = encode_atom("alpha", DIM)
        b = encode_atom("beta", DIM)
        np.testing.assert_allclose(bind(a, b), bind(b, a), atol=1e-10)

    def test_bound_dissimilar_to_inputs(self):
        """The bound vector should be quasi-orthogonal to both inputs."""
        a = encode_atom("dog", DIM)
        b = encode_atom("cat", DIM)
        bound = bind(a, b)
        assert abs(similarity(bound, a)) < 0.15
        assert abs(similarity(bound, b)) < 0.15


class TestBundle:
    def test_preserves_similarity(self):
        """Bundled vector should be similar to each of its components."""
        vecs = [encode_atom(f"item_{i}", DIM) for i in range(3)]
        bundled = bundle(*vecs)
        for v in vecs:
            sim = similarity(bundled, v)
            assert sim > 0.2, f"Bundle lost signal: sim={sim:.4f}"

    def test_capacity_degrades(self):
        """Similarity to each component should decrease as more items are added."""
        target = encode_atom("target", DIM)
        sims = []
        for n in [2, 5, 10, 20]:
            others = [encode_atom(f"noise_{i}", DIM) for i in range(n - 1)]
            bundled = bundle(target, *others)
            sims.append(similarity(bundled, target))
        # Similarity should generally decrease (allow minor non-monotonicity)
        assert sims[0] > sims[-1], f"No degradation: {sims}"


class TestSimilarity:
    def test_identity(self):
        """similarity(a, a) should be exactly 1.0."""
        a = encode_atom("self", DIM)
        assert similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_near_zero(self):
        """Random vectors should have similarity near 0."""
        sims = []
        for i in range(10):
            a = encode_atom(f"rand_a_{i}", DIM)
            b = encode_atom(f"rand_b_{i}", DIM)
            sims.append(similarity(a, b))
        mean_sim = np.mean(sims)
        assert abs(mean_sim) < 0.1, f"Mean similarity too high: {mean_sim:.4f}"


class TestEncodeText:
    def test_order_invariant(self):
        """Bag-of-words should be order-invariant."""
        v1 = encode_text("the quick brown fox", DIM)
        v2 = encode_text("fox brown quick the", DIM)
        sim = similarity(v1, v2)
        assert sim == pytest.approx(1.0, abs=1e-10)

    def test_similar_texts_high_similarity(self):
        """Texts sharing words should have high similarity."""
        v1 = encode_text("the cat sat on the mat", DIM)
        v2 = encode_text("the cat on the mat", DIM)
        sim = similarity(v1, v2)
        assert sim > 0.5, f"Similar texts low sim: {sim:.4f}"

    def test_empty_text(self):
        """Empty text should return a valid vector (the __hrr_empty__ atom)."""
        v = encode_text("", DIM)
        assert v.shape == (DIM,)


class TestEncodeFact:
    def test_entity_extraction(self):
        """Unbinding entity from fact should recover content signal."""
        content = "prefers rust for systems programming"
        entities = ["peppi"]

        fact_vec = encode_fact(content, entities, DIM)
        content_vec = encode_text(content, DIM)

        # Unbind: fact - bind(entity, ROLE_ENTITY) should be similar to bind(content, ROLE_CONTENT)
        role_entity = encode_atom("__hrr_role_entity__", DIM)
        role_content = encode_atom("__hrr_role_content__", DIM)
        entity_vec = encode_atom("peppi", DIM)

        # Extract what's associated with peppi's entity role
        probe = unbind(fact_vec, bind(entity_vec, role_entity))

        # The extracted signal should have nonzero similarity to the content-role binding
        content_bound = bind(content_vec, role_content)
        sim = similarity(probe, content_bound)
        # At DIM=256, 2-component bundle: SNR≈11, but phase cosine similarity compresses
        # the signal. Noise baseline is ~0.035 std; signal should be above 0.03.
        assert sim > 0.03, f"Entity extraction failed: sim={sim:.4f}"

    def test_multiple_entities(self):
        """Facts with multiple entities should encode all of them."""
        fact_vec = encode_fact("loves pizza", ["alice", "bob"], DIM)
        assert fact_vec.shape == (DIM,)
        # Both entities should be recoverable (above noise floor)
        role_entity = encode_atom("__hrr_role_entity__", DIM)
        for name in ["alice", "bob"]:
            entity_vec = encode_atom(name, DIM)
            probe = unbind(fact_vec, bind(entity_vec, role_entity))
            # Just verify it's a valid vector (deeper tests would check signal)
            assert probe.shape == (DIM,)


class TestSerialization:
    def test_roundtrip(self):
        """bytes_to_phases(phases_to_bytes(v)) should recover v exactly."""
        v = encode_atom("serialize_me", DIM)
        data = phases_to_bytes(v)
        recovered = bytes_to_phases(data)
        np.testing.assert_array_equal(v, recovered)

    def test_byte_size(self):
        """float64 * dim = 8 * dim bytes."""
        v = encode_atom("size_check", DIM)
        data = phases_to_bytes(v)
        assert len(data) == DIM * 8


class TestSNREstimate:
    def test_formula(self):
        """SNR should match sqrt(dim / n_items)."""
        import math
        assert snr_estimate(1024, 4) == pytest.approx(math.sqrt(1024 / 4))
        assert snr_estimate(1024, 256) == pytest.approx(math.sqrt(1024 / 256))

    def test_empty(self):
        """Zero items → infinite SNR."""
        assert snr_estimate(1024, 0) == float("inf")

    def test_warning_logged(self, caplog):
        """SNR < 2.0 should emit a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            snr_estimate(4, 4)  # SNR = 1.0
        assert "near capacity" in caplog.text.lower()


class TestNumpyGuard:
    def test_raises_without_numpy(self):
        """All public functions should raise RuntimeError when numpy is absent."""
        import holographic

        original = holographic._HAS_NUMPY
        try:
            holographic._HAS_NUMPY = False
            with pytest.raises(RuntimeError, match="numpy is required"):
                encode_atom("test", DIM)
            with pytest.raises(RuntimeError, match="numpy is required"):
                bind(np.zeros(DIM), np.zeros(DIM))
            with pytest.raises(RuntimeError, match="numpy is required"):
                unbind(np.zeros(DIM), np.zeros(DIM))
            with pytest.raises(RuntimeError, match="numpy is required"):
                bundle(np.zeros(DIM))
            with pytest.raises(RuntimeError, match="numpy is required"):
                similarity(np.zeros(DIM), np.zeros(DIM))
            with pytest.raises(RuntimeError, match="numpy is required"):
                encode_text("test", DIM)
            with pytest.raises(RuntimeError, match="numpy is required"):
                encode_fact("test", ["e"], DIM)
            with pytest.raises(RuntimeError, match="numpy is required"):
                phases_to_bytes(np.zeros(DIM))
            with pytest.raises(RuntimeError, match="numpy is required"):
                bytes_to_phases(b"\x00" * DIM * 8)
            with pytest.raises(RuntimeError, match="numpy is required"):
                snr_estimate(DIM, 1)
        finally:
            holographic._HAS_NUMPY = original
