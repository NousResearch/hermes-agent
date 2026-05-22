"""Storage-size regression tests for holographic HRR vectors."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from plugins.memory.holographic import holographic as hrr
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


pytestmark = pytest.mark.skipif(
    not hrr._HAS_NUMPY,
    reason="holographic vector storage requires numpy",
)


def _float32_blob_size(dim: int) -> int:
    return len(hrr._FLOAT32_BLOB_PREFIX) + dim * np.dtype(np.float32).itemsize


def test_phases_to_bytes_stores_float32_and_round_trips_with_dim() -> None:
    dim = 1024
    phases = hrr.encode_atom("storage-size-regression", dim=dim)

    blob = hrr.phases_to_bytes(phases)

    assert len(blob) == _float32_blob_size(dim)
    restored = hrr.bytes_to_phases(blob, dim=dim)
    assert restored.shape == (dim,)
    np.testing.assert_allclose(restored, phases, rtol=0, atol=1e-6)


def test_phases_to_bytes_round_trips_without_dim() -> None:
    dim = 1024
    phases = hrr.encode_atom("dimensionless-round-trip", dim=dim)

    restored = hrr.bytes_to_phases(hrr.phases_to_bytes(phases))

    assert restored.shape == (dim,)
    np.testing.assert_allclose(restored, phases, rtol=0, atol=1e-6)


def test_phases_to_bytes_round_trips_ambiguous_small_dims_without_dim() -> None:
    dim = 2
    phases = hrr.encode_atom("ambiguous-small-dimension", dim=dim)

    restored = hrr.bytes_to_phases(hrr.phases_to_bytes(phases))

    assert restored.shape == (dim,)
    np.testing.assert_allclose(restored, phases, rtol=0, atol=1e-6)


def test_bytes_to_phases_rejects_malformed_float32_blobs() -> None:
    phases = hrr.encode_atom("malformed-float32-blob", dim=2)
    blob = hrr.phases_to_bytes(phases)

    with pytest.raises(ValueError, match="expected .* for dim=3"):
        hrr.bytes_to_phases(blob, dim=3)

    with pytest.raises(ValueError, match="invalid payload byte length"):
        hrr.bytes_to_phases(hrr._FLOAT32_BLOB_PREFIX + b"x")


def test_bytes_to_phases_reads_legacy_float64_blobs_with_and_without_dim() -> None:
    dim = 1024
    phases = hrr.encode_atom("legacy-float64-regression", dim=dim)
    legacy_blob = phases.astype(np.float64, copy=False).tobytes()

    assert len(legacy_blob) == dim * np.dtype(np.float64).itemsize
    restored_with_dim = hrr.bytes_to_phases(legacy_blob, dim=dim)
    restored_without_dim = hrr.bytes_to_phases(legacy_blob)

    assert restored_with_dim.shape == (dim,)
    assert restored_without_dim.shape == (dim,)
    np.testing.assert_allclose(restored_with_dim, phases, rtol=0, atol=0)
    np.testing.assert_allclose(restored_without_dim, phases, rtol=0, atol=0)


def test_bytes_to_phases_prefers_dim_matched_legacy_float64_on_prefix_collision() -> None:
    dim = 4
    legacy_blob = hrr._FLOAT32_BLOB_PREFIX + b"\0" * (
        dim * np.dtype(np.float64).itemsize - len(hrr._FLOAT32_BLOB_PREFIX)
    )

    restored = hrr.bytes_to_phases(legacy_blob, dim=dim)

    assert restored.shape == (dim,)
    np.testing.assert_array_equal(
        restored,
        np.frombuffer(legacy_blob, dtype=np.float64).copy(),
    )


def test_memory_store_reads_legacy_float64_vectors(tmp_path) -> None:
    dim = 64
    db_path = tmp_path / "legacy_memory_store.db"

    with MemoryStore(db_path=db_path, hrr_dim=dim) as store:
        fact_id = store.add_fact(
            'Bob Stone keeps "legacy HRR vectors" searchable.',
            category="compat",
            tags="legacy storage",
        )

        fact_blob = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()["hrr_vector"]
        bank_blob = store._conn.execute(
            "SELECT vector FROM memory_banks WHERE bank_name = ?",
            ("cat:compat",),
        ).fetchone()["vector"]

        legacy_fact_blob = hrr.bytes_to_phases(fact_blob, dim=dim).astype(np.float64).tobytes()
        legacy_bank_blob = hrr.bytes_to_phases(bank_blob, dim=dim).astype(np.float64).tobytes()
        store._conn.execute(
            "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
            (legacy_fact_blob, fact_id),
        )
        store._conn.execute(
            "UPDATE memory_banks SET vector = ? WHERE bank_name = ?",
            (legacy_bank_blob, "cat:compat"),
        )
        store._conn.commit()

        assert len(legacy_fact_blob) == dim * np.dtype(np.float64).itemsize
        assert len(legacy_bank_blob) == dim * np.dtype(np.float64).itemsize

        retriever = FactRetriever(store, hrr_dim=dim)
        results = retriever.search("legacy HRR vectors", category="compat", limit=1)

    assert results
    assert results[0]["fact_id"] == fact_id


def test_memory_store_persists_fact_and_bank_vectors_as_float32(tmp_path) -> None:
    dim = 64
    db_path = tmp_path / "memory_store.db"

    with MemoryStore(db_path=db_path, hrr_dim=dim) as store:
        fact_id = store.add_fact(
            'Alice Smith stores "compact HRR vectors" for Python tests.',
            category="perf",
            tags="hrr storage",
        )

        fact_blob = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()["hrr_vector"]
        bank_blob = store._conn.execute(
            "SELECT vector FROM memory_banks WHERE bank_name = ?",
            ("cat:perf",),
        ).fetchone()["vector"]

        assert len(fact_blob) == _float32_blob_size(dim)
        assert len(bank_blob) == _float32_blob_size(dim)

        retriever = FactRetriever(store, hrr_dim=dim)
        results = retriever.search("compact HRR vectors", category="perf", limit=1)

    assert results
    assert results[0]["fact_id"] == fact_id
