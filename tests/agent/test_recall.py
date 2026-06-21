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
        s.append(session_id="sess-A", turn_seq=1, role="user",
                 content="how do I deploy?",
                 vec=np.ones(VEC_DIM, dtype=np.float32))
        s.append(session_id="sess-A", turn_seq=2, role="assistant",
                 content="use vercel promote",
                 vec=np.zeros(VEC_DIM, dtype=np.float32))
        items = s.recent_embeddings(session_id="sess-A")
        assert len(items) == 2
        # DESC order: turn_seq=2 first, turn_seq=1 second
        assert items[0][1] == 2
        assert items[1][1] == 1
        assert items[0][2] == "assistant"
        # Other sessions don't see these rows.
        assert s.recent_embeddings(session_id="sess-B") == []
    finally:
        s.close()


def test_recall_store_window_eviction(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db", max_rows=3)
    try:
        for i in range(5):
            v = np.full(VEC_DIM, float(i), dtype=np.float32)
            s.append(session_id="sess", turn_seq=i + 1,
                     role="user", content=str(i), vec=v)
        assert s.count() == 3
        items = s.recent_embeddings(session_id="sess")
        seqs = [seq for _sid, seq, *_ in items]
        # Most recent 3 kept: turn_seq 3, 4, 5 (1, 2 evicted)
        assert seqs == [5, 4, 3]
    finally:
        s.close()


def test_recall_store_clear(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        s.append(session_id="sess", turn_seq=1, role="user", content="x",
                 vec=np.ones(VEC_DIM, dtype=np.float32))
        s.append(session_id="other-sess", turn_seq=1, role="user",
                 content="y", vec=np.ones(VEC_DIM, dtype=np.float32))
        assert s.count() == 2
        # Clear one session only
        s.clear(session_id="sess")
        assert s.count() == 1
        # Wipe all
        s.clear()
        assert s.count() == 0
    finally:
        s.close()


def test_recall_store_validates_vec_shape(tmp_path):
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        with pytest.raises(ValueError):
            s.append(session_id="sess", turn_seq=1, role="user",
                     content="x", vec=np.ones(100, dtype=np.float32))
    finally:
        s.close()


def test_recall_store_next_turn_seq(tmp_path):
    """next_turn_seq must return max+1, scoped per session_id."""
    from agent.recall import RecallStore
    s = RecallStore(tmp_path / "recall.db")
    try:
        # Fresh session starts at 1
        assert s.next_turn_seq("sess-A") == 1
        s.append(session_id="sess-A", turn_seq=1, role="user",
                 content="a", vec=np.ones(VEC_DIM, dtype=np.float32))
        s.append(session_id="sess-A", turn_seq=2, role="assistant",
                 content="b", vec=np.ones(VEC_DIM, dtype=np.float32))
        # Next should be 3 (continues numbering)
        assert s.next_turn_seq("sess-A") == 3
        # Different session_id starts at 1 (independent namespace)
        assert s.next_turn_seq("sess-B") == 1
        # After sess-B inserts, sess-A still continues from 3
        s.append(session_id="sess-B", turn_seq=1, role="user",
                 content="c", vec=np.ones(VEC_DIM, dtype=np.float32))
        assert s.next_turn_seq("sess-A") == 3
        assert s.next_turn_seq("sess-B") == 2
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
        session_id="test-session",
    )
    assert service.enabled
    assert service.session_id == "test-session"

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
    assert service.store is not None and service.store.count(session_id="test-session") == 2

    # Query for alpha — should return the user turn first.
    block = service.ephemeral_block("I have a question about alpha")
    assert "<recalled_context>" in block
    assert "alpha" in block.lower()
    # The assistant turn should also be present, but ranked lower.
    assert "first letter" in block


def test_recall_service_session_resume_continues_turn_seq(tmp_path):
    """The whole point of the session_id refactor: resuming must continue
    numbering rather than clobbering prior turns."""
    from agent.recall import (
        build_recall_service, NumpyBackend,
    )

    service = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "max_turns": 10},
        session_id="resume-test",
    )
    fake = NumpyBackend()
    fake._model = object()
    fake._healthy = True
    fake.embed = lambda text: np.ones(VEC_DIM, dtype=np.float32)  # type: ignore[assignment]
    service.backend = fake

    # Original session: 2 turns
    service.record_turn("user", "first")
    service.record_turn("assistant", "answer 1")
    assert service.store.count(session_id="resume-test") == 2

    # Simulate resume: a NEW service instance with the SAME session_id.
    # Without the fix, this would clobber t1, t2 with t1, t2.
    service2 = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "max_turns": 10},
        session_id="resume-test",
    )
    service2.backend = fake
    service2.record_turn("user", "second question")
    service2.record_turn("assistant", "answer 2")

    # Now we should have 4 turns, NOT 2 (which is what the old code would
    # have given us — INSERT OR REPLACE on t1/t2 collision).
    assert service2.store.count(session_id="resume-test") == 4


def test_recall_service_scopes_to_session_id(tmp_path):
    """Two concurrent services for different sessions must not see each
    other's rows."""
    from agent.recall import build_recall_service, NumpyBackend

    service_a = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "max_turns": 10},
        session_id="sess-A",
    )
    service_b = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "max_turns": 10},
        session_id="sess-B",
    )
    fake = NumpyBackend()
    fake._model = object()
    fake._healthy = True
    fake.embed = lambda text: np.ones(VEC_DIM, dtype=np.float32)  # type: ignore[assignment]
    service_a.backend = fake
    service_b.backend = fake

    service_a.record_turn("user", "this is for session A only")
    service_b.record_turn("user", "this is for session B only")

    block_a = service_a.ephemeral_block("any query")
    block_b = service_b.ephemeral_block("any query")

    assert "session A" in block_a
    assert "session B" not in block_a
    assert "session B" in block_b
    assert "session A" not in block_b


def test_recall_store_v1_schema_migration(tmp_path):
    """A v1 recall.db (turn_id-only PK) should migrate to v2 cleanly
    rather than crash on open."""
    from agent.recall import RecallStore
    import sqlite3

    # Construct a v1 schema manually
    db_path = tmp_path / "recall.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE recall_embeddings ("
        "  turn_id TEXT PRIMARY KEY,"
        "  role TEXT NOT NULL,"
        "  content TEXT NOT NULL,"
        "  vec BLOB NOT NULL,"
        "  ts INTEGER NOT NULL"
        ")"
    )
    conn.execute(
        "INSERT INTO recall_embeddings VALUES (?, ?, ?, ?, ?)",
        ("t1", "user", "old data", b"\x00" * (VEC_DIM * 4), 1000),
    )
    conn.commit()
    conn.close()

    # Open with RecallStore — should migrate to v2
    s = RecallStore(db_path)
    try:
        # Migration preserved the row under session_id="legacy-v1"
        rows = s.recent_embeddings(session_id="legacy-v1")
        assert len(rows) == 1
        # New writes use the new schema (separate session_id)
        s.append(session_id="new-session", turn_seq=1, role="user",
                 content="new", vec=np.ones(VEC_DIM, dtype=np.float32))
        assert s.count() == 2
        # Counts split per-session
        assert s.count(session_id="legacy-v1") == 1
        assert s.count(session_id="new-session") == 1
    finally:
        s.close()


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
        "embeddings_total": 0,
        "max_rows": 0,
        "top_k": 5,
        "max_tokens": 1500,
        "session_id": "",
    }


def test_recall_health_summary_scoped_count(tmp_path):
    """When session_id is set, embeddings_stored is scoped to that
    session; embeddings_total is the global count."""
    from agent.recall import build_recall_service, recall_health_summary, NumpyBackend

    s = build_recall_service(
        profile_dir=tmp_path,
        config={"enabled": True, "backend": "numpy", "max_turns": 10},
        session_id="doctor-sess",
    )
    fake = NumpyBackend()
    fake._model = object()
    fake._healthy = True
    fake.embed = lambda text: np.ones(VEC_DIM, dtype=np.float32)  # type: ignore[assignment]
    s.backend = fake

    # Write 2 rows in this session
    s.record_turn("user", "x")
    s.record_turn("assistant", "y")
    h = recall_health_summary(s)
    assert h["embeddings_stored"] == 2
    assert h["embeddings_total"] == 2
    assert h["session_id"] == "doctor-sess"


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
