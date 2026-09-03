"""Tests for retrieval_count increments on the agent tool path.

The agent reads facts through the fact_store tool (search / probe / related),
which routes through FactRetriever — not through store.search_facts(), the
only path that used to bump retrieval_count. The memory janitor's staleness
check ("facts with retrieval_count=0 older than 30 days are stale") therefore
flagged every fact as never retrieved even when the agent actually used them.
The tool path must count retrievals too (#78801).
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    db_path = tmp_path / "memory.db"
    provider = HolographicMemoryProvider(config={"db_path": str(db_path)})
    provider.initialize(session_id="test-session")
    return provider


def _retrieval_count(provider, fact_id: int) -> int:
    row = provider._store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    assert row is not None, f"fact {fact_id} not found"
    return row[0]


def _seed_fact(provider, content: str) -> int:
    out = json.loads(
        provider.handle_tool_call("fact_store", {"action": "add", "content": content})
    )
    return out["fact_id"]


def _run(provider, action: str, **extra) -> dict:
    args = {"action": action, **extra}
    return json.loads(provider.handle_tool_call("fact_store", args))


def test_search_increments_retrieval_count(provider):
    fid = _seed_fact(provider, "Alice likes oolong tea with jasmine")
    assert _retrieval_count(provider, fid) == 0

    out = _run(provider, "search", query="oolong tea")

    assert out["count"] >= 1
    returned_ids = {f["fact_id"] for f in out["results"]}
    assert fid in returned_ids
    assert _retrieval_count(provider, fid) == 1


def test_probe_increments_retrieval_count(provider):
    fid = _seed_fact(provider, "Alice prefers oolong tea")
    assert _retrieval_count(provider, fid) == 0

    out = _run(provider, "probe", entity="alice")

    assert out["count"] >= 1
    returned_ids = {f["fact_id"] for f in out["results"]}
    assert fid in returned_ids
    assert _retrieval_count(provider, fid) == 1


def test_related_increments_retrieval_count(provider):
    fid = _seed_fact(provider, "Alice works on the deployment rollback")
    assert _retrieval_count(provider, fid) == 0

    out = _run(provider, "related", entity="alice")

    assert out["count"] >= 1
    returned_ids = {f["fact_id"] for f in out["results"]}
    assert fid in returned_ids
    assert _retrieval_count(provider, fid) == 1


def test_non_retrieval_actions_do_not_bump(provider):
    fid = _seed_fact(provider, "Bob prefers coffee")
    assert _retrieval_count(provider, fid) == 0

    # add / reason / feedback must not touch retrieval_count
    _run(provider, "reason", entities=["bob"])
    assert _retrieval_count(provider, fid) == 0
