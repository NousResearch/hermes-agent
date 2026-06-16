"""Tests for agent_pool.py — AgentContextPool (CICS TCB pool model)."""

import json
from pathlib import Path

import pytest

from src.orchestration.agent_context import AgentContext
from src.orchestration.agent_pool import AgentContextPool


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """Pool backed by a temp agents directory."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    monkeypatch.setattr(
        "src.orchestration.agent_context.get_hermes_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "src.orchestration.agent_pool.get_hermes_home",
        lambda: tmp_path,
    )
    return AgentContextPool(agents_dir=agents_dir)


# ── tests ────────────────────────────────────────────────────────────


class TestAgentContextPool:

    def test_get_or_create_loads_from_disk(self, pool):
        """First call triggers from_disk; second returns the same instance."""
        ctx1 = pool.get_or_create("alpha")
        assert ctx1.id == "alpha"
        assert "alpha" in pool

        ctx2 = pool.get_or_create("alpha")
        assert ctx2 is ctx1  # same object

    def test_snapshot_persists_state(self, pool):
        ctx = pool.get_or_create("beta")
        ctx.state = {"turn": 5, "data": [1, 2, 3]}
        pool.snapshot("beta")

        # reload should pick up the persisted state
        pool2 = AgentContextPool(agents_dir=pool._agents_dir)
        ctx2 = pool2.get_or_create("beta")
        assert ctx2.state == {"turn": 5, "data": [1, 2, 3]}

    def test_register_and_unregister(self, pool):
        """Dynamic register / unregister (CICS RDO)."""
        ctx = AgentContext(id="dynamic", workspace=pool._agents_dir / "dynamic")
        ctx.workspace.mkdir(parents=True)
        pool.register(ctx)
        assert "dynamic" in pool

        pool.unregister("dynamic")
        assert "dynamic" not in pool

    def test_agent_ids_and_len(self, pool):
        pool.get_or_create("a")
        pool.get_or_create("b")
        assert pool.agent_ids == ["a", "b"]
        assert len(pool) == 2

    def test_snapshot_all(self, pool):
        for name in ("x", "y", "z"):
            ctx = pool.get_or_create(name)
            ctx.state = {"name": name}
        pool.snapshot_all()

        # all three should have persisted state
        pool2 = AgentContextPool(agents_dir=pool._agents_dir)
        for name in ("x", "y", "z"):
            ctx = pool2.get_or_create(name)
            assert ctx.state == {"name": name}
