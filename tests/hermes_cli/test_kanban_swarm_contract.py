"""Behavior-contract tests for kanban_swarm — create_swarm against a real
(temporary) HERMES_HOME so that the full connect() / init_db() path is
exercised, catching regressions like the create_task tuple-unpack bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SwarmCreated,
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Provide a temporary HERMES_HOME with an initialized kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_swarm_against_temp_home(kanban_home):
    """create_swarm end-to-end against a temp HERMES_HOME creates
    root + workers + verifier + synthesizer without raising.

    This would have caught the tuple-unpack regression where
    ``root, _ = kb.create_task(...)`` raised ValueError because
    create_task returns a bare str, not a tuple.
    """
    conn = kb.connect()
    try:
        created = create_swarm(
            conn,
            goal="Analyze market and produce strategy memo.",
            workers=[
                SwarmWorkerSpec(
                    profile="researcher",
                    title="Market research",
                    body="Find top 5 competitors",
                ),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
    finally:
        conn.close()

    # Verify the returned structure.
    assert isinstance(created, SwarmCreated)
    assert created.root_id.startswith("t_")
    assert len(created.worker_ids) == 1
    assert created.verifier_id.startswith("t_")
    assert created.synthesizer_id.startswith("t_")

    # All ids are distinct.
    all_ids = [created.root_id] + created.worker_ids + [created.verifier_id, created.synthesizer_id]
    assert len(set(all_ids)) == 4

    # Blackboard topology was posted on root.
    conn = kb.connect()
    try:
        board = latest_blackboard(conn, created.root_id)
    finally:
        conn.close()
    assert "topology" in board
    topo = board["topology"]
    assert topo["root_id"] == created.root_id
    assert topo["worker_ids"] == created.worker_ids
    assert topo["verifier_id"] == created.verifier_id
    assert topo["synthesizer_id"] == created.synthesizer_id


def test_create_swarm_multi_worker_against_temp_home(kanban_home):
    """Multi-worker swarm against a temp HERMES_HOME: 3 workers,
    verifier gated on all 3, synthesizer gated on verifier.
    """
    conn = kb.connect()
    try:
        created = create_swarm(
            conn,
            goal="Triangulate the problem from three angles.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Angle A", body="Investigate A"),
                SwarmWorkerSpec(profile="b", title="Angle B", body="Investigate B"),
                SwarmWorkerSpec(profile="c", title="Angle C", body="Investigate C"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
    finally:
        conn.close()

    assert len(created.worker_ids) == 3

    # Verify task states.
    conn = kb.connect()
    try:
        root = kb.get_task(conn, created.root_id)
        assert root.status == "done"

        for wid in created.worker_ids:
            w = kb.get_task(conn, wid)
            assert w.status == "ready"

        verifier = kb.get_task(conn, created.verifier_id)
        assert verifier.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)

        synth = kb.get_task(conn, created.synthesizer_id)
        assert synth.status == "todo"
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
    finally:
        conn.close()
