"""Tests for the Hermes Chronos Time-Travel / Branching Tree-of-Thought Trajectory Debugger."""

import json
from pathlib import Path
import pytest

from agent.chronos_debugger import (
    ChronosFrame,
    ChronosTrajectoryDebugger,
    WorkspacePatch,
)


@pytest.fixture
def temp_chronos_dir(tmp_path):
    cdir = tmp_path / "chronos_test_session"
    cdir.mkdir(parents=True, exist_ok=True)
    return cdir


def test_record_sequential_steps(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)

    f0 = dbg.record_step(
        thought="Initial user prompt received",
        messages_snapshot=[{"role": "user", "content": "Calculate stats"}],
    )
    assert f0.turn_index == 0
    assert f0.parent_frame_id is None
    assert f0.branch_name == "main"

    f1 = dbg.record_step(
        thought="Querying database",
        action_name="sql_query",
        action_args={"query": "SELECT * FROM users"},
        observation="Found 100 rows",
    )
    assert f1.turn_index == 1
    assert f1.parent_frame_id == f0.frame_id

    f2 = dbg.record_step(
        thought="Formatting summary report",
        action_name="write_file",
        observation="Saved summary.txt",
    )
    assert f2.turn_index == 2
    assert f2.parent_frame_id == f1.frame_id
    assert dbg.current_frame_id == f2.frame_id
    assert dbg.branches["main"] == f2.frame_id


def test_get_lineage(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)
    f0 = dbg.record_step(thought="Step 0")
    f1 = dbg.record_step(thought="Step 1")
    f2 = dbg.record_step(thought="Step 2")

    lineage = dbg.get_lineage(f2.frame_id)
    assert len(lineage) == 3
    assert [f.frame_id for f in lineage] == [f0.frame_id, f1.frame_id, f2.frame_id]


def test_rewind_to_historical_frame(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)
    f0 = dbg.record_step(thought="Step 0", memory_snapshot={"counter": 0})
    f1 = dbg.record_step(thought="Step 1", memory_snapshot={"counter": 1})
    f2 = dbg.record_step(thought="Step 2 - Failed path", memory_snapshot={"counter": 2, "error": True})

    # Rewind to f1
    target = dbg.rewind(f1.frame_id)
    assert target.frame_id == f1.frame_id
    assert dbg.current_frame_id == f1.frame_id
    assert target.memory_snapshot == {"counter": 1}

    # Record new step from rewound state
    f1_retry = dbg.record_step(thought="Step 2 - Corrected path", memory_snapshot={"counter": 2, "fixed": True})
    assert f1_retry.parent_frame_id == f1.frame_id


def test_fork_alternate_branch(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)
    f0 = dbg.record_step(thought="Root plan")
    f1 = dbg.record_step(thought="Linear approach")

    # Fork alternate branch from f0
    base = dbg.fork_branch(f0.frame_id, "branch-parallel")
    assert base.frame_id == f0.frame_id
    assert dbg.active_branch == "branch-parallel"
    assert dbg.current_frame_id == f0.frame_id

    # Record step on alternate branch
    f_alt = dbg.record_step(thought="Tree search approach")
    assert f_alt.branch_name == "branch-parallel"
    assert f_alt.parent_frame_id == f0.frame_id

    # Check both branches exist and are distinct
    assert dbg.branches["main"] == f1.frame_id
    assert dbg.branches["branch-parallel"] == f_alt.frame_id


def test_diff_frames_across_branches(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)
    f0 = dbg.record_step(thought="Root", memory_snapshot={"base_val": 10})
    f_a = dbg.record_step(
        thought="Branch A action",
        action_name="quick_sort",
        memory_snapshot={"base_val": 10, "algo": "quicksort"},
        messages_snapshot=[{"msg": 1}, {"msg": 2}],
    )

    dbg.fork_branch(f0.frame_id, "branch-b")
    f_b = dbg.record_step(
        thought="Branch B action",
        action_name="merge_sort",
        memory_snapshot={"base_val": 10, "algo": "mergesort", "extra": True},
        messages_snapshot=[{"msg": 1}, {"msg": 2}, {"msg": 3}],
    )

    diff = dbg.diff_frames(f_a.frame_id, f_b.frame_id)
    assert diff["branch_a"] == "main"
    assert diff["branch_b"] == "branch-b"
    assert diff["action_diff"]["frame_a_action"] == "quick_sort"
    assert diff["action_diff"]["frame_b_action"] == "merge_sort"
    assert diff["memory_diff"]["added"] == {"extra": True}
    assert diff["memory_diff"]["modified"]["algo"] == {"from": "quicksort", "to": "mergesort"}
    assert diff["messages_count_diff"] == 1


def test_render_ascii_tree(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="test-session", storage_dir=temp_chronos_dir)
    dbg.record_step(thought="Start", action_name="start_app")
    dbg.record_step(thought="Build", action_name="npm_build", observation="Success")

    tree = dbg.render_ascii_tree()
    assert "=== Chronos Trajectory DAG ===" in tree
    assert "Branch: main [ACTIVE]" in tree
    assert "Turn 0:" in tree
    assert "Turn 1:" in tree
    assert "npm_build" in tree


def test_dag_persistence_and_reload(temp_chronos_dir):
    dbg = ChronosTrajectoryDebugger(session_id="persist-session", storage_dir=temp_chronos_dir)
    f0 = dbg.record_step(thought="Persisted step 0")
    f1 = dbg.record_step(thought="Persisted step 1")

    # Load from new debugger instance
    dbg2 = ChronosTrajectoryDebugger(session_id="persist-session", storage_dir=temp_chronos_dir)
    assert len(dbg2.frames) == 2
    assert dbg2.current_frame_id == f1.frame_id
    assert dbg2.frames[f1.frame_id].parent_frame_id == f0.frame_id


def test_workspace_patch_tracking():
    patch = WorkspacePatch(
        filepath="src/core.py",
        before_hash="abc1234",
        after_hash="def5678",
        diff_content="+ def optimize(): pass",
    )
    d = patch.to_dict()
    assert d["filepath"] == "src/core.py"
    restored = WorkspacePatch.from_dict(d)
    assert restored.filepath == patch.filepath
    assert restored.diff_content == patch.diff_content
