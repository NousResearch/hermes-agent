"""Tests for the graph-loop-orchestrator plugin control plane."""

from __future__ import annotations

import json

import pytest

from plugins.graph_loop_orchestrator.control_plane import Plane


@pytest.fixture()
def plane(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.graph_loop_orchestrator.control_plane.get_hermes_home",
        lambda: tmp_path,
    )
    return Plane()


def _checks(graph):
    return [
        {"criterion": c, "passed": True, "evidence": "verified"}
        for c in graph["definition_of_done"]
    ]


def test_create_requires_goal_and_dod(plane):
    with pytest.raises(ValueError):
        plane.create("", ["a"])
    with pytest.raises(ValueError):
        plane.create("goal", [])


def test_full_loop_passes_verification(plane):
    snap = plane.create(
        "build widget",
        ["widget builds", "tests pass"],
        nodes=[{"title": "make"}, {"title": "test", "depends_on": []}],
    )
    gid = snap["id"]
    for node in plane.runnable(gid):
        plane.claim(gid, node["id"], agent="worker-1")
        plane.complete(gid, node["id"], {"ok": True}, evidence=["log"])
    result = plane.verify(gid, _checks(plane.get(gid)))
    assert result["status"] == "complete"
    assert result["verification"]["status"] == "passed"


def test_failed_verification_loops_then_blocks(plane):
    snap = plane.create("goal", ["criterion"], nodes=[{"title": "n1"}])
    gid = snap["id"]
    node = plane.runnable(gid)[0]
    plane.claim(gid, node["id"])
    plane.complete(gid, node["id"], {"ok": True})
    plane.get(gid)["max_iterations"] = 2
    for _ in range(2):
        result = plane.verify(gid, [{"criterion": "criterion", "passed": False}])
    assert result["status"] == "blocked"
    assert result["loop_iteration"] == 2


def test_state_persists_across_instances(plane, tmp_path):
    plane.create("goal", ["criterion"], nodes=[{"title": "n1"}])
    from plugins.graph_loop_orchestrator.control_plane import Plane as P2

    fresh = P2()
    # Rebind path to the tmp home (module-level get_hermes_home was patched)
    fresh.path = tmp_path / "graph-loop-orchestrator" / "state.json"
    fresh.load()
    assert len(fresh.graphs) == 1


def test_ping_validation(plane):
    with pytest.raises(ValueError):
        plane.ping("a", [], "hello")
    ping = plane.ping("a", ["b"], "hello")
    assert ping["status"] == "queued"


def test_dependency_gating(plane):
    snap = plane.create(
        "g", ["c"], nodes=[{"id": "a"}, {"id": "b", "depends_on": ["a"]}]
    )
    gid = snap["id"]
    assert [n["id"] for n in plane.runnable(gid)] == ["a"]
    plane.claim(gid, "a")
    plane.complete(gid, "a", {"ok": True})
    assert [n["id"] for n in plane.runnable(gid)] == ["b"]
