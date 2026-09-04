"""pm.receipt: the universal machine-readable surface for venv ops.

Every pm sync writes one; same schema/dir as update receipts; the
updater embeds the sync sections via snapshot().
"""

from __future__ import annotations

import json

import pytest

import pm.receipt as receipt


@pytest.fixture
def homed(tmp_path, monkeypatch):
    """Receipt dir inside a temp hermes home."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def test_begin_record_finalize_roundtrip(homed):
    receipt.begin("sync")
    receipt.record_step("uv-lock", True)
    receipt.record_venv_rebuild(True)
    receipt.record_bisect(
        [{"plugin": "bad", "action": "disabled", "reason": "conflict"}]
    )
    receipt.record_feature_list(["web", "acp"])
    path = receipt.finalize("bisected")
    assert path is not None and path.is_file()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kind"] == "sync"
    assert data["outcome"] == "bisected"
    assert data["venv_rebuild"] == {"ok": True, "reason": ""}
    assert data["plugin_bisect"][0]["plugin"] == "bad"
    assert data["feature_list"] == ["web", "acp"]


def test_latest_points_at_newest(homed):
    receipt.begin("sync")
    receipt.finalize("ok")
    latest = receipt.latest()
    assert latest is not None
    assert latest["outcome"] == "ok"

    receipt.begin("sync")
    receipt.record_venv_rebuild(False, "uv sync exited 1")
    receipt.finalize("failed", 1)
    latest = receipt.latest()
    assert latest["outcome"] == "failed"
    assert latest["venv_rebuild"]["reason"] == "uv sync exited 1"


def test_finalize_without_begin_is_none(homed):
    assert receipt.finalize("ok") is None


def test_snapshot_returns_inflight(homed):
    receipt.begin("sync")
    snap = receipt.snapshot()
    assert snap is not None and snap["kind"] == "sync"
    receipt.finalize("ok")
    assert receipt.snapshot() is None


def test_latest_none_when_empty(homed):
    assert receipt.latest() is None
