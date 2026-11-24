"""Durable ``reset_pending`` on the existing gateway runtime status record."""

from __future__ import annotations

import json

import pytest

from gateway import status


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    """Point status I/O at a temp home and drop any leaked in-process snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status._runtime_status_state = None
    status._runtime_status_state_path = None
    yield tmp_path
    status._runtime_status_state = None
    status._runtime_status_state_path = None


def test_reset_pending_is_durable_until_explicitly_cleared(isolated_status):
    absent = status.read_runtime_status()
    assert not (isinstance(absent, dict) and absent.get("reset_pending") is True)

    assert status.write_runtime_status(gateway_state="starting", wait_timeout=10.0)
    seeded = status.read_runtime_status()
    assert isinstance(seeded, dict)
    assert seeded.get("reset_pending") is not True

    assert status.write_runtime_status(reset_pending=True, wait_timeout=10.0)
    stored = status.read_runtime_status()
    assert isinstance(stored, dict)
    assert stored.get("reset_pending") is True
    on_disk = json.loads((isolated_status / "gateway_state.json").read_text(encoding="utf-8"))
    assert on_disk["reset_pending"] is True

    assert status.write_runtime_status(gateway_state="running", wait_timeout=10.0)
    kept = status.read_runtime_status()
    assert isinstance(kept, dict)
    assert kept.get("gateway_state") == "running"
    assert kept.get("reset_pending") is True

    assert status.write_runtime_status(reset_pending=False, wait_timeout=10.0)
    cleared = status.read_runtime_status()
    assert isinstance(cleared, dict)
    assert cleared.get("reset_pending") is False
    cleared_disk = json.loads((isolated_status / "gateway_state.json").read_text(encoding="utf-8"))
    assert cleared_disk["reset_pending"] is False
