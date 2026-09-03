"""Tests for rollback.list checkpoint metadata and reason-to-message mapping (#101743)."""

from unittest.mock import MagicMock
import tui_gateway.server as server


def test_rollback_list_maps_reason_to_message_and_preserves_metadata(monkeypatch):
    handler = server._methods.get("rollback.list")
    assert handler is not None

    fake_checkpoints = [
        {
            "hash": "1111222233334444",
            "short_hash": "1111222",
            "timestamp": "2026-09-03T01:00:00Z",
            "reason": "Fix parser regression in parser.py",
            "files_changed": 2,
            "insertions": 10,
            "deletions": 3,
        }
    ]

    class FakeMgr:
        enabled = True

        def list_checkpoints(self, cwd):
            return fake_checkpoints

    fake_agent = MagicMock()
    fake_agent._checkpoint_mgr = FakeMgr()

    fake_session = {
        "cwd": "/tmp/test",
        "agent": fake_agent,
    }

    monkeypatch.setattr(server, "_sess", lambda params, rid: (fake_session, None))

    resp = handler(1, {})
    assert "result" in resp
    checkpoints = resp["result"]["checkpoints"]
    assert len(checkpoints) == 1
    cp = checkpoints[0]
    assert cp["hash"] == "1111222233334444"
    assert cp["short_hash"] == "1111222"
    assert cp["shortHash"] == "1111222"
    assert cp["message"] == "Fix parser regression in parser.py"
    assert cp["reason"] == "Fix parser regression in parser.py"
    assert cp["files_changed"] == 2
    assert cp["filesChanged"] == 2
    assert cp["insertions"] == 10
    assert cp["deletions"] == 3
