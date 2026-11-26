"""Legacy A2A protocol fallback for pre-v1.0 peers.

When a peer only implements the legacy tasks/send method (not v1.0 SendMessage),
the client should detect the -32601 error and fall back to tasks/send + polling
tasks/get until terminal state.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import plugins.platforms.a2a.tools as a2at
import plugins.platforms.a2a.protocol as protocol


class TestLegacyFallback(unittest.TestCase):
    """Test that a2a_call falls back to legacy tasks/send when SendMessage returns -32601."""

    def setUp(self):
        # Mock config to have a peer configured
        self.mock_config = {
            "a2a_agents": {
                "legacy-peer": {"url": "http://localhost:9999", "timeout": 30}
            }
        }

    def test_sendmessage_method_not_found_triggers_fallback(self):
        """When SendMessage returns -32601, should fall back to tasks/send."""
        call_count = {"v1": 0, "legacy_send": 0, "legacy_get": 0}

        def mock_http_post(url, body, headers, timeout):
            method = body.get("method", "")
            if method == "SendMessage":
                call_count["v1"] += 1
                # Simulate method not found error
                return {"jsonrpc": "2.0", "id": body.get("id"),
                        "error": {"code": -32601, "message": "method not found"}}
            elif method == "tasks/send":
                call_count["legacy_send"] += 1
                return {"jsonrpc": "2.0", "id": body.get("id"),
                        "result": {"id": body["params"]["id"], "status": {"state": "working"}}}
            elif method == "tasks/get":
                call_count["legacy_get"] += 1
                # Return completed on second poll
                if call_count["legacy_get"] >= 2:
                    return {"jsonrpc": "2.0", "id": body.get("id"),
                            "result": {"id": body["params"]["id"], "status": {"state": "completed"},
                                      "artifacts": [{"parts": [{"text": "legacy reply"}]}]}}
                return {"jsonrpc": "2.0", "id": body.get("id"),
                        "result": {"id": body["params"]["id"], "status": {"state": "working"}}}
            return {"error": {"code": -32601, "message": "unknown method"}}

        with patch.object(a2at, "_load_config", return_value=self.mock_config):
            with patch.object(a2at, "_fetch_card", return_value=None):
                with patch.object(a2at, "_http_post_json", side_effect=mock_http_post):
                    with patch.object(protocol, "persist_message"):
                        peer = {"url": "http://localhost:9999", "auth": {}, "timeout": 30}
                        reply, ctx, state = a2at._send_task("legacy-peer", peer, "hello", "")

        # Verify fallback happened
        self.assertEqual(call_count["v1"], 1, "Should try SendMessage first")
        self.assertEqual(call_count["legacy_send"], 1, "Should fall back to tasks/send")
        self.assertGreaterEqual(call_count["legacy_get"], 2, "Should poll tasks/get")
        self.assertEqual(reply, "legacy reply")
        self.assertEqual(state, "completed")

    def test_v1_success_no_fallback(self):
        """When SendMessage succeeds, should NOT fall back to legacy."""
        call_count = {"v1": 0, "legacy": 0}

        def mock_http_post(url, body, headers, timeout):
            method = body.get("method", "")
            if method == "SendMessage":
                call_count["v1"] += 1
                return {"jsonrpc": "2.0", "id": body.get("id"),
                        "result": {"status": {"state": "completed"},
                                  "artifacts": [{"parts": [{"text": "v1 reply"}]}]}}
            else:
                call_count["legacy"] += 1
                return {}

        with patch.object(a2at, "_load_config", return_value=self.mock_config):
            with patch.object(a2at, "_fetch_card", return_value=None):
                with patch.object(a2at, "_http_post_json", side_effect=mock_http_post):
                    with patch.object(protocol, "persist_message"):
                        peer = {"url": "http://localhost:9999", "auth": {}, "timeout": 30}
                        reply, ctx, state = a2at._send_task("legacy-peer", peer, "hello", "")

        self.assertEqual(call_count["v1"], 1)
        self.assertEqual(call_count["legacy"], 0, "Should NOT call legacy methods when v1 succeeds")
        self.assertEqual(reply, "v1 reply")

    def test_other_error_no_fallback(self):
        """When SendMessage returns a different error (not -32601), should NOT fall back."""
        call_count = {"v1": 0, "legacy": 0}

        def mock_http_post(url, body, headers, timeout):
            method = body.get("method", "")
            if method == "SendMessage":
                call_count["v1"] += 1
                return {"jsonrpc": "2.0", "id": body.get("id"),
                        "error": {"code": -32600, "message": "invalid request"}}
            else:
                call_count["legacy"] += 1
                return {}

        with patch.object(a2at, "_load_config", return_value=self.mock_config):
            with patch.object(a2at, "_fetch_card", return_value=None):
                with patch.object(a2at, "_http_post_json", side_effect=mock_http_post):
                    with patch.object(protocol, "persist_message"):
                        peer = {"url": "http://localhost:9999", "auth": {}, "timeout": 30}
                        with self.assertRaises(ValueError) as cm:
                            a2at._send_task("legacy-peer", peer, "hello", "")
                        self.assertIn("invalid request", str(cm.exception))

        self.assertEqual(call_count["v1"], 1)
        self.assertEqual(call_count["legacy"], 0, "Should NOT fall back for non--32601 errors")


if __name__ == "__main__":
    unittest.main()
