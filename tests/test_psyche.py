#!/usr/bin/env python3
"""
Tests for the Psyche Network Monitor tool.

Covers:
- All action handlers (list_runs, run_details, checkpoints, pool_status, network_stats, contribute)
- HTTP helper functions
- Solana RPC integration
- Error handling and edge cases
- Schema validation

Run with: python -m pytest tests/test_psyche.py -v
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that tools/__init__.py eagerly imports.
# This allows the test to run without firecrawl, fal_client, etc.
# ---------------------------------------------------------------------------
_OPTIONAL_DEPS = [
    "firecrawl", "fal_client", "browserbase", "playwright",
    "agent", "agent.auxiliary_client", "agent.display",
]
for _dep in _OPTIONAL_DEPS:
    if _dep not in sys.modules:
        sys.modules[_dep] = types.ModuleType(_dep)

# Provide stubs for agent.auxiliary_client used at import time
_agent_aux = sys.modules.setdefault("agent.auxiliary_client", types.ModuleType("agent.auxiliary_client"))
_agent_aux.get_text_auxiliary_client = lambda: (None, "stub-model")
_agent_aux.get_vision_auxiliary_client = lambda: (None, "stub-vision-model")

# Provide a stub for Firecrawl class
_firecrawl = sys.modules.setdefault("firecrawl", types.ModuleType("firecrawl"))
_firecrawl.Firecrawl = MagicMock

# Provide a stub for DebugSession
_debug_mod = types.ModuleType("tools.debug_helpers")
class _StubDebugSession:
    def __init__(self, *a, **kw):
        self.active = False
        self.session_id = "test"
        self.log_dir = "/tmp"
    def log_call(self, *a, **kw): pass
    def save(self, *a, **kw): pass
    def get_session_info(self): return {}
_debug_mod.DebugSession = _StubDebugSession
sys.modules.setdefault("tools.debug_helpers", _debug_mod)

# Now safe to import
from tools.psyche_tool import (
    psyche_monitor,
    PSYCHE_MONITOR_SCHEMA,
    check_psyche_available,
    _http_get,
    _solana_rpc_call,
)
import tools.psyche_tool as _mod


class TestPsycheMonitorSchema(unittest.TestCase):
    """Test tool schema and registry registration."""

    def test_schema_name(self):
        self.assertEqual(PSYCHE_MONITOR_SCHEMA["name"], "psyche_monitor")

    def test_schema_has_required_action(self):
        self.assertIn("action", PSYCHE_MONITOR_SCHEMA["parameters"]["required"])

    def test_schema_action_enum(self):
        action_prop = PSYCHE_MONITOR_SCHEMA["parameters"]["properties"]["action"]
        expected_actions = [
            "list_runs", "run_details", "checkpoints",
            "pool_status", "network_stats", "contribute",
        ]
        self.assertEqual(action_prop["enum"], expected_actions)

    def test_schema_optional_params(self):
        props = PSYCHE_MONITOR_SCHEMA["parameters"]["properties"]
        self.assertIn("run_id", props)
        self.assertIn("model_id", props)
        self.assertIn("limit", props)

    def test_schema_has_description(self):
        self.assertIn("Psyche", PSYCHE_MONITOR_SCHEMA["description"])

    def test_availability_check_always_true(self):
        """Psyche monitor should always be available (no API keys needed)."""
        self.assertTrue(check_psyche_available())


class TestPsycheMonitorDispatch(unittest.TestCase):
    """Test the main dispatcher."""

    def test_unknown_action(self):
        result = json.loads(psyche_monitor({"action": "nonexistent"}))
        self.assertIn("error", result)
        self.assertIn("Unknown action", result["error"])

    def test_unknown_action_lists_available(self):
        result = json.loads(psyche_monitor({"action": "bad_action"}))
        self.assertIn("list_runs", result["error"])

    @patch.object(_mod, "_http_get")
    @patch.object(_mod, "_solana_rpc_call")
    def test_default_action_is_network_stats(self, mock_rpc, mock_get):
        """When no action provided, defaults to network_stats."""
        mock_get.return_value = {"success": True, "data": [], "status": 200}
        mock_rpc.return_value = {"success": True, "data": "ok"}
        result = json.loads(psyche_monitor({}))
        self.assertIn("network", result)


class TestListRuns(unittest.TestCase):
    """Test the list_runs action."""

    @patch.object(_mod, "_http_get")
    def test_list_runs_with_hf_models(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "data": [
                {
                    "modelId": "PsycheFoundation/test-model",
                    "lastModified": "2026-02-20T10:00:00Z",
                    "downloads": 100,
                    "likes": 50,
                    "pipeline_tag": "text-generation",
                }
            ],
            "status": 200,
        }
        result = json.loads(psyche_monitor({"action": "list_runs"}))
        self.assertTrue(result["success"])
        self.assertIn("training_runs", result)
        self.assertIn("huggingface_models", result)
        self.assertGreater(len(result["training_runs"]), 0)
        self.assertEqual(len(result["huggingface_models"]), 1)

    @patch.object(_mod, "_http_get")
    def test_list_runs_hf_failure_still_returns_known_runs(self, mock_get):
        """Should still return known runs even if HuggingFace is unreachable."""
        mock_get.return_value = {"success": False, "error": "timeout", "status": 0}
        result = json.loads(psyche_monitor({"action": "list_runs"}))
        self.assertTrue(result["success"])
        self.assertGreater(len(result["training_runs"]), 0)
        self.assertEqual(len(result["huggingface_models"]), 0)

    @patch.object(_mod, "_http_get")
    def test_list_runs_contains_dashboard_url(self, mock_get):
        mock_get.return_value = {"success": True, "data": [], "status": 200}
        result = json.loads(psyche_monitor({"action": "list_runs"}))
        self.assertIn("dashboard_url", result)
        self.assertIn("psyche.network", result["dashboard_url"])


class TestRunDetails(unittest.TestCase):
    """Test the run_details action."""

    @patch.object(_mod, "_http_get")
    def test_known_run_consilience(self, mock_get):
        mock_get.return_value = {
            "success": True,
            "data": {
                "modelId": "PsycheFoundation/consilience-40b-CqX3FUm4",
                "lastModified": "2026-02-25T10:00:00Z",
                "downloads": 500,
                "likes": 200,
                "tags": ["transformers", "pytorch"],
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
            },
            "status": 200,
        }
        result = json.loads(psyche_monitor({"action": "run_details", "run_id": "consilience-40b-1"}))
        self.assertTrue(result["success"])
        self.assertEqual(result["run_id"], "consilience-40b-1")
        self.assertEqual(result["model_size"], "40B parameters")
        self.assertIn("huggingface", result)
        self.assertEqual(result["huggingface"]["downloads"], 500)

    def test_unknown_run_returns_error(self):
        result = json.loads(psyche_monitor({"action": "run_details", "run_id": "nonexistent-run"}))
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Unknown run", result["error"])

    def test_unknown_run_suggests_list_runs(self):
        result = json.loads(psyche_monitor({"action": "run_details", "run_id": "bad-id"}))
        self.assertIn("tip", result)

    @patch.object(_mod, "_http_get")
    def test_run_details_hf_failure_still_returns_info(self, mock_get):
        mock_get.return_value = {"success": False, "error": "timeout", "status": 0}
        result = json.loads(psyche_monitor({"action": "run_details", "run_id": "consilience-40b-1"}))
        self.assertTrue(result["success"])
        self.assertEqual(result["architecture"], "MLA (Multi-head Latent Attention)")
        self.assertNotIn("huggingface", result)


class TestCheckpoints(unittest.TestCase):
    """Test the checkpoints action."""

    @patch.object(_mod, "_http_get")
    def test_checkpoints_lists_model_files(self, mock_get):
        def side_effect(url, *a, **kw):
            if "/tree/" in url:
                return {
                    "success": True,
                    "data": [
                        {"path": "config.json", "size": 1024, "type": "file"},
                        {"path": "model.safetensors", "size": 80_000_000_000, "type": "file"},
                        {"path": "tokenizer.json", "size": 2048, "type": "file"},
                        {"path": "README.md", "size": 512, "type": "file"},
                    ],
                    "status": 200,
                }
            return {
                "success": True,
                "data": {
                    "modelId": "PsycheFoundation/consilience-40b-CqX3FUm4",
                    "lastModified": "2026-02-25T10:00:00Z",
                    "downloads": 500,
                    "likes": 200,
                    "tags": [],
                },
                "status": 200,
            }

        mock_get.side_effect = side_effect
        result = json.loads(psyche_monitor({"action": "checkpoints"}))
        self.assertTrue(result["success"])
        self.assertIn("checkpoint_files", result)
        self.assertGreaterEqual(len(result["checkpoint_files"]), 3)

    @patch.object(_mod, "_http_get")
    def test_checkpoints_model_not_found(self, mock_get):
        mock_get.return_value = {"success": False, "error": "HTTP 404: Not Found", "status": 404}
        result = json.loads(psyche_monitor({"action": "checkpoints", "model_id": "nonexistent/model"}))
        self.assertFalse(result["success"])

    @patch.object(_mod, "_http_get")
    def test_checkpoints_custom_model_id(self, mock_get):
        mock_get.return_value = {"success": True, "data": {"modelId": "test/model"}, "status": 200}
        result = json.loads(psyche_monitor({"action": "checkpoints", "model_id": "test/model"}))
        self.assertEqual(result["model_id"], "test/model")


class TestPoolStatus(unittest.TestCase):
    """Test the pool_status action."""

    def test_pool_status_returns_info(self):
        result = json.loads(psyche_monitor({"action": "pool_status"}))
        self.assertTrue(result["success"])
        self.assertIn("pool_info", result)
        self.assertEqual(result["pool_info"]["blockchain"], "Solana")

    def test_pool_status_has_how_it_works(self):
        result = json.loads(psyche_monitor({"action": "pool_status"}))
        self.assertIn("how_it_works", result["pool_info"])
        self.assertIsInstance(result["pool_info"]["how_it_works"], list)

    def test_pool_status_has_smart_contract_features(self):
        result = json.loads(psyche_monitor({"action": "pool_status"}))
        self.assertIn("smart_contract_features", result["pool_info"])


class TestNetworkStats(unittest.TestCase):
    """Test the network_stats action."""

    @patch.object(_mod, "_solana_rpc_call")
    @patch.object(_mod, "_http_get")
    def test_network_stats_full_success(self, mock_get, mock_rpc):
        mock_get.return_value = {"success": True, "data": [], "status": 200}
        mock_rpc.side_effect = [
            {"success": True, "data": "ok"},       # getHealth
            {"success": True, "data": 350000000},   # getSlot
        ]
        result = json.loads(psyche_monitor({"action": "network_stats"}))
        self.assertTrue(result["success"])
        self.assertIn("network", result)
        self.assertIn("technology", result)
        self.assertIn("ecosystem", result)
        self.assertEqual(result["solana_status"], "healthy")
        self.assertEqual(result["solana_latest_slot"], 350000000)

    @patch.object(_mod, "_solana_rpc_call")
    @patch.object(_mod, "_http_get")
    def test_network_stats_solana_down(self, mock_get, mock_rpc):
        mock_get.return_value = {"success": True, "data": [], "status": 200}
        mock_rpc.return_value = {"success": False, "error": "connection refused"}
        result = json.loads(psyche_monitor({"action": "network_stats"}))
        self.assertTrue(result["success"])
        self.assertEqual(result["solana_status"], "unknown")

    @patch.object(_mod, "_solana_rpc_call")
    @patch.object(_mod, "_http_get")
    def test_network_stats_has_timestamp(self, mock_get, mock_rpc):
        mock_get.return_value = {"success": True, "data": [], "status": 200}
        mock_rpc.return_value = {"success": True, "data": "ok"}
        result = json.loads(psyche_monitor({"action": "network_stats"}))
        self.assertIn("timestamp", result)


class TestContributeGuide(unittest.TestCase):
    """Test the contribute action."""

    def test_contribute_guide_structure(self):
        result = json.loads(psyche_monitor({"action": "contribute"}))
        self.assertTrue(result["success"])
        self.assertIn("contribution_guide", result)
        paths = result["contribution_guide"]["paths"]
        self.assertIn("compute_contribution", paths)
        self.assertIn("mining_pool", paths)
        self.assertIn("code_contribution", paths)
        self.assertIn("atropos_environments", paths)
        self.assertIn("community", paths)

    def test_contribute_guide_has_atropos_bounty(self):
        result = json.loads(psyche_monitor({"action": "contribute"}))
        atropos = result["contribution_guide"]["paths"]["atropos_environments"]
        self.assertIn("$2,500", atropos["bounty"])

    def test_contribute_guide_has_github_repo(self):
        result = json.loads(psyche_monitor({"action": "contribute"}))
        code = result["contribution_guide"]["paths"]["code_contribution"]
        self.assertIn("github.com", code["repo"])


class TestHTTPHelper(unittest.TestCase):
    """Test the HTTP helper function."""

    @patch.object(_mod.urllib.request, "urlopen")
    def test_http_get_json_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"key": "value"}'
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _http_get("https://example.com/api")
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], {"key": "value"})
        self.assertEqual(result["status"], 200)

    @patch.object(_mod.urllib.request, "urlopen")
    def test_http_get_text_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"plain text response"
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _http_get("https://example.com/text")
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], "plain text response")

    @patch.object(_mod.urllib.request, "urlopen")
    def test_http_get_connection_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        result = _http_get("https://down.example.com")
        self.assertFalse(result["success"])
        self.assertIn("Connection failed", result["error"])


class TestSolanaRPC(unittest.TestCase):
    """Test the Solana RPC helper."""

    @patch.object(_mod.urllib.request, "urlopen")
    def test_rpc_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "jsonrpc": "2.0",
            "result": "ok",
            "id": 1,
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _solana_rpc_call("getHealth")
        self.assertTrue(result["success"])
        self.assertEqual(result["data"], "ok")

    @patch.object(_mod.urllib.request, "urlopen")
    def test_rpc_error_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": -32600, "message": "Invalid request"},
            "id": 1,
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _solana_rpc_call("invalidMethod")
        self.assertFalse(result["success"])
        self.assertIn("Invalid request", result["error"])

    @patch.object(_mod.urllib.request, "urlopen")
    def test_rpc_connection_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection timeout")

        result = _solana_rpc_call("getSlot")
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["error"])


if __name__ == "__main__":
    unittest.main()
