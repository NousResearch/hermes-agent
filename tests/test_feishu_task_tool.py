"""Tests for Feishu Task Tool.

TDD Phase 1: Write tests that describe expected behavior.
These tests will FAIL until the tool is implemented.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timezone


# Shared mock for lark_oapi module tree
_mock_lark_module = MagicMock()


def _make_mock_builder_chain():
    """Create a mock request builder that simulates the lark_oapi builder chain."""
    mock_request = MagicMock()
    # Simulate builder().build() returning a request with queries/paths
    # queries is a list of (key, value) tuples in lark_oapi
    mock_request.queries = []
    mock_request.paths = {}
    return mock_request


class TestFeishuTaskToolUnit(unittest.TestCase):
    """Unit tests for Feishu task tool functions."""

    # -------------------------------------------------------------------------
    # feishu_task_list
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_success(self, mock_get_client):
        """List tasks returns formatted tasks."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "items": [
                {
                    "guid": "task_guid/abc123",
                    "summary": "Review PR",
                    "due": {"timestamp": "1743000000", "is_all_day": False},
                    "origin": {"platform_i18n_name": {"zh_cn": "Lark"}},
                    "completed_at": None,
                },
                {
                    "guid": "task_guid/def456",
                    "summary": "Write tests",
                    "due": {"timestamp": "1743100000", "is_all_day": True},
                    "origin": {"platform_i18n_name": {"zh_cn": "Lark"}},
                    "completed_at": {"timestamp": "1742000000"},
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({})

        self.assertIn("success", result)
        self.assertIn("Review PR", result)
        self.assertIn("Write tests", result)
        self.assertIn("task_guid/abc123", result)
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_empty(self, mock_get_client):
        """List tasks handles empty response."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({})

        self.assertIn("success", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_with_completed_filter(self, mock_get_client):
        """List tasks respects completed parameter."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({"completed": True})

        self.assertIn("success", result)
        # Verify request was made with the builder
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_with_due_range(self, mock_get_client):
        """List tasks respects due_start and due_end parameters."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({
            "due_start": "2026-04-01T00:00:00Z",
            "due_end": "2026-04-30T23:59:59Z",
        })

        self.assertIn("success", result)
        # Verify request was made
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_limit(self, mock_get_client):
        """List tasks respects limit parameter."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({"limit": 10})

        self.assertIn("success", result)
        # Verify request was made
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_client_unavailable(self, mock_get_client):
        """Returns error when Feishu client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_list({})

        self.assertIn("error", result)
        self.assertIn("not available", result.lower())

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_list_tasks_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "app_access_token is invalid"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_list({})

        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_task_create
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_success(self, mock_get_client):
        """Create task returns the created task details."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/new123",
                "summary": "New Task",
                "due": {"timestamp": "1743000000", "is_all_day": False},
                "description": "Task description",
                "members": [
                    {"id": "ou_abc", "type": "user", "role": "assignee"},
                ],
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "New Task",
            "due": "2026-04-01T10:00:00Z",
            "description": "Task description",
            "assignee": "ou_abc",
        })

        self.assertIn("success", result)
        self.assertIn("task_guid/new123", result)
        self.assertIn("New Task", result)
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_minimal(self, mock_get_client):
        """Create task works with only required fields."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/minimal",
                "summary": "Minimal Task",
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "Minimal Task",
        })

        self.assertIn("success", result)
        self.assertIn("Minimal Task", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_missing_summary(self, mock_get_client):
        """Returns error when summary is missing."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_create({})

        self.assertIn("error", result)
        self.assertIn("summary", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_with_follower(self, mock_get_client):
        """Create task handles follower parameter."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/follower",
                "summary": "Task with Follower",
                "members": [
                    {"id": "ou_follower", "type": "user", "role": "follower"},
                ],
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "Task with Follower",
            "follower": "ou_follower",
        })

        self.assertIn("success", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_create({
            "summary": "Test Task",
        })

        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_create_task_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "permission denied"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "Test Task",
            "assignee": "ou_external",
        })

        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_task_complete
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_complete_task_success(self, mock_get_client):
        """Complete task returns success."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/complete123",
                "summary": "Completed Task",
                "completed_at": {"timestamp": "1743000000"},
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_complete({
            "task_guid": "task_guid/complete123",
        })

        self.assertIn("success", result)
        # Verify request was made
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_complete_task_missing_guid(self, mock_get_client):
        """Returns error when task_guid is missing."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_complete({})

        self.assertIn("error", result)
        self.assertIn("task_guid", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_complete_task_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_complete({
            "task_guid": "task_guid/abc",
        })

        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_complete_task_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "task not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_complete({
            "task_guid": "task_guid/nonexistent",
        })

        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_task_reopen
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_reopen_task_success(self, mock_get_client):
        """Reopen task returns success."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/reopen123",
                "summary": "Reopened Task",
                "completed_at": None,
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_reopen({
            "task_guid": "task_guid/reopen123",
        })

        self.assertIn("success", result)
        # Verify request was made
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_reopen_task_missing_guid(self, mock_get_client):
        """Returns error when task_guid is missing."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_reopen({})

        self.assertIn("error", result)
        self.assertIn("task_guid", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_reopen_task_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_reopen({
            "task_guid": "task_guid/abc",
        })

        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_reopen_task_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "task not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_reopen({
            "task_guid": "task_guid/nonexistent",
        })

        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_task_search
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_success(self, mock_get_client):
        """Search tasks returns matching tasks."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "items": [
                {
                    "guid": "task_guid/search1",
                    "summary": "Buy groceries",
                },
                {
                    "guid": "task_guid/search2",
                    "summary": "Clean groceries closet",
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_search({
            "query": "groceries",
        })

        self.assertIn("success", result)
        self.assertIn("Buy groceries", result)
        self.assertIn("Clean groceries closet", result)
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_empty_query(self, mock_get_client):
        """Returns error when query is empty."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_search({
            "query": "",
        })

        self.assertIn("error", result)
        self.assertIn("query", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_empty_results(self, mock_get_client):
        """Search tasks handles empty response."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_search({
            "query": "nonexistent",
        })

        self.assertIn("success", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_with_filters(self, mock_get_client):
        """Search tasks respects assignee, creator, completed filters."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_search({
            "query": "test",
            "assignee": "ou_abc",
            "creator": "ou_def",
            "completed": True,
            "limit": 5,
        })

        self.assertIn("success", result)
        # Verify request was made with filters
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_search({
            "query": "test",
        })

        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_search_tasks_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "search failed"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_search({
            "query": "test",
        })

        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_task_delete
    # -------------------------------------------------------------------------

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_delete_task_success(self, mock_get_client):
        """Delete task returns success."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_delete({
            "task_guid": "task_guid/delete123",
        })

        self.assertIn("success", result)
        # Verify request was made
        mock_client.request.assert_called_once()

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_delete_task_missing_guid(self, mock_get_client):
        """Returns error when task_guid is missing."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_delete({})

        self.assertIn("error", result)
        self.assertIn("task_guid", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_delete_task_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_task_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_task_delete({
            "task_guid": "task_guid/abc",
        })

        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_delete_task_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "task not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_delete({
            "task_guid": "task_guid/nonexistent",
        })

        self.assertIn("error", result)
        self.assertIn("99991663", result)


class TestFeishuTaskToolSchema(unittest.TestCase):
    """Tests for tool schemas."""

    def test_schema_list_has_required_fields(self):
        """Schema for feishu_task_list has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_LIST_SCHEMA

        self.assertEqual(FEISHU_TASK_LIST_SCHEMA["name"], "feishu_task_list")
        self.assertIn("description", FEISHU_TASK_LIST_SCHEMA)
        self.assertIn("parameters", FEISHU_TASK_LIST_SCHEMA)
        props = FEISHU_TASK_LIST_SCHEMA["parameters"]["properties"]
        self.assertIn("completed", props)
        self.assertIn("due_start", props)
        self.assertIn("due_end", props)
        self.assertIn("limit", props)

    def test_schema_create_has_required_fields(self):
        """Schema for feishu_task_create has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_CREATE_SCHEMA

        schema = FEISHU_TASK_CREATE_SCHEMA
        self.assertEqual(schema["name"], "feishu_task_create")
        props = schema["parameters"]["properties"]
        self.assertIn("summary", props)
        self.assertIn("due", props)
        self.assertIn("description", props)
        self.assertIn("assignee", props)
        self.assertIn("follower", props)
        required = schema["parameters"]["required"]
        self.assertIn("summary", required)

    def test_schema_complete_has_required_fields(self):
        """Schema for feishu_task_complete has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_COMPLETE_SCHEMA

        schema = FEISHU_TASK_COMPLETE_SCHEMA
        self.assertEqual(schema["name"], "feishu_task_complete")
        props = schema["parameters"]["properties"]
        self.assertIn("task_guid", props)
        required = schema["parameters"]["required"]
        self.assertIn("task_guid", required)

    def test_schema_reopen_has_required_fields(self):
        """Schema for feishu_task_reopen has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_REOPEN_SCHEMA

        schema = FEISHU_TASK_REOPEN_SCHEMA
        self.assertEqual(schema["name"], "feishu_task_reopen")
        props = schema["parameters"]["properties"]
        self.assertIn("task_guid", props)
        required = schema["parameters"]["required"]
        self.assertIn("task_guid", required)

    def test_schema_search_has_required_fields(self):
        """Schema for feishu_task_search has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_SEARCH_SCHEMA

        schema = FEISHU_TASK_SEARCH_SCHEMA
        self.assertEqual(schema["name"], "feishu_task_search")
        props = schema["parameters"]["properties"]
        self.assertIn("query", props)
        self.assertIn("assignee", props)
        self.assertIn("creator", props)
        self.assertIn("completed", props)
        self.assertIn("limit", props)
        required = schema["parameters"]["required"]
        self.assertIn("query", required)

    def test_schema_delete_has_required_fields(self):
        """Schema for feishu_task_delete has correct structure."""
        from tools.feishu_task_tool import FEISHU_TASK_DELETE_SCHEMA

        schema = FEISHU_TASK_DELETE_SCHEMA
        self.assertEqual(schema["name"], "feishu_task_delete")
        props = schema["parameters"]["properties"]
        self.assertIn("task_guid", props)
        required = schema["parameters"]["required"]
        self.assertIn("task_guid", required)


class TestFeishuTaskToolEdgeCases(unittest.TestCase):
    """Edge case tests for Feishu task tools."""

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_invalid_iso8601_due_date(self, mock_get_client):
        """Create task handles invalid due date gracefully."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"task": {"guid": "task_guid/test", "summary": "Test"}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "Test",
            "due": "not-a-valid-time",
        })

        # Should return error for invalid due date
        self.assertIn("error", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_empty_summary(self, mock_get_client):
        """Create task rejects empty summary."""
        import tools.feishu_task_tool as ft

        result = ft._handle_feishu_task_create({
            "summary": "   ",
        })

        self.assertIn("error", result)
        self.assertIn("summary", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_task_guid_whitespace_stripped(self, mock_get_client):
        """Task GUID is stripped of whitespace."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"task": {"guid": "task_guid/padded", "summary": "Test"}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_complete({
            "task_guid": "  task_guid/padded  ",
        })

        self.assertIn("success", result)

    @patch.dict("sys.modules", {"lark_oapi": _mock_lark_module, "lark_oapi.api": _mock_lark_module, "lark_oapi.api.task": _mock_lark_module, "lark_oapi.api.task.v2": _mock_lark_module})
    @patch("tools.feishu_task_tool._get_client")
    def test_all_task_fields_preserved(self, mock_get_client):
        """All task fields are preserved in response."""
        import tools.feishu_task_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "task": {
                "guid": "task_guid/full",
                "summary": "Full Task",
                "due": {"timestamp": "1743000000", "is_all_day": False},
                "description": "Full description",
                "completed_at": {"timestamp": "1743100000"},
                "members": [
                    {"id": "ou_assignee", "type": "user", "role": "assignee"},
                    {"id": "ou_follower", "type": "user", "role": "follower"},
                ],
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_task_create({
            "summary": "Full Task",
            "due": "2026-04-01T10:00:00Z",
            "description": "Full description",
            "assignee": "ou_assignee",
            "follower": "ou_follower",
        })

        self.assertIn("success", result)
        self.assertIn("Full Task", result)
        self.assertIn("task_guid/full", result)


if __name__ == "__main__":
    unittest.main()
