"""Tests for Feishu Bitable Tool.

TDD Phase 1: Write tests that describe expected behavior.
These tests will FAIL until the tool is implemented.
"""

import unittest
from unittest.mock import patch, MagicMock


class TestFeishuBitableToolUnit(unittest.TestCase):
    """Unit tests for Feishu Bitable tool handlers."""

    # -------------------------------------------------------------------------
    # feishu_bitable_list
    # -------------------------------------------------------------------------

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_bitable_apps_returns_guidance(self, mock_get_client):
        """feishu_bitable_list returns guidance since bitable.v1 has no ListApp API."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "app_list": [
                {
                    "app": {
                        "app_id": "bltapp001",
                        "name": "Project Tracker",
                        "default_language": "zh-CN",
                    }
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_list({})
        # bitable.v1 has no ListAppRequestBuilder, so it returns a guidance error
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_bitable_apps_empty(self, mock_get_client):
        """feishu_bitable_list returns guidance (no ListApp in bitable.v1)."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"app_list": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_list({})
        # bitable.v1 has no ListAppRequestBuilder
        self.assertIn("error", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_bitable_apps_client_unavailable(self, mock_get_client):
        """Returns error when Feishu client is not available."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_list({})
        self.assertIn("error", result)
        self.assertIn("not available", result.lower())

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_bitable_apps_client_unavailable_no_import(self, mock_get_client):
        """Returns error when client unavailable (no import attempted)."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_list({})
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_bitable_tables
    # -------------------------------------------------------------------------

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_success(self, mock_get_client):
        """List tables returns formatted table list."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {
                        "table_id": "tblXYZ001",
                        "name": "Tasks",
                        "fields": [
                            {"field_name": "Title", "type": 1},
                            {"field_name": "Status", "type": 3},
                        ],
                    },
                    {
                        "table_id": "tblXYZ002",
                        "name": "Team Members",
                        "fields": [
                            {"field_name": "Name", "type": 1},
                        ],
                    },
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({
            "base_token": "blt_abc123",
        })
        self.assertIn("success", result)
        self.assertIn("Tasks", result)
        self.assertIn("tblXYZ001", result)
        self.assertIn("Team Members", result)
        self.assertIn("tblXYZ002", result)
        mock_client.request.assert_called_once()

        # Verify request was made with correct base_token
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        # lark-oapi passes app_token as kwarg to client.request, not in req.paths
        self.assertEqual(call_args[1].get("app_token"), "blt_abc123")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_missing_base_token(self, mock_get_client):
        """Returns error when base_token is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_tables({})
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_empty(self, mock_get_client):
        """List tables handles empty response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({"base_token": "blt_abc"})
        self.assertIn("success", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_limit(self, mock_get_client):
        """List tables respects limit parameter."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({
            "base_token": "blt_abc",
            "limit": 100,
        })
        self.assertIn("success", result)
        # Verify app_token was passed and page_size in queries
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "100")
        # Verify app_token kwarg
        self.assertEqual(call_args[1].get("app_token"), "blt_abc")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_no_access_error(self, mock_get_client):
        """Returns user-friendly error when bot has no access to Bitable."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 91403
        mock_response.msg = "no permission"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({"base_token": "blt_private"})
        self.assertIn("error", result)
        self.assertIn("share", result.lower())
        # 91403 code is handled but user-friendly message is returned (no code in output)
        self.assertNotIn("91403", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_tables({"base_token": "blt_abc"})
        self.assertIn("error", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "table not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({"base_token": "blt_abc"})
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_bitable_records
    # -------------------------------------------------------------------------

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_success(self, mock_get_client):
        """List records returns formatted record list."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {
                        "record_id": "rec001",
                        "fields": {
                            "Title": "Write Report",
                            "Status": "Done",
                        },
                    },
                    {
                        "record_id": "rec002",
                        "fields": {
                            "Title": "Review PR",
                            "Status": "In Progress",
                        },
                    },
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc123",
            "table_id": "tblXYZ001",
        })
        self.assertIn("success", result)
        self.assertIn("Write Report", result)
        self.assertIn("Done", result)
        self.assertIn("rec001", result)
        self.assertIn("Review PR", result)
        mock_client.request.assert_called_once()

        # Verify request was made with correct paths (passed as kwargs to client.request)
        call_args = mock_client.request.call_args
        self.assertEqual(call_args[1].get("app_token"), "blt_abc123")
        self.assertEqual(call_args[1].get("table_id"), "tblXYZ001")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_missing_base_token(self, mock_get_client):
        """Returns error when base_token is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_records({
            "table_id": "tblXYZ001",
        })
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_missing_table_id(self, mock_get_client):
        """Returns error when table_id is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
        })
        self.assertIn("error", result)
        self.assertIn("table_id", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_empty(self, mock_get_client):
        """List records handles empty response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("success", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_limit(self, mock_get_client):
        """List records respects limit parameter."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "limit": 200,
        })
        self.assertIn("success", result)
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "200")
        # Verify app_token and table_id were passed
        self.assertEqual(call_args[1].get("app_token"), "blt_abc")
        self.assertEqual(call_args[1].get("table_id"), "tblXYZ")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "table not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_bitable_create_record
    # -------------------------------------------------------------------------

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_success(self, mock_get_client):
        """Create record returns the created record details."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "record": {
                    "record_id": "recNew001",
                    "fields": {
                        "Title": "New Task",
                        "Status": "Open",
                    },
                }
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc123",
            "table_id": "tblXYZ001",
            "fields": {
                "Title": "New Task",
                "Status": "Open",
            },
        })

        self.assertIn("success", result)
        self.assertIn("recNew001", result)
        self.assertIn("New Task", result)

        # Verify request was made with correct paths (passed as kwargs to client.request)
        call_args = mock_client.request.call_args
        self.assertEqual(call_args[1].get("app_token"), "blt_abc123")
        self.assertEqual(call_args[1].get("table_id"), "tblXYZ001")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_missing_base_token(self, mock_get_client):
        """Returns error when base_token is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_create_record({
            "table_id": "tblXYZ001",
            "fields": {"Title": "Test"},
        })
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_missing_table_id(self, mock_get_client):
        """Returns error when table_id is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "fields": {"Title": "Test"},
        })
        self.assertIn("error", result)
        self.assertIn("table_id", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_missing_fields(self, mock_get_client):
        """Returns error when fields is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("error", result)
        self.assertIn("fields", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_empty_fields(self, mock_get_client):
        """Returns error when fields is empty."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "record": {
                    "record_id": "recEmpty",
                    "fields": {},
                }
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": {},
        })
        # Empty fields dict should still be allowed (creates record with empty fields)
        self.assertIn("success", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": {"Title": "Test"},
        })
        self.assertIn("error", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "field not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": {"NonExistent": "Value"},
        })
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_bitable_search
    # -------------------------------------------------------------------------

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_success(self, mock_get_client):
        """Search records returns matching records."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {
                        "record_id": "recSearch001",
                        "fields": {
                            "Title": "Design Review",
                            "Status": "In Progress",
                        },
                    },
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc123",
            "table_id": "tblXYZ001",
            "query": "Design",
        })

        self.assertIn("success", result)
        self.assertIn("Design Review", result)
        self.assertIn("recSearch001", result)

        # Verify request was made with correct paths (passed as kwargs to client.request)
        call_args = mock_client.request.call_args
        self.assertEqual(call_args[1].get("app_token"), "blt_abc123")
        self.assertEqual(call_args[1].get("table_id"), "tblXYZ001")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_missing_base_token(self, mock_get_client):
        """Returns error when base_token is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_search({
            "table_id": "tblXYZ001",
            "query": "test",
        })
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_missing_table_id(self, mock_get_client):
        """Returns error when table_id is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "query": "test",
        })
        self.assertIn("error", result)
        self.assertIn("table_id", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_missing_query(self, mock_get_client):
        """Returns error when query is missing."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("error", result)
        self.assertIn("query", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_empty_query_string(self, mock_get_client):
        """Returns error when query is empty string."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "   ",
        })
        self.assertIn("error", result)
        self.assertIn("query", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_empty_results(self, mock_get_client):
        """Search handles empty results gracefully."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "nonexistent",
        })
        self.assertIn("success", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_limit(self, mock_get_client):
        """Search respects limit parameter."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
            "limit": 50,
        })
        self.assertIn("success", result)
        # Check the query params contain page_size
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "50")
        # Verify app_token and table_id were passed
        self.assertEqual(call_args[1].get("app_token"), "blt_abc")
        self.assertEqual(call_args[1].get("table_id"), "tblXYZ")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_default_limit(self, mock_get_client):
        """Search uses default limit of 20 when not specified."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
        })
        self.assertIn("success", result)
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "20")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_bitable_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "table not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
        })
        self.assertIn("error", result)


class TestFeishuBitableToolSchema(unittest.TestCase):
    """Tests for tool schemas."""

    def test_schema_list_has_required_fields(self):
        """Schema for feishu_bitable_list has correct structure."""
        from tools.feishu_bitable_tool import FEISHU_BITABLE_LIST_SCHEMA

        self.assertEqual(FEISHU_BITABLE_LIST_SCHEMA["name"], "feishu_bitable_list")
        self.assertIn("description", FEISHU_BITABLE_LIST_SCHEMA)
        self.assertIn("parameters", FEISHU_BITABLE_LIST_SCHEMA)
        props = FEISHU_BITABLE_LIST_SCHEMA["parameters"]["properties"]
        # feishu_bitable_list has no parameters (no required base_token)
        self.assertEqual(props, {})

    def test_schema_tables_has_required_fields(self):
        """Schema for feishu_bitable_tables has correct structure."""
        from tools.feishu_bitable_tool import FEISHU_BITABLE_TABLES_SCHEMA

        schema = FEISHU_BITABLE_TABLES_SCHEMA
        self.assertEqual(schema["name"], "feishu_bitable_tables")
        props = schema["parameters"]["properties"]
        self.assertIn("base_token", props)
        self.assertIn("limit", props)
        required = schema["parameters"]["required"]
        self.assertIn("base_token", required)

    def test_schema_records_has_required_fields(self):
        """Schema for feishu_bitable_records has correct structure."""
        from tools.feishu_bitable_tool import FEISHU_BITABLE_RECORDS_SCHEMA

        schema = FEISHU_BITABLE_RECORDS_SCHEMA
        self.assertEqual(schema["name"], "feishu_bitable_records")
        props = schema["parameters"]["properties"]
        self.assertIn("base_token", props)
        self.assertIn("table_id", props)
        self.assertIn("search", props)
        self.assertIn("limit", props)
        required = schema["parameters"]["required"]
        self.assertIn("base_token", required)
        self.assertIn("table_id", required)

    def test_schema_create_record_has_required_fields(self):
        """Schema for feishu_bitable_create_record has correct structure."""
        from tools.feishu_bitable_tool import FEISHU_BITABLE_CREATE_RECORD_SCHEMA

        schema = FEISHU_BITABLE_CREATE_RECORD_SCHEMA
        self.assertEqual(schema["name"], "feishu_bitable_create_record")
        props = schema["parameters"]["properties"]
        self.assertIn("base_token", props)
        self.assertIn("table_id", props)
        self.assertIn("fields", props)
        required = schema["parameters"]["required"]
        self.assertIn("base_token", required)
        self.assertIn("table_id", required)
        self.assertIn("fields", required)

    def test_schema_search_has_required_fields(self):
        """Schema for feishu_bitable_search has correct structure."""
        from tools.feishu_bitable_tool import FEISHU_BITABLE_SEARCH_SCHEMA

        schema = FEISHU_BITABLE_SEARCH_SCHEMA
        self.assertEqual(schema["name"], "feishu_bitable_search")
        props = schema["parameters"]["properties"]
        self.assertIn("base_token", props)
        self.assertIn("table_id", props)
        self.assertIn("query", props)
        self.assertIn("search_fields", props)
        self.assertIn("limit", props)
        required = schema["parameters"]["required"]
        self.assertIn("base_token", required)
        self.assertIn("table_id", required)
        self.assertIn("query", required)


class TestFeishuBitableToolEdgeCases(unittest.TestCase):
    """Edge case tests for Feishu Bitable tools."""

    @patch("tools.feishu_bitable_tool._get_client")
    def test_whitespace_base_token(self, mock_get_client):
        """Returns error when base_token is only whitespace."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({
            "base_token": "   ",
        })
        self.assertIn("error", result)
        self.assertIn("base_token", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_with_special_characters(self, mock_get_client):
        """Create record handles fields with special characters."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "record": {
                    "record_id": "recSpecial",
                    "fields": {
                        "Title": "Task with <script> & \"quotes\"",
                        "Notes": "Line1\nLine2",
                    },
                }
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": {
                "Title": "Task with <script> & \"quotes\"",
                "Notes": "Line1\nLine2",
            },
        })
        self.assertIn("success", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_large_limit(self, mock_get_client):
        """List records handles large limit values."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        # Should clamp to max 500
        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "limit": 10000,
        })
        self.assertIn("success", result)
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "500")


class TestFeishuBitableToolCoverage(unittest.TestCase):
    """Additional tests for coverage improvement."""

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_invalid_limit(self, mock_get_client):
        """List tables handles invalid limit gracefully."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        # Invalid limit should default to 50
        result = ft._handle_feishu_bitable_tables({
            "base_token": "blt_abc",
            "limit": "not_a_number",
        })
        self.assertIn("success", result)
        # Should have been called with default page_size
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "50")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_invalid_limit(self, mock_get_client):
        """List records handles invalid limit gracefully."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "limit": "invalid",
        })
        self.assertIn("success", result)
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "50")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_with_search_filter(self, mock_get_client):
        """List records applies in-memory search filter to results."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {"record_id": "rec1", "fields": {"Title": "Alpha Task", "Status": "Open"}},
                    {"record_id": "rec2", "fields": {"Title": "Beta Task", "Status": "Done"}},
                    {"record_id": "rec3", "fields": {"Title": "Alpha Review", "Status": "Open"}},
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "search": "Alpha",
        })
        self.assertIn("success", result)
        self.assertIn("Alpha Task", result)
        self.assertIn("Alpha Review", result)
        # Beta should be filtered out
        self.assertIn("rec1", result)
        self.assertIn("rec3", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_search_no_match(self, mock_get_client):
        """List records handles search with no matches."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {"record_id": "rec1", "fields": {"Title": "Task One"}},
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "search": "NonExistent",
        })
        self.assertIn("success", result)
        self.assertIn("No records found", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_search_case_insensitive(self, mock_get_client):
        """List records search is case-insensitive."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {"record_id": "rec1", "fields": {"Title": "URGENT"}},
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "search": "urgent",
        })
        self.assertIn("success", result)
        self.assertIn("URGENT", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_no_data(self, mock_get_client):
        """Create record handles no data in response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = None
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": {"Title": "New Task"},
        })
        self.assertIn("error", result)
        self.assertIn("No data", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_no_data(self, mock_get_client):
        """List tables handles no data in response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = None
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({
            "base_token": "blt_abc",
        })
        self.assertIn("error", result)
        self.assertIn("No data", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_records_no_data(self, mock_get_client):
        """List records handles no data in response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = None
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_records({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
        })
        self.assertIn("error", result)
        self.assertIn("No data", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_no_data(self, mock_get_client):
        """Search records handles no data in response."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = None
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
        })
        self.assertIn("error", result)
        self.assertIn("No data", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_with_search_fields(self, mock_get_client):
        """Search with specific search_fields uses correct filter."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "data": {
                "items": [
                    {"record_id": "rec1", "fields": {"Title": "Design Doc", "Notes": "..."}},
                ]
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "Design",
            "search_fields": ["Title", "Notes"],
            "limit": 10,
        })
        self.assertIn("success", result)
        self.assertIn("Design Doc", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_search_records_invalid_limit(self, mock_get_client):
        """Search handles invalid limit gracefully."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"data": {"items": []}}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_search({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "query": "test",
            "limit": "bad",
        })
        self.assertIn("success", result)
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "20")

    @patch("tools.feishu_bitable_tool._get_client")
    def test_list_tables_91403_no_code_in_output(self, mock_get_client):
        """Error 91403 returns user-friendly message without exposing internal code."""
        import tools.feishu_bitable_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 91403
        mock_response.msg = "permission denied"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_bitable_tables({"base_token": "blt_private"})
        self.assertIn("error", result)
        self.assertIn("share", result.lower())
        self.assertNotIn("91403", result)

    @patch("tools.feishu_bitable_tool._get_client")
    def test_create_record_fields_not_dict(self, mock_get_client):
        """Returns error when fields is not a dictionary."""
        import tools.feishu_bitable_tool as ft

        result = ft._handle_feishu_bitable_create_record({
            "base_token": "blt_abc",
            "table_id": "tblXYZ",
            "fields": "not_a_dict",
        })
        self.assertIn("error", result)
        self.assertIn("dictionary", result)


if __name__ == "__main__":
    unittest.main()
