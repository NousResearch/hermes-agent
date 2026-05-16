"""Tests for Feishu Calendar Tool.

TDD Phase 1: Write tests that describe expected behavior.
These tests will FAIL until the tool is implemented.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


class TestFeishuCalendarToolUnit(unittest.TestCase):
    """Unit tests for Feishu calendar tool functions."""

    # -------------------------------------------------------------------------
    # feishu_calendar_list
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_calendars_success(self, mock_get_client):
        """List calendars returns a formatted list of calendars."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock response with calendars
        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "calendar_list": [
                {
                    "calendar": {
                        "calendar_id": "feishu_calendar_1",
                        "summary": "Primary Calendar",
                        "type": "primary",
                        "summary_alias": "Main",
                    }
                },
                {
                    "calendar": {
                        "calendar_id": "feishu_calendar_2",
                        "summary": "Work Calendar",
                        "type": "calendar",
                        "summary_alias": "Work",
                    }
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_list({})
        self.assertIn("success", result)
        self.assertIn("Primary Calendar", result)
        self.assertIn("feishu_calendar_1", result)
        self.assertIn("Work Calendar", result)
        mock_client.request.assert_called_once()

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_calendars_primary_calendar(self, mock_get_client):
        """List with calendar_id='primary' returns the user's primary calendar."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "calendar_list": [
                {
                    "calendar": {
                        "calendar_id": "primary",
                        "summary": "My Primary Calendar",
                        "type": "primary",
                    }
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_list({"calendar_id": "primary"})
        self.assertIn("My Primary Calendar", result)
        # Verify the request was made
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_calendars_client_unavailable(self, mock_get_client):
        """Returns error when Feishu client is not available."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_list({})
        self.assertIn("error", result)
        self.assertIn("not available", result.lower())

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_calendars_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "app_access_token is invalid"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_list({})
        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_calendar_events
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_success(self, mock_get_client):
        """List events returns formatted events."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "items": [
                {
                    "event_id": "evt_abc123",
                    "summary": "Team Meeting",
                    "start_time": {"timestamp": "1743000000", "timezone": "Asia/Shanghai"},
                    "end_time": {"timestamp": "1743003600", "timezone": "Asia/Shanghai"},
                    "status": "confirmed",
                },
                {
                    "event_id": "evt_def456",
                    "summary": "Project Review",
                    "start_time": {"timestamp": "1743100000", "timezone": "Asia/Shanghai"},
                    "end_time": {"timestamp": "1743103600", "timezone": "Asia/Shanghai"},
                    "status": "tentative",
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_events({
            "calendar_id": "primary",
        })
        self.assertIn("success", result)
        self.assertIn("Team Meeting", result)
        self.assertIn("evt_abc123", result)
        self.assertIn("Project Review", result)

        # Verify request was made
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_missing_calendar_id(self, mock_get_client):
        """Returns error when calendar_id is missing."""
        import tools.feishu_calendar_tool as ft

        result = ft._handle_feishu_calendar_events({})
        self.assertIn("error", result)
        self.assertIn("calendar_id", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_with_time_range(self, mock_get_client):
        """List events respects start_time and end_time parameters."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        start = "2026-04-01T00:00:00Z"
        end = "2026-04-30T23:59:59Z"
        result = ft._handle_feishu_calendar_events({
            "calendar_id": "primary",
            "start_time": start,
            "end_time": end,
        })

        self.assertIn("success", result)
        # Verify time range in request
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        # Check that request has queries with start_time and end_time
        queries_list = getattr(req, "queries", [])
        query_keys = [k for k, v in queries_list]
        self.assertIn("start_time", query_keys)
        self.assertIn("end_time", query_keys)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_limit(self, mock_get_client):
        """List events respects limit parameter."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_events({
            "calendar_id": "primary",
            "limit": 5,
        })
        self.assertIn("success", result)
        # Verify page_size was set in request
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_dict = {k: v for k, v in queries_list}
        self.assertEqual(query_dict.get("page_size"), "5")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "app_access_token is invalid"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_events({"calendar_id": "primary"})
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_calendar_create_event
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_create_event_success(self, mock_get_client):
        """Create event returns the created event details."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "event": {
                "event_id": "new_evt_789",
                "summary": "New Meeting",
                "start_time": {"timestamp": "1743000000"},
                "end_time": {"timestamp": "1743003600"},
                "status": "confirmed",
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "summary": "New Meeting",
            "start_time": "2026-04-01T10:00:00Z",
            "end_time": "2026-04-01T11:00:00Z",
        })

        self.assertIn("success", result)
        self.assertIn("new_evt_789", result)
        self.assertIn("New Meeting", result)

        # Verify request was made to correct endpoint
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_create_event_missing_required_fields(self, mock_get_client):
        """Returns error when required fields are missing."""
        import tools.feishu_calendar_tool as ft

        # Missing summary
        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "start_time": "2026-04-01T10:00:00Z",
            "end_time": "2026-04-01T11:00:00Z",
        })
        self.assertIn("error", result)
        self.assertIn("summary", result)

        # Missing start_time
        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "summary": "Test",
            "end_time": "2026-04-01T11:00:00Z",
        })
        self.assertIn("error", result)
        self.assertIn("start_time", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_create_event_with_optional_fields(self, mock_get_client):
        """Create event handles optional description, location, attendees."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "event": {
                "event_id": "evt_xyz",
                "summary": "Conference",
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "summary": "Conference",
            "start_time": "2026-04-01T09:00:00Z",
            "end_time": "2026-04-01T17:00:00Z",
            "description": "Annual conference",
            "location": "Beijing Office",
            "attendees": [
                {"type": "user", "user_id": "ou_abc123"},
                {"type": "user", "user_id": "ou_def456"},
            ],
        })

        self.assertIn("success", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_create_event_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "summary": "Test",
            "start_time": "2026-04-01T10:00:00Z",
            "end_time": "2026-04-01T11:00:00Z",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_create_event_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "calendar not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "invalid",
            "summary": "Test",
            "start_time": "2026-04-01T10:00:00Z",
            "end_time": "2026-04-01T11:00:00Z",
        })
        self.assertIn("error", result)
        self.assertIn("99991663", result)

    # -------------------------------------------------------------------------
    # feishu_calendar_update_event
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_update_event_success(self, mock_get_client):
        """Update event returns the updated event details."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "event": {
                "event_id": "evt_update",
                "summary": "Updated Title",
                "status": "confirmed",
            }
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_update_event({
            "calendar_id": "primary",
            "event_id": "evt_update",
            "summary": "Updated Title",
        })

        self.assertIn("success", result)
        self.assertIn("evt_update", result)
        self.assertIn("Updated Title", result)

        # Verify PATCH request was made
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")
        self.assertEqual(req.paths.get("event_id"), "evt_update")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_update_event_missing_event_id(self, mock_get_client):
        """Returns error when event_id is missing."""
        import tools.feishu_calendar_tool as ft

        result = ft._handle_feishu_calendar_update_event({
            "calendar_id": "primary",
            "summary": "New Title",
        })
        self.assertIn("error", result)
        self.assertIn("event_id", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_update_event_partial_update(self, mock_get_client):
        """Update event allows partial updates (only summary, only time, etc)."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "event": {
                "event_id": "evt_partial",
                "summary": "Renamed",
            }
        }
        mock_client.request.return_value = mock_response

        # Only update summary
        result = ft._handle_feishu_calendar_update_event({
            "calendar_id": "primary",
            "event_id": "evt_partial",
            "summary": "Renamed",
        })
        self.assertIn("success", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_update_event_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_update_event({
            "calendar_id": "primary",
            "event_id": "evt_123",
            "summary": "New Title",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_update_event_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "event not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_update_event({
            "calendar_id": "primary",
            "event_id": "nonexistent",
            "summary": "Title",
        })
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_calendar_delete_event
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_delete_event_success(self, mock_get_client):
        """Delete event returns success."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_delete_event({
            "calendar_id": "primary",
            "event_id": "evt_delete",
        })

        self.assertIn("success", result)

        # Verify DELETE request was made
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")
        self.assertEqual(req.paths.get("event_id"), "evt_delete")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_delete_event_missing_event_id(self, mock_get_client):
        """Returns error when event_id is missing."""
        import tools.feishu_calendar_tool as ft

        result = ft._handle_feishu_calendar_delete_event({
            "calendar_id": "primary",
        })
        self.assertIn("error", result)
        self.assertIn("event_id", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_delete_event_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_delete_event({
            "calendar_id": "primary",
            "event_id": "evt_123",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_delete_event_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "event not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_delete_event({
            "calendar_id": "primary",
            "event_id": "nonexistent",
        })
        self.assertIn("error", result)

    # -------------------------------------------------------------------------
    # feishu_calendar_attendees
    # -------------------------------------------------------------------------

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_attendees_success(self, mock_get_client):
        """List attendees returns formatted attendee list."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "items": [
                {
                    "attendee_id": {"user_id": "ou_abc123"},
                    "display_name": "Alice",
                    "type": "user",
                    "rsvp_status": "accepted",
                },
                {
                    "attendee_id": {"user_id": "ou_def456"},
                    "display_name": "Bob",
                    "type": "user",
                    "rsvp_status": "tentative",
                },
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
            "event_id": "evt_abc",
        })

        self.assertIn("success", result)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertIn("accepted", result)

        # Verify request was made
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        self.assertEqual(req.paths.get("calendar_id"), "primary")
        self.assertEqual(req.paths.get("event_id"), "evt_abc")

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_attendees_missing_event_id(self, mock_get_client):
        """Returns error when event_id is missing."""
        import tools.feishu_calendar_tool as ft

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
        })
        self.assertIn("error", result)
        self.assertIn("event_id", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_add_attendees_success(self, mock_get_client):
        """Add attendees to an event."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {
            "items": [
                {
                    "attendee_id": {"user_id": "ou_new"},
                    "display_name": "Charlie",
                    "type": "user",
                    "rsvp_status": "needs_action",
                }
            ]
        }
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
            "event_id": "evt_abc",
            "attendees": [
                {"type": "user", "user_id": "ou_new"},
            ],
        })

        self.assertIn("success", result)
        mock_client.request.assert_called_once()

    @patch("tools.feishu_calendar_tool._get_client")
    def test_attendees_client_unavailable(self, mock_get_client):
        """Returns error when client is not available."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
            "event_id": "evt_123",
        })
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_attendees_api_error(self, mock_get_client):
        """Returns error when API returns non-zero code."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 99991663
        mock_response.msg = "event not found"
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
            "event_id": "nonexistent",
        })
        self.assertIn("error", result)


class TestFeishuCalendarToolSchema(unittest.TestCase):
    """Tests for tool schemas."""

    def test_schema_list_calendars_has_required_fields(self):
        """Schema for feishu_calendar_list has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_LIST_SCHEMA

        self.assertEqual(FEISHU_CALENDAR_LIST_SCHEMA["name"], "feishu_calendar_list")
        self.assertIn("description", FEISHU_CALENDAR_LIST_SCHEMA)
        self.assertIn("parameters", FEISHU_CALENDAR_LIST_SCHEMA)
        props = FEISHU_CALENDAR_LIST_SCHEMA["parameters"]["properties"]
        self.assertIn("calendar_id", props)

    def test_schema_events_has_required_fields(self):
        """Schema for feishu_calendar_events has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_EVENTS_SCHEMA

        self.assertEqual(FEISHU_CALENDAR_EVENTS_SCHEMA["name"], "feishu_calendar_events")
        self.assertIn("parameters", FEISHU_CALENDAR_EVENTS_SCHEMA)
        props = FEISHU_CALENDAR_EVENTS_SCHEMA["parameters"]["properties"]
        self.assertIn("calendar_id", props)
        self.assertIn("start_time", props)
        self.assertIn("end_time", props)
        self.assertIn("limit", props)
        required = FEISHU_CALENDAR_EVENTS_SCHEMA["parameters"]["required"]
        self.assertIn("calendar_id", required)

    def test_schema_create_event_has_required_fields(self):
        """Schema for feishu_calendar_create_event has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_CREATE_EVENT_SCHEMA

        schema = FEISHU_CALENDAR_CREATE_EVENT_SCHEMA
        self.assertEqual(schema["name"], "feishu_calendar_create_event")
        props = schema["parameters"]["properties"]
        self.assertIn("calendar_id", props)
        self.assertIn("summary", props)
        self.assertIn("start_time", props)
        self.assertIn("end_time", props)
        self.assertIn("description", props)
        self.assertIn("location", props)
        self.assertIn("attendees", props)
        required = schema["parameters"]["required"]
        self.assertIn("calendar_id", required)
        self.assertIn("summary", required)
        self.assertIn("start_time", required)
        self.assertIn("end_time", required)

    def test_schema_update_event_has_required_fields(self):
        """Schema for feishu_calendar_update_event has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_UPDATE_EVENT_SCHEMA

        schema = FEISHU_CALENDAR_UPDATE_EVENT_SCHEMA
        self.assertEqual(schema["name"], "feishu_calendar_update_event")
        props = schema["parameters"]["properties"]
        self.assertIn("calendar_id", props)
        self.assertIn("event_id", props)
        self.assertIn("summary", props)
        self.assertIn("start_time", props)
        self.assertIn("end_time", props)
        self.assertIn("description", props)
        self.assertIn("location", props)
        required = schema["parameters"]["required"]
        self.assertIn("calendar_id", required)
        self.assertIn("event_id", required)

    def test_schema_delete_event_has_required_fields(self):
        """Schema for feishu_calendar_delete_event has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_DELETE_EVENT_SCHEMA

        schema = FEISHU_CALENDAR_DELETE_EVENT_SCHEMA
        self.assertEqual(schema["name"], "feishu_calendar_delete_event")
        props = schema["parameters"]["properties"]
        self.assertIn("calendar_id", props)
        self.assertIn("event_id", props)
        required = schema["parameters"]["required"]
        self.assertIn("calendar_id", required)
        self.assertIn("event_id", required)

    def test_schema_attendees_has_required_fields(self):
        """Schema for feishu_calendar_attendees has correct structure."""
        from tools.feishu_calendar_tool import FEISHU_CALENDAR_ATTENDEES_SCHEMA

        schema = FEISHU_CALENDAR_ATTENDEES_SCHEMA
        self.assertEqual(schema["name"], "feishu_calendar_attendees")
        props = schema["parameters"]["properties"]
        self.assertIn("calendar_id", props)
        self.assertIn("event_id", props)
        self.assertIn("attendees", props)
        self.assertIn("list_only", props)
        required = schema["parameters"]["required"]
        self.assertIn("calendar_id", required)
        self.assertIn("event_id", required)


class TestFeishuCalendarToolEdgeCases(unittest.TestCase):
    """Edge case tests for Feishu calendar tools."""

    @patch("tools.feishu_calendar_tool._get_client")
    def test_empty_calendar_list(self, mock_get_client):
        """List calendars handles empty response."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"calendar_list": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_list({})
        self.assertIn("success", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_empty_events_list(self, mock_get_client):
        """List events handles empty response."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_events({"calendar_id": "primary"})
        self.assertIn("success", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_list_events_default_time_range(self, mock_get_client):
        """List events uses reasonable default time range when not specified."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_events({"calendar_id": "primary"})
        self.assertIn("success", result)
        # Default range should be set in request
        call_args = mock_client.request.call_args
        req = call_args[0][0]
        queries_list = getattr(req, "queries", [])
        query_keys = [k for k, v in queries_list]
        self.assertIn("start_time", query_keys)
        self.assertIn("end_time", query_keys)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_invalid_iso8601_time(self, mock_get_client):
        """Handles invalid ISO8601 timestamp gracefully."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        # Invalid time should either error or handle gracefully
        result = ft._handle_feishu_calendar_create_event({
            "calendar_id": "primary",
            "summary": "Test",
            "start_time": "not-a-valid-time",
            "end_time": "2026-04-01T11:00:00Z",
        })
        # Should return an error for invalid ISO8601
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_lark_oapi_not_installed(self, mock_get_client):
        """Handles case when lark_oapi is not installed."""
        import tools.feishu_calendar_tool as ft

        mock_get_client.return_value = None

        result = ft._handle_feishu_calendar_list({})
        self.assertIn("error", result)

    @patch("tools.feishu_calendar_tool._get_client")
    def test_attendees_empty_list(self, mock_get_client):
        """Handles empty attendees list."""
        import tools.feishu_calendar_tool as ft

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.code = 0
        mock_response.data = {"items": []}
        mock_client.request.return_value = mock_response

        result = ft._handle_feishu_calendar_attendees({
            "calendar_id": "primary",
            "event_id": "evt_abc",
            "attendees": [],
        })
        self.assertIn("success", result)


if __name__ == "__main__":
    unittest.main()
