"""Tests for the calendar plugin.

Covers the plugin at ``plugins/calendar/``:
  * calendar_plugin.py library: subprocess calls to gws with correct args,
    JSON parsing, error handling.
  * Plugin __init__: tool registration with correct schemas and check_fn.
  * Fake gws stub binary returning canned JSON (no real gws dependency).
"""

from __future__ import annotations

import importlib
import json
import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def fake_gws_bin(tmp_path, monkeypatch):
    """Create a fake gws stub binary that returns canned JSON responses.

    The stub inspects argv[1:] to determine which response to return.
    """
    stub_path = tmp_path / "gws"
    stub_path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        'if args[0] == "calendar":\n'
        '    sub = args[1] if len(args) > 1 else ""\n'
        '    if sub == "list":\n'
        "        print(json.dumps([\n"
        '            {"id": "ev1", "title": "Team Standup", "start": "2026-05-28T09:00:00+03:00", "end": "2026-05-28T09:30:00+03:00"},\n'
        '            {"id": "ev2", "title": "\u05e4\u05d2\u05d9\u05e9\u05ea \u05dc\u05e7\u05d5\u05d7", "start": "2026-05-28T11:00:00+03:00", "end": "2026-05-28T12:00:00+03:00"},\n'
        "        ]))\n"
        '    elif sub == "create":\n'
        '        title = args[args.index("--title") + 1] if "--title" in args else "Untitled"\n'
        "        print(json.dumps({\n"
        '            "id": "ev-new",\n'
        '            "title": title,\n'
        '            "status": "confirmed",\n'
        "        }))\n"
        '    elif sub == "update":\n'
        "        print(json.dumps({\n"
        '            "id": args[2] if len(args) > 2 else "ev-unknown",\n'
        '            "status": "updated",\n'
        "        }))\n"
        '    elif sub == "delete":\n'
        "        print(json.dumps({\n"
        '            "id": args[2] if len(args) > 2 else "ev-unknown",\n'
        '            "status": "cancelled",\n'
        "        }))\n"
        '    elif sub == "freebusy":\n'
        "        print(json.dumps([\n"
        '            {"start": "2026-05-28T12:00:00+03:00", "end": "2026-05-28T12:30:00+03:00"},\n'
        '            {"start": "2026-05-28T15:00:00+03:00", "end": "2026-05-28T15:30:00+03:00"},\n'
        "        ]))\n"
        '    elif sub == "auth" and "status" in args:\n'
        '        print(json.dumps({"account": "operator@agente.dev", "connected": True}))\n'
        "    else:\n"
        '        print(json.dumps({"status": "ok"}))\n'
        "else:\n"
        '    print(json.dumps({"status": "ok"}))\n'
        "sys.exit(0)\n"
    )
    stub_path.chmod(stub_path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("AGENTE_GWS_BIN", str(stub_path))
    return stub_path


@pytest.fixture
def _gws_bin_unset(monkeypatch):
    """Ensure AGENTE_GWS_BIN is unset (every test sets it explicitly)."""
    monkeypatch.delenv("AGENTE_GWS_BIN", raising=False)


def _load_calendar_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "calendar"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.calendar",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_lib():
    repo_root = Path(__file__).resolve().parents[2]
    lib_path = repo_root / "plugins" / "calendar" / "calendar_plugin.py"
    spec = importlib.util.spec_from_file_location(
        "calendar_plugin_under_test", lib_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGwsSubprocess:
    """Tests for calendar_plugin.py subprocess calls."""

    def test_list_events(self, fake_gws_bin):
        lib = _load_lib()
        events = lib.list_events("2026-05-28", "2026-06-04")
        assert isinstance(events, list)
        assert len(events) == 2
        assert events[0]["id"] == "ev1"
        assert events[0]["title"] == "Team Standup"

    def test_list_events_with_calendar_id(self, fake_gws_bin):
        lib = _load_lib()
        events = lib.list_events("2026-05-28", "2026-06-04", calendar_id="alt@group.calendar.google.com")
        assert len(events) == 2

    def test_create_event(self, fake_gws_bin):
        lib = _load_lib()
        event = lib.create_event(
            start="2026-05-28T14:00:00+03:00",
            end="2026-05-28T15:00:00+03:00",
            title="Test Meeting",
        )
        assert event["id"] == "ev-new"
        assert event["status"] == "confirmed"

    def test_create_event_with_optional_fields(self, fake_gws_bin):
        lib = _load_lib()
        event = lib.create_event(
            start="2026-05-28T14:00:00+03:00",
            end="2026-05-28T15:00:00+03:00",
            title="Test with location",
            description="Agenda here",
            location="Room A",
        )
        assert event["status"] == "confirmed"

    def test_update_event(self, fake_gws_bin):
        lib = _load_lib()
        event = lib.update_event(event_id="ev1", title="Updated Title")
        assert event["id"] == "ev1"
        assert event["status"] == "updated"

    def test_cancel_event(self, fake_gws_bin):
        lib = _load_lib()
        event = lib.cancel_event("ev1")
        assert event["status"] == "cancelled"

    def test_find_free_slots(self, fake_gws_bin):
        lib = _load_lib()
        slots = lib.find_free_slots(duration_minutes=30, within="2026-05-28")
        assert isinstance(slots, list)
        assert len(slots) == 2
        assert slots[0]["start"] == "2026-05-28T12:00:00+03:00"

    def test_find_free_slots_custom_duration(self, fake_gws_bin):
        lib = _load_lib()
        slots = lib.find_free_slots(duration_minutes=60, within="2026-05-28")
        assert len(slots) == 2

    def test_hebrew_title_preserved(self, fake_gws_bin):
        lib = _load_lib()
        events = lib.list_events("2026-05-28", "2026-06-04")
        hebrew_event = events[1]
        assert hebrew_event["title"] == "פגישת לקוח"

    def test_gws_bin_unset_raises(self, _gws_bin_unset):
        lib = _load_lib()
        with pytest.raises(RuntimeError, match="AGENTE_GWS_BIN is not set"):
            lib.list_events("2026-05-28", "2026-06-04")


class TestPluginRegistration:
    """Test that the __init__.py register() function works correctly."""

    def test_register_tools(self, fake_gws_bin):
        registered = []

        class FakeCtx:
            def register_tool(self, name, toolset, schema, handler, check_fn, emoji):
                registered.append({
                    "name": name,
                    "toolset": toolset,
                    "schema": schema,
                    "handler": handler,
                    "check_fn": check_fn,
                    "emoji": emoji,
                })

        mod = _load_calendar_plugin()
        mod.register(FakeCtx())

        assert len(registered) == 5
        tool_names = {r["name"] for r in registered}
        assert tool_names == {
            "list_events",
            "create_event",
            "update_event",
            "cancel_event",
            "find_free_slots",
        }

        for r in registered:
            assert r["toolset"] == "calendar"
            assert callable(r["handler"])
            assert callable(r["check_fn"])

    def test_check_fn_returns_none_when_gws_available(self, fake_gws_bin):
        mod = _load_calendar_plugin()
        result = mod._check_gws()
        assert result is None

    def test_check_fn_returns_error_when_gws_unset(self, _gws_bin_unset):
        mod = _load_calendar_plugin()
        result = mod._check_gws()
        assert result is not None
        assert len(result) == 1
        assert "AGENTE_GWS_BIN" in result[0]

    def test_check_fn_returns_error_when_gws_not_found(self, tmp_path, monkeypatch):
        missing_path = str(tmp_path / "nonexistent-gws")
        monkeypatch.setenv("AGENTE_GWS_BIN", missing_path)
        mod = _load_calendar_plugin()
        result = mod._check_gws()
        assert result is not None
        assert len(result) == 1
        assert "not found" in result[0]


class TestSchemas:
    """Verify tool schemas have the expected shape."""

    def test_all_schemas_present(self, fake_gws_bin):
        mod = _load_calendar_plugin()
        from plugins.calendar.schemas import TOOL_SCHEMAS

        assert set(TOOL_SCHEMAS.keys()) == {
            "list_events",
            "create_event",
            "update_event",
            "cancel_event",
            "find_free_slots",
        }

    def test_schemas_have_required_fields(self, fake_gws_bin):
        from plugins.calendar.schemas import TOOL_SCHEMAS

        for name, schema in TOOL_SCHEMAS.items():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert "type" in schema["parameters"]
            assert "properties" in schema["parameters"]
            assert "required" in schema["parameters"]

    def test_list_events_requires_start_end(self, fake_gws_bin):
        from plugins.calendar.schemas import TOOL_SCHEMAS
        assert TOOL_SCHEMAS["list_events"]["parameters"]["required"] == ["start", "end"]

    def test_create_event_requires_start_end_title(self, fake_gws_bin):
        from plugins.calendar.schemas import TOOL_SCHEMAS
        assert TOOL_SCHEMAS["create_event"]["parameters"]["required"] == ["start", "end", "title"]
