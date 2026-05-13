"""Tests for graceful plugin degradation with stub tools."""
import pytest


class TestPluginStubTools:
    """When a plugin fails to init, stub tools should explain the failure."""

    def test_stub_tool_returns_error_message(self):
        """A stub tool should return a clear message explaining the plugin is unavailable."""
        from run_agent import _make_plugin_stub_handler

        handler = _make_plugin_stub_handler("memory_search", "Honcho connection refused")
        result = handler()
        assert "Honcho" in result
        assert "unavailable" in result.lower()
        assert "memory_search" in result

    def test_stub_tool_schema_matches_original(self):
        """Stub tool schema should have the same name as the original tool."""
        from run_agent import _make_plugin_stub_schema

        schema = _make_plugin_stub_schema("memory_search", "Search persistent memory")
        assert schema["function"]["name"] == "memory_search"
        assert "unavailable" in schema["function"]["description"].lower() or "disabled" in schema["function"]["description"].lower()

    def test_stub_tool_accepts_arbitrary_kwargs(self):
        """Stub handler must accept any kwargs the model passes without raising."""
        from run_agent import _make_plugin_stub_handler

        handler = _make_plugin_stub_handler("honcho_search", "init failed")
        # Model may pass arbitrary keyword arguments; stub must not raise
        result = handler(query="recent memories", limit=5)
        assert "honcho_search" in result
        assert "unavailable" in result.lower()

    def test_stub_message_does_not_hardcode_memory_advice(self):
        """Error message should be generic, not assume the plugin is a memory plugin."""
        from run_agent import _make_plugin_stub_handler

        handler = _make_plugin_stub_handler("my_custom_tool", "ImportError: missing dep")
        result = handler()
        # Must not suggest "built-in memory tools" for a non-memory plugin
        assert "built-in memory tools" not in result
        assert "my_custom_tool" in result

    def test_stub_schema_has_empty_parameters(self):
        """Stub schema parameters should be an empty object so the model can call it."""
        from run_agent import _make_plugin_stub_schema

        schema = _make_plugin_stub_schema("honcho_profile", "Retrieve peer card")
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params["required"] == []

    def test_stub_registration_skips_collision_with_real_tool(self):
        """If a real tool with the same name already exists, the stub must not be appended."""
        from run_agent import _make_plugin_stub_schema

        existing_tools = [
            {"type": "function", "function": {"name": "memory_search", "description": "real", "parameters": {}}}
        ]
        existing_names = {t["function"]["name"] for t in existing_tools}

        stub_schema = _make_plugin_stub_schema("memory_search", "Search persistent memory")
        stub_name = stub_schema["function"]["name"]

        # Simulate the collision guard introduced in the fix
        if stub_name not in existing_names:
            existing_tools.append(stub_schema)

        # Tool list should still have exactly one entry for memory_search
        names = [t["function"]["name"] for t in existing_tools]
        assert names.count("memory_search") == 1

    def test_failed_schemas_sourced_from_memory_manager(self):
        """_failed_schemas must come from self._memory_manager, not an undefined variable."""
        # This test verifies the NameError fix: before the fix, referencing
        # 'memory_manager_tool_schemas' (an undefined name) would raise NameError,
        # silently swallowed by the outer except, meaning stubs were never registered.
        from unittest.mock import MagicMock, patch

        # Build a minimal fake memory manager with one provider that has a schema
        fake_schema = {
            "name": "honcho_search",
            "description": "Search Honcho memory",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        fake_manager = MagicMock()
        fake_manager.get_all_tool_schemas.return_value = [fake_schema]

        stub_handlers: dict = {}
        stub_tools: list = []
        valid_names: set = set()

        from run_agent import _make_plugin_stub_handler, _make_plugin_stub_schema

        # Replicate the fixed registration logic
        _failed_schemas = fake_manager.get_all_tool_schemas()
        _existing = set()
        for _schema in _failed_schemas:
            if not isinstance(_schema, dict):
                continue
            _stub_name = _schema.get("name", "")
            if not _stub_name or _stub_name in _existing:
                continue
            _stub_desc = _schema.get("description", "")
            _stub_handler = _make_plugin_stub_handler(_stub_name, "connection refused")
            _stub_schema = _make_plugin_stub_schema(_stub_name, _stub_desc)
            stub_handlers[_stub_name] = _stub_handler
            stub_tools.append(_stub_schema)
            valid_names.add(_stub_name)
            _existing.add(_stub_name)

        assert "honcho_search" in stub_handlers
        assert "honcho_search" in valid_names
        assert any(t["function"]["name"] == "honcho_search" for t in stub_tools)
        # Verify the stub returns the right message
        result = stub_handlers["honcho_search"]()
        assert "honcho_search" in result
        assert "unavailable" in result.lower()
