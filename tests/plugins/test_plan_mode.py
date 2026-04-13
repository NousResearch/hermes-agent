"""Tests for the Plan Mode hook plugin (Phase B2)."""

import importlib
import os

import pytest
from tools.registry import registry

# Load plan_mode_hook via importlib (hyphenated directory name)
_PLUGIN_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "plugins", "hongxing-enhancements"
)
_spec = importlib.util.spec_from_file_location(
    "plan_mode_hook",
    os.path.join(_PLUGIN_DIR, "plan_mode_hook.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

enter_plan_mode = _mod.enter_plan_mode
exit_plan_mode = _mod.exit_plan_mode
is_active = _mod.is_active
pre_tool_call = _mod.pre_tool_call


@pytest.fixture(autouse=True)
def _reset_plan_mode():
    """Ensure plan mode is off before and after each test."""
    _mod._session_states.clear()
    yield
    _mod._session_states.clear()


@pytest.fixture
def mock_registry_metadata(monkeypatch):
    """Replace registry metadata lookups with test-controlled values."""

    def _apply(metadata_by_tool):
        monkeypatch.setattr(
            registry,
            "get_metadata",
            lambda name: metadata_by_tool.get(name, {}),
        )

    return _apply


# ── Inactive state ─────────────────────────────────────────────────────

class TestInactive:
    def test_inactive_allows_all(self):
        assert pre_tool_call("write_file", {"path": "/tmp/x"}) is None

    def test_inactive_allows_terminal(self):
        assert pre_tool_call("terminal", {"command": "rm -rf /"}) is None


# ── Activation / deactivation ─────────────────────────────────────────

class TestActivation:
    def test_enter_plan_mode(self):
        enter_plan_mode()
        assert is_active() is True

    def test_exit_plan_mode(self):
        enter_plan_mode()
        exit_plan_mode()
        assert is_active() is False


class TestSessionIsolation:
    def test_session_a_plan_mode_does_not_affect_session_b(self):
        enter_plan_mode("session-a")

        assert is_active("session-a") is True
        assert is_active("session-b") is False
        assert pre_tool_call("write_file", {"path": "/tmp/x"}, session_id="session-a")["action"] == "deny"
        assert pre_tool_call("write_file", {"path": "/tmp/x"}, session_id="session-b") is None

    def test_exit_session_a_does_not_change_session_b_state(self):
        enter_plan_mode("session-a")
        enter_plan_mode("session-b")

        exit_plan_mode("session-a")

        assert is_active("session-a") is False
        assert is_active("session-b") is True
        assert pre_tool_call("write_file", {"path": "/tmp/x"}, session_id="session-a") is None
        assert pre_tool_call("write_file", {"path": "/tmp/x"}, session_id="session-b")["action"] == "deny"

    def test_default_session_id_behaves_as_before(self):
        enter_plan_mode()

        assert is_active() is True
        assert is_active("default") is True
        assert pre_tool_call("write_file", {"path": "/tmp/x"})["action"] == "deny"
        assert pre_tool_call("write_file", {"path": "/tmp/x"}, session_id="default")["action"] == "deny"

    def test_session_isolation_still_works(self):
        enter_plan_mode("session-a")

        assert pre_tool_call("terminal", {"command": "ls"}, session_id="session-a")["action"] == "deny"
        assert pre_tool_call("terminal", {"command": "ls"}, session_id="session-b") is None


# ── Allowed tools in plan mode ─────────────────────────────────────────

class TestAllowedInPlanMode:
    def test_read_file_allowed(self):
        import tools.file_tools  # noqa: F401

        enter_plan_mode()
        assert pre_tool_call("read_file", {}) is None

    def test_search_files_allowed(self):
        import tools.file_tools  # noqa: F401

        enter_plan_mode()
        assert pre_tool_call("search_files", {"query": "foo"}) is None

    def test_session_search_allowed(self):
        enter_plan_mode()
        assert pre_tool_call("session_search", {"query": "bar"}) is None

    def test_plan_mode_tool_allowed(self):
        enter_plan_mode()
        assert pre_tool_call("plan_mode", {"action": "status"}) is None


class TestRegistryMetadataInPlanMode:
    def test_metadata_true_allows_tool(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {"registry_allowed_tool": {"allowed_in_plan_mode_default": True}}
        )

        assert pre_tool_call("registry_allowed_tool", {}) is None

    def test_metadata_false_denies_tool(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {"registry_denied_tool": {"allowed_in_plan_mode_default": False}}
        )

        result = pre_tool_call("registry_denied_tool", {})
        assert result is not None
        assert result["action"] == "deny"

    def test_missing_metadata_falls_back_to_legacy_allowlist(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata({})

        assert pre_tool_call("read_file", {}) is None

    def test_plan_mode_override_is_always_allowed(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {"plan_mode": {"allowed_in_plan_mode_default": False}}
        )

        assert pre_tool_call("plan_mode", {"action": "status"}) is None

    def test_mutates_agent_state_denied(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {
                "stateful_tool": {
                    "allowed_in_plan_mode_default": True,
                    "mutates_agent_state": True,
                }
            }
        )

        result = pre_tool_call("stateful_tool", {})
        assert result is not None
        assert result["action"] == "deny"

    def test_mutates_local_fs_denied(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {
                "file_mutation_tool": {
                    "mutates_local_fs": True,
                }
            }
        )

        result = pre_tool_call("file_mutation_tool", {})
        assert result is not None
        assert result["action"] == "deny"

    def test_high_risk_denied(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {
                "high_risk_tool": {
                    "allowed_in_plan_mode_default": True,
                    "risk_level": "high",
                }
            }
        )

        result = pre_tool_call("high_risk_tool", {})
        assert result is not None
        assert result["action"] == "deny"

    def test_metadata_missing_non_allowlist_denied(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata({})

        result = pre_tool_call("unknown_non_allowlisted_tool", {})
        assert result is not None
        assert result["action"] == "deny"

    def test_metadata_conflict_conservative(self, mock_registry_metadata):
        enter_plan_mode()
        mock_registry_metadata(
            {
                "conflicted_tool": {
                    "allowed_in_plan_mode_default": True,
                    "mutates_external_world": True,
                }
            }
        )

        result = pre_tool_call("conflicted_tool", {})
        assert result is not None
        assert result["action"] == "deny"


# ── Denied tools in plan mode ──────────────────────────────────────────

class TestDeniedInPlanMode:
    def test_write_file_denied(self):
        import tools.file_tools  # noqa: F401

        enter_plan_mode()
        result = pre_tool_call("write_file", {"path": "/tmp/x"})
        assert result is not None
        assert result["action"] == "deny"

    def test_patch_denied(self):
        import tools.file_tools  # noqa: F401

        enter_plan_mode()
        result = pre_tool_call("patch", {"path": "/tmp/x"})
        assert result is not None
        assert result["action"] == "deny"

    def test_terminal_denied(self):
        enter_plan_mode()
        result = pre_tool_call("terminal", {"command": "ls"})
        assert result is not None
        assert result["action"] == "deny"

    def test_delegate_task_denied(self):
        enter_plan_mode()
        result = pre_tool_call("delegate_task", {"goal": "do stuff"})
        assert result is not None
        assert result["action"] == "deny"

    def test_todo_denied_in_plan_mode(self):
        import tools.todo_tool  # noqa: F401

        enter_plan_mode()
        result = pre_tool_call("todo", {"todos": []})
        assert result is not None
        assert result["action"] == "deny"


# ── Memory action-aware filtering ─────────────────────────────────────

class TestMemoryInPlanMode:
    def test_memory_read_allowed(self):
        enter_plan_mode()
        assert pre_tool_call("memory", {"action": "read"}) is None

    def test_memory_add_denied(self):
        enter_plan_mode()
        result = pre_tool_call("memory", {"action": "add", "content": "x"})
        assert result is not None
        assert result["action"] == "deny"

    def test_memory_replace_denied(self):
        enter_plan_mode()
        result = pre_tool_call("memory", {"action": "replace"})
        assert result is not None
        assert result["action"] == "deny"


# ── Exit restores normal behavior ─────────────────────────────────────

class TestExitRestores:
    def test_exit_restores_write_file(self):
        enter_plan_mode()
        assert pre_tool_call("write_file", {})["action"] == "deny"
        exit_plan_mode()
        assert pre_tool_call("write_file", {}) is None

    def test_exit_restores_memory_add(self):
        enter_plan_mode()
        assert pre_tool_call("memory", {"action": "add"})["action"] == "deny"
        exit_plan_mode()
        assert pre_tool_call("memory", {"action": "add"}) is None
