"""Tests for the permission engine plugin."""

import importlib
import os

import pytest

# The plugin directory uses a hyphen (hongxing-enhancements) which is not
# a valid Python identifier, so we use importlib to load the module.
_PLUGIN_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "plugins", "hongxing-enhancements"
)
_spec = importlib.util.spec_from_file_location(
    "permission_engine",
    os.path.join(_PLUGIN_DIR, "permission_engine.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate_permission = _mod.evaluate_permission


@pytest.fixture(autouse=True)
def stub_tool_metadata(monkeypatch):
    monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {})


# ── Layer 1: Whitelist (always allow) ───────────────────────────────────

class TestWhitelist:
    def test_read_file_allowed(self):
        assert evaluate_permission("read_file", {}) is None

    def test_search_files_allowed(self):
        assert evaluate_permission("search_files", {}) is None


# ── Layer 4: Terminal risk rules ────────────────────────────────────────

class TestTerminalRules:
    def test_safe_ls(self):
        assert evaluate_permission("terminal", {"command": "ls -la /tmp"}) is None

    def test_safe_cat(self):
        assert evaluate_permission("terminal", {"command": "cat /etc/hosts"}) is None

    def test_safe_grep(self):
        assert evaluate_permission("terminal", {"command": "grep foo bar.txt"}) is None

    def test_terminal_metadata_does_not_force_global_confirmation(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "risk_level": "high",
            "mutates_external_world": True,
            "requires_confirmation_default": True,
        })
        assert evaluate_permission("terminal", {"command": "git status --short --branch"}) is None

    @pytest.mark.parametrize(
        "command",
        [
            "source venv/bin/activate && git status --short --branch",
            "cd /tmp/project && git log --oneline -5",
            "bash -lc 'git diff --stat'",
            "zsh -lc 'python -m pytest -q tests/plugins/test_permission_engine.py'",
        ],
    )
    def test_wrapped_safe_local_dev_commands_allowed(self, command):
        assert evaluate_permission("terminal", {"command": command}) is None

    def test_rm_rf_root_denied(self):
        result = evaluate_permission("terminal", {"command": "rm -rf /"})
        assert result is not None
        assert result["action"] == "deny"

    def test_rm_rf_home_denied(self):
        result = evaluate_permission("terminal", {"command": "rm -rf /home/user"})
        assert result is not None
        assert result["action"] == "deny"

    def test_mkfs_denied(self):
        result = evaluate_permission("terminal", {"command": "mkfs.ext4 /dev/sda"})
        assert result is not None
        assert result["action"] == "deny"

    def test_sudo_asks(self):
        result = evaluate_permission("terminal", {"command": "sudo apt install foo"})
        assert result is not None
        assert result["action"] == "ask"


# ── Layer 4: Write / patch path rules ──────────────────────────────────

class TestWritePathRules:
    def test_write_safe_path(self):
        assert evaluate_permission("write_file", {"file_path": "/tmp/test.py"}) is None

    def test_write_etc_asks(self):
        result = evaluate_permission("write_file", {"file_path": "/etc/passwd"})
        assert result is not None
        assert result["action"] == "ask"

    def test_write_ssh_asks(self):
        ssh_path = os.path.expanduser("~/.ssh/authorized_keys")
        result = evaluate_permission("write_file", {"file_path": ssh_path})
        assert result is not None
        assert result["action"] == "ask"

    def test_patch_usr_asks(self):
        result = evaluate_permission("patch", {"path": "/usr/local/bin/script"})
        assert result is not None
        assert result["action"] == "ask"

    def test_patch_safe_path(self):
        assert evaluate_permission("patch", {"path": "/home/user/project/main.py"}) is None


# ── Layer 3: Registry metadata rules ────────────────────────────────────

class TestRegistryMetadataRules:
    def test_metadata_high_risk_asks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "risk_level": "high",
        })
        result = evaluate_permission("browser_click", {})
        assert result is not None
        assert result["action"] == "ask"
        assert result["reason"] == "high risk tool"

    def test_metadata_mutates_external_world_asks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "mutates_external_world": True,
        })
        result = evaluate_permission("send_message", {})
        assert result is not None
        assert result["action"] == "ask"
        assert result["reason"] == "mutates external world"

    def test_metadata_requires_confirmation_asks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "requires_confirmation_default": True,
        })
        result = evaluate_permission("browser_click", {})
        assert result is not None
        assert result["action"] == "ask"
        assert result["reason"] == "requires confirmation"

    def test_no_metadata_falls_through_to_pattern_rules(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {})
        result = evaluate_permission("terminal", {"command": "sudo apt install foo"})
        assert result is not None
        assert result["action"] == "ask"
        assert "requires confirmation" in result["reason"]

    def test_whitelist_takes_priority_over_metadata(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "risk_level": "high",
            "mutates_external_world": True,
            "requires_confirmation_default": True,
        })
        assert evaluate_permission("read_file", {}) is None


# ── Layer 5: Default allow ──────────────────────────────────────────────

class TestDefaultAllow:
    def test_memory_tool(self):
        assert evaluate_permission("memory", {"action": "add"}) is None

    def test_random_tool(self):
        assert evaluate_permission("some_random_tool", {}) is None
