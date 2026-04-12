"""Tests for the permission engine plugin (Phase B1).

16 test cases covering the four evaluation layers.
"""

import importlib
import os
import sys

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


# ── Layer 1: Whitelist (always allow) ───────────────────────────────────

class TestWhitelist:
    def test_read_file_allowed(self):
        assert evaluate_permission("read_file", {}) is None

    def test_search_files_allowed(self):
        assert evaluate_permission("search_files", {}) is None


# ── Layer 3: Terminal risk rules ────────────────────────────────────────

class TestTerminalRules:
    def test_safe_ls(self):
        assert evaluate_permission("terminal", {"command": "ls -la /tmp"}) is None

    def test_safe_cat(self):
        assert evaluate_permission("terminal", {"command": "cat /etc/hosts"}) is None

    def test_safe_grep(self):
        assert evaluate_permission("terminal", {"command": "grep foo bar.txt"}) is None

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


# ── Layer 3: Write / patch path rules ──────────────────────────────────

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


# ── Layer 4: Registry metadata defaults (Phase B2) ─────────────────────

class TestRegistryMetadataRules:
    def test_metadata_requires_confirmation_asks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "requires_confirmation_default": True,
            "risk_level": "medium",
        })
        result = evaluate_permission("browser_click", {})
        assert result is not None
        assert result["action"] == "ask"

    def test_metadata_critical_external_world_denied(self, monkeypatch):
        monkeypatch.setattr(_mod, "_get_tool_metadata", lambda tool_name: {
            "mutates_external_world": True,
            "risk_level": "critical",
        })
        result = evaluate_permission("send_message", {})
        assert result is not None
        assert result["action"] == "deny"


# ── Layer 5: Default allow ──────────────────────────────────────────────

class TestDefaultAllow:
    def test_memory_tool(self):
        assert evaluate_permission("memory", {"action": "add"}) is None

    def test_random_tool(self):
        assert evaluate_permission("some_random_tool", {}) is None
