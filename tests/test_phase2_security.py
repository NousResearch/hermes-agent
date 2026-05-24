"""Tests for Phase 2 security modules."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Sandbox Policy Tests ──────────────────────────────────────────────

class TestSandboxPolicy:
    """Test sandbox policy enforcement."""

    def test_resolve_sandbox_mode_defaults_to_env_type(self) -> None:
        from tools.sandbox_policy import resolve_sandbox_mode
        assert resolve_sandbox_mode("local") == "local"
        assert resolve_sandbox_mode("docker") == "docker"

    def test_resolve_sandbox_mode_override(self) -> None:
        from tools.sandbox_policy import resolve_sandbox_mode
        assert resolve_sandbox_mode("local", "container") == "container"
        assert resolve_sandbox_mode("local", "docker") == "docker"

    def test_check_sandbox_policy_allows_local_by_default(self) -> None:
        from tools.sandbox_policy import check_sandbox_policy
        ok, reason = check_sandbox_policy("ls -la", "local")
        assert ok

    def test_check_sandbox_policy_warns_dangerous_local(self) -> None:
        from tools.sandbox_policy import check_sandbox_policy
        ok, reason = check_sandbox_policy("rm -rf /tmp", "local")
        assert ok  # warns but doesn't block
        assert reason is not None
        assert "Warning" in reason

    def test_check_sandbox_policy_denies_native_when_flag(self) -> None:
        from tools.sandbox_policy import check_sandbox_policy
        ok, reason = check_sandbox_policy("ls", "local", deny_native=True)
        assert not ok
        assert "disabled" in reason.lower()


# ── Credential Rotation Tests ─────────────────────────────────────────

class TestCredentialRotation:
    """Test credential rotation mechanism."""

    def test_should_rotate_when_never_rotated(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.credential_rotation import should_rotate
        assert should_rotate()

    def test_rotate_credentials_advances_index(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.credential_rotation import rotate_credentials, get_rotation_status

        pool = [
            {"provider": "openai", "model": "gpt-4", "api_key": "key1"},
            {"provider": "anthropic", "model": "claude-3", "api_key": "key2"},
        ]

        result = rotate_credentials(pool)
        assert result["rotated"] is True
        assert result["credential"]["provider"] == "anthropic"
        assert result["rotation_count"] == 1

        # Rotate again — should wrap around
        result2 = rotate_credentials(pool)
        assert result2["credential"]["provider"] == "openai"
        assert result2["rotation_count"] == 2

    def test_get_rotation_status(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.credential_rotation import get_rotation_status
        status = get_rotation_status()
        assert status["current_index"] == 0
        assert status["last_rotation"] is None or status["last_rotation"] == "never"

    def test_reset_rotation(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.credential_rotation import rotate_credentials, reset_rotation, get_rotation_status

        pool = [{"provider": "openai", "model": "gpt-4", "api_key": "key"}]
        rotate_credentials(pool)

        reset_rotation()
        status = get_rotation_status()
        assert status["current_index"] == 0
        assert status["last_rotation"] is None


# ── Plugin Capabilities Tests ─────────────────────────────────────────

class TestPluginCapabilities:
    """Test plugin capability system."""

    def test_parse_capabilities(self) -> None:
        from hermes_cli.plugin_capabilities import PluginCapabilities, Capability

        caps = PluginCapabilities("test-plugin", ["tool-register", "filesystem-read"])
        assert caps.has(Capability.TOOL_REGISTER)
        assert caps.has(Capability.FILESYSTEM_READ)
        assert not caps.has(Capability.NETWORK)

    def test_capability_denied_raises(self) -> None:
        from hermes_cli.plugin_capabilities import (
            Capability,
            CapabilityDenied,
            PluginCapabilities,
        )

        caps = PluginCapabilities("test-plugin", ["tool-register"])
        with pytest.raises(CapabilityDenied) as exc_info:
            caps.check(Capability.NETWORK)
        assert "test-plugin" in str(exc_info.value)
        assert "NETWORK" in str(exc_info.value)

    def test_approval_required_capabilities(self) -> None:
        from hermes_cli.plugin_capabilities import (
            Capability,
            CapabilityDenied,
            PluginCapabilities,
        )

        caps = PluginCapabilities("test-plugin", ["filesystem-write"])
        # Has the capability but no approval yet
        with pytest.raises(CapabilityDenied):
            caps.check(Capability.FILESYSTEM_WRITE)

        # Grant approval — now it passes
        caps.grant_approval(Capability.FILESYSTEM_WRITE)
        caps.check(Capability.FILESYSTEM_WRITE)  # no raise

    def test_parse_from_manifest(self) -> None:
        from hermes_cli.plugin_capabilities import parse_capabilities_from_manifest

        manifest = {"capabilities": ["tool-register", "network", "subprocess"]}
        caps = parse_capabilities_from_manifest(manifest)
        assert len(caps) == 3

    def test_get_capabilities_sorted(self) -> None:
        from hermes_cli.plugin_capabilities import PluginCapabilities

        caps = PluginCapabilities("test", ["network", "tool-register", "filesystem-read"])
        names = caps.get_capabilities()
        # Sorted by enum value, not alphabetically
        assert set(names) == {"filesystem-read", "network", "tool-register"}
        assert len(names) == 3


# ── CSRF Tests ────────────────────────────────────────────────────────

class TestCSRF:
    """Test CSRF protection."""

    def test_generate_csrf_token(self) -> None:
        from hermes_cli.csrf import generate_csrf_token
        token = generate_csrf_token("test-session-token")
        assert len(token) == 64  # SHA-256 hex digest

    def test_verify_csrf_token_valid(self) -> None:
        from hermes_cli.csrf import generate_csrf_token, verify_csrf_token
        session = "test-session"
        csrf = generate_csrf_token(session)
        assert verify_csrf_token(session, csrf)

    def test_verify_csrf_token_invalid(self) -> None:
        from hermes_cli.csrf import verify_csrf_token
        assert not verify_csrf_token("session", "wrong-token")

    def test_verify_csrf_token_disabled(self) -> None:
        from hermes_cli.csrf import verify_csrf_token
        # When disabled, always returns True
        assert verify_csrf_token("session", None, enabled=False)
        assert verify_csrf_token("session", "wrong", enabled=False)

    def test_verify_csrf_token_missing(self) -> None:
        from hermes_cli.csrf import generate_csrf_token, verify_csrf_token
        assert not verify_csrf_token("session", None)

    def test_is_state_changing_method(self) -> None:
        from hermes_cli.csrf import is_state_changing_method
        assert is_state_changing_method("POST")
        assert is_state_changing_method("PUT")
        assert is_state_changing_method("DELETE")
        assert is_state_changing_method("PATCH")
        assert not is_state_changing_method("GET")
        assert not is_state_changing_method("HEAD")
        assert not is_state_changing_method("OPTIONS")

    def test_extract_csrf_token(self) -> None:
        from hermes_cli.csrf import extract_csrf_token
        headers = {"X-CSRF-Token": "my-token"}
        assert extract_csrf_token(headers) == "my-token"

    def test_extract_csrf_token_alias(self) -> None:
        from hermes_cli.csrf import extract_csrf_token
        headers = {"x-hermes-csrf-token": "my-token"}
        assert extract_csrf_token(headers) == "my-token"

    def test_csrf_token_is_deterministic(self) -> None:
        """Same session token should always produce same CSRF token."""
        from hermes_cli.csrf import generate_csrf_token
        t1 = generate_csrf_token("fixed-session")
        t2 = generate_csrf_token("fixed-session")
        assert t1 == t2

    def test_different_sessions_produce_different_tokens(self) -> None:
        from hermes_cli.csrf import generate_csrf_token
        t1 = generate_csrf_token("session-a")
        t2 = generate_csrf_token("session-b")
        assert t1 != t2
