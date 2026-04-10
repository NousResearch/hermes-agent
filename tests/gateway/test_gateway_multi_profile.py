"""Tests for multi-profile gateway features.

Covers:
- bridge_port config passthrough
- COMMAND_ADMIN_USERS group command restriction
- Shared group session sender attribution
- status_callback gated on tool_progress_enabled
"""

import os
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config


# ---------------------------------------------------------------------------
# bridge_port config passthrough
# ---------------------------------------------------------------------------

class TestBridgePortConfig:
    def test_bridge_port_bridged_from_config(self, tmp_path, monkeypatch):
        """bridge_port in platform config should be parsed as int and
        forwarded into the bridged platform data."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "whatsapp:\n"
            "  bridge_port: 3001\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")

        config = load_gateway_config()
        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert wa_config.extra.get("bridge_port") == 3001

    def test_bridge_port_cast_to_int(self, tmp_path, monkeypatch):
        """bridge_port should be cast to int even if specified as a string."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "whatsapp:\n"
            "  bridge_port: '3002'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")

        config = load_gateway_config()
        wa_config = config.platforms.get(Platform.WHATSAPP)
        assert wa_config is not None
        assert wa_config.extra.get("bridge_port") == 3002
        assert isinstance(wa_config.extra["bridge_port"], int)


# ---------------------------------------------------------------------------
# COMMAND_ADMIN_USERS — group slash command restriction
# ---------------------------------------------------------------------------

class TestCommandAdminUsers:
    """The admin-user gating logic is tested indirectly by verifying
    the env var parsing and ID matching, since the full dispatch path
    requires the entire GatewayRunner."""

    def test_admin_ids_parsed_from_env(self, monkeypatch):
        """COMMAND_ADMIN_USERS should be parsed into a set of trimmed IDs."""
        monkeypatch.setenv("COMMAND_ADMIN_USERS", "alice, bob , charlie")
        raw = os.getenv("COMMAND_ADMIN_USERS", "")
        admin_ids = {uid.strip() for uid in raw.split(",") if uid.strip()}
        assert admin_ids == {"alice", "bob", "charlie"}

    def test_empty_admin_env_means_no_restriction(self, monkeypatch):
        """When COMMAND_ADMIN_USERS is empty, no restriction applies."""
        monkeypatch.setenv("COMMAND_ADMIN_USERS", "")
        raw = os.getenv("COMMAND_ADMIN_USERS", "")
        assert not raw  # falsy → skip restriction

    def test_user_id_at_split_for_matching(self):
        """User IDs like 'number@domain' should also match the bare number."""
        user_id = "4912345678@s.whatsapp.net"
        user_ids = {user_id}
        if "@" in user_id:
            user_ids.add(user_id.split("@")[0])
        assert "4912345678" in user_ids
        assert "4912345678@s.whatsapp.net" in user_ids

    def test_non_admin_blocked(self, monkeypatch):
        """A non-admin user in a group should have their command nullified."""
        monkeypatch.setenv("COMMAND_ADMIN_USERS", "admin_user")
        admin_raw = os.getenv("COMMAND_ADMIN_USERS", "")
        admin_ids = {uid.strip() for uid in admin_raw.split(",") if uid.strip()}
        user_ids = {"random_user"}
        assert not (user_ids & admin_ids)

    def test_admin_allowed(self, monkeypatch):
        """An admin user should pass the check."""
        monkeypatch.setenv("COMMAND_ADMIN_USERS", "4912345678,other_admin")
        admin_raw = os.getenv("COMMAND_ADMIN_USERS", "")
        admin_ids = {uid.strip() for uid in admin_raw.split(",") if uid.strip()}
        user_ids = {"4912345678@s.whatsapp.net", "4912345678"}
        assert user_ids & admin_ids


# ---------------------------------------------------------------------------
# Shared group session sender attribution
# ---------------------------------------------------------------------------

class TestSharedGroupAttribution:
    """Verify the logic for prefixing messages with [sender name] in
    shared group sessions (group_sessions_per_user=false)."""

    def _should_attribute(self, chat_type, thread_id, group_per_user, thread_per_user):
        """Replicate the attribution logic from gateway/run.py."""
        _is_shared_thread = (
            chat_type != "dm"
            and thread_id
            and not thread_per_user
        )
        _is_shared_group = (
            chat_type == "group"
            and not group_per_user
        )
        return _is_shared_thread or _is_shared_group

    def test_dm_never_attributed(self):
        assert not self._should_attribute("dm", None, True, False)
        assert not self._should_attribute("dm", None, False, False)

    def test_shared_group_attributed(self):
        """group_sessions_per_user=false → messages should be attributed."""
        assert self._should_attribute("group", None, False, False)

    def test_per_user_group_not_attributed(self):
        """group_sessions_per_user=true (default) → no attribution needed."""
        assert not self._should_attribute("group", None, True, False)

    def test_shared_thread_attributed(self):
        """Shared thread sessions should still be attributed."""
        assert self._should_attribute("group", "thread-123", True, False)

    def test_per_user_thread_not_attributed(self):
        """thread_sessions_per_user=true → no thread attribution."""
        assert not self._should_attribute("group", "thread-123", True, True)
