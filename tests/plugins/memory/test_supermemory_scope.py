"""Tests for Supermemory provider scope support."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _sanitize_tag,
    _save_supermemory_config,
)


@pytest.fixture
def provider():
    return SupermemoryMemoryProvider()


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key-1234567890")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    _save_supermemory_config({"container_tag": "hermes_{identity}"}, str(tmp_path))
    return tmp_path


class TestSupermemoryScopeInitialization:
    def test_no_scope_suffix_uses_identity_tag(self, provider, temp_config):
        provider.initialize("test-session", hermes_home=str(temp_config), agent_identity="default")
        assert provider._container_tag == "hermes_default"

    def test_scope_suffix_appended_to_container(self, provider, temp_config):
        provider.initialize(
            "test-session", hermes_home=str(temp_config), agent_identity="default",
            memory_scope_key="a4c981e7f2b3d5c9",
        )
        assert provider._container_tag == "hermes_default_a4c981e7f2b3d5c9"

    def test_scope_suffix_sanitized(self, provider, temp_config):
        provider.initialize(
            "test-session", hermes_home=str(temp_config), agent_identity="default",
            memory_scope_key="!@#$%^&*()",
        )
        tag = provider._container_tag
        assert "!" not in tag
        assert "@" not in tag

    def test_scoped_primary_in_allowed_containers(self, provider, temp_config):
        provider.initialize(
            "test-session", hermes_home=str(temp_config), agent_identity="default",
            memory_scope_key="abc123",
        )
        assert provider._container_tag in provider._allowed_containers

    def test_different_scopes_produce_different_tags(self, provider, temp_config):
        p1 = SupermemoryMemoryProvider()
        p1.initialize("session-a", hermes_home=str(temp_config), agent_identity="default",
                      memory_scope_key="aaa111")
        p2 = SupermemoryMemoryProvider()
        p2.initialize("session-b", hermes_home=str(temp_config), agent_identity="default",
                      memory_scope_key="bbb222")
        assert p1._container_tag != p2._container_tag

    def test_session_switch_rebinds_scoped_container(self, provider, temp_config):
        provider.initialize("a", hermes_home=str(temp_config), agent_identity="default", memory_scope_key="aaa")
        provider._session_turns = []
        provider.on_session_switch("b", memory_scope_key="bbb")
        assert provider._container_tag == "hermes_default_bbb"
        assert provider._client._container_tag == "hermes_default_bbb"


class TestSupermemoryScopeBackwardsCompat:
    def test_identity_scope_unchanged(self, provider, temp_config):
        provider.initialize(
            "test-session", hermes_home=str(temp_config), agent_identity="myprofile",
        )
        assert provider._container_tag == "hermes_myprofile"
        assert provider._allowed_containers[0] == "hermes_myprofile"


class TestSupermemoryScopeMultiContainer:
    def test_custom_containers_preserved_with_scope(self, provider, temp_config):
        _save_supermemory_config(
            {
                "container_tag": "hermes_{identity}",
                "enable_custom_container_tags": True,
                "custom_containers": ["project-alpha", "project-beta"],
            },
            str(temp_config),
        )
        provider.initialize(
            "test-session", hermes_home=str(temp_config), agent_identity="default",
            memory_scope_key="conv123",
        )
        assert provider._container_tag == "hermes_default_conv123"
        assert "project_alpha" in provider._allowed_containers
        assert "project_beta" in provider._allowed_containers
        assert provider._container_tag in provider._allowed_containers
