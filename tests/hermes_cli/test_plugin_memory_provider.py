"""Tests for register_memory_provider in the plugin system."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_memory_provider,
)


# ── Helpers ────────────────────────────────────────────────────────────────


class _StubMemoryProvider:
    """Minimal MemoryProvider subclass for testing."""

    def __init__(self, name: str = "stub", available: bool = True):
        self._name = name
        self._available = available

    @property
    def name(self) -> str:  # noqa: D401 – satisfies ABC
        return self._name

    def is_available(self) -> bool:
        return self._available


def _make_context(name: str = "test_plugin") -> PluginContext:
    """Create a PluginContext backed by a fresh PluginManager."""
    manifest = PluginManifest(name=name)
    manager = PluginManager()
    return PluginContext(manifest, manager)


# ── TestRegisterMemoryProvider ─────────────────────────────────────────────


class TestRegisterMemoryProvider:
    """Tests for PluginContext.register_memory_provider()."""

    def test_successful_registration(self):
        """A valid MemoryProvider is stored on the manager."""
        # We patch the isinstance check by making _StubMemoryProvider inherit
        # from MemoryProvider (via lazy import).
        from agent.memory_provider import MemoryProvider

        class RealStub(MemoryProvider):
            @property
            def name(self):
                return "real_stub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

        ctx = _make_context("my_provider")
        provider = RealStub()
        ctx.register_memory_provider(provider)

        assert ctx._manager._memory_provider is provider
        assert ctx._manager._memory_provider_name == "my_provider"

    def test_duplicate_registration_rejected(self, caplog):
        """A second plugin's registration attempt is rejected with a warning."""
        from agent.memory_provider import MemoryProvider

        class RealStub(MemoryProvider):
            @property
            def name(self):
                return "real_stub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

        manager = PluginManager()
        manifest1 = PluginManifest(name="first_plugin")
        manifest2 = PluginManifest(name="second_plugin")
        ctx1 = PluginContext(manifest1, manager)
        ctx2 = PluginContext(manifest2, manager)
        provider = RealStub()

        ctx1.register_memory_provider(provider)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx2.register_memory_provider(provider)

        # First plugin still owns the slot
        assert manager._memory_provider is provider
        assert manager._memory_provider_name == "first_plugin"
        # Warning was emitted
        assert any("already registered" in r.message for r in caplog.records)

    def test_non_memory_provider_rejected(self, caplog):
        """Passing an object that isn't a MemoryProvider is rejected."""
        ctx = _make_context("bad_plugin")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx.register_memory_provider("not_a_provider")

        assert ctx._manager._memory_provider is None
        assert any("does not inherit" in r.message for r in caplog.records)

    def test_non_memory_provider_object_rejected(self, caplog):
        """A plain object (not MemoryProvider subclass) is rejected."""
        ctx = _make_context("bad_plugin2")

        class FakeProvider:
            @property
            def name(self):
                return "fake"

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx.register_memory_provider(FakeProvider())

        assert ctx._manager._memory_provider is None
        assert any("does not inherit" in r.message for r in caplog.records)


# ── TestGetPluginMemoryProvider ────────────────────────────────────────────


class TestGetPluginMemoryProvider:
    """Tests for the module-level get_plugin_memory_provider() helper."""

    def test_returns_none_by_default(self, monkeypatch):
        """Returns None when no provider has been registered."""
        mgr = PluginManager()
        monkeypatch.setattr(
            "hermes_cli.plugins._plugin_manager", mgr
        )
        assert get_plugin_memory_provider() is None

    def test_returns_registered_provider(self, monkeypatch):
        """Returns the provider after registration."""
        from agent.memory_provider import MemoryProvider

        class RealStub(MemoryProvider):
            @property
            def name(self):
                return "real_stub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

        mgr = PluginManager()
        manifest = PluginManifest(name="prov_plugin")
        ctx = PluginContext(manifest, mgr)
        provider = RealStub()
        ctx.register_memory_provider(provider)

        monkeypatch.setattr(
            "hermes_cli.plugins._plugin_manager", mgr
        )
        assert get_plugin_memory_provider() is provider
