"""Behavior contract for third-party browser-login backends."""

import pytest

from agent.vault_backends.base import (
    LoginBackend,
    backend_for_handle,
    enabled_backends,
    external_backend_classes,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class PluginBackend(LoginBackend):
    name = "fixturepass"
    display_name = "Fixture Pass"
    prefix = "fp:"

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    @classmethod
    def is_available(cls, config):
        return config.get("token") == "ready"

    def list_items(self):
        return []

    def get_meta(self, handle):
        return None

    def resolve_password(self, handle):
        return self.cfg["password"]


def test_plugin_backend_receives_config_routes_handles_and_unloads(monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="fixture-pass", key="fixture-pass"), manager)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"vault": {"fixturepass": {"enabled": True, "token": "ready", "password": "secret"}}},
    )

    assert context.register_login_backend(PluginBackend) is not None
    try:
        assert external_backend_classes()[-1] is PluginBackend
        enabled_names = [backend.name for backend in enabled_backends()]
        assert enabled_names[0] == "local"
        assert enabled_names[-1] == "fixturepass"
        assert backend_for_handle("fp:item").resolve_password("fp:item") == "secret"
    finally:
        manager.unload("fixture-pass")

    assert PluginBackend not in external_backend_classes()


def test_plugin_backend_cannot_shadow_builtin_name_or_handle_prefix():
    class NamedLikeLocal(PluginBackend):
        name = "local"
        prefix = "other:"

    class OverlapsBitwarden(PluginBackend):
        name = "other"
        prefix = "bw:child:"

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="collision", key="collision"), manager)
    try:
        assert context.register_login_backend(NamedLikeLocal) is None
        assert context.register_login_backend(OverlapsBitwarden) is None
    finally:
        manager.unload("collision")


def test_plugin_backend_rejects_malformed_names_without_aborting_registration():
    class MissingName(LoginBackend):
        display_name = "Missing Name"
        prefix = "missing:"

        def list_items(self):
            return []

        def get_meta(self, handle):
            return None

        def resolve_password(self, handle):
            return "secret"

    class NonStringName(PluginBackend):
        name = 7

    class UnhashableName(PluginBackend):
        name = []

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="malformed", key="malformed"), manager)
    try:
        assert context.register_login_backend(MissingName) is None
        assert context.register_login_backend(NonStringName) is None
        assert context.register_login_backend(UnhashableName) is None
    finally:
        manager.unload("malformed")


def test_plugin_backend_availability_wins_over_existing_binary_path(monkeypatch, tmp_path):
    class UnavailableBackend(PluginBackend):
        name = "unavailable"
        prefix = "unavailable:"

        @classmethod
        def is_available(cls, config):
            return False

    binary = tmp_path / "manager-cli"
    binary.write_text("present", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"vault": {"unavailable": {"binary_path": str(binary)}}},
    )
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="unavailable", key="unavailable"), manager)
    assert context.register_login_backend(UnavailableBackend) is not None
    try:
        from agent.vault_backends.base import is_installed

        assert is_installed("unavailable") is False
        assert "unavailable" not in {backend.name for backend in enabled_backends()}
    finally:
        manager.unload("unavailable")


@pytest.mark.parametrize(("first_prefix", "second_name", "second_prefix"), [
    ("fixture:", "second", "fixture:child:"),
    ("fixture:child:", "second", "fixture:"),
    ("fixture:", "second", "fixture:"),
    ("fixture:", "fixturepass", "other:"),
    ("fixture:", "second", "op:"),
    ("fixture:", "second", "op:child:"),
    ("fixture:", "second", "op"),
    ("fixture:", "second", "vault_"),
    ("fixture:", "second", "vault_child:"),
])
def test_collision_never_creates_ambiguous_routing(first_prefix, second_name, second_prefix):
    from agent.vault_backends.registry import list_backend_classes

    first = type("First", (PluginBackend,), {"prefix": first_prefix})
    second = type("Second", (PluginBackend,), {"name": second_name, "prefix": second_prefix})
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="collision", key="collision"), manager)
    try:
        assert context.register_login_backend(first) is not None
        assert context.register_login_backend(second) is None
        assert list_backend_classes() == (first,)
    finally:
        manager.unload("collision")
