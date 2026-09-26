"""Real plugin discovery, configuration isolation and fail-closed login routing."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from agent import secret_scope
from agent.vault_backends.base import backend_for_handle, enabled_backends, is_installed
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.plugins import (
    _reset_plugin_managers_for_tests,
    discover_plugins,
    get_plugin_manager,
)

_SOURCE = '''
from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta

class Backend(LoginBackend):
    name = "example"
    prefix = "example:"
    display_name = "Example"

    @classmethod
    def is_available(cls, config):
        if config.get("probe_error"):
            raise RuntimeError("PRIVATE-LOADER-DETAIL")
        if config.get("mutation"):
            config["nested"]["value"] = "probe-mutated"
        return config.get("available", True)

    def __init__(self, config):
        if config.get("construct_error"):
            raise RuntimeError("PRIVATE-LOADER-DETAIL")
        self.profile = config["profile"]
        if config.get("mutation"):
            config["nested"]["value"] = "constructor-mutated"

    def list_items(self):
        return [VaultItemMeta(id="example:item", kind="login", label=self.profile,
                             origin="https://example.com", created_at="")]

    def get_meta(self, handle):
        return self.list_items()[0]

    def resolve_password(self, handle):
        return "synthetic-password"

def register(ctx):
    ctx.register_login_backend(Backend)
'''


def _install(home: Path, settings: dict[str, object], *, plugin_enabled: bool = True) -> None:
    plugin = home / "plugins" / "example-login"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: example-login\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_SOURCE, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["example-login"] if plugin_enabled else [],
                    "disabled": [] if plugin_enabled else ["example-login"]},
        "vault": {"onepassword": {"enabled": False}, "bitwarden": {"enabled": False},
                  "example": settings},
    }), encoding="utf-8")


@contextmanager
def _profile(home: Path) -> Iterator[None]:
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def isolated_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from agent.vault_backends.registry import _reset_for_tests

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    old_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    _reset_plugin_managers_for_tests()
    _reset_for_tests()
    try:
        yield tmp_path
    finally:
        _reset_plugin_managers_for_tests()
        _reset_for_tests()
        secret_scope.set_multiplex_active(old_multiplex)


def test_real_discovery_profile_switch_config_copy_reload_and_unload(isolated_profiles: Path) -> None:
    from hermes_cli.config import load_config_readonly
    from tools.browser_vault_tool import browser_vault_list

    import json

    a, b, c = (isolated_profiles / name for name in ("a", "b", "c"))
    for home in (a, b, c):
        _install(home, {"enabled": True, "profile": home.name, "mutation": True,
                        "nested": {"value": "original"}}, plugin_enabled=home != c)
    instances = []
    for home in (a, b, a):
        with _profile(home):
            discover_plugins()
            backend = backend_for_handle("example:item")
            assert backend is not None
            meta = backend.get_meta("example:item")
            assert meta is not None and meta.label == home.name
            assert load_config_readonly()["vault"]["example"]["nested"]["value"] == "original"
            listing = json.loads(browser_vault_list())
            assert listing["items"][0]["label"] == home.name
            assert "synthetic-password" not in json.dumps(listing)
            instances.append(backend)
    assert instances[0] is not instances[2]
    with _profile(c):
        discover_plugins()
        assert backend_for_handle("example:item") is None  # no fallback to launch profile
    with _profile(a):
        discover_plugins(force=True)
        assert backend_for_handle("example:item") is not None
        get_plugin_manager().unload("example-login")
        assert backend_for_handle("example:item") is None
    with _profile(b):
        assert backend_for_handle("example:item") is not None


@pytest.mark.parametrize("settings", [
    {}, {"enabled": False}, {"enabled": "true"}, {"enabled": 1},
    {"enabled": True, "available": False},
    {"enabled": True, "available": "true"},
    {"enabled": True, "probe_error": True},
    {"enabled": True, "construct_error": True},
])
def test_unavailable_or_unapproved_provider_cannot_route_or_leak(
    isolated_profiles: Path, settings: dict[str, object], caplog: pytest.LogCaptureFixture,
) -> None:
    home = isolated_profiles / "a"
    _install(home, {"profile": "a", **settings})
    with _profile(home):
        discover_plugins()
        assert backend_for_handle("example:item") is None
        assert [backend.name for backend in enabled_backends()] == ["local"]
        if settings.get("construct_error"):
            # Status probing never constructs the backend.
            assert is_installed("example")
    assert "PRIVATE-LOADER-DETAIL" not in caplog.text
    assert not any(record.exc_info for record in caplog.records if record.name == "agent.vault_backends.base")
