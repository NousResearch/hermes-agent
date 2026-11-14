"""Profile-scoped dashboard-auth stays refused, and a later accept drops that sentence."""
from pathlib import Path

import pytest

from hermes_cli.dashboard_auth import clear_providers, list_providers
from hermes_cli.plugins import PluginManager, _reset_plugin_managers_for_tests
from hermes_constants import hermes_home_key

_SOURCE = """\
from hermes_cli.dashboard_auth.base import DashboardAuthProvider

class ToyAuth(DashboardAuthProvider):
    name = "basictoy"
    display_name = "Toy"
    supports_session = True
    interactive_login = False
    def start_login(self, *, redirect_uri):
        raise NotImplementedError
    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        raise NotImplementedError
    def verify_session(self, *, access_token):
        return None
    def refresh_session(self, *, refresh_token):
        raise NotImplementedError
    def revoke_session(self, *, refresh_token):
        return None

def register(ctx):
    ctx.register_dashboard_auth_provider(ToyAuth())
"""


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_providers()
    _reset_plugin_managers_for_tests()
    yield
    clear_providers()
    _reset_plugin_managers_for_tests()


def _write_plugin(home: Path) -> None:
    plugin = home / "plugins" / "basictoy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: basictoy\nversion: '0.0.1'\ndescription: toy\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(_SOURCE, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - basictoy\n  load_timeout_seconds: 0\n",
        encoding="utf-8",
    )


def _isolate(monkeypatch, tmp_path: Path) -> None:
    bundled = tmp_path / "bundled-empty"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)


def _refusal() -> str:
    from hermes_cli.web_server import _no_auth_provider_message

    return _no_auth_provider_message("127.0.0.1")


def test_profile_scope_refusal_names_the_plugin_and_profile(monkeypatch, tmp_path):
    launch = tmp_path / "launch"
    profile = launch / "profiles" / "operator"
    _isolate(monkeypatch, tmp_path)
    _write_plugin(profile)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    PluginManager(str(profile)).discover_and_load(force=True)

    assert [provider.name for provider in list_providers()] == []
    refusal = _refusal()
    assert "basictoy" in refusal
    assert hermes_home_key(profile) in refusal


def test_accepted_registration_drops_the_profile_refusal(monkeypatch, tmp_path):
    launch = tmp_path / "launch"
    profile = launch / "profiles" / "operator"
    _isolate(monkeypatch, tmp_path)
    _write_plugin(profile)
    monkeypatch.setenv("HERMES_HOME", str(launch))

    PluginManager(str(profile)).discover_and_load(force=True)
    assert "Plugin 'basictoy'" in _refusal()

    _write_plugin(launch)
    PluginManager(str(launch)).discover_and_load(force=True)

    assert [provider.name for provider in list_providers()] == ["basictoy"]
    assert "Plugin 'basictoy'" not in _refusal()
