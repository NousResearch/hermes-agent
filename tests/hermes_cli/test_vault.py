"""Behavior contracts for the ``hermes vault`` command."""

from types import SimpleNamespace

import pytest


def _write_login_backend_plugin(home):
    plugin = home / "plugins" / "cold-vault"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: cold-vault\nversion: 1.0.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        """from agent.vault_backends.base import LoginBackend

class ColdVaultBackend(LoginBackend):
    name = "coldvault"
    display_name = "Cold Vault"
    prefix = "cold:"
    def __init__(self, config=None): self.config = config or {}
    def is_available(self): return True
    def list_items(self): return []
    def get_meta(self, handle): return None
    def resolve_password(self, handle): return "secret"

def register(ctx):
    ctx.register_login_backend(ColdVaultBackend)
""",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - cold-vault\n",
        encoding="utf-8",
    )


@pytest.fixture
def cold_login_backend_plugin(tmp_path, monkeypatch):
    from agent.vault_backends.registry import _reset_for_tests
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_login_backend_plugin(home)
    _reset_plugin_managers_for_tests()
    _reset_for_tests()
    yield
    _reset_plugin_managers_for_tests()
    _reset_for_tests()


def test_cold_sources_discovers_login_backend(cold_login_backend_plugin, monkeypatch):
    from hermes_cli import vault

    lines = []
    monkeypatch.setattr(vault, "_console", lambda: SimpleNamespace(print=lines.append))

    vault._cmd_sources(SimpleNamespace(enable=None, disable=None))

    assert any("Cold Vault" in line and "detected" in line for line in lines)