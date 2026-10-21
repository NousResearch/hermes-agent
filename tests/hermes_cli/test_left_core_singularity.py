"""The Singularity terminal backend left core for the ``singularity`` catalog plugin: homes whose backend
is ``singularity`` get the plugin once, the name is free for the plugin to register, and a home without
it hears the install command instead of a bare "unknown backend".

Real config/.env files under a temp home; the catalog lookup and the network install are stand-ins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(lcm, "_undelivered", {})
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def _home(tmp_path: Path, *, env: str = "", config: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize(("env", "config", "in_use"), [
    ("", "terminal:\n  backend: singularity\n", True),
    ("", "terminal:\n  env_type: singularity\n", True),                  # legacy key the bridge maps
    ("TERMINAL_ENV=singularity\n", "", True),
    ("TERMINAL_ENV=singularity\n", "terminal:\n  backend: docker\n", False),  # config.yaml wins, as in the bridge
    ("", "terminal:\n  backend: docker\n  singularity_image: docker://x\n", False),  # an image key alone is not use
    ("", "", False),
])
def test_singularity_in_use_is_the_configured_backend(tmp_path, env, config, in_use):
    assert lcm.singularity_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_process_env_backend_counts_only_for_the_active_home(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "singularity")
    assert lcm.singularity_in_use(home) is False
    assert lcm.singularity_in_use(home, process_env=True) is True


def test_a_singularity_home_gets_the_plugin_once(tmp_path):
    home = _home(tmp_path, config="terminal:\n  backend: singularity\n  singularity_image: docker://python:3.12\n")
    calls, said = [], []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert lcm.migrate_home(home, install=install, say=said.append) == ["singularity"]
    assert calls == ["singularity"]
    assert "Singularity/Apptainer terminal backend moved out of core" in said[0]
    assert "terminal.singularity_image" in said[0]
    said.clear()
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["singularity"] and said == []


def test_the_backend_name_is_free_for_the_plugin(monkeypatch):
    from agent import terminal_env_registry as reg
    from agent.terminal_env_provider import TerminalEnvironmentProvider

    class _Stand(TerminalEnvironmentProvider):
        name = "singularity"

        def is_available(self):
            return True

        def create_environment(self, **kwargs):
            return object()

    previous = reg.snapshot_registration("singularity")
    provider = _Stand()
    reg.register_provider(provider)
    try:
        assert reg.get_provider("singularity") is provider
    finally:
        reg.restore_registration("singularity", provider, previous)


def test_without_the_plugin_the_backend_error_names_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path)))
    from tools import terminal_tool_backends as backends
    assert backends._check_plugin_requirements({"env_type": "singularity"}) is False
    reason = backends.terminal_backend_unavailable_reason()
    assert "moved out of core into the 'singularity' plugin" in reason
    assert "hermes plugins install singularity" in reason
    with pytest.raises(ValueError, match="hermes plugins install singularity"):
        backends._create_environment("singularity", image="", cwd="/root", timeout=10)
    assert lcm.terminal_backend_install_hint("acmebox") == ""
