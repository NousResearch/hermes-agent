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


@pytest.mark.parametrize(("home_env", "managed_env", "in_use"), [
    ("TERMINAL_ENV=local\n", "TERMINAL_ENV=singularity\n", True),
    ("TERMINAL_ENV=singularity\n", "TERMINAL_ENV=local\n", False),
    ("", "TERMINAL_ENV=singularity\n", True),
])
def test_the_managed_env_beats_the_homes_env_as_in_load_hermes_dotenv(tmp_path, home_env, managed_env, in_use):
    # load_hermes_dotenv applies the administrator-managed .env last, with override.
    (tmp_path / "managed" / ".env").write_text(managed_env, encoding="utf-8")
    assert lcm.singularity_in_use(_home(tmp_path, env=home_env)) is in_use


@pytest.mark.parametrize("env", [
    "HPC_BACKEND=singularity\nTERMINAL_ENV=${HPC_BACKEND}\n",
    "TERMINAL_ENV=${SITE_BACKEND}\n",                    # the process environment supplies the value
    "TERMINAL_ENV=${UNSET_BACKEND:-singularity}\n",
])
def test_dotenv_references_resolve_as_in_load_hermes_dotenv(tmp_path, monkeypatch, env):
    # load_hermes_dotenv expands ${VAR} / ${VAR:-default} against the file's earlier lines and the
    # process environment before TERMINAL_ENV picks the backend.
    monkeypatch.setenv("SITE_BACKEND", "singularity")
    monkeypatch.delenv("UNSET_BACKEND", raising=False)
    assert lcm.singularity_in_use(_home(tmp_path, env=env)) is True


def test_process_env_backend_counts_only_for_the_launch_home(tmp_path, monkeypatch):
    launch = _home(tmp_path)
    secondary = tmp_path / "profiles" / "work"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "singularity")  # the launch process's backend (systemd unit, export)
    assert lcm.singularity_in_use(launch) is False
    assert lcm.singularity_in_use(launch, process_env=True) is True
    # Gateway start runs the startup migration for every served home: a secondary with no backend of its
    # own runs `local`, whatever the launch process exported.
    assert lcm.singularity_in_use(secondary, process_env=True) is False


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


def test_config_unset_removes_the_env_copy_core_used_to_write(tmp_path, monkeypatch):
    # While core shipped the backend, `hermes config set terminal.singularity_image` also wrote
    # TERMINAL_SINGULARITY_IMAGE to .env, and the plugin still falls back to it: unset must remove both,
    # or the old image comes back instead of the default.
    home = _home(tmp_path, env="TERMINAL_SINGULARITY_IMAGE=docker://old:1\n",
                 config="terminal:\n  backend: singularity\n  singularity_image: docker://old:1\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import unset_config_value
    unset_config_value("terminal.singularity_image")
    assert "singularity_image" not in (home / "config.yaml").read_text(encoding="utf-8")
    assert "TERMINAL_SINGULARITY_IMAGE" not in (home / ".env").read_text(encoding="utf-8")


def test_config_set_and_get_still_recognize_the_image_key(tmp_path, monkeypatch, capsys):
    # The plugin reads terminal.singularity_image and its setup text says `hermes config set
    # terminal.singularity_image <image>`; leaving DEFAULT_CONFIG must not make it an unknown key.
    home = _home(tmp_path, config="terminal:\n  backend: singularity\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import get_config_value, set_config_value
    set_config_value("terminal.singularity_image", "docker://python:3.12")
    get_config_value("terminal.singularity_image")
    out = capsys.readouterr()
    assert "docker://python:3.12" in out.out
    assert "not a recognized config key" not in out.out + out.err
