import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _CANCELLED, _curses_select








def test_get_installable_providers_filters_catalog_entries_already_installed():
    installed = [("openbrain", "API key / local", object())]

    assert memory_setup._get_installable_providers(installed) == []


def test_cmd_setup_lists_catalogued_provider_when_not_installed(monkeypatch):
    captured = {}
    save_config = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    def select_cancel(title, items, **kwargs):
        captured["items"] = items
        return kwargs["cancel_returns"]

    monkeypatch.setattr(memory_setup, "_curses_select", select_cancel)
    monkeypatch.setattr("hermes_cli.config.load_config", MagicMock())
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    assert captured["items"][0][0] == "openbrain"
    assert "install standalone plugin" in captured["items"][0][1]
    save_config.assert_not_called()


def test_cmd_setup_installs_catalogued_provider_then_runs_post_setup(monkeypatch):
    events = []

    class PostSetupProvider:
        def post_setup(self, hermes_home, config):
            events.append(("post_setup", hermes_home, config))

    provider = PostSetupProvider()
    calls = iter([[], [("openbrain", "API key / local", provider)]])

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: next(calls))
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: 0)
    monkeypatch.setattr(memory_setup, "_clear_interactive_transition", lambda: events.append("clear"))
    monkeypatch.setattr(
        memory_setup,
        "_install_standalone_provider",
        lambda entry: events.append(("install", entry["name"])) or "openbrain",
    )
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: events.append(("deps", name)))
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: "/tmp/hermes-test")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})

    memory_setup.cmd_setup(SimpleNamespace())

    assert events == [
        "clear",
        ("install", "openbrain"),
        "clear",
        ("deps", "openbrain"),
        ("post_setup", "/tmp/hermes-test", {"memory": {}}),
    ]


def test_catalog_install_refreshes_real_discovery_then_runs_post_setup(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    hermes_home = tmp_path / "hermes-home"
    repo = tmp_path / "fixture-memory-provider"
    repo.mkdir()
    (repo / "plugin.yaml").write_text(
        "name: fixture_memory\n"
        "manifest_version: 1\n"
        "description: Filesystem-backed memory fixture\n",
        encoding="utf-8",
    )
    (repo / "__init__.py").write_text(
        "from pathlib import Path\n\n"
        "class FixtureMemoryProvider:\n"
        "    def is_available(self):\n"
        "        return True\n\n"
        "    def get_config_schema(self):\n"
        "        return []\n\n"
        "    def post_setup(self, hermes_home, config):\n"
        "        Path(hermes_home, 'post-setup-ran').write_text('ok')\n\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(FixtureMemoryProvider())\n",
        encoding="utf-8",
    )

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hermes test",
        "GIT_AUTHOR_EMAIL": "hermes-test@example.invalid",
        "GIT_COMMITTER_NAME": "Hermes test",
        "GIT_COMMITTER_EMAIL": "hermes-test@example.invalid",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=git_env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=git_env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repo,
        check=True,
        env=git_env,
    )

    from hermes_cli import memory_provider_catalog

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        memory_provider_catalog,
        "INSTALLABLE_MEMORY_PROVIDERS",
        [{
            "name": "fixture_memory",
            "label": "fixture_memory",
            "setup_hint": "install test plugin",
            "identifier": repo.as_uri(),
            "description": "Filesystem-backed memory fixture",
        }],
    )

    module_prefix = "_hermes_user_memory.fixture_memory"
    try:
        memory_setup.cmd_setup_provider("fixture_memory")

        assert (hermes_home / "plugins" / "fixture_memory" / "__init__.py").is_file()
        assert (hermes_home / "post-setup-ran").read_text() == "ok"
    finally:
        for module_name in list(sys.modules):
            if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
                sys.modules.pop(module_name, None)


def test_cmd_setup_generic_choice_cancel_writes_nothing(tmp_path, monkeypatch):
    class ChoiceProvider:
        def __init__(self):
            self.save_config = MagicMock()

        def get_config_schema(self):
            return [{
                "key": "mode",
                "description": "Mode",
                "default": "one",
                "choices": ["one", "two"],
            }]

    provider = ChoiceProvider()
    selections = iter([0, _CANCELLED])
    save_config = MagicMock()
    install_dependencies = MagicMock()

    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [("fake", "local", provider)])
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr(memory_setup, "_install_dependencies", install_dependencies)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr("hermes_cli.config.save_config", save_config)

    memory_setup.cmd_setup(SimpleNamespace())

    install_dependencies.assert_called_once_with("fake")
    save_config.assert_not_called()
    provider.save_config.assert_not_called()
    assert not (tmp_path / ".env").exists()


def test_write_env_vars_strips_line_separators_and_nul(tmp_path):
    """A pasted secret with embedded CR/LF/NUL must not inject an extra
    KEY=VALUE line into .env (mirrors the openviking plugin's writer)."""
    env_path = tmp_path / ".env"

    memory_setup._write_env_vars(
        env_path,
        {"PROVIDER_API_KEY": "good\nINJECTED_KEY=attacker\r\u2028\x00tail"},
    )

    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["PROVIDER_API_KEY=goodINJECTED_KEY=attackertail"]
    parsed = dict(line.split("=", 1) for line in lines if "=" in line)
    assert set(parsed) == {"PROVIDER_API_KEY"}




# ---------------------------------------------------------------------------
# _provider_pip_dependencies — mode-aware dep expansion (#70636)
# ---------------------------------------------------------------------------





def test_install_dependencies_force_reinstalls_versioned_specs(tmp_path, monkeypatch):
    """force=True hands every declared spec (version ranges intact) to pip,
    so a downgraded/stripped bridge package is restored on hermes update."""
    import yaml as _yaml

    plugin_dir = tmp_path / "mem0"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text(
        _yaml.safe_dump({"pip_dependencies": ["mem0ai>=2.0.10,<3"]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "plugins.memory.find_provider_dir", lambda name: plugin_dir
    )

    installed = []

    def fake_install_specs(specs, timeout=120):
        installed.append(list(specs))
        return SimpleNamespace(ok=True, blocked=False, reason="", stderr="")

    monkeypatch.setattr("tools.lazy_deps.install_specs", fake_install_specs)

    memory_setup._install_dependencies("mem0", force=True)

    assert installed, "force=True must reach the install step"
    assert any("mem0ai>=2.0.10,<3" in specs for specs in installed)
