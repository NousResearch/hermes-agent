import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from hermes_cli import memory_setup


def test_memory_setup_cli_accepts_direct_provider_argument(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    hermes_home = tmp_path / ".hermes"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "memory", "setup", "memory_fragmentation"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Memory provider: memory_fragmentation" in result.stdout
    assert (hermes_home / "memory_fragmentation" / "config.json").exists()


def test_memory_setup_with_provider_name_skips_interactive_picker(monkeypatch):
    direct_calls = []

    def fake_cmd_setup_provider(provider_name: str) -> None:
        direct_calls.append(provider_name)

    def fail_interactive_setup(args) -> None:
        raise AssertionError("interactive setup should not run when provider_name is supplied")

    monkeypatch.setattr(memory_setup, "cmd_setup_provider", fake_cmd_setup_provider)
    monkeypatch.setattr(memory_setup, "cmd_setup", fail_interactive_setup)

    memory_setup.memory_command(
        Namespace(memory_command="setup", provider_name="memory_fragmentation")
    )

    assert direct_calls == ["memory_fragmentation"]


def test_memory_setup_without_provider_name_uses_interactive_picker(monkeypatch):
    interactive_calls = []

    def fail_direct_setup(provider_name: str) -> None:
        raise AssertionError("direct setup should not run without provider_name")

    def fake_cmd_setup(args) -> None:
        interactive_calls.append(args)

    monkeypatch.setattr(memory_setup, "cmd_setup_provider", fail_direct_setup)
    monkeypatch.setattr(memory_setup, "cmd_setup", fake_cmd_setup)
    args = Namespace(memory_command="setup", provider_name=None)

    memory_setup.memory_command(args)

    assert interactive_calls == [args]


def test_interactive_setup_builtin_only_saves_empty_provider(monkeypatch):
    saved_configs = []

    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("fake_schema", "local", object())],
    )
    monkeypatch.setattr(
        memory_setup,
        "_curses_select",
        lambda title, items, default=0: len(items) - 1,
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {"provider": "old"}})
    monkeypatch.setattr(
        "hermes_cli.config.save_config",
        lambda config: saved_configs.append({"memory": dict(config.get("memory", {}))}),
    )

    memory_setup.cmd_setup(Namespace())

    assert saved_configs == [{"memory": {"provider": ""}}]


def test_direct_provider_setup_runs_generic_schema_flow(tmp_path, monkeypatch):
    class FakeSchemaProvider:
        def get_config_schema(self):
            return [
                {
                    "key": "endpoint",
                    "description": "Endpoint URL",
                    "default": "http://localhost:1234",
                },
                {
                    "key": "api_key",
                    "description": "API key",
                    "secret": True,
                    "env_var": "FAKE_MEMORY_API_KEY",
                },
            ]

        def save_config(self, values, hermes_home):
            saved_provider_configs.append((dict(values), hermes_home))

    saved_configs = []
    saved_provider_configs = []
    responses = iter(["https://memory.example.test", "secret-value"])

    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("fake_schema", "local", FakeSchemaProvider())],
    )
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda provider_name: None)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"memory": {}})
    monkeypatch.setattr(
        "hermes_cli.config.save_config",
        lambda config: saved_configs.append({"memory": dict(config.get("memory", {}))}),
    )
    monkeypatch.setattr(
        memory_setup,
        "_prompt",
        lambda label, default=None, secret=False: next(responses),
    )

    memory_setup.cmd_setup_provider("fake_schema")

    assert saved_configs[-1]["memory"]["provider"] == "fake_schema"
    assert saved_provider_configs == [
        ({"endpoint": "https://memory.example.test"}, str(tmp_path))
    ]
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "FAKE_MEMORY_API_KEY=secret-value\n"
