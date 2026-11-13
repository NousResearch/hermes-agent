"""Tests for --ignore-user-config and --ignore-rules flags on `hermes chat`.

Ported from openai/codex#18646 (`feat: add --ignore-user-config and --ignore-rules`).
Codex's flags fully isolate a run from user-level config and exec-policy .rules
files. In Hermes the equivalent isolation is:

* ``--ignore-user-config`` → skip ``~/.hermes/config.yaml`` in ``load_cli_config()``
  (credentials in ``.env`` are still loaded).
* ``--ignore-rules`` → skip AGENTS.md / SOUL.md / .cursorrules auto-injection
  and persistent memory (maps to ``AIAgent(skip_context_files=True,
  skip_memory=True)``).

Both flags are wired via env vars so they work cleanly across the
argparse → cmd_chat → cli.main() → HermesCLI → AIAgent call chain.
"""

from __future__ import annotations

import os
import textwrap

import pytest


def _default(*keys: str):
    from hermes_cli.config import DEFAULT_CONFIG, cfg_get
    return cfg_get(DEFAULT_CONFIG, *keys)

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure the two env-var gates start AND end each test in a known state.

    Some tests here write directly to ``os.environ`` (mirroring the real
    ``cmd_chat`` logic), so ``monkeypatch.delenv`` alone isn't enough —
    those writes aren't tracked by monkeypatch and won't be undone by it.
    We add explicit cleanup on yield to prevent cross-test pollution.
    """
    for var in ("HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES"):
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ("HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES"):
        os.environ.pop(var, None)

class TestIgnoreUserConfigEnvGate:
    """``load_cli_config()`` must honour ``HERMES_IGNORE_USER_CONFIG=1``.

    When the env var is set, user config at ``<hermes_home>/config.yaml`` is
    skipped even if present — the function returns only the built-in defaults.
    """

    def _write_user_config(self, tmp_path, model_default):
        config_yaml = textwrap.dedent(
            f"""
            model:
              default: {model_default}
              provider: openrouter
            agent:
              system_prompt: "from user config"
            """
        ).lstrip()
        (tmp_path / "config.yaml").write_text(config_yaml)

    def _reload_cli(self, monkeypatch, tmp_path):
        """Point cli._hermes_home at tmp_path and return a fresh load_cli_config."""
        import cli
        monkeypatch.setattr(cli, "_hermes_home", tmp_path)
        return cli.load_cli_config

    def test_user_config_loaded_when_flag_unset(self, tmp_path, monkeypatch):
        self._write_user_config(tmp_path, "test-vendor/ignore-user-config-sentinel")
        load_cli_config = self._reload_cli(monkeypatch, tmp_path)

        cfg = load_cli_config()

        # User config value wins
        assert cfg["model"]["default"] == "test-vendor/ignore-user-config-sentinel"
        assert cfg["agent"]["system_prompt"] == "from user config"

    def test_user_config_skipped_when_flag_set(self, tmp_path, monkeypatch):
        """With HERMES_IGNORE_USER_CONFIG=1, user config.yaml is ignored.

        The built-in default ``model.default`` is empty string (no user override),
        and the user's ``agent.system_prompt`` is not seen.
        """
        self._write_user_config(tmp_path, "test-vendor/ignore-user-config-sentinel")
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

        load_cli_config = self._reload_cli(monkeypatch, tmp_path)
        cfg = load_cli_config()

        # User-set "system_prompt: from user config" MUST NOT leak through
        assert cfg["agent"].get("system_prompt", "") != "from user config"

        # User-set model.default MUST NOT leak through — either the built-in
        # default ("" or unset) or a project-level fallback, but never the
        # user's value
        assert cfg["model"].get("default", "") != "test-vendor/ignore-user-config-sentinel"

    def test_flag_ignored_when_set_to_other_value(self, tmp_path, monkeypatch):
        """Only the literal value "1" activates the bypass, matching the yolo pattern."""
        self._write_user_config(tmp_path, "test-vendor/ignore-user-config-sentinel")
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "true")  # not "1"

        load_cli_config = self._reload_cli(monkeypatch, tmp_path)
        cfg = load_cli_config()

        # "true" != "1", so user config IS loaded
        assert cfg["model"]["default"] == "test-vendor/ignore-user-config-sentinel"

    @pytest.mark.parametrize("flag", [True, False], ids=["ignore-user-config", "no-user-config"])
    def test_a_project_cli_config_at_the_install_root_is_never_read(self, tmp_path, monkeypatch, flag):
        """``--ignore-user-config`` and a missing user config both mean built-in defaults, as documented:
        a developer's gitignored ``cli-config.yaml`` at the checkout root is not a config source."""
        home, root = tmp_path / "home", tmp_path / "install"
        home.mkdir()
        root.mkdir()
        if flag:
            self._write_user_config(home, "test-vendor/user-sentinel")
            monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
        package = root / "hermes_cli"
        package.mkdir()
        for where in (root, package):  # the install root, and beside the loader where the fallback looked
            (where / "cli-config.yaml").write_text("model:\n  default: test-vendor/project-sentinel\n")
        monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
        monkeypatch.setattr("hermes_cli.cli_config_load.__file__", str(package / "cli_config_load.py"))

        cfg = self._reload_cli(monkeypatch, home)()

        assert cfg["model"].get("default", "") not in ("test-vendor/project-sentinel", "test-vendor/user-sentinel")

class TestIgnoreUserConfigDelegationDefault:
    """``--ignore-user-config`` must not silently shrink the subagent iteration budget: the
    delegation reader must give the same cap as ``DEFAULT_CONFIG``."""

    def test_builtin_delegation_default_matches_the_delegate_tool_default(self):
        """The CLI's built-in default (``hermes_cli/cli_config_load.py``, re-exported by ``cli``) is the
        value a `--ignore-user-config` run actually uses."""
        import cli
        from hermes_agent.tools.delegate_tool import DEFAULT_MAX_ITERATIONS

        assert DEFAULT_MAX_ITERATIONS == _default("delegation", "max_iterations")
        assert cli._cli_config_defaults()["delegation"]["max_iterations"] == DEFAULT_MAX_ITERATIONS

    def test_ignore_user_config_resolves_the_shared_iteration_cap(self, tmp_path):
        """Fresh startup sees the user cap without the flag and the shared cap with it."""
        import json
        from pathlib import Path
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[2]
        (tmp_path / "config.yaml").write_text("delegation:\n  max_iterations: 7\n")
        script = (
            "import json, cli; from hermes_agent.tools import delegate_tool_config; "
            "print(json.dumps([cli.CLI_CONFIG['delegation']['max_iterations'], "
            "delegate_tool_config._load_config()['max_iterations']]))"
        )
        for ignored in (False, True):
            env = {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "HERMES_HOME": str(tmp_path),
                   "PATH": os.environ["PATH"], "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
            for name in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
                if name in os.environ:
                    env[name] = os.environ[name]
            if ignored:
                env["HERMES_IGNORE_USER_CONFIG"] = "1"
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root, env=env,
                                    capture_output=True, text=True, check=True, timeout=60)
            expected = _default("delegation", "max_iterations") if ignored else 7
            assert json.loads(result.stdout.splitlines()[-1]) == [expected, expected]


    def test_example_config_documents_the_same_cap(self):
        """cli-config.yaml.example is the user-facing statement of the same number."""
        from pathlib import Path

        import re

        body = (Path(__file__).resolve().parents[2] / "cli-config.yaml.example").read_text(
            encoding="utf-8")
        documented = re.search(r"max_iterations: (\d+)\s+#.*\(default: (\d+)\)", body)
        assert documented is not None
        default = _default("delegation", "max_iterations")
        assert [int(n) for n in documented.groups()] == [default, default]


class TestSharedDefaults:
    """The classic CLI reads shared defaults without changing ignore-config policy."""

    def _home(self, tmp_path, monkeypatch, text=""):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(text)
        import cli
        monkeypatch.setattr(cli, "_hermes_home", tmp_path)
        return tmp_path

    def test_classic_cli_defaults_are_the_shared_defaults(self, tmp_path, monkeypatch):
        """An empty config.yaml gives ``hermes chat`` the turn cap and subagent cap that load_config
        (gateway, TUI, dashboard) gives: unlimited and 250."""
        from hermes_cli.config import load_config

        self._home(tmp_path, monkeypatch)
        import cli
        cfg, shared = cli.load_cli_config(), load_config()

        assert cfg["agent"]["max_turns"] == shared["agent"]["max_turns"] == _default("agent", "max_turns")
        assert cfg["delegation"]["max_iterations"] == shared["delegation"]["max_iterations"] == (
            _default("delegation", "max_iterations"))

    def test_a_shell_terminal_timeout_survives_a_file_terminal_section(self, tmp_path, monkeypatch):
        """``hermes chat`` mirrors the file's terminal section to env; a key the file does not set
        must not export a built-in default over the user's TERMINAL_TIMEOUT."""
        self._home(tmp_path, monkeypatch, "terminal:\n  backend: local\n")
        monkeypatch.setenv("TERMINAL_TIMEOUT", "600")
        import cli
        cli.load_cli_config()
        assert os.environ["TERMINAL_TIMEOUT"] == "600"



@pytest.mark.parametrize("limit,label", [(19, "19"), (None, "unlimited")])
def test_classic_config_renders_the_resolved_budget(capsys, limit, label):
    from datetime import datetime
    from cli import HermesCLI, _t
    from hermes_cli.config import resolve_turn_limit
    resolved = resolve_turn_limit(limit)
    stand_in = HermesCLI.__new__(HermesCLI)
    stand_in.api_key, stand_in.agent, stand_in.model, stand_in.base_url = None, None, "test/model", "local"
    stand_in.max_turns, stand_in.enabled_toolsets, stand_in.verbose = resolved, [], False
    stand_in.session_start = datetime(2026, 1, 1)
    HermesCLI.show_config(stand_in)
    output = capsys.readouterr().out
    turn_label = _t("cli.config.label_max_turns")
    row = next(line.strip() for line in output.splitlines() if line.strip().startswith(turn_label + " "))
    assert row.removeprefix(turn_label).strip() == label
    if limit is None:
        assert str(resolved) not in output
