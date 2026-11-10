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
    """``--ignore-user-config`` must not silently shrink the subagent iteration budget.

    ``tools.delegate_tool_config._load_config()`` documents that
    ``HERMES_IGNORE_USER_CONFIG=1`` "is only honored by the legacy loader, so it stays
    authoritative when that flag is set" — i.e. the whole ``delegation`` section comes from
    ``cli._cli_config_defaults()``, bypassing ``hermes_cli.config.DEFAULT_CONFIG``. Those two
    default tables have to agree, or an isolated run gets a different cap from every other run.
    """

    def test_builtin_delegation_default_matches_the_delegate_tool_default(self):
        """The CLI's built-in default (``hermes_cli/cli_config_load.py``, re-exported by ``cli``) is the
        value a `--ignore-user-config` run actually uses."""
        import cli
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS

        assert DEFAULT_MAX_ITERATIONS == DEFAULT_CONFIG["delegation"]["max_iterations"] == 250
        assert cli._cli_config_defaults()["delegation"]["max_iterations"] == DEFAULT_MAX_ITERATIONS

    def test_ignore_user_config_resolves_the_shared_iteration_cap(self, monkeypatch):
        """End-to-end through the reader ``delegate_task`` calls for its iteration cap.

        ``CLI_CONFIG`` is pinned to the built-in defaults so the assertion cannot be flipped by a
        developer's untracked repo-root ``cli-config.yaml`` (the file ``load_cli_config`` falls back
        to when the flag is set — see the NOTE on ``_write_user_config`` above).
        """
        import cli
        from tools import delegate_tool_config
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS

        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
        monkeypatch.setattr(cli, "CLI_CONFIG", cli._cli_config_defaults())

        cfg = delegate_tool_config._load_config()

        assert cfg.get("max_iterations", DEFAULT_MAX_ITERATIONS) == DEFAULT_MAX_ITERATIONS, (
            "an --ignore-user-config run resolved a different subagent iteration cap than "
            "delegation.max_iterations; migration 36 raised this to 250 because a smaller cap "
            "truncated substantial delegated work"
        )

    def test_example_config_documents_the_same_cap(self):
        """cli-config.yaml.example is the user-facing statement of the same number."""
        from pathlib import Path

        body = (Path(__file__).resolve().parents[2] / "cli-config.yaml.example").read_text(
            encoding="utf-8")
        assert "max_iterations: 250" in body
        assert "(default: 250)" in body


class TestOneDefaultTableAndOneBypass:
    """The classic CLI reads the shared ``DEFAULT_CONFIG`` for the keys both tables carry, and
    ``--ignore-user-config`` means the same thing to every layered reader: config.yaml is absent."""

    def _home(self, tmp_path, monkeypatch, text=""):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(text)
        import cli
        monkeypatch.setattr(cli, "_hermes_home", tmp_path)
        return tmp_path

    def test_classic_cli_defaults_are_the_shared_defaults(self, tmp_path, monkeypatch):
        """An empty config.yaml gives ``hermes chat`` the turn cap, subagent cap and terminal timeout
        that load_config (gateway, TUI, dashboard) gives: unlimited, 250, 180."""
        from hermes_cli.config import load_config
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        self._home(tmp_path, monkeypatch)
        import cli
        cfg, shared = cli.load_cli_config(), load_config()

        assert cfg["agent"]["max_turns"] == shared["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
        assert cfg["delegation"]["max_iterations"] == shared["delegation"]["max_iterations"] == 250
        assert cfg["terminal"]["timeout"] == shared["terminal"]["timeout"] == DEFAULT_CONFIG["terminal"]["timeout"]

    def test_every_layered_reader_drops_the_user_file(self, tmp_path, monkeypatch):
        """With the flag set no layered read returns a value only config.yaml sets; the raw readers
        (the write-back primitive) still see the file."""
        import gateway.run as gateway_run
        from hermes_cli import config as C
        from hermes_cli.config_effective import load_user_config_effective
        from tools import delegate_tool_config

        home = self._home(tmp_path, monkeypatch, (
            "model:\n  default: test-vendor/user-only\n  base_url: https://user-only.example\n"
            "delegation:\n  max_iterations: 7\nterminal:\n  timeout: 9\n"))
        assert C.load_config()["model"]["base_url"] == "https://user-only.example"  # read before the flag
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

        layered = {
            "load_config": C.load_config(), "load_config_readonly": C.load_config_readonly(),
            "load_user_config_effective": load_user_config_effective(),
            "_load_gateway_config": gateway_run._load_gateway_config(home / "config.yaml"),
        }
        for name, cfg in layered.items():
            model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
            assert model.get("base_url") != "https://user-only.example", name
            assert (cfg.get("terminal") or {}).get("timeout") != 9, name
        assert delegate_tool_config._load_config().get("max_iterations") == 250
        # No file terminal section exists, so defaults only backfill: a .env/shell value is kept.
        env = {"TERMINAL_TIMEOUT": "55"}
        C.apply_terminal_config_to_env(env=env)
        assert env["TERMINAL_TIMEOUT"] == "55"
        assert C.read_raw_config()["model"]["base_url"] == "https://user-only.example"

    def test_a_save_under_the_flag_keeps_the_user_file(self, tmp_path, monkeypatch):
        """load_config() served defaults, so a read-modify-write must land only its own edit: the
        user's sections stay on disk instead of being replaced by the defaults."""
        from hermes_cli import config as C
        from hermes_cli.config_backend import read_config_doc

        home = self._home(tmp_path, monkeypatch, (
            "model:\n  default: test-vendor/user-only\napprovals:\n  deny:\n  - rm -rf /\n"
            "custom_providers:\n- name: mine\n  base_url: https://mine.example\n"))
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

        cfg = C.load_config()
        cfg.setdefault("display", {})["skin"] = "mono"
        C.save_config(cfg)

        on_disk = read_config_doc(home / "config.yaml")
        assert on_disk["display"]["skin"] == "mono"
        assert on_disk["model"]["default"] == "test-vendor/user-only"
        assert on_disk["approvals"]["deny"] == ["rm -rf /"]
        assert on_disk["custom_providers"][0]["name"] == "mine"
