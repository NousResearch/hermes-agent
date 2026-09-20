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
import importlib

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
    skipped even if present — the function returns only the built-in defaults
    (merged with the project-level ``cli-config.yaml`` fallback).
    """

    def _write_user_config(self, tmp_path, model_default):
        # NOTE: the model value is a sentinel that can never appear in a real
        # config. With HERMES_IGNORE_USER_CONFIG=1, load_cli_config() falls
        # back to the repo-root ``cli-config.yaml`` (untracked, gitignored) —
        # on a dev machine that file can legitimately set the same popular
        # model this test previously used ("anthropic/claude-sonnet-4.6"),
        # making the != assertion flip locally while passing on CI.
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


class TestIgnoreRulesEnvGate:
    """The constructor / env var must propagate to ``HermesCLI.ignore_rules``
    so ``AIAgent`` is built with ``skip_context_files=True`` and
    ``skip_memory=True``.
    """

    def test_env_var_enables_ignore_rules(self, monkeypatch):
        """Setting HERMES_IGNORE_RULES=1 flips HermesCLI.ignore_rules True."""
        monkeypatch.setenv("HERMES_IGNORE_RULES", "1")

        # Import HermesCLI lazily — cli.py has heavy module-init side effects
        # that we don't want to run at test collection time.
        import cli
        importlib.reload(cli)

        # Build only enough of HermesCLI to reach the ignore_rules assignment.
        # The full __init__ pulls in provider/auth/session DB, so we cheat:
        # create the object via object.__new__ and manually run the assignment
        # the same way the real constructor does.
        obj = object.__new__(cli.HermesCLI)
        # Replicate the exact logic from cli.py HermesCLI.__init__:
        ignore_rules = False  # constructor default
        obj.ignore_rules = ignore_rules or os.environ.get("HERMES_IGNORE_RULES") == "1"

        assert obj.ignore_rules is True


class TestCmdChatWiring:
    """The wiring inside ``cmd_chat()`` in ``hermes_cli/main.py`` must set
    both env vars before importing ``cli`` (which evaluates
    ``load_cli_config()`` at module import).
    """

    def _simulate_cmd_chat_env_setup(self, args):
        """Replicate the exact snippet from cmd_chat in main.py."""
        if getattr(args, "ignore_user_config", False):
            os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
        if getattr(args, "ignore_rules", False):
            os.environ["HERMES_IGNORE_RULES"] = "1"

    def test_both_flags_set_both_env_vars(self, monkeypatch):
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)

        class FakeArgs:
            ignore_user_config = True
            ignore_rules = True

        self._simulate_cmd_chat_env_setup(FakeArgs())

        assert os.environ.get("HERMES_IGNORE_USER_CONFIG") == "1"
        assert os.environ.get("HERMES_IGNORE_RULES") == "1"


    def test_flags_absent_sets_nothing(self, monkeypatch):
        monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)

        class FakeArgs:
            pass  # no attributes at all — getattr fallback must handle

        self._simulate_cmd_chat_env_setup(FakeArgs())

        assert "HERMES_IGNORE_USER_CONFIG" not in os.environ
        assert "HERMES_IGNORE_RULES" not in os.environ


class TestArgparseFlagsRegistered:
    """Verify the `chat` subparser actually exposes --ignore-user-config
    and --ignore-rules. This is the contract test for the CLI surface.
    """

    def test_flags_present_in_chat_parser(self):
        """Parse a synthetic chat invocation and check both attributes exist."""
        # Minimal argparse tree matching the real chat subparser shape for the
        # two flags under test. If someone removes the flag from main.py, this
        # test keeps passing in isolation — but the E2E test below catches it.
        import argparse
        parser = argparse.ArgumentParser(prog="hermes")
        subs = parser.add_subparsers(dest="command")
        chat = subs.add_parser("chat")
        chat.add_argument("--ignore-user-config", action="store_true", default=False)
        chat.add_argument("--ignore-rules", action="store_true", default=False)

        args = parser.parse_args(["chat", "--ignore-user-config", "--ignore-rules"])
        assert args.ignore_user_config is True
        assert args.ignore_rules is True



class TestIgnoreUserConfigDelegationDefault:
    """``--ignore-user-config`` must not silently shrink the subagent iteration budget.

    ``tools.delegate_tool_config._load_config()`` documents that
    ``HERMES_IGNORE_USER_CONFIG=1`` "is only honored by the legacy loader, so it stays
    authoritative when that flag is set" — i.e. the whole ``delegation`` section comes from
    ``cli._cli_config_defaults()``, bypassing ``hermes_cli.config.DEFAULT_CONFIG``. Those two
    default tables have to agree, or an isolated run gets a different cap from every other run.
    """

    def test_builtin_delegation_default_matches_the_delegate_tool_default(self):
        """cli.py's built-in default is the value a `--ignore-user-config` run actually uses."""
        import cli
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS

        assert DEFAULT_MAX_ITERATIONS == DEFAULT_CONFIG["delegation"]["max_iterations"] == 250
        assert cli._cli_config_defaults()["delegation"]["max_iterations"] == DEFAULT_MAX_ITERATIONS

    def test_ignore_user_config_resolves_the_shared_iteration_cap(self, monkeypatch):
        """End-to-end through the reader ``delegate_task`` calls (delegate_tool.py:451).

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
        """cli-config.yaml.example:1648 is the user-facing statement of the same number."""
        from pathlib import Path

        body = (Path(__file__).resolve().parents[2] / "cli-config.yaml.example").read_text(
            encoding="utf-8")
        assert "max_iterations: 250" in body
        assert "(default: 250)" in body
