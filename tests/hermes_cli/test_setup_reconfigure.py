"""Tests for the setup wizard's returning-user behavior.

On an existing install:
- Bare `hermes setup` drops straight into the full reconfigure wizard
  (every prompt shows the current value as its default).
- `hermes setup --quick` runs the narrower "fill in missing items" flow.
- `hermes setup --reconfigure` is a backwards-compat alias for the
  bare-setup default.

On a fresh install, all three are no-ops — fall through to first-time setup.
"""

from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import patch

import pytest


def _make_setup_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", False),
        section=overrides.get("section", None),
        reset=overrides.get("reset", False),
        reconfigure=overrides.get("reconfigure", False),
        quick=overrides.get("quick", False),
    )


@pytest.fixture
def existing_install(tmp_path, monkeypatch):
    """Simulate a returning user with an existing configured install."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def fresh_install(tmp_path, monkeypatch):
    """Simulate a first-time user with no existing configuration."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _enter_existing_install_patches(stack, **extra):
    """Apply standard existing-install mocks via an ExitStack.

    Returns a dict of mocks from the `extra` kwargs (which map mock-name to
    target path) so callers can assert on them.
    """
    # Unconditional mocks (no return values to assert against).
    for target, kwargs in [
        ("hermes_cli.setup.ensure_hermes_home", {}),
        ("hermes_cli.setup.is_interactive_stdin", {"return_value": True}),
        ("hermes_cli.config.is_managed", {"return_value": False}),
        ("hermes_cli.setup.load_config", {"return_value": {}}),
        ("hermes_cli.setup.save_config", {}),
        ("hermes_cli.setup.get_env_value", {"return_value": None}),
        ("hermes_cli.auth.get_active_provider", {"return_value": "openrouter"}),
        ("hermes_cli.setup._print_setup_summary", {}),
        ("hermes_cli.setup._offer_openclaw_migration", {"return_value": False}),
    ]:
        stack.enter_context(patch(target, **kwargs))

    # Named mocks caller wants to assert on.
    named = {}
    for name, target in extra.items():
        named[name] = stack.enter_context(patch(target))
    return named


def _enter_fresh_install_patches(stack, **extra):
    for target, kwargs in [
        ("hermes_cli.setup.ensure_hermes_home", {}),
        ("hermes_cli.setup.is_interactive_stdin", {"return_value": True}),
        ("hermes_cli.config.is_managed", {"return_value": False}),
        ("hermes_cli.setup.load_config", {"return_value": {}}),
        ("hermes_cli.setup.save_config", {}),
        ("hermes_cli.auth.get_active_provider", {"return_value": None}),
        ("hermes_cli.setup.get_env_value", {"return_value": None}),
        ("hermes_cli.main._has_any_provider_configured", {"return_value": False}),
        ("hermes_cli.setup._offer_openclaw_migration", {"return_value": False}),
    ]:
        stack.enter_context(patch(target, **kwargs))

    named = {}
    for name, target_spec in extra.items():
        if isinstance(target_spec, tuple):
            target, kwargs = target_spec
            named[name] = stack.enter_context(patch(target, **kwargs))
        else:
            named[name] = stack.enter_context(patch(target_spec))
    return named


class TestExistingInstallDefault:
    """Bare `hermes setup` on an existing install = full reconfigure wizard."""

    def test_bare_setup_runs_full_reconfigure_without_menu(self, existing_install):
        """No menu, no prompt_choice — just run every section in sequence."""
        args = _make_setup_args()  # no flags

        with ExitStack() as stack:
            m = _enter_existing_install_patches(
                stack,
                prompt_choice="hermes_cli.setup.prompt_choice",
                quick="hermes_cli.setup._run_quick_setup",
                model="hermes_cli.setup.setup_model_provider",
                terminal="hermes_cli.setup.setup_terminal_backend",
                agent="hermes_cli.setup.setup_agent_settings",
                gateway="hermes_cli.setup.setup_gateway",
                tools="hermes_cli.setup.setup_tools",
            )
            from hermes_cli.setup import run_setup_wizard
            run_setup_wizard(args)

        # No menu shown.
        m["prompt_choice"].assert_not_called()
        # Quick-setup path NOT taken.
        m["quick"].assert_not_called()
        # Model/terminal/gateway/tools run; agent settings are no longer
        # prompted on existing installs (they keep their tuned values).
        m["model"].assert_called_once()
        m["terminal"].assert_called_once()
        m["agent"].assert_not_called()
        m["gateway"].assert_called_once()
        m["tools"].assert_called_once()


class TestExistingInstallApiKeyProviders:
    """``_run_setup_wizard_impl`` recognises API-key providers via ``_has_any_provider_configured``
    so installs configured with e.g. ANTHROPIC_API_KEY / Z.AI / MiniMax / Kimi are no longer
    misclassified as first-time setups (#13024)."""

    def test_anthropic_api_key_treated_as_existing_install(self, existing_install):
        """The previous narrow check (``OPENROUTER_API_KEY`` / ``OPENAI_BASE_URL`` / OAuth) missed
        API-key providers like Anthropic / Z.AI / MiniMax / Kimi. After the fix the wizard defers
        to ``_has_any_provider_configured()`` which scans ``PROVIDER_REGISTRY.api_key_env_vars``."""
        args = _make_setup_args()

        with ExitStack() as stack:
            # ``get_active_provider`` is None (no OAuth) and the two legacy env keys are
            # unset — the OLD check would have routed this to the first-time-setup path.
            m = _enter_existing_install_patches(
                stack,
                provider="hermes_cli.auth.get_active_provider",
                env_value="hermes_cli.setup.get_env_value",
                has_provider="hermes_cli.main._has_any_provider_configured",
                prompt_choice="hermes_cli.setup.prompt_choice",
                quick="hermes_cli.setup._run_quick_setup",
                model="hermes_cli.setup.setup_model_provider",
                terminal="hermes_cli.setup.setup_terminal_backend",
                gateway="hermes_cli.setup.setup_gateway",
                tools="hermes_cli.setup.setup_tools",
            )
            from hermes_cli.setup import run_setup_wizard
            run_setup_wizard(args)

        # The new check saw a provider configured (e.g. ANTHROPIC_API_KEY).
        m["has_provider"].assert_called_once()
        # Wizard took the existing-install path: full reconfigure sections run, no first-time menu.
        m["prompt_choice"].assert_not_called()
        m["model"].assert_called_once()
        m["terminal"].assert_called_once()
        m["gateway"].assert_called_once()
        m["tools"].assert_called_once()
        m["quick"].assert_not_called()


    def test_no_provider_anywhere_treated_as_fresh_install(self, fresh_install):
        """Counter-test: with no provider configured anywhere, the wizard must still fall
        through to the first-time-setup path. This pins that the helper delegation hasn't
        flipped the polarity (#13024)."""
        args = _make_setup_args()

        with ExitStack() as stack:
            # The standard fresh-install patches + explicit False on the new check so
            # ``is_existing = bool(_has_any_provider_configured())`` evaluates to False.
            m = _enter_fresh_install_patches(
                stack,
                provider="hermes_cli.auth.get_active_provider",
                env_value="hermes_cli.setup.get_env_value",
                has_provider=("hermes_cli.main._has_any_provider_configured", {"return_value": False}),
                prompt=("hermes_cli.setup.prompt_choice", {"return_value": 0}),
                first="hermes_cli.setup_quick._run_first_time_quick_setup",
            )
            from hermes_cli.setup import run_setup_wizard
            run_setup_wizard(args)

        # The new check returned False; wizard fell through to first-time-setup.
        m["has_provider"].assert_called_once()
        m["first"].assert_called_once()
        m["prompt"].assert_called_once()


class TestQuickFlag:
    """`--quick` on an existing install runs the fill-missing flow."""

    def test_quick_flag_runs_quick_setup_only(self, existing_install):
        args = _make_setup_args(quick=True)

        with ExitStack() as stack:
            m = _enter_existing_install_patches(
                stack,
                quick="hermes_cli.setup._run_quick_setup",
                model="hermes_cli.setup.setup_model_provider",
                terminal="hermes_cli.setup.setup_terminal_backend",
                agent="hermes_cli.setup.setup_agent_settings",
                gateway="hermes_cli.setup.setup_gateway",
                tools="hermes_cli.setup.setup_tools",
            )
            from hermes_cli.setup import run_setup_wizard
            from hermes_cli import setup as setup_mod

            section_indexes = []
            m["quick"].side_effect = lambda *_args: section_indexes.append(
                setup_mod._SETUP_NAVIGATION.get().section_index
            )
            run_setup_wizard(args)

        m["quick"].assert_called_once()
        assert section_indexes == [0]
        # Full reconfigure sections must NOT run.
        m["model"].assert_not_called()
        m["terminal"].assert_not_called()
        m["agent"].assert_not_called()
        m["gateway"].assert_not_called()
        m["tools"].assert_not_called()


class TestFreshInstall:
    """On a fresh install (no active provider), flags are no-ops."""


    def test_reconfigure_on_fresh_install_falls_through(self, fresh_install):
        args = _make_setup_args(reconfigure=True)

        with ExitStack() as stack:
            m = _enter_fresh_install_patches(
                stack,
                prompt=("hermes_cli.setup.prompt_choice", {"return_value": 0}),
                first="hermes_cli.setup_quick._run_first_time_quick_setup",
            )
            from hermes_cli.setup import run_setup_wizard
            from hermes_cli import setup as setup_mod

            section_indexes = []
            m["first"].side_effect = lambda *_args: section_indexes.append(
                setup_mod._SETUP_NAVIGATION.get().section_index
            )
            run_setup_wizard(args)

        m["prompt"].assert_called_once()
        m["first"].assert_called_once()
        assert section_indexes == [0]

    def test_blank_slate_runs_inside_navigation_step(self, fresh_install):
        args = _make_setup_args()

        with ExitStack() as stack:
            m = _enter_fresh_install_patches(
                stack,
                prompt=("hermes_cli.setup.prompt_choice", {"return_value": 2}),
                blank="hermes_cli.setup_quick._run_blank_slate_setup",
            )
            from hermes_cli import setup as setup_mod

            section_indexes = []
            m["blank"].side_effect = lambda *_args: section_indexes.append(
                setup_mod._SETUP_NAVIGATION.get().section_index
            )
            setup_mod.run_setup_wizard(args)

        m["blank"].assert_called_once()
        assert section_indexes == [0]


class TestArgparse:
    """The flags are plumbed through argparse to cmd_setup."""

    def test_reconfigure_flag_reaches_cmd_setup(self, monkeypatch):
        import sys
        from hermes_cli.main import main

        captured = {}
        monkeypatch.setattr(
            "hermes_cli.setup.run_setup_wizard",
            lambda args: captured.setdefault("args", args),
        )
        monkeypatch.setattr(sys, "argv", ["hermes", "setup", "--reconfigure"])
        try:
            main()
        except SystemExit:
            pass
        assert captured["args"].reconfigure is True
        assert captured["args"].quick is False
