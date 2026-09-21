"""``hermes chat -m <alias>`` must expand configured model aliases.

Symptom on main: a ``model.aliases`` / ``model_aliases:`` entry passed to
``chat``'s ``-m/--model`` was forwarded as a literal model id.  With no
``--provider`` the configured provider's fuzzy normalizer silently substituted
an unrelated default (``sol`` -> the profile default model, so the user got a
completely different model with no error), and with an explicit ``--provider``
the upstream API rejected the alias with a 400.

``hermes -z ... -m <alias>`` (oneshot) and ``/model <alias>`` already resolved
direct aliases; only the ``chat`` path did not.  The fix routes all three
through ``model_switch.resolve_direct_alias()`` so they cannot drift again.

Behavior contracts asserted here (not implementation snapshots):
  * the model half always becomes the alias's model;
  * an explicit ``--provider`` wins over the alias's provider;
  * a non-alias model id passes through untouched (no accidental rewriting);
  * the resolution happens before dispatch, so the CLI/quiet-query path
    receives the expanded pair.
"""

import argparse
import sys
import types

import pytest

import hermes_cli.main as main_mod
import hermes_cli.model_switch as ms
from hermes_cli.model_switch import DirectAlias


@pytest.fixture
def aliases(monkeypatch):
    """Install a known alias table, bypassing config loading."""
    table = {
        "sol": DirectAlias("gpt-5.6-sol", "openai-codex", ""),
        "local-qwen": DirectAlias("qwen3-local", "custom", "http://127.0.0.1:8080/v1"),
    }
    monkeypatch.setattr(ms, "DIRECT_ALIASES", dict(table))
    monkeypatch.setattr(ms, "_ensure_direct_aliases", lambda: None)
    return table


def _args(**overrides) -> argparse.Namespace:
    base = {
        "model": None,
        "provider": None,
        "toolsets": "all",
        "query": "hi",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestResolveDirectAlias:
    """The shared lookup used by every entry path."""

    def test_case_insensitive_hit(self, aliases):
        assert ms.resolve_direct_alias("SOL").model == "gpt-5.6-sol"

    @pytest.mark.parametrize("value", [None, "", "   ", "not-an-alias", "gpt-5.6-sol"])
    def test_miss_returns_none(self, aliases, value):
        assert ms.resolve_direct_alias(value) is None


class TestExpandModelAliasFlag:
    """The ``chat`` -m/--model expansion."""

    def test_alias_fills_model_and_provider(self, aliases):
        args = _args(model="sol")
        main_mod._expand_model_alias_flag(args)
        assert args.model == "gpt-5.6-sol"
        assert args.provider == "openai-codex"

    def test_explicit_provider_wins(self, aliases):
        """`-m sol --provider openrouter` still asks OpenRouter for that model."""
        args = _args(model="sol", provider="openrouter")
        main_mod._expand_model_alias_flag(args)
        assert args.model == "gpt-5.6-sol"
        assert args.provider == "openrouter"

    def test_alias_base_url_propagates(self, aliases):
        args = _args(model="local-qwen")
        main_mod._expand_model_alias_flag(args)
        assert args.model == "qwen3-local"
        assert args.provider == "custom"
        assert args.base_url == "http://127.0.0.1:8080/v1"

    def test_literal_model_id_untouched(self, aliases):
        """A real model id must not be rewritten or given a provider."""
        args = _args(model="deepseek-v4-flash")
        main_mod._expand_model_alias_flag(args)
        assert args.model == "deepseek-v4-flash"
        assert args.provider is None
        assert not hasattr(args, "base_url")

    def test_no_model_flag_is_noop(self, aliases):
        args = _args(model=None)
        main_mod._expand_model_alias_flag(args)
        assert args.model is None
        assert args.provider is None


class TestChatDispatchReceivesExpandedModel:
    """Resolution must happen before dispatch, not only in the helper."""

    @pytest.fixture
    def dispatch(self, monkeypatch):
        """Neutralize cmd_chat's side effects and capture cli.main's kwargs."""
        captured = {}

        cli_stub = types.ModuleType("cli")

        def _cli_main(**kwargs):
            captured.update(kwargs)

        cli_stub.main = _cli_main
        monkeypatch.setitem(sys.modules, "cli", cli_stub)

        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: False)
        monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda args: None)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
        monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        return captured

    def test_cli_receives_expanded_model(self, aliases, dispatch):
        args = _args(model="sol", verbose=None, quiet=False, image=None)
        main_mod.cmd_chat(args)
        assert dispatch["model"] == "gpt-5.6-sol"
        assert dispatch["provider"] == "openai-codex"

    def test_cli_receives_literal_model_unchanged(self, aliases, dispatch):
        args = _args(model="deepseek-v4-flash", verbose=None, quiet=False, image=None)
        main_mod.cmd_chat(args)
        assert dispatch["model"] == "deepseek-v4-flash"
        assert "provider" not in dispatch

    def test_tui_child_gets_expanded_model(self, aliases, monkeypatch):
        """TUI mode forwards the resolved model via HERMES_INFERENCE_MODEL."""
        seen = {}

        def _fake_tui(resume_session_id=None, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda args: True)
        monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda args: None)
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
        monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
        monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
        monkeypatch.setattr(main_mod, "_launch_tui", _fake_tui)

        args = _args(model="sol", verbose=None, quiet=False, image=None)
        main_mod.cmd_chat(args)

        assert seen["model"] == "gpt-5.6-sol"
        assert seen["provider"] == "openai-codex"
