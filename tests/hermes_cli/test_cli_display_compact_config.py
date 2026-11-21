"""``display.compact`` in config.yaml reaches the classic CLI (regression for #4848).

``hermes chat`` has no ``--compact`` flag, so cmd_chat used to hand ``compact=False`` down to
``HermesCLI``, and the ``compact if compact is not None else display.compact`` fallback never ran."""
import argparse

import pytest


class _Built(Exception):
    """Stops cli.main right after HermesCLI resolved its display options."""


def _chat_args(**kw):
    # The real chat parser defines no ``compact`` attribute; only Termux's fast path sets one.
    base = dict(command="chat", query=None, model=None, provider=None, toolsets=None, skills=None,
                verbose=None, quiet=False, image=None, resume=None, worktree=False, checkpoints=False,
                pass_session_id=False, max_turns=None, output_format="text", tui=False, cli=True,
                safe_mode=False, ignore_user_config=False, continue_last=None, yolo=False,
                ignore_rules=False, source=None, query_file=None, oneshot_exit=False,
                run_budget=None, reasoning=None, accept_hooks=False)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def resolved_compact(monkeypatch):
    """Run cmd_chat through the real cli.main/HermesCLI and return the compact value HermesCLI used."""
    import cli
    import hermes_cli.main as m
    from hermes_cli.cli_init_mixin import CLIInitMixin

    for name in ("_resolve_chat_session_args", "_warn_retired_xai_models", "_start_chat_background_prefetch",
                 "_pin_kanban_board_env", "_confirm_startup_expensive_model_override",
                 "_guard_noninteractive_user_config"):
        monkeypatch.setattr(m, name, lambda *a, **k: None)
    monkeypatch.setattr(m, "_resolve_use_tui", lambda args: False)
    monkeypatch.setattr(m, "_has_any_provider_configured", lambda: True)
    import hermes_cli.free_tier_bootstrap as boot
    monkeypatch.setattr(boot, "run_bootstrap", lambda **k: None)
    import hermes_cli.observability.shared_metrics_consent as consent
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda args: None)
    import hermes_cli.observability.shared_metrics_process as proc
    monkeypatch.setattr(proc, "begin_process", lambda name: None)

    seen = {}
    real_init = CLIInitMixin._init_display_options

    def spy(self, verbose, compact):
        real_init(self, verbose, compact)
        seen["compact"] = self.compact
        raise _Built

    monkeypatch.setattr(CLIInitMixin, "_init_display_options", spy)

    def run(config_compact, **arg_overrides):
        monkeypatch.setitem(cli.CLI_CONFIG["display"], "compact", config_compact)
        with pytest.raises(_Built):
            m.cmd_chat(_chat_args(**arg_overrides))
        return seen["compact"]

    return run


@pytest.mark.parametrize("config_compact", [True, False])
def test_chat_without_flag_follows_display_compact(resolved_compact, config_compact):
    assert resolved_compact(config_compact) is config_compact


def test_explicit_compact_arg_still_wins_over_config(resolved_compact):
    # Termux's fast interactive path sets args.compact=True regardless of config.
    assert resolved_compact(False, compact=True) is True
