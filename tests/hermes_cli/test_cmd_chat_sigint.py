"""Regression tests for #80210: SIGINT during `hermes chat` startup/run must
exit cleanly with code 130, not crash with an unhandled traceback and a null
exit code.

The single-query signal handler in cli.py (_signal_handler_q) raises
KeyboardInterrupt to unwind the main thread. cli_main() catches it around
run_conversation (exit 130), but if the interrupt lands BEFORE the run loop —
during agent init or credential validation, both of which make network calls —
the exception escapes cmd_chat() entirely (cmd_chat only caught ValueError),
producing a raw traceback and a signal-death exit. This test pins the
cmd_chat-level catch.
"""

from __future__ import annotations

import argparse
import sys

import pytest


def _cmd_chat_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        model=None,
        provider=None,
        reasoning=None,
        toolsets=None,
        skills=None,
        verbose=False,
        quiet=True,
        query="hi",
        image=None,
        resume=None,
        worktree=False,
        checkpoints=False,
        pass_session_id=False,
        max_turns=None,
        ignore_rules=False,
        ignore_user_config=False,
        compact=False,
        yolo=False,
        continue_last=None,
        no_restore_cwd=True,
        source=None,
        tui=False,
        tui_dev=False,
        accept_hooks=False,
        safe_mode=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_cmd_chat_swallows_keyboard_interrupt_during_startup(
    monkeypatch,
) -> None:
    """SIGINT before the run loop must become a clean exit 130, not a crash.

    The interrupt lands during agent init / credential validation (network
    calls outside cli_main's own try/except). cmd_chat must catch it and
    sys.exit(130) — the standard 128+SIGINT convention.
    """

    def boom(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("cli.main", boom)
    # Hermetic: a clean CI has no provider configured, which makes cmd_chat
    # exit at the first-run guard (code 1) before the interrupt can land.
    # Force the configured path so the KeyboardInterrupt is genuinely reached.
    monkeypatch.setattr(
        "hermes_cli.main._has_any_provider_configured", lambda: True
    )

    from hermes_cli.main import cmd_chat

    with pytest.raises(SystemExit) as excinfo:
        cmd_chat(_cmd_chat_args())
    assert excinfo.value.code == 130, (
        "SIGINT during startup must exit 130, got "
        f"{excinfo.value.code!r}"
    )


def test_cmd_chat_still_propagates_value_error(monkeypatch) -> None:
    """The existing ValueError handling must be unaffected."""
    captured: dict[str, object] = {}

    def value_error(**kwargs):
        captured["kwargs"] = kwargs
        raise ValueError("bad config")

    monkeypatch.setattr("cli.main", value_error)
    # Non-hermetic guard: in clean CI no provider is configured, so
    # cmd_chat's first-run guard exits before cli.main is ever reached.
    monkeypatch.setattr(
        "hermes_cli.main._has_any_provider_configured", lambda: True
    )

    from hermes_cli.main import cmd_chat

    with pytest.raises(SystemExit) as excinfo:
        cmd_chat(_cmd_chat_args())
    assert excinfo.value.code == 1
    assert captured["kwargs"]["query"] == "hi"
