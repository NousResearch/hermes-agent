"""Regression tests for the CLI's Ctrl+C/Ctrl+Q clear/interrupt/exit priority.

Before this fix, ``handle_ctrl_c``/``handle_ctrl_q`` checked whether the
agent was running BEFORE checking whether the composer had a draft — so
pressing Ctrl+C to clear a typo or half-typed correction while the agent was
mid-stream instead interrupted (killed) the running turn. The TUI had the
identical bug and fixed it with a pure priority function,
``resolveCtrlCComposerAction`` (PR #89171, ``ui-tui/src/app/useInputHandlers.ts``):
a non-empty draft always wins over interrupting. ``resolve_ctrl_c_composer_action``
in ``cli.py`` mirrors that fix for the CLI's two Ctrl+C-shaped handlers.
"""

from __future__ import annotations

import cli as cli_mod


def test_draft_wins_over_interrupt_while_agent_running():
    assert (
        cli_mod.resolve_ctrl_c_composer_action(has_draft=True, agent_running=True)
        == "clear"
    )


def test_interrupts_running_agent_when_draft_is_empty():
    assert (
        cli_mod.resolve_ctrl_c_composer_action(has_draft=False, agent_running=True)
        == "interrupt"
    )


def test_clears_an_idle_draft_instead_of_exiting():
    assert (
        cli_mod.resolve_ctrl_c_composer_action(has_draft=True, agent_running=False)
        == "clear"
    )


def test_exits_when_idle_with_an_empty_draft():
    assert (
        cli_mod.resolve_ctrl_c_composer_action(has_draft=False, agent_running=False)
        == "exit"
    )
