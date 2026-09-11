"""Regression test: bare /hatch must not call raw input() off the main thread.

Slash commands dispatch from the process_loop daemon thread while
prompt_toolkit owns stdin. The old code called ``input()`` directly, which
rendered nothing and silently swallowed subsequent keystrokes until Ctrl+C
(found in the Aug 2026 full-surface CLI QA sweep). The handler must route
through the thread-aware ``_prompt_text_input`` helper, which cancels
cleanly (returns None) when prompting isn't safe.
"""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _StandIn(SimpleNamespace):
    pass


def test_bare_hatch_uses_thread_aware_prompt_not_raw_input(capsys):
    calls = []

    def fake_prompt(prompt_text):
        calls.append(prompt_text)
        return None  # helper cancelled (e.g. unsafe thread context)

    stand_in = _StandIn(_prompt_text_input=fake_prompt)

    def _boom(*a, **k):  # any raw input() call is the regression
        raise AssertionError("raw input() called from /hatch handler")

    with patch.object(builtins, "input", _boom):
        CLICommandsMixin._handle_hatch_command(stand_in, "/hatch")

    assert calls, "expected /hatch to route through _prompt_text_input"
    out = capsys.readouterr().out
    assert "Usage: /hatch" in out  # cancelled prompt falls through to usage
