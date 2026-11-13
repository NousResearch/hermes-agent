"""Text typed while a turn runs must not land in plaintext files under HERMES_HOME (#75461).

Users paste credentials into the prompt (bot tokens during platform setup). The interrupt
paths used to append the raw message to ``interrupt_debug.log`` with ``open().write()``,
bypassing the logging module's RedactingFormatter.
"""

from __future__ import annotations

import queue
from unittest.mock import patch

from agent.onboarding import BUSY_INPUT_FLAG
from tests.hermes_cli.test_cli_interrupt_ack_race import _make_cli, _StubAgent

SECRET = "1234567890:" + "A" * 35  # Telegram bot-token shape, synthetic


def _files_containing(root, needle):
    return [p for p in root.rglob("*") if p.is_file() and needle in p.read_text(errors="ignore")]


def test_interrupt_fired_during_chat_does_not_persist_the_message(tmp_path):
    cli = _make_cli()
    agent = _StubAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._interrupt_queue.put(f"my token is {SECRET}")

    import cli as cli_mod

    with patch.object(cli_mod, "_hermes_home", tmp_path), \
         patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("original")

    assert agent.interrupt_calls == [f"my token is {SECRET}"]
    assert _files_containing(tmp_path, SECRET) == []


def test_enter_while_busy_interrupt_does_not_persist_the_message(tmp_path):
    import cli as cli_mod

    shell = object.__new__(cli_mod.HermesCLI)
    shell.busy_input_mode = "interrupt"
    shell.agent = None  # no redirect support -> legacy interrupt queue
    shell._agent_running = True
    shell._pending_input, shell._interrupt_queue = queue.Queue(), queue.Queue()
    config = {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}}

    with patch.object(cli_mod, "_hermes_home", tmp_path), patch.dict(cli_mod.__dict__, {"CLI_CONFIG": config}):
        shell._tui_enter_while_busy(SECRET, [], SECRET)

    assert shell._interrupt_queue.get_nowait() == SECRET
    assert _files_containing(tmp_path, SECRET) == []


def test_no_tip_points_users_at_the_removed_debug_file():
    """The startup tips must not send users to ``interrupt_debug.log``: nothing writes it now."""
    from agent import i18n
    from agent.i18n import t
    from hermes_cli import tips

    stale = [(lang, key) for lang in i18n.supported_languages() for key in tips.tip_keys()
             if "interrupt_debug.log" in t(key, lang=lang)]
    assert stale == []
