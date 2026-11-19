"""The documented CLI stop --force is also recovery for a disconnected viewer's lease."""

import argparse

from hermes_cli.subcommands.computer_use_screen import build_screen_parser
import pytest


@pytest.mark.platforms("linux")
def test_screen_stop_hands_back_even_when_the_desktop_has_already_exited():
    from tools.bot_desktop import lease

    parser = argparse.ArgumentParser()
    build_screen_parser(parser.add_subparsers(), lambda sub, help_text: sub.add_argument('--json', action='store_true'))
    lease.acquire('disconnected-viewer')
    args = parser.parse_args(['screen', 'stop', '--force'])
    assert args.screen_func(args) == 0
    assert lease.get().holder == lease.AGENT


@pytest.mark.platforms("linux")
def test_screen_stop_refuses_while_a_human_holds_unless_forced(monkeypatch):
    """Same door as display.stop: a runbook or stray `screen stop` must not yank a live takeover."""
    from tools.bot_desktop import lease, runtime

    stopped: list = []
    monkeypatch.setattr(runtime, "stop", lambda: stopped.append(1) or True)
    parser = argparse.ArgumentParser()
    build_screen_parser(parser.add_subparsers(), lambda sub, help_text: sub.add_argument('--json', action='store_true'))
    lease.acquire('human-at-keyboard')
    args = parser.parse_args(['screen', 'stop'])
    assert args.screen_func(args) == 1
    assert lease.get().holder == lease.HUMAN and not stopped
    args = parser.parse_args(['screen', 'stop', '--force'])
    assert args.screen_func(args) == 0
    assert lease.get().holder == lease.AGENT and stopped
