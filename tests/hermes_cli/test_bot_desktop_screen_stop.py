"""The documented CLI stop is also recovery for a disconnected viewer's lease."""

import argparse

from hermes_cli.subcommands.computer_use_screen import build_screen_parser
import pytest


@pytest.mark.linux_only
def test_screen_stop_hands_back_even_when_the_desktop_has_already_exited():
    from tools.bot_desktop import lease

    parser = argparse.ArgumentParser()
    build_screen_parser(parser.add_subparsers(), lambda sub, help_text: sub.add_argument('--json', action='store_true'))
    lease.acquire('disconnected-viewer')
    args = parser.parse_args(['screen', 'stop'])
    assert args.screen_func(args) == 0
    assert lease.get().holder == lease.AGENT
    assert lease.wait_for_release(timeout=0)
