"""GW-1 (#134889): `hermes pause` holds every gateway turn, interactive ones included.

Since #106742 every local turn is a gateway turn, so the emergency stop holding CLI, TUI and Desktop
turns is the stop working as designed. What every surface must get is the same named refusal
(``agent.estop.paused_reply``: the reason and "run `hermes resume`"), and a turn already running
when the pause lands must finish. Real GatewayRunner + authority + WS API, loopback model.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


@pytest.fixture(scope='module')
def receipt(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    tmp = tmp_path_factory.mktemp('estop')
    home, user = tmp / 'state', tmp / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/estop_interactive_peer.py')],
                            cwd=root, env=env, capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((home / 'receipt.json').read_text())


def _text(payload):
    return str(payload.get('text') or payload.get('content') or payload.get('final_response') or '')


def test_a_turn_already_running_when_pause_lands_finishes(receipt):
    assert 'RECOVERY_ACK_BLOCK_STARTED' in _text(receipt['running']), receipt['running']


def test_a_new_interactive_turn_gets_the_named_refusal_with_reason_and_resume(receipt):
    notice = receipt['expected_notice']
    assert 'maintenance window' in notice and 'hermes resume' in notice, notice
    assert _text(receipt['paused']).strip() == notice, receipt['paused']
    assert receipt['paused_model_calls'] == 0, receipt


def test_resume_lets_the_next_turn_run(receipt):
    assert 'RECOVERY_ACK_AFTER_RESUME' in _text(receipt['after']), receipt['after']


def test_pause_help_and_docs_say_interactive_turns_are_held_too():
    """The decided contract is documented where operators read it (`hermes pause --help`, the CLI
    reference): interactive CLI, TUI and Desktop turns are gateway turns and are held too."""
    import argparse
    from hermes_cli.subcommands.pause import build_pause_parser
    parser = argparse.ArgumentParser(prog='hermes')
    build_pause_parser(parser.add_subparsers())
    sub = parser._subparsers._group_actions[0].choices['pause']
    text = ' '.join(sub.format_help().split())
    for needle in ('CLI', 'TUI', 'Desktop', 'hermes resume'):
        assert needle in text, text
    doc = (Path(__file__).resolve().parents[2] / 'website/docs/reference/cli-commands.md').read_text()
    row = next(line for line in doc.splitlines() if line.startswith('| `hermes pause`'))
    assert 'CLI, TUI and Desktop' in row and 'finish' in row, row
