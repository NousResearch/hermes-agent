"""Ink/Desktop verbs the owner answered -32601 or refused now run their gateway handlers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env


def probe(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir()
    user.mkdir()
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/tui_desktop_routes_peer.py'), mode],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((home / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['mode'] == mode and receipt['authenticated_ws']
    return receipt


def test_message_react_writes_the_owned_transcript_row(tmp_path):
    assert probe(tmp_path, 'react')['reacted']


def test_session_control_buttons_run_the_typed_command_handlers(tmp_path):
    assert probe(tmp_path, 'control')['controlled']


def test_a_criterion_button_that_cannot_act_is_refused(tmp_path):
    assert probe(tmp_path, 'control_refusals')['refused']


def test_the_connection_card_drives_the_sessions_open_operation(tmp_path):
    assert probe(tmp_path, 'connection_card')['card']


@pytest.mark.platforms("posix")
def test_exec_quick_and_plugin_commands_run_for_a_controlling_viewer(tmp_path):
    assert probe(tmp_path, 'commands')['commands']
