"""A real native background child leaves its parent available for voice tools."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
def test_native_background_child_does_not_block_parent_tools(tmp_path, api_mode):
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HERMES_HOME=str(tmp_path), HERMES_TEST_ISOLATION=str(tmp_path),
               HERMES_DISABLE_LAZY_INSTALLS='1', PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures/plugin_exact_peer.py'),
                             'default', 'strict', api_mode],
                            cwd=root, env=env, capture_output=True, text=True, timeout=110, check=False)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    receipt = json.loads((tmp_path / 'hybrid-receipt.json').read_text())
    assert receipt['direct_completed_while_child_running']
    assert receipt['child_session'] != receipt['parent_session']
