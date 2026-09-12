"""Bot relay uses the ordinary daemon's durable FIFO, not an attached viewer."""
import os
from pathlib import Path
import subprocess
import sys


def test_bot_delivery_busy_retry_keeps_exact_target(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env['PYTHONPATH'] = str(root)
    proc = subprocess.run([sys.executable, str(root / 'tests/gateway/fixtures/bot_authority_peer.py'),
                           str(tmp_path)], cwd=root, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=150)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout)
