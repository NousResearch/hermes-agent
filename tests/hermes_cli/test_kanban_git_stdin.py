"""Noninteractive board Git queries must not consume the worker's control input (#144287)."""
import hashlib
from pathlib import Path
import subprocess
import sys


PROBE = """
from pathlib import Path
import sys
from hermes_cli.kanban_db import _git_out
print('git_hash=' + str(_git_out(Path(sys.argv[1]), 'hash-object', '--stdin', timeout=15)))
"""


def test_git_out_does_not_consume_inherited_control_bytes(tmp_path):
    # No parked reader here: a real Git child must see EOF, not the owner's supplied bytes.
    # Closing this input is intentional; the separate worker test exercises a parked read.
    result = subprocess.run(
        [sys.executable, '-c', PROBE, str(tmp_path)], cwd=Path(__file__).resolve().parents[2],
        input=b'{"type":"finish"}\n', capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    seen = dict(line.split('=', 1) for line in result.stdout.decode().splitlines() if '=' in line)
    empty_blob = hashlib.sha1(b'blob 0\0').hexdigest()
    assert seen.get('git_hash') == empty_blob, 'Git consumed control bytes instead of reading EOF'
