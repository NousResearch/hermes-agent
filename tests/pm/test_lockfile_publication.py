"""Independent pin editors preserve winners and reject changed evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from pm.lock import Lockfile


def test_pin_writers_merge_without_losing_other_rows(tmp_path):
    path = tmp_path / 'lock.json'
    seed = Lockfile(path)
    seed.set_pin('left', '1', {})
    seed.set_pin('right', '1', {})
    seed.save()
    code = '''
from pathlib import Path
import sys
from pm.lock import Lockfile
lock = Lockfile(Path(sys.argv[1]))
lock.set_pin(sys.argv[2], '2', {})
print('ready', flush=True)
assert sys.stdin.readline().strip() == 'save'
lock.save()
'''
    root = Path(__file__).resolve().parents[2]
    children = [subprocess.Popen([sys.executable, '-u', '-c', code, str(path), name], cwd=root,
                                env=dict(os.environ), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for name in ('left', 'right')]
    try:
        # Pipes form a barrier: both writers read their snapshot before either saves.
        for child in children:
            ready = threading.Event()
            line = []
            threading.Thread(target=lambda c=child, e=ready, out=line: (out.append(c.stdout.readline()), e.set()),
                             daemon=True).start()
            assert ready.wait(10), 'writer did not reach the snapshot barrier'
            assert line == ['ready\n']
        for child in children:
            child.stdin.write('save\n')
            child.stdin.flush()
        for child in children:
            out, error = child.communicate(timeout=30)
            assert child.returncode == 0, out + error
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
    final = Lockfile(path)
    assert final.version('left') == final.version('right') == '2'


def test_stale_or_invalid_evidence_never_gets_overwritten(tmp_path):
    path = tmp_path / 'lock.json'
    initial = {'schema': 1, 'packages': {'tool': {'version': '1', 'artifacts': {}},
                                       'keep': {'version': '1', 'artifacts': {}}}}
    for change in ('changed', 'deleted', 'corrupt', 'removed-file'):
        path.write_text(json.dumps(initial), encoding='utf-8')
        stale = Lockfile(path)
        stale.set_pin('extra', '2', {})
        stale.set_pin('tool', '2', {})
        if change == 'corrupt':
            path.write_bytes(b'not JSON')
        elif change == 'removed-file':
            path.unlink()
        else:
            current = json.loads(path.read_text(encoding='utf-8'))
            if change == 'changed':
                current['packages']['tool']['version'] = '3'
            else:
                del current['packages']['tool']
            path.write_text(json.dumps(current), encoding='utf-8')
        before = path.read_bytes() if path.exists() else None
        with pytest.raises((RuntimeError, ValueError)):
            stale.save()
        assert (path.read_bytes() if path.exists() else None) == before
        assert not path.with_suffix('.corrupt').exists()
    path.write_text(json.dumps(initial), encoding='utf-8')
    first, retry = Lockfile(path), Lockfile(path)
    first.set_pin('tool', '2', {})
    first.save()
    before = path.read_bytes()
    os.utime(path, ns=(1234567890, 1234567890))
    modified = path.stat().st_mtime_ns
    retry.set_pin('tool', '2', {})
    retry.save()
    assert path.stat().st_mtime_ns == modified
    retry.save()
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == modified
