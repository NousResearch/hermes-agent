"""A killed predecessor releases the existing profile lock, never a second live writer."""
import os
import select
import subprocess
import sys

import pytest

from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership


@pytest.mark.platforms('linux', 'macos')
def test_successor_reserves_only_after_predecessor_exit(tmp_path):
    home = tmp_path / 'profile'
    code = '''
import sys
from pathlib import Path
from gateway.runtime_ownership import ProfileOwnership
owner = ProfileOwnership()
owner.reserve([Path(sys.argv[1])])
print('RESERVED', flush=True)
sys.stdin.read()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(home)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ))
    successor = ProfileOwnership()
    try:
        assert select.select([child.stdout], [], [], 10)[0], 'fixture never reserved'
        assert child.stdout.readline().strip() == 'RESERVED'
        with pytest.raises(OwnershipConflict):
            successor.reserve([home])
        assert not successor.owns(home)
        child.kill()
        child.wait(timeout=10)
        successor.reserve([home])
        assert successor.owns(home)
    finally:
        successor.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
