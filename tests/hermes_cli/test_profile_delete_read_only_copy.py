"""A read-only profile copy (0555 dirs, 0444 files) can still be deleted (N106): the owner check
cannot reserve gateway.lock there, so it proves the lock unheld through a read-only flock."""
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.platforms("linux", "macos"),
              pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason="root writes 0555")]


def _read_only_copy(home, *, lock):
    (home / 'sub').mkdir(parents=True)
    (home / 'config.yaml').write_text('model: x\n')
    if lock:
        (home / 'gateway.lock').write_text('')
    for parent, _dirs, files in os.walk(home, topdown=False):
        for name in files:
            os.chmod(os.path.join(parent, name), 0o444)
        os.chmod(parent, 0o555)


@pytest.mark.parametrize('lock', [False, True])
def test_read_only_copy_is_unowned_unless_its_lock_is_held(tmp_path, lock):
    from hermes_cli.profiles_remove import wait_profile_unowned
    home = tmp_path / 'profiles' / 'ro'
    _read_only_copy(home, lock=lock)
    try:
        assert wait_profile_unowned(home, timeout=0) is True
        if lock:  # a live holder of the read-only inode still blocks the delete
            holder = subprocess.Popen(
                [sys.executable, '-c', 'import fcntl,sys;f=open(sys.argv[1]);fcntl.flock(f,fcntl.LOCK_SH);'
                 'print(1,flush=True);sys.stdin.read()', str(home / 'gateway.lock')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                assert holder.stdout.readline().strip() == '1'
                assert wait_profile_unowned(home, timeout=0) is False
            finally:
                holder.stdin.close()
                holder.wait(timeout=10)
    finally:
        for parent, _dirs, _files in os.walk(tmp_path):
            os.chmod(parent, 0o755)
