"""A gateway.lock this user cannot read and cannot prove unheld (a root run on macOS, which has no
/proc/locks) is refused with a terminal reason and the exact recovery command, never guessed at."""
import os

import pytest

pytestmark = [pytest.mark.platforms("linux", "macos"),
              pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason="root reads mode 000")]


def _unreadable_lock(tmp_path):
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    lock = home / 'gateway.lock'
    lock.write_text('{"pid": 1}', encoding='utf-8')
    lock.chmod(0)
    return home, lock


def test_reserve_without_proc_locks_refuses_with_recovery_command(tmp_path, monkeypatch):
    from gateway import runtime_ownership
    home, lock = _unreadable_lock(tmp_path)
    monkeypatch.setattr(runtime_ownership, '_inode_locked_per_proc', lambda ino: None)  # macOS: no table
    inode = lock.stat().st_ino
    with pytest.raises(PermissionError) as raised:
        runtime_ownership.ProfileOwnership().reserve([home])
    assert 'foreign_stale_lock' in str(raised.value) and f'sudo rm {lock}' in str(raised.value)
    assert lock.stat().st_ino == inode


def test_discovery_names_the_unreadable_lock_and_its_recovery(tmp_path):
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    home, lock = _unreadable_lock(tmp_path)
    result = discover_gateway_endpoint(home, timeout=1.0)
    assert (result.state, result.reason_code) == ('inaccessible', 'foreign_stale_lock')
    assert f'sudo lsof {lock}' in result.detail
