"""A hardlinked gateway.lock (a cp -al / rsync --link-dest profile clone) is never written."""
import os

import pytest


@pytest.mark.platforms("linux")
def test_private_hardlinked_gateway_lock_is_refused_unwritten(tmp_path):
    from gateway import runtime_ownership
    clone, home = tmp_path / 'clone', tmp_path / 'home'
    for path in (clone, home):
        path.mkdir(mode=0o700)
    sentinel = clone / 'gateway.lock'
    sentinel.write_text('sentinel', encoding='utf-8')
    sentinel.chmod(0o600)  # already private: the chmod branch never runs
    os.link(sentinel, home / 'gateway.lock')
    with pytest.raises(PermissionError, match='link count'):
        runtime_ownership.ProfileOwnership().reserve([home])
    assert sentinel.read_text(encoding='utf-8') == 'sentinel'
