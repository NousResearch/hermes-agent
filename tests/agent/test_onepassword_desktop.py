"""Desktop authentication must retain Hermes's profile and lock boundaries."""
import sys

import pytest

from agent.secret_scope import set_secret_scope, reset_secret_scope
from agent.vault_backends import unlock
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.onepassword import OnePasswordLoginBackend

pytestmark = pytest.mark.platforms('posix')


def fake_op(tmp_path, *, signin=0, whoami=0, token=''):
    exe = tmp_path / 'op'
    exe.write_text(f'''#!{sys.executable}
import os, sys
args = sys.argv[1:]
if args[0] == 'signin':
    print({token!r}, end='')
    sys.exit({signin})
if args[0] == 'whoami':
    print('{{"user_uuid":"synthetic-user"}}')
    sys.exit({whoami})
if args[:2] == ['item', 'list']:
    sessions = {{k:v for k,v in os.environ.items() if k.startswith('OP_SESSION')}}
    assert sessions == ({{'OP_SESSION': {token!r}}} if {bool(token)!r} else {{}})
    print('[]')
    sys.exit(0)
sys.exit(2)
''')
    exe.chmod(0o700)
    return exe


@pytest.mark.parametrize('token', ['', 'synthetic-session-token'])
def test_unlock_survives_backend_recreation_but_not_profile_switch_or_lock(tmp_path, monkeypatch, token):
    exe = fake_op(tmp_path, token=token)
    scope = set_secret_scope({})
    try:
        a, b = tmp_path/'a', tmp_path/'b'
        monkeypatch.setenv('HERMES_HOME', str(a))
        unlock.lock('onepassword')
        backend = OnePasswordLoginBackend({'binary_path': str(exe)})
        backend.unlock('synthetic-password')
        assert OnePasswordLoginBackend({'binary_path': str(exe)}).is_unlocked()
        assert backend.list_items() == []
        monkeypatch.setenv('HERMES_HOME', str(b))
        assert not backend.is_unlocked()
        with pytest.raises(UnlockRequired):
            backend._run('item', 'list')
        monkeypatch.setenv('HERMES_HOME', str(a))
        assert backend.is_unlocked()
        unlock.lock('onepassword')
        with pytest.raises(UnlockRequired):
            backend._run('item', 'list')
    finally:
        unlock.lock('onepassword')
        reset_secret_scope(scope)


@pytest.mark.parametrize('signin,whoami', [(0, 1), (1, 0)])
def test_no_token_requires_successful_signin_and_identity_check(tmp_path, monkeypatch, signin, whoami):
    exe = fake_op(tmp_path, signin=signin, whoami=whoami)
    scope = set_secret_scope({})
    try:
        monkeypatch.setenv('HERMES_HOME', str(tmp_path/'home'))
        unlock.lock('onepassword')
        backend = OnePasswordLoginBackend({'binary_path': str(exe)})
        with pytest.raises(RuntimeError):
            backend.unlock('synthetic-password')
        assert not backend.is_unlocked()
    finally:
        unlock.lock('onepassword')
        reset_secret_scope(scope)
