"""Private discovery must not assume another process shares its temp root."""
import json
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from gateway.control_socket import _home_hash
from hermes_cli.gateway_runtime_discovery import DiscoveryError, home_mode_unsafe, query_identify


@pytest.mark.platforms("linux")
def test_fallback_pointer_uses_owner_location_and_keeps_identity_checks(tmp_path):
    home = tmp_path / 'profile'
    home.mkdir(mode=0o700)
    # The owner has a distinct TMPDIR, as a service or native app commonly does.
    with tempfile.TemporaryDirectory(prefix='h-owner-') as root:
        directory = Path(root) / f'hermes-gw-{_home_hash(home)}'
        directory.mkdir(mode=0o700)
        target = directory / 'control.sock'
        pointer = home / 'gateway.sock.path'
        pointer.write_text(str(target), encoding='utf-8')
        pointer.chmod(0o600)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(target))
            target.chmod(0o600)
            server.listen(1)
            server.settimeout(3)
            requests = []

            def respond():
                try:
                    peer, _ = server.accept()
                    with peer:
                        requests.append(peer.recv(4096))
                        peer.sendall(b'{"ok":true,"protocol":1,"id":1,"result":{"state":"ready"}}\n')
                except TimeoutError:
                    return

            thread = threading.Thread(target=respond)
            thread.start()
            try:
                assert query_identify(home, timeout=2) == {'state': 'ready'}
            finally:
                thread.join(timeout=4)
            assert json.loads(requests[0])['verb'] == 'identify'
            pointer.write_text(str(directory.with_name('hermes-gw-wrong') / 'control.sock'), encoding='utf-8')
            with pytest.raises(DiscoveryError, match='invalid_control_pointer'):
                query_identify(home, timeout=2)
            pointer.write_text(str(target), encoding='utf-8')
            directory.chmod(0o755)
            with pytest.raises(DiscoveryError, match='unsafe_control_permissions'):
                query_identify(home, timeout=2)


@pytest.mark.platforms("linux", "macos")
def test_home_mode_refuses_only_writes_another_user_can_make(tmp_path, monkeypatch):
    import grp
    import os
    import pwd
    home = tmp_path / 'home'
    home.mkdir()
    user = pwd.getpwuid(os.getuid())
    monkeypatch.setattr(pwd, 'getpwall', lambda: [user])  # no other account on this gid
    private = grp.struct_group((user.pw_name, 'x', user.pw_gid, []))
    shared = grp.struct_group(('staff', 'x', user.pw_gid, [user.pw_name, 'someone-else']))
    for mode, group, unsafe in ((0o700, private, False), (0o755, shared, False), (0o757, private, True),
                                (0o775, private, False), (0o775, shared, True)):
        home.chmod(mode)
        monkeypatch.setattr(grp, 'getgrgid', lambda _gid, g=group: g)
        assert home_mode_unsafe(home.lstat(), home) is unsafe, (oct(mode), group.gr_name)


@pytest.mark.platforms("linux", "macos")
def test_group_write_is_private_only_when_no_other_account_has_that_primary_group(tmp_path, monkeypatch):
    import grp
    import os
    import pwd
    home = tmp_path / 'home'
    home.mkdir()
    home.chmod(0o775)
    user = pwd.getpwuid(os.getuid())
    # gr_mem never lists accounts whose PRIMARY gid is the group: an empty member list
    # with another passwd record on that gid is a shared group, not a private one.
    monkeypatch.setattr(grp, 'getgrgid', lambda _gid: grp.struct_group((user.pw_name, 'x', user.pw_gid, [])))
    other = pwd.struct_passwd(('intruder', 'x', user.pw_uid + 1, user.pw_gid, '', '/', '/bin/sh'))
    monkeypatch.setattr(pwd, 'getpwall', lambda: [user, other])
    assert home_mode_unsafe(home.lstat(), home) is True
    monkeypatch.setattr(pwd, 'getpwall', lambda: [user])
    assert home_mode_unsafe(home.lstat(), home) is False

    def unreadable():
        raise OSError('passwd database unavailable')
    monkeypatch.setattr(pwd, 'getpwall', unreadable)
    assert home_mode_unsafe(home.lstat(), home) is True
