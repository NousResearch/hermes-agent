"""Private discovery must not assume another process shares its temp root."""
import json
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from gateway.control_socket import _home_hash
from hermes_cli.gateway_runtime_discovery import DiscoveryError, home_mode_unsafe, query_identify


def _ready_owner_payload(home):
    return {
        "runtime_protocol": 1,
        "state": "ready",
        "served_profiles": [{"home": str(home), "profile_id": str(home)}],
        "capabilities": ["session-authority-v1"],
        "authority_epoch": 1,
        "instance_id": "owner",
        "api_origin": "http://127.0.0.1:8765",
        "supervisor": "none",
    }


@pytest.mark.platforms("linux", "macos")
def test_discovery_resolves_a_private_overlay_lock_to_its_owner(tmp_path, monkeypatch):
    """ACP task homes share the served profile's lock instead of publishing a socket of their own."""
    import hermes_cli.gateway_runtime as runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    owner = tmp_path / "owner"
    overlay = tmp_path / "task-home"
    owner.mkdir(mode=0o700)
    overlay.mkdir(mode=0o700)
    lock = owner / "gateway.lock"
    lock.touch(mode=0o600)
    (overlay / "gateway.lock").symlink_to(lock)
    queried = []

    def identify(home, *, timeout):
        queried.append(home)
        return _ready_owner_payload(owner)

    monkeypatch.setattr(discovery, "query_identify", identify)
    result = runtime.discover_gateway_endpoint(overlay)

    assert result.state == "ready"
    assert result.endpoint is not None and result.endpoint.profile_id == str(owner)
    assert queried == [owner]

    checked = []

    def ready(home, *, timeout):
        checked.append(home)
        return runtime.GatewayDiscovery("ready")

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", ready)
    assert runtime.ensure_gateway_runtime(overlay).state == "ready"
    assert checked == [owner]


@pytest.mark.platforms("linux", "macos")
def test_discovery_does_not_follow_an_overlay_lock_from_a_writable_home(tmp_path, monkeypatch):
    """A task directory another user can replace must not select a gateway owner."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    import hermes_cli.gateway_runtime_discovery as discovery

    owner = tmp_path / "owner"
    overlay = tmp_path / "task-home"
    owner.mkdir(mode=0o700)
    overlay.mkdir(mode=0o700)
    overlay.chmod(0o707)
    lock = owner / "gateway.lock"
    lock.touch(mode=0o600)
    (overlay / "gateway.lock").symlink_to(lock)
    queried = []

    def identify(home, *, timeout):
        queried.append(home)
        return _ready_owner_payload(owner)

    monkeypatch.setattr(discovery, "query_identify", identify)
    result = discover_gateway_endpoint(overlay)

    assert result.state == "inaccessible"
    assert queried == [overlay]


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
def test_the_client_verdict_names_the_unsafe_node_and_its_fix(monkeypatch, request):
    """``Gateway inaccessible: <code> (<detail>)`` is what a client prints (``connect_gateway``): the
    detail must survive discovery and ensure, with the code unchanged for callers that match it."""
    import os
    import shutil
    from hermes_cli.gateway_runtime import discover_gateway_endpoint, ensure_gateway_runtime
    # A short root: the socket path must fit AF_UNIX's limit.
    root = Path(tempfile.mkdtemp(prefix='h-'))
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))
    home = root / 'h'
    home.mkdir(mode=0o700)
    control = home / 'gateway.sock'
    # Discovery names the canonical home, the path a chmod/chown must act on: on macOS the temp
    # root sits under /var, a symlink to /private/var, so the caller's spelling is not what it prints.
    shown, shown_control = home.resolve(), home.resolve() / 'gateway.sock'
    control.write_text('not a socket', encoding='utf-8')
    verdict = ensure_gateway_runtime(home, timeout=2)
    assert (verdict.state, verdict.reason_code) == ('inaccessible', 'unsafe_control_path')
    detail = verdict.detail or ''
    assert f'{shown_control} is -rw' in detail and 'expected socket' in detail, verdict

    control.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(control))
        control.chmod(0o660)
        verdict = discover_gateway_endpoint(home)
        assert verdict.reason_code == 'unsafe_control_permissions'
        assert verdict.detail == f'{shown_control} has mode 0660, expected 0600; chmod 600 {shown_control}', verdict
    control.unlink()

    home.chmod(0o757)
    verdict = discover_gateway_endpoint(home)
    assert verdict.reason_code == 'unsafe_control_permissions'
    assert verdict.detail == (f'{shown} has mode 0757, writable by another user; chmod go-w {shown}'), verdict
    home.chmod(0o700)

    owner = home.lstat().st_uid
    monkeypatch.setattr(os, 'getuid', lambda: owner + 1)
    verdict = discover_gateway_endpoint(home)
    assert verdict.reason_code == 'unsafe_control_path'
    assert verdict.detail == (f'{shown} is owned by uid {owner}, expected uid {owner + 1} (gateway user); '
                              'chown it to the gateway user'), verdict


@pytest.mark.platforms("linux", "macos")
def test_hermes_chat_prints_the_whole_repair_command(monkeypatch, request):
    """``hermes chat`` shows a failed attach through ``agent_init_failure_message``, whose 120-character
    summary cut the hint mid-path: the whole command must reach the user."""
    import asyncio
    import shutil
    from hermes_cli.cli_chat_error_copy import agent_init_failure_message
    from hermes_cli.gateway_client import GatewayUnavailableError, connect_gateway
    root = Path(tempfile.mkdtemp(prefix='h-'))
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))
    home = root / ('profile-' + 'x' * 12)
    home.mkdir(mode=0o700)
    control = home / 'gateway.sock'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)

    async def attach():
        async with connect_gateway():
            pass

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(control))
        control.chmod(0o660)
        with pytest.raises(GatewayUnavailableError) as unavailable:
            asyncio.run(attach())
    message = agent_init_failure_message(unavailable.value)
    assert f'chmod 600 {home.resolve() / "gateway.sock"})' in message, message


@pytest.mark.platforms("linux", "macos")
def test_a_long_home_keeps_the_repair_command_whole_or_drops_it(tmp_path):
    """The detail is bounded to 500 characters; a long profile home (named twice: the node and the
    command) must not leave the client printing a chmod cut off mid-path or missing its quote."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    omitted = '; repair command omitted (path too long to print)'
    home = tmp_path.resolve() / ('p' * 200) / "profile's home"
    home.mkdir(parents=True, mode=0o700)
    home.chmod(0o757)
    verdict = discover_gateway_endpoint(home)
    assert verdict.reason_code == 'unsafe_control_permissions'
    assert verdict.detail == f'{home} has mode 0757, writable by another user{omitted}', verdict

    # Too long even without the command: the description is cut with a visible marker.
    deeper = home / ('q' * 250) / ('r' * 100)
    deeper.mkdir(parents=True, mode=0o700)
    deeper.chmod(0o757)
    detail = discover_gateway_endpoint(deeper).detail or ''
    assert len(detail) == 500 and detail.endswith(f'… [truncated]{omitted}'), detail
    assert detail.startswith(str(deeper)[:300]) and 'chmod' not in detail, detail


@pytest.mark.platforms("linux", "macos")
def test_a_permission_hint_runs_as_printed_for_a_path_with_spaces(tmp_path):
    """The hint is a command the user pastes into a shell: a profile home with a space must survive it."""
    import subprocess
    from hermes_cli.gateway_runtime_discovery import _private_node

    home = tmp_path / 'profile home'
    home.mkdir(mode=0o700)
    control = home / 'control file'
    control.write_text('', encoding='utf-8')
    for path, kind, is_home, mode in ((control, 'file', False, 0o660), (home, 'directory', True, 0o702)):
        path.chmod(mode)
        with pytest.raises(DiscoveryError, match='unsafe_control_permissions') as error:
            _private_node(path, kind=kind, home=is_home)
        hint = str(error.value).split('; ', 1)[1]
        ran = subprocess.run(['sh', '-c', hint], capture_output=True, text=True, check=False)
        assert ran.returncode == 0, (hint, ran.stderr)
        _private_node(path, kind=kind, home=is_home)


@pytest.mark.platforms("linux")
def test_an_open_control_directory_names_its_path_and_mode(tmp_path):
    home = tmp_path / 'profile'
    home.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='h-owner-') as root:
        directory = Path(root) / f'hermes-gw-{_home_hash(home)}'
        directory.mkdir(mode=0o755)
        pointer = home / 'gateway.sock.path'
        pointer.write_text(str(directory / 'control.sock'), encoding='utf-8')
        pointer.chmod(0o600)
        with pytest.raises(DiscoveryError, match='unsafe_control_permissions') as refused:
            query_identify(home, timeout=2)
        assert refused.value.reason == 'unsafe_control_permissions'
        assert getattr(refused.value, 'detail', None) == f'{directory} has mode 0755, expected 0700; chmod 700 {directory}'


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


@pytest.mark.platforms("linux", "macos")
def test_upgraded_same_owner_0644_lock_is_narrowed_not_inaccessible(tmp_path):
    """Main created gateway.lock with open("a+") under the umask (0644); the inode is never
    replaced, so discovery and reserve must narrow it in place instead of refusing it forever.
    A hardlinked lock (another name shares the inode) is never chmodded."""
    import os
    import stat
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli.gateway_runtime_discovery import missing_owner_state
    home = tmp_path / 'profile'
    home.mkdir(mode=0o700)
    lock = home / 'gateway.lock'
    lock.touch()
    lock.chmod(0o644)
    assert missing_owner_state(home) == 'absent'
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    lock.chmod(0o644)
    owner = ProfileOwnership()
    owner.reserve([home])
    try:
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600
        assert missing_owner_state(home) == 'starting'
    finally:
        owner.close()
    lock.chmod(0o644)
    os.link(lock, tmp_path / 'alias')
    assert missing_owner_state(home) == 'inaccessible'
    assert stat.S_IMODE(lock.stat().st_mode) == 0o644
