"""Private discovery must not assume another process shares its temp root."""
import json
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from gateway.control_socket import _home_hash
from hermes_cli.gateway_runtime_discovery import DiscoveryError, query_identify


@pytest.mark.linux_only
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
