import socket
import pytest
from hermes_cli import gateway


@pytest.mark.parametrize('occupied', [False, True])
def test_timeout_requires_all_resolved_addresses_free(monkeypatch, occupied):
    attempts = []
    class Probe:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def setsockopt(self, *args): pass
        def bind(self, address):
            attempts.append(address)
            if occupied and address[0] == '127.0.0.1':
                raise OSError('occupied')
    addresses = [(socket.AF_INET6, socket.SOCK_STREAM, 0, '', ('::1', 8134)), (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', 8134))]
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **kw: addresses)
    monkeypatch.setattr(socket, 'socket', Probe)
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
    assert gateway._wait_for_tcp_port_free('localhost', 8134, timeout=0.02) is (not occupied)
    assert attempts[:2] == [('::1', 8134), ('127.0.0.1', 8134)]
