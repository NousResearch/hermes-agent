"""Loopback OpenAI-compatible model peer shared by the ordinary-daemon wire tests.

Registered in ``tests/conftest.py`` (see the note there on why fixtures are imported rather than
listed in ``pytest_plugins``) so gateway, ACP and CLI tests receive ``model_peer`` by name.
"""
import pytest


@pytest.fixture
def model_peer():
    from http.server import ThreadingHTTPServer
    import threading
    from tests.gateway.fixtures.shared_authority_peer import ModelPeer
    server = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    server.requests, server.metadata_requests = [], []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
