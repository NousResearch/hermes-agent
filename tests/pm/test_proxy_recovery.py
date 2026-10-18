"""A CONNECT failure must retain its HTTP policy through urllib's OSError wrapper."""
import http.server
import threading
import urllib.error
import urllib.request

import pytest

from pm.downloader import DownloadTransportError
from pm.network import retry_network


@pytest.mark.parametrize(('status', 'recoverable', 'attempts'), [(503, True, 4), (407, False, 1)])
def test_proxy_status_retains_recovery_policy(status, recoverable, attempts):
    calls = []

    class Proxy(http.server.BaseHTTPRequestHandler):
        def do_CONNECT(self):
            calls.append(self.path)
            self.send_response(status)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = 'https://recovery.invalid/model.gguf'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({
        'https': f'http://127.0.0.1:{server.server_port}'}))
    try:
        with pytest.raises(urllib.error.URLError) as caught:
            retry_network(lambda: opener.open(url), wait=lambda _: None)
        error = DownloadTransportError(url, caught.value)
        assert error.status == status
        assert error.recoverable is recoverable
        assert len(calls) == attempts
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
