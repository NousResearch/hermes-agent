"""Exercise the recommended-models HTTP transport against gzip and plain servers."""
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli.models import fetch_nous_recommended_models


@pytest.mark.parametrize("compress", [False, True])
def test_recommendations_request_gzip_and_decode_the_servers_response(compress):
    payload = {"paidRecommendedModels": [{"modelName": "test-model"}] * 50}
    encoded = json.dumps(payload).encode("utf-8")
    requests = []
    transferred = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Accept-Encoding"))
            use_gzip = compress and "gzip" in self.headers.get("Accept-Encoding", "")
            body = gzip.compress(encoded) if use_gzip else encoded
            transferred.append(len(body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = fetch_nous_recommended_models(
            f"http://127.0.0.1:{server.server_port}", force_refresh=True)
        assert result == payload
        assert requests == ["gzip"]
        if compress:
            assert transferred[0] < len(encoded) / 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
