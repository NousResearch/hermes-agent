"""#134335: a token GitHub refuses with 403 (suspended/restricted account, still printed by
`gh auth token`) falls back to anonymous like a 401 does; a spent rate limit does not.

Real HTTP server on loopback, real urllib, real credential ladder (GITHUB_TOKEN rung).
"""

import http.server
import threading
import urllib.error

import pytest

from hermes_cli import source_check


class _Api(http.server.BaseHTTPRequestHandler):
    spent = False

    def do_GET(self):
        if self.headers.get("Authorization"):
            self.send_response(403)
            self.send_header("x-ratelimit-remaining", "0" if self.spent else "4999")
            self.end_headers()
            self.wfile.write(b'{"message": "Forbidden"}')
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"sha": "anonymous-ok"}')

    def log_message(self, *args):
        pass


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "stale-token-of-a-suspended-account")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/repos/o/r/commits/main"
    server.shutdown()


def test_a_refused_token_falls_back_to_anonymous(api, monkeypatch):
    monkeypatch.setattr(_Api, "spent", False)
    assert source_check._request(api) == '{"sha": "anonymous-ok"}'


def test_a_spent_rate_limit_still_fails(api, monkeypatch):
    monkeypatch.setattr(_Api, "spent", True)
    with pytest.raises(urllib.error.HTTPError) as caught:
        source_check._request(api)
    assert caught.value.code == 403 and caught.value.hermes_authenticated is True
