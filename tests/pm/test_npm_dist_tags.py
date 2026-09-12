"""npm update discovery reads the tag endpoint, not the package history."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from pm.packages import AgentBrowser, Npm
from pm.update import npm_dist_tags, resolve_package


def test_tag_endpoint_drives_package_updates_and_preserves_escaped_names(monkeypatch):
    from hermes_cli import urllib_security

    tags = {
        "/-/package/npm/dist-tags": {"latest": "2.3.4", "next": "3.0.0-beta.1"},
        "/-/package/agent-browser/dist-tags": {"latest": "4.5.6"},
        "/-/package/@scope%2Ftool/dist-tags": {},
    }
    requests = []

    class Registry(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("User-Agent"), self.headers.get("Authorization")))
            if len(requests) == 1:
                self.send_error(503)
                return
            if self.path not in tags:
                self.send_error(404)
                return
            body = json.dumps(tags[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    real_open = urllib_security.open_credentialed_url
    with ThreadingHTTPServer(("127.0.0.1", 0), Registry) as server:
        def loopback(request, **kwargs):
            # Route transport only; retain the production JSON reader, headers,
            # retry policy, and credential-safe opener.
            origin = "https://registry.npmjs.org"
            assert request.full_url.startswith(origin + "/")
            request.full_url = request.full_url.replace(origin, f"http://127.0.0.1:{server.server_port}", 1)
            return real_open(request, **kwargs)

        monkeypatch.setattr(urllib_security, "open_credentialed_url", loopback)
        monkeypatch.setenv("GH_TOKEN", "not-for-npm")
        monkeypatch.setenv("HF_TOKEN", "not-for-npm-either")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for package, version in [(Npm(), "2.3.4"), (AgentBrowser(), "4.5.6")]:
                decision = resolve_package(package, ["linux-x64"], locked="1.0.0")
                assert decision.version == version
                assert decision.changed
            assert npm_dist_tags("@scope%2Ftool") == {}
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert [path for path, _, _ in requests] == [
        "/-/package/npm/dist-tags", "/-/package/npm/dist-tags",
        "/-/package/agent-browser/dist-tags", "/-/package/@scope%2Ftool/dist-tags",
    ]
    assert all(agent == "hermes-pm" and auth is None for _, agent, auth in requests)
