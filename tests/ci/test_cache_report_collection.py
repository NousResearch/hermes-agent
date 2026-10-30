"""The API token must not reach a redirected log download."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from scripts.ci import timings_report


def test_log_redirect_does_not_forward_api_token(monkeypatch) -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            redirect = self.path.endswith("/logs")
            self.send_response(302 if redirect else 200)
            if redirect:
                self.send_header("Location", "/blob")
            self.end_headers()
            self.wfile.write(b"log")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setattr(timings_report, "API_BASE", f"http://127.0.0.1:{server.server_port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert timings_report._job_log(1, "fixture/cache", "fixture-token") == "log"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert requests == [
        ("/repos/fixture/cache/actions/jobs/1/logs", "Bearer fixture-token"),
        ("/blob", None),
    ]
