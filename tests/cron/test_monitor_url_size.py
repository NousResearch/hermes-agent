"""A bounded URL read must not turn a partial response into a monitor baseline."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from cron import jobs, monitor
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _source(body):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body[0])))
            self.end_headers()
            self.wfile.write(body[0])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/status"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_oversized_url_preserves_last_complete_observation(tmp_path):
    token = set_hermes_home_override(tmp_path / "home")
    body = [b"baseline"]
    try:
        with _source(body) as url:
            job = jobs.create_job(prompt="Report changes", schedule="every 1h", monitor_url=url)
            assert monitor.check_monitor(job).changed
            baseline = jobs.get_job(job["id"])["monitor_state"]
            snapshot = monitor._snapshot_path(job["id"]).read_bytes()

            body[0] = b"x" * (monitor.MAX_URL_BYTES + 1)
            outcome = monitor.check_monitor(jobs.get_job(job["id"]))
            assert not outcome.ok
            assert "exceeds" in outcome.error
            assert jobs.get_job(job["id"])["monitor_state"] == baseline
            assert monitor._snapshot_path(job["id"]).read_bytes() == snapshot

            body[0] = b"baseline"
            assert not monitor.check_monitor(jobs.get_job(job["id"])).changed
            body[0] = b"new complete observation"
            assert monitor.check_monitor(jobs.get_job(job["id"])).changed
    finally:
        reset_hermes_home_override(token)


def test_url_at_byte_limit_is_complete():
    body = [b"x" * monitor.MAX_URL_BYTES]
    with _source(body) as url:
        ok, output = monitor._fetch_monitor_url(url)
    assert ok
    assert output.encode("utf-8") == body[0]
