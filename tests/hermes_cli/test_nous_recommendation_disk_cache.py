"""Recommended models reuse their bounded disk cache across real client processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def portal():
    requests = []
    state = {"version": 1, "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = json.dumps({"paidRecommendedModels": [{"modelName": f"model-{state['version']}"}]}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fetch_in_new_process(home, base, *, force=False):
    result = subprocess.run(
        [sys.executable, "-c", "from hermes_cli.models import fetch_nous_recommended_models; "
         "import json, sys; print(json.dumps(fetch_nous_recommended_models(sys.argv[1], force_refresh=sys.argv[2] == 'True')))",
         base, str(force)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(home)},
        capture_output=True, text=True, check=True, timeout=30,
    )
    return json.loads(result.stdout)


def test_process_restarts_reuse_fresh_disk_but_force_and_expiry_fetch(tmp_path, portal):
    base, requests, state = portal
    home = tmp_path / "client"
    first = fetch_in_new_process(home, base)
    assert fetch_in_new_process(home, base) == first
    assert len(requests) == 1

    state["version"] = 2
    second = fetch_in_new_process(home, base, force=True)
    assert second != first
    assert len(requests) == 2

    path = home / "cache" / "nous_recommended_cache.json"
    disk = json.loads(path.read_text(encoding="utf-8"))
    disk[base]["ts"] = time.time() - 3600
    path.write_text(json.dumps(disk), encoding="utf-8")
    state["version"] = 3
    assert fetch_in_new_process(home, base) != second
    assert len(requests) == 3

    # A failed refresh retains the last good payload without making it fresh on disk.
    disk = json.loads(path.read_text(encoding="utf-8"))
    disk[base]["ts"] = time.time() - 3600
    path.write_text(json.dumps(disk), encoding="utf-8")
    state["status"] = 503
    assert fetch_in_new_process(home, base) == disk[base]["data"]
    assert json.loads(path.read_text(encoding="utf-8"))[base]["ts"] == disk[base]["ts"]

    # Bad local cache bytes/metadata must never stop a healthy API refresh.
    state["status"] = 200
    path.write_bytes(b"\xff")
    assert fetch_in_new_process(home, base) == disk[base]["data"]
    for timestamp in [None, True, "invalid", float("nan"), time.time() + 3600, 10 ** 400]:
        disk[base]["ts"] = timestamp
        path.write_text(json.dumps(disk), encoding="utf-8")
        before = len(requests)
        assert fetch_in_new_process(home, base) == disk[base]["data"]
        assert len(requests) == before + 1


def test_disk_cache_is_scoped_to_home_and_portal(tmp_path, portal):
    base, requests, state = portal
    a, b = tmp_path / "a", tmp_path / "b"
    first = fetch_in_new_process(a, base)
    state["version"] = 2
    assert fetch_in_new_process(b, base) != first
    assert fetch_in_new_process(a, base) == first
    assert len(requests) == 2
    assert fetch_in_new_process(a, base + "/staging") != first
    assert len(requests) == 3

    # The same process can switch profiles under the multiplex gateway too.
    from hermes_cli.models import fetch_nous_recommended_models, _nous_recommended_cache
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    _nous_recommended_cache.clear()
    for home, expected in [(a, first), (b, {"paidRecommendedModels": [{"modelName": "model-2"}]}), (a, first)]:
        token = set_hermes_home_override(home)
        try:
            assert fetch_nous_recommended_models(base) == expected
        finally:
            reset_hermes_home_override(token)
    assert len(requests) == 3
    _nous_recommended_cache.clear()
