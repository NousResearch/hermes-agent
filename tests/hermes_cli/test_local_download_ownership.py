"""A second process must not restore a live model download as interrupted."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from threading import Event

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._fixtures import threaded_server
from tests.waiting import wait_until


@pytest.mark.parametrize("outcome", ["complete", "terminated"])
def test_restore_skips_live_plan_and_reconciles_owner_exit(client, outcome):
    from hermes_cli.web_routers import local_models as lm

    block = 1 << 20
    payload = b"w" * (12 * block)
    release = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"live-model"')
            self.end_headers()
            if (start, end) == (0, 0):
                self.wfile.write(payload[:1])
                return
            self.wfile.write(payload[:8 * block])
            self.wfile.flush()
            if release.wait(20) and outcome == "complete":
                self.wfile.write(payload[8 * block:])

        def log_message(self, *args):
            pass

    dest = lm.bootstrap.models_dir() / "live.gguf"
    with threaded_server(Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/model"
        code = (
            "import sys\nfrom pathlib import Path\nfrom functools import partial\n"
            "from hermes_cli.web_routers import local_models as lm\n"
            "lm.Download = partial(lm.Download, connections=1)\n"
            "job = lm._job('model-download', 'live model', model_id='live')\n"
            "lm._download_job(job, [(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))])\n"
        )
        child = subprocess.Popen([sys.executable, "-c", code, url, str(dest), str(len(payload))],
                                 cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            side = lm.download_plans._root().parent / (hashlib.sha256(url.encode()).hexdigest() + ".ranges")
            wait_until(side.is_file, timeout=10)
            lm._restore_download_plans()
            assert client.get("/api/local-models/jobs").json()["jobs"] == []
            if outcome == "complete":
                release.set()
                _, error = child.communicate(timeout=5)
                assert child.returncode == 0, error.decode()
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
            release.set()

    lm._restore_download_plans()
    restored = client.get("/api/local-models/jobs").json()["jobs"]
    if outcome == "complete":
        assert restored == [] and dest.read_bytes() == payload
        return
    assert len(restored) == 1
    assert restored[0]["status"] == "paused" and restored[0]["can_resume"]
    assert restored[0]["done_bytes"] == 8 * block
    assert not dest.exists()
