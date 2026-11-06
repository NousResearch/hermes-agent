"""An active model-download job survives an outage without another user action."""
from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler
from itertools import pairwise
import re
from threading import Event
from types import SimpleNamespace
import time

import pytest

from pm import downloader
from pm.downloader import Download
from tests.pm._fixtures import threaded_server
from tests.waiting import wait_until


@pytest.mark.parametrize("outage_phase", ["probe", "transfer"])
@pytest.mark.parametrize("required", [True, False], ids=["required", "optional"])
def test_model_job_completes_when_connectivity_returns(tmp_path, monkeypatch, outage_phase, required):
    from hermes_cli.web_routers import local_models

    block = 1 << 20
    payload = b"model bytes" * (2 * block // 11) + b"!"
    failures = []
    transfers = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers["Range"]).groups())
            probe = (start, end) == (0, 0)
            if (probe == (outage_phase == "probe") and len(failures) < 6
                    and (probe or transfers)):
                failures.append(clock[0])
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"same-model"')
            self.end_headers()
            stop = end + 1
            if not probe:
                transfers.append(start)
                if outage_phase == "transfer" and len(transfers) == 1:
                    stop = block
            self.wfile.write(payload[start:stop])
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    destination = tmp_path / "model.gguf"
    finished = Event()
    clock = [time.monotonic()]

    def wait_retry(self, delay):
        clock[0] += delay

    monkeypatch.setattr(local_models, "Download", partial(Download, connections=1, partials_dir=tmp_path / "parts"))
    monkeypatch.setattr(local_models, "_refresh_runtime", lambda _: None)
    monkeypatch.setattr(downloader, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    monkeypatch.setattr(Download, "_wait_retry", wait_retry)
    job = local_models._job("download", "outage model")
    try:
        with threaded_server(Handler) as server:
            plan = [(f"http://127.0.0.1:{server.server_port}/model", destination, len(payload))]
            action = local_models._download_job if required else local_models._download_optional
            local_models._spawn_job(job, "test-model-outage", lambda: action(job, plan),
                                    resumable=True, on_exit=finished.set)
            assert finished.wait(10), "model job did not finish after connectivity returned"
            wait_until(lambda: job["status"] != "running", timeout=5)
        assert job["status"] == "done", job["error"]
        assert len(failures) == (6 if required else 4)
        assert all(1 <= later - earlier <= 30 for earlier, later in pairwise(failures))
        if required:
            assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(payload).digest()
            if outage_phase == "transfer":
                assert transfers == [0, block]
        else:
            assert not destination.exists()
    finally:
        running = local_models._RUNNING.get(job["job_id"])
        if running is not None:
            running["pause"].set()
            wait_until(lambda: job["status"] != "running", timeout=5)
        local_models._JOBS.pop(job["job_id"], None)
        local_models._RUNNING.pop(job["job_id"], None)
