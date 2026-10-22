"""A second process must not restore a live model download as interrupted."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from threading import Event, Lock

import pytest

from tests.hermes_cli.test_local_download_jobs import client as client  # noqa: PLC0414 -- pytest fixture re-export
from tests.pm._fixtures import threaded_server
from tests.waiting import wait_until


def test_worker_does_not_enter_body_when_ownership_is_cancelled(client):
    from hermes_cli.web_routers import local_models as lm

    @contextmanager
    def cancelled():
        yield False

    entered = Event()
    job = lm._job("quickstart", "cancelled setup")
    lm._spawn_job(job, "test-cancelled-owner", entered.set, resumable=True, ownership=cancelled)
    assert wait_until(lambda: job["status"] != "running", timeout=5)
    assert not entered.is_set()
    assert job["status"] == "paused" and lm._job_view(job)["can_resume"]


def test_setup_reports_waiting_and_pause_cancels_ownership(client, monkeypatch):
    from hermes_cli.web_routers import local_models as lm, local_models_setup

    entered, quickstart = Event(), Lock()
    quickstart.acquire()
    monkeypatch.setattr(lm, "_QUICKSTART_LOCK", quickstart)
    monkeypatch.setattr(local_models_setup, "_run_setup", lambda job: entered.set())
    job = lm._job("quickstart", "waiting setup", model_id="pending")
    job["setup"] = {"model_id": "pending", "profile": "default", "backend": "cpu", "variant": "pending"}
    job["plan_key"] = lm.download_plans.plan_key([], job["setup"])
    with lm.download_plans.plan_lock(job["plan_key"]):
        local_models_setup._spawn(job)
        try:
            assert wait_until(lambda: job["phase"] == "waiting", timeout=5)
        finally:
            lm._RUNNING[job["job_id"]]["pause"].set()
            assert wait_until(lambda: job["status"] != "running", timeout=5)
        assert job["status"] == "paused" and not entered.is_set()
        assert not lm.download_plans.has_plan(job["plan_key"])
    assert lm._RUNNING[job["job_id"]]["discard"]()
    assert not quickstart.locked()


def test_worker_checkpoints_before_releasing_plan_ownership(client, monkeypatch):
    from urllib.error import HTTPError
    from hermes_cli.web_routers import local_models as lm

    plan = [("https://example.invalid/model", lm.bootstrap.models_dir() / "pending.gguf", 4)]
    job, _ = lm._plan_job("pending model", "pending", plan)
    checkpointing, release = Event(), Event()
    original = lm.download_plans.checkpoint

    def checkpoint(job):
        checkpointing.set()
        assert release.wait(5)
        original(job)

    def unavailable(download, progress=None):
        target = download.sources[0].url
        raise lm.DownloadTransportError(target, HTTPError(target, 503, "Unavailable", {}, None))

    monkeypatch.setattr(lm.Download, "run", unavailable)
    monkeypatch.setattr(lm.download_plans, "checkpoint", checkpoint)
    lm._spawn_job(job, "test-checkpoint-owner", lambda: lm._run_download_plan(job, plan, job["target"]), resumable=True)
    try:
        assert checkpointing.wait(5)
        discarded_during_checkpoint = lm.download_plans.discard_plan(job["plan_key"])
    finally:
        release.set()
        assert wait_until(lambda: job["status"] != "running", timeout=5)
    assert not discarded_during_checkpoint
    assert job["status"] == "paused"
    old_resume = lm._RUNNING[job["job_id"]]["resume"]
    assert lm.download_plans.discard_plan(job["plan_key"])
    # A second process can discard between this process's reconciliation ticks.
    assert old_resume()
    assert wait_until(lambda: job["status"] != "running", timeout=5)
    assert job["status"] == "error" and "discarded" in job["error"]
    assert not lm.download_plans.has_plan(job["plan_key"])
    lm._restore_download_plans()
    assert not lm.download_plans.has_plan(job["plan_key"])


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
            "lm._run_download_plan(job, [(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))], 'live model')\n"
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
