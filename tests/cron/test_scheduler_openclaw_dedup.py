import asyncio
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cron.scheduler as scheduler
import gateway.config as gateway_config
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.openclaw import send_openclaw_direct
from gateway.platforms.base import SendResult


class _DedupReceiverHandler(BaseHTTPRequestHandler):
    secret = "test-secret"
    queue = []
    seen = set()

    def do_GET(self):
        if self.path != "/queue":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"queue": self.queue}).encode("utf-8")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        expected = "sha256=" + hmac.new(self.secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        if self.headers.get("x-cron-signature") != expected:
            self.send_response(401)
            self.end_headers()
            return
        payload = json.loads(body.decode("utf-8"))
        key = f"{payload['profile']}/{payload['job']}/{payload['run_id']}"
        if key not in self.seen:
            self.seen.add(key)
            self.queue.append({"key": key, "payload": payload})
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b'{"accepted":true}')

    def log_message(self, *_args):
        return


class _TimeoutAfterPostAdapter:
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        result = await send_openclaw_direct(
            PlatformConfig(extra={"secret": "test-secret"}),
            chat_id,
            content,
            metadata=metadata,
        )
        if result.get("error"):
            return SendResult(success=False, error=result["error"], raw_response=result)
        raise TimeoutError("post landed but live adapter timed out locally")


def _start_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(timeout=5)
    return loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _start_receiver():
    _DedupReceiverHandler.queue = []
    _DedupReceiverHandler.seen = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DedupReceiverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/cron"


def _openclaw_config():
    return GatewayConfig(
        platforms={
            Platform.OPENCLAW: PlatformConfig(
                enabled=True,
                extra={"secret": "test-secret"},
            )
        }
    )


def test_openclaw_metadata_built_once_before_live_and_standalone(monkeypatch):
    calls = []

    def metadata(job, content):
        calls.append((job["id"], content))
        return {
            "profile": "ratatosk",
            "job": job["id"],
            "run_id": f"run-{len(calls)}",
            "severity": "info",
        }

    monkeypatch.setattr(gateway_config, "load_gateway_config", _openclaw_config)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(scheduler, "_openclaw_delivery_metadata", metadata)

    loop, thread = _start_loop()
    server, url = _start_receiver()
    try:
        job = {"id": "oc202-dedup", "name": "OC202 Dedup", "deliver": f"openclaw:{url}"}
        error = scheduler._deliver_result(
            job,
            "cron output",
            adapters={Platform.OPENCLAW: _TimeoutAfterPostAdapter()},
            loop=loop,
        )
    finally:
        server.shutdown()
        server.server_close()
        _stop_loop(loop, thread)

    assert error is None
    assert len(calls) == 1


def test_openclaw_run_id_stable_across_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "ratatosk")
    monkeypatch.setattr(gateway_config, "load_gateway_config", _openclaw_config)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    timestamps = iter([1000.0, 1001.0])
    monkeypatch.setattr(scheduler.time, "time", lambda: next(timestamps, 1001.0))

    loop, thread = _start_loop()
    server, url = _start_receiver()
    try:
        job = {"id": "oc202-dedup", "name": "OC202 Dedup", "deliver": f"openclaw:{url}"}
        error = scheduler._deliver_result(
            job,
            "cron output",
            adapters={Platform.OPENCLAW: _TimeoutAfterPostAdapter()},
            loop=loop,
        )
        queue_url = url.rsplit("/", 1)[0] + "/queue"
        import urllib.request

        with urllib.request.urlopen(queue_url, timeout=2) as response:
            queue = json.loads(response.read().decode("utf-8"))["queue"]
    finally:
        server.shutdown()
        server.server_close()
        _stop_loop(loop, thread)

    assert error is None
    assert len(queue) == 1
    assert queue[0]["payload"]["run_id"] == "oc202-dedup-1000000"
