"""Outbound paths must be readable by a sidecar without exposing Hermes' private filesystem."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from gateway.config import Platform, load_gateway_config
from gateway.platforms.signal import SignalAdapter
from gateway.platforms.signal_rate_limit import _reset_scheduler
from tools.send_message_tool import _send_to_platform


@pytest.fixture
def sidecar(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    received = []
    reject = [False]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            paths = [Path(p) for p in request["params"].get("attachments", [])]
            # signal-cli can only open paths in its shared mount.
            readable = all(p.is_relative_to(shared) and p.is_file() for p in paths)
            received.append((paths, [p.read_bytes() for p in paths] if readable else None))
            result = {"error": {"code": -1, "message": "attachment unavailable"}} if reject[0] or not readable else {
                "result": {"timestamp": 123}}
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"], **result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _reset_scheduler()
    try:
        yield shared, received, reject, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        _reset_scheduler()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["gateway", "batch", "standalone"])
@pytest.mark.parametrize("reject", [False, True])
async def test_outbound_staging_through_http(route, reject, sidecar, tmp_path, monkeypatch):
    shared, received, rejection, url = sidecar
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    rejection[0] = reject
    source = tmp_path / "private" / "chart.png"
    source.parent.mkdir()
    source.write_bytes(b"private attachment")
    existing = shared / "chart.png"
    existing.write_bytes(b"already shared attachment")
    extra = {"http_url": url, "account": "+15551234567", "attachment_staging_dir": str(shared)}
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        json.dumps({"platforms": {"signal": {"enabled": True, "extra": extra}}}), encoding="utf-8")
    config = load_gateway_config().platforms[Platform.SIGNAL]
    adapter = SignalAdapter(config)
    async with httpx.AsyncClient() as client:
        adapter.client = client
        if route == "gateway":
            result = await adapter.send_image_file("+15557654321", str(source))
            success = result.success
        elif route == "batch":
            result = await adapter.send_multiple_images("+15557654321", [(source.as_uri(), ""), (existing.as_uri(), "")])
            success = result.success
        else:
            result = await _send_to_platform(Platform.SIGNAL, config, "+15557654321", "",
                                             media_files=[(str(source), False), (str(existing), False)])
            success = result.get("success", False)
    assert success is not reject
    attachment_calls = [(paths, contents) for paths, contents in received if paths]
    assert attachment_calls
    for paths, contents in attachment_calls:
        assert contents == ([b"private attachment"] if route == "gateway" else
                            [b"private attachment", b"already shared attachment"])
        assert all(p.parent == shared for p in paths)
        assert all(not p.exists() for p in paths if p != existing)
    assert source.read_bytes() == b"private attachment"
    assert existing.read_bytes() == b"already shared attachment"
    assert list(shared.iterdir()) == [existing]


def test_partial_staging_failure_removes_only_temporary_copies(tmp_path):
    from gateway.platforms.signal_attachments import staged_signal_attachments

    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    shared = tmp_path / "shared"
    with pytest.raises(FileNotFoundError):
        with staged_signal_attachments([str(source), str(tmp_path / "removed.pdf")], shared):
            pytest.fail("An incomplete batch must not be sent")
    assert source.read_bytes() == b"report"
    assert list(shared.iterdir()) == []
