"""Per-session outbound request log from iron-proxy v0.50 per-request records.

The fixture lines are the exact shape the pinned v0.50.0 daemon writes to ``iron-proxy.log``
(captured from a live run): a ``msg: "request"`` line with an ``audit`` group, plus transform
traces Hermes must never surface.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_constants import get_hermes_home

_SECRET_PATH_TOKEN = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def _line(ts: str, remote: str, method: str, host: str, path: str, status: int, action: str = "allow") -> str:
    return json.dumps({
        "time": ts, "level": "INFO", "msg": "request",
        "audit": {"host": host, "method": method, "path": path, "remote_addr": remote, "sni": host,
                  "mode": "mitm", "action": action, "status_code": status, "duration_ms": 257.4},
        "request_transforms": [{"name": "secrets", "action": "allow", "duration_ms": 0.004,
                                "annotations": {"swapped": [{"secret": "OPENAI_API_KEY",
                                                             "locations": ["header:Authorization"]}]}}],
    })


@pytest.fixture
def client(_isolate_hermes_home):
    c = TestClient(web_server.app)
    c.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return c


def _write_egress_state(enabled: bool) -> None:
    home = get_hermes_home()
    (home / "config.yaml").write_text(f"proxy:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8")
    proxy = home / "proxy"
    proxy.mkdir(parents=True, exist_ok=True)
    # What the Docker backend appends when it starts an egress-routed sandbox.
    (proxy / "sandboxes.jsonl").write_text("".join(
        json.dumps({"ts": 1.0, "ip": ip, "task_id": task, "container": cid}) + "\n"
        for ip, task, cid in (("172.17.0.4", "sess_A", "edddc2d5da6f"), ("172.17.0.5", "sess_B", "0ce974ee2506"))
    ), encoding="utf-8")
    later = "2999-01-01T00:00:0{}.000000000-08:00"
    (proxy / "iron-proxy.log").write_text("\n".join([
        json.dumps({"time": later.format(0), "level": "INFO", "msg": "iron-proxy started"}),
        _line(later.format(1), "172.17.0.4:5001", "CONNECT", "httpbin.org:443", "", 200),
        _line(later.format(2), "172.17.0.4:5001", "GET", "httpbin.org", f"/key/{_SECRET_PATH_TOKEN}/x?api_key=QSECRET", 200),
        _line(later.format(3), "172.17.0.4:5002", "CONNECT", "evil.example.org:443", "", 403, action="reject"),
        _line(later.format(4), "172.17.0.5:6001", "GET", "httpbin.org", "/status/418", 418),
    ]) + "\n", encoding="utf-8")


def test_session_log_lists_only_that_sandboxs_requests_without_secrets(client):
    _write_egress_state(enabled=True)

    resp = client.get("/api/egress/requests", params={"session": "sess_A"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    rows = body["requests"]
    # Newest first; the allowed CONNECT is folded into the request it tunnelled, the denied one stays.
    assert [(r["method"], r["host"], r["status"]) for r in rows] == [
        ("CONNECT", "evil.example.org:443", 403), ("GET", "httpbin.org", 200)]
    assert set(rows[0]) == {"ts", "method", "host", "path", "status", "ms", "action"}
    text = json.dumps(body)
    for leaked in (_SECRET_PATH_TOKEN, "QSECRET", "api_key", "OPENAI_API_KEY", "Authorization", "/status/418"):
        assert leaked not in text
    assert rows[1]["path"].startswith("/key/") and rows[1]["path"].endswith("/x")


def test_egress_off_exposes_no_request_log(client):
    _write_egress_state(enabled=False)

    resp = client.get("/api/egress/requests", params={"session": "sess_A"})

    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "requests": []}
