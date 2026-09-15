"""Invariant tests for profile-safe MCP discovery and explicit connection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture
def discovery_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "profiles" / "target").mkdir(parents=True)
    (root / "profiles" / "source").mkdir(parents=True)
    return root


@pytest.fixture
def dashboard_client(discovery_home):
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


class _MCPHandler(BaseHTTPRequestHandler):
    valid = True

    def do_POST(self):  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path not in {"/mcp", "/sse"} or request.get("method") != "initialize":
            self.send_error(404)
            return
        result = (
            {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "mock-local", "version": "1"},
                "capabilities": {},
            }
            if self.valid
            else {"serverInfo": {"name": "not-mcp"}}
        )
        body = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


@pytest.fixture
def local_mcp_servers():
    servers = []
    threads = []
    for valid in (True, False):
        handler = type(f"Handler{valid}", (_MCPHandler,), {"valid": valid})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
    try:
        yield servers
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def test_discovery_hides_configs_and_only_accepts_verified_http(
    discovery_home, dashboard_client, local_mcp_servers, monkeypatch
):
    source_cfg = {
        "command": "uvx",
        "args": ["safe-package"],
        "env": {"LOG_LEVEL": "debug"},
        "tools": {"include": ["read"]},
    }
    _write_yaml(
        discovery_home / "profiles" / "source" / "config.yaml",
        {"mcp_servers": {"safe-source": source_cfg}},
    )
    inline_secret = "unit-test-inline-secret"
    (Path.home() / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "credentialed": {
                        "command": "node",
                        "args": ["server.js"],
                        "env": {"API_TOKEN": inline_secret},
                    }
                },
                "projects": {"/work": {"mcpServers": {"project-source": source_cfg}}},
            }
        ),
        encoding="utf-8",
    )

    import psutil

    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind="inet": [
            SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=s.server_port))
            for s in local_mcp_servers
        ],
    )

    response = dashboard_client.post("/api/mcp/discovery?profile=target")
    assert response.status_code == 200
    assert dashboard_client.post("/api/mcp/discovery", params={"profile": "../target"}).status_code == 404
    payload = response.json()
    serialized = json.dumps(payload)
    assert inline_secret not in serialized
    allowed_keys = {"id", "name", "source", "transport", "summary", "connectable", "reason"}
    assert all(set(candidate) <= allowed_keys for candidate in payload["candidates"])

    by_name = {candidate["name"]: candidate for candidate in payload["candidates"]}
    assert by_name["safe-source"]["connectable"] is True
    assert by_name["project-source"]["connectable"] is True
    assert by_name["credentialed"]["connectable"] is False
    assert "credential" in by_name["credentialed"]["reason"].lower()

    local_candidates = [c for c in payload["candidates"] if c["source"].startswith("Local HTTP endpoint ")]
    assert len(local_candidates) == 1
    assert local_candidates[0]["name"] == "mock-local"
    assert f":{local_mcp_servers[0].server_port}/mcp" in local_candidates[0]["source"]
    assert str(local_mcp_servers[1].server_port) not in serialized

    repeated = dashboard_client.post("/api/mcp/discovery?profile=target").json()["candidates"]
    repeated_safe = next(candidate for candidate in repeated if candidate["name"] == "safe-source")
    assert repeated_safe["id"] == by_name["safe-source"]["id"]


@pytest.mark.parametrize("outcome", ["success", "stale", "collision"])
def test_connect_revalidates_source_and_writes_only_target(
    discovery_home, dashboard_client, monkeypatch, outcome
):
    import psutil

    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [])
    original = {
        "command": "uvx",
        "args": ["package"],
        "env": {"LOG_LEVEL": "info"},
        "tools": {"exclude": ["delete*"]},
    }
    source_path = discovery_home / "profiles" / "source" / "config.yaml"
    target_path = discovery_home / "profiles" / "target" / "config.yaml"
    default_path = discovery_home / "config.yaml"
    _write_yaml(source_path, {"mcp_servers": {"shared": original}, "source_sibling": True})
    _write_yaml(target_path, {"target_sibling": {"kept": True}})
    _write_yaml(default_path, {"default_sibling": True})

    found = dashboard_client.post("/api/mcp/discovery?profile=target")
    assert found.status_code == 200
    candidate = next(c for c in found.json()["candidates"] if c["name"] == "shared")

    if outcome == "stale":
        changed = {**original, "args": ["different-package"]}
        _write_yaml(source_path, {"mcp_servers": {"shared": changed}, "source_sibling": True})
    elif outcome == "collision":
        _write_yaml(target_path, {"target_sibling": {"kept": True}, "mcp_servers": {"shared": {"command": "other"}}})

    connected = dashboard_client.post(
        "/api/mcp/discovery/connect?profile=target",
        json={"candidate_id": candidate["id"]},
    )
    if outcome == "success":
        assert connected.status_code == 200
        assert connected.json() == {"ok": True, "name": "shared"}
        target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        assert target["target_sibling"] == {"kept": True}
        assert target["mcp_servers"]["shared"] == original
    else:
        assert connected.status_code == 409
        target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        if outcome == "stale":
            assert "mcp_servers" not in target
        else:
            assert target["mcp_servers"]["shared"] == {"command": "other"}

    assert yaml.safe_load(default_path.read_text(encoding="utf-8")) == {"default_sibling": True}


def test_corrupt_profile_is_reported_without_aborting_other_profiles(discovery_home, monkeypatch):
    import psutil
    from hermes_cli.mcp_discovery import discover_candidates

    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [])
    (discovery_home / "profiles" / "source" / "config.yaml").write_text("mcp_servers: [\n", encoding="utf-8")
    _write_yaml(
        discovery_home / "profiles" / "healthy" / "config.yaml",
        {"mcp_servers": {"healthy-server": {"command": "uvx", "args": ["healthy"]}}},
    )

    result = discover_candidates("target")

    assert [candidate["name"] for candidate in result["candidates"]] == ["healthy-server"]
    assert any("YAMLError" in warning and "source" in warning for warning in result["warnings"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO behavior")
def test_fifo_config_is_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "config.yaml"
    os.mkfifo(fifo)
    script = (
        "from pathlib import Path; "
        "from hermes_cli.mcp_discovery import _read_mapping; "
        f"warnings=[]; assert _read_mapping(Path({str(fifo)!r}), yaml_file=True, warnings=warnings) is None; "
        "assert warnings"
    )

    completed = subprocess.run([sys.executable, "-c", script], timeout=2, check=False)

    assert completed.returncode == 0


@pytest.mark.parametrize(
    "result",
    [
        {"protocolVersion": "bogus", "serverInfo": {"name": "server", "version": "1"}, "capabilities": {}},
        {"protocolVersion": "2025-03-26", "serverInfo": {"name": "", "version": "1"}, "capabilities": {}},
        {"protocolVersion": "2025-03-26", "serverInfo": {"name": "server", "version": ""}, "capabilities": {}},
    ],
)
def test_initialize_requires_supported_version_and_complete_server_info(result):
    from hermes_cli.mcp_discovery import _parse_initialize_response

    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode()

    assert _parse_initialize_response(body, "application/json") is None


def test_sse_multiline_data_event_is_joined_before_parsing():
    from hermes_cli.mcp_discovery import _parse_initialize_response

    body = b"\n".join(
        [
            b"event: message",
            b'data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26",',
            b'data: "serverInfo":{"name":"joined","version":"1"},"capabilities":{}}}',
            b"",
            b"",
        ]
    )

    result = _parse_initialize_response(body, "text/event-stream")

    assert result is not None
    assert result["serverInfo"]["name"] == "joined"


def test_probe_rejects_nonliteral_loopback_without_opening(monkeypatch):
    import hermes_cli.mcp_discovery as discovery

    monkeypatch.setattr(
        discovery,
        "_probe_http",
        lambda *_args, **_kwargs: pytest.fail("non-literal endpoint reached the network"),
    )

    assert discovery._probe_endpoint("http://localhost:1234/mcp") is None
    assert discovery._probe_endpoint("http://example.com:1234/mcp") is None


def test_probe_ignores_proxy_and_closes_allocated_session(monkeypatch):
    seen: list[tuple[str, str | None, str | None]] = []

    class SessionHandler(_MCPHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "session-server", "version": "1"},
                        "capabilities": {},
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Mcp-Session-Id", "allocated-session")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):  # noqa: N802
            seen.append(
                (
                    self.path,
                    self.headers.get("Mcp-Session-Id"),
                    self.headers.get("MCP-Protocol-Version"),
                )
            )
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), SessionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/mcp"
        script = (
            "from hermes_cli.mcp_discovery import _probe_endpoint; "
            f"raise SystemExit(0 if _probe_endpoint({endpoint!r}) is not None else 1)"
        )
        completed = subprocess.run([sys.executable, "-c", script], timeout=3, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert completed.returncode == 0
    assert seen == [("/mcp", "allocated-session", "2025-03-26")]


@pytest.mark.parametrize("slow_phase", ["body", "headers"])
def test_slow_stream_probe_leaves_no_discovery_workers(monkeypatch, slow_phase):
    import psutil
    import hermes_cli.mcp_discovery as discovery

    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if slow_phase == "headers":
                self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
            for _ in range(100):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.02)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(discovery, "_HTTP_PROBE_TIMEOUT", 0.15)
    monkeypatch.setattr(discovery, "_HTTP_DISCOVERY_DEADLINE", 0.25)
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind="inet": [
            SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=server.server_port))
        ],
    )
    started = time.monotonic()
    try:
        assert discovery._http_candidates(set(), []) == []
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 0.75
    assert not [worker for worker in threading.enumerate() if worker.name.startswith("mcp-discovery")]


def test_connect_enables_disabled_source_without_dropping_fields(discovery_home, dashboard_client, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "net_connections", lambda kind="inet": [])
    original = {
        "command": "uvx",
        "args": ["package"],
        "enabled": False,
        "env": {"LOG_LEVEL": "info"},
        "tools": {"include": ["read"]},
    }
    source_path = discovery_home / "profiles" / "source" / "config.yaml"
    target_path = discovery_home / "profiles" / "target" / "config.yaml"
    _write_yaml(source_path, {"mcp_servers": {"disabled": original}})

    found = dashboard_client.post("/api/mcp/discovery?profile=target").json()
    candidate = next(item for item in found["candidates"] if item["name"] == "disabled")
    response = dashboard_client.post(
        "/api/mcp/discovery/connect?profile=target", json={"candidate_id": candidate["id"]}
    )

    assert response.status_code == 200
    saved = yaml.safe_load(target_path.read_text(encoding="utf-8"))["mcp_servers"]["disabled"]
    assert saved == {**original, "enabled": True}
