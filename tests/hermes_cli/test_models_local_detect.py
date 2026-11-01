"""Local model server detection: real loopback HTTP servers on ephemeral ports stand in for
Ollama / LM Studio / vLLM; the pickers, the switch path and ``hermes model`` read them for real."""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import models_local_detect as detect

pytestmark = pytest.mark.real_local_model_probe


class _Server:
    """One loopback server. ``routes`` maps a GET path to ``(status, body, headers)``."""

    def __init__(self, routes: dict, *, hang: bool = False):
        self.requests: list[tuple[str, str | None]] = []
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Authorization")))
                if hang:
                    outer.release.wait(10)
                status, body, headers = routes.get(self.path, (404, b"{}", {}))
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def _models(ids, owner):
    return (200, json.dumps({"object": "list", "data": [{"id": i, "owned_by": owner} for i in ids]}).encode(), {})


@pytest.fixture
def servers(monkeypatch):
    started: list[_Server] = []

    def start(kind: str, routes: dict, **kw) -> _Server:
        srv = _Server(routes, **kw)
        started.append(srv)
        label = {"ollama": "Ollama", "lmstudio": "LM Studio", "vllm": "vLLM"}[kind]
        known = [k for k in detect.KNOWN_SERVERS if k.kind != kind] + [detect._Kind(kind, label, srv.port)]
        monkeypatch.setattr(detect, "KNOWN_SERVERS", tuple(known))
        monkeypatch.setattr(detect, "_KIND_BY_NAME", {k.kind: k for k in known})
        return srv

    # Point every default port at a closed one first, so nothing on the host can answer.
    closed = _Server({})
    closed_port = closed.port
    closed.close()
    blank = tuple(detect._Kind(k.kind, k.label, closed_port) for k in detect.KNOWN_SERVERS)
    monkeypatch.setattr(detect, "KNOWN_SERVERS", blank)
    monkeypatch.setattr(detect, "_KIND_BY_NAME", {k.kind: k for k in blank})
    monkeypatch.setattr(detect, "_cache", None)
    yield start
    for srv in started:
        srv.close()


def test_detects_servers_from_their_own_model_lists_without_sending_a_key(servers):
    ollama = servers("ollama", {"/v1/models": _models(["llama3.2:3b", "qwen2.5:7b"], "library")})
    vllm = servers("vllm", {"/v1/models": _models(["meta-llama/Llama-3.1-8B-Instruct"], "vllm")})

    found = {s.kind: s for s in detect.detect_local_servers(refresh=True)}

    assert found["ollama"].models == ("llama3.2:3b", "qwen2.5:7b")
    assert found["ollama"].base_url == f"http://127.0.0.1:{ollama.port}/v1"
    assert found["ollama"].row_label == "Detected: Ollama (2 models)"
    assert found["vllm"].models == ("meta-llama/Llama-3.1-8B-Instruct",)
    assert "lmstudio" not in found
    assert all(auth is None for srv in (ollama, vllm) for _path, auth in srv.requests)


def test_ollama_native_tag_list_is_the_fallback(servers):
    servers("ollama", {"/api/tags": (200, json.dumps({"models": [{"name": "mistral:latest"}]}).encode(), {})})
    (server,) = detect.detect_local_servers(refresh=True)
    assert (server.kind, server.models) == ("ollama", ("mistral:latest",))


@pytest.mark.parametrize("routes", [
    {"/v1/models": (200, b"<html>dev server</html>", {})},
    {"/v1/models": (200, json.dumps({"data": []}).encode(), {})},
    {"/v1/models": (401, b"{}", {})},
    {"/v1/models": (302, b"", {"Location": "/elsewhere"}),
     "/elsewhere": _models(["redirected"], "vllm")},
    {"/v1/models": (200, b'{"data": [' + b'{"id": "x"},' * 200_000 + b'{"id": "y"}]}', {})},
])
def test_a_port_that_is_not_a_model_server_yields_no_row(servers, routes):
    servers("vllm", routes)
    assert detect.detect_local_servers(refresh=True) == []


def test_a_hanging_port_cannot_stall_the_scan(servers):
    servers("vllm", {"/v1/models": _models(["slow"], "vllm")}, hang=True)
    servers("ollama", {"/v1/models": _models(["fast"], "library")})
    started = time.monotonic()
    found = detect.detect_local_servers(refresh=True, timeout=0.3)
    assert time.monotonic() - started < 2
    assert [s.kind for s in found] == ["ollama"]


def test_only_an_opened_picker_probes_and_other_reads_reuse_its_rows(servers):
    from hermes_cli.inventory import ConfigContext, build_models_payload

    srv = servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    ctx = ConfigContext("openrouter", "x", "", {}, [])

    def detected(mode):
        rows = build_models_payload(ctx, for_picker=True, local_detection=mode)["providers"]
        return [r for r in rows if r.get("source") == "detected"]

    assert detected("off") == [] and detected("cached") == []
    assert srv.requests == []
    (row,) = detected("probe")
    assert (row["slug"], row["name"], row["models"]) == ("local-ollama", "Detected: Ollama", ["llama3.2:3b"])
    probes = len(srv.requests)
    assert [r["slug"] for r in detected("cached")] == ["local-ollama"]
    assert len(srv.requests) == probes


def test_saved_server_is_not_offered_again(servers):
    from hermes_cli.inventory import ConfigContext, build_models_payload

    srv = servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    saved = {"my-ollama": {"name": "Box", "base_url": f"http://localhost:{srv.port}/v1"}}
    ctx = ConfigContext("openrouter", "x", "", saved, [])
    rows = build_models_payload(ctx, for_picker=True, local_detection="probe")["providers"]
    assert not [r for r in rows if r.get("source") == "detected"]


def test_picking_a_detected_row_saves_a_provider_and_switches(servers):
    from hermes_cli.config import load_config
    from hermes_cli.model_switch import switch_model

    srv = servers("ollama", {"/v1/models": _models(["llama3.2:3b", "qwen2.5:7b"], "library")})
    detect.detect_local_servers(refresh=True)

    result = switch_model("qwen2.5:7b", "openrouter", "x", explicit_provider="local-ollama",
                          user_providers={}, custom_providers=[])

    assert result.success, result.error_message
    assert (result.target_provider, result.new_model) == ("local-ollama", "qwen2.5:7b")
    assert result.base_url == f"http://127.0.0.1:{srv.port}/v1"
    entry = load_config()["providers"]["local-ollama"]
    assert entry == {"name": "Ollama (local)", "base_url": f"http://127.0.0.1:{srv.port}/v1"}


def test_unknown_local_slug_writes_nothing(servers):
    from hermes_cli.config import load_config
    from hermes_cli.model_switch import switch_model

    result = switch_model("m", "openrouter", "x", explicit_provider="local-ollama",
                          user_providers={}, custom_providers=[])
    assert not result.success
    assert "local-ollama" not in (load_config().get("providers") or {})


def test_hermes_model_offers_detected_row_and_pick_saves_named_provider(servers):
    from hermes_cli.config import load_config
    from hermes_cli.main_provider_setup import (
        _adopt_detected_pick, _build_provider_picker_rows, _named_custom_provider_map)

    srv = servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})
    rows, _ = _build_provider_picker_rows(load_config(), "", {}, {})
    (key, label, _members), = [r for r in rows if r[0].startswith("detected:")]
    assert label.startswith("Detected: LM Studio (1 model)")
    assert key == "detected:local-lmstudio"

    slug = _adopt_detected_pick(key)

    assert slug == "custom:local-lmstudio"
    assert _named_custom_provider_map(load_config())[slug]["base_url"] == f"http://127.0.0.1:{srv.port}/v1"
    assert _adopt_detected_pick("openrouter") == "openrouter"


def test_model_options_rpc_accepts_the_picker_scan_flag(servers):
    """Desktop sends ``detect_local`` from an opened picker; the RPC contract must accept it."""
    import tui_gateway.methods_complete  # registers model.options
    from tui_gateway.server import handle_request

    servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    reply = handle_request({"jsonrpc": "2.0", "id": 1, "method": "model.options",
                            "params": {"detect_local": True, "explicit_only": True}})

    assert "error" not in reply, reply
    assert "local-ollama" in [p["slug"] for p in reply["result"]["providers"]]


def test_a_new_chat_pinned_to_a_detected_row_builds_on_that_server(servers, monkeypatch):
    """Desktop pins a composer pick on the new chat (``session.create`` model + provider) with no
    ``/model`` switch; the session must still resolve the picked server, not fail on an unknown
    provider at the first turn."""
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tui_gateway import server

    for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_enable_gateway_prompts"):
        monkeypatch.setattr(server, name, lambda *a: None)
    monkeypatch.setattr(server, "_sessions", {})
    srv = servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})

    reply = server._methods["session.create"]("r1", {"cols": 80, "model": "qwen/qwen3-4b", "provider": "local-lmstudio"})

    assert "error" not in reply, reply
    assert load_config()["providers"]["local-lmstudio"]["base_url"] == f"http://127.0.0.1:{srv.port}/v1"
    runtime = resolve_runtime_provider(requested="local-lmstudio", target_model="qwen/qwen3-4b")
    assert runtime["base_url"] == f"http://127.0.0.1:{srv.port}/v1"


@pytest.mark.asyncio
async def test_the_gateway_model_picker_scans_when_an_opened_picker_asks(servers, monkeypatch):
    """The TUI (and a Desktop attached to the gateway) read ``model.options`` from the gateway's
    session authority, which refused any flag it did not list: the picker showed an error instead."""
    from types import SimpleNamespace

    from gateway import session_config
    from hermes_constants import get_hermes_home

    servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(get_hermes_home()), agent=lambda ref: None))
    monkeypatch.setattr(session_config, "_authorize", lambda connection, ref, params: None)

    result = await session_config.model_options(
        connection, SimpleNamespace(session_id=""), {"detect_local": True, "include_unconfigured": True})

    assert "local-ollama" in [p["slug"] for p in result["providers"]]


@pytest.mark.asyncio
async def test_a_gateway_session_picking_a_detected_row_runs_on_that_server(servers, tmp_path):
    """The TUI's ``/model`` goes through the gateway's session mutation, which freezes the picked
    provider entry from a profile read taken before the switch saved it: the pick succeeded and the
    next turn failed on an unknown provider."""
    from dataclasses import asdict
    from types import SimpleNamespace

    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy
    from hermes_state import SessionDB

    srv = servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})
    (tmp_path / "config.yaml").write_text("model: {provider: custom, default: old, base_url: http://127.0.0.1:9/v1}\n")
    with SessionDB(tmp_path / "state.db") as db:
        authority = SimpleNamespace(instance_id="i", epoch=1, profile_id="p", db=db)
        runner = SimpleNamespace(session_authority=authority, _resolve_session_agent_runtime=lambda **k: (None, {}))
        authority.runner = runner
        policy = build_policy({"cwd": str(tmp_path), "model": "old"},
                              {"model": {"provider": "custom", "default": "old", "base_url": "http://127.0.0.1:9/v1"}})
        prepared = {"snapshot": {"receipt": {"session_id": "sid", "policy": asdict(policy)}}}

        committed = await prepare_model(authority, SimpleNamespace(source=None, route="r"),
                                        {"model": "qwen/qwen3-4b", "provider": "local-lmstudio"}, prepared)

        _, runtime = _resolve_policy_agent_runtime(runner, restore_policy(committed["policy"]))
    assert runtime["base_url"].rstrip("/") == f"http://127.0.0.1:{srv.port}/v1"


@pytest.fixture
def raw_server():
    """Factory for loopback ports whose every connection gets ``reply(conn)`` after the request
    arrives; each listener closes at teardown, which also ends its accept thread."""
    listeners: list[socket.socket] = []

    def start(reply) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        listeners.append(sock)

        def serve():
            while True:
                try:
                    conn, _ = sock.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(65536)
                    try:
                        reply(conn)
                    except OSError:
                        pass

        threading.Thread(target=serve, daemon=True).start()
        return sock.getsockname()[1]

    yield start
    for sock in listeners:
        sock.close()


def _point(monkeypatch, kind: str, port: int) -> None:
    known = [k if k.kind != kind else detect._Kind(k.kind, k.label, port) for k in detect.KNOWN_SERVERS]
    monkeypatch.setattr(detect, "KNOWN_SERVERS", tuple(known))
    monkeypatch.setattr(detect, "_KIND_BY_NAME", {k.kind: k for k in known})


def test_a_port_speaking_broken_http_does_not_fail_the_scan(servers, raw_server, monkeypatch):
    """Whatever listens on a scanned port may not speak HTTP; that port yields no row and the other
    servers are still offered (the scan used to raise and the picker lost every row)."""
    servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    _point(monkeypatch, "vllm", raw_server(lambda conn: conn.sendall(b"SSH-2.0-OpenSSH_9.6\r\n\r\n")))

    assert [s.kind for s in detect.detect_local_servers(refresh=True)] == ["ollama"]


def test_a_server_dripping_bytes_cannot_hold_a_probe(servers, raw_server):
    """Each byte arrives under the socket timeout, so only a total deadline ends the probe."""
    def drip(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
        for _ in range(100):
            conn.sendall(b" ")
            time.sleep(0.05)  # health: allow HX013 -- the drip delay is the behaviour under test

    port = raw_server(drip)
    started = time.monotonic()
    assert detect.probe_port(port, expected=detect._Kind("vllm", "vLLM", port), timeout=0.4) is None
    assert time.monotonic() - started < 2


def test_a_key_saved_for_another_address_is_never_claimed(servers):
    """``providers.local-ollama`` already names a remote box; the local Ollama must not be offered
    under that key, since picking it would select (and report saving) the remote entry."""
    from hermes_cli.config import load_config, save_config

    servers("ollama", {"/v1/models": _models(["llama3.2:3b"], "library")})
    cfg = load_config()
    remote = {"name": "GPU box", "base_url": "http://10.0.0.5:11434/v1"}
    cfg["providers"] = {"local-ollama": dict(remote)}
    save_config(cfg)

    assert detect.unsaved_servers(load_config()) == []
    (server,) = detect.detect_local_servers()
    assert detect.save_detected_server(server) is None
    assert load_config()["providers"]["local-ollama"] == remote


async def _gateway_owner(monkeypatch):
    """A session authority over the test home (its state.db and config.yaml share one home, as in
    production) and one client connection to it."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "config.yaml").write_text("platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, "
                                      "base_url: http://127.0.0.1:9/v1}\n", encoding="utf-8")
    monkeypatch.setattr(run, "_hermes_home", home)
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: (None, {}))
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id="fixture")
    return home, runner, owner, AuthorityConnection(owner, object(), {"user_id": "owner"})


@pytest.mark.asyncio
async def test_a_new_gateway_chat_pinned_to_a_detected_row_runs_on_that_server(servers, monkeypatch):
    """Desktop creates chats through the gateway's session authority; a composer pick of a detected
    row froze ``local-lmstudio`` with no ``providers:`` entry behind it, and the first turn failed."""
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    srv = servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})
    home, runner, owner, connection = await _gateway_owner(monkeypatch)
    try:
        await connection.create(None, {"request_id": "r1", "source": "cli", "cwd": str(home), "toolsets": [],
                                       "model": "qwen/qwen3-4b", "provider": "local-lmstudio"})
        sid = local_identity(owner.profile_id, connection.actor.subject, "r1")
        policy = restore_policy(local_receipt(owner.db, sid)["policy"])
        _, runtime = _resolve_policy_agent_runtime(runner, policy)
    finally:
        await connection.close()
        owner.db.close()
    assert runtime["base_url"].rstrip("/") == f"http://127.0.0.1:{srv.port}/v1"


@pytest.mark.asyncio
async def test_a_refused_create_retry_saves_no_detected_server(servers, monkeypatch):
    """Retrying a staged request_id with a different (detected) pick is refused; the refused request
    must not have written that server into the profile's config.yaml first."""
    from hermes_state_runtime import RuntimeStoreError

    servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})
    home, _, owner, connection = await _gateway_owner(monkeypatch)
    base = {"request_id": "r1", "source": "cli", "cwd": str(home), "toolsets": []}
    try:
        await connection.create(None, dict(base))
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            await connection.create(None, {**base, "model": "qwen/qwen3-4b", "provider": "local-lmstudio"})
    finally:
        await connection.close()
        owner.db.close()
    assert "local-lmstudio" not in (home / "config.yaml").read_text(encoding="utf-8")


def test_an_unprobed_lmstudio_row_does_not_hide_the_running_server(servers):
    """``providers.lmstudio`` holding only a timeout makes an empty ``lmstudio`` row; the detected
    server with its real models must still be offered."""
    from hermes_cli.inventory import ConfigContext, build_models_payload

    servers("lmstudio", {"/v1/models": _models(["qwen/qwen3-4b"], "organization_owner")})
    ctx = ConfigContext("openrouter", "x", "", {"lmstudio": {"request_timeout_seconds": 120}}, [])
    rows = build_models_payload(ctx, for_picker=True, local_detection="probe")["providers"]

    assert [r["models"] for r in rows if r.get("source") == "detected"] == [["qwen/qwen3-4b"]]
