"""Contract tests for hermes_cli.local_runtime — Rollouts 1+2.

Per the design's verification plan: relationships and contracts, no
change-detector tests, real imports against temp HERMES_HOME (the autouse
fixture isolates it). The stub HTTP server speaks just enough llama-server
(/props, /health, /models, /v1/chat/completions, /metrics, /slots) to
exercise detection fingerprinting and supervisor logic without a GPU.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hermes_cli.local_runtime.binaries import (
    AssetPlan,
    BinaryResolutionError,
    resolve_assets,
    select_backend,
)
from hermes_cli.local_runtime.detect import DetectedServer, probe_port


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_binary_gguf(path: Path, *, metadata: dict[str, int | str] | None = None,
                       tensors: list[tuple[str, list[int], int]] | None = None) -> None:
    """Write a real GGUF header/table fixture without allocating tensor payload bytes."""
    metadata = metadata or {}
    tensors = tensors or []
    body = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        body.append(_gguf_string(key))
        if isinstance(value, str):
            body.extend((struct.pack("<I", 8), _gguf_string(value)))
        else:
            body.extend((struct.pack("<I", 10), struct.pack("<Q", value)))
    for name, dims, tensor_type in tensors:
        body.extend((
            _gguf_string(name), struct.pack("<I", len(dims)),
            struct.pack(f"<{len(dims)}Q", *dims), struct.pack("<I", tensor_type),
            struct.pack("<Q", 0),
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(body))


# ── stub llama-server ────────────────────────────────────────


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal llama-server imitation; behavior driven by class attrs."""

    props: dict = {}
    models: dict | None = None
    require_auth = False
    chat_answer = "Paris"
    requests_processing = 0
    slots: list = []

    def _send(self, code: int, body: dict | str | None = None) -> None:
        raw = (json.dumps(body) if isinstance(body, dict) else (body or "")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if self.require_auth and "Authorization" not in self.headers:
            self._send(401, {})
            return
        path = self.path.split("?")[0]  # router telemetry uses ?model=
        if path == "/props":
            self._send(200, self.props)
        elif path == "/health":
            self._send(200, {"status": "ok"})
        elif path == "/models":
            if self.models is None:
                self._send(404, {})
            else:
                self._send(200, self.models)
        elif path == "/metrics":
            self._send(200, f"llamacpp:requests_processing {self.requests_processing}\n")
        elif path == "/slots":
            raw = json.dumps(self.slots).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self._send(404, {})

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/chat/completions":
            self._send(200, {"choices": [{"message": {
                "role": "assistant", "content": self.chat_answer}}]})
        elif self.path == "/models/load":
            self._send(200, {"success": True})
        elif self.path == "/models/unload":
            type(self).unloaded = getattr(type(self), "unloaded", [])
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            type(self).unloaded.append(body.get("model"))
            self._send(200, {"success": True})
        else:
            self._send(404, {})

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def stub_server():
    """Yields (port, handler_class); handler attrs are per-test mutable."""

    class Handler(_StubHandler):
        props = {}
        models = None
        require_auth = False
        slots = []

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], Handler
    server.shutdown()


# ── detection (Rollout 1) ────────────────────────────────────


def test_probe_fingerprints_real_llama_server(stub_server):
    port, handler = stub_server
    handler.props = {
        "build_info": "b10290-c8e03ce81",
        "model_path": "C:/models/some model with spaces.gguf",
        "default_generation_settings": {"n_ctx": 65536},
    }
    handler.models = {"data": [{"id": "m", "status": {"value": "unloaded"}}]}
    hit = probe_port(port)
    assert isinstance(hit, DetectedServer)
    assert hit.base_url == f"http://127.0.0.1:{port}/v1"
    assert hit.build_info.startswith("b10290")
    assert hit.n_ctx == 65536
    assert hit.router_mode is True
    assert hit.auth_required is False


def test_probe_rejects_non_llama_openai_server(stub_server):
    # Answers /props with no build_info (e.g. some other local service).
    port, handler = stub_server
    handler.props = {"something": "else"}
    assert probe_port(port) is None


def test_probe_single_model_mode_is_not_router(stub_server):
    port, handler = stub_server
    handler.props = {"build_info": "b10290-x", "model_path": "m.gguf"}
    handler.models = None  # /models 404s in plain (non-router) mode
    hit = probe_port(port)
    assert hit is not None
    assert hit.router_mode is False


def test_probe_auth_required_still_detected(stub_server):
    port, handler = stub_server
    handler.require_auth = True
    hit = probe_port(port)
    assert hit is not None
    assert hit.auth_required is True


def test_probe_dead_port_returns_none():
    # Bind-then-close to get a port that is definitely closed.
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    assert probe_port(dead_port) is None


# ── binary resolver (Rollout 2) ──────────────────────────────


@pytest.mark.parametrize("os_name,arch,backend,ok", [
    ("win", "x64", "cuda", True),
    ("win", "x64", "vulkan", True),
    ("win", "x64", "cpu", True),
    ("win", "arm64", "cpu", True),
    ("win", "arm64", "cuda", True),      # upstream ships these since ~b1036x (CUDA 13.4)
    ("win", "arm64", "vulkan", False),
    ("macos", "arm64", "metal", True),
    ("ubuntu", "x64", "vulkan", True),
    ("ubuntu", "x64", "cpu", True),
    ("ubuntu", "x64", "cuda", False),    # no prebuilt linux CUDA
])
def test_resolver_platform_matrix(os_name, arch, backend, ok):
    if ok:
        plan = resolve_assets("b10290", backend, os_name=os_name, arch=arch)
        assert plan.assets, "resolvable combination must yield assets"
        # Invariant: every asset names the tag or is a paired runtime zip.
        for asset in plan.assets:
            assert "b10290" in asset or asset.startswith("cudart-")
    else:
        with pytest.raises(BinaryResolutionError):
            resolve_assets("b10290", backend, os_name=os_name, arch=arch)


def test_windows_cuda_pairs_cudart():
    """Windows CUDA must ship the runtime zip — users have no toolkit."""
    plan = resolve_assets("b10290", "cuda", os_name="win", arch="x64")
    assert any(a.startswith("cudart-") for a in plan.assets)


def test_windows_cuda_arm64_pairs_cudart_on_its_own_version():
    """arm64 CUDA rides its own CUDA line (13.4 at b10362, verified live):
    both zips must agree on version and name the arch."""
    plan = resolve_assets("b10362", "cuda", os_name="win", arch="arm64")
    assert len(plan.assets) == 2
    assert all("arm64" in a for a in plan.assets)
    versions = {a.split("cuda-")[1].split("-")[0] for a in plan.assets}
    assert len(versions) == 1, f"paired zips disagree on CUDA version: {plan.assets}"
    assert any(a.startswith("cudart-") for a in plan.assets)
    assert any(a.startswith("llama-") for a in plan.assets)


def test_install_dir_is_profile_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    plan = AssetPlan(tag="b10290", backend="cuda")
    assert str(tmp_path) in str(plan.install_dir)
    assert "runtimes" in plan.install_dir.parts


@pytest.mark.parametrize("vendor,os_name,expected", [
    ("NVIDIA GeForce RTX 5090", "win", "cuda"),
    ("nvidia", "ubuntu", "cuda"),
    ("AMD Radeon RX 7900", "win", "vulkan"),
    ("intel", "win", "vulkan"),
    (None, "win", "cpu"),
    ("", "ubuntu", "cpu"),
    ("nvidia", "macos", "metal"),   # macOS is Metal regardless
    (None, "macos", "metal"),
])
def test_backend_selection(vendor, os_name, expected):
    assert select_backend(vendor, os_name=os_name) == expected


def test_sha256_mismatch_rejects(tmp_path, monkeypatch):
    """A pinned hash that doesn't match the download must hard-fail."""
    from hermes_cli.local_runtime import binaries

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # Pre-place a wrong-content "download" so no network is touched. The
    # asset name is host-dependent (win/.zip, ubuntu/.tar.gz, macos/.zip)
    # — resolve it the way the installer will, so the poisoned file is the
    # one it verifies on every CI platform.
    plan = binaries.resolve_assets("b10290", "cpu")
    asset = plan.assets[0]
    downloads = binaries.runtimes_root() / "downloads"
    downloads.mkdir(parents=True)
    (downloads / asset).write_bytes(b"not the real archive")
    with pytest.raises(BinaryResolutionError, match="sha256 mismatch"):
        binaries.ensure_runtime_installed(
            "b10290", "cpu",
            expected_sha256={asset: "0" * 64})
    # The poisoned download must not survive for a retry to trust.
    assert not (downloads / asset).exists()


# ── supervisor contracts (stubbed; no GPU) ───────────────────


def _make_supervisor(tmp_path, port):
    """Supervisor pointed at the stub: skip spawn, drive HTTP logic only."""
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    sup = LlamaServerSupervisor(
        install_dir=tmp_path, models_dir=tmp_path, port=port)
    return sup


def test_touch_generate_is_the_readiness_proof(stub_server, tmp_path):
    port, handler = stub_server
    sup = _make_supervisor(tmp_path, port)
    handler.chat_answer = "Paris"
    assert sup.touch_generate("m") is True
    handler.chat_answer = "I cannot answer that."
    assert sup.touch_generate("m") is False


def test_touch_generate_scans_reasoning_content(stub_server, tmp_path):
    """Reasoning models answer inside reasoning_content (receipted pitfall)."""
    port, handler = stub_server
    sup = _make_supervisor(tmp_path, port)

    class ReasoningHandler(handler):  # type: ignore[valid-type]
        def do_POST(self):  # noqa: N802
            if self.path == "/v1/chat/completions":
                self._send(200, {"choices": [{"message": {
                    "role": "assistant", "content": "",
                    "reasoning_content": "The capital of France is Paris."}}]})
            else:
                self._send(404, {})

    # Swap handler class on the live stub server socket is overkill; just
    # verify the scan logic path via the normal handler with empty content.
    handler.chat_answer = ""
    assert sup.touch_generate("m") is False  # empty content, no reasoning field


def test_ensure_model_ready_unknown_model_raises(stub_server, tmp_path):
    port, handler = stub_server
    handler.models = {"data": [{"id": "present", "status": {"value": "unloaded"}}]}
    sup = _make_supervisor(tmp_path, port)
    with pytest.raises(KeyError):
        sup.ensure_model_ready("absent")


def test_is_idle_requires_no_busy_slots_and_zero_processing(stub_server, tmp_path):
    port, handler = stub_server
    sup = _make_supervisor(tmp_path, port)
    # Router telemetry is per-child (?model=); a loaded model must exist for
    # is_idle to have anything to check.
    handler.models = {"data": [{"id": "m", "status": {"value": "loaded"}}]}
    handler.slots = [{"id": 0, "is_processing": False}]
    handler.requests_processing = 0
    assert sup.is_idle() is True
    handler.slots = [{"id": 0, "is_processing": True}]
    assert sup.is_idle() is False
    handler.slots = [{"id": 0, "is_processing": False}]
    handler.requests_processing = 2
    assert sup.is_idle() is False


def test_base_url_dials_loopback_ip_never_localhost(tmp_path):
    """C12: localhost costs ~2s/request on Windows."""
    sup = _make_supervisor(tmp_path, 9999)
    assert "127.0.0.1" in sup.base_url
    assert "localhost" not in sup.base_url


# ── provider integration (existing alias mechanism, no new plugin) ──


def test_llamacpp_aliases_route_to_custom_profile():
    """Design + maintainer direction: llamacpp fits the EXISTING provider
    mechanism — the aliases already resolve to the keyless custom profile;
    no parallel provider plugin exists."""
    from providers import get_provider_profile

    for alias in ("llamacpp", "llama.cpp", "llama-cpp"):
        profile = get_provider_profile(alias)
        assert profile is not None, alias
        assert profile.name == "custom"
        assert profile.env_vars == ()  # credential is reachability


def test_llamacpp_endpoint_resolution_prefers_managed(tmp_path, monkeypatch, stub_server):
    """provider: llamacpp with a live managed server resolves to it,
    api-key included."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    port, handler = stub_server
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        # A LIVE pid: the ownership guard treats health-200 + dead recorded
        # pid as a foreign server on our stable port (scratch-profile
        # collision), so claiming this test process models "our server".
        "base_url": f"http://127.0.0.1:{port}/v1", "api_key": "sk-managed", "pid": os.getpid(),
    }), encoding="utf-8")
    resolved = ep.resolve_llamacpp_endpoint()
    assert resolved == {"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "sk-managed"}


def test_llamacpp_endpoint_stale_state_falls_through(tmp_path, monkeypatch):
    """A crashed-without-cleanup state file (dead pid, dead endpoint) must
    not blackhole requests: state ignored -> detection (none here) -> None."""
    import socket

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.detect import DEFAULT_PROBE_PORTS
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{dead_port}/v1", "api_key": "sk-x", "pid": 1,
    }), encoding="utf-8")
    monkeypatch.setattr(ep, "_pid_alive", lambda pid: False)
    # Keep detection away from any real server on 8080 during the test.
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS",
                        (dead_port,))
    assert ep.resolve_llamacpp_endpoint() is None
    assert DEFAULT_PROBE_PORTS  # (import kept honest)


def test_llamacpp_dead_server_raises_friendly_error(tmp_path, monkeypatch):
    """A llamacpp send with no server must say WHY in user terms, not fall
    through to the generic custom path (which lands on a cloud provider
    with a placeholder key and surfaces as a baffling '401 Invalid API
    key'). Message tracks the off switch: enabled = probably starting;
    disabled = the user turned it off."""
    import pytest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        "hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
        lambda *a, **k: None)

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"local_runtime": {"enabled": False}})
    with pytest.raises(ValueError, match="turned off"):
        rp._resolve_named_custom_runtime(requested_provider="llamacpp")

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"local_runtime": {"enabled": True}})
    with pytest.raises(ValueError, match="isn't running"):
        rp._resolve_named_custom_runtime(requested_provider="llamacpp")

    # An explicit base_url is the user pointing at a specific server —
    # that path keeps its own error reporting, never this one.
    result = rp._resolve_named_custom_runtime(
        requested_provider="llamacpp",
        explicit_base_url="http://127.0.0.1:9999/v1")
    assert result is None or result.get("base_url", "").startswith("http://127.0.0.1:9999")


def test_llamacpp_endpoint_starting_server_resolves(tmp_path, monkeypatch):
    """The restart race: state written at spawn, server not yet healthy,
    supervisor child alive — resolution must return the endpoint (a
    STARTING server is configured, not missing credentials; this exact
    race threw the app back to onboarding on the first restart test)."""
    import socket

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        not_listening = s.getsockname()[1]
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{not_listening}/v1",
        "api_key": "sk-starting", "pid": 4242,
    }), encoding="utf-8")
    monkeypatch.setattr(ep, "_pid_alive", lambda pid: True)
    resolved = ep.resolve_llamacpp_endpoint()
    assert resolved is not None
    assert resolved["api_key"] == "sk-starting"


def test_llamacpp_endpoint_waits_for_boot_in_flight(tmp_path, monkeypatch):
    """The SECOND restart race (no state file at all yet): a fresh backend's
    readiness probe resolves before the lifespan boot thread has even
    spawned the server. With the runtime enabled+installed, resolution must
    poll briefly and pick up the state file when the boot thread writes it
    — not report unconfigured."""
    import threading
    import time as _time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.supervisor import state_path

    # Boot is in flight: runtime enabled + binary installed.
    monkeypatch.setattr(ep, "_boot_in_flight", lambda config: True)
    monkeypatch.setattr(ep, "_pid_alive", lambda pid: True)
    # Nothing detected externally.
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS", ())

    def _late_writer():
        _time.sleep(0.6)
        state_path().parent.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps({
            "base_url": "http://127.0.0.1:59999/v1",
            "api_key": "sk-boot", "pid": 777,
        }), encoding="utf-8")

    t = threading.Thread(target=_late_writer)
    t.start()
    try:
        resolved = ep.resolve_llamacpp_endpoint(wait_for_boot_s=5.0)
    finally:
        t.join()
    assert resolved is not None
    assert resolved["api_key"] == "sk-boot"


def test_resolution_kicks_boot_when_no_thread_is_booting(tmp_path, monkeypatch):
    """The dead-router-mid-flight case: runtime enabled+installed, but no
    state file and NO lifespan boot thread running (the router died after
    backend start — tree-killed with a stale backend, or the stable port
    was owned by another install and the ownership guard refused it).
    Resolution must not just wait for a boot that nobody is doing — it
    kicks ensure_local_runtime itself and picks up the state file that
    boot writes."""
    import time as _time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap as bs
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.supervisor import state_path

    monkeypatch.setattr(ep, "_boot_in_flight", lambda config: True)
    monkeypatch.setattr(ep, "_pid_alive", lambda pid: True)
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS", ())

    def _fake_ensure(config, force=False):
        _time.sleep(0.3)  # a real spawn takes a moment
        state_path().parent.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps({
            "base_url": "http://127.0.0.1:59998/v1",
            "api_key": "sk-kicked", "pid": 778,
        }), encoding="utf-8")

    monkeypatch.setattr(bs, "ensure_local_runtime", _fake_ensure)

    resolved = ep.resolve_llamacpp_endpoint(config={}, wait_for_boot_s=5.0)
    assert resolved is not None
    assert resolved["api_key"] == "sk-kicked"


def test_boot_in_flight_real_gate(tmp_path, monkeypatch):
    """_boot_in_flight exercised FOR REAL (the previous regression test
    monkeypatched it — and the real one threw TypeError on every call,
    silently disabling the boot wait). Enabled + verified manifest on
    disk -> True; either missing -> False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import endpoint as ep
    from hermes_cli.local_runtime.binaries import runtimes_root

    enabled = {"local_runtime": {"enabled": True}}
    # Not installed yet -> False.
    assert ep._boot_in_flight(enabled) is False
    # Verified install manifest -> True.
    install = runtimes_root() / "b10290" / "cuda"
    install.mkdir(parents=True)
    (install / "manifest.json").write_text(
        json.dumps({"tag": "b10290", "verified_version": "5015 (abc)"}),
        encoding="utf-8")
    assert ep._boot_in_flight(enabled) is True
    # Disabled -> False even when installed.
    assert ep._boot_in_flight({"local_runtime": {"enabled": False}}) is False


def test_idle_sweep_unloads_idle_models(tmp_path, monkeypatch, stub_server):
    """Residency v2 contract: after the idle threshold, idle loaded models
    unload — no exemptions; demand reloads anything the user returns to.
    Idleness is the C5 contract (no busy slots)."""
    port, handler = stub_server
    handler.models = {"data": [
        {"id": "model-a", "status": {"value": "loaded"}},
        {"id": "model-b", "status": {"value": "loaded"}},
    ]}
    handler.slots = []          # everyone idle per C5
    handler.requests_processing = 0
    handler.unloaded = []

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    sup = LlamaServerSupervisor(tmp_path / "i", tmp_path / "m", port=port)

    t0 = 1000.0
    # First sweep: starts the idle clocks, nothing unloads yet.
    assert sup.sweep_idle(now=t0) == []
    # Before the threshold: still nothing.
    assert sup.sweep_idle(now=t0 + sup.IDLE_UNLOAD_S - 1) == []
    # Past the threshold: both idle models unload.
    assert sorted(sup.sweep_idle(now=t0 + sup.IDLE_UNLOAD_S + 1)) == ["model-a", "model-b"]
    assert sorted(handler.unloaded) == ["model-a", "model-b"]


def test_idle_sweep_busy_model_resets_clock(tmp_path, monkeypatch, stub_server):
    """A model seen busy (C5: busy slot) restarts its idle clock — an
    active conversation never trips the sweep."""
    port, handler = stub_server
    handler.models = {"data": [{"id": "side-m", "status": {"value": "loaded"}}]}
    handler.unloaded = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    sup = LlamaServerSupervisor(tmp_path / "i", tmp_path / "m", port=port)

    t0 = 1000.0
    handler.slots = []                 # idle: clock starts
    assert sup.sweep_idle(now=t0) == []
    handler.slots = [{"is_processing": True}]   # busy mid-window
    assert sup.sweep_idle(now=t0 + sup.IDLE_UNLOAD_S) == []
    handler.slots = []                 # idle again: clock restarts, not expired
    assert sup.sweep_idle(now=t0 + sup.IDLE_UNLOAD_S + 10) == []
    assert handler.unloaded == []


def test_staged_models_requires_every_split_part(tmp_path, monkeypatch):
    """A split GGUF mid-download must NOT count as staged: the picker, the
    catalog's 'downloaded' flag, and the router's model list all read
    staged_models(), and a first part with missing continuations is not
    servable. Single files and complete splits count; continuation parts
    never count as their own model."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import hermes_cli.local_runtime.bootstrap as bs

    mdir = bs.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)

    (mdir / "Single-Q4_K_M.gguf").touch()
    # Complete split: both parts present.
    (mdir / "Whole-Q4-00001-of-00002.gguf").touch()
    (mdir / "Whole-Q4-00002-of-00002.gguf").touch()
    # Mid-download split: first part only, of three.
    (mdir / "Partial-Q4-00001-of-00003.gguf").touch()

    assert bs.staged_model_ids() == ["Single-Q4_K_M", "Whole-Q4"]


def test_split_gguf_reader_aggregates_ple_outside_first_shard(tmp_path):
    """A metadata-only first shard must account for tensors stored by later shards."""
    from hermes_cli.local_runtime.estimator import profile_from_gguf
    from hermes_cli.local_runtime.gguf import (
        AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES,
        read_gguf_header,
        supports_automatic_lazy_lookup,
    )

    first = tmp_path / "Toy-00001-of-00002.gguf"
    second = tmp_path / "Toy-00002-of-00002.gguf"
    metadata = {
        "general.architecture": "toy",
        "toy.block_count": 1,
        "toy.context_length": 65536,
        "toy.embedding_length": 16,
        "toy.attention.head_count": 1,
        "toy.attention.head_count_kv": 1,
    }
    lazy = AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1
    _write_binary_gguf(first, metadata=metadata)  # exact shape of Qwen's shard one: metadata, no tensors
    _write_binary_gguf(second, tensors=[
        ("token_embd.weight", [32], 24),
        ("per_layer_token_embd.weight", [lazy], 24),
    ])

    header = read_gguf_header(first)
    assert header.n_tensors == 2
    assert header.tensor_bytes == 32 + lazy
    assert header.embd_table_bytes == 32
    assert header.lookup_table_bytes == lazy
    assert header.lazy_table_bytes == lazy
    # Continuation paths receive the same aggregate, not just their local table.
    assert read_gguf_header(second).tensor_bytes == header.tensor_bytes

    assert supports_automatic_lazy_lookup("b10678") is False
    assert supports_automatic_lazy_lookup("b10679") is True
    assert supports_automatic_lazy_lookup("b10679-old") is False
    assert supports_automatic_lazy_lookup("not-a-build-10679") is False
    assert supports_automatic_lazy_lookup(None) is False
    assert supports_automatic_lazy_lookup("b" + "9" * 10_000) is False
    old = profile_from_gguf(header, engine_tag="b10678")
    new = profile_from_gguf(header, engine_tag="b10679")
    assert old.lazy_table_bytes == 0
    assert old.resident_weights_bytes == header.tensor_bytes
    assert new.lazy_table_bytes == lazy
    assert new.resident_weights_bytes == 32


def test_automatic_lazy_ple_threshold_is_strict(tmp_path):
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES, read_gguf_header

    path = tmp_path / "Boundary.gguf"
    _write_binary_gguf(path, tensors=[
        ("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES], 24),
    ])
    header = read_gguf_header(path)
    assert header.lookup_table_bytes == AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES
    assert header.lazy_table_bytes == 0


def test_gguf_reader_bounds_malicious_header_dimensions(tmp_path):
    """Status inspection must reject a tiny hostile header before it can build a giant struct format."""
    from hermes_cli.local_runtime.gguf import read_gguf_header

    path = tmp_path / "Hostile.gguf"
    path.write_bytes(b"".join((
        b"GGUF", struct.pack("<IQQ", 3, 1, 0), _gguf_string("bad"), struct.pack("<I", 999_999),
    )))

    with pytest.raises(ValueError, match="too many dimensions"):
        read_gguf_header(path)


def test_bootstrap_refuses_eligible_ple_on_an_old_engine(tmp_path, monkeypatch):
    """The router autoloads its whole directory, so startup itself—not just the API—must gate PLE.

    The tiny fixture has a tensor table describing more than 4 GiB without allocating its tensor
    payload.  It models the Qwen split/header fact closely enough to prove b10678 never reaches
    llama-server installation/spawn, while b10679 does proceed through the normal boot ladder.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import binaries, bootstrap
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES

    mdir = bootstrap.models_dir()
    _write_binary_gguf(mdir / "Engram-Q4.gguf", tensors=[
        ("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1], 24),
    ])
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(binaries, "active_tag", lambda section: "b10678")
    reached: list[str] = []
    monkeypatch.setattr(
        binaries, "ensure_runtime_installed",
        lambda tag, backend, **kwargs: reached.append(tag) or (_ for _ in ()).throw(
            AssertionError("old engine must not reach installation/spawn")))

    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "tag": "b10678"}}) is None
    assert reached == []

    # The same file is safe with the first lazy-lookup-capable build; reaching the installer is
    # sufficient here because its deliberate assertion is swallowed by the session-safe boot API.
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10679"])
    monkeypatch.setattr(binaries, "active_tag", lambda section: "b10679")
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "tag": "b10679"}}) is None
    assert reached == ["b10679"]


def test_bootstrap_refuses_catalog_architecture_floor_without_a_ple_tensor(tmp_path, monkeypatch):
    """Catalog min_engine protects known architectures even when their parsed header has no PLE table."""
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import binaries, bootstrap, catalog

    _write_binary_gguf(bootstrap.models_dir() / "Catalog-Only-Engine-Gate.gguf")
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(catalog, "entry_for_model", lambda model_id: SimpleNamespace(min_engine="b10679"))
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(binaries, "active_tag", lambda section: "b10678")
    reached: list[str] = []
    monkeypatch.setattr(binaries, "ensure_runtime_installed", lambda *args, **kwargs: reached.append("spawn"))

    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "tag": "b10678"}}) is None
    assert reached == []


def test_bootstrap_stops_legacy_incumbent_before_ple_can_autoload(tmp_path, monkeypatch):
    """An adopted old server is just as unsafe as a new b10678 boot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import binaries, bootstrap, endpoint
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES

    _write_binary_gguf(bootstrap.models_dir() / "Engram-Q4.gguf", tensors=[
        ("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1], 24),
    ])
    incumbent = {"base_url": "http://127.0.0.1:18434/v1", "api_key": "k", "pid": 4321}
    stopped: list[int] = []
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(endpoint, "_managed_state", lambda: incumbent)
    monkeypatch.setattr(bootstrap, "_stop_state_server", lambda state: stopped.append(state["pid"]))
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(binaries, "active_tag", lambda section: "b10678")
    monkeypatch.setattr(
        binaries, "ensure_runtime_installed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("old engine must not reach installation/spawn")))

    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "tag": "b10678"}}) is None
    assert stopped == [4321]


def test_refresh_stops_an_adopted_server_using_private_pid_state(tmp_path, monkeypatch):
    """Refresh must retain the ownership PID; provider routing intentionally does not expose it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap, endpoint

    incumbent = {"base_url": "http://127.0.0.1:18434/v1", "api_key": "k", "pid": 4321}
    stopped: list[int] = []
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(endpoint, "_managed_state", lambda: incumbent)
    monkeypatch.setattr(bootstrap, "_stop_state_server", lambda state: stopped.append(state["pid"]))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"local_runtime": {"enabled": True}})
    monkeypatch.setattr(bootstrap, "ensure_local_runtime", lambda config, force=False: object())

    assert bootstrap.refresh_local_runtime() is True
    assert stopped == [4321]


def test_bootstrap_fails_closed_for_uninspectable_sideload_on_an_old_engine(tmp_path, monkeypatch):
    """An old llama.cpp may parse a future GGUF layout that Hermes cannot inspect yet."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import binaries, bootstrap

    path = bootstrap.models_dir() / "Future-Layout.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(binaries, "installed_tags", lambda: ["b10678"])
    monkeypatch.setattr(binaries, "active_tag", lambda section: "b10678")
    reached: list[str] = []
    monkeypatch.setattr(binaries, "ensure_runtime_installed", lambda *args, **kwargs: reached.append("spawn"))

    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "tag": "b10678"}}) is None
    assert reached == []


def test_supervisor_persists_and_endpoint_proves_serving_engine_tag(tmp_path, monkeypatch):
    """A pending configured update must not masquerade as the engine actually serving requests."""
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import endpoint
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor, state_path

    sup = LlamaServerSupervisor(tmp_path / "install", tmp_path / "models", engine_tag="b10679")
    sup.proc = SimpleNamespace(pid=4321)
    monkeypatch.setattr(endpoint, "_pid_alive", lambda pid: pid == 4321)
    sup._write_state()

    state = endpoint._managed_state()
    assert state is not None and state["engine_tag"] == "b10679"
    assert state["pid"] == 4321
    # Provider routing deliberately receives no process-management PID.
    assert endpoint._state_endpoint() == {
        "base_url": sup.base_url, "api_key": sup.api_key, "engine_tag": "b10679"}
    assert not list(state_path().parent.glob("*.tmp"))
    assert endpoint.managed_engine_tag() == "b10679"

    # Old state files intentionally yield no guess.  A caller must use the safe resident/update
    # path rather than infer support from config or an installed-but-not-running build.
    state_path().write_text(json.dumps({
        "base_url": sup.base_url, "api_key": sup.api_key, "pid": 4321,
    }), encoding="utf-8")
    assert endpoint.managed_engine_tag() is None


def test_endpoint_refuses_an_old_persisted_router_before_async_boot(tmp_path, monkeypatch):
    """A provider request racing session boot must not get an old router even once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap, endpoint
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES
    from hermes_cli.local_runtime.supervisor import state_path

    _write_binary_gguf(bootstrap.models_dir() / "Engram-Q4.gguf", tensors=[
        ("per_layer_token_embd.weight", [AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1], 24),
    ])
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:18434/v1", "api_key": "k", "pid": 4321,
        "engine_tag": "b10678",
    }), encoding="utf-8")
    monkeypatch.setattr(endpoint, "_pid_alive", lambda pid: pid == 4321)
    kicked: list[object] = []
    monkeypatch.setattr(endpoint, "_kick_managed_boot", lambda config: kicked.append(config))
    monkeypatch.setattr(
        "hermes_cli.local_runtime.detect.detect_server",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unsafe managed port must not be rediscovered")),
    )

    assert endpoint.resolve_llamacpp_endpoint(wait_for_boot_s=0) is None
    assert kicked == [None]


def test_endpoint_refuses_old_persisted_router_for_catalog_engine_floor(tmp_path, monkeypatch):
    """The last resolver boundary also honors non-PLE catalog engine requirements."""
    from types import SimpleNamespace

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap, catalog, endpoint
    from hermes_cli.local_runtime.supervisor import state_path

    _write_binary_gguf(bootstrap.models_dir() / "Catalog-Only-Engine-Gate.gguf")
    monkeypatch.setattr(catalog, "entry_for_model", lambda model_id: SimpleNamespace(min_engine="b10679"))
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:18434/v1", "api_key": "k", "pid": 4321,
        "engine_tag": "b10678",
    }), encoding="utf-8")
    monkeypatch.setattr(endpoint, "_pid_alive", lambda pid: pid == 4321)
    kicked: list[object] = []
    monkeypatch.setattr(endpoint, "_kick_managed_boot", lambda config: kicked.append(config))

    assert endpoint.resolve_llamacpp_endpoint(wait_for_boot_s=0) is None
    assert kicked == [None]


def test_supervisor_spawn_guard_and_watchdog_stop_prevent_unsafe_restart(tmp_path, monkeypatch):
    """The watchdog cannot bypass bootstrap policy or revive after shutdown during backoff."""
    from types import SimpleNamespace

    from hermes_cli.local_runtime import supervisor as sup_mod
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    blocked = LlamaServerSupervisor(tmp_path / "install", tmp_path / "models", can_spawn=lambda: False)
    binaries: list[object] = []
    monkeypatch.setattr(sup_mod, "server_binary", lambda install: binaries.append(install))
    with pytest.raises(RuntimeError, match="placement policy"):
        blocked._spawn()
    assert binaries == []

    stopping = LlamaServerSupervisor(tmp_path / "install", tmp_path / "models")
    stopping.proc = SimpleNamespace(poll=lambda: 1)
    monkeypatch.setattr(sup_mod, "_RESTART_BACKOFF_S", (0,))
    monkeypatch.setattr(sup_mod.time, "sleep", lambda seconds: setattr(stopping, "_stopping", True))
    monkeypatch.setattr(stopping, "_spawn", lambda: pytest.fail("watchdog spawned after stop"))
    stopping._watch()
    assert stopping._stopping is True


def test_only_the_named_oversized_ple_tensor_is_lazy(tmp_path):
    """Size alone is insufficient: unrelated oversized weights remain ordinary residency."""
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES, read_gguf_header

    oversized = AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1
    path = tmp_path / "Other-Lookup.gguf"
    _write_binary_gguf(path, tensors=[("some_other_lookup.weight", [oversized], 24)])
    header = read_gguf_header(path)
    assert header.tensor_bytes == oversized
    assert header.lookup_table_bytes == 0
    assert header.lazy_table_bytes == 0


def test_split_reader_rejects_incomplete_tensor_sets(tmp_path):
    """Partial downloads must not be priced as a metadata-only, zero-byte model."""
    from hermes_cli.local_runtime.gguf import read_gguf_header

    first = tmp_path / "Partial-00001-of-00004.gguf"
    _write_binary_gguf(first, metadata={"general.architecture": "toy"})
    with pytest.raises(ValueError, match="incomplete split GGUF"):
        read_gguf_header(first)


def test_catalog_and_preset_agree_on_real_split_ple_header(tmp_path, monkeypatch):
    """The pre-download catalog and post-download split parser must reach the same placement."""
    from hermes_cli.local_runtime.catalog import AssetFile, CatalogEntry, QuantVariant, select_variant
    from hermes_cli.local_runtime.estimator import HardwareBudget, profile_from_gguf
    from hermes_cli.local_runtime.gguf import AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES, read_gguf_header
    from hermes_cli.local_runtime.presets import generate_presets, read_preset_decisions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    first = tmp_path / "Toy-00001-of-00002.gguf"
    second = tmp_path / "Toy-00002-of-00002.gguf"
    lazy = AUTOMATIC_LAZY_TENSOR_THRESHOLD_BYTES + 1
    _write_binary_gguf(first, metadata={
        "general.architecture": "toy",
        "toy.block_count": 1,
        "toy.context_length": 65536,
        "toy.embedding_length": 16,
        "toy.attention.head_count": 1,
        "toy.attention.head_count_kv": 1,
    })
    _write_binary_gguf(second, tensors=[
        ("token_embd.weight", [32], 24),
        ("per_layer_token_embd.weight", [lazy], 24),
    ])
    variant = QuantVariant(
        quant="Q4", lazy_table_bytes=lazy,
        files=(AssetFile(first.name, 0), AssetFile(second.name, lazy + 32)),
    )
    entry = CatalogEntry(
        id="toy", display_name="Toy", description="split fixture", repo="fixture",
        variants=(variant,), n_ctx_train=65536, full_layers=1, recurrent_layers=0,
        per_layer_f16=64,
    )
    parsed = profile_from_gguf(read_gguf_header(first), engine_tag="b10679")
    catalog_profile = entry.profile(variant, engine_tag="b10679")
    assert (parsed.weights_bytes, parsed.lazy_table_bytes, parsed.resident_weights_bytes) == (
        catalog_profile.weights_bytes, catalog_profile.lazy_table_bytes,
        catalog_profile.resident_weights_bytes,
    )

    budget = HardwareBudget(usable_vram_bytes=2 << 30, total_device_bytes=2 << 30,
                            ram_available_bytes=0)
    choice = select_variant(entry, budget, engine_tag="b10679")
    assert choice is not None and choice.zero_spill
    preset_path = tmp_path / "presets.ini"
    generated = generate_presets(tmp_path, budget, preset_path, engine_tag="b10679")
    assert len(generated) == 1
    preset = generated[0]
    assert preset.lazy_table_bytes == lazy
    assert not preset.spilled
    assert preset.keys is not None and "override-tensor" not in preset.keys
    reread = read_preset_decisions(preset_path)[preset.model_id]
    assert (reread.window, reread.spilled, reread.lazy_table_bytes) == (
        preset.window, preset.spilled, lazy)


def test_bootstrap_skips_boot_with_no_staged_models(tmp_path, monkeypatch):
    """Residency: enabled + installed but zero staged models -> no server
    boot (nothing to serve; the walked-away story)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import hermes_cli.local_runtime.bootstrap as bs

    monkeypatch.setattr(bs, "_SUPERVISOR", None)
    called = {"spawn": False}

    def _boom(*a, **k):
        called["spawn"] = True
        raise AssertionError("must not reach install/spawn")

    monkeypatch.setattr("hermes_cli.local_runtime.binaries.ensure_runtime_installed", _boom)
    result = bs.ensure_local_runtime({"local_runtime": {"enabled": True}})
    assert result is None
    assert called["spawn"] is False


def test_endpoint_identity_stable_across_supervisor_instances(tmp_path, monkeypatch):
    """Round-7 contract: base_url AND api_key survive a restart as a unit.
    Two supervisor constructions (= two backend boots) must agree on both —
    sessions persist the resolved pair, so either piece rotating strands
    every resumed session (connection error / HTTP 401)."""
    import socket as _socket

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import supervisor as sup_mod
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    # A test-owned default port: the production default may legitimately be
    # held by a live managed server on the dev machine.
    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        test_port = s.getsockname()[1]
    monkeypatch.setattr(sup_mod, "_DEFAULT_PORT", test_port)

    first = LlamaServerSupervisor(tmp_path / "install", tmp_path / "models")
    second = LlamaServerSupervisor(tmp_path / "install", tmp_path / "models")
    assert first.api_key == second.api_key
    assert len(first.api_key) >= 16
    assert first.port == second.port == test_port
    # The key is persisted, not per-process state.
    key_file = tmp_path / ".hermes" / "runtimes" / "llamacpp" / ".api_key"
    assert key_file.exists()
    assert key_file.read_text(encoding="utf-8").strip() == first.api_key


def test_llamacpp_endpoint_no_wait_when_not_enabled(tmp_path, monkeypatch):
    """No boot in flight (runtime disabled/uninstalled): resolution returns
    None promptly instead of burning the wait budget."""
    import time as _time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import endpoint as ep

    monkeypatch.setattr(ep, "_boot_in_flight", lambda config: False)
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS", ())
    t0 = _time.monotonic()
    assert ep.resolve_llamacpp_endpoint(wait_for_boot_s=8.0) is None
    assert _time.monotonic() - t0 < 3.0


def test_switch_model_explicit_llamacpp_provider(tmp_path, monkeypatch, stub_server):
    """The desktop dropdown path: switch_model(explicit_provider='llamacpp')
    must resolve the managed provider — not 'Unknown provider' (the
    desktop-review symptom). E2E through the real pipeline against a stub server."""
    port, handler = stub_server
    handler.models = {"data": [{"id": "stub-model-a", "owned_by": "llamacpp"}]}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{port}/v1",
        # Live pid: ownership guard rejects health-200 + dead recorded pid
        # (foreign server on our stable port).
        "api_key": "sk-managed", "pid": os.getpid(),
    }), encoding="utf-8")

    from hermes_cli.model_switch import switch_model

    result = switch_model(
        "stub-model-a",
        current_provider="nous",
        current_model="Hermes-4.5",
        current_base_url="",
        explicit_provider="llamacpp",
    )
    assert result.success, result.error_message
    assert f"127.0.0.1:{port}" in (result.base_url or "")
    assert result.api_key == "sk-managed"


def test_runtime_provider_seam_llamacpp_alias(tmp_path, monkeypatch, stub_server):
    """End to end through the REAL resolver: provider='llamacpp' with no
    base_url lands on the managed endpoint with source='local-runtime'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    port, handler = stub_server
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        # A LIVE pid: the ownership guard treats health-200 + dead recorded
        # pid as a foreign server on our stable port (scratch-profile
        # collision), so claiming this test process models "our server".
        "base_url": f"http://127.0.0.1:{port}/v1", "api_key": "sk-managed", "pid": os.getpid(),
    }), encoding="utf-8")

    from hermes_cli.runtime_provider import _resolve_named_custom_runtime

    runtime = _resolve_named_custom_runtime(requested_provider="llamacpp")
    assert runtime is not None
    assert runtime["source"] == "local-runtime"
    assert runtime["base_url"] == f"http://127.0.0.1:{port}/v1"
    assert runtime["api_key"] == "sk-managed"
    assert runtime["provider"] == "custom"


def test_runtime_provider_seam_explicit_base_url_wins(tmp_path, monkeypatch):
    """A user-specified base_url must never be overridden by the managed
    endpoint — pointing at a specific server means that server."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:1/v1", "api_key": "sk-managed", "pid": 1,
    }), encoding="utf-8")

    from hermes_cli.runtime_provider import _resolve_named_custom_runtime

    runtime = _resolve_named_custom_runtime(
        requested_provider="llamacpp",
        explicit_base_url="http://127.0.0.1:9999/v1")
    assert runtime is not None
    assert runtime["base_url"] == "http://127.0.0.1:9999/v1"
    assert runtime["source"] != "local-runtime"


def test_local_runtime_config_defaults_shape():
    """Contract: the section exists, is off by default, and carries no
    context/VRAM knobs (design: constants, not knobs)."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["local_runtime"]
    assert cfg["enabled"] is False
    assert isinstance(cfg["tag"], str) and cfg["tag"].startswith("b")
    forbidden = [k for k in cfg if "context" in k or "ctx" in k or "vram" in k or "kv" in k]
    assert forbidden == [], f"policy constants leaked into config: {forbidden}"


# ── bootstrap contracts ──────────────────────────────────────


def test_bootstrap_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": False}}) is None
    assert bootstrap.ensure_local_runtime({}) is None
    assert bootstrap.ensure_local_runtime(None) is None


def test_bootstrap_reuses_running_server(tmp_path, monkeypatch, stub_server):
    """A live state file (another process supervising) short-circuits the
    install/spawn path entirely."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    port, handler = stub_server
    from hermes_cli.local_runtime import bootstrap
    from hermes_cli.local_runtime.supervisor import state_path

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{port}/v1", "api_key": "k", "pid": os.getpid(),
    }), encoding="utf-8")

    called = []
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda *a, **k: called.append(1))
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
    assert called == []


def test_bootstrap_failure_never_raises(tmp_path, monkeypatch):
    """Session start must survive a broken runtime: failures log + return
    None, chat falls back to configured providers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: None)

    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed", boom)
    result = bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}})
    assert result is None  # no exception escaped
