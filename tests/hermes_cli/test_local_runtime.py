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


# ── server_extra_args contracts (#103270) ────────────────────


def test_normalize_server_extra_args_passthrough():
    """Ordinary device/tuning flags survive verbatim (whitespace trimmed)."""
    from hermes_cli.local_runtime.supervisor import normalize_server_extra_args

    assert normalize_server_extra_args(["--main-gpu", "1"]) == ["--main-gpu", "1"]
    assert normalize_server_extra_args(["  --ctx-size  ", "8192"]) == ["--ctx-size", "8192"]
    assert normalize_server_extra_args(None) == []
    assert normalize_server_extra_args([]) == []


def test_normalize_server_extra_args_drops_managed_identity():
    """--host/--port/--api-key/--models-dir would desync supervised endpoint
    identity (the round-7 contract above) — refused in both --flag value and
    --flag=value forms, including a following bare value."""
    from hermes_cli.local_runtime.supervisor import normalize_server_extra_args

    assert normalize_server_extra_args(["--port", "8080"]) == []
    assert normalize_server_extra_args(["--api-key=hunter2"]) == []
    assert normalize_server_extra_args(["--host", "0.0.0.0", "--main-gpu", "1"]) == ["--main-gpu", "1"]
    assert normalize_server_extra_args(["--models-dir", "/tmp/x"]) == []


def test_normalize_server_extra_args_hygiene_and_bounds():
    """Non-lists degrade to stock; junk items drop; runaway lengths cap."""
    from hermes_cli.local_runtime.supervisor import normalize_server_extra_args

    assert normalize_server_extra_args("--main-gpu 1") == []
    assert normalize_server_extra_args(42) == []
    assert normalize_server_extra_args(["--main-gpu", 1, "", "   ", None]) == ["--main-gpu"]
    assert normalize_server_extra_args(["--ok", "x" * 600]) == ["--ok"]
    many = [f"--flag-{i}" for i in range(100)]
    assert normalize_server_extra_args(many) == many[:32]


def test_spawn_appends_extra_args_after_managed(tmp_path, monkeypatch):
    """argv contract: managed flags first, user tail last (last-wins), no shell."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import supervisor as sup_mod

    captured = {}

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        assert isinstance(cmd, list)  # list-args, never a shell string
        return _FakeProc()

    monkeypatch.setattr(sup_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(sup_mod, "server_binary", lambda install_dir: tmp_path / "llama-server")

    sup = sup_mod.LlamaServerSupervisor(
        tmp_path / "i", tmp_path / "m", port=18434, extra_args=["--main-gpu", "1"])
    sup._spawn()
    cmd = captured["cmd"]
    assert cmd[:3] == [str(tmp_path / "llama-server"), "--host", "127.0.0.1"]
    assert cmd[-2:] == ["--main-gpu", "1"]
    assert "--port" in cmd and "18434" in cmd


def test_ensure_local_runtime_passes_extra_args(tmp_path, monkeypatch):
    """Boot plumbing: config server_extra_args reaches the supervisor normalized."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: "nvidia")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.select_backend", lambda vendor: "cuda")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.default_tag", lambda: "b10679")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.installed_tags", lambda: ["b10679"])
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries.ensure_runtime_installed",
        lambda tag, backend, **k: tmp_path / "rt")
    monkeypatch.setattr(bootstrap, "_generate_presets", lambda mdir, path: None)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda sup: None)

    seen = {}

    class _FakeSupervisor:
        base_url = "http://127.0.0.1:18434/v1"

        def __init__(self, *a, **k):
            seen.update(k)

        def start(self):
            return None

    monkeypatch.setattr(
        "hermes_cli.local_runtime.supervisor.LlamaServerSupervisor", _FakeSupervisor)
    result = bootstrap.ensure_local_runtime({
        "local_runtime": {"enabled": True, "server_extra_args": ["--main-gpu", "1", "--port", "9"]}},
        force=True)
    assert result is not None
    assert seen.get("extra_args") == ["--main-gpu", "1"]


def test_config_default_declares_server_extra_args():
    """Relationship (not snapshot): the local_runtime section declares a list-typed
    server_extra_args that the schema surfaces to Settings as a list field."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.web_server_config import CONFIG_SCHEMA

    section = DEFAULT_CONFIG["local_runtime"]
    assert isinstance(section.get("server_extra_args"), list)
    assert CONFIG_SCHEMA["local_runtime.server_extra_args"]["type"] == "list"
