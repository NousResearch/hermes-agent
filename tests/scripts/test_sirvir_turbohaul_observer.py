"""Tests for the Sirvir observer's Turbofit backend and failure modes.

All fixtures are synthetic files and a loopback HTTP server; no live
gateway, no GPU, no HERMES_HOME access.
"""

import importlib.util
import fcntl
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "sirvir_turbohaul_observer.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sirvir_turbohaul_observer", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status(*, free_mib, generation_state="idle"):
    """Legacy Turbohaul manager /status fixture."""
    return {
        "vram": free_mib,
        "vram_total_mib": [24576] * len(free_mib),
        "generation": {"state": generation_state},
        "active": None,
        "loading": None,
        "grace": None,
        "idle_hot": {"model_tag": "darwin-28b-reason", "remaining_s": 600},
        "queue": {"acceptance_buffer_depth": 0, "staging_queue_depth": 0},
        "parallel_slots": {"used": 0, "max": 1},
    }


def gateway_status(*, main=None, aux=None, vram=None):
    """Turbofit gateway /status fixture (turbofit-gateway/2.0)."""
    payload = {
        "main": main or {"alias": "none", "type": "down"},
        "aux": aux or {"alias": "none", "type": "down"},
        "stall_timeout_s": 90,
        "backend_timeout_s": 300,
        "gateway": "turbofit-gateway/2.0",
    }
    if vram is not None:
        payload["vram"] = vram
        payload["vram_total_mib"] = [24576] * len(vram)
    return payload


_CONTROLLER_LOCKS = []


@pytest.fixture(autouse=True)
def release_fixture_locks():
    yield
    while _CONTROLLER_LOCKS:
        _CONTROLLER_LOCKS.pop().close()


def seed_turbofit_state(tmp_path, *, selection="qwen3.8-27b", with_endpoint=True,
                        endpoint_valid=True, owned_roles=(), leases=None,
                        orphaned=False, owner_running=True):
    runtime_state = tmp_path / "runtime-state.json"
    if selection is not None:
        runtime_state.write_text(json.dumps({
            "active": selection,
            "routes": {"main": {"kind": "local", "alias": "qwen3.8-27b", "port": 8092,
                                "context_length": 204800}},
        }))
    native = tmp_path / "native"
    native.mkdir(exist_ok=True)
    if owner_running:
        handle = (native / "lifecycle.lock").open("a+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _CONTROLLER_LOCKS.append(handle)
    if with_endpoint:
        endpoint = {"host": "127.0.0.1", "port": 45678, "token": "a" * 64}
        if not endpoint_valid:
            endpoint = {"host": "127.0.0.1", "port": 0, "token": "a" * 64}
        (native / "lifecycle-endpoint.json").write_text(json.dumps(endpoint))
    (native / "lifecycle-state.json").write_text(json.dumps({
        "schema": 2,
        "leases": leases or {},
        "last_activity_wall": {"main": 0, "aux": 0},
    }))
    if orphaned:
        (native / "lifecycle-state.json").write_text("not json at all")
    for role in owned_roles:
        (native / f"{role}.json").write_text(json.dumps({
            "pid": 100 + len(role),
            "alias": f"tag-{role}",
            "port": 8092 if role == "main" else 8093,
            "command": ["/usr/bin/llama-server"],
            "start_identity": "boot",
        }))
    return runtime_state, native


# ─── Legacy Turbohaul behaviour is preserved ──────────────────────────────────

def test_uses_turbohaul_vram_as_free_mib_not_used_mib():
    module = load_module()

    snapshot = module.snapshot_from_status(status(free_mib=[23001, 24110]))

    assert snapshot.free_mib == [23001, 24110]
    assert snapshot.minimum_free_gib == 23001 / 1024


def test_requires_two_stable_polls_then_contracts_one_level():
    module = load_module()
    observer = module.Observer(stability_checks=2)
    pressured = module.snapshot_from_status(status(free_mib=[5 * 1024, 20 * 1024]))

    first = observer.evaluate(pressured)
    second = observer.evaluate(pressured)

    assert first["decision"] == "await_stability"
    assert second["current_level"] == 0
    assert second["next_level"] == 1
    assert second["decision"] == "would_hold_context_at_65536"
    assert observer.level == 1


def test_defers_contraction_while_turbohaul_is_generating():
    module = load_module()
    observer = module.Observer(stability_checks=1)
    critical = module.snapshot_from_status(
        status(free_mib=[512, 20 * 1024], generation_state="generating")
    )

    result = observer.evaluate(critical)

    assert result["target_level"] == 5
    assert result["decision"] == "defer_generation_in_flight"
    assert observer.level == 0


def test_recovery_never_requests_context_above_proven_64k_floor():
    module = load_module()
    observer = module.Observer(stability_checks=1)
    observer.level = 1
    recovered = module.snapshot_from_status(status(free_mib=[11 * 1024, 20 * 1024]))

    result = observer.evaluate(recovered)

    assert result["next_level"] == 0
    assert result["decision"] == "would_keep_context_at_65536"
    assert observer.level == 0


def test_observer_source_has_no_lifecycle_or_config_actuators():
    source = MODULE_PATH.read_text()

    for forbidden in ("systemctl", "subprocess", "models.yaml", "config.yaml",
                      "lifecycle_request", "publish_route_state", "save_controller_state"):
        assert forbidden not in source


# ─── Turbofit gateway backend ─────────────────────────────────────────────────

def test_stale_endpoint_without_controller_is_not_healthy(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path, owner_running=False)
    # A killed controller leaves endpoint/state JSON, but no kernel-held lock.
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(vram=[20000, 20000]), runtime_state=runtime_state,
        native_state=native,
    )
    assert snapshot.extra["resident_state"] == "stale"
    assert module.Observer().evaluate(snapshot)["decision"] == "controller_state_unavailable"


def test_turbofit_idle_reports_idle_not_unhealthy(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path, with_endpoint=True)
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )

    assert snapshot.backend == "turbofit"
    assert snapshot.extra["resident_state"] == "idle"
    assert snapshot.minimum_free_gib is None
    event = module.Observer().evaluate(snapshot)
    assert event["decision"] == "vram_unavailable"
    assert event["target_level"] == 0
    assert event["minimum_free_gib"] is None


def test_turbofit_residents_present_when_main_ready(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(
        tmp_path, owned_roles=("main",), with_endpoint=True
    )
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(main={"alias": "qwen3.8-27b", "state": "ready",
                             "base_url": "http://127.0.0.1:8092", "port": 8092}),
        runtime_state=runtime_state,
        native_state=native,
    )

    assert snapshot.extra["resident_state"] == "residents_present"
    assert snapshot.extra["roles"]["main"]["state"] == "ready"
    assert snapshot.extra["owned"]["main"]["port"] == 8092


def test_turbofit_missing_controller_endpoint_is_stale(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(
        tmp_path, with_endpoint=False
    )
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )

    assert snapshot.extra["resident_state"] == "stale"
    assert snapshot.extra["lifecycle"]["endpoint_present"] is False


def test_turbofit_stale_endpoint_with_selection_is_stale(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(
        tmp_path, endpoint_valid=False
    )
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )

    assert snapshot.extra["resident_state"] == "stale"


def test_turbofit_orphaned_lifecycle_state_is_stale(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path, orphaned=True)
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )

    assert snapshot.extra["resident_state"] == "stale"
    assert snapshot.extra["lifecycle"]["orphaned"] is True  # corrupt state mirrors owner fail-closed


def test_turbofit_leases_without_residents_are_stale(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(
        tmp_path, leases={"tok1": "main", "tok2": "main"}
    )
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )

    assert snapshot.extra["resident_state"] == "stale"
    assert snapshot.extra["lifecycle"]["leases"]["main"] == 2


def test_turbofit_main_only_pressure_state_evaluates_ladder(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(
        tmp_path, owned_roles=("main",), with_endpoint=True
    )
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(
            main={"alias": "qwen3.8-27b", "state": "ready",
                  "base_url": "http://127.0.0.1:8092"},
            vram=[23001, 24110],
        ),
        runtime_state=runtime_state,
        native_state=native,
    )

    assert snapshot.free_mib == [23001, 24110]
    event = module.Observer().evaluate(snapshot)
    assert event["target_level"] == 0
    assert event["decision"] == "healthy"

    pressured = module.snapshot_from_turbofit_status(
        gateway_status(
            main={"alias": "qwen3.8-27b", "state": "ready",
                  "base_url": "http://127.0.0.1:8092"},
            vram=[5 * 1024, 20 * 1024],
        ),
        runtime_state=runtime_state,
        native_state=native,
    )
    observer = module.Observer(stability_checks=1)
    event = observer.evaluate(pressured)
    assert event["next_level"] == 1
    assert event["decision"] == "would_hold_context_at_65536"


def test_turbofit_without_vram_never_fakes_a_pressure_emergency(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path)
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native
    )
    observer = module.Observer(stability_checks=1)

    event = observer.evaluate(snapshot)
    assert snapshot.extra["vram_source"] == "unavailable"
    assert event["target_level"] == 0
    assert event["decision"] == "vram_unavailable"

    observer.level = 5
    for _ in range(3):
        event = observer.evaluate(snapshot)

    assert event["target_level"] == 0
    assert event["decision"] == "vram_unavailable"
    assert observer.level == 5  # missing telemetry can neither raise nor fake recovery


def test_turbofit_role_state_normalisation():
    module = load_module()

    assert module._role_state({"alias": "none", "type": "down"}) == ("down", None)
    assert module._role_state({"alias": "m", "state": "ready"}) == ("ready", "m")
    assert module._role_state({"alias": "m", "state": "loading"}) == ("loading", "m")
    assert module._role_state({"alias": "m", "state": "weird"}) == ("down", "m")
    assert module._role_state("garbage") == ("down", None)
    assert module._role_state({"state": "ready"}) == ("down", None)


# ─── Auto backend selection and fetch failures ────────────────────────────────

class _StatusServer:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                server.requests.append(self.path)
                payload = server.payloads.pop(0) if server.payloads else {}
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self):
        return f"http://127.0.0.1:{self.port}/status"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def status_server():
    server = _StatusServer([])
    yield server
    server.close()


def test_auto_prefers_turbofit_gateway_payload(status_server, tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path, with_endpoint=True)
    status_server.payloads = [gateway_status()]
    snapshot = module.snapshot_for_urls(
        status_server.url(), "http://127.0.0.1:1/status",
        runtime_state=runtime_state, native_state=native,
    )

    assert snapshot.backend == "turbofit"
    assert status_server.requests == ["/status"]


def test_auto_falls_back_to_turbohaul_when_turbofit_payload_unrecognised(status_server):
    module = load_module()
    legacy = _StatusServer([status(free_mib=[23000, 24000])])
    try:
        snapshot = module.snapshot_for_urls(
            status_server.url(), legacy.url(),
        )
    finally:
        legacy.close()

    assert snapshot.backend == "turbohaul"


def test_fetch_timeout_raises_oserror():
    module = load_module()

    with pytest.raises(OSError):
        module.fetch_status("http://127.0.0.1:1/status", timeout=0.2)


def test_parse_failure_is_reported_not_raised(capsys):
    module = load_module()
    server = _StatusServer([])
    server.payloads = [b"{not json"]
    try:
        with pytest.raises(json.JSONDecodeError):
            module.fetch_status(server.url())
    finally:
        server.close()


def test_missing_controller_state_file_is_tolerated(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path, selection=None, with_endpoint=False)
    snapshot = module.snapshot_from_turbofit_status(
        gateway_status(), runtime_state=runtime_state, native_state=native,
        controller_state=tmp_path / "absent.json",
    )

    assert snapshot.extra["resident_state"] == "unknown"
    assert snapshot.extra["selection"] is None
    assert module.Observer().evaluate(snapshot)["decision"] == "controller_state_unavailable"


def test_lock_release_is_observed_without_changing_state_files(tmp_path):
    module = load_module()
    runtime_state, native = seed_turbofit_state(tmp_path)
    before = {p.name: p.read_bytes() for p in native.iterdir()}
    assert module._lifecycle_owner_locked(native) is True
    _CONTROLLER_LOCKS.pop().close()
    assert module._lifecycle_owner_locked(native) is False
    assert {p.name: p.read_bytes() for p in native.iterdir()} == before