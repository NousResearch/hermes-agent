from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hermes_cli.local_model_status import (
    LocalModelProbeResult,
    LocalModelStatusMonitor,
    LocalModelState,
    build_local_model_display,
    build_local_model_route,
    probe_llamacpp_status,
    _resolve_probe_verify,
)


class _PropsHandler(BaseHTTPRequestHandler):
    payload = {
        "default_generation_settings": {"n_ctx": 131072},
        "is_sleeping": False,
    }
    expected_auth = "Bearer test-secret"
    expected_extra_header: tuple[str, str] | None = None
    response_status = 200
    raw_body: bytes | None = None
    paths: list[str] = []

    def do_GET(self):
        type(self).paths.append(self.path)
        if self.headers.get("Authorization") != self.expected_auth:
            self.send_response(401)
            self.end_headers()
            return
        if self.expected_extra_header is not None:
            name, value = self.expected_extra_header
            if self.headers.get(name) != value:
                self.send_response(403)
                self.end_headers()
                return
        if self.path not in {"/v1/props", "/props"}:
            self.send_response(404)
            self.end_headers()
            return
        if self.response_status != 200:
            self.send_response(self.response_status)
            self.end_headers()
            return
        body = self.raw_body if self.raw_body is not None else json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args
        return


def test_probe_reads_loaded_state_from_authenticated_llamacpp():
    _PropsHandler.response_status = 200
    _PropsHandler.payload = {
        "default_generation_settings": {"n_ctx": 131072},
        "is_sleeping": False,
    }
    _PropsHandler.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PropsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = build_local_model_route(
            provider="custom",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local.gguf",
            api_key="test-secret",
        )
        assert route is not None

        result = probe_llamacpp_status(route)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.state is LocalModelState.LOADED
    assert result.supported is True
    assert _PropsHandler.paths == ["/props"]
    assert "test-secret" not in repr(route)
    assert "test-secret" not in repr(result)


def test_probe_includes_configured_route_headers_without_exposing_them():
    _PropsHandler.response_status = 200
    _PropsHandler.expected_extra_header = ("X-Local-Proxy", "header-secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PropsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = build_local_model_route(
            provider="custom",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local.gguf",
            api_key="test-secret",
            extra_headers={"X-Local-Proxy": "header-secret"},
        )
        assert route is not None
        result = probe_llamacpp_status(route)
    finally:
        _PropsHandler.expected_extra_header = None
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.state is LocalModelState.LOADED
    assert "header-secret" not in repr(route)


def test_probe_reports_sleeping_only_when_llamacpp_confirms_it():
    _PropsHandler.response_status = 200
    _PropsHandler.payload = {
        "default_generation_settings": {"n_ctx": 131072},
        "is_sleeping": True,
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PropsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = build_local_model_route(
            provider="custom",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local.gguf",
            api_key="test-secret",
        )
        assert route is not None

        result = probe_llamacpp_status(route)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.state is LocalModelState.SLEEPING
    assert result.supported is True


def test_probe_reports_loading_on_props_service_unavailable():
    _PropsHandler.response_status = 503
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PropsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        route = build_local_model_route(
            provider="custom",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="local.gguf",
            api_key="test-secret",
        )
        assert route is not None
        result = probe_llamacpp_status(route)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.state is LocalModelState.LOADING
    assert result.supported is None


def test_probe_failures_never_report_sleeping():
    _PropsHandler.response_status = 200
    _PropsHandler.raw_body = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PropsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        unauthorized_route = build_local_model_route(
            provider="custom",
            base_url=base_url,
            model="local.gguf",
            api_key="wrong-secret",
        )
        assert unauthorized_route is not None
        unauthorized = probe_llamacpp_status(unauthorized_route)

        _PropsHandler.raw_body = b"{not-json"
        valid_auth_route = build_local_model_route(
            provider="custom",
            base_url=base_url,
            model="local.gguf",
            api_key="test-secret",
        )
        assert valid_auth_route is not None
        invalid_json = probe_llamacpp_status(valid_auth_route)

        _PropsHandler.raw_body = None
        _PropsHandler.payload = {"default_generation_settings": {"n_ctx": 131072}}
        missing_sleep_flag = probe_llamacpp_status(valid_auth_route)
    finally:
        _PropsHandler.raw_body = None
        _PropsHandler.payload = {
            "default_generation_settings": {"n_ctx": 131072},
            "is_sleeping": False,
        }
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    unavailable_route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:1/v1",
        model="local.gguf",
    )
    assert unavailable_route is not None
    transport_error = probe_llamacpp_status(unavailable_route, timeout=0.1)

    failures = (unauthorized, invalid_json, missing_sleep_flag, transport_error)
    assert all(result.state is LocalModelState.UNKNOWN for result in failures)
    assert all(result.state is not LocalModelState.SLEEPING for result in failures)


def test_monitor_probes_off_thread_and_keeps_one_request_in_flight():
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def _probe(_route):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=2)
        return LocalModelProbeResult(
            state=LocalModelState.LOADED,
            supported=True,
            checked_at=time.monotonic(),
        )

    route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="local.gguf",
        api_key="test-secret",
    )
    assert route is not None
    monitor = LocalModelStatusMonitor(probe=_probe, poll_interval=60)

    before = time.monotonic()
    initial = monitor.observe(route)
    elapsed = time.monotonic() - before
    assert elapsed < 0.1
    assert initial is not None
    assert initial.state is LocalModelState.UNKNOWN
    assert started.wait(timeout=1)

    monitor.observe(route)
    assert calls == 1

    release.set()
    deadline = time.monotonic() + 2
    current = monitor.observe(route)
    assert current is not None
    while current.state is not LocalModelState.LOADED and time.monotonic() < deadline:
        time.sleep(0.01)
        current = monitor.observe(route)
        assert current is not None

    assert current.state is LocalModelState.LOADED
    assert calls == 1


def test_monitor_auto_polls_without_additional_render_observations():
    second_probe = threading.Event()
    calls = 0

    def _probe(_route):
        nonlocal calls
        calls += 1
        if calls >= 2:
            second_probe.set()
        return LocalModelProbeResult(
            state=LocalModelState.LOADED,
            supported=True,
            checked_at=time.monotonic(),
        )

    route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="local.gguf",
    )
    assert route is not None
    monitor = LocalModelStatusMonitor(
        probe=_probe,
        poll_interval=0.1,
        auto_poll=True,
    )
    try:
        monitor.observe(route)
        assert second_probe.wait(timeout=1)
    finally:
        monitor.close()

    assert calls >= 2


def test_monitor_uses_exponential_backoff_after_probe_failures():
    now = [100.0]
    calls = 0

    def _probe(_route):
        nonlocal calls
        calls += 1
        return LocalModelProbeResult(
            state=LocalModelState.UNKNOWN,
            supported=None,
            checked_at=now[0],
        )

    route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="local.gguf",
    )
    assert route is not None
    monitor = LocalModelStatusMonitor(
        probe=_probe,
        poll_interval=5,
        max_backoff=30,
        clock=lambda: now[0],
    )

    def _wait_for_calls(expected):
        deadline = time.monotonic() + 1
        while calls < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == expected

    monitor.observe(route)
    _wait_for_calls(1)

    now[0] = 105.0
    monitor.observe(route)
    _wait_for_calls(2)

    now[0] = 110.0
    monitor.observe(route)
    time.sleep(0.05)
    assert calls == 2

    now[0] = 115.0
    monitor.observe(route)
    _wait_for_calls(3)

    switched_route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="other-local.gguf",
    )
    assert switched_route is not None
    now[0] = 116.0
    monitor.observe(switched_route)
    _wait_for_calls(4)

    # A new route gets a fresh backoff budget instead of inheriting the old
    # route's 30-second failure delay.
    now[0] = 121.0
    monitor.observe(switched_route)
    _wait_for_calls(5)


def test_monitor_uses_a_long_negative_cache_for_unsupported_routes():
    now = [100.0]
    calls = 0

    def _probe(_route):
        nonlocal calls
        calls += 1
        return LocalModelProbeResult(
            state=LocalModelState.UNKNOWN,
            supported=False,
            checked_at=now[0],
        )

    route = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="local.gguf",
    )
    assert route is not None
    monitor = LocalModelStatusMonitor(
        probe=_probe,
        poll_interval=5,
        unsupported_interval=300,
        clock=lambda: now[0],
    )

    monitor.observe(route)
    deadline = time.monotonic() + 1
    while calls < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == 1

    now[0] = 105.0
    assert monitor.observe(route) is None
    time.sleep(0.05)
    assert calls == 1

    now[0] = 400.0
    assert monitor.observe(route) is None
    deadline = time.monotonic() + 1
    while calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == 2


def test_loaded_display_uses_binary_bar_and_estimated_five_minute_countdown():
    loaded = LocalModelProbeResult(
        state=LocalModelState.LOADED,
        supported=True,
        checked_at=100.0,
    )

    active = build_local_model_display(
        loaded,
        idle_timeout_seconds=300,
        last_activity_at=100.0,
        now=117.0,
        turn_live=False,
    )
    expired_but_still_loaded = build_local_model_display(
        loaded,
        idle_timeout_seconds=300,
        last_activity_at=100.0,
        now=401.0,
        turn_live=False,
    )

    assert active.bar == "[██████████]"
    assert active.label == "~4m43s"
    assert expired_but_still_loaded.bar == "[██████████]"
    assert expired_but_still_loaded.label == "on"


def test_monitor_discards_a_stale_result_after_route_change():
    started = {"a": threading.Event(), "b": threading.Event()}
    release = {"a": threading.Event(), "b": threading.Event()}

    def _probe(route):
        suffix = route.model
        started[suffix].set()
        assert release[suffix].wait(timeout=2)
        return LocalModelProbeResult(
            state=LocalModelState.LOADED if suffix == "a" else LocalModelState.SLEEPING,
            supported=True,
            checked_at=time.monotonic(),
        )

    route_a = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="a",
    )
    route_b = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="b",
    )
    assert route_a is not None
    assert route_b is not None
    monitor = LocalModelStatusMonitor(probe=_probe, poll_interval=60)

    monitor.observe(route_a)
    assert started["a"].wait(timeout=1)
    monitor.observe(route_b)
    assert not started["b"].wait(timeout=0.05)

    release["a"].set()
    assert started["b"].wait(timeout=1)
    stale_snapshot = monitor.observe(route_b)
    assert stale_snapshot is not None
    assert stale_snapshot.state is LocalModelState.UNKNOWN

    release["b"].set()
    deadline = time.monotonic() + 2
    result = monitor.observe(route_b)
    assert result is not None
    while result.state is not LocalModelState.SLEEPING and time.monotonic() < deadline:
        time.sleep(0.01)
        result = monitor.observe(route_b)
        assert result is not None
    assert result.state is LocalModelState.SLEEPING


def test_monitor_serializes_an_a_b_a_route_cycle():
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []
    active = 0
    max_active = 0

    def _probe(route):
        nonlocal active, max_active
        calls.append(route.model)
        active += 1
        max_active = max(max_active, active)
        try:
            if len(calls) == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            return LocalModelProbeResult(
                state=LocalModelState.LOADED,
                supported=True,
                checked_at=time.monotonic(),
            )
        finally:
            active -= 1

    route_a = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="a",
    )
    route_b = build_local_model_route(
        provider="custom",
        base_url="http://127.0.0.1:8080/v1",
        model="b",
    )
    assert route_a is not None
    assert route_b is not None
    monitor = LocalModelStatusMonitor(probe=_probe, poll_interval=60)

    monitor.observe(route_a)
    assert first_started.wait(timeout=1)
    monitor.observe(route_b)
    monitor.observe(route_a)
    time.sleep(0.05)
    assert calls == ["a"]

    release_first.set()
    deadline = time.monotonic() + 1
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert calls == ["a", "a"]
    assert max_active == 1


def test_route_builder_rejects_cloud_and_non_http_endpoints():
    assert build_local_model_route(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-5",
    ) is None
    assert build_local_model_route(
        provider="moa",
        base_url="moa://local",
        model="ensemble",
    ) is None


def test_route_builder_preserves_localhost_for_https_certificates():
    route = build_local_model_route(
        provider="custom",
        base_url="https://localhost:8443/v1",
        model="local.gguf",
    )

    assert route is not None
    assert route.base_url == "https://localhost:8443/v1"


def test_probe_verify_uses_the_shared_provider_tls_resolution(monkeypatch):
    _resolve_probe_verify.cache_clear()
    calls = 0
    captured = {}

    def _resolve_httpx_verify(*, ca_bundle, ssl_verify, base_url):
        nonlocal calls
        calls += 1
        captured.update(
            ca_bundle=ca_bundle,
            ssl_verify=ssl_verify,
            base_url=base_url,
        )
        return False

    monkeypatch.setattr(
        "hermes_cli.local_model_status.resolve_httpx_verify",
        _resolve_httpx_verify,
    )

    args = ("https://localhost:8443/v1", "/tmp/test-ca.pem", False)
    assert _resolve_probe_verify(*args) is False
    assert _resolve_probe_verify(*args) is False
    assert calls == 1
    assert captured == {
        "ca_bundle": "/tmp/test-ca.pem",
        "ssl_verify": False,
        "base_url": "https://localhost:8443/v1",
    }
    _resolve_probe_verify.cache_clear()
