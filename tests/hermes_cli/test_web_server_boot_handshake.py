"""Integration tests for the desktop backend readiness handshake."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import gateway.restart as restart
import hermes_cli.web_server as web_server_mod
from hermes_cli import __version__
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS

SLOW_SECONDS = 3


def _make_slow_drain(seconds: float):
    """Return a resolver replacement that sleeps in its worker thread."""
    def _slow():
        time.sleep(seconds)
        return 180.0

    return _slow


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("healthz touched a heavyweight status dependency")


def test_lifespan_has_no_gateway_module_warmup():
    assert not hasattr(web_server_mod, "_warm_gateway_module")


def test_get_status_does_not_block_event_loop():
    """Slow status config resolution must not block unrelated requests."""
    import httpx

    results: dict[str, float | int] = {}

    async def _run():
        transport = httpx.ASGITransport(app=web_server_mod.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with asyncio.TaskGroup() as tg:
                async def _status():
                    started = time.perf_counter()
                    response = await client.get(
                        "/api/status", timeout=SLOW_SECONDS + 5
                    )
                    results["status_ms"] = (
                        time.perf_counter() - started
                    ) * 1000
                    results["status_code"] = response.status_code

                async def _version():
                    await asyncio.sleep(0.1)
                    started = time.perf_counter()
                    response = await client.get("/api/version", timeout=5)
                    results["version_ms"] = (
                        time.perf_counter() - started
                    ) * 1000
                    results["version_code"] = response.status_code

                tg.create_task(_status())
                tg.create_task(_version())

    with patch.object(
        web_server_mod,
        "resolve_restart_drain_timeout",
        _make_slow_drain(SLOW_SECONDS),
    ):
        asyncio.run(_run())

    assert "version_ms" in results, "Fast endpoint never responded"
    assert "status_ms" in results, "/api/status never responded"
    assert float(results["version_ms"]) < SLOW_SECONDS * 1000
    assert results.get("status_code") == 200


def test_concurrent_status_probes_all_respond():
    """Concurrent status probes must all complete while resolution is slow."""
    import httpx

    responses: list[int] = []

    async def _run():
        transport = httpx.ASGITransport(app=web_server_mod.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            requests = [
                client.get("/api/status", timeout=SLOW_SECONDS + 5)
                for _ in range(3)
            ]
            results = await asyncio.gather(*requests, return_exceptions=True)
            for result in results:
                responses.append(
                    -1 if isinstance(result, Exception) else result.status_code
                )

    with patch.object(
        web_server_mod,
        "resolve_restart_drain_timeout",
        _make_slow_drain(SLOW_SECONDS),
    ):
        asyncio.run(_run())

    assert responses == [200, 200, 200]


def test_healthz_is_fixed_uncached_and_independent_of_status_state(
    monkeypatch, tmp_path
):
    """Readiness must not load config, database, gateway, or plugin state."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "this: [is: intentionally: invalid\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    for name in (
        "check_config_version",
        "get_running_pid_cached",
        "read_runtime_status",
        "load_config",
        "read_raw_config",
        "resolve_restart_drain_timeout",
    ):
        monkeypatch.setattr(web_server_mod, name, _fail_if_called)
    monkeypatch.setattr(
        web_server_mod, "_status_active_sessions", _fail_if_called
    )

    previous = getattr(web_server_mod.app.state, "auth_required", None)
    web_server_mod.app.state.auth_required = False
    try:
        response = TestClient(web_server_mod.app).get("/api/healthz")
    finally:
        web_server_mod.app.state.auth_required = previous

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "ready",
        "version": __version__,
    }
    assert response.headers["cache-control"] == "no-store"


def test_healthz_is_in_shared_public_allowlist():
    assert "/api/healthz" in PUBLIC_API_PATHS


@pytest.mark.parametrize("auth_required", [False, True])
def test_healthz_is_public_under_both_dashboard_auth_modes(auth_required):
    previous_required = getattr(
        web_server_mod.app.state, "auth_required", None
    )
    previous_host = getattr(web_server_mod.app.state, "bound_host", None)
    previous_port = getattr(web_server_mod.app.state, "bound_port", None)
    web_server_mod.app.state.auth_required = auth_required
    web_server_mod.app.state.bound_host = "127.0.0.1"
    web_server_mod.app.state.bound_port = 9119
    try:
        client = TestClient(
            web_server_mod.app, base_url="http://127.0.0.1:9119"
        )
        response = client.get("/api/healthz")
    finally:
        web_server_mod.app.state.auth_required = previous_required
        web_server_mod.app.state.bound_host = previous_host
        web_server_mod.app.state.bound_port = previous_port

    assert response.status_code == 200


def test_healthz_over_real_loopback_server_with_temp_hermes_home(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    ready_file = tmp_path / "backend-ready.json"
    env = os.environ.copy()
    env.update({
        "HERMES_DESKTOP_READY_FILE": str(ready_file),
        "HERMES_HOME": str(hermes_home),
        "HERMES_SERVE_HEADLESS": "1",
    })
    repo_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from hermes_cli.web_server import start_server; "
                "start_server(host='127.0.0.1', port=0, open_browser=False, "
                "headless=True)"
            ),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not ready_file.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)

        if not ready_file.exists():
            output = process.stdout.read() if process.poll() is not None else ""
            pytest.fail(
                f"backend did not publish its ready file; "
                f"exit={process.poll()} output={output}"
            )

        port = json.loads(ready_file.read_text(encoding="utf-8"))["port"]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/healthz", timeout=5
        ) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == {
                "ok": True,
                "status": "ready",
                "version": __version__,
            }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def test_restart_drain_timeout_resolver_precedence(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "agent:\n  restart_drain_timeout: 14\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)
    assert restart.resolve_restart_drain_timeout() == 14.0

    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "9")
    assert restart.resolve_restart_drain_timeout() == 9.0

    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "invalid")
    assert (
        restart.resolve_restart_drain_timeout()
        == restart.DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    )

    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT")
    (hermes_home / "config.yaml").unlink()
    assert (
        restart.resolve_restart_drain_timeout()
        == restart.DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    )


def test_restart_drain_timeout_zero_from_real_config(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "agent:\n  restart_drain_timeout: 0\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)

    assert restart.resolve_restart_drain_timeout() == 0.0
