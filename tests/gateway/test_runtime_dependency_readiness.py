"""A bound port, or an answering ``/health``, is not proof a gateway can accept a turn.

Verified incident: the gateway served ``/health`` (200) while ``import
pydantic_core._pydantic_core`` failed under the running interpreter, so every real request
died inside the provider client. These tests pin a side-effect-free local dependency probe:
it imports the client stack an API turn needs, names the adapter import each enabled
platform needs, reports the running interpreter/generation identity, and never contacts a
provider, reads user config, or leaks a path or credential into the authenticated payload.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import types
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import readiness
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import BasePlatformAdapter
from gateway.readiness import collect_dependency_readiness, collect_runtime_readiness
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status
from pm import environments

_RUNNING = {"gateway_state": "running", "platforms": {}, "updated_at": "2026-09-26T00:00:00Z"}


def _failing_import(module: str, error: Exception | None = None):
    """A probe import seam that fails for exactly one module (every other module imports for real)."""
    real = readiness._import_for_probe

    def _import(name: str):
        if name == module:
            raise error or ImportError(f"cannot load {module} for this interpreter")
        return real(name)

    return _import


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_provider_client_import_failure_degrades_readiness(tmp_path, monkeypatch):
    """The acceptance contract: a runtime whose provider client cannot import is not healthy,
    whatever ``/health`` answers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(readiness, "_import_for_probe", _failing_import("pydantic_core"))

    result = collect_runtime_readiness(configured_model="test/model", runtime_status=_RUNNING)

    assert result["status"] == "degraded"
    dependencies = result["checks"]["dependencies"]
    assert dependencies["status"] == "degraded"
    assert dependencies["checks"]["provider_client"]["status"] == "degraded"
    assert "pydantic_core" in dependencies["checks"]["provider_client"]["detail"]
    # Detail names the module and exception class only: no exception text, no paths.
    assert "ImportError" in dependencies["checks"]["provider_client"]["detail"]
    assert "cannot load" not in json.dumps(result)


def test_healthy_runtime_reports_ok_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = collect_runtime_readiness(configured_model="test/model", runtime_status=_RUNNING)

    dependencies = result["checks"]["dependencies"]
    assert dependencies["checks"]["provider_client"]["status"] == "ok"
    environment = dependencies["checks"]["environment"]
    assert environment["status"] == "ok"
    assert environment["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert environment["abi"] == sys.implementation.cache_tag
    assert "site-packages" not in json.dumps(dependencies)
    assert str(tmp_path) not in json.dumps(dependencies)


def test_enabled_api_adapter_import_failure_is_named(tmp_path, monkeypatch):
    """An enabled API adapter whose module cannot import is reported by platform name."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        readiness, "_import_for_probe", _failing_import("gateway.platforms.api_server")
    )

    result = collect_dependency_readiness(platforms=["telegram", "api_server"])

    assert result["status"] == "degraded"
    adapters = result["checks"]["adapters"]
    assert adapters["status"] == "degraded"
    assert "api_server" in adapters["detail"]
    assert "telegram" not in adapters["detail"]


def test_probe_is_side_effect_free(tmp_path, monkeypatch):
    """No config read, no socket, no environment or sys.path mutation: the probe is safe to run
    against a live gateway and against a candidate interpreter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    before_env = dict(os.environ)
    before_path = list(sys.path)

    def _no_config(*_args, **_kwargs):
        raise AssertionError("the dependency probe must not read user config")

    def _no_socket(*_args, **_kwargs):
        raise AssertionError("the dependency probe must not touch the network")

    monkeypatch.setattr(readiness, "load_yaml_file_readonly", _no_config)
    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)

    result = collect_dependency_readiness(platforms=["api_server"])

    assert result["status"] == "ok"
    assert dict(os.environ) == before_env
    assert sys.path == before_path


def test_wrong_dependency_generation_is_degraded(tmp_path, monkeypatch):
    """A committed generation this process is not running from is the ABI-mix signature."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "checkout"
    (root / "gateway").mkdir(parents=True)
    state = environments.install_state_dir(root)
    generation = state / "environments" / "gen-1"
    (generation / "Lib" / "site-packages").mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("version_info = 3.14\n", encoding="utf-8")
    (state / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(generation)}}}), encoding="utf-8"
    )

    result = collect_dependency_readiness(project_root=root)

    environment = result["checks"]["environment"]
    assert environment["status"] == "degraded"
    assert environment["committed_generation"] is True
    assert environment["on_selected_environment"] is False
    assert str(generation) not in json.dumps(result)


# ---------------------------------------------------------------------------
# Surfaces that consume the probe
# ---------------------------------------------------------------------------


def _create_health_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    return app


@pytest.mark.asyncio
async def test_health_detailed_reports_provider_import_failure(tmp_path, monkeypatch):
    """``/health`` stays the cheap liveness answer; the authenticated readiness surface tells the
    truth about the provider client."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(readiness, "_import_for_probe", _failing_import("pydantic_core"))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    app = _create_health_app(adapter)
    with patch("gateway.status.read_runtime_status", return_value=dict(_RUNNING)), patch(
        "gateway.run._resolve_gateway_model", return_value="test/model"
    ), patch(
        "gateway.readiness.shutil.disk_usage",
        return_value=types.SimpleNamespace(total=100, used=25, free=75),
    ):
        async with TestClient(TestServer(app)) as cli:
            liveness = await cli.get("/health")
            assert liveness.status == 200
            assert (await liveness.json())["status"] == "ok"

            detailed = await cli.get("/health/detailed")
            assert detailed.status == 200
            payload = await detailed.json()

    assert payload["status"] == "degraded"
    dependencies = payload["readiness"]["checks"]["dependencies"]
    assert dependencies["status"] == "degraded"
    assert "pydantic_core" in dependencies["checks"]["provider_client"]["detail"]


class _HealthyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, platforms) -> GatewayRunner:
    # gateway.run binds the PROCESS home (`_hermes_home`) and voice-mode state at import, before
    # any fixture runs, so a startup test would otherwise read real state (.clean_shutdown,
    # gateway_voice_mode.json). Redirect both to the test home first.
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    config = GatewayConfig(platforms=platforms, sessions_dir=tmp_path / "sessions")
    runner = GatewayRunner(config)

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


@pytest.mark.asyncio
async def test_startup_reports_degraded_when_the_client_stack_cannot_import(
    monkeypatch, tmp_path, caplog
):
    """The gateway must not stamp ``running`` when its own interpreter cannot import the provider
    client — it can answer a probe but never serve a turn."""
    runner = _runner(
        monkeypatch, tmp_path, {Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: _HealthyAdapter())
    monkeypatch.setattr(readiness, "_import_for_probe", _failing_import("pydantic_core"))

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True, "a degraded gateway keeps serving what it can; only status changes"
        assert read_runtime_status()["gateway_state"] == "degraded"
        assert any(
            record.levelno >= logging.ERROR
            and "pydantic_core" in record.getMessage()
            and "DEGRADED" in record.getMessage()
            for record in caplog.records
        ), "the startup failure must be logged at ERROR, naming the module that cannot import"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_enabled_platform_with_missing_dependencies_is_reported_not_omitted(
    monkeypatch, tmp_path, caplog
):
    """An enabled platform whose adapter cannot be built (aiohttp missing) was dropped with one
    WARNING while the run reported healthy. It must be a visible startup failure."""
    import gateway.platforms.api_server as api_server_module

    runner = _runner(
        monkeypatch,
        tmp_path,
        {
            Platform.API_SERVER: PlatformConfig(enabled=True, extra={"port": 8642}),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
    )
    real_create = runner._create_adapter
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, config: (
            _HealthyAdapter() if platform is Platform.TELEGRAM else real_create(platform, config)
        ),
    )
    monkeypatch.setattr(api_server_module, "AIOHTTP_AVAILABLE", False)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True
        state = read_runtime_status()
        assert state["gateway_state"] == "degraded"
        assert state["platforms"]["api_server"]["state"] == "fatal"
        assert state["platforms"]["api_server"]["error_code"] == "dependencies_unavailable"
        assert state["platforms"]["telegram"]["state"] == "connected"
        assert any(
            record.levelno >= logging.ERROR and "api_server" in record.getMessage()
            for record in caplog.records
        )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_sole_enabled_platform_unavailable_stays_running_for_cron(
    monkeypatch, tmp_path, caplog
):
    """The ONLY enabled platform cannot be built (its dependency is missing): the gateway must
    report ``degraded`` and keep running so cron jobs still execute. Exiting 78 here would take
    the whole process down — and the scheduler with it — over a per-platform dependency problem."""
    import gateway.platforms.api_server as api_server_module
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    runner = _runner(
        monkeypatch,
        tmp_path,
        {Platform.API_SERVER: PlatformConfig(enabled=True, extra={"port": 8642})},
    )
    monkeypatch.setattr(api_server_module, "AIOHTTP_AVAILABLE", False)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True, "an unavailable platform must not abort startup"
        state = read_runtime_status()
        assert state["gateway_state"] == "degraded"
        assert state["gateway_state"] != "startup_failed"
        assert state["platforms"]["api_server"]["state"] == "fatal"
        assert state["platforms"]["api_server"]["error_code"] == "dependencies_unavailable"
        assert runner.exit_code != GATEWAY_FATAL_CONFIG_EXIT_CODE
        assert runner.should_exit_cleanly is False
        assert runner._running is True, "cron/service gateway must keep running"
        assert any(
            record.levelno >= logging.ERROR and "api_server" in record.getMessage()
            for record in caplog.records
        )
    finally:
        await runner.stop()


def test_environment_probe_degrades_when_dependency_module_unimportable(tmp_path, monkeypatch):
    """``pm.environments`` failing to import must not escape the probe. The import lives inside
    the guarded block, so a broken dependency-generation module reports ``degraded`` instead of
    propagating out of ``collect_runtime_readiness`` (which 500s ``/health/detailed``)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    broken = types.ModuleType("pm.environments")  # present but missing its public names
    monkeypatch.setitem(sys.modules, "pm.environments", broken)

    result = collect_dependency_readiness(project_root=tmp_path)

    environment = result["checks"]["environment"]
    assert environment["status"] == "degraded"
    assert "ImportError" in environment["detail"]


@pytest.mark.asyncio
async def test_health_detailed_survives_unimportable_dependency_module(tmp_path, monkeypatch):
    """The authenticated readiness surface must answer 200 (degraded) — never 500 — when the
    dependency-generation module cannot be imported under the running interpreter."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pm.environments", types.ModuleType("pm.environments"))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    app = _create_health_app(adapter)
    with patch("gateway.status.read_runtime_status", return_value=dict(_RUNNING)), patch(
        "gateway.run._resolve_gateway_model", return_value="test/model"
    ), patch(
        "gateway.readiness.shutil.disk_usage",
        return_value=types.SimpleNamespace(total=100, used=25, free=75),
    ):
        async with TestClient(TestServer(app)) as cli:
            detailed = await cli.get("/health/detailed")
            assert detailed.status == 200
            payload = await detailed.json()

    assert payload["status"] == "degraded"
    environment = payload["readiness"]["checks"]["dependencies"]["checks"]["environment"]
    assert environment["status"] == "degraded"


class _NonRetryableConflictAdapter(BasePlatformAdapter):
    """A live foreign token holder at zero-connected startup: fatal, non-retryable."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram-bot-token_lock",
            "Telegram bot token already in use (PID 999).",
            retryable=True,  # emitted retryable; the startup router must still treat it as fatal
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_genuine_nonretryable_conflict_still_exits_ex_config(monkeypatch, tmp_path):
    """The degraded-not-exit change is scoped to *unavailable adapters* only. A real startup
    conflict with nothing else serving (live foreign token holder) must still exit 78 so the
    supervisor does not restart-loop a second writer onto a held token."""
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    runner = _runner(
        monkeypatch, tmp_path, {Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    monkeypatch.setattr(
        runner, "_create_adapter", lambda platform, config: _NonRetryableConflictAdapter()
    )

    ok = await runner.start()
    try:
        assert ok is True
        assert runner.should_exit_cleanly is True
        assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
        assert read_runtime_status()["gateway_state"] == "startup_failed"
    finally:
        await runner.stop()