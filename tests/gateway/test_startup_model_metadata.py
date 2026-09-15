"""Startup moves metadata I/O off the first turn, not inference or profile state."""

import asyncio
import json
import threading
from pathlib import Path

import pytest
import requests

from agent import secret_scope
from gateway import run as gateway_run
from gateway.run_model_context import _resolve_gateway_model_context
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [False, True])
async def test_startup_warms_profile_metadata_without_changing_context(tmp_path, monkeypatch, pinned):
    """Real config, credential and metadata resolution; only the HTTP transport is fake."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("STARTUP_MODEL_API_KEY", "wrong-process-profile")
    # Tool discovery is an independent, already-tested startup phase.
    monkeypatch.setattr(gateway_run, "_warm_turn_machinery_sync", lambda: 0)
    loop_thread = threading.get_ident()
    calls = []
    catalog = {}

    def send(_session, request, **_kwargs):
        assert request.method == "GET"
        entry, credential = catalog[request.url]
        assert request.headers["Authorization"] == "Bearer " + credential
        assert threading.get_ident() != loop_thread
        calls.append(request.url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"data": [entry]}).encode()
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    for index in range(2):
        profile = tmp_path / f"profile-{index}"
        profile.mkdir()
        model = f"startup-model-{index}"
        endpoint = f"https://profile-{index}.example.test/v1"
        credential = f"profile-{index}-fixture-key"
        live_context = 64000 + index * 1000
        configured_context = live_context // 2
        catalog[endpoint + "/models"] = ({"id": model, "context_length": live_context}, credential)
        model_config = {"default": model, "provider": "custom:startup-fixture", "base_url": endpoint}
        if pinned:
            model_config["context_length"] = configured_context
        (profile / "config.yaml").write_text(json.dumps({
            "model": model_config,
            "providers": {"startup-fixture": {"base_url": endpoint, "key_env": "STARTUP_MODEL_API_KEY"}},
        }), encoding="utf-8")
        home_token = set_hermes_home_override(profile)
        secret_token = secret_scope.set_secret_scope({"STARTUP_MODEL_API_KEY": credential})
        try:
            runner = object.__new__(gateway_run.GatewayRunner)
            before = len(calls)
            await runner._warm_turn_prerequisites()
            assert len(calls) == before + (0 if pinned else 1)
            # First-turn/display resolution must reuse the same metadata and honor pins.
            resolved = await asyncio.to_thread(_resolve_gateway_model_context)
            assert resolved.model == model
            assert resolved.base_url == endpoint
            assert resolved.context_length == (configured_context if pinned else live_context)
            assert len(calls) == before + (0 if pinned else 1)
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["error", "timeout", "disabled", "bedrock", "bedrock-url", "moa", "auth-failed"])
async def test_warmup_keeps_startup_bounded_and_never_runs_inference_probes(monkeypatch, case):
    """A failed/slow catalog cannot hold the gate; inference-based discovery stays lazy."""
    from agent import model_metadata

    started = threading.Event()
    release = threading.Event()
    calls = []
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()

    def metadata(*_args, **_kwargs):
        calls.append(True)
        started.set()
        loop.call_soon_threadsafe(entered.set)
        if case == "timeout":
            assert release.wait(5)
            return 64000
        raise OSError("fixture catalog unavailable")

    def runtime():
        if case == "auth-failed":
            raise RuntimeError("fixture credentials unavailable")
        return {
            "provider": case if case in {"bedrock", "moa"} else "custom",
            "base_url": ("https://bedrock-runtime.us-east-1.amazonaws.com" if case == "bedrock-url"
                         else "https://startup.example.test/v1"),
            "api_key": "fixture-key",
        }

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", runtime)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"model": {"default": "fixture-model"}})
    monkeypatch.setattr(gateway_run, "_warm_turn_machinery_sync", lambda: 0)
    monkeypatch.setattr(gateway_run, "_startup_warmup_timeout_secs",
                        lambda: 0 if case == "disabled" else .05 if case == "timeout" else 5)
    monkeypatch.setattr(model_metadata, "get_model_context_length", metadata)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._startup_restore_queue = []
    runner._start_startup_warmup()
    try:
        if case == "timeout":
            await asyncio.wait_for(entered.wait(), timeout=5)
        await asyncio.wait_for(runner._finish_startup_restore(), timeout=5)
        assert runner._startup_restore_in_progress is False
        assert bool(calls) == (case in {"error", "timeout"})
        if case == "timeout":
            assert started.is_set()
            assert not runner._startup_warmup_task.done()
        if case == "disabled":
            assert runner._startup_warmup_task is None
    finally:
        release.set()
        if runner._startup_warmup_task is not None:
            await asyncio.wait_for(runner._startup_warmup_task, timeout=5)
