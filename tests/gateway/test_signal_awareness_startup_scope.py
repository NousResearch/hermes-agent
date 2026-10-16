"""Service-equivalent profile-scope coverage for Signal awareness startup."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_primary_startup_connect_binds_awareness_and_background_worker_scope(
    monkeypatch, tmp_path,
):
    from agent import secret_scope
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.signal.adapter import SignalAdapter
    import gateway.platforms.signal_reaction_awareness_runtime as awareness_runtime
    from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
    from gateway.run_startup import GatewayStartupMixin
    from hermes_constants import get_hermes_home

    primary = tmp_path / "primary"
    secondary = tmp_path / "profiles" / "secondary"
    primary.mkdir(parents=True)
    secondary.mkdir(parents=True)
    (primary / ".env").write_text(
        "HERMES_INFERENCE_PROVIDER=openai\n"
        "OPENAI_API_KEY=synthetic-primary\n"
        "OPENAI_BASE_URL=https://primary.invalid/v1\n"
        "HERMES_CODEX_BASE_URL=https://primary-codex.invalid/v1\n"
    )
    (secondary / ".env").write_text(
        "HERMES_INFERENCE_PROVIDER=openai\n"
        "OPENAI_API_KEY=synthetic-secondary\n"
        "OPENAI_BASE_URL=https://secondary.invalid/v1\n"
        "HERMES_CODEX_BASE_URL=https://secondary-codex.invalid/v1\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(primary))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)

    observed = {}
    owner = "00000000-0000-0000-0000-000000000002"
    configured = {
        "model": {"default": "openai/test-configured"},
        "signal": {"reaction_awareness": True},
    }
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: configured)

    class ModelRouteRunner:
        _provider_routing = {}
        @staticmethod
        def _profile_name_for_source(_source, adapter_profile=None):
            return adapter_profile or "main"
        def _resolve_session_agent_runtime(self, *, source=None, user_config=None):
            observed["resolver_codex_url"] = secret_scope.get_secret("HERMES_CODEX_BASE_URL", "")
            return _resolve_gateway_model(user_config), _resolve_runtime_agent_kwargs()
        @staticmethod
        def _resolve_turn_agent_config(_prompt, model, runtime):
            return {"model": model, "runtime": runtime}
    model_runner = ModelRouteRunner()
    signal_adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": "00000000-0000-0000-0000-000000000001"}))
    signal_adapter.dm_allow_from = {owner}
    signal_adapter.gateway_runner = model_runner

    async def connect(**_kwargs):
        await awareness_runtime.start_awareness(signal_adapter)
        service = awareness_runtime._services[id(signal_adapter)]
        observed["home"] = str(get_hermes_home().resolve())
        observed["codex_base"] = secret_scope.get_secret("HERMES_CODEX_BASE_URL", "")
        observed["runtime"] = _resolve_runtime_agent_kwargs()
        observed["model"] = service.lane.model_name
        observed["client"] = service.lane.model_client

        async def worker_probe(profile, scope, **authorization):
            assert authorization["authorized_sender"] == owner
            assert authorization["authorization_check"](owner)
            observed["worker_home"] = str(get_hermes_home().resolve())
            observed["worker_secret"] = secret_scope.get_secret("HERMES_CODEX_BASE_URL", "")
            # Make the synthetic queue terminal so the completion callback does not reschedule.
            service.lane.db.execute("UPDATE awareness_outbox SET status='completed'")
            service.lane.db.commit()
            return "empty"
        startup_ts = __import__('time').monotonic_ns()
        assert service.lane.register_target(profile=str(primary.resolve()),
            scope=next(iter(service.owner_scopes.values())), timestamp=startup_ts, author="bot", body="synthetic target")
        assert service.lane.enqueue({"owner_authorized": True, "chat_type": "dm",
            "profile": str(primary.resolve()), "scope": next(iter(service.owner_scopes.values())),
            "event_id": "synthetic-startup-" + str(startup_ts), "sender": owner, "target_author": "bot",
            "target_timestamp": startup_ts, "action": "added", "emoji": "👍", "event_timestamp": startup_ts + 1}) == "queued"
        service.lane.process_one = worker_probe
        service.schedule((str(primary.resolve()), next(iter(service.owner_scopes.values()))))
        await asyncio.gather(*tuple(service.tasks.values()))
        assert signal_adapter._reaction_awareness_callback is not None
        assert signal_adapter._reaction_awareness_note_getter is not None
        assert (primary / "signal-reaction-awareness.sqlite3").is_file()
        return True

    adapter = SimpleNamespace(connect=connect)

    from gateway.run_adapters import GatewayAdapterLifecycleMixin

    class ServiceEquivalentRunner(GatewayAdapterLifecycleMixin, GatewayStartupMixin):
        pass

    runner = ServiceEquivalentRunner()
    async def abort(*_args):
        return False
    runner._abort_startup_if_shutdown_requested = abort
    runner._startup_should_abort = lambda: False
    runner._update_platform_runtime_status = lambda *_args, **_kwargs: None
    runner._platform_lock_takeover_on_start = False
    runner._multiplex_on = lambda: True

    result = await GatewayStartupMixin._start_connect_pending(
        runner, [(Platform.SIGNAL, PlatformConfig(enabled=True), adapter)]
    )
    assert result and result[0][3] == "ok", result
    assert observed["home"] == str(primary.resolve())
    assert observed["runtime"]["base_url"] == "https://primary.invalid/v1"
    assert observed["resolver_codex_url"] == "https://primary-codex.invalid/v1"
    assert observed["codex_base"] == "https://primary-codex.invalid/v1"
    assert observed["worker_home"] == str(primary.resolve())
    assert observed["worker_secret"] == "https://primary-codex.invalid/v1"
    assert observed["model"] == "openai/test-configured"
    assert observed["client"].model == "openai/test-configured"
    assert observed["client"].runtime_kwargs["base_url"] == "https://primary.invalid/v1"

    # Multiplex profile isolation remains fail-closed and routes to the owning profile's endpoint.
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(secondary):
        secondary_runtime = _resolve_runtime_agent_kwargs()
        assert secondary_runtime["base_url"] == "https://secondary.invalid/v1"
        assert secret_scope.get_secret("HERMES_CODEX_BASE_URL", "") == "https://secondary-codex.invalid/v1"

    assert observed["runtime"]["base_url"] != secondary_runtime["base_url"]
    with pytest.raises(secret_scope.UnscopedSecretError):
        secret_scope.get_secret("HERMES_CODEX_BASE_URL", "")
    await awareness_runtime.stop_awareness(signal_adapter)
    assert signal_adapter._reaction_awareness_callback is None
    assert signal_adapter._reaction_awareness_note_getter is None

    # The primary watcher rebuilds and reconnects adapters outside startup's task scope.
    # A successful Signal connect starts awareness again, so retry must bind the same tenant.
    observed.clear()
    adapter.has_fatal_error = False
    adapter.fatal_error_retryable = True
    adapter.fatal_error_code = None
    adapter.fatal_error_message = None
    runner._running = True
    runner._failed_platforms = {
        Platform.SIGNAL: {
            "config": PlatformConfig(enabled=True), "attempts": 0,
            "next_retry": 0, "predecessor": None,
        }
    }
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._flag_reconnect_needs_attention = lambda *_args, **_kwargs: None
    async def run_in_executor_with_context(fn, *args):
        return fn(*args)
    runner._run_in_executor_with_context = run_in_executor_with_context
    runner._create_adapter = lambda *_args: adapter
    runner._wire_adapter_handlers = lambda *_args: None
    runner._platform_connect_timeout_secs = lambda *_args, **_kwargs: 0
    async def install_reconnected_adapter(*_args):
        return None
    runner._install_reconnected_adapter = install_reconnected_adapter
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *_args: True)

    await runner._reconnect_failed_platform(Platform.SIGNAL, 0)
    assert observed["home"] == str(primary.resolve())
    assert observed["resolver_codex_url"] == "https://primary-codex.invalid/v1"
    assert observed["runtime"]["base_url"] == "https://primary.invalid/v1"
    assert observed["worker_home"] == str(primary.resolve())
    assert observed["worker_secret"] == "https://primary-codex.invalid/v1"
    assert observed["model"] == "openai/test-configured"
    await awareness_runtime.stop_awareness(signal_adapter)
