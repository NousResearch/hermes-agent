from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import yaml


def _sealed_config() -> dict:
    platforms = {
        "api_server": {
            "enabled": True,
            "extra": {
                "host": "127.0.0.1",
                "port": 8642,
                "key_credential": "api-server.key",
            },
        },
        "relay": {
            "enabled": True,
            "extra": {
                "relay_url": "unix:///run/muncho-discord-connector/connector.sock"
            },
        },
    }
    return {
        "gateway": {"isolated_runtime": False, "platforms": platforms},
        "platforms": platforms,
    }


def test_capability_loader_pins_exact_plan_derived_bytes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import canonical_capability_canary_runtime as runtime
    from gateway import run
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    config = tmp_path / "gateway.yaml"
    value = _sealed_config()
    raw = yaml.safe_dump(value, sort_keys=True).encode()
    config.write_bytes(raw)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config)
    plan = SimpleNamespace(
        gateway_config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    observed = {}
    monkeypatch.setattr(runtime, "DEFAULT_GATEWAY_CONFIG", config)
    monkeypatch.setattr(runtime, "load_capability_plan", lambda: plan)
    monkeypatch.setattr(runtime, "render_gateway_config", lambda _plan: raw)
    monkeypatch.setattr(
        runtime,
        "validate_capability_gateway_config",
        lambda loaded: observed.update(validated=loaded),
    )
    monkeypatch.setattr(
        runtime,
        "capability_gateway_effective_environment_is_sealed",
        lambda env, loaded: observed.update(environment=(env, loaded)) or True,
    )
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    parsed = run._load_required_capability_gateway_config(str(config))
    assert set(parsed.platforms) == {run.Platform.API_SERVER, run.Platform.RELAY}
    assert observed["validated"] == value
    assert observed["environment"][1] == value
    assert config_module.load_config() == value


def test_capability_loader_rejects_any_post_plan_config_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import canonical_capability_canary_runtime as runtime
    from gateway import run
    from hermes_cli import managed_scope

    expected = yaml.safe_dump(_sealed_config(), sort_keys=True).encode()
    config = tmp_path / "gateway.yaml"
    config.write_bytes(expected + b"future_semantic_default: true\n")
    monkeypatch.setattr(runtime, "DEFAULT_GATEWAY_CONFIG", config)
    monkeypatch.setattr(
        runtime,
        "load_capability_plan",
        lambda: SimpleNamespace(
            gateway_config_sha256=hashlib.sha256(expected).hexdigest()
        ),
    )
    monkeypatch.setattr(runtime, "render_gateway_config", lambda _plan: expected)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    with pytest.raises(RuntimeError, match="config bytes drifted"):
        run._load_required_capability_gateway_config(str(config))


def test_gateway_main_uses_distinct_capability_contract(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    config = tmp_path / "gateway.yaml"
    config.write_text("{}\n", encoding="utf-8")
    captured = {}

    async def fake_start_gateway(parsed, **kwargs):
        captured["config"] = parsed
        captured["kwargs"] = kwargs
        return True

    expected = run.GatewayConfig.from_dict({})
    monkeypatch.setattr(run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(
        run,
        "_load_required_capability_gateway_config",
        lambda _path: expected,
    )
    monkeypatch.setattr(
        run,
        "_exit_after_graceful_shutdown",
        lambda code: captured.update(exit_code=code),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gateway.run",
            "--config",
            str(config),
            "--require-capability-canary",
        ],
    )
    run.main()
    assert captured == {
        "config": expected,
        "kwargs": {"require_capability_canary": True},
        "exit_code": 0,
    }


def test_capability_adapter_boundary_requires_live_privileged_relay() -> None:
    from gateway import run
    from gateway.canonical_writer_client import ExactServerMainPidAuthorizer
    from gateway.discord_connector_service import (
        DEFAULT_DISCORD_CONNECTOR_SOCKET,
        DEFAULT_DISCORD_CONNECTOR_UNIT,
        DiscordConnectorRuntime,
    )
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.discord_connector_transport import (
        DiscordConnectorRelayTransport,
    )

    authorizer = ExactServerMainPidAuthorizer(
        server_unit=DEFAULT_DISCORD_CONNECTOR_UNIT,
        expected_server_uid=501,
        main_pid_provider=SimpleNamespace(main_pid=lambda _unit: 123),
    )
    transport = DiscordConnectorRelayTransport(
        DEFAULT_DISCORD_CONNECTOR_SOCKET,
        server_authorizer=authorizer,
    )
    transport._connected = True
    transport._poller = SimpleNamespace(done=lambda: False)
    descriptor = CapabilityDescriptor.from_json(
        json.dumps(DiscordConnectorRuntime.descriptor())
    )
    relay = RelayAdapter(
        run.PlatformConfig(enabled=True),
        descriptor,
        transport=transport,
    )
    adapters = {
        run.Platform.API_SERVER: SimpleNamespace(),
        run.Platform.RELAY: relay,
    }
    assert run._capability_canary_adapters_are_ready(adapters) is True
    transport._connected = False
    assert run._capability_canary_adapters_are_ready(adapters) is False
    assert (
        run._capability_canary_adapters_are_ready({run.Platform.RELAY: relay}) is False
    )


@pytest.mark.parametrize("readiness_fails", (False, True))
def test_capability_execution_probe_is_last_gate_before_ready_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    readiness_fails: bool,
) -> None:
    from gateway import canonical_capability_canary_runtime as runtime
    from gateway import canonical_writer_boundary
    from gateway import canonical_writer_readiness
    from gateway import code_skew
    from gateway import run
    from gateway import status
    from hermes_cli import config as config_module
    from hermes_cli import plugins
    from hermes_cli import security_audit_startup
    from tools import skills_sync
    import hermes_logging

    events: list[str] = []
    plan = object()

    class FakeThread:
        def __init__(self, *args, name=None, **kwargs):
            self.name = name

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class FakeRunner:
        instance = None

        def __init__(self, config, **kwargs):
            self.config = config
            self.hooks = object()
            self.adapters = {"connected": object()}
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self._running = True
            self._restart_requested = False
            self._restart_via_service = False
            FakeRunner.instance = self

        async def start(self):
            events.append("runner.start")
            return True

        async def stop(self):
            events.append("runner.stop")
            self._running = False

        async def wait_for_shutdown(self):
            events.append("runner.wait_for_shutdown")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(code_skew, "record_boot_fingerprint", lambda: None)
    monkeypatch.setattr(
        canonical_writer_boundary,
        "harden_gateway_process_for_writer_boundary",
        lambda _policy: True,
    )
    monkeypatch.setattr(
        config_module,
        "effective_config_projection_is_pinned",
        lambda: True,
    )
    monkeypatch.setattr(
        config_module,
        "attest_pinned_effective_config_projection",
        lambda: {"pinned": True},
    )
    monkeypatch.setattr(config_module, "load_config", _sealed_config)
    monkeypatch.setattr(
        runtime,
        "validate_capability_gateway_config",
        lambda _value: None,
    )
    monkeypatch.setattr(runtime, "load_capability_plan", lambda: plan)
    monkeypatch.setattr(
        runtime,
        "validate_capability_extension_surface",
        lambda manager, hooks, *, plan: events.append("extensions.live"),
    )
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: object())
    monkeypatch.setattr(status, "get_running_pid", lambda: None)
    monkeypatch.setattr(status, "acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(status, "write_pid_file", lambda: None)
    monkeypatch.setattr(status, "remove_pid_file", lambda: events.append("pid.remove"))
    monkeypatch.setattr(
        status,
        "release_gateway_runtime_lock",
        lambda: events.append("lock.release"),
    )
    monkeypatch.setattr(skills_sync, "sync_skills", lambda **_kwargs: None)
    monkeypatch.setattr(
        hermes_logging,
        "setup_logging",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        security_audit_startup,
        "log_startup_security_warnings",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(run, "GatewayRunner", FakeRunner)
    monkeypatch.setattr(run.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        run.threading,
        "current_thread",
        lambda: SimpleNamespace(name="capability-test-thread"),
    )
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        canonical_writer_readiness,
        "attest_canonical_writer_startup_readiness",
        lambda: events.append("writer.attest") or {"ready": True},
    )

    def notify(_receipt, *, ready):
        events.append("writer.READY" if ready else "writer.unready")
        return True

    monkeypatch.setattr(
        canonical_writer_readiness,
        "notify_systemd_writer_readiness",
        notify,
    )
    monkeypatch.setattr(
        run,
        "_capability_canary_adapters_are_ready",
        lambda _adapters: events.append("adapters.live") or True,
    )

    def execution_readiness(observed_plan):
        assert observed_plan is plan
        events.append("execution.readiness")
        if readiness_fails:
            raise RuntimeError("worker/browser probe failed")
        return {"ok": True}

    monkeypatch.setattr(
        runtime,
        "attest_capability_execution_readiness",
        execution_readiness,
    )
    monkeypatch.setattr(
        run,
        "_start_gateway_cron_scheduler",
        lambda **_kwargs: (None, None),
    )

    result = asyncio.run(
        run.start_gateway(
            config=run.GatewayConfig.from_dict(_sealed_config()),
            verbosity=None,
            require_capability_canary=True,
        )
    )

    assert events.index("runner.start") < events.index("adapters.live")
    assert events.index("adapters.live") < events.index("execution.readiness")
    if readiness_fails:
        assert result is False
        assert "writer.READY" not in events
        assert events.index("execution.readiness") < events.index("runner.stop")
        assert "runner.wait_for_shutdown" not in events
    else:
        assert result is True
        assert events.index("execution.readiness") < events.index("writer.READY")
        assert events.index("writer.READY") < events.index(
            "runner.wait_for_shutdown"
        )


def test_startup_contract_booleans_are_strict_and_mutually_exclusive() -> None:
    from gateway import run

    with pytest.raises(TypeError, match="require_capability_canary"):
        asyncio.run(run.start_gateway(require_capability_canary=1))
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(
            run.start_gateway(
                require_canonical_writer=True,
                require_capability_canary=True,
            )
        )
