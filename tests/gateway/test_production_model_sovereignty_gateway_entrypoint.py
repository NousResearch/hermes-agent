from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml


_REVISION = "a" * 40


def _production_shaped_gateway_config() -> dict:
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
        "production_capabilities": {"sealed": True},
    }


def test_production_keeps_builtin_memory_without_external_provider() -> None:
    from gateway import run

    production = object.__new__(run.GatewayRunner)
    production._isolated_runtime = False
    production._require_production_model_sovereignty = True

    assert production._agent_startup_isolation_kwargs() == {
        "skip_memory": False,
        "skip_context_files": False,
    }


def _production_discord_source(user_id: str, *, trusted: bool = True):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="222222222222222222",
        chat_type="channel",
        user_id=user_id,
        delivered_via_upstream_relay=trusted,
    )


def test_production_owner_team_agent_projection_is_frozen_before_creation() -> None:
    from gateway import run
    from gateway.production_access_policy import (
        PRODUCTION_OWNER_DISCORD_USER_ID,
    )

    runner = object.__new__(run.GatewayRunner)
    runner._isolated_runtime = False
    runner._require_production_model_sovereignty = True
    original = [
        "skills",
        "session_search",
        "memory",
        "file",
        "todo",
        "code_execution",
    ]

    owner_source = _production_discord_source(
        PRODUCTION_OWNER_DISCORD_USER_ID
    )
    owner = runner._production_agent_access(original, owner_source)
    assert owner is not None
    assert owner.role == "owner"
    assert set(owner.enabled_toolsets) == {
        "skills",
        "session_search",
        "memory",
        "file",
        "todo",
        "cronjob",
    }
    assert "code_execution" not in owner.enabled_toolsets
    assert runner._agent_startup_isolation_kwargs(
        owner_source,
        production_access=owner,
    ) == {"skip_memory": False, "skip_context_files": False}

    team_source = _production_discord_source("1279454038731264062")
    team = runner._production_agent_access(original, team_source)
    assert team is not None
    assert team.role == "team"
    assert set(team.enabled_toolsets) == {"file", "skills_readonly", "todo"}
    assert runner._agent_startup_isolation_kwargs(
        team_source,
        production_access=team,
    ) == {"skip_memory": True, "skip_context_files": False}

    # The owner ID alone is not authority: only the connector-authenticated
    # source marker can grant owner capabilities.
    forged = runner._production_agent_access(
        original,
        _production_discord_source(
            PRODUCTION_OWNER_DISCORD_USER_ID,
            trusted=False,
        ),
    )
    assert forged is not None
    assert forged.role == "team"
    assert forged.skip_memory is True


def test_production_discord_resolution_cannot_recover_unreviewed_toolsets() -> None:
    from gateway.production_access_policy import (
        PRODUCTION_OWNER_DISCORD_USER_ID,
        project_production_agent_access,
    )
    from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS
    from hermes_cli.tools_config import _get_platform_tools

    config = {
        "platform_toolsets": {"discord": list(FIRST_WAVE_TOOLSETS)},
        "plugins": {"enabled": [], "disabled": []},
        "mcp_servers": {},
    }
    resolved = _get_platform_tools(config, "discord") | {
        "kanban",
        "code_execution",
    }
    # The production identity projection must strip any unreviewed surface
    # recovered by generic platform/plugin resolution before agent creation.

    owner = project_production_agent_access(
        resolved,
        _production_discord_source(PRODUCTION_OWNER_DISCORD_USER_ID),
    )
    assert owner is not None
    assert set(owner.enabled_toolsets) == set(FIRST_WAVE_TOOLSETS) | {"cronjob"}
    assert "kanban" not in owner.enabled_toolsets

    team = project_production_agent_access(
        resolved,
        _production_discord_source("1279454038731264062"),
    )
    assert team is not None
    assert "cronjob" not in team.enabled_toolsets
    assert "kanban" not in team.enabled_toolsets


def test_production_two_user_real_agent_tool_schemas_are_disjoint(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run
    from gateway.production_access_policy import (
        PRODUCTION_OWNER_DISCORD_USER_ID,
    )
    from run_agent import AIAgent

    runner = object.__new__(run.GatewayRunner)
    runner._isolated_runtime = False
    runner._require_production_model_sovereignty = True
    configured = [
        "skills",
        "session_search",
        "memory",
        "file",
        "todo",
        "code_execution",
    ]
    owner_source = _production_discord_source(
        PRODUCTION_OWNER_DISCORD_USER_ID
    )
    team_source = _production_discord_source("1279454038731264062")
    owner_access = runner._production_agent_access(configured, owner_source)
    team_access = runner._production_agent_access(configured, team_source)
    assert owner_access is not None and team_access is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    from gateway.session_context import clear_session_vars, set_session_vars

    session_tokens = set_session_vars(platform="discord")

    common = {
        "model": "anthropic/claude-sonnet-4",
        "api_key": "test",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "max_iterations": 2,
        "quiet_mode": True,
        "skip_context_files": True,
    }
    owner_agent = AIAgent(
        **common,
        enabled_toolsets=list(owner_access.enabled_toolsets),
        skip_memory=owner_access.skip_memory,
        platform="discord",
        user_id=PRODUCTION_OWNER_DISCORD_USER_ID,
    )
    team_agent = AIAgent(
        **common,
        enabled_toolsets=list(team_access.enabled_toolsets),
        skip_memory=team_access.skip_memory,
        platform="discord",
        user_id="1279454038731264062",
    )
    try:
        assert {"memory", "session_search", "skill_manage"}.issubset(
            owner_agent.valid_tool_names
        )
        assert "cronjob" in owner_agent.valid_tool_names
        assert "execute_code" not in owner_agent.valid_tool_names
        assert not (
            {"memory", "session_search", "skill_manage"}
            & team_agent.valid_tool_names
        )
        assert {"skills_list", "skill_view"}.issubset(
            team_agent.valid_tool_names
        )
        assert "cronjob" not in team_agent.valid_tool_names
        assert owner_agent._cached_system_prompt is None
        assert team_agent._cached_system_prompt is None
    finally:
        owner_agent.close()
        team_agent.close()
        clear_session_vars(session_tokens)


def test_production_loader_pins_exact_path_digest_and_effective_mapping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import run
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    config = tmp_path / "config.yaml"
    value = _production_shaped_gateway_config()
    raw = yaml.safe_dump(value, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    config.write_bytes(raw)
    observed: dict[str, object] = {}

    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config)
    monkeypatch.setattr(runtime, "PRODUCTION_CONFIG_PATH", config)
    monkeypatch.setattr(runtime, "load_strict_production_config", lambda _raw: value)
    monkeypatch.setattr(
        runtime,
        "validate_production_gateway_config",
        lambda loaded: observed.update(config=loaded),
    )
    monkeypatch.setattr(
        runtime,
        "validate_production_gateway_environment",
        lambda environ, **bindings: observed.update(
            environment=environ,
            bindings=bindings,
        ),
    )
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    parsed = run._load_required_production_gateway_config(
        str(config),
        revision=_REVISION,
        config_sha256=digest,
    )

    assert set(parsed.platforms) == {run.Platform.API_SERVER, run.Platform.RELAY}
    assert observed["config"] == value
    assert observed["bindings"] == {
        "revision": _REVISION,
        "config_sha256": digest,
        "topology": {"sealed": True},
    }
    assert config_module.load_config() == value


def test_production_loader_rejects_path_and_digest_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import run
    from hermes_cli import managed_scope

    exact = tmp_path / "exact.yaml"
    other = tmp_path / "other.yaml"
    raw = b"platforms: {}\n"
    exact.write_bytes(raw)
    other.write_bytes(raw)
    monkeypatch.setattr(runtime, "PRODUCTION_CONFIG_PATH", exact)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)

    with pytest.raises(RuntimeError, match="path is not exact"):
        run._load_required_production_gateway_config(
            str(other),
            revision=_REVISION,
            config_sha256=hashlib.sha256(raw).hexdigest(),
        )
    with pytest.raises(RuntimeError, match="digest drifted"):
        run._load_required_production_gateway_config(
            str(exact),
            revision=_REVISION,
            config_sha256="0" * 64,
        )


def test_gateway_main_passes_exact_production_bindings(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    expected = run.GatewayConfig.from_dict(_production_shaped_gateway_config())
    captured: dict[str, object] = {}

    async def fake_start_gateway(parsed, **kwargs):
        captured["config"] = parsed
        captured["kwargs"] = kwargs
        return True

    def fake_loader(path, **bindings):
        captured["loader"] = (path, bindings)
        return expected

    monkeypatch.setattr(run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(run, "_load_required_production_gateway_config", fake_loader)
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
            "--require-production-model-sovereignty",
            "--production-release-revision",
            _REVISION,
            "--production-config-sha256",
            digest,
        ],
    )

    run.main()

    assert captured == {
        "loader": (
            str(config),
            {"revision": _REVISION, "config_sha256": digest},
        ),
        "config": expected,
        "kwargs": {
            "require_production_model_sovereignty": True,
            "production_release_revision": _REVISION,
            "production_config_sha256": digest,
        },
        "exit_code": 0,
    }


def test_production_startup_contract_arguments_fail_closed_before_side_effects() -> None:
    from gateway import run

    with pytest.raises(TypeError, match="require_production_model_sovereignty"):
        asyncio.run(run.start_gateway(require_production_model_sovereignty=1))
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(
            run.start_gateway(
                require_capability_canary=True,
                require_production_model_sovereignty=True,
                production_release_revision=_REVISION,
                production_config_sha256="0" * 64,
            )
        )
    with pytest.raises(TypeError, match="production_release_revision"):
        asyncio.run(run.start_gateway(require_production_model_sovereignty=True))
    with pytest.raises(ValueError, match="bindings require"):
        asyncio.run(run.start_gateway(production_release_revision=_REVISION))


def test_production_relay_registration_is_exact_and_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import relay
    from gateway import run

    observed: list[tuple[bool, str | None]] = []

    def successful_registration(*, force=False, url=None):
        observed.append((force, url))
        return True

    monkeypatch.setattr(relay, "register_relay_adapter", successful_registration)
    run._register_production_relay_adapter()
    assert observed == [(True, runtime.RELAY_URL)]

    monkeypatch.setattr(relay, "register_relay_adapter", lambda **_kwargs: False)
    with pytest.raises(RuntimeError, match="returned false"):
        run._register_production_relay_adapter()


def test_production_pre_ready_attestation_covers_all_live_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import production_capability_prerequisites as prerequisites
    from gateway import production_execution_readiness as execution_readiness
    from gateway import run
    from hermes_cli import plugins

    calls: list[object] = []
    runner = SimpleNamespace(hooks=object(), adapters={"live": object()})
    plugin_manager = object()
    jobs = [{"id": "mechanical"}]
    topology = {
        "isolated_worker": {
            "socket_path": "/run/worker.sock",
            "server_uid": 1001,
            "server_gid": 1002,
            "socket_uid": 0,
            "socket_gid": 1003,
        },
        "browser": {"socket_path": "/run/browser.sock", "service_uid": 1004},
    }
    browser_client_config = object()

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: plugin_manager)
    monkeypatch.setattr(
        runtime,
        "validate_production_extension_surface",
        lambda manager, hooks, providers: calls.append(
            ("extensions", manager, hooks, providers)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "validate_production_gateway_adapters",
        lambda adapters: calls.append(("adapters", adapters)),
    )
    monkeypatch.setattr(
        runtime,
        "validate_production_cron_jobs",
        lambda observed: calls.append(("cron", observed)),
    )
    monkeypatch.setattr(
        runtime,
        "attest_current_production_release_import_identity",
        lambda **bindings: calls.append(("release", bindings)) or {"ok": True},
    )
    monkeypatch.setattr(
        runtime,
        "attest_current_production_capability_prerequisites",
        lambda **bindings: calls.append(("prerequisites", bindings))
        or {
            "ready": True,
            "topology_identity_sha256": "f" * 64,
            "lifecycle_phase": prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED,
        },
    )
    monkeypatch.setattr(
        runtime,
        "attest_current_production_gateway_service_identity",
        lambda **bindings: calls.append(("gateway_service", bindings))
        or {"process_identity_matches_unit": True},
    )
    monkeypatch.setattr(
        runtime,
        "production_browser_controller_client_config",
        lambda value: calls.append(("browser_config", value))
        or browser_client_config,
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_production_capability_topology",
        lambda value: calls.append(("topology", value)) or value,
    )
    monkeypatch.setattr(
        prerequisites,
        "production_capability_topology_identity_sha256",
        lambda value: "f" * 64,
    )
    worker_receipt = {
        "schema": "muncho-production-isolated-worker-readiness.v1",
        "lease_identity_sha256": "1" * 64,
        "socket_path": "/run/worker.sock",
        "server_uid": 1001,
        "server_gid": 1002,
        "socket_uid": 0,
        "socket_gid": 1003,
        "execution_round_trip": True,
        "output_sha256": "2" * 64,
        "secret_material_recorded": False,
    }
    browser_receipt = {
        "schema": "muncho-production-browser-controller-readiness.v1",
        "session_identity_sha256": "3" * 64,
        "socket_path": "/run/browser.sock",
        "server_uid": 1004,
        "command_round_trip": True,
        "secret_material_recorded": False,
    }
    monkeypatch.setattr(
        execution_readiness,
        "attest_isolated_worker_execution",
        lambda **bindings: calls.append(("worker_execution", bindings))
        or worker_receipt,
    )
    monkeypatch.setattr(
        execution_readiness,
        "attest_browser_controller_execution",
        lambda **bindings: calls.append(("browser_execution", bindings))
        or browser_receipt,
    )
    monkeypatch.setattr(run, "_load_stable_production_cron_jobs", lambda: jobs)

    receipt = run._attest_production_gateway_before_ready(
        runner,
        revision=_REVISION,
        config_sha256="0" * 64,
        topology=topology,
    )

    assert [call[0] for call in calls] == [
        "extensions",
        "adapters",
        "cron",
        "release",
        "prerequisites",
        "gateway_service",
        "topology",
        "worker_execution",
        "browser_config",
        "browser_execution",
    ]
    assert calls[0][1:3] == (plugin_manager, runner.hooks)
    assert calls[1] == ("adapters", runner.adapters)
    assert calls[2] == ("cron", jobs)
    assert calls[7] == (
        "worker_execution",
        {
            "socket_path": Path("/run/worker.sock"),
            "server_uid": 1001,
            "server_gid": 1002,
            "socket_uid": 0,
            "socket_gid": 1003,
            "revision": _REVISION,
            "config_sha256": "0" * 64,
        },
    )
    assert calls[8] == ("browser_config", topology)
    assert calls[9] == (
        "browser_execution",
        {
            "client_config": browser_client_config,
            "revision": _REVISION,
            "config_sha256": "0" * 64,
        },
    )
    assert receipt == {
        "browser_execution": browser_receipt,
        "config_sha256": "0" * 64,
        "gateway_service": {"process_identity_matches_unit": True},
        "prerequisites": {
            "ready": True,
            "topology_identity_sha256": "f" * 64,
            "lifecycle_phase": prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED,
        },
        "release": {"ok": True},
        "worker_execution": worker_receipt,
    }


@pytest.mark.parametrize("failed_boundary", ["worker", "browser"])
def test_production_execution_round_trip_failure_keeps_pre_ready_unattested(
    monkeypatch: pytest.MonkeyPatch,
    failed_boundary: str,
) -> None:
    from gateway import production_capability_prerequisites as prerequisites
    from gateway import production_execution_readiness as execution_readiness
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import run
    from hermes_cli import plugins

    topology = {
        "isolated_worker": {
            "socket_path": "/run/worker.sock",
            "server_uid": 1001,
            "server_gid": 1002,
            "socket_uid": 0,
            "socket_gid": 1003,
        },
        "browser": {"socket_path": "/run/browser.sock", "service_uid": 1004},
    }
    worker_receipt = {
        "schema": "muncho-production-isolated-worker-readiness.v1",
        "lease_identity_sha256": "1" * 64,
        "socket_path": "/run/worker.sock",
        "server_uid": 1001,
        "server_gid": 1002,
        "socket_uid": 0,
        "socket_gid": 1003,
        "execution_round_trip": True,
        "output_sha256": "2" * 64,
        "secret_material_recorded": False,
    }

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: object())
    monkeypatch.setattr(runtime, "validate_production_extension_surface", lambda *_a: None)
    monkeypatch.setattr(runtime, "validate_production_gateway_adapters", lambda *_a: None)
    monkeypatch.setattr(runtime, "validate_production_cron_jobs", lambda *_a: None)
    monkeypatch.setattr(
        runtime,
        "attest_current_production_release_import_identity",
        lambda **_k: {},
    )
    monkeypatch.setattr(
        runtime,
        "attest_current_production_capability_prerequisites",
        lambda **_k: {
            "topology_identity_sha256": "f" * 64,
            "lifecycle_phase": prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED,
        },
    )
    monkeypatch.setattr(
        runtime,
        "attest_current_production_gateway_service_identity",
        lambda **_k: {},
    )
    monkeypatch.setattr(
        runtime,
        "production_browser_controller_client_config",
        lambda _topology: object(),
    )
    monkeypatch.setattr(
        prerequisites,
        "validate_production_capability_topology",
        lambda value: value,
    )
    monkeypatch.setattr(
        prerequisites,
        "production_capability_topology_identity_sha256",
        lambda _value: "f" * 64,
    )
    monkeypatch.setattr(run, "_load_stable_production_cron_jobs", lambda: [])

    if failed_boundary == "worker":
        monkeypatch.setattr(
            execution_readiness,
            "attest_isolated_worker_execution",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("worker offline")),
        )
        monkeypatch.setattr(
            execution_readiness,
            "attest_browser_controller_execution",
            lambda **_k: pytest.fail("browser probe must not run"),
        )
    else:
        monkeypatch.setattr(
            execution_readiness,
            "attest_isolated_worker_execution",
            lambda **_k: worker_receipt,
        )
        monkeypatch.setattr(
            execution_readiness,
            "attest_browser_controller_execution",
            lambda **_k: (_ for _ in ()).throw(RuntimeError("browser offline")),
        )

    with pytest.raises(RuntimeError, match=f"{failed_boundary} offline"):
        run._attest_production_gateway_before_ready(
            SimpleNamespace(hooks=object(), adapters={}),
            revision=_REVISION,
            config_sha256="0" * 64,
            topology=topology,
        )


def test_production_cron_boundary_latches_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cron import production_policy
    from gateway import run

    active = False

    def activate() -> None:
        nonlocal active
        active = True

    monkeypatch.setattr(production_policy, "activate_production_cron_policy", activate)
    monkeypatch.setattr(
        production_policy,
        "production_cron_policy_active",
        lambda: active,
    )
    run._activate_production_cron_boundary()
    assert active is True

    monkeypatch.setattr(
        production_policy,
        "production_cron_policy_active",
        lambda: False,
    )
    with pytest.raises(RuntimeError, match="did not activate"):
        run._activate_production_cron_boundary()


def test_production_cron_boundary_activates_before_runner_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No adapter or inbound API can exist before the write latch is active."""
    from gateway import canonical_writer_boundary
    from gateway import code_skew
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import run
    from gateway import status
    from hermes_cli import config as config_module
    from hermes_cli import security_audit_startup
    from tools import browser_controller_client
    from tools import skills_sync
    import hermes_logging

    class ConstructionReached(RuntimeError):
        pass

    events: list[str] = []

    monkeypatch.setattr(code_skew, "record_boot_fingerprint", lambda: None)
    monkeypatch.setattr(
        canonical_writer_boundary,
        "harden_gateway_process_for_writer_boundary",
        lambda _policy: True,
    )
    monkeypatch.setattr(config_module, "effective_config_projection_is_pinned", lambda: True)
    monkeypatch.setattr(
        config_module,
        "attest_pinned_effective_config_projection",
        lambda: {"pinned": True},
    )
    monkeypatch.setattr(config_module, "load_config", _production_shaped_gateway_config)
    monkeypatch.setattr(runtime, "validate_production_gateway_config", lambda _cfg: None)
    monkeypatch.setattr(
        runtime,
        "validate_production_gateway_environment",
        lambda _env, **_bindings: None,
    )
    monkeypatch.setattr(status, "get_running_pid", lambda: None)
    monkeypatch.setattr(skills_sync, "sync_skills", lambda **_kwargs: None)
    monkeypatch.setattr(hermes_logging, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        security_audit_startup,
        "log_startup_security_warnings",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run,
        "_activate_production_cron_boundary",
        lambda: events.append("cron_boundary"),
    )
    monkeypatch.setattr(
        browser_controller_client,
        "activate_browser_controller_required",
        lambda: events.append("browser_boundary"),
    )

    def stop_at_runner(*_args, **_kwargs):
        events.append("runner")
        raise ConstructionReached

    monkeypatch.setattr(run, "GatewayRunner", stop_at_runner)
    config = run.GatewayConfig.from_dict(_production_shaped_gateway_config())

    with pytest.raises(ConstructionReached):
        asyncio.run(
            run.start_gateway(
                config=config,
                verbosity=None,
                require_production_model_sovereignty=True,
                production_release_revision=_REVISION,
                production_config_sha256="0" * 64,
            )
        )

    assert events == ["cron_boundary", "browser_boundary", "runner"]


def test_production_cron_store_read_is_stable_and_nonrepairing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_model_sovereignty_runtime as runtime
    from gateway import run

    monkeypatch.setattr(runtime, "PRODUCTION_HOME", tmp_path)
    assert run._load_stable_production_cron_jobs() == []

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    jobs = [{"id": "mechanical", "enabled": False}]
    path = cron_dir / "jobs.json"
    original = json.dumps({"jobs": jobs, "updated_at": "fixed"}).encode()
    path.write_bytes(original)

    assert run._load_stable_production_cron_jobs() == jobs
    assert path.read_bytes() == original


def test_production_runtime_ignores_persisted_session_and_channel_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    runner = object.__new__(run.GatewayRunner)
    runner._require_production_model_sovereignty = True
    runner._session_model_overrides = {
        "session": {
            "model": "stale-model",
            "provider": "stale-provider",
            "api_key": "must-not-be-used",
        }
    }
    runner.config = None
    runner._last_resolved_model = {}
    runner._rehydrate_session_model_override = MagicMock()
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda _config: "gpt-5.6-sol")
    monkeypatch.setattr(
        run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )

    model, runtime = runner._resolve_session_agent_runtime(
        session_key="session",
        user_config={},
    )

    assert model == "gpt-5.6-sol"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"
    assert "api_key" not in runtime
    runner._rehydrate_session_model_override.assert_not_called()


def test_production_runtime_ignores_session_reasoning_override() -> None:
    from gateway import run

    runner = object.__new__(run.GatewayRunner)
    runner._require_production_model_sovereignty = True
    runner._session_reasoning_overrides = {"session": {"effort": "none"}}
    runner._load_reasoning_config = MagicMock(return_value={"effort": "high"})

    assert runner._resolve_session_reasoning_config(session_key="session") == {
        "effort": "high"
    }
    runner._load_reasoning_config.assert_called_once_with()


def _slash_test_runner():
    from gateway import run

    runner = object.__new__(run.GatewayRunner)
    runner.config = run.GatewayConfig.from_dict({})
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._require_production_model_sovereignty = True
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()
    return runner


@pytest.mark.parametrize(
    "command",
    ["commands", "help", "status", "usage", "version", "whoami"],
)
def test_production_team_slash_policy_allows_only_exact_read_only_set(
    command: str,
) -> None:
    runner = _slash_test_runner()
    source = _production_discord_source("1279454038731264062")

    assert runner._check_slash_access(source, command) is None


@pytest.mark.parametrize(
    "command",
    [
        "approve",
        "background",
        "cron",
        "memory",
        "model",
        "new",
        "reload-skills",
        "restart",
        "skills",
        "snapshot",
        "tools",
        "yolo",
    ],
)
def test_production_team_slash_policy_denies_mutating_and_global_commands(
    command: str,
) -> None:
    runner = _slash_test_runner()
    source = _production_discord_source("1279454038731264062")

    denial = runner._check_slash_access(source, command)

    assert denial is not None
    assert "owner-only" in denial


def test_production_owner_slash_policy_requires_connector_identity() -> None:
    from gateway.production_access_policy import (
        PRODUCTION_OWNER_DISCORD_USER_ID,
    )

    runner = _slash_test_runner()
    trusted = _production_discord_source(PRODUCTION_OWNER_DISCORD_USER_ID)
    forged = _production_discord_source(
        PRODUCTION_OWNER_DISCORD_USER_ID,
        trusted=False,
    )

    assert runner._check_slash_access(trusted, "approve") is None
    assert "owner-only" in runner._check_slash_access(forged, "approve")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    sorted(
        {
            "model",
            "moa",
            "codex-runtime",
            "fast",
            "reasoning",
            "yolo",
            "reload-mcp",
        }
    ),
)
async def test_production_route_changing_slash_commands_are_blocked(
    command: str,
) -> None:
    from gateway import run
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _slash_test_runner()
    event = MessageEvent(
        text=f"/{command}",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=run.Platform.API_SERVER,
            chat_id="control",
            chat_type="channel",
            user_id="owner",
        ),
    )

    assert await runner._handle_message(event) == run._PRODUCTION_ROUTE_CHANGE_BLOCKED

    session_key = runner._session_key_for_source(event.source)
    active = MagicMock()
    active.get_activity_summary.return_value = {"seconds_since_activity": 0}
    runner._running_agents[session_key] = active
    runner._running_agents_ts[session_key] = time.time()
    assert await runner._handle_message(event) == run._PRODUCTION_ROUTE_CHANGE_BLOCKED
    active.interrupt.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "/personality helpful",
        "/footer",
        "/footer on",
        "/verbose",
        "/memory approval on",
        "/skills mode off",
    ],
)
async def test_production_config_mutation_slash_arms_are_blocked(
    text: str,
) -> None:
    from gateway import run
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    runner = _slash_test_runner()
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=run.Platform.API_SERVER,
            chat_id="control",
            chat_type="channel",
            user_id="owner",
        ),
    )

    assert await runner._handle_message(event) == run._PRODUCTION_CONFIG_CHANGE_BLOCKED

    session_key = runner._session_key_for_source(event.source)
    active = MagicMock()
    active.get_activity_summary.return_value = {"seconds_since_activity": 0}
    runner._running_agents[session_key] = active
    runner._running_agents_ts[session_key] = time.time()
    assert await runner._handle_message(event) == run._PRODUCTION_CONFIG_CHANGE_BLOCKED
    active.interrupt.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "/personality",
        "/footer status",
        "/memory pending",
        "/memory approve exact-id",
        "/skills pending",
        "/skills reject exact-id",
    ],
)
def test_production_read_only_or_review_slash_arms_do_not_match_config_mutation(
    text: str,
) -> None:
    from gateway import run
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from hermes_cli.commands import resolve_command

    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=run.Platform.API_SERVER,
            chat_id="control",
            chat_type="channel",
            user_id="owner",
        ),
    )
    command = event.get_command()
    resolved = resolve_command(command)

    assert resolved is not None
    assert run._production_slash_requests_config_mutation(
        resolved.name,
        event,
    ) is False


@pytest.mark.asyncio
async def test_blocked_config_command_preserves_pin_and_next_turn_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    config_path = tmp_path / "config.yaml"
    exact = {
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "agent": {
            "personalities": {
                "helpful": {"system_prompt": "must-not-be-applied"}
            }
        },
    }
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=digest,
        exact_mapping=exact,
    )

    runner = _slash_test_runner()
    runner._external_drain_active = False
    runner._claim_active_session_slot = MagicMock(return_value=(None, None))
    runner._persist_active_agents = MagicMock()
    runner._begin_session_run_generation = MagicMock(return_value=1)
    runner._handle_message_with_agent = AsyncMock(return_value="model-dispatched")
    source = SessionSource(
        platform=run.Platform.API_SERVER,
        chat_id="control",
        chat_type="channel",
        user_id="owner",
    )

    blocked = MessageEvent(
        text="/personality helpful",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert await runner._handle_message(blocked) == run._PRODUCTION_CONFIG_CHANGE_BLOCKED
    assert config_path.read_bytes() == raw
    assert config_module.attest_pinned_effective_config_projection() == digest

    normal = MessageEvent(
        text="Continue the approved plan.",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert await runner._handle_message(normal) == "model-dispatched"
    runner._handle_message_with_agent.assert_awaited_once()
    assert config_module.attest_pinned_effective_config_projection() == digest
