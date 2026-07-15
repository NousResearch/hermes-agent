from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from gateway import production_model_sovereignty_runtime as runtime
from gateway import production_capability_prerequisites as prerequisites


REVISION = "a1234567890bcdef1234567890abcdef12345678"


def _topology() -> dict:
    digest = "1" * 64
    release = prerequisites.production_release_root(REVISION)
    return {
        "schema": prerequisites.TOPOLOGY_SCHEMA,
        "prerequisite_receipt_path": str(prerequisites.PREREQUISITE_PATH),
        "collector_contract_sha256": (
            prerequisites.packaged_prerequisite_contract_sha256()
        ),
        "isolated_worker": {
            "socket_unit": "muncho-isolated-worker.socket",
            "socket_fragment_sha256": "2" * 64,
            "service_unit": "muncho-isolated-worker.service",
            "service_fragment_sha256": "3" * 64,
            "config_path": "/etc/muncho/isolated-worker.json",
            "config_sha256": "4" * 64,
            "socket_path": "/run/muncho-isolated-worker/worker.sock",
            "socket_uid": 0,
            "socket_gid": 993,
            "server_uid": 996,
            "server_gid": 992,
            "gateway_uid": 995,
            "gateway_gid": 991,
            "bwrap_path": "/usr/bin/bwrap",
            "bwrap_sha256": "5" * 64,
            "shell_path": "/bin/bash",
            "shell_sha256": "6" * 64,
        },
        "browser": {
            "unit": prerequisites.BROWSER_UNIT,
            "fragment_sha256": "7" * 64,
            "config_path": str(prerequisites.BROWSER_CONFIG_PATH),
            "config_sha256": "8" * 64,
            "socket_path": str(prerequisites.BROWSER_SOCKET_PATH),
            "service_uid": 997,
            "service_gid": 994,
            "node_path": str(
                release
                / "ops/muncho/runtime/dependencies/node-linux-x64/bin/node"
            ),
            "node_sha256": "9" * 64,
            "wrapper_path": str(
                release / "node_modules/agent-browser/bin/agent-browser.js"
            ),
            "wrapper_sha256": "a" * 64,
            "native_path": str(
                release
                / "node_modules/agent-browser/bin/agent-browser-linux-x64"
            ),
            "native_sha256": "b" * 64,
            "executable": str(
                release
                / "ops/muncho/runtime/dependencies/chrome-linux64/chrome"
            ),
            "executable_sha256": "c" * 64,
            "agent_browser_config_path": str(
                release
                / "ops/muncho/runtime/dependencies/agent-browser.json"
            ),
            "agent_browser_config_sha256": "d" * 64,
        },
        "mac_ops": {
            "unit": prerequisites.MAC_OPS_UNIT,
            "fragment_sha256": "7" * 64,
            "config_sha256": "b" * 64,
            "config_path": str(prerequisites.MAC_OPS_CONFIG_PATH),
            "socket_path": str(prerequisites.MAC_OPS_SOCKET_PATH),
            "credential_path": str(prerequisites.MAC_OPS_CREDENTIAL_PATH),
            "journal_path": str(prerequisites.MAC_OPS_JOURNAL_PATH),
        },
        "routeback_edge": {
            "unit": prerequisites.ROUTEBACK_EDGE_UNIT,
            "fragment_sha256": "8" * 64,
            "config_sha256": "c" * 64,
            "config_path": str(prerequisites.ROUTEBACK_EDGE_CONFIG_PATH),
            "socket_path": str(prerequisites.ROUTEBACK_EDGE_SOCKET_PATH),
            "credential_path": str(prerequisites.ROUTEBACK_EDGE_CREDENTIAL_PATH),
            "readiness_path": str(prerequisites.ROUTEBACK_EDGE_READINESS_PATH),
        },
        "public_connector": {
            "unit": prerequisites.PUBLIC_CONNECTOR_UNIT,
            "fragment_sha256": "9" * 64,
            "config_path": str(prerequisites.PUBLIC_CONNECTOR_CONFIG_PATH),
            "socket_path": str(prerequisites.PUBLIC_CONNECTOR_SOCKET_PATH),
            "credential_path": str(
                prerequisites.PUBLIC_CONNECTOR_CREDENTIAL_PATH
            ),
            "readiness_path": str(
                prerequisites.PUBLIC_CONNECTOR_READINESS_PATH
            ),
        },
        "phase_b": {
            "unit": prerequisites.PHASE_B_UNIT,
            "fragment_sha256": "a" * 64,
            "readiness_path": str(prerequisites.PHASE_B_RECEIPT_PATH),
        },
        "codex_auth_file": str(prerequisites.CODEX_AUTH_PATH),
        "api_control_credential_file": str(
            prerequisites.API_SERVER_CREDENTIAL_PATH
        ),
        "api_approval_credential_file": str(
            prerequisites.API_APPROVAL_CREDENTIAL_PATH
        ),
        "gateway_identity": {"uid": 995, "gid": 991},
    }


def _source_mapping() -> dict:
    return {
        "model": {
            "default": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
        "agent": {
            "max_turns": 90,
            "reasoning_effort": "high",
            "tool_use_enforcement": "auto",
            "verify_on_stop": None,
            "background_review_enabled": None,
            "task_completion_guidance": True,
            "parallel_tool_call_guidance": True,
            "environment_hint": (
                "Use OpenAI Codex OAuth/openai-codex gpt-5.6-sol; do not "
                "route GPT-5.5 through OPENAI_API_KEY. Keep working."
            ),
        },
        "compression": {
            "enabled": True,
            "threshold": 0.5,
            "target_ratio": 0.2,
            "abort_on_summary_failure": False,
        },
        "auxiliary": {
            "compression": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
                "timeout": 120,
            },
            "vision": {"provider": "auto", "model": ""},
        },
        "context": {"engine": "compressor"},
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "provider": "",
        },
        "curator": {
            "enabled": False,
            "consolidate": False,
            "prune_builtins": False,
            "interval_hours": 168,
        },
        "kanban": {
            "dispatch_in_gateway": True,
            "auto_decompose": False,
            "failure_limit": 2,
        },
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": False,
            "warn_after": {"same_tool_failure": 3},
        },
        "tools": {
            "tool_search": {
                "enabled": "on",
                "threshold_pct": 0,
                "search_default_limit": 20,
                "max_search_limit": 50,
            },
        },
        "delegation": {
            "provider": "openrouter",
            "model": "other-model",
            "base_url": "",
            "api_key": "",
            "max_iterations": 90,
        },
        "canonical_brain": {"retention_days": 365},
        "plugins": {},
        "hooks": {},
        "hooks_auto_accept": False,
        "mcp_servers": {"legacy": {"url": "https://invalid.example/mcp"}},
        "fallback_model": {"provider": "openrouter", "model": "fallback"},
        "fallback_providers": ["openrouter"],
        "provider_routing": {"order": ["openrouter"]},
        "command_allowlist": ["*"],
        "thread_sessions_per_user": False,
        "cron": {
            "enabled": True,
            "provider": "builtin",
            "output_retention": 50,
        },
        "approvals": {"mode": "off", "timeout": 60, "cron_mode": "approve"},
        "production_capabilities": _topology(),
        "mac_ops_edge": {
            "enabled": True,
            "socket_path": str(prerequisites.MAC_OPS_SOCKET_PATH),
            "service_unit": prerequisites.MAC_OPS_UNIT,
            "service_uid": 991,
            "socket_gid": 992,
            "service_identity_sha256": "6" * 64,
            "connect_timeout_seconds": 2.0,
            "request_timeout_seconds": 30.0,
        },
        "platforms": {"discord": {"enabled": True}},
        "gateway": {
            "isolated_runtime": False,
            "multiplex_profiles": False,
            "platforms": {"discord": {"enabled": True}},
        },
        "platform_toolsets": {
            "discord": ["terminal", "web", "canonical_brain", "todo"]
        },
        "terminal": {"backend": "local", "cwd": "/srv/muncho"},
        "browser": {"provider": "local"},
        "company": {
            "skyvision": {"workspace": "/srv/skyvision", "read_only": False},
            "adventico": {"workspace": "/srv/adventico", "read_only": False},
        },
    }


def _source_bytes() -> bytes:
    return yaml.safe_dump(
        _source_mapping(), sort_keys=False, allow_unicode=True
    ).encode()


def _contract():
    source = _source_bytes()
    return runtime.produce_production_gateway_contract(
        source,
        expected_source_sha256=hashlib.sha256(source).hexdigest(),
        revision=REVISION,
        gateway_user="hermes-gateway",
        gateway_group="hermes-gateway",
    )


def test_gateway_service_attestation_binds_config_identity_and_exact_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    observed_calls: dict[str, object] = {}
    service = {
        "unit": runtime.GATEWAY_UNIT,
        "fragment_sha256": "1" * 64,
        "unit_service_contract_sha256": "2" * 64,
        "main_pid": 4321,
        "main_pid_executable": "/release/python",
        "main_pid_uid": 995,
        "main_pid_gid": 991,
        "main_pid_groups": [991, 992],
        "main_pid_cmdline_sha256": "3" * 64,
        "main_pid_cgroup": f"/system.slice/{runtime.GATEWAY_UNIT}",
        "main_pid_mount_namespace_inode": 1001,
        "main_pid_network_namespace_inode": 1002,
        "process_identity_matches_unit": True,
        "effective_uid": 995,
        "effective_gid": 991,
    }
    monkeypatch.setattr(
        runtime,
        "_read_stable_production_config",
        lambda: contract.config_bytes,
    )
    monkeypatch.setattr(
        runtime.pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_name=f"gateway-{uid}"),
    )
    monkeypatch.setattr(
        runtime.grp,
        "getgrgid",
        lambda gid: SimpleNamespace(gr_name=f"gateway-{gid}"),
    )
    monkeypatch.setattr(
        runtime,
        "render_production_gateway_unit",
        lambda **kwargs: observed_calls.update(render=kwargs) or b"exact-unit\n",
    )
    monkeypatch.setattr(
        runtime,
        "attest_live_production_gateway_service_identity",
        lambda **kwargs: observed_calls.update(attest=kwargs) or service,
    )

    receipt = runtime.attest_current_production_gateway_service_identity(
        revision=REVISION,
        config_sha256=hashlib.sha256(contract.config_bytes).hexdigest(),
    )

    assert observed_calls["render"] == {
        "revision": REVISION,
        "config_sha256": hashlib.sha256(contract.config_bytes).hexdigest(),
        "gateway_user": "gateway-995",
        "gateway_group": "gateway-991",
        "topology": _topology(),
    }
    assert observed_calls["attest"] == {"expected_unit": b"exact-unit\n"}
    assert receipt["main_pid"] == 4321
    assert receipt["process_identity_matches_unit"] is True
    assert receipt["ready_not_yet_published"] is True


def _provider_registry():
    profile = SimpleNamespace(
        name="openai-codex",
        aliases=("codex", "openai_codex"),
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
        auth_type="oauth_external",
        env_vars=(),
    )
    return SimpleNamespace(
        _REGISTRY={"openai-codex": profile},
        _ALIASES={"codex": "openai-codex", "openai_codex": "openai-codex"},
        _discovered=True,
        _discovery_error=None,
        _isolated_provider_allowlist=frozenset({"openai-codex"}),
        _isolated_discovery_validated=True,
    )


def _plugin_manager():
    manifest = SimpleNamespace(
        key=runtime.PRODUCTION_WEB_PLUGIN_KEY,
        name="web-ddgs",
        source="bundled",
        kind="backend",
        path="/release/plugins/web/ddgs",
    )
    loaded = SimpleNamespace(
        manifest=manifest,
        enabled=True,
        error=None,
        deferred=False,
        module=object(),
        tools_registered=[],
        hooks_registered=[],
        middleware_registered=[],
        commands_registered=[],
    )
    return SimpleNamespace(
        _discovered=True,
        _isolated_allowlist=runtime.PRODUCTION_PLUGIN_ALLOWLIST,
        _isolated_discovery_failure=None,
        _plugins={runtime.PRODUCTION_WEB_PLUGIN_KEY: loaded},
        _hooks={},
        _middleware={},
        _plugin_tool_names=set(),
        _plugin_platform_names=set(),
        _cli_commands={},
        _plugin_commands={},
        _plugin_skills={},
        _aux_tasks={},
        _slack_action_handlers=[],
        _context_engine=None,
        _cli_ref=None,
    )


def _environment() -> dict[str, str]:
    release = str(runtime.production_release_root(REVISION))
    topology = _topology()
    worker = topology["isolated_worker"]
    return {
        "HOME": str(runtime.PRODUCTION_HOME),
        "HERMES_CONFIG": str(runtime.PRODUCTION_CONFIG_PATH),
        "HERMES_HOME": str(runtime.PRODUCTION_HOME),
        "HERMES_MAX_ITERATIONS": "90",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": release,
        "GATEWAY_RELAY_URL": runtime.RELAY_URL,
        "GATEWAY_RELAY_PLATFORMS": "discord",
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": worker["socket_path"],
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(worker["server_uid"]),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(worker["server_gid"]),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": str(worker["socket_uid"]),
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(worker["socket_gid"]),
        "CREDENTIALS_DIRECTORY": f"/run/credentials/{runtime.GATEWAY_UNIT}",
        "_HERMES_GATEWAY": "1",
        "NOTIFY_SOCKET": "/run/systemd/notify",
    }


def _import_origins() -> dict[str, str]:
    release = runtime.production_release_root(REVISION)
    return {
        module: str(release / relative)
        for module, relative in runtime.REQUIRED_IMPORT_ORIGINS.items()
    }


def test_producer_preserves_unrelated_config_and_seals_target() -> None:
    source = _source_mapping()
    contract = _contract()
    effective = runtime.load_strict_production_config(contract.config_bytes)

    runtime.validate_production_gateway_config(effective)
    assert effective["company"] == source["company"]
    worker = _topology()["isolated_worker"]
    assert effective["terminal"] == {
        "backend": "isolated_worker",
        "cwd": "/workspace",
        "timeout": 180,
        "home_mode": "profile",
        "lifetime_seconds": 900,
        "isolated_worker_socket": worker["socket_path"],
        "isolated_worker_server_uid": worker["server_uid"],
        "isolated_worker_server_gid": worker["server_gid"],
        "isolated_worker_socket_uid": worker["socket_uid"],
        "isolated_worker_socket_gid": worker["socket_gid"],
    }
    assert effective["browser"] == {
        "controller": {
            "schema": "hermes-browser-controller-client.v1",
            "socket_path": prerequisites.BROWSER_SOCKET_PATH.as_posix(),
            "server_uid": _topology()["browser"]["service_uid"],
            "artifact_root": prerequisites.BROWSER_ARTIFACT_PATH.as_posix(),
            "connect_timeout_seconds": 5,
            "request_timeout_seconds": 120,
        }
    }
    assert effective["platforms"]["api_server"]["extra"] == {
        "host": prerequisites.API_SERVER_HOST,
        "port": prerequisites.API_SERVER_PORT,
        "key_verifier_credential": prerequisites.API_SERVER_CREDENTIAL_NAME,
        "approval_verifier_credential": (
            prerequisites.API_APPROVAL_CREDENTIAL_NAME
        ),
    }
    assert effective["auxiliary"]["vision"]["provider"] == "openai-codex"
    assert effective["auxiliary"]["vision"]["fallback_chain"] == []
    assert effective["curator"]["interval_hours"] == 168
    assert effective["kanban"]["failure_limit"] == 2
    assert effective["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "max",
    }
    assert effective["agent"]["background_review_enabled"] is False
    assert effective["agent"]["verification_ledger_enabled"] is False
    assert effective["compression"]["abort_on_summary_failure"] is True
    assert effective["tools"]["tool_search"] == {"enabled": "off"}
    assert effective["auxiliary"]["compression"]["provider"] == "openai-codex"
    assert effective["auxiliary"]["compression"]["model"] == "gpt-5.6-sol"
    assert effective["context"] == {"engine": "compressor"}
    assert effective["memory"]["provider"] == ""
    assert effective["memory"]["memory_enabled"] is True
    assert effective["memory"]["user_profile_enabled"] is True
    assert {
        key: effective["delegation"][key]
        for key in runtime._PRODUCTION_DELEGATION_CONFIG
    } == runtime._PRODUCTION_DELEGATION_CONFIG
    assert "memory" in effective["platform_toolsets"]["api_server"]
    assert "memory" in effective["platform_toolsets"]["relay"]
    assert "memory" in effective["platform_toolsets"]["discord"]
    assert "memory" in effective["platform_toolsets"]["cron"]
    assert effective["plugins"] == {"enabled": [], "disabled": []}
    assert effective["web"] == {
        "backend": "",
        "search_backend": "ddgs",
        "extract_backend": "",
    }
    assert effective["hooks"] == {}
    assert effective["mcp_servers"] == {}
    assert effective["thread_sessions_per_user"] is True
    assert effective["platforms"] == runtime._PLATFORMS
    assert effective["platform_toolsets"] == {
        "api_server": list(prerequisites.FIRST_WAVE_TOOLSETS),
        "relay": list(prerequisites.FIRST_WAVE_TOOLSETS),
        "discord": list(prerequisites.FIRST_WAVE_TOOLSETS),
        "cron": list(prerequisites.FIRST_WAVE_TOOLSETS),
    }
    from gateway.production_access_policy import production_access_config

    assert effective["production_access"] == production_access_config()
    assert effective["command_allowlist"] == []
    assert effective["approvals"]["mode"] == "manual"
    assert effective["approvals"]["cron_mode"] == "deny"
    assert effective["approvals"]["plan_owner_user_ids"] == [
        runtime.PRODUCTION_OWNER_DISCORD_USER_ID
    ]
    assert effective["approvals"]["gateway_authorized_user_ids"] == [
        runtime.PRODUCTION_OWNER_DISCORD_USER_ID
    ]
    assert effective["approvals"]["gateway_authorized_user_names"] == []
    assert effective["approvals"]["gateway_authorized_labels"] == ["Емо"]
    assert effective["approvals"]["gateway_owner_escalation"] == {
        "enabled": True,
        "owner_user_id": runtime.PRODUCTION_OWNER_DISCORD_USER_ID,
        "owner_guild_id": runtime.SKYVISION_GUILD_ID,
        "owner_channel_id": runtime.SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "owner_target_type": "guild_channel",
    }
    assert effective["goals"] == {"max_turns": 0}
    assert "discord" not in effective["platforms"]


def test_production_thread_transcripts_are_isolated_per_exact_author() -> None:
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    effective = runtime.load_strict_production_config(_contract().config_bytes)
    common = {
        "platform": Platform.RELAY,
        "chat_id": "public-channel",
        "chat_type": "channel",
        "thread_id": "public-thread",
    }
    owner = SessionSource(**common, user_id="owner-id")
    teammate = SessionSource(**common, user_id="team-id")

    owner_key = build_session_key(
        owner,
        thread_sessions_per_user=effective["thread_sessions_per_user"],
    )
    teammate_key = build_session_key(
        teammate,
        thread_sessions_per_user=effective["thread_sessions_per_user"],
    )

    assert owner_key.endswith(":owner-id")
    assert teammate_key.endswith(":team-id")
    assert owner_key != teammate_key


def test_production_approval_prompts_are_owner_id_only() -> None:
    from gateway.approval_authority import gateway_approval_authority_decision

    effective = runtime.load_strict_production_config(_contract().config_bytes)
    owner = SimpleNamespace(
        user_id=runtime.PRODUCTION_OWNER_DISCORD_USER_ID,
        user_id_alt="",
        user_name="arbitrary-display-name",
        chat_name="arbitrary-chat-name",
    )
    teammate = SimpleNamespace(
        user_id="1282940574533423125",
        user_id_alt="",
        user_name="Емо",
        chat_name="Емо",
    )

    owner_decision = gateway_approval_authority_decision(effective, owner)
    teammate_decision = gateway_approval_authority_decision(effective, teammate)

    assert owner_decision.restricted is True
    assert owner_decision.allowed is True
    assert teammate_decision.restricted is True
    assert teammate_decision.allowed is False
    assert teammate_decision.authorized_labels == ("Емо",)


@pytest.mark.parametrize("value", [False, None])
def test_validator_rejects_shared_or_missing_thread_transcript_lanes(value) -> None:
    effective = runtime.load_strict_production_config(_contract().config_bytes)
    if value is None:
        effective.pop("thread_sessions_per_user", None)
    else:
        effective["thread_sessions_per_user"] = value

    with pytest.raises(
        runtime.ProductionContractError,
        match="production_thread_session_isolation_not_exact",
    ):
        runtime.validate_production_gateway_config(effective)


@pytest.mark.parametrize("value", [20, False, None])
def test_validator_rejects_arbitrary_production_goal_pause(value) -> None:
    effective = runtime.load_strict_production_config(_contract().config_bytes)
    if value is None:
        effective.pop("goals", None)
    else:
        effective["goals"] = {"max_turns": value}

    with pytest.raises(
        runtime.ProductionContractError,
        match="production_goal_continuation_not_exact",
    ):
        runtime.validate_production_gateway_config(effective)


def test_overlay_repairs_disabled_builtin_memory_without_external_provider() -> None:
    source = _source_mapping()
    source["memory"]["memory_enabled"] = False
    source["memory"]["user_profile_enabled"] = False

    effective = runtime.overlay_production_gateway_config(source)

    assert effective["memory"] == {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "provider": "",
    }


def test_producer_binds_source_target_release_and_public_manifest() -> None:
    contract = _contract()
    manifest = contract.public_manifest()

    assert contract.release_root == (
        runtime.PRODUCTION_RELEASES / f"hermes-agent-{REVISION[:12]}"
    )
    assert contract.config_sha256 == hashlib.sha256(contract.config_bytes).hexdigest()
    assert contract.unit_sha256 == hashlib.sha256(contract.unit_bytes).hexdigest()
    assert manifest["release_revision"] == REVISION
    assert manifest["target_config_sha256"] == contract.config_sha256
    assert manifest["target_unit_sha256"] == contract.unit_sha256
    assert manifest["direct_discord_enabled"] is False
    assert manifest["discord_dm_allowed"] is False
    assert manifest["isolated_worker_required"] is True
    assert manifest["browser_controller_required"] is True
    assert manifest["host_workspace_projection_enabled"] is False
    assert manifest["mcp_auto_discovery"] is False
    assert manifest["api_approval_owner_credential"] == (
        prerequisites.API_APPROVAL_CREDENTIAL_NAME
    )
    assert manifest["api_positive_approval_requires_owner_authority"] is True
    assert manifest["plugin_allowlist"] == sorted(
        runtime.PRODUCTION_PLUGIN_ALLOWLIST
    )
    assert manifest["secret_material_recorded"] is False
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "contract_sha256"
    }
    assert manifest["contract_sha256"] == hashlib.sha256(
        runtime._canonical_bytes(unsigned)
    ).hexdigest()
    encoded = repr(manifest)
    assert "environment_hint" not in encoded
    assert "company" not in encoded
    assert "api_key" not in encoded
    assert "docker" not in encoded.casefold()
    assert "cdp" not in encoded.casefold()


def test_browser_controller_client_binding_is_exact_and_topology_derived() -> None:
    topology = _topology()
    config = runtime.production_browser_controller_client_config(topology)

    assert config.socket_path == prerequisites.BROWSER_SOCKET_PATH
    assert config.server_uid == topology["browser"]["service_uid"]
    assert config.artifact_root == prerequisites.BROWSER_ARTIFACT_PATH
    assert config.connect_timeout_seconds == 5
    assert config.request_timeout_seconds == 120

    drifted = copy.deepcopy(topology)
    drifted["browser"]["socket_path"] = "/tmp/browser.sock"
    with pytest.raises(
        runtime.ProductionContractError,
        match="browser_controller_client_config_invalid",
    ):
        runtime.production_browser_controller_client_config(drifted)


def test_rendered_unit_is_the_normal_sha_pinned_production_contract() -> None:
    contract = _contract()
    unit = contract.unit_bytes.decode()
    release = contract.release_root

    assert "Type=notify" in unit
    assert "Restart=on-failure" in unit
    assert f"BindsTo={prerequisites.ROUTEBACK_EDGE_UNIT}" in unit
    assert runtime.WRITER_UNIT in unit
    assert runtime.CONNECTOR_UNIT in unit
    assert prerequisites.MAC_OPS_UNIT in unit
    assert prerequisites.BROWSER_UNIT in unit
    topology = _topology()
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    requires = next(
        line.removeprefix("Requires=")
        for line in unit.splitlines()
        if line.startswith("Requires=")
    ).split()
    binds_to = next(
        line.removeprefix("BindsTo=")
        for line in unit.splitlines()
        if line.startswith("BindsTo=")
    ).split()
    assert set(requires) == {
        prerequisites.PHASE_B_UNIT,
        prerequisites.ROUTEBACK_EDGE_UNIT,
        runtime.CONNECTOR_UNIT,
        prerequisites.MAC_OPS_UNIT,
        worker["socket_unit"],
        worker["service_unit"],
        browser["unit"],
        runtime.WRITER_UNIT,
    }
    assert set(binds_to) == set(requires) - {prerequisites.PHASE_B_UNIT}
    assert worker["socket_unit"] in unit
    assert worker["service_unit"] in unit
    assert f"AssertPathExists={worker['config_path']}" in unit
    assert f"AssertPathExists={worker['socket_path']}" in unit
    assert f"ReadOnlyPaths={worker['config_path']}" in unit
    assert f"ReadOnlyPaths={prerequisites.BROWSER_CONFIG_PATH}" in unit
    assert f"ReadOnlyPaths={prerequisites.BROWSER_SOCKET_PATH.parent}" in unit
    for field in (
        "node_path",
        "wrapper_path",
        "native_path",
        "executable",
        "agent_browser_config_path",
    ):
        assert f"InaccessiblePaths={browser[field]}" in unit
        assert f"AssertPathExists={browser[field]}" not in unit
    assert "/usr/bin/chromium" not in unit
    assert prerequisites.PHASE_B_UNIT in unit
    assert (
        f"LoadCredential={prerequisites.API_SERVER_CREDENTIAL_NAME}:"
        f"{prerequisites.API_SERVER_CREDENTIAL_PATH}" in unit
    )
    assert (
        f"LoadCredential={prerequisites.API_APPROVAL_CREDENTIAL_NAME}:"
        f"{prerequisites.API_APPROVAL_CREDENTIAL_PATH}" in unit
    )
    assert "ExecStartPre=+" in unit
    assert "gateway.production_capability_prerequisites collect" in unit
    assert (
        f"--lifecycle-phase {prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED}"
    ) in unit
    assert f"ReadWritePaths={prerequisites.PREREQUISITE_PATH.parent}" in unit
    assert f"AssertPathExists={prerequisites.PREREQUISITE_PATH}" not in unit
    assert "SupplementaryGroups=muncho-writer-client muncho-discord-egress" in unit
    assert "Environment=TERMINAL_ENV=isolated_worker" in unit
    assert (
        "Environment=TERMINAL_ISOLATED_WORKER_SOCKET="
        f"{worker['socket_path']}"
    ) in unit
    assert (
        "Environment=TERMINAL_ISOLATED_WORKER_SERVER_UID="
        f"{worker['server_uid']}"
    ) in unit
    assert (
        "Environment=TERMINAL_ISOLATED_WORKER_SERVER_GID="
        f"{worker['server_gid']}"
    ) in unit
    assert (
        "Environment=TERMINAL_ISOLATED_WORKER_SOCKET_UID="
        f"{worker['socket_uid']}"
    ) in unit
    assert (
        "Environment=TERMINAL_ISOLATED_WORKER_SOCKET_GID="
        f"{worker['socket_gid']}"
    ) in unit
    assert "TERMINAL_DOCKER" not in unit
    assert "docker.sock" not in unit
    assert "remote-debugging" not in unit
    assert "9222" not in unit
    assert "muncho-worker-clients" in unit
    assert "muncho-capability-browser" in unit
    supplementary = next(
        line.removeprefix("SupplementaryGroups=")
        for line in unit.splitlines()
        if line.startswith("SupplementaryGroups=")
    ).split()
    assert supplementary == [
        "muncho-writer-client",
        "muncho-discord-egress",
        "muncho-discord-connector",
        "muncho-mac-ops-edge",
        "muncho-worker-clients",
        "muncho-capability-browser",
    ]
    assert f"ReadWritePaths={prerequisites.CODEX_AUTH_PATH}" in unit
    assert f"ReadOnlyPaths={prerequisites.CODEX_AUTH_PATH}" not in unit
    assert f"Environment=PYTHONPATH={release}" in unit
    assert (
        f"ExecStart={release}/.venv/bin/python -B -P -s -m gateway.run "
        in unit
    )
    assert f"--config {runtime.PRODUCTION_CONFIG_PATH}" in unit
    assert runtime.STARTUP_FLAG in unit
    assert f"--production-release-revision {REVISION}" in unit
    assert f"--production-config-sha256 {contract.config_sha256}" in unit
    assert "--require-capability-canary" not in unit
    assert "--require-canonical-writer" not in unit
    assert "EnvironmentFile=" not in unit
    assert "PassEnvironment=" not in unit
    assert "DISCORD_BOT_TOKEN=" not in unit
    assert "HERMES_YOLO_MODE" in unit
    assert "InaccessiblePaths=-/etc/muncho/discord-connector-credentials" in unit
    assert (
        f"InaccessiblePaths=-{runtime.LEGACY_API_BEARER_SOURCE_PATH}" in unit
    )
    assert (
        f"InaccessiblePaths=-{runtime.OWNER_APPROVAL_PASSKEY_SOURCE_PATH}"
        in unit
    )
    assert str(runtime.LEGACY_API_BEARER_SOURCE_PATH) not in "\n".join(
        line for line in unit.splitlines() if line.startswith("LoadCredential=")
    )
    assert str(runtime.OWNER_APPROVAL_PASSKEY_SOURCE_PATH) not in "\n".join(
        line for line in unit.splitlines() if line.startswith("LoadCredential=")
    )


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("agent", "background_review_enabled"), True, "agent_policy"),
        (("agent", "adaptive_reasoning"), {"enabled": False}, "agent_policy"),
        (("compression", "abort_on_summary_failure"), False, "compression"),
        (("context", "engine"), "lcm", "context_engine"),
        (("memory", "provider"), "honcho", "memory_provider"),
        (("memory", "memory_enabled"), False, "memory_provider"),
        (("memory", "user_profile_enabled"), False, "memory_provider"),
        (("delegation", "provider"), "openrouter", "delegation_route"),
        (("delegation", "model"), "other-model", "delegation_route"),
        (("delegation", "base_url"), "https://example.invalid", "delegation_route"),
        (("delegation", "api_key"), "not-a-real-key", "delegation_route"),
        (("delegation", "api_mode"), "chat_completions", "delegation_route"),
        (("delegation", "reasoning_effort"), "low", "delegation_route"),
        (("delegation", "max_iterations"), 89, "delegation_route"),
        (("delegation", "child_timeout_seconds"), 30, "delegation_route"),
        (("delegation", "max_concurrent_children"), 5, "delegation_route"),
        (("delegation", "orchestrator_enabled"), False, "delegation_route"),
        (("delegation", "max_spawn_depth"), 3, "delegation_route"),
        (("delegation", "subagent_auto_approve"), True, "delegation_route"),
        (("kanban", "dispatch_in_gateway"), True, "kanban"),
        (("curator", "enabled"), True, "curator"),
        (("agent", "verification_ledger_enabled"), True, "agent_policy"),
        (
            ("approvals", "plan_owner_user_ids"),
            ["1282940574533423125"],
            "approvals",
        ),
        (
            ("approvals", "gateway_authorized_user_names"),
            ["Emil"],
            "approvals",
        ),
        (
            ("approvals", "gateway_owner_escalation"),
            {"enabled": False},
            "approvals",
        ),
        (("command_allowlist",), ["*"], "command_allowlist"),
        (("plugins", "enabled"), ["observability"], "plugin_config"),
        (("web", "search_backend"), "firecrawl", "web_boundary"),
        (("mcp_servers", "external"), {"url": "https://x"}, "mcp_config"),
        (
            ("quick_commands", "unsafe"),
            {"type": "exec", "command": "id"},
            "quick_commands",
        ),
        (("skills", "inline_shell"), True, "skills_inline_shell"),
        (("security", "allow_lazy_installs"), True, "lazy_installs"),
        (("terminal", "backend"), "docker", "terminal_boundary"),
        (
            ("browser", "controller"),
            {"cdp_url": "http://127.0.0.1:9222"},
            "browser_boundary",
        ),
        (
            ("production_access", "owner_discord_user_id"),
            "1279454038731264062",
            "production_access_policy",
        ),
        (("platform_toolsets", "cron"), ["code_execution"], "toolsets"),
        (("platform_toolsets", "discord"), ["code_execution"], "toolsets"),
        (("platforms", "discord"), {"enabled": True}, "platform_boundary"),
        (
            ("tools", "tool_search"),
            {"enabled": "on", "threshold_pct": 0},
            "tool_search",
        ),
    ],
)
def test_config_validator_fails_closed_on_runtime_drift(path, value, code) -> None:
    effective = runtime.load_strict_production_config(_contract().config_bytes)
    parent = effective
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value

    with pytest.raises(runtime.ProductionContractError, match=code):
        runtime.validate_production_gateway_config(effective)


def test_source_digest_and_strict_yaml_are_fail_closed() -> None:
    source = _source_bytes()
    with pytest.raises(runtime.ProductionContractError, match="sha256_mismatch"):
        runtime.render_production_gateway_config(
            source, expected_source_sha256="0" * 64
        )
    with pytest.raises(runtime.ProductionContractError, match="alias_forbidden"):
        runtime.load_strict_production_config(b"a: &x {}\nb: *x\n")
    with pytest.raises(runtime.ProductionContractError, match="key_invalid"):
        runtime.load_strict_production_config(b"a: 1\na: 2\n")


def test_extension_surface_has_only_exact_runnable_mechanical_web_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.web_search_registry import _reset_for_tests, register_provider
    from plugins.web.ddgs.provider import DDGSWebSearchProvider

    hooks = SimpleNamespace(_handlers={}, _loaded_hooks=[])
    monkeypatch.setattr(
        runtime.importlib_metadata,
        "version",
        lambda name: (
            runtime.PRODUCTION_WEB_DISTRIBUTION_VERSION
            if name == runtime.PRODUCTION_WEB_DISTRIBUTION
            else "unexpected"
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "ddgs", SimpleNamespace())
    _reset_for_tests()
    register_provider(DDGSWebSearchProvider())
    try:
        runtime.validate_production_extension_surface(
            _plugin_manager(), hooks, _provider_registry()
        )

        manager = _plugin_manager()
        manager._middleware["llm_request"] = [lambda **_kwargs: None]
        with pytest.raises(runtime.ProductionContractError, match="surface_not_empty"):
            runtime.validate_production_extension_surface(
                manager, hooks, _provider_registry()
            )

        provider = _provider_registry()
        provider._REGISTRY["openrouter"] = SimpleNamespace(name="openrouter")
        with pytest.raises(runtime.ProductionContractError, match="registry_not_exact"):
            runtime.validate_production_extension_surface(
                _plugin_manager(), hooks, provider
            )

        from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider

        register_provider(FirecrawlWebSearchProvider())
        with pytest.raises(
            runtime.ProductionContractError, match="web_executor_not_exact"
        ):
            runtime.validate_production_extension_surface(
                _plugin_manager(), hooks, _provider_registry()
            )
    finally:
        _reset_for_tests()


def test_adapter_surface_is_exact_live_loopback_api_and_privileged_unix_relay() -> None:
    from gateway.config import Platform
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.api_verifier_credentials import (
        build_api_approval_scrypt_verifier,
        build_api_bearer_verifier,
        parse_api_approval_scrypt_verifier,
        parse_api_bearer_verifier,
    )
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.discord_connector_transport import (
        DiscordConnectorRelayTransport,
    )

    transport = object.__new__(DiscordConnectorRelayTransport)
    transport._connected = True
    transport._poller = SimpleNamespace(done=lambda: False)
    transport.socket_path = str(runtime.CONNECTOR_SOCKET)
    transport.server_authorizer = SimpleNamespace(
        server_unit=runtime.CONNECTOR_UNIT
    )
    adapter = object.__new__(RelayAdapter)
    adapter._transport = transport
    adapter.descriptor = CapabilityDescriptor(
        contract_version=1,
        platform="discord",
        label="Discord public connector",
        max_message_length=2_000,
        supports_draft_streaming=False,
        supports_edit=False,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )
    api = object.__new__(APIServerAdapter)
    api._running = True
    api._host = prerequisites.API_SERVER_HOST
    api._port = prerequisites.API_SERVER_PORT
    api._api_key = ""
    api._api_bearer_verifier = parse_api_bearer_verifier(
        build_api_bearer_verifier("x" * 32)
    )
    api._approval_passkey = ""
    api._approval_passkey_verifier = parse_api_approval_scrypt_verifier(
        build_api_approval_scrypt_verifier("y" * 32, salt=b"s" * 32)
    )
    api._model_routes = {}
    api._max_concurrent_runs = 1
    api._app = object()
    api._runner = object()
    api._site = SimpleNamespace(
        _server=SimpleNamespace(is_serving=lambda: True)
    )
    adapters = {Platform.API_SERVER: api, Platform.RELAY: adapter}
    runtime.validate_production_gateway_adapters(adapters)

    api._approval_passkey_verifier = None
    with pytest.raises(runtime.ProductionContractError, match="not_ready"):
        runtime.validate_production_gateway_adapters(adapters)
    api._approval_passkey_verifier = parse_api_approval_scrypt_verifier(
        build_api_approval_scrypt_verifier("y" * 32, salt=b"s" * 32)
    )

    transport._connected = False
    with pytest.raises(runtime.ProductionContractError, match="not_ready"):
        runtime.validate_production_gateway_adapters(adapters)
    with pytest.raises(runtime.ProductionContractError, match="set_not_exact"):
        runtime.validate_production_gateway_adapters({})


def test_environment_and_release_import_identity_defeat_editable_venv() -> None:
    contract = _contract()
    environment = _environment()
    runtime.validate_production_gateway_environment(
        environment,
        revision=REVISION,
        config_sha256=contract.config_sha256,
        topology=_topology(),
    )
    release = str(contract.release_root)
    sys_path = [
        release,
        "/usr/lib/python3.12.zip",
        "/usr/lib/python3.12",
        f"{release}/.venv/lib/python3.12/site-packages",
    ]
    runtime.validate_production_release_import_identity(
        revision=REVISION,
        source_commit_bytes=f"{REVISION}\n".encode(),
        executable=f"{release}/.venv/bin/python",
        sys_path=sys_path,
        import_origins=_import_origins(),
    )

    with pytest.raises(runtime.ProductionContractError, match="editable_import"):
        runtime.validate_production_release_import_identity(
            revision=REVISION,
            source_commit_bytes=f"{REVISION}\n".encode(),
            executable=f"{release}/.venv/bin/python",
            sys_path=[release, "/opt/adventico-ai-platform/hermes-agent"],
            import_origins=_import_origins(),
        )
    origins = _import_origins()
    origins["gateway.run"] = "/opt/adventico-ai-platform/hermes-agent/gateway/run.py"
    with pytest.raises(runtime.ProductionContractError, match="not_revision_bound"):
        runtime.validate_production_release_import_identity(
            revision=REVISION,
            source_commit_bytes=f"{REVISION}\n".encode(),
            executable=f"{release}/.venv/bin/python",
            sys_path=sys_path,
        import_origins=origins,
    )


def test_environment_is_bound_to_exact_isolated_worker_topology() -> None:
    contract = _contract()
    environment = _environment()
    topology = _topology()
    drifted = copy.deepcopy(topology)
    drifted["isolated_worker"]["server_uid"] += 10

    with pytest.raises(
        runtime.ProductionContractError,
        match="environment_values_drifted",
    ):
        runtime.validate_production_gateway_environment(
            environment,
            revision=REVISION,
            config_sha256=contract.config_sha256,
            topology=drifted,
        )

    environment["TERMINAL_DOCKER_IMAGE"] = "legacy"
    with pytest.raises(
        runtime.ProductionContractError,
        match="environment_name_not_allowed",
    ):
        runtime.validate_production_gateway_environment(
            environment,
            revision=REVISION,
            config_sha256=contract.config_sha256,
            topology=topology,
        )


def test_environment_rejects_direct_discord_and_provider_credentials() -> None:
    contract = _contract()
    for name in (
        "DISCORD_BOT_TOKEN",
        "OPENAI_API_KEY",
        "HERMES_YOLO_MODE",
        "HERMES_IGNORE_USER_CONFIG",
        "CUSTOM_SECRET",
    ):
        environment = _environment()
        environment[name] = "opaque"
        with pytest.raises(
            runtime.ProductionContractError,
            match="secret_or_override_present",
        ):
            runtime.validate_production_gateway_environment(
                environment,
                revision=REVISION,
                config_sha256=contract.config_sha256,
                topology=_topology(),
            )


@pytest.mark.parametrize(
    "name",
    [
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "BROWSER_CDP_URL",
        "AGENT_BROWSER_CONFIG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "DOCKER_HOST",
        "HERMES_ARBITRARY_OVERRIDE",
    ],
)
def test_environment_rejects_every_unreviewed_name(name: str) -> None:
    contract = _contract()
    environment = _environment()
    environment[name] = "opaque"
    with pytest.raises(
        runtime.ProductionContractError,
        match="environment_name_not_allowed",
    ):
        runtime.validate_production_gateway_environment(
            environment,
            revision=REVISION,
            config_sha256=contract.config_sha256,
            topology=_topology(),
        )


def test_environment_rejects_malformed_systemd_metadata() -> None:
    contract = _contract()
    environment = _environment()
    environment["INVOCATION_ID"] = "not-an-id"
    with pytest.raises(
        runtime.ProductionContractError,
        match="environment_value_not_allowed",
    ):
        runtime.validate_production_gateway_environment(
            environment,
            revision=REVISION,
            config_sha256=contract.config_sha256,
            topology=_topology(),
        )


def test_cron_boundary_allows_only_exact_primary_model_jobs() -> None:
    runtime.validate_production_cron_jobs(
        [
            {
                "id": "model-check",
                "enabled": True,
                "no_agent": False,
                "prompt": "Inspect the exact source and decide what to report.",
                "script": None,
                "deliver": "local",
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "base_url": None,
            },
            {
                "id": "disabled-legacy",
                "enabled": False,
                "no_agent": False,
                "prompt": "",
                "provider": None,
                "model": None,
            },
        ]
    )
    semantic = {
        "id": "review",
        "enabled": True,
        "no_agent": False,
        "prompt": "Review",
        "script": "/opt/muncho/bin/review",
        "provider": "openrouter",
        "model": "other-model",
    }
    with pytest.raises(runtime.ProductionContractError, match="local_script_forbidden"):
        runtime.validate_production_cron_jobs([semantic])
    pinned = {
        "id": "sync",
        "enabled": True,
        "no_agent": True,
        "script": "/opt/muncho/bin/fork-sync",
        "model": "gpt-5.5",
    }
    with pytest.raises(runtime.ProductionContractError, match="local_script_forbidden"):
        runtime.validate_production_cron_jobs([pinned])

    pre_run = {
        "id": "pre-run",
        "enabled": True,
        "no_agent": False,
        "prompt": "Review the collected facts",
        "script": "collect.py",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": None,
    }
    with pytest.raises(runtime.ProductionContractError, match="local_script_forbidden"):
        runtime.validate_production_cron_jobs([pre_run])


@pytest.mark.parametrize("script", ["", 0, False, {}, []])
def test_production_cron_rejects_every_non_null_script_shape(script) -> None:
    with pytest.raises(runtime.ProductionContractError, match="local_script_forbidden"):
        runtime.validate_production_cron_jobs(
            [_exact_production_cron_job(script=script)]
        )


def test_production_cron_rejects_host_workdir() -> None:
    with pytest.raises(runtime.ProductionContractError, match="workdir_forbidden"):
        runtime.validate_production_cron_jobs(
            [_exact_production_cron_job(workdir="/srv/muncho")]
        )


def _exact_production_cron_job(**overrides):
    job = {
        "id": "exact-toolsets",
        "enabled": True,
        "no_agent": False,
        "prompt": "Inspect the exact source and decide what to report.",
        "script": None,
        "deliver": "local",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": None,
        "enabled_toolsets": None,
    }
    job.update(overrides)
    return job


def test_production_cron_resolver_inherits_or_narrows_exact_first_wave() -> None:
    config = {
        "platform_toolsets": {
            "cron": list(prerequisites.FIRST_WAVE_TOOLSETS),
        }
    }

    assert runtime.resolve_production_cron_enabled_toolsets(
        _exact_production_cron_job(), config
    ) == list(prerequisites.FIRST_WAVE_TOOLSETS)
    assert runtime.resolve_production_cron_enabled_toolsets(
        _exact_production_cron_job(enabled_toolsets=["web", "file"]),
        config,
    ) == ["web", "file"]


def test_production_cron_allows_only_local_or_public_discord_origin_delivery() -> None:
    public_origin = {
        "platform": "discord",
        "chat_id": "1504852355588423801",
        "chat_name": None,
        "thread_id": None,
        "user_id": "1279454038731264061",
    }
    runtime.validate_production_cron_jobs(
        [
            _exact_production_cron_job(),
            _exact_production_cron_job(
                deliver="origin",
                origin=public_origin,
            ),
            _exact_production_cron_job(
                deliver="origin",
                origin={
                    **public_origin,
                    "chat_id": "1504852355588423802",
                    "thread_id": "1504852355588423802",
                },
            ),
        ]
    )


@pytest.mark.parametrize(
    ("deliver", "origin"),
    [
        (None, None),
        ([], None),
        ("all", None),
        ("discord:1504852355588423801", None),
        ("origin", None),
        (
            "origin",
            {"platform": "relay", "chat_id": "1504852355588423801"},
        ),
        ("origin", {"platform": "discord", "chat_id": "not-a-snowflake"}),
        (
            "origin",
            {
                "platform": "discord",
                "chat_id": "1504852355588423801",
                "thread_id": "1504852355588423802",
            },
        ),
        (
            "origin",
            {
                "platform": "discord",
                "chat_id": "1504852355588423801",
                "scope_id": "1282725267068157972",
            },
        ),
    ],
)
def test_production_cron_rejects_unreviewed_delivery_or_origin(
    deliver,
    origin,
) -> None:
    with pytest.raises(runtime.ProductionContractError):
        runtime.validate_production_cron_jobs(
            [
                _exact_production_cron_job(
                    deliver=deliver,
                    origin=origin,
                )
            ]
        )


@pytest.mark.parametrize(
    "enabled_toolsets",
    [
        [],
        "web",
        ["web", "web"],
        ["web", 1],
        ["code_execution"],
        ["web", " code_execution"],
    ],
)
def test_production_cron_rejects_malformed_or_widening_job_toolsets(
    enabled_toolsets,
) -> None:
    with pytest.raises(runtime.ProductionContractError):
        runtime.validate_production_cron_jobs(
            [_exact_production_cron_job(enabled_toolsets=enabled_toolsets)]
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"platform_toolsets": {}},
        {"platform_toolsets": {"cron": None}},
        {"platform_toolsets": {"cron": ["web"]}},
    ],
)
def test_production_cron_resolver_rejects_missing_or_drifted_platform_surface(
    config,
) -> None:
    with pytest.raises(
        runtime.ProductionContractError,
        match="production_cron_toolsets_not_exact",
    ):
        runtime.resolve_production_cron_enabled_toolsets(
            _exact_production_cron_job(), config
        )


def test_overlay_does_not_mutate_observed_mapping() -> None:
    source = _source_mapping()
    before = copy.deepcopy(source)
    runtime.overlay_production_gateway_config(source)
    assert source == before


def test_production_tool_search_cannot_replace_model_visible_tools() -> None:
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    effective = runtime.overlay_production_gateway_config(_source_mapping())
    config = ToolSearchConfig.from_raw(effective["tools"]["tool_search"])
    deferrable = [
        {
            "type": "function",
            "function": {
                "name": f"external_business_tool_{index}",
                "description": "x" * 10_000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(25)
    ]

    assembly = assemble_tool_defs(
        deferrable,
        context_length=1,
        config=config,
    )

    assert config.enabled == "off"
    assert assembly.activated is False
    assert assembly.tool_defs == deferrable
    assert {
        item["function"]["name"] for item in assembly.tool_defs
    }.isdisjoint({"tool_search", "tool_describe", "tool_call"})


def test_pre_ready_prerequisite_attestation_is_committed_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_raw = b"exact production config\n"
    topology = _topology()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runtime,
        "_read_stable_production_config",
        lambda: config_raw,
    )
    monkeypatch.setattr(
        runtime,
        "load_strict_production_config",
        lambda _raw: {"production_capabilities": topology},
    )
    monkeypatch.setattr(
        runtime,
        "validate_production_gateway_config",
        lambda _config: None,
    )

    def load_receipt(**kwargs):
        captured.update(kwargs)
        return {
            "schema": prerequisites.PREREQUISITE_SCHEMA,
            "release_revision": REVISION,
            "lifecycle_phase": kwargs["lifecycle_phase"],
            "boot_id_sha256": "b" * 64,
            "observed_at_unix": 1_800_000_000,
            "receipt_sha256": "c" * 64,
        }

    monkeypatch.setattr(
        runtime,
        "load_production_capability_prerequisite_receipt",
        load_receipt,
    )
    result = runtime.attest_current_production_capability_prerequisites(
        revision=REVISION,
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
    )
    assert captured["lifecycle_phase"] == (
        prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED
    )
    assert result["lifecycle_phase"] == (
        prerequisites.PREREQUISITE_LIFECYCLE_COMMITTED
    )


def test_production_model_authored_skill_inline_shell_stays_literal(
    tmp_path,
    monkeypatch,
) -> None:
    effective = runtime.overlay_production_gateway_config(_source_mapping())
    skill = tmp_path / "model-authored" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\nname: model-authored\ndescription: test\n---\n\n"
        "Never execute: !`printf GATEWAY_CHILD_RAN`\n",
        encoding="utf-8",
    )

    def _must_not_spawn(*_args, **_kwargs):
        raise AssertionError("production skill preprocessing spawned a child")

    monkeypatch.setattr("tools.skills_tool.SKILLS_DIR", tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: effective)
    monkeypatch.setattr(
        "agent.skill_preprocessing.subprocess.run",
        _must_not_spawn,
    )
    from tools.skills_tool import skill_view

    result = json.loads(skill_view("model-authored"))
    assert result["success"] is True
    assert "!`printf GATEWAY_CHILD_RAN`" in result["content"]
    assert "Never execute: GATEWAY_CHILD_RAN" not in result["content"]
