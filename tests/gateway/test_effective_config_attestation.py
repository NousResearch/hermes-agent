import asyncio
import hashlib
import json
import os
from pathlib import Path

import gateway.run as gateway_run
import gateway.config as gateway_config
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _resolve_gateway_model, start_gateway
from gateway.status import read_runtime_status, write_runtime_status


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_mapping_keys(nested)


def test_running_gateway_attests_exact_sanitized_profile_config(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_bytes = (
        b"model:\n"
        b"  provider: openrouter\n"
        b"  default: openai/gpt-5.5\n"
        b"  api_key: primary-secret-value\n"
        b"  base_url: https://user:password@example.invalid/v1\n"
        b"fallback_providers:\n"
        b"  - provider: anthropic\n"
        b"    model: claude-sonnet-4\n"
        b"    api_key: fallback-secret-value\n"
        b"    key_env: FALLBACK_API_KEY\n"
        b"    headers:\n"
        b"      Authorization: bearer-secret-value\n"
    )
    (home / "config.yaml").write_bytes(config_bytes)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))
    snapshot_loaded_at = runner._startup_config_loaded_at

    assert asyncio.run(runner.start()) is True

    state = read_runtime_status()
    receipt = state["effective_config"]
    assert state["gateway_state"] == "running"
    assert receipt["schema"] == 1
    assert receipt["complete"] is True
    assert receipt["pid"] == os.getpid() == state["pid"]
    assert receipt["start_time"] == state["start_time"]
    assert receipt["loaded_at"] == snapshot_loaded_at
    assert receipt["profiles"] == {
        "default": {
            "profile_path": str(home.resolve()),
            "config_path": str((home / "config.yaml").resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "primary": {"provider": "openrouter", "model": "openai/gpt-5.5"},
            "fallbacks": [{"provider": "anthropic", "model": "claude-sonnet-4"}],
        }
    }

    forbidden_keys = {
        "api_key",
        "key_env",
        "api_key_env",
        "base_url",
        "headers",
        "token",
        "secret",
        "password",
        "command",
        "args",
        "credential_pool",
    }
    assert forbidden_keys.isdisjoint(_all_mapping_keys(receipt))
    serialized = json.dumps(receipt)
    for secret in (
        "primary-secret-value",
        "fallback-secret-value",
        "bearer-secret-value",
        "user:password",
        "FALLBACK_API_KEY",
    ):
        assert secret not in serialized


def test_single_profile_receipt_uses_explicit_custom_hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "fleet-managed-agent-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: hermes-4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))

    assert asyncio.run(runner.start()) is True

    receipt = read_runtime_status()["effective_config"]
    assert receipt["complete"] is True
    assert receipt["profiles"]["default"]["profile_path"] == str(home.resolve())
    assert receipt["profiles"]["default"]["config_path"] == str(
        (home / "config.yaml").resolve()
    )


def test_config_bom_is_decoded_but_original_bytes_are_hashed(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_bytes = (
        b"\xef\xbb\xbfmodel:\n  provider: openrouter\n  default: bom/model\n"
    )
    (home / "config.yaml").write_bytes(config_bytes)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))

    assert asyncio.run(runner.start()) is True

    profile = read_runtime_status()["effective_config"]["profiles"]["default"]
    assert profile["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert profile["primary"] == {
        "provider": "openrouter",
        "model": "bom/model",
    }


def test_multiplex_receipt_is_keyed_by_every_profile(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.5\n",
        encoding="utf-8",
    )
    (coder_home / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-sonnet-4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={},
            sessions_dir=home / "sessions",
            multiplex_profiles=True,
        )
    )

    assert asyncio.run(runner.start()) is True

    receipt = read_runtime_status()["effective_config"]
    assert receipt["complete"] is True
    assert set(receipt["profiles"]) == {"default", "coder"}
    assert receipt["profiles"]["default"]["primary"] == {
        "provider": "openrouter",
        "model": "openai/gpt-5.5",
    }
    assert receipt["profiles"]["coder"]["profile_path"] == str(coder_home.resolve())
    assert receipt["profiles"]["coder"]["primary"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-4",
    }


def test_failed_secondary_config_load_cannot_claim_complete(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    broken_home = home / "profiles" / "broken"
    broken_home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.5\n",
        encoding="utf-8",
    )
    (broken_home / "config.yaml").write_text("model: [", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={},
            sessions_dir=home / "sessions",
            multiplex_profiles=True,
        )
    )

    assert asyncio.run(runner.start()) is True

    state = read_runtime_status()
    receipt = state["effective_config"]
    assert state["gateway_state"] == "running"
    assert receipt["complete"] is False
    assert set(receipt["profiles"]) == {"default"}


def test_multiplex_receipt_keeps_secondary_snapshot_consumed_at_startup(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: primary\n  default: primary/model\n",
        encoding="utf-8",
    )
    coder_config = coder_home / "config.yaml"
    original = b"model:\n  provider: first\n  default: first/model\n"
    replacement = b"model:\n  provider: second\n  default: second/model\n"
    coder_config.write_bytes(original)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={},
            sessions_dir=home / "sessions",
            multiplex_profiles=True,
        )
    )
    original_start = runner._start_one_profile_adapters

    async def start_then_replace(profile_name, profile_home, claimed):
        result = await original_start(profile_name, profile_home, claimed)
        coder_config.write_bytes(replacement)
        return result

    monkeypatch.setattr(runner, "_start_one_profile_adapters", start_then_replace)

    assert asyncio.run(runner.start()) is True

    profile = read_runtime_status()["effective_config"]["profiles"]["coder"]
    assert profile["config_sha256"] == hashlib.sha256(original).hexdigest()
    assert profile["primary"] == {"provider": "first", "model": "first/model"}
    assert coder_config.read_bytes() == replacement


def test_secondary_without_config_yaml_uses_fallback_without_attestation(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    legacy_home = home / "profiles" / "legacy"
    legacy_home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: primary\n  default: primary/model\n",
        encoding="utf-8",
    )
    (legacy_home / "gateway.json").write_text(
        '{"always_log_local": false}', encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={},
            sessions_dir=home / "sessions",
            multiplex_profiles=True,
        )
    )
    real_load_gateway_config = gateway_config.load_gateway_config
    loaded_fallback_values = []

    def track_gateway_config_fallback():
        loaded = real_load_gateway_config()
        loaded_fallback_values.append(loaded.always_log_local)
        return loaded

    monkeypatch.setattr(
        "gateway.config.load_gateway_config", track_gateway_config_fallback
    )

    assert asyncio.run(runner.start()) is True

    receipt = read_runtime_status()["effective_config"]
    assert loaded_fallback_values == [False]
    assert receipt["complete"] is False
    assert set(receipt["profiles"]) == {"default"}


def test_failed_startup_never_publishes_effective_config(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: openai/gpt-5.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.YUANBAO: PlatformConfig(
                    enabled=True,
                    extra={"dm_policy": "open"},
                ),
            },
            sessions_dir=home / "sessions",
        )
    )

    assert asyncio.run(runner.start()) is True

    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    assert state["effective_config"] is None


def test_public_startup_clears_stale_receipt_when_runner_construction_fails(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)
    gateway_config = GatewayConfig(platforms={}, sessions_dir=home / "sessions")
    write_runtime_status(
        gateway_state="running",
        effective_config={"schema": 1, "profiles": {"stale": {}}},
    )
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.is_gateway_runtime_lock_active", lambda: False)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda **kwargs: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.security_audit_startup.log_startup_security_warnings",
        lambda **kwargs: None,
    )

    def fail_constructor(config):
        raise RuntimeError("constructor failed")

    monkeypatch.setattr("gateway.run.GatewayRunner", fail_constructor)

    assert asyncio.run(start_gateway(gateway_config, verbosity=None)) is False

    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"
    assert state["effective_config"] is None


def test_constructor_failing_contender_preserves_live_owner_receipt(
    monkeypatch, tmp_path
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)
    owner_receipt = {"schema": 1, "profiles": {"owner": {"config_sha256": "abc"}}}
    write_runtime_status(gateway_state="running", effective_config=owner_receipt)
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    # Both contenders passed the initial PID check; the winner then acquired
    # the authoritative runtime lock before this contender's constructor failed.
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.is_gateway_runtime_lock_active", lambda: True)
    monkeypatch.setattr(
        "gateway.status.acquire_gateway_runtime_lock",
        lambda: (_ for _ in ()).throw(AssertionError("must not steal owner lock")),
    )
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda **kwargs: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.security_audit_startup.log_startup_security_warnings",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner",
        lambda config: (_ for _ in ()).throw(RuntimeError("constructor failed")),
    )

    assert asyncio.run(
        start_gateway(
            GatewayConfig(platforms={}, sessions_dir=home / "sessions"),
            verbosity=None,
        )
    ) is False

    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["effective_config"] == owner_receipt


def test_receipt_uses_config_snapshot_consumed_by_runtime(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    loaded_bytes = (
        b"model:\n  provider: first\n  default: first/model\n"
        b"agent:\n  system_prompt: original startup prompt\n"
    )
    replacement_bytes = (
        b"model:\n  provider: second\n  default: second/model\n"
        b"agent:\n  system_prompt: replacement prompt\n"
    )
    config_path.write_bytes(loaded_bytes)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))
    assert runner._ephemeral_system_prompt == "original startup prompt"

    # Mutate disk after the real runtime load but before the running receipt.
    config_path.write_bytes(replacement_bytes)

    assert asyncio.run(runner.start()) is True

    profile = read_runtime_status()["effective_config"]["profiles"]["default"]
    assert profile["config_sha256"] == hashlib.sha256(loaded_bytes).hexdigest()
    assert profile["primary"] == {"provider": "first", "model": "first/model"}
    assert config_path.read_bytes() == replacement_bytes


def test_attested_model_matches_normalized_runtime_model_path(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: ${RUNTIME_MODEL}\n"
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: ${FALLBACK_MODEL}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_MODEL", "expanded/model")
    monkeypatch.setenv("FALLBACK_MODEL", "expanded/fallback")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))
    startup_snapshot = gateway_run._load_gateway_config_snapshot()
    token = gateway_run._startup_config_snapshot_override.set(
        startup_snapshot
    )
    try:
        runtime_model = _resolve_gateway_model()
        runtime_fallbacks = runner._load_fallback_model()
    finally:
        gateway_run._startup_config_snapshot_override.reset(token)

    assert asyncio.run(runner.start()) is True

    profile = read_runtime_status()["effective_config"]["profiles"]["default"]
    assert runtime_model == "${RUNTIME_MODEL}"
    assert profile["primary"] == {
        "provider": "openrouter",
        "model": runtime_model,
    }
    assert runtime_fallbacks == [
        {"provider": "openrouter", "model": "${FALLBACK_MODEL}"}
    ]
    assert profile["fallbacks"] == runtime_fallbacks


def test_gateway_receipt_matches_fleet_v1_contract_example(monkeypatch, tmp_path):
    """Keep Hermes' producer shape executable against Fleet's public contract."""
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "contracts"
        / "gateway-effective-config-v1.example.json"
    )
    contract_state = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_receipt = contract_state["effective_config"]
    contract_profile = contract_receipt["profiles"]["default"]

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-5.5\n"
        "fallback_providers:\n"
        "  - provider: xai-oauth\n    model: grok-4.3\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._hermes_home", home)

    runner = GatewayRunner(GatewayConfig(platforms={}, sessions_dir=home / "sessions"))
    assert asyncio.run(runner.start()) is True

    runtime_state = read_runtime_status()
    runtime_receipt = runtime_state["effective_config"]
    runtime_profile = runtime_receipt["profiles"]["default"]
    assert contract_state["gateway_state"] == runtime_state["gateway_state"] == "running"
    assert set(runtime_receipt) == set(contract_receipt)
    assert set(runtime_profile) == set(contract_profile)
    assert runtime_receipt["schema"] == contract_receipt["schema"] == 1
    assert runtime_receipt["complete"] is contract_receipt["complete"] is True
    assert isinstance(runtime_receipt["pid"], type(contract_receipt["pid"]))
    assert isinstance(runtime_receipt["start_time"], (int, float))
    assert isinstance(contract_receipt["start_time"], (int, float))
    assert isinstance(runtime_receipt["loaded_at"], type(contract_receipt["loaded_at"]))
    assert isinstance(runtime_profile["fallbacks"], type(contract_profile["fallbacks"]))
