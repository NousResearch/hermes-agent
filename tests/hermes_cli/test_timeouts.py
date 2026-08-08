from __future__ import annotations

import textwrap

from hermes_cli.timeouts import (
    ProviderStallRecoveryConfig,
    get_provider_request_timeout,
    get_provider_stale_timeout,
    get_provider_stall_recovery_config,
)


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def test_provider_stall_recovery_defaults(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})

    assert get_provider_stall_recovery_config() == ProviderStallRecoveryConfig(
        enabled=True,
        health_probe_enabled=False,
        health_probe_timeout_seconds=5.0,
        same_provider_retries=1,
    )


def test_provider_stall_recovery_explicit_values(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "agent": {
                "provider_stall_recovery": {
                    "enabled": False,
                    "health_probe_enabled": True,
                    "health_probe_timeout_seconds": 12.5,
                    "same_provider_retries": 0,
                }
            }
        },
    )

    assert get_provider_stall_recovery_config() == ProviderStallRecoveryConfig(
        enabled=False,
        health_probe_enabled=True,
        health_probe_timeout_seconds=12.5,
        same_provider_retries=0,
    )


def test_provider_stall_recovery_malformed_values_use_safe_defaults(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "agent": {
                "provider_stall_recovery": {
                    "enabled": "false",
                    "health_probe_enabled": "true",
                    "health_probe_timeout_seconds": "invalid",
                    "same_provider_retries": object(),
                }
            }
        },
    )

    assert get_provider_stall_recovery_config() == ProviderStallRecoveryConfig()


def test_provider_stall_recovery_non_dict_config_uses_defaults(monkeypatch):
    for config in (None, [], {"agent": []}, {"agent": {"provider_stall_recovery": []}}):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", lambda config=config: config
        )
        assert get_provider_stall_recovery_config() == ProviderStallRecoveryConfig()


def test_provider_stall_recovery_clamps_probe_timeout_and_retry_count(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "agent": {
                "provider_stall_recovery": {
                    "health_probe_timeout_seconds": 999,
                    "same_provider_retries": 8,
                }
            }
        },
    )

    config = get_provider_stall_recovery_config()
    assert config.health_probe_timeout_seconds == 30.0
    assert config.same_provider_retries == 1


def test_provider_stall_recovery_clamps_values_below_bounds(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "agent": {
                "provider_stall_recovery": {
                    "health_probe_timeout_seconds": 0.25,
                    "same_provider_retries": -4,
                }
            }
        },
    )

    config = get_provider_stall_recovery_config()
    assert config.health_probe_timeout_seconds == 1.0
    assert config.same_provider_retries == 0










def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the 900s default read timeout."""
    pytest = __import__("pytest")
    anthropic = pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Default stays at 900s; custom overrides; invalid falls back to default
    assert c_default.timeout.read == 900.0
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == 900.0
    # Connect timeout always stays at 10s regardless
    assert c_default.timeout.connect == 10.0
    assert c_custom.timeout.connect == 10.0


def test_resolved_api_call_timeout_priority(monkeypatch, tmp_path):
    """AIAgent._resolved_api_call_timeout() honors config > env > default priority."""
    # Isolate HERMES_HOME
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    # Case A: config wins over env var
    _write_config(tmp_path, """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
        """)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # Per-model override wins
    assert agent._resolved_api_call_timeout() == 42.0

    # Provider-level (different model, no per-model override)
    agent.model = "some/other-model"
    assert agent._resolved_api_call_timeout() == 77.0

    # Case B: no config → env wins
    _write_config(tmp_path, "")
    # Clear the cached config load
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="some/model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_timeout() == 999.0

    # Case C: no config, no env → 1800.0 default
    monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_timeout() == 1800.0



