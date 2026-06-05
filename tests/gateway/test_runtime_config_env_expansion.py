"""Regression tests for gateway runtime config env-var expansion."""

from __future__ import annotations

import json

import pytest

import gateway.run as gateway_run


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.fixture
def gateway_home(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.delenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INPUT_MODE", raising=False)
    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
    return tmp_path


def test_load_prefill_messages_expands_env_var_path(monkeypatch, gateway_home):
    prefill = [{"role": "system", "content": "few-shot"}]
    (gateway_home / "prefill.json").write_text(json.dumps(prefill), encoding="utf-8")
    _write_config(gateway_home, "prefill_messages_file: ${PREFILL_FILE}\n")
    monkeypatch.setenv("PREFILL_FILE", "prefill.json")

    assert gateway_run.GatewayRunner._load_prefill_messages() == prefill


def test_load_prefill_messages_accepts_legacy_agent_key(monkeypatch, gateway_home):
    prefill = [{"role": "system", "content": "legacy few-shot"}]
    (gateway_home / "prefill.json").write_text(json.dumps(prefill), encoding="utf-8")
    _write_config(gateway_home, "agent:\n  prefill_messages_file: prefill.json\n")

    assert gateway_run.GatewayRunner._load_prefill_messages() == prefill


def test_load_prefill_messages_prefers_top_level_over_legacy(monkeypatch, gateway_home):
    top_level = [{"role": "system", "content": "top-level"}]
    legacy = [{"role": "system", "content": "legacy"}]
    (gateway_home / "top.json").write_text(json.dumps(top_level), encoding="utf-8")
    (gateway_home / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    _write_config(
        gateway_home,
        "prefill_messages_file: top.json\n"
        "agent:\n"
        "  prefill_messages_file: legacy.json\n",
    )

    assert gateway_run.GatewayRunner._load_prefill_messages() == top_level


@pytest.mark.parametrize(
    ("config_body", "env_name", "env_value", "loader_name", "expected"),
    [
        (
            "agent:\n  system_prompt: ${GW_PROMPT}\n",
            "GW_PROMPT",
            "expanded prompt",
            "_load_ephemeral_system_prompt",
            "expanded prompt",
        ),
        (
            "agent:\n  reasoning_effort: ${REASONING_LEVEL}\n",
            "REASONING_LEVEL",
            "high",
            "_load_reasoning_config",
            {"enabled": True, "effort": "high"},
        ),
        (
            "agent:\n  service_tier: ${SERVICE_TIER}\n",
            "SERVICE_TIER",
            "priority",
            "_load_service_tier",
            "priority",
        ),
        (
            "display:\n  busy_input_mode: ${BUSY_MODE}\n",
            "BUSY_MODE",
            "steer",
            "_load_busy_input_mode",
            "steer",
        ),
        (
            "agent:\n  restart_drain_timeout: ${DRAIN_TIMEOUT}\n",
            "DRAIN_TIMEOUT",
            "12",
            "_load_restart_drain_timeout",
            12.0,
        ),
        (
            "display:\n  background_process_notifications: ${BG_MODE}\n",
            "BG_MODE",
            "error",
            "_load_background_notifications_mode",
            "error",
        ),
    ],
)
def test_gateway_runtime_loaders_expand_env_var_templates(
    monkeypatch,
    gateway_home,
    config_body,
    env_name,
    env_value,
    loader_name,
    expected,
):
    _write_config(gateway_home, config_body)
    monkeypatch.setenv(env_name, env_value)

    loader = getattr(gateway_run.GatewayRunner, loader_name)

    assert loader() == expected
