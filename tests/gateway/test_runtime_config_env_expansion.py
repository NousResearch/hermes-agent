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


