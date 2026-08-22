"""Configuration coverage for engine-driven preflight maintenance."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from hermes_state import SessionDB
from run_agent import AIAgent


def _config(preflight_enabled=None) -> dict:
    compression = {
        "enabled": True,
        "threshold": 0.50,
        "target_ratio": 0.20,
        "protect_first_n": 3,
        "protect_last_n": 20,
    }
    if preflight_enabled is not None:
        compression["preflight_enabled"] = preflight_enabled
    return {
        "compression": compression,
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_agent(monkeypatch, tmp_path: Path, *, preflight_enabled=None):
    from hermes_cli import config as config_mod

    config = _config(preflight_enabled=preflight_enabled)
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)
    db = SessionDB(db_path=tmp_path / "state.db")
    with contextlib.redirect_stdout(io.StringIO()):
        return AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model="gpt-5.5",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=True,
            skip_memory=True,
            session_db=db,
            session_id="preflight-config-test",
        )


def test_engine_preflight_defaults_enabled_when_unset(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)
    assert agent.compression_preflight_enabled is True


def test_engine_preflight_can_be_disabled(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, preflight_enabled=False)
    assert agent.compression_preflight_enabled is False
