"""``compression.warn_after_compressions`` survives agent construction.

The emit-path tests assign the attribute directly. This file is the
config → ``AIAgent`` path: custom, disabled, and invalid values.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


def _config(warn_after_compressions) -> dict:
    return {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            "warn_after_compressions": warn_after_compressions,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_agent(monkeypatch, tmp_path: Path, warn_after_compressions):
    from hermes_cli import config as config_mod

    cfg = _config(warn_after_compressions)
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
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
            session_id="warn-after-config-test",
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (5, 5),
        (0, 0),
        (None, 2),
        ("abc", 2),
        (True, 2),
    ],
)
def test_warn_after_compressions_config_to_agent_attribute(monkeypatch, tmp_path, raw, expected):
    agent = _make_agent(monkeypatch, tmp_path, raw)
    assert agent.compression_warn_after_compressions == expected
