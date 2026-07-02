"""Tests for the Codex gpt-5.5 autoraise notice UX switch.

The autoraise behavior and its user-facing notice are intentionally separate:
Codex gpt-5.5 should still compact at 85%, while gateway/CLI notices can be
suppressed for long-running chat sessions where the notice is noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from hermes_state import SessionDB
from run_agent import AIAgent


def _make_codex_gpt55_agent(compression_cfg: dict, tmp_path: Path) -> Any:
    cfg = {
        "compression": compression_cfg,
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
    }
    db = SessionDB(db_path=tmp_path / "state.db")
    with patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://api.openai.com/v1",
            model="gpt-5.5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
        )


def test_codex_gpt55_notice_can_be_suppressed_without_disabling_autoraise(tmp_path: Path) -> None:
    agent = _make_codex_gpt55_agent(
        {
            "enabled": True,
            "threshold": 0.50,
            "codex_gpt55_autoraise": True,
            "codex_gpt55_autoraise_notice": False,
        },
        tmp_path,
    )

    assert getattr(agent, "_compression_threshold_autoraised") == {"from": 0.50, "to": 0.85}
    assert agent.context_compressor.context_length == 272_000
    assert agent.context_compressor.threshold_tokens == 231_200
    assert getattr(agent, "_compression_warning") is None


def test_codex_gpt55_notice_defaults_visible_for_existing_users(tmp_path: Path) -> None:
    agent = _make_codex_gpt55_agent(
        {
            "enabled": True,
            "threshold": 0.50,
            "codex_gpt55_autoraise": True,
        },
        tmp_path,
    )

    warning = getattr(agent, "_compression_warning")
    assert getattr(agent, "_compression_threshold_autoraised") == {"from": 0.50, "to": 0.85}
    assert warning is not None
    assert "auto-compaction was raised to 85%" in warning
