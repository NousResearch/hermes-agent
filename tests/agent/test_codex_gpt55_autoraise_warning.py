"""Regression tests for the Codex gpt-5.5 autoraise warning gate."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from hermes_state import SessionDB
from run_agent import AIAgent


def _config(*, show_warning: bool, legacy_notice_key: bool = False) -> dict:
    warning_key = (
        "codex_gpt55_autoraise_notice"
        if legacy_notice_key
        else "codex_gpt55_autoraise_warning"
    )
    return {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            "codex_gpt55_autoraise": True,
            warning_key: show_warning,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_codex_agent(
    monkeypatch,
    tmp_path: Path,
    *,
    show_warning: bool,
    legacy_notice_key: bool = False,
):
    """Construct a real Codex gpt-5.5 agent under an isolated config."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: _config(
            show_warning=show_warning,
            legacy_notice_key=legacy_notice_key,
        ),
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        agent = AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model="gpt-5.5",
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=False,
            skip_memory=True,
            session_db=db,
            session_id="codex-warning-test",
        )

    return agent, stdout.getvalue()


def _threshold_ratio(agent: AIAgent) -> float:
    compressor = getattr(agent, "context_compressor")
    return round(compressor.threshold_tokens / compressor.context_length, 2)


def test_codex_gpt55_autoraise_warning_enabled_by_default(monkeypatch, tmp_path):
    agent, stdout = _make_codex_agent(monkeypatch, tmp_path, show_warning=True)

    assert _threshold_ratio(agent) == 0.85
    warning = getattr(agent, "_compression_warning")
    assert warning is not None
    assert "auto-compaction was raised" in warning
    assert "auto-compaction was raised" in stdout


def test_codex_gpt55_autoraise_warning_can_be_suppressed_without_disabling_autoraise(
    monkeypatch, tmp_path
):
    agent, stdout = _make_codex_agent(monkeypatch, tmp_path, show_warning=False)

    assert _threshold_ratio(agent) == 0.85
    assert getattr(agent, "_compression_warning") is None
    assert "auto-compaction was raised" not in stdout


def test_legacy_notice_key_still_suppresses_warning(monkeypatch, tmp_path):
    agent, stdout = _make_codex_agent(
        monkeypatch,
        tmp_path,
        show_warning=False,
        legacy_notice_key=True,
    )

    assert _threshold_ratio(agent) == 0.85
    assert getattr(agent, "_compression_warning") is None
    assert "auto-compaction was raised" not in stdout
