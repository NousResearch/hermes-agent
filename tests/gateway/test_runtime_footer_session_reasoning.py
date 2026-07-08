"""Tests for the runtime footer honoring the SESSION reasoning override (P1).

Regression coverage for the reported bug: ``/reasoning high`` (a session-scoped
override) left the runtime footer showing ``r:xhigh`` (the global config default)
because the footer builder fell back to ``agent.reasoning_effort`` when the caller
passed no ``reasoning=`` value. ``_reasoning_effort_for_footer`` resolves the
session-aware value so the footer reports what the turn actually ran at.
"""

import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.runtime_footer import build_footer_line
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._session_reasoning_overrides = {}
    return runner


def _write_config(tmp_path, monkeypatch, effort="xhigh"):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        f"agent:\n  reasoning_effort: {effort}\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    return hermes_home


class TestReasoningEffortForFooter:
    def test_reflects_session_reasoning_override(self, tmp_path, monkeypatch):
        # config default xhigh, but this session was /reasoning high → footer must say high.
        _write_config(tmp_path, monkeypatch, effort="xhigh")
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

        assert runner._reasoning_effort_for_footer(source=source, session_key=session_key) == "high"

    def test_falls_back_to_config_when_no_override(self, tmp_path, monkeypatch):
        # No override → return "" so build_footer_line applies its config fallback (xhigh).
        _write_config(tmp_path, monkeypatch, effort="xhigh")
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)

        assert runner._reasoning_effort_for_footer(source=source, session_key=session_key) == ""

    def test_disabled_override_shows_none(self, tmp_path, monkeypatch):
        _write_config(tmp_path, monkeypatch, effort="xhigh")
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)
        runner._session_reasoning_overrides[session_key] = {"enabled": False}

        assert runner._reasoning_effort_for_footer(source=source, session_key=session_key) == "none"

    def test_e2e_footer_shows_override_not_config_default(self, tmp_path, monkeypatch):
        # End-to-end through the REAL build_footer_line: session override high vs config xhigh.
        _write_config(tmp_path, monkeypatch, effort="xhigh")
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

        footer_cfg = {
            "agent": {"reasoning_effort": "xhigh"},
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["provider_model", "reasoning", "context_full"],
                }
            },
        }
        resolved = runner._reasoning_effort_for_footer(source=source, session_key=session_key)
        line = build_footer_line(
            user_config=footer_cfg,
            platform_key="discord",
            model="claude-fable-5",
            provider="claude-app",
            context_tokens=184_400,
            context_length=1_000_000,
            reasoning=(resolved or None),
        )
        assert "r:high" in line
        assert "r:xhigh" not in line

    def test_e2e_footer_config_fallback_without_override(self, tmp_path, monkeypatch):
        _write_config(tmp_path, monkeypatch, effort="xhigh")
        runner = _make_runner()
        source = _make_event().source
        session_key = runner._session_key_for_source(source)  # no override set

        footer_cfg = {
            "agent": {"reasoning_effort": "xhigh"},
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["provider_model", "reasoning", "context_full"],
                }
            },
        }
        resolved = runner._reasoning_effort_for_footer(source=source, session_key=session_key)
        line = build_footer_line(
            user_config=footer_cfg,
            platform_key="discord",
            model="claude-opus-4-8",
            provider="claude-app",
            context_tokens=100_000,
            context_length=1_000_000,
            reasoning=(resolved or None),
        )
        assert "r:xhigh" in line
