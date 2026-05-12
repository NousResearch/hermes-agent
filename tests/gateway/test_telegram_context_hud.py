"""Telegram context HUD: gateway-side integration tests.

Covers the helpers that decide whether to render a HUD for a given turn,
plus the new ``/context`` and ``/hud`` command handlers.  Uses mocked
agent state — no real LLM calls, no real platform calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform


# ── isolated harness ──────────────────────────────────────────────────────


def _telegram_source(chat_id: str = "100", user_id: str = "42", thread_id=None):
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
        chat_type="dm",
    )


def _slack_source():
    return SimpleNamespace(
        platform=Platform.SLACK,
        chat_id="C1",
        user_id="U1",
        thread_id=None,
        chat_type="dm",
    )


def _harness(user_config: dict | None = None, agent=None):
    """Return a stub object with just enough surface to call HUD helpers."""
    from gateway.run import GatewayRunner

    obj = GatewayRunner.__new__(GatewayRunner)
    obj._session_hud_overrides = {}
    obj._read_user_config = MagicMock(return_value=user_config or {})
    # _resolve_session_token_state probes _running_agents and _agent_cache,
    # then falls back to session_store.  Wire all of them.
    obj._running_agents = {}
    obj._agent_cache_lock = None
    obj._agent_cache = None
    if agent is not None:
        obj._running_agents = {"agent:main:telegram:dm:100:42": agent}
    obj.session_store = MagicMock()
    obj.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="sess_abc"
    )
    obj.session_store.load_transcript.return_value = []
    return obj


def _fake_agent(used: int, limit: int, model="claude-sonnet-4-6", provider="openrouter"):
    ctx = SimpleNamespace(
        last_prompt_tokens=used,
        context_length=limit,
        compression_count=0,
    )
    return SimpleNamespace(
        context_compressor=ctx,
        model=model,
        provider=provider,
    )


# ── _is_telegram_hud_enabled ──────────────────────────────────────────────


class TestHudEnabledGate:
    def test_off_for_non_telegram(self):
        h = _harness()
        assert h._is_telegram_hud_enabled(_slack_source(), "k") is False

    def test_on_by_default_for_telegram(self):
        h = _harness()
        assert h._is_telegram_hud_enabled(_telegram_source(), "k") is True

    def test_config_disabled(self):
        h = _harness({"platforms": {"telegram": {"extra": {"context_hud": {"enabled": False}}}}})
        assert h._is_telegram_hud_enabled(_telegram_source(), "k") is False

    def test_session_override_beats_config(self):
        h = _harness({"platforms": {"telegram": {"extra": {"context_hud": {"enabled": False}}}}})
        h._session_hud_overrides["k"] = True
        assert h._is_telegram_hud_enabled(_telegram_source(), "k") is True

    def test_session_off_overrides_config_on(self):
        h = _harness()  # default enabled=True
        h._session_hud_overrides["k"] = False
        assert h._is_telegram_hud_enabled(_telegram_source(), "k") is False


# ── _compute_telegram_hud_prefix ──────────────────────────────────────────


class TestComputeHudPrefix:
    def test_none_for_non_telegram(self):
        h = _harness()
        assert h._compute_telegram_hud_prefix(_slack_source(), "k") is None

    def test_none_when_no_context_length_known(self):
        # No agent and empty transcript — limit unknown
        h = _harness()
        assert h._compute_telegram_hud_prefix(_telegram_source(), "agent:main:telegram:dm:100:42") is None

    def test_uses_live_agent_state(self):
        agent = _fake_agent(used=22_500, limit=250_000)
        h = _harness(agent=agent)
        out = h._compute_telegram_hud_prefix(_telegram_source(), "agent:main:telegram:dm:100:42")
        assert out is not None
        first, second = out.split("\n", 1)
        assert "22k" in first
        assert "250k" in first
        assert second.startswith("[") and second.endswith("]")

    def test_hide_below_threshold_returns_none(self):
        agent = _fake_agent(used=100, limit=250_000)  # 0.04%
        h = _harness(agent=agent)
        assert h._compute_telegram_hud_prefix(_telegram_source(), "agent:main:telegram:dm:100:42") is None

    def test_config_overrides_propagate(self):
        agent = _fake_agent(used=100, limit=250_000)  # 0.04%
        h = _harness(
            user_config={
                "platforms": {
                    "telegram": {
                        "extra": {
                            "context_hud": {
                                "hide_below_percent": 0,
                                "bar_width": 10,
                            }
                        }
                    }
                }
            },
            agent=agent,
        )
        out = h._compute_telegram_hud_prefix(_telegram_source(), "agent:main:telegram:dm:100:42")
        assert out is not None
        _, second = out.split("\n", 1)
        assert len(second) == 12  # 10 cells + 2 brackets


# ── HUD prefix: session title line ────────────────────────────────────────


class TestComputeHudPrefixTitle:
    def test_prepends_quoted_session_title_line(self):
        agent = _fake_agent(used=22_500, limit=250_000)
        h = _harness(agent=agent)
        h._session_db = MagicMock()
        h._session_db.get_session_title.return_value = "Update Help"
        out = h._compute_telegram_hud_prefix(
            _telegram_source(), "agent:main:telegram:dm:100:42"
        )
        assert out is not None
        lines = out.split("\n")
        # First line is exactly the quoted title — no label, no prefix, no suffix.
        assert lines[0] == '"Update Help"'
        # Existing HUD body (metric line + bar) is preserved beneath it.
        assert "22k" in lines[1]
        assert "250k" in lines[1]
        assert lines[2].startswith("[") and lines[2].endswith("]")
        h._session_db.get_session_title.assert_called_with("sess_abc")

    def test_falls_back_to_session_id_when_title_missing(self):
        agent = _fake_agent(used=22_500, limit=250_000)
        h = _harness(agent=agent)
        h._session_db = MagicMock()
        h._session_db.get_session_title.return_value = None
        out = h._compute_telegram_hud_prefix(
            _telegram_source(), "agent:main:telegram:dm:100:42"
        )
        assert out is not None
        assert out.split("\n", 1)[0] == '"sess_abc"'

    def test_no_title_line_when_session_db_unavailable(self):
        # Defensive path: when no session DB is wired up there is nothing to
        # look the title up against, so the HUD body renders unchanged.
        agent = _fake_agent(used=22_500, limit=250_000)
        h = _harness(agent=agent)  # _session_db not set on harness
        out = h._compute_telegram_hud_prefix(
            _telegram_source(), "agent:main:telegram:dm:100:42"
        )
        assert out is not None
        assert "22k" in out.split("\n", 1)[0]


# ── /hud command ──────────────────────────────────────────────────────────


def _event_for(args: str = "", source=None, command="hud"):
    src = source or _telegram_source()
    ev = SimpleNamespace(source=src)
    ev.get_command_args = lambda: args
    ev.get_command = lambda: command
    return ev


def _harness_with_session_key(harness):
    harness._session_key_for_source = lambda source: "agent:main:telegram:dm:100:42"
    return harness


@pytest.mark.asyncio
class TestHudCommand:
    async def test_status_default(self):
        h = _harness_with_session_key(_harness())
        out = await h._handle_hud_command(_event_for(""))
        assert "HUD: on" in out
        assert "Default (config): on" in out

    async def test_on_sets_override(self):
        h = _harness_with_session_key(_harness(
            {"platforms": {"telegram": {"extra": {"context_hud": {"enabled": False}}}}}
        ))
        out = await h._handle_hud_command(_event_for("on"))
        assert "HUD on" in out
        assert h._session_hud_overrides["agent:main:telegram:dm:100:42"] is True

    async def test_off_sets_override(self):
        h = _harness_with_session_key(_harness())
        out = await h._handle_hud_command(_event_for("off"))
        assert "HUD off" in out
        assert h._session_hud_overrides["agent:main:telegram:dm:100:42"] is False

    async def test_reset_clears_override(self):
        h = _harness_with_session_key(_harness())
        h._session_hud_overrides["agent:main:telegram:dm:100:42"] = False
        out = await h._handle_hud_command(_event_for("reset"))
        assert "cleared" in out.lower()
        assert "agent:main:telegram:dm:100:42" not in h._session_hud_overrides

    async def test_rejects_non_telegram_platform(self):
        h = _harness_with_session_key(_harness())
        out = await h._handle_hud_command(_event_for("", source=_slack_source()))
        assert "Telegram-only" in out


# ── /context command ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestContextCommand:
    async def test_reports_session_and_limit_when_agent_available(self):
        agent = _fake_agent(used=10_000, limit=200_000, model="anthropic/claude-sonnet-4-6")
        h = _harness_with_session_key(_harness(agent=agent))
        out = await h._handle_context_command(_event_for(""))
        assert "sess_abc" in out
        assert "anthropic/claude-sonnet-4-6" in out
        assert "10,000" in out
        assert "200,000" in out
        assert "5%" in out
        assert "/reset" in out
        assert "HUD" in out  # Telegram-only line included

    async def test_reports_estimate_when_no_agent(self):
        h = _harness_with_session_key(_harness())
        out = await h._handle_context_command(_event_for(""))
        assert "limit unknown" in out
        assert "/reset" in out
