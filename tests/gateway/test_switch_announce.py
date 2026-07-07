"""Tests for the deliberate /reasoning + /model switch channel announce (P2).

A deliberate switch's confirmation is visible only to whoever ran the command
(ephemeral on native Discord slash), so the rest of the conversation only saw the
footer move. P2 posts a channel-visible ``🔀`` line whenever the *effective* state
changes, gated by ``model.announce_switch`` (default on), best-effort.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text, platform=Platform.DISCORD, user_id="u1", chat_id="c1"):
    source = SessionSource(platform=platform, user_id=user_id, chat_id=chat_id, user_name="ace")
    return MessageEvent(text=text, source=source)


def _make_runner(monkeypatch, tmp_path, config_yaml="agent:\n  reasoning_effort: xhigh\n"):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    # _load_gateway_config reads the live config path; point it at our temp file.
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _yaml_load(hermes_home / "config.yaml"))

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_reasoning_overrides = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._show_reasoning = False
    runner._reasoning_config = None
    runner._evict_cached_agent = MagicMock()
    # Stub adapter capturing send(chat_id, text, metadata=...)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.DISCORD: adapter}
    return runner, adapter


def _yaml_load(path):
    import yaml
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _sent_texts(adapter):
    return [c.args[1] for c in adapter.send.call_args_list]


class TestReasoningSwitchAnnounce:
    @pytest.mark.asyncio
    async def test_reasoning_switch_announces(self, tmp_path, monkeypatch):
        # config xhigh, session already high → /reasoning xhigh announces high→xhigh
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        ev = _make_event("/reasoning xhigh")
        session_key = runner._session_key_for_source(ev.source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
        await runner._handle_reasoning_command(ev)
        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: high → xhigh" in tx for tx in texts), texts

    @pytest.mark.asyncio
    async def test_no_announce_on_noop_reasoning_config_default(self, tmp_path, monkeypatch):
        # RC-1: NO override, config default xhigh, /reasoning xhigh → silent (no-op).
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        ev = _make_event("/reasoning xhigh")
        await runner._handle_reasoning_command(ev)
        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_reasoning_switch_old_side_is_config_default(self, tmp_path, monkeypatch):
        # RC-1: NO override, config xhigh, /reasoning high → 🔀 Reasoning: xhigh → high
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        ev = _make_event("/reasoning high")
        await runner._handle_reasoning_command(ev)
        texts = _sent_texts(adapter)
        assert any("🔀 Reasoning: xhigh → high" in tx for tx in texts), texts

    @pytest.mark.asyncio
    async def test_no_announce_on_noop_reasoning_explicit_override(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        ev = _make_event("/reasoning high")
        session_key = runner._session_key_for_source(ev.source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
        await runner._handle_reasoning_command(ev)
        assert adapter.send.call_count == 0, _sent_texts(adapter)

    @pytest.mark.asyncio
    async def test_no_announce_on_display_toggle(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        await runner._handle_reasoning_command(_make_event("/reasoning show"))
        assert adapter.send.call_count == 0

    @pytest.mark.asyncio
    async def test_announce_baseline_matches_footer_resolver(self, tmp_path, monkeypatch):
        # RC-A: for every config state, the announce baseline == the footer-resolved
        # effective effort, so a no-op never spuriously announces.
        for effort in ("xhigh", "high", "medium"):
            runner, _ = _make_runner(monkeypatch, tmp_path, f"agent:\n  reasoning_effort: {effort}\n")
            ev = _make_event("/reasoning")
            session_key = runner._session_key_for_source(ev.source)
            baseline = runner._resolved_effort_label(source=ev.source, session_key=session_key)
            assert baseline == effort


class TestModelSwitchAnnounce:
    @pytest.mark.asyncio
    async def test_model_switch_announces(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        source = _make_event("/model x").source
        await runner._announce_switch(
            source, "Model", "claude-app/claude-opus-4-8", "claude-app/claude-fable-5"
        )
        texts = _sent_texts(adapter)
        assert any("🔀 Model: claude-app/claude-opus-4-8 → claude-app/claude-fable-5" in tx for tx in texts)

    @pytest.mark.asyncio
    async def test_no_announce_on_same_model(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        source = _make_event("/model x").source
        await runner._announce_switch(source, "Model", "claude-app/opus", "claude-app/opus")
        assert adapter.send.call_count == 0


class TestGateAndBestEffort:
    @pytest.mark.asyncio
    async def test_announce_switch_gate_off_suppresses(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(
            monkeypatch, tmp_path,
            "agent:\n  reasoning_effort: xhigh\nmodel:\n  announce_switch: false\n",
        )
        source = _make_event("/x").source
        await runner._announce_switch(source, "Reasoning", "high", "xhigh")
        assert adapter.send.call_count == 0

    @pytest.mark.asyncio
    async def test_announce_send_failure_does_not_break_handler(self, tmp_path, monkeypatch):
        runner, adapter = _make_runner(monkeypatch, tmp_path)
        adapter.send = AsyncMock(side_effect=RuntimeError("boom"))
        ev = _make_event("/reasoning high")
        # Handler must still return its confirmation string despite the raising adapter.
        result = await runner._handle_reasoning_command(ev)
        assert isinstance(result, str) and result
