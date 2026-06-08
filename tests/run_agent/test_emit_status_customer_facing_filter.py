# -*- coding: utf-8 -*-
"""Unit tests for the ``_emit_status`` customer-facing platform filter.

Refs: https://github.com/NousResearch/hermes-agent/issues/28208

When Hermes runs over WhatsApp (or any other customer-facing messaging
platform), internal Hermes diagnostics like the compression banner
``"🗜️ Context too large (~156k tokens) — compressing..."`` must not be
sent into end-user chat threads. The fix routes these emissions through a
platform-aware filter in ``AIAgent._emit_status``:

* CLI / admin / api surfaces still receive every status (existing behavior).
* Customer-facing platforms (whatsapp, slack, signal, telegram, discord)
  drop messages flagged ``customer_facing=False`` and, defensively, any
  message starting with the compression-banner glyph for un-migrated callers.
"""
from __future__ import annotations

import pytest

import run_agent


def _stub_agent(platform: str = "") -> tuple[object, list[tuple[str, str]]]:
    """Build a minimal AIAgent for _emit_status testing without full init."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = platform
    agent.log_prefix = ""
    captured: list[tuple[str, str]] = []
    agent.status_callback = lambda evt, msg: captured.append((evt, msg))
    agent._vprint = lambda *a, **k: None  # silence the CLI path
    return agent, captured


@pytest.mark.parametrize(
    "platform", ["whatsapp", "slack", "signal", "telegram", "discord"],
)
def test_compression_status_suppressed_on_customer_facing_platforms(platform):
    """Explicit customer_facing=False compression banner stays out of customer chats."""
    agent, captured = _stub_agent(platform)
    agent._emit_status(
        "🗜️ Context too large (~156,287 tokens) — compressing (1/3)...",
        customer_facing=False,
    )
    assert captured == [], f"compression status leaked to {platform}: {captured!r}"


@pytest.mark.parametrize("platform", ["whatsapp", "slack"])
def test_glyph_fallback_drops_unmigrated_callers(platform):
    """Defensive glyph detection drops 🗜 banners even without the kwarg."""
    agent, captured = _stub_agent(platform)
    agent._emit_status("🗜️ Compressed 120 → 7 messages, retrying...")
    assert captured == [], f"glyph-fallback failed on {platform}: {captured!r}"


def test_customer_facing_default_lets_normal_messages_through_on_wa():
    """Real customer-facing messages (no glyph) still reach the gateway."""
    agent, captured = _stub_agent("whatsapp")
    agent._emit_status("Müberra Hanım, kalite skoru hazır")
    assert captured == [("lifecycle", "Müberra Hanım, kalite skoru hazır")]


def test_explicit_customer_facing_false_drops_message_without_glyph():
    """customer_facing=False suppresses even messages without a glyph prefix."""
    agent, captured = _stub_agent("whatsapp")
    agent._emit_status("any internal-only banner", customer_facing=False)
    assert captured == []


def test_cli_platform_receives_compression_status():
    """CLI/admin surface keeps full diagnostic visibility — no suppression."""
    agent, captured = _stub_agent("cli")
    agent._emit_status("🗜️ Context too large (~156k)", customer_facing=False)
    assert captured == [("lifecycle", "🗜️ Context too large (~156k)")]


def test_non_customer_facing_platform_passes_through():
    """Platforms outside the customer-facing set keep visibility."""
    agent, captured = _stub_agent("api_server")
    agent._emit_status("🗜️ Compressed", customer_facing=False)
    assert captured == [("lifecycle", "🗜️ Compressed")]


def test_empty_platform_treated_as_non_customer_facing():
    """A blank ``platform`` attribute is admin-equivalent."""
    agent, captured = _stub_agent("")
    agent._emit_status("🗜️ Context too large", customer_facing=False)
    assert captured == [("lifecycle", "🗜️ Context too large")]


def test_status_callback_exception_is_swallowed():
    """An exception inside ``status_callback`` must not escape ``_emit_status``."""
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = "whatsapp"
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None

    def _explode(evt, msg):
        raise RuntimeError("boom")

    agent.status_callback = _explode
    agent._emit_status("Normal customer message")
