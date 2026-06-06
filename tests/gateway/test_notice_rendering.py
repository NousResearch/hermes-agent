"""Unit tests for messaging-gateway credit-notice rendering.

Covers render_notice_line — the pure helper that turns an AgentNotice into the
single plaintext line pushed standalone over a messaging platform (no status
bar, unlike the TUI). Behavior contracts, not data snapshots.
"""
from agent.credits_tracker import AgentNotice
from gateway.run import render_notice_line


class TestRenderNoticeLine:
    def test_info_gets_bullet_glyph(self):
        line = render_notice_line(AgentNotice(text="Credits 50% used", level="info"))
        assert "Credits 50% used" in line
        assert line.startswith("•")

    def test_warn_gets_warning_glyph(self):
        line = render_notice_line(AgentNotice(text="Credits 90% used", level="warn"))
        assert line.startswith("⚠")

    def test_error_gets_block_glyph(self):
        line = render_notice_line(AgentNotice(text="Credits depleted", level="error"))
        assert line.startswith("⛔")

    def test_success_gets_check_glyph(self):
        line = render_notice_line(
            AgentNotice(text="✓ Credit access restored", level="success")
        )
        assert line.startswith("✓")

    def test_unknown_level_degrades_to_bare_text(self):
        # An unrecognized level must not raise or prepend a stray glyph —
        # it falls back to the text so the notice still reaches the user.
        line = render_notice_line(AgentNotice(text="something", level="weird"))
        assert line == "something"

    def test_empty_text_returns_empty_string(self):
        # Empty/whitespace text → empty line → the callback suppresses the push
        # (no glyph-only message). Fail-soft, not raise.
        assert render_notice_line(AgentNotice(text="", level="warn")) == ""
        assert render_notice_line(AgentNotice(text="   ", level="warn")) == ""

    def test_text_is_stripped(self):
        line = render_notice_line(AgentNotice(text="  padded  ", level="info"))
        assert line == "• padded"

    def test_malformed_notice_does_not_raise(self):
        # Duck-typed: a stand-in lacking the expected attrs degrades to "".
        class _Bare:
            pass

        assert render_notice_line(_Bare()) == ""


# ── Delivery seam: a rendered notice line goes out via _deliver_platform_notice ──

import threading
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_source(platform_value="telegram", chat_id="555", user_id="u1"):
    src = MagicMock()
    plat = MagicMock()
    plat.value = platform_value
    src.platform = plat
    src.chat_id = chat_id
    src.user_id = user_id
    return src


def _make_runner_with_adapter(source, adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {source.platform: adapter}
    runner.config = MagicMock()
    runner.config.get_notice_delivery = MagicMock(return_value="public")
    runner._thread_metadata_for_source = MagicMock(return_value={"thread": "t"})
    return runner


class TestDeliverNoticeLine:
    """The seam between render_notice_line and the platform adapter.

    Proves a rendered credit-notice line reaches adapter.send (public) /
    send_private_notice (private) through the shared _deliver_platform_notice
    rail — the path the gateway notice_callback schedules onto the loop.
    """

    @pytest.mark.asyncio
    async def test_public_delivery_sends_rendered_line(self):
        source = _make_source()
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        runner = _make_runner_with_adapter(source, adapter)

        line = render_notice_line(
            AgentNotice(text="Credits 90% used", level="warn")
        )
        await runner._deliver_platform_notice(source, line)

        adapter.send.assert_awaited_once()
        args, kwargs = adapter.send.call_args
        assert args[0] == "555"
        assert "⚠ Credits 90% used" in args[1]

    @pytest.mark.asyncio
    async def test_private_delivery_prefers_private_notice(self):
        source = _make_source()
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        adapter.send_private_notice = AsyncMock(return_value=MagicMock(success=True))
        runner = _make_runner_with_adapter(source, adapter)
        runner.config.get_notice_delivery = MagicMock(return_value="private")

        line = render_notice_line(
            AgentNotice(text="✓ Credit access restored", level="success")
        )
        await runner._deliver_platform_notice(source, line)

        adapter.send_private_notice.assert_awaited_once()
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_adapter_is_a_noop(self):
        source = _make_source()
        runner = object.__new__(__import__("gateway.run", fromlist=["GatewayRunner"]).GatewayRunner)
        runner.adapters = {}
        # Must not raise when the platform has no registered adapter.
        await runner._deliver_platform_notice(source, "• anything")

