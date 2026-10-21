"""Regression: whitespace-only reasoning must not render a reasoning box (#empty-box).

Some providers return ``reasoning = "\\n\\n"`` — a whitespace-only payload with no
thought content. Every render site treated that as real reasoning and drew an empty
"Reasoning" box / panel, which reads to the user as a broken feature.

Covers the three sites that gate on reasoning text:
  * ``CLIStreamMixin._stream_reasoning_delta``  (live streaming box, classic layout)
  * ``CLIChatTurnMixin._chat_print_reasoning_box`` (post-turn box, classic + modern layout)
  * ``GatewayTurnPersistenceMixin._hmwa_prepend_reasoning`` (messaging platforms)
"""
from __future__ import annotations

import types

import pytest


def _stream_cli():
    """A bare stand-in carrying only the attributes `_stream_reasoning_delta` touches."""
    return types.SimpleNamespace(
        show_reasoning=True,
        streaming_enabled=True,
        verbose=False,
        _stream_box_opened=False,
        _reasoning_box_opened=False,
        _reasoning_shown_this_turn=False,
        _reasoning_buf="",
        _scrollback_box_width=lambda: 78,
    )


def _collect_cprint(monkeypatch, calls: list[str]) -> None:
    """The mixins do ``from cli import _cprint`` at call time; swap it on the cli module."""
    import cli

    monkeypatch.setattr(cli, "_cprint", lambda text="": calls.append(str(text)))


class TestStreamReasoningDelta:
    """Live streaming box: classic and modern layout both key off the same guard."""

    @pytest.mark.parametrize("payload", ["\n\n", "", "   \n\t  \n", "\r\n"])
    def test_whitespace_only_delta_opens_no_box(self, monkeypatch, payload):
        from hermes_cli.cli_stream_mixin import CLIStreamMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = _stream_cli()

        CLIStreamMixin._stream_reasoning_delta(cli, payload)

        assert calls == [], f"whitespace-only delta {payload!r} drew output"
        assert cli._reasoning_box_opened is False
        assert cli._reasoning_shown_this_turn is False

    def test_real_text_still_opens_the_box(self, monkeypatch):
        from hermes_cli.cli_stream_mixin import CLIStreamMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = _stream_cli()

        CLIStreamMixin._stream_reasoning_delta(cli, "Because air scatters blue.")

        assert any("┌" in line for line in calls)
        assert cli._reasoning_box_opened is True
        assert cli._reasoning_shown_this_turn is True

    def test_blank_delta_then_real_text_still_opens_the_box(self, monkeypatch):
        """A blank delta must not latch the box closed for the real text after it."""
        from hermes_cli.cli_stream_mixin import CLIStreamMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = _stream_cli()

        CLIStreamMixin._stream_reasoning_delta(cli, "\n\n")
        CLIStreamMixin._stream_reasoning_delta(cli, "real text after a blank delta")

        assert any("┌" in line for line in calls)

    def test_second_real_delta_adds_no_second_header(self, monkeypatch):
        from hermes_cli.cli_stream_mixin import CLIStreamMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = _stream_cli()

        CLIStreamMixin._stream_reasoning_delta(cli, "part one")
        CLIStreamMixin._stream_reasoning_delta(cli, "part two")

        assert sum("┌" in line for line in calls) == 1


class TestChatPrintReasoningBox:
    """Post-turn box. `_modern_layout` is exercised both ways so neither branch regresses."""

    def _print(self, monkeypatch, last_reasoning, *, modern: bool):
        from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = types.SimpleNamespace(
            show_reasoning=True,
            _reasoning_shown_this_turn=False,
            reasoning_full=False,
            _modern_layout=modern,
            _scrollback_box_width=lambda: 78,
        )
        turn = types.SimpleNamespace(result={"last_reasoning": last_reasoning})
        CLIChatTurnMixin._chat_print_reasoning_box(cli, turn)
        return calls

    @pytest.mark.parametrize("modern", [False, True], ids=["classic", "modern"])
    @pytest.mark.parametrize("payload", ["\n\n", "", None, "   \n\t \n"], ids=["lf-lf", "empty", "none", "mixed"])
    def test_whitespace_only_renders_nothing(self, monkeypatch, payload, modern):
        assert self._print(monkeypatch, payload, modern=modern) == []

    @pytest.mark.parametrize("modern", [False, True], ids=["classic", "modern"])
    def test_real_text_still_renders(self, monkeypatch, modern):
        calls = self._print(monkeypatch, "Real reasoning text.", modern=modern)
        assert calls, "real reasoning rendered nothing"
        assert any("Real reasoning text." in line for line in calls)

    def test_suppressed_when_streaming_already_showed_it(self, monkeypatch):
        from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

        calls: list[str] = []
        _collect_cprint(monkeypatch, calls)
        cli = types.SimpleNamespace(
            show_reasoning=True,
            _reasoning_shown_this_turn=True,  # streaming already drew it
            reasoning_full=False,
            _modern_layout=False,
            _scrollback_box_width=lambda: 78,
        )
        turn = types.SimpleNamespace(result={"last_reasoning": "Real reasoning text."})
        CLIChatTurnMixin._chat_print_reasoning_box(cli, turn)
        assert calls == []


class TestGatewayPrependReasoning:
    """Messaging platforms get the reasoning prepended as text, not a box."""

    @staticmethod
    def _runner():
        from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

        return types.SimpleNamespace(
            _show_reasoning=True,
            _REASONING_QUOTE_STYLES=GatewayTurnPersistenceMixin._REASONING_QUOTE_STYLES,
        )

    @staticmethod
    def _source():
        from gateway.run_turn_persistence import Platform

        for name in ("TELEGRAM", "SLACK", "DISCORD"):
            if hasattr(Platform, name):
                return types.SimpleNamespace(platform=getattr(Platform, name))
        raise AssertionError("no messaging Platform member found to exercise the prepend")

    def _prepend(self, monkeypatch, last_reasoning):
        import gateway.run as grun
        from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

        monkeypatch.setattr(grun, "_load_gateway_config", lambda: {"display": {"show_reasoning": True}})
        monkeypatch.setattr(grun, "_platform_config_key", lambda platform: "telegram")
        monkeypatch.setattr(grun, "_resolve_gateway_display_bool", lambda *a, **k: True)
        return GatewayTurnPersistenceMixin._hmwa_prepend_reasoning(
            self._runner(), {"last_reasoning": last_reasoning}, "FINAL RESPONSE", self._source(), False
        )

    @pytest.mark.parametrize("payload", ["\n\n", "", None, ["not", "a", "string"]], ids=["lf-lf", "empty", "none", "list"])
    def test_blank_reasoning_leaves_response_untouched(self, monkeypatch, payload):
        assert self._prepend(monkeypatch, payload) == "FINAL RESPONSE"

    def test_real_reasoning_is_prepended(self, monkeypatch):
        out = self._prepend(monkeypatch, "Reasoning with real words.")
        assert out != "FINAL RESPONSE"
        assert "real words" in out
        assert "FINAL RESPONSE" in out
