"""Tests for ``PluginContext.ask_user`` — the AskUserQuestion-style overlay
proxy added so plugins can drive the CLI's existing clarify-callback
machinery (arrow-key selection, timeout, plays nicely with the live
display).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _make_ctx(cli=None) -> PluginContext:
    """Build a real PluginContext with a synthetic manifest + manager."""
    mgr = PluginManager()
    mgr._cli_ref = cli
    manifest = PluginManifest(
        name="test-plugin",
        version="0.0.0",
        path=None,  # type: ignore[arg-type]
    )
    return PluginContext(manifest, mgr)


class TestAskUser:
    def test_proxies_to_clarify_callback(self):
        clarify = MagicMock(return_value="accept")
        cli = SimpleNamespace(_clarify_callback=clarify)
        ctx = _make_ctx(cli=cli)

        result = ctx.ask_user("Use rewrite?", ["accept", "reject"])

        assert result == "accept"
        clarify.assert_called_once_with("Use rewrite?", ["accept", "reject"])

    def test_returns_none_when_no_cli_ref(self):
        """Gateway / ACP / script contexts have no CLI — must return None."""
        ctx = _make_ctx(cli=None)
        assert ctx.ask_user("Q", ["a", "b"]) is None

    def test_returns_none_when_cli_has_no_clarify_callback(self):
        cli = SimpleNamespace()  # no _clarify_callback attr
        ctx = _make_ctx(cli=cli)
        assert ctx.ask_user("Q", ["a", "b"]) is None

    def test_returns_none_on_clarify_exception(self):
        """Plugin must not crash if the overlay raises (e.g. detached TTY)."""
        cli = SimpleNamespace(
            _clarify_callback=MagicMock(side_effect=RuntimeError("boom"))
        )
        ctx = _make_ctx(cli=cli)
        assert ctx.ask_user("Q", ["a", "b"]) is None

    def test_choices_passed_as_list_copy(self):
        """Mutating the caller's list after the call must not affect the cb."""
        seen = []
        clarify = MagicMock(side_effect=lambda q, c: seen.append(c) or "a")
        cli = SimpleNamespace(_clarify_callback=clarify)
        ctx = _make_ctx(cli=cli)

        choices = ["a", "b"]
        ctx.ask_user("Q", choices)
        choices.append("c")

        assert seen == [["a", "b"]]

    def test_timeout_path_returns_none(self):
        """Clarify returns None on timeout — propagate as-is."""
        cli = SimpleNamespace(_clarify_callback=MagicMock(return_value=None))
        ctx = _make_ctx(cli=cli)
        assert ctx.ask_user("Q", ["a", "b"]) is None
