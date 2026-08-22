"""Regression for #71453: finite ``chat -q`` cannot deliver async results."""

from types import SimpleNamespace

import cli as cli_mod
from gateway.session_context import async_delivery_supported, reset_session_vars


def test_single_query_declares_stateless_channel_before_agent_turn(monkeypatch):
    """A detached delegate must fall back inline before the finite turn starts."""
    observed_delivery_capability: list[bool] = []

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_kw: None)
            self.session_id = "chat-q-session"
            self.agent = SimpleNamespace(
                session_id="chat-q-session",
                platform="cli",
            )

        def _claim_active_session(self, _surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            return None

        def chat(self, _query, images=None):
            observed_delivery_capability.append(async_delivery_supported())
            return "done"

        def _print_exit_summary(self, clear_screen=True):
            return None

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)

    reset_session_vars()
    try:
        assert async_delivery_supported() is True
        cli_mod.main(query="delegate this", quiet=False, toolsets="terminal")
    finally:
        reset_session_vars()

    assert observed_delivery_capability == [False]


def test_interactive_cli_keeps_async_delivery_capability(monkeypatch):
    """Long-lived CLI sessions still accept late delegation completions."""
    observed_delivery_capability: list[bool] = []

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.agent = None

        def run(self):
            observed_delivery_capability.append(async_delivery_supported())

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)

    reset_session_vars()
    try:
        cli_mod.main(query=None, quiet=False, toolsets="terminal")
    finally:
        reset_session_vars()

    assert observed_delivery_capability == [True]
