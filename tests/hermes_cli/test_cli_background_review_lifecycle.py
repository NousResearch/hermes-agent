"""CLI lifecycle coverage for automatic background reviews."""

from __future__ import annotations

import types

import pytest

from hermes_cli import mcp_startup


@pytest.mark.parametrize(
    ("single_query", "expected_skip"),
    [(True, True), (False, False)],
)
def test_agent_background_review_matches_cli_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    single_query: bool,
    expected_skip: bool,
) -> None:
    """One-shot agents skip daemon review work; interactive agents retain it."""
    import cli as cli_mod

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._single_query_mode = single_query

    captured: dict[str, object] = {}

    def _fake_agent(*_args, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace()

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)

    assert cli._init_agent() is True
    assert captured["skip_background_review"] is expected_skip
