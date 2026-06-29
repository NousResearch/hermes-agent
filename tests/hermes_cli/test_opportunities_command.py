"""Tests for the shared /opportunities command handler."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def opportunities_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import agent.opportunities as opportunities

    importlib.reload(hermes_constants)
    importlib.reload(opportunities)
    return opportunities


def test_list_empty_mentions_enable(opportunities_env):
    from hermes_cli.opportunities_cmd import handle_opportunities_command

    out = handle_opportunities_command("").text

    assert "No pending opportunities" in out
    assert "/opportunities enable" in out


def test_seed_lists_and_accept_returns_agent_seed(opportunities_env):
    from hermes_cli.opportunities_cmd import handle_opportunities_command

    seeded = handle_opportunities_command("seed")
    listed = handle_opportunities_command("")
    accepted = handle_opportunities_command("accept 1")

    assert "Added 5 starter" in seeded.text
    assert "Learn a repeated workflow" in listed.text
    assert "Accepted opportunity" in accepted.text
    assert accepted.agent_seed
    assert "skill_manage" in accepted.agent_seed


def test_dismiss_removes_pending(opportunities_env):
    from hermes_cli.opportunities_cmd import handle_opportunities_command

    handle_opportunities_command("seed")
    out = handle_opportunities_command("dismiss 1").text

    assert "Dismissed" in out
    assert "Learn a repeated workflow" not in handle_opportunities_command("").text


def test_enable_disable_persist_config(monkeypatch, opportunities_env):
    from hermes_cli import opportunities_cmd

    calls = []
    monkeypatch.setattr(
        opportunities_cmd,
        "_save_proactive_enabled",
        lambda enabled: calls.append(enabled) or True,
    )

    enabled = opportunities_cmd.handle_opportunities_command("enable").text
    disabled = opportunities_cmd.handle_opportunities_command("disable").text

    assert calls == [True, False]
    assert "enabled" in enabled
    assert "disabled" in disabled
