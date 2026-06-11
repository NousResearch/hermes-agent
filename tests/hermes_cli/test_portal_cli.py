"""Tests for `hermes portal` dispatch.

`hermes portal` (no subcommand) is the human-readable alias for the Nous Portal
one-shot onboarding (`hermes auth add nous --type oauth` / `hermes setup
--portal`). The prior status default moved to `hermes portal info`, with
`status` retained as a back-compat alias.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli import portal_cli


def _args(portal_command):
    return SimpleNamespace(portal_command=portal_command)


@pytest.mark.parametrize("sub", [None, "", "login"])
def test_bare_portal_and_login_run_one_shot(monkeypatch, sub):
    """`hermes portal`, `hermes portal login` -> one-shot onboarding."""
    calls = {"login": 0, "status": 0}

    def fake_one_shot(config):
        calls["login"] += 1

    def fake_status(args):
        calls["status"] += 1
        return 0

    monkeypatch.setattr(
        "hermes_cli.setup._run_portal_one_shot", fake_one_shot
    )
    monkeypatch.setattr(portal_cli, "_cmd_status", fake_status)
    monkeypatch.setattr(portal_cli, "load_config", lambda: {})

    rc = portal_cli.portal_command(_args(sub))

    assert rc == 0
    assert calls["login"] == 1
    assert calls["status"] == 0


@pytest.mark.parametrize("sub", ["info", "status"])
def test_info_and_status_alias_run_status(monkeypatch, sub):
    """`hermes portal info` and the `status` back-compat alias -> status."""
    calls = {"login": 0, "status": 0}

    monkeypatch.setattr(
        "hermes_cli.setup._run_portal_one_shot",
        lambda config: calls.__setitem__("login", calls["login"] + 1),
    )

    def fake_status(args):
        calls["status"] += 1
        return 0

    monkeypatch.setattr(portal_cli, "_cmd_status", fake_status)

    rc = portal_cli.portal_command(_args(sub))

    assert rc == 0
    assert calls["status"] == 1
    assert calls["login"] == 0


def test_open_and_tools_dispatch(monkeypatch):
    seen = []
    monkeypatch.setattr(portal_cli, "_cmd_open", lambda a: seen.append("open") or 0)
    monkeypatch.setattr(portal_cli, "_cmd_tools", lambda a: seen.append("tools") or 0)

    assert portal_cli.portal_command(_args("open")) == 0
    assert portal_cli.portal_command(_args("tools")) == 0
    assert seen == ["open", "tools"]


def test_unknown_subcommand_returns_error(capsys):
    rc = portal_cli.portal_command(_args("bogus"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Unknown portal subcommand" in err


def test_login_cancelled_returns_one(monkeypatch):
    def boom(config):
        raise KeyboardInterrupt

    monkeypatch.setattr("hermes_cli.setup._run_portal_one_shot", boom)
    monkeypatch.setattr(portal_cli, "load_config", lambda: {})

    rc = portal_cli.portal_command(_args(None))
    assert rc == 1


def test_parser_registers_subcommands():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    portal_cli.add_parser(subparsers)

    # Bare `portal` resolves to portal_command with no portal_command set.
    ns = parser.parse_args(["portal"])
    assert ns.func is portal_cli.portal_command
    assert getattr(ns, "portal_command", None) in (None, "")

    # All documented subcommands parse.
    for sub in ("login", "info", "status", "open", "tools"):
        ns = parser.parse_args(["portal", sub])
        assert ns.portal_command == sub


def test_one_shot_delegates_to_model_flow_nous(monkeypatch):
    """`hermes portal` must run the quick-setup Nous flow (login + MODEL PICK +
    provider + Tool Gateway), i.e. delegate to `_model_flow_nous` — not the
    lighter auth-only path that skipped model selection.
    """
    import hermes_cli.setup as setup_mod

    calls = {"model_flow": 0}

    def fake_model_flow(config):
        calls["model_flow"] += 1

    # _model_flow_nous lives in hermes_cli.main and is imported lazily inside
    # _run_portal_one_shot, so patch it at the source module.
    monkeypatch.setattr("hermes_cli.main._model_flow_nous", fake_model_flow)
    # Keep the disk re-sync a no-op so the test never touches real config.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    setup_mod._run_portal_one_shot({})

    assert calls["model_flow"] == 1, (
        "`hermes portal` must route through _model_flow_nous so the model "
        "picker runs every time (matching quick setup)."
    )


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError, SystemExit])
def test_one_shot_swallows_cancel_and_systemexit(monkeypatch, exc):
    """A cancel/abort from the delegated Nous flow must NOT escape and kill the
    CLI. `_login_nous` raises SystemExit(130)/(1) on cancel/failure, and the
    expired-session re-login path inside `_model_flow_nous` only catches
    Exception — so SystemExit could otherwise propagate out. The portal handler
    must treat KeyboardInterrupt/EOFError/SystemExit as a graceful cancel.
    """
    import hermes_cli.setup as setup_mod

    def boom(config):
        raise exc

    monkeypatch.setattr("hermes_cli.main._model_flow_nous", boom)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    # Must return normally (None), not propagate the exception.
    assert setup_mod._run_portal_one_shot({}) is None
