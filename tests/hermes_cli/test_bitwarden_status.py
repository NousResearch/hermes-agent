from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock


def _mock_bw(**overrides):
    """Build a MagicMock standing in for the bitwarden module.

    #70697: secrets_cli now lazy-imports bitwarden via ``_bw()`` instead of
    a module-level ``bw`` binding, so tests patch ``_bw`` rather than ``bw``.
    """
    mock = MagicMock()
    mock.find_bws.return_value = overrides.get("find_bws", Path("/tmp/bws"))
    mock._BWS_VERSION = overrides.get("_BWS_VERSION", "2.0.0")
    return mock


def test_broken_bitwarden_import_does_not_crash_cli_startup(monkeypatch, capsys):
    """#70697: a broken ``cryptography`` import must not crash CLI startup.

    The lazy ``_bw()`` helper defers the Bitwarden import until first call.
    Importing ``secrets_cli`` must succeed even when
    ``agent.secret_sources.bitwarden`` is unimportable; the error must be
    deferred to the first ``_bw()`` call (i.e. when a bitwarden subcommand
    actually runs).
    """
    import importlib

    # Poison the bitwarden module so import fails.
    monkeypatch.setitem(
        sys.modules, "agent.secret_sources.bitwarden", None
    )

    # Re-import secrets_cli — must not raise despite broken bitwarden.
    from hermes_cli import secrets_cli
    importlib.reload(secrets_cli)

    # _bw() must now raise when actually called, not at import time.
    raised = False
    try:
        secrets_cli._bw()
    except ImportError:
        raised = True
    assert raised, (
        "_bw() must raise ImportError when bitwarden is broken, "
        "but only when actually called — not at module import time"
    )


def test_register_cli_builds_secrets_parser_with_broken_bitwarden(monkeypatch):
    """#70697: ``secrets_cli.register_cli`` must wire the bitwarden
    subcommand tree without importing the bitwarden module.

    Reviewer follow-up (#74703): the unit-level check above only proves
    ``_bw()`` defers the import. This test drives the real registration
    path — the same ``register_cli`` call ``main()`` makes when it builds
    the ``hermes secrets`` parser — with the bitwarden module poisoned, and
    asserts the parser tree still constructs and parses normally.
    """
    import importlib

    # Poison the bitwarden module so any eager import fails loudly.
    monkeypatch.setitem(
        sys.modules, "agent.secret_sources.bitwarden", None
    )

    # Fresh module state for the lazy cache, mirroring a cold process.
    from hermes_cli import secrets_cli

    secrets_cli._bw_module = None
    importlib.reload(secrets_cli)

    # Build a minimal parent parser exactly like main() does for the
    # `hermes secrets` subcommand, then register the bitwarden tree on it.
    import argparse

    secrets_parser = argparse.ArgumentParser(prog="hermes secrets")
    bw_sub = secrets_parser.add_subparsers(dest="secrets_command")
    bitwarden_parser = bw_sub.add_parser(
        "bitwarden",
        aliases=["bw"],
        help="Bitwarden Secrets Manager integration",
    )
    secrets_cli.register_cli(bitwarden_parser)

    # The tree must parse a status invocation without touching bitwarden.
    args = secrets_parser.parse_args(["bitwarden", "status"])
    assert args.secrets_command == "bitwarden"
    assert args.secrets_bw_command == "status"

    # And calling _bw() afterwards must raise — import is still deferred.
    raised = False
    try:
        secrets_cli._bw()
    except ImportError:
        raised = True
    assert raised, "_bw() must still raise after parser registration"




def _bitwarden_config(*, enabled: bool = True, server_url: str = "") -> dict:
    return {
        "secrets": {
            "bitwarden": {
                "enabled": enabled,
                "access_token_env": "BWS_ACCESS_TOKEN",
                "project_id": "proj-123",
                "server_url": server_url,
                "cache_ttl_seconds": 300,
                "override_existing": True,
                "auto_install": True,
            }
        }
    }


def test_status_surfaces_failed_token_validation(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    seen = {}

    monkeypatch.setattr(
        secrets_cli,
        "load_config",
        lambda: _bitwarden_config(server_url="https://vault.bitwarden.eu"),
    )
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.invalid-token")
    monkeypatch.setattr(secrets_cli, "_bw", lambda: _mock_bw(find_bws=Path("/tmp/bws")))
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")

    def _fake_list_projects(binary, token, console, *, server_url=""):
        seen["binary"] = binary
        seen["token"] = token
        seen["server_url"] = server_url
        console.print("  bws project list failed: Doesn't contain a decryption key")
        console.print(
            "  This usually means the access token is wrong or revoked. "
            "Double-check it in the Bitwarden web app."
        )
        return None

    monkeypatch.setattr(secrets_cli, "_list_projects", _fake_list_projects)

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token in env" in out
    assert "Token validation" in out
    assert "failed" in out
    assert "Doesn't contain a decryption key" in out
    assert "wrong or revoked" in out
    assert seen == {
        "binary": Path("/tmp/bws"),
        "token": "0.invalid-token",
        "server_url": "https://vault.bitwarden.eu",
    }


def test_status_warns_when_token_does_not_look_like_bsm_token(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    monkeypatch.setattr(secrets_cli, "load_config", lambda: _bitwarden_config())
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "not-a-bitwarden-token")
    monkeypatch.setattr(secrets_cli, "_bw", lambda: _mock_bw(find_bws=Path("/tmp/bws")))
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws v2.1.0")
    monkeypatch.setattr(
        secrets_cli,
        "_list_projects",
        lambda binary, token, console, *, server_url="": [],
    )

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token validation" in out
    assert "passed" in out
    assert "doesn't start with '0.'" in out


def test_status_marks_validation_as_not_checked_without_bws_binary(monkeypatch, capsys):
    from hermes_cli import secrets_cli

    monkeypatch.setattr(secrets_cli, "load_config", lambda: _bitwarden_config())
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.token-present")
    monkeypatch.setattr(secrets_cli, "_bw", lambda: _mock_bw(find_bws=None))

    assert secrets_cli.cmd_status(Namespace()) == 0

    out = capsys.readouterr().out
    assert "Token validation" in out
    assert "not checked" in out
    assert "bws not installed" in out
