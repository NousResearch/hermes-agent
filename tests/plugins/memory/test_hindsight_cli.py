"""Tests for the hindsight plugin CLI registration (hermes hindsight ...)."""

import argparse

import pytest

from plugins.memory.hindsight import cli as hindsight_cli


def _build_parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    hindsight_parser = subparsers.add_parser("hindsight")
    hindsight_cli.register_cli(hindsight_parser)
    return parser


def test_parses_import_sessions_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "hindsight", "import-sessions",
            "--dry-run", "--skip-existing", "-y",
            "--since", "2026-01-01", "--until", "2026-02-01",
            "--days", "30", "--limit", "10", "--retain-timeout", "120",
            "--doc-id-prefix", "hist-", "--extra-tags", "a,b",
            "--bank-id", "bank",
        ]
    )

    assert args.hindsight_command == "import-sessions"
    assert args.dry_run and args.skip_existing and args.yes
    assert args.since == "2026-01-01"
    assert args.until == "2026-02-01"
    assert args.days == 30
    assert args.limit == 10
    assert args.retain_timeout == 120
    assert args.doc_id_prefix == "hist-"
    assert args.extra_tags == "a,b"
    assert args.bank_id == "bank"


def test_defaults():
    args = _build_parser().parse_args(["hindsight", "import-sessions"])

    assert not args.dry_run and not args.yes and not args.skip_existing
    assert args.since is None and args.until is None
    assert args.days is None and args.limit is None
    assert args.retain_timeout == 600
    assert args.doc_id_prefix == ""
    assert args.extra_tags == "hermes-backfill"
    assert args.bank_id is None


@pytest.mark.parametrize(
    "argv",
    [
        ["hindsight", "import-sessions", "--since", "01-01-2026"],
        ["hindsight", "import-sessions", "--until", "not-a-date"],
        ["hindsight", "import-sessions", "--days", "-3"],
        ["hindsight", "import-sessions", "--limit", "ten"],
    ],
)
def test_invalid_values_exit_with_usage_error(argv, capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2
    capsys.readouterr()  # swallow argparse usage output


def test_dispatch_routes_to_import_sessions_handler(monkeypatch):
    called = {}
    monkeypatch.setattr(
        "plugins.memory.hindsight.import_sessions.handle_import_sessions_command",
        lambda args: called.setdefault("args", args),
    )

    args = argparse.Namespace(hindsight_command="import-sessions")
    hindsight_cli.hindsight_command(args)

    assert called["args"] is args


def test_no_subcommand_prints_usage(capsys):
    hindsight_cli.hindsight_command(argparse.Namespace(hindsight_command=None))

    out = capsys.readouterr().out
    assert "import-sessions" in out


def test_discovery_registers_hindsight_cli(monkeypatch):
    """discover_plugin_cli_commands() picks up the real cli.py when hindsight is active."""
    import plugins.memory as pm

    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "hindsight")

    cmds = pm.discover_plugin_cli_commands()

    assert len(cmds) == 1
    assert cmds[0]["name"] == "hindsight"
    assert callable(cmds[0]["setup_fn"])
    assert cmds[0]["handler_fn"].__name__ == "hindsight_command"
