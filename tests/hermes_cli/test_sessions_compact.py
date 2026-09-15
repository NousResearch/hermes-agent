"""CLI contracts for selective in-place session compaction."""

import argparse
import sys

import pytest

from hermes_cli.subcommands.sessions import build_sessions_parser


def test_compact_parser_requires_one_retention_boundary():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_sessions_parser(subparsers, cmd_sessions=lambda *_args, **_kwargs: 0)

    args = parser.parse_args(["sessions", "compact", "session-prefix", "--keep-last", "3"])
    assert args.sessions_action == "compact"
    assert args.keep_last == 3
    assert args.keep_until is None

    with pytest.raises(SystemExit):
        parser.parse_args(["sessions", "compact", "session-prefix"])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "sessions", "compact", "session-prefix", "--keep-last", "3", "--keep-until", "2026-01-01"
        ])


def test_compact_cli_resolves_prefix_and_passes_iso_boundary(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    seen = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            seen["input"] = session_id
            return "full-session-id"

        def compact_session(self, session_id, **kwargs):
            seen["call"] = (session_id, kwargs)
            return {"compacted": 2, "remaining": 3}

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "compact", "session-prefix", "--keep-until", "2026-01-01T00:00:00Z"],
    )

    main_mod.main()

    assert seen["input"] == "session-prefix"
    assert seen["call"][0] == "full-session-id"
    assert seen["call"][1]["keep_last"] is None
    assert seen["call"][1]["dry_run"] is False
    assert seen["call"][1]["keep_until"] == pytest.approx(1767225600.0)
    assert seen["closed"] is True
    assert "Compacted 2 message(s)" in capsys.readouterr().out
