"""`hermes sessions` exit status tracks the outcome: a maintenance subcommand that reports a failure
exits non-zero, so cron/CI wrappers and `hermes sessions repair && ...` chains can tell."""

import sys

import pytest


def _run_cli(monkeypatch, argv_tail):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    try:
        main_mod.main()
    except SystemExit as exc:
        return exc.code or 0
    return 0


def test_sessions_repair_that_fails_exits_nonzero(monkeypatch, capsys):
    import hermes_state
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    db_path.write_bytes(b"not a sqlite database" * 200)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)

    code = _run_cli(monkeypatch, ["repair"])

    assert "Repair failed" in capsys.readouterr().out
    assert code != 0


@pytest.mark.parametrize("refused, accepted", [
    (["archive"], ["archive", "--source", "cli", "--yes"]),
    (["prune", "--never-active", "--older-than", "not-a-duration"],
     ["prune", "--never-active", "--older-than", "2d", "--yes"]),
])
def test_refused_bulk_action_exits_nonzero_where_the_valid_form_exits_zero(monkeypatch, refused, accepted):
    from hermes_state import SessionDB

    SessionDB().close()

    assert _run_cli(monkeypatch, accepted) == 0
    assert _run_cli(monkeypatch, refused) != 0
