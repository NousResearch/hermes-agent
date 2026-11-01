"""Priority aliases flow through the actual kanban parser and task storage."""
import argparse

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def parser():
    root = argparse.ArgumentParser()
    kc.build_parser(root.add_subparsers(dest="command"))
    return root


@pytest.mark.parametrize("action,required", [
    ("create", ["example"]),
    ("swarm", ["example", "--worker", "a:task", "--verifier", "b", "--synthesizer", "a"]),
    ("edit", ["t_example"]),
])
@pytest.mark.parametrize("raw,expected", [("high", 3), ("H", 3), ("Medium", 2), ("m", 2), ("LOW", 1), ("l", 1), ("0", 0), ("-7", -7), ("42", 42)])
def test_aliases_and_integer_compatibility(action, required, raw, expected):
    assert parser().parse_args(["kanban", action, *required, "--priority", raw]).priority == expected


@pytest.mark.parametrize("raw", ["urgent", "1.5", ""])
def test_invalid_priority_reports_accepted_forms(raw, capsys):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["kanban", "create", "example", "--priority", raw])
    assert exc.value.code == 2
    assert "expected an integer or high/h, medium/m, low/l" in capsys.readouterr().err


def test_alias_is_persisted_by_slash_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    result = kc.run_slash("create 'alias task' --priority HIGH")
    assert "error" not in result.lower()
    with kbc.connect() as conn:
        row = conn.execute("SELECT priority FROM tasks WHERE title = ?", ("alias task",)).fetchone()
    assert row[0] == 3
