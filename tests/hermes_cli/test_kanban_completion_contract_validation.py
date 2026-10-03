"""Shared completion-contract validation across Kanban creation surfaces."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from collections.abc import Generator

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_pr_acceptance import validate_contract


@pytest.fixture
def kanban_home(monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    root = Path(tempfile.mkdtemp(prefix="hermes-contract-", dir="/var/tmp"))
    home = root / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: root)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    try:
        yield home
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("placeholder", ["OWNER/REPO", "owner/repo", "Owner/Repo", "OWNER/repo"])
def test_validate_contract_rejects_literal_owner_repo_placeholder(placeholder: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        validate_contract(placeholder)


@pytest.mark.parametrize("contract", ["qwickapps/aos", "https://github.com/qwickapps/aos/pull/659"])
def test_validate_contract_accepts_real_repo_and_exact_pr(contract: str) -> None:
    assert validate_contract(contract) == contract


def test_create_task_rejects_placeholder_before_persistence(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        before = conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="placeholder"):
            kb.create_task(conn, title="bad contract", completion_contract="OWNER/REPO")
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == before


def test_cli_create_rejects_placeholder_before_persistence(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import kanban_parser

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kanban_parser.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "create", "bad contract", "--assignee", "worker",
        "--completion-contract", "owner/repo",
    ])
    assert kanban_cli.kanban_command(args) != 0
    assert "placeholder" in capsys.readouterr().err
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_kanban_create_tool_rejects_placeholder_before_persistence(kanban_home: Path) -> None:
    from tools import kanban_tools

    result = json.loads(kanban_tools._handle_create({
        "title": "bad contract", "assignee": "worker", "completion_contract": "Owner/Repo",
    }))
    assert "placeholder" in json.dumps(result)
    with kbc.connect_closing() as conn:
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("placeholder", ["OWNER/REPO", "owner/repo", "Owner/Repo", "OWNER/repo"])
def test_rebind_contract_rejects_placeholder_before_mutation(kanban_home: Path, placeholder: str) -> None:
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="bound", completion_contract="qwickapps/aos")
        with pytest.raises(ValueError, match="placeholder"):
            kb.rebind_contract(conn, task_id, new_contract=placeholder, reason="bad placeholder")
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.completion_contract == "qwickapps/aos"
        assert not [event for event in kb.list_events(conn, task_id) if event.kind == "contract_rebound"]


def test_kanban_create_tool_accepts_real_repo_contract(kanban_home: Path) -> None:
    from tools import kanban_tools

    result = json.loads(kanban_tools._handle_create({
        "title": "real contract", "assignee": "worker", "completion_contract": "qwickapps/aos",
    }))
    assert result["ok"] is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["task_id"])
        assert task is not None
        assert task.completion_contract == "qwickapps/aos"
