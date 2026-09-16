"""Persistence and validation for task execution contracts."""
from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def contract_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb._INITIALIZED_PATHS.clear()
    return tmp_path / "kanban.db"


def test_create_task_persists_normalized_execution_contract(contract_board):
    declared = {
        "kind": "argocd",
        "targets": [{
            "server": "argocd-applications-diiastage-3dc.diia.digital",
            "application": "example-app",
            "operation": "sync",
        }],
    }
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Deploy example", execution_contract=declared)
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.execution_contract == declared


@pytest.mark.parametrize("contract", [
    {},
    {"kind": "argocd", "targets": []},
    {"kind": "argocd", "targets": [{"server": "stage", "application": "app", "operation": "actions"}]},
    {"kind": "argocd", "targets": [{"server": "stage", "application": "app", "operation": "delete"}]},
])
def test_rejects_ambiguous_execution_contract(contract_board, contract):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="execution_contract"):
            kb.create_task(conn, title="Deploy example", execution_contract=contract)
