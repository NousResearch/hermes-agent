"""Contrato do aviso heurístico: aresta invertida acusa, ordem correta cala."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


@pytest.mark.parametrize("review,execution", [
    ({"title": "Revisar a matriz E2E"}, {"title": "Executar a matriz E2E"}),
    ({"title": "Verification of certificates"}, {"title": "Build certificate inventory"}),
    ({"title": "Gate", "body": "Emitir parecer independente."},
     {"title": "Delivery", "body": "Construir o inventário."}),
    ({"title": "Gate", "assignee": "revisor"},
     {"title": "Delivery", "assignee": "executor"}),
])
def test_inverted_edge_warns_and_corrected_edge_clears(tmp_path, monkeypatch, capsys, review, execution):
    from argparse import Namespace
    from hermes_cli.kanban import _cmd_diagnostics

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, **review)
        child = kb.create_task(conn, parents=[parent], **execution)
        before = list(conn.iterdump())
        assert _cmd_diagnostics(Namespace(task=None, json=True, severity=None)) == 0
        payload = json.loads(capsys.readouterr().out)
        warnings = [d for row in payload for d in row["diagnostics"]
                    if d["kind"] == "suspected_inverted_dependency"]
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning["severity"] == "warning"
        assert warning["data"]["parent_id"] == parent
        assert warning["data"]["child_id"] == child
        assert parent in warning["detail"] and child in warning["detail"]
        assert warning["data"]["parent_signal"] and warning["data"]["child_signal"]
        assert list(conn.iterdump()) == before
        with capsys.disabled():
            print(f"INVERTED: {parent} -> {child}: {warning['kind']} (warning)")

        assert kb.unlink_tasks(conn, parent, child)
        kb.link_tasks(conn, child, parent)
        before = list(conn.iterdump())
        assert _cmd_diagnostics(Namespace(task=None, json=True, severity=None)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert not [d for row in payload for d in row["diagnostics"]
                    if d["kind"] == "suspected_inverted_dependency"]
        assert list(conn.iterdump()) == before
        with capsys.disabled():
            print(f"CORRECTED: {child} -> {parent}: no inverted dependency warning")


@pytest.mark.parametrize("task,graph", [
    ({"id": "c", "title": "Execute checks"}, None),
    ({"id": "c", "title": "Preview runtime"},
     {"parents": [{"id": "p", "title": "Review result"}]}),
    ({"id": "c", "title": "Build inventory"},
     {"parents": [{"id": "p", "title": "Audit and implement changes"}]}),
    ({"id": "c", "title": "Review output", "body": "Run tests too"},
     {"parents": [{"id": "p", "title": "Review plan"}]}),
    ({"id": "c", "title": "Build inventory"}, {"parents": []}),
])
def test_unproven_roles_do_not_warn(task, graph):
    assert not [d for d in kd.compute_task_diagnostics(task, [], [], graph=graph)
                if d.kind == "suspected_inverted_dependency"]
