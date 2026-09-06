"""Strict owner lifecycle evidence on the mounted authenticated host."""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tests.plugins.test_kanban_owner_reconciliation import host


def test_snapshot_keeps_deleted_parent_and_retirement_after_restore(host):
    client, _ = host
    conn = kbc.connect()
    parent = kb.create_task(conn, title="Root", assignee="worker", created_by="operator")
    child = kb.create_task(conn, title="Child", assignee="worker", created_by="auto-decomposer")
    # Authoritative creation evidence, not dependency edges, defines containment.
    conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='created'",
                 ('{"by":"auto-decomposer","from_decompose_of":"' + parent + '"}', child))
    conn.execute("UPDATE owner_events SET payload=? WHERE task_id=? AND kind='created'",
                 ('{"by":"auto-decomposer","from_decompose_of":"' + parent + '"}', child))
    conn.commit()
    headers = {"Authorization": "Bearer this-host-fixture"}
    kb.archive_task(conn, parent)
    response = client.patch(f"/api/plugins/kanban/tasks/{parent}",
                            json={"status": "ready"},
                            headers={"X-Hermes-Session-Token": "this-host-fixture"})
    assert response.status_code == 200, response.text
    snapshot = client.get("/api/plugins/kanban/owner-snapshot", headers=headers)
    assert snapshot.status_code == 200, snapshot.text
    data = snapshot.json()
    assert data["contract_version"] == 2
    assert parent in data["retired_task_ids"]
    assert not next(r for r in data["receipts"] if r["task"]["id"] == parent)["archived"]
    kb.delete_task(conn, parent)
    data = client.get("/api/plugins/kanban/owner-snapshot", headers=headers).json()
    receipts = {r["task"]["id"]: r for r in data["receipts"]}
    assert receipts[parent]["archived"] is True
    assert receipts[child]["created_event"]["payload"]["from_decompose_of"] == parent
    conn.close()


@pytest.mark.parametrize("payload,expected", [({}, {}), ({"body": "private"}, {}),
    ({"from_decompose_of": " parent "}, None), ({"from_decompose_of": 12}, None),
    ({"by": " operator "}, None), (None, None)])
def test_creation_redaction_distinguishes_empty_from_invalid(payload, expected):
    assert kb._owner_contract_payload("created", payload) == expected
