from __future__ import annotations

from hermes_cli import kanban_delivery_acceptance as delivery


SOURCE = "github:acme/repo:issue:41:intake"
PR = "https://github.com/acme/repo/pull/270"
HEAD = "a" * 40


def _snapshot(*, merged=True, base="main", default="main", issue="CLOSED", linked=True):
    refs = [{"number": 41, "repository": {"nameWithOwner": "acme/repo"}}] if linked else []
    return {"data": {"repository": {
        "defaultBranchRef": {"name": default},
        "pullRequest": {
            "state": "MERGED" if merged else "OPEN",
            "mergedAt": "2026-09-24T17:00:00Z" if merged else None,
            "headRefOid": HEAD,
            "baseRefName": base,
            "body": "Closes #41" if linked else "Work",
            "closingIssuesReferences": {"nodes": refs, "pageInfo": {"hasNextPage": False}},
        },
        "issue": {"state": issue},
    }}}


def test_delivery_acceptance_requires_default_branch_merge_and_closed_linked_issue(monkeypatch):
    state = {"snapshot": _snapshot()}
    calls = []

    def fake_api(endpoint, *, query=None, paginate=False):
        calls.append((endpoint, query, paginate))
        assert endpoint == "graphql"
        assert not paginate
        return state["snapshot"]

    monkeypatch.setattr(delivery, "_api", fake_api)
    receipt = delivery.collect_delivery_acceptance(SOURCE, PR, expected_head_sha=HEAD)
    assert receipt["ok"] is True
    assert receipt["classification"] == "delivered"
    assert receipt["issue_number"] == 41
    assert receipt["default_branch"] == "main"
    assert receipt["head_sha"] == HEAD
    assert len(calls) == 1


def test_delivery_acceptance_rejects_open_pr_open_issue_and_nondefault_merge(monkeypatch):
    state = {"snapshot": None}
    monkeypatch.setattr(delivery, "_api", lambda *a, **k: state["snapshot"])

    for snapshot, classification in (
        (_snapshot(merged=False), "not_merged"),
        (_snapshot(issue="OPEN"), "issue_open"),
        (_snapshot(base="release/other"), "wrong_base"),
    ):
        state["snapshot"] = snapshot
        receipt = delivery.collect_delivery_acceptance(SOURCE, PR, expected_head_sha=HEAD)
        assert receipt["ok"] is False
        assert receipt["classification"] == classification


def test_delivery_acceptance_rejects_unlinked_or_mismatched_head(monkeypatch):
    state = {"snapshot": _snapshot(linked=False)}
    monkeypatch.setattr(delivery, "_api", lambda *a, **k: state["snapshot"])
    receipt = delivery.collect_delivery_acceptance(SOURCE, PR, expected_head_sha=HEAD)
    assert receipt["ok"] is False
    assert receipt["classification"] == "issue_unlinked"

    state["snapshot"] = _snapshot()
    receipt = delivery.collect_delivery_acceptance(SOURCE, PR, expected_head_sha="b" * 40)
    assert receipt["ok"] is False
    assert receipt["classification"] == "stale"


def test_delivery_acceptance_rejects_wrong_repository_and_malformed_source():
    receipt = delivery.collect_delivery_acceptance(
        SOURCE, "https://github.com/other/repo/pull/270", expected_head_sha=HEAD,
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "pr_mismatch"

    receipt = delivery.collect_delivery_acceptance(
        "github:acme/repo:issue:0:intake", PR, expected_head_sha=HEAD,
    )
    assert receipt["ok"] is False
    assert receipt["classification"] == "missing"


def test_source_issue_task_cannot_complete_before_delivery(monkeypatch):
    import json

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance_store as store
    from hermes_cli.kanban_db_connect import connect

    kb.init_db()
    pr_receipt = {"ok": True, "classification": "success", "head_sha": HEAD,
                  "pr_url": PR, "checks": [], "recovery": "retry"}
    delivery_receipt = {"ok": False, "classification": "not_merged", "head_sha": HEAD,
                        "pr_url": PR, "recovery": "wait for merge"}
    monkeypatch.setattr(store, "collect_acceptance", lambda *a, **k: pr_receipt)
    monkeypatch.setattr(store, "collect_delivery_acceptance", lambda *a, **k: delivery_receipt,
                        raising=False)

    with connect() as conn:
        task_id = kb.create_task(conn, title="Issue delivery", completion_contract="acme/repo",
                                 idempotency_key=SOURCE)
        ok = kb.complete_task(conn, task_id, result="PR opened",
                              metadata={"published_pr": PR})
        assert ok is False
        assert kb.get_task(conn, task_id).status != "done"
        receipts = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='delivery_acceptance'", (task_id,))]
        assert receipts[-1]["classification"] == "not_merged"


def test_source_issue_task_completes_only_after_delivery_receipt(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance_store as store
    from hermes_cli.kanban_db_connect import connect

    kb.init_db()
    pr_receipt = {"ok": True, "classification": "success", "head_sha": HEAD,
                  "pr_url": PR, "checks": [], "recovery": "retry"}
    delivery_receipt = {"ok": True, "classification": "delivered", "head_sha": HEAD,
                        "pr_url": PR, "recovery": "verified"}
    monkeypatch.setattr(store, "collect_acceptance", lambda *a, **k: pr_receipt)
    monkeypatch.setattr(store, "collect_delivery_acceptance", lambda *a, **k: delivery_receipt,
                        raising=False)

    with connect() as conn:
        task_id = kb.create_task(conn, title="Issue delivery", completion_contract="acme/repo",
                                 idempotency_key=SOURCE)
        assert kb.complete_task(conn, task_id, result="Delivered",
                                metadata={"published_pr": PR}) is True
        assert kb.get_task(conn, task_id).status == "done"


def test_local_only_intake_does_not_require_or_query_delivery(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance_store as store
    from hermes_cli.kanban_db_connect import connect

    kb.init_db()
    called = []
    monkeypatch.setattr(store, "collect_delivery_acceptance", lambda *a, **k: called.append(a),
                        raising=False)
    with connect() as conn:
        task_id = kb.create_task(conn, title="Read-only analysis", completion_contract="local-only",
                                 idempotency_key=SOURCE)
        assert kb.complete_task(conn, task_id, result="Analysis complete") is True
    assert called == []


def test_coordinator_local_gate_receipt_is_passed_only_for_issue_root(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance_store as store
    from hermes_cli.kanban_db_connect import connect

    kb.init_db()
    local_gate = {"head_sha": HEAD, "result": "passed"}
    captured = {}
    pr_receipt = {"ok": True, "classification": "success", "head_sha": HEAD,
                  "pr_url": PR, "checks": [], "recovery": "retry"}
    delivery_receipt = {"ok": True, "classification": "delivered", "head_sha": HEAD,
                        "pr_url": PR, "recovery": "verified"}

    def fake_acceptance(contract, published_pr, **kwargs):
        captured.update(kwargs)
        return pr_receipt

    monkeypatch.setattr(store, "collect_acceptance", fake_acceptance)
    monkeypatch.setattr(store, "collect_delivery_acceptance", lambda *a, **k: delivery_receipt,
                        raising=False)
    with connect() as conn:
        task_id = kb.create_task(
            conn, title="Issue root", assignee="meetitcoordinator",
            completion_contract="acme/repo", idempotency_key=SOURCE,
        )
        assert kb.complete_task(conn, task_id, result="Delivered",
                                metadata={"published_pr": PR, "local_gate": local_gate})
    assert captured == {"local_gate": local_gate, "coordinator": "meetitcoordinator"}


def test_malformed_github_issue_key_fails_delivery_closed(monkeypatch):
    import json

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_pr_acceptance_store as store
    from hermes_cli.kanban_db_connect import connect

    kb.init_db()
    pr_receipt = {"ok": True, "classification": "success", "head_sha": HEAD,
                  "pr_url": PR, "checks": [], "recovery": "retry"}
    monkeypatch.setattr(store, "collect_acceptance", lambda *a, **k: pr_receipt)
    with connect() as conn:
        task_id = kb.create_task(
            conn, title="Malformed source key", completion_contract="acme/repo",
            idempotency_key="github:acme/repo:issue:000:intake",
        )
        assert kb.complete_task(conn, task_id, result="Done", metadata={"published_pr": PR}) is False
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='delivery_acceptance'", (task_id,),
        ).fetchone()
        assert json.loads(event[0])["classification"] == "missing"
