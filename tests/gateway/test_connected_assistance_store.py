"""Durable connected assistance state against real profile-local SQLite."""

import sqlite3

import pytest


@pytest.fixture
def home(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    return profile


def evidence(source="gmail:account-a", record="message-1", excerpt="Due Friday"):
    return {"source_id": source, "record_id": record, "excerpt": excerpt,
            "url": "https://example.test/message-1", "data_use": {"no_training": True}}


def finding(**updates):
    row = {"kind": "deadline", "entity_key": "project-1", "title": "Review deadline",
           "summary": "Review is due", "why_now": "Friday is near",
           "suggested_action": "Review the draft", "evidence": [evidence()]}
    return row | updates


def scan(*findings, records=None, **updates):
    return {"findings": list(findings), "evidence": records or [], "errors": [],
            "connected": True} | updates


def test_runtime_publication_fence_rolls_back_scan_if_revoked_before_commit(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    checks = []

    def current():
        checks.append(True)
        if len(checks) == 2:
            raise PermissionError("profile retired while persisting")

    with pytest.raises(PermissionError, match="retired"):
        store.ingest_scan(scan(finding(), records=[evidence()]), check_current=current)
    assert store.list_findings() == []
    assert store.prior_evidence() == []


def test_round_trip_findings_survive_new_store_and_are_sqlite(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    created = store.ingest_scan(scan(finding()))
    assert len(created) == 1
    row = created[0]
    assert row["id"] and row["status"] == "open"
    assert row.items() >= finding().items()
    reopened = AssistanceStore(home)
    assert reopened.get_finding(row["id"]) == row
    assert reopened.list_findings() == [row]
    assert reopened.get_finding("missing") is None
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_evidence_not_model_wording_drives_changes_and_history(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    first = store.ingest_scan(scan(finding()))[0]
    reworded = finding(title="A different title", summary="New prose, same evidence")
    assert AssistanceStore(home).ingest_scan(scan(reworded)) == []
    assert store.get_finding(first["id"])["title"] == first["title"]
    updated = finding(evidence=[evidence(excerpt="Due Thursday")])
    changed = store.ingest_scan(scan(updated))
    assert len(changed) == 1 and changed[0]["id"] == first["id"]
    assert store.ingest_scan(scan(finding())) == []
    assert store.get_finding(first["id"])["evidence"] == updated["evidence"]
    assert len(store.list_findings()) == 1


@pytest.mark.parametrize("status", ["open", "dismissed", "resolved"])
def test_evidence_dedup_survives_entity_rewording_and_terminal_decisions(home, status):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    first = store.ingest_scan(scan(finding(entity_key="customer-123")))[0]
    if status != "open":
        store.transition(first["id"], status)
    assert AssistanceStore(home).ingest_scan(scan(finding(entity_key="customer 123"))) == []
    assert len(store.list_findings()) == 1
    assert store.get_finding(first["id"])["status"] == status
    assert len(store.ingest_scan(scan(finding(kind="direct_request", entity_key="customer 123")))) == 1
    assert len(store.ingest_scan(scan(finding(entity_key="customer 456",
                                            evidence=[evidence(record="message-2")])))) == 1


def test_account_identity_and_distinct_evidence_are_not_merged(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    rows = store.ingest_scan(scan(
        finding(), finding(evidence=[evidence(source="gmail:account-b")]),
        finding(entity_key="project-2", evidence=[evidence(record="message-2")]), finding(kind="unanswered")))
    assert len({row["id"] for row in rows}) == 4


def test_evidence_order_and_duplicates_do_not_create_changes(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    a, b = evidence(), evidence(source="slack:account-a", record="thread-2")
    store.ingest_scan(scan(finding(evidence=[a, b])))
    assert store.ingest_scan(scan(finding(evidence=[b, a, a]))) == []


@pytest.mark.parametrize("status", ["dismissed", "resolved"])
def test_terminal_decision_survives_restart_and_changed_evidence(home, status):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    row = store.ingest_scan(scan(finding()))[0]
    assert store.transition(row["id"], status)["status"] == status
    reopened = AssistanceStore(home)
    assert reopened.ingest_scan(scan(finding())) == []
    assert reopened.ingest_scan(scan(finding(evidence=[evidence(excerpt="Due now")]))) == []
    assert reopened.get_finding(row["id"])["status"] == status
    assert len(reopened.list_findings(status=status)) == 1
    assert reopened.list_findings(status="open") == []
    assert reopened.transition("missing", status) is None
    with pytest.raises(ValueError):
        reopened.transition(row["id"], "open")


def test_absence_or_source_errors_never_resolve(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    row = store.ingest_scan(scan(finding()))[0]
    assert store.ingest_scan(scan()) == []
    assert store.ingest_scan(scan(connected=False, errors=["read_failed"])) == []
    assert store.get_finding(row["id"])["status"] == "open"


def test_pause_is_durable_and_profiles_are_isolated(home, tmp_path, monkeypatch):
    from gateway.connected_assistance_store import AssistanceStore

    other = tmp_path / "other-profile"
    other.mkdir()
    a, b = AssistanceStore(home), AssistanceStore(other)
    assert not a.is_paused() and not b.is_paused()
    a.set_paused(True)
    row = a.ingest_scan(scan(finding()))[0]
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert AssistanceStore(home).is_paused()
    assert not b.is_paused() and b.list_findings() == []
    assert a.get_finding(row["id"])
    a.set_paused(False)
    assert not AssistanceStore(home).is_paused()
    with pytest.raises(ValueError):
        a.set_paused("false")


def test_prior_evidence_retains_restrictions_and_latest_account_record(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    record = evidence() | {"data": {"text": "Due Friday"}, "pointer": "/messages/0"}
    assert store.prior_evidence() == []
    store.ingest_scan(scan(records=[record, evidence(source="slack:work")]))
    assert record in AssistanceStore(home).prior_evidence()
    changed = record | {"excerpt": "Due Thursday", "data_use": ["no_external_llm"]}
    store.ingest_scan(scan(records=[changed]))
    records = AssistanceStore(home).prior_evidence()
    assert changed in records and record not in records
    assert len(records) == 2
    store.ingest_scan(scan(connected=False, errors=["offline"]))
    assert store.prior_evidence() == records


def test_prior_evidence_is_bounded_without_truncating_data_use(home):
    from gateway.connected_assistance_store import (
        AssistanceStore, MAX_EVIDENCE_BYTES, MAX_EVIDENCE_RECORDS,
    )

    store = AssistanceStore(home)
    for number in range(MAX_EVIDENCE_RECORDS + 3):
        store.ingest_scan(scan(records=[evidence(record=str(number))]))
    records = store.prior_evidence()
    assert len(records) == MAX_EVIDENCE_RECORDS
    assert {r["record_id"] for r in records}.isdisjoint({"0", "1", "2"})
    assert all(r["data_use"] == {"no_training": True} for r in records)
    oversized = evidence(record="large") | {"data_use": "x" * MAX_EVIDENCE_BYTES}
    with pytest.raises(ValueError, match="evidence"):
        store.ingest_scan(scan(records=[oversized]))
    assert store.prior_evidence() == records


@pytest.mark.platforms("posix")
def test_private_directory_database_and_sqlite_journal(home):
    from gateway.connected_assistance_store import AssistanceStore

    directory = home / "connected_assistance"
    directory.mkdir(mode=0o777)
    store = AssistanceStore(home)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600
    store.path.chmod(0o666)
    AssistanceStore(home)
    assert store.path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(store.path) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT INTO settings VALUES ('journal-test', 'private')")
        journal = store.path.with_name(store.path.name + "-journal")
        assert journal.stat().st_mode & 0o077 == 0
    assert not journal.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("target_kind", ["directory", "database", "journal", "hardlink"])
def test_refuses_linked_storage_without_touching_target(home, tmp_path, target_kind):
    import os
    from gateway.connected_assistance_store import AssistanceStore

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_text("do not touch")
    directory = home / "connected_assistance"
    if target_kind == "directory":
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir()
        target = directory / ("state.sqlite3-journal" if target_kind == "journal" else "state.sqlite3")
        if target_kind == "hardlink":
            os.link(secret, target)
        else:
            target.symlink_to(secret)
    before = secret.stat().st_mode
    with pytest.raises(OSError):
        AssistanceStore(home)
    assert secret.read_text() == "do not touch" and secret.stat().st_mode == before
    assert not (outside / "state.sqlite3").exists()


def test_missing_home_is_not_recreated_by_stale_store(home):
    from gateway.connected_assistance_store import AssistanceStore

    missing = home / "deleted-profile"
    with pytest.raises(FileNotFoundError):
        AssistanceStore(missing)
    assert not missing.exists()
    store = AssistanceStore(home)
    import shutil
    shutil.rmtree(home)
    with pytest.raises(FileNotFoundError):
        store.set_paused(True)
    assert not home.exists()


@pytest.mark.parametrize("bad", [
    finding(evidence=[]), finding(evidence=[evidence(source="")]),
    finding(evidence=[{"source_id": "gmail:a", "excerpt": "missing id"}]),
    finding(title=""), finding(kind=1), finding(summary="x" * 4097),
    finding(evidence=[evidence() | {"data_use": float("nan")}]),
])
def test_invalid_scan_is_atomic_and_does_not_log_evidence(home, bad, caplog):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    with pytest.raises(ValueError):
        store.ingest_scan(scan(finding(), bad, records=[evidence()]))
    assert store.list_findings() == [] and store.prior_evidence() == []
    assert "Due Friday" not in caplog.text


def test_disconnected_scan_does_not_add_findings_or_evidence(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    assert store.ingest_scan(scan(finding(), records=[evidence()], connected=False)) == []
    assert store.list_findings() == [] and store.prior_evidence() == []


@pytest.mark.parametrize("status,limit", [("invalid", 1), (None, -1), (None, True), (None, "20")])
def test_list_rejects_invalid_filters(home, status, limit):
    from gateway.connected_assistance_store import AssistanceStore

    with pytest.raises(ValueError):
        AssistanceStore(home).list_findings(status=status, limit=limit)


def test_list_is_bounded_and_returns_latest_inserted(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    for number in range(105):
        store.ingest_scan(scan(finding(entity_key=str(number), evidence=[evidence(record=str(number))])))
    assert len(store.list_findings()) == 20
    assert len(store.list_findings(limit=1000)) == 100
    assert store.list_findings(limit=1)[0]["entity_key"] == "104"
    assert store.list_findings(limit=0) == []


def test_prior_evidence_keeps_capability_and_pointer_provenance_separate(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    base = {"source_id": "gmail:account-a", "record_id": "same-id", "capability_id": "read-mail",
            "json_pointer": "/messages/0", "content": {"id": "same-id", "text": "Due Friday"},
            "data_use": ["no_external_llm"]}
    sibling = base | {"json_pointer": "/messages/1", "data_use": ["no_training"]}
    another_capability = base | {"capability_id": "read-drafts"}
    store.ingest_scan(scan(records=[base, sibling, another_capability]))
    assert store.prior_evidence() == [another_capability, sibling, base]
    assert AssistanceStore(home).prior_evidence()[2]["data_use"] == ["no_external_llm"]


def observed_scan(*, text="Please review Atlas today before lunch.", entity="Atlas", quote="Please review Atlas today",
                  source="gmail:work", capability="READ", pointer="", kind="direct_request", url=False, record="m1"):
    from agent.connected_assistance_evidence import observed_records, validate_findings

    response = {"id": record, "text": text, "url": "https://mail.test/m1"}
    if pointer:
        response = [response]
    records = observed_records([{"source_id": source, "capability_id": capability,
                                 "response": response, "data_use": ["no_training"]}])
    citation = {"source_id": source, "record_id": record, "json_pointer": pointer, "excerpt": quote}
    if url:
        citation["url"] = "https://mail.test/m1"
    row = {"kind": kind, "entity_key": entity, "title": "Review Atlas", "summary": "Review requested",
           "why_now": "Please review Atlas today", "suggested_action": "review", "evidence": [citation]}
    return scan(*validate_findings([row], records), records=records)


@pytest.mark.parametrize("status", ["open", "dismissed", "resolved"])
def test_same_observed_content_version_survives_model_rewording(home, status):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    first = store.ingest_scan(observed_scan())[0]
    if status != "open":
        store.transition(first["id"], status)
    assert AssistanceStore(home).ingest_scan(observed_scan(entity="Atlas review", quote="Please review Atlas today before lunch.", url=True)) == []
    assert len(store.list_findings()) == 1
    saved = store.get_finding(first["id"])
    assert saved is not None and saved["status"] == status


def test_actual_content_versions_not_forged_digests_drive_changes(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    original = observed_scan()
    first = store.ingest_scan(original)[0]
    updated = observed_scan(text="Please review Atlas today before dinner.")
    updated["findings"][0]["evidence"][0]["content_digest"] = first["evidence"][0].get("content_digest")
    updated["evidence"][0]["content_digest"] = "forged"
    changed = store.ingest_scan(updated)
    assert len(changed) == 1 and changed[0]["id"] == first["id"]
    assert changed[0]["evidence"][0]["content_digest"] != first["evidence"][0]["content_digest"]
    assert store.ingest_scan(original) == []
    for kwargs in ({"source": "gmail:personal"}, {"capability": "READ-OTHER"}, {"pointer": "/0"}):
        assert len(store.ingest_scan(observed_scan(entity=str(kwargs), **kwargs))) == 1


def test_legacy_without_observed_content_does_not_trust_digest(home):
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    first = store.ingest_scan(scan(finding()))[0]
    forged = finding(entity_key="reworded", evidence=[evidence() | {"content_digest": "f" * 64}])
    store.transition(first["id"], "dismissed")
    assert store.ingest_scan(scan(forged)) == []


def test_legacy_dismissal_is_upgraded_from_retained_actual_content(home):
    import json
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    result = observed_scan()
    first = store.ingest_scan(result)[0]
    store.transition(first["id"], "dismissed")
    # Simulate a pre-digest database with real observed data but legacy citation fingerprints.
    legacy = {k: v for k, v in first.items() if k not in {"id", "status"}}
    legacy["evidence"][0].pop("content_digest", None)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE findings SET payload=? WHERE id=?", (json.dumps(legacy), first["id"]))
        db.execute("DELETE FROM fingerprints")
    assert AssistanceStore(home).ingest_scan(observed_scan(entity="Atlas review", quote="Please review Atlas today before lunch.")) == []
    assert len(store.list_findings()) == 1


@pytest.mark.parametrize("receipt_status", ["unknown", "shown"])
def test_legacy_notification_receipts_survive_content_version_upgrade(home, receipt_status):
    import json
    from gateway.connected_assistance_store import AssistanceStore, notification_key

    store = AssistanceStore(home)
    first = store.ingest_scan(observed_scan())[0]
    legacy = {k: v for k, v in first.items() if k not in {"id", "status"}}
    legacy["evidence"][0].pop("content_digest", None)
    legacy_key = notification_key(legacy | {"id": first["id"]})
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE findings SET payload=? WHERE id=?", (json.dumps(legacy), first["id"]))
        db.execute("INSERT INTO assistance_notifications VALUES (?, ?, ?)",
                   (legacy_key, first["id"], receipt_status))
    reopened = AssistanceStore(home)
    assert reopened.list_findings(status="open", pending_notification=True) == []
    with sqlite3.connect(store.path) as db:
        statuses = db.execute("SELECT status FROM assistance_notifications WHERE finding_id=?", (first["id"],)).fetchall()
        assert statuses == [(receipt_status,)]


def test_legacy_local_management_never_loads_agent_or_connectors(home, monkeypatch):
    import builtins
    import json
    from gateway.connected_assistance_store import AssistanceStore

    store = AssistanceStore(home)
    first = store.ingest_scan(observed_scan())[0]
    payload = {k: v for k, v in first.items() if k not in {"id", "status"}}
    payload["evidence"][0].pop("content_digest")
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE findings SET payload=? WHERE id=?", (json.dumps(payload), first["id"]))
    original_import = builtins.__import__

    def no_apps(name, *args, **kwargs):
        if name.startswith(("agent.", "tools.connectors")):
            raise AssertionError("local management must not load the scan pipeline")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_apps)
    reopened = AssistanceStore(home)
    assert reopened.list_findings()
    assert reopened.get_finding(first["id"])
    assert reopened.prior_evidence()
    reopened.transition(first["id"], "dismissed")
    reopened.set_paused(True)
    assert reopened.is_paused()
