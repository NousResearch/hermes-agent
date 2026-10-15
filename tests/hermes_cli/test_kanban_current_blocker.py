"""Offline evidence and human-output contracts for the current blocking episode."""
from pathlib import Path

import pytest

from hermes_cli import kanban_diagnostics as kd


TASK_ID = "t_demo00"


def task(**fields):
    return {"id": TASK_ID, "status": "blocked", "consecutive_failures": 0, **fields}


def event(kind="blocked", reason="needs approval", **fields):
    return {"id": 1, "task_id": TASK_ID, "run_id": None, "kind": kind,
            "payload": {"reason": reason}, "created_at": 100, **fields}


def current(events, runs=None, task_fields=None):
    assert Path(kd.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
    return [d for d in kd.compute_task_diagnostics(
        task(**(task_fields or {})), events, runs or [], now=101
    ) if d.kind == "current_blocker"]


@pytest.mark.parametrize("with_kind", [False, True])
def test_current_blocker_is_visible_in_human_cli(capsys, with_kind):
    from hermes_cli.kanban import _print_diagnostics
    _print_diagnostics(current([event()]), "", with_kind=with_kind)
    line = capsys.readouterr().out.strip().splitlines()[0]
    assert "Blocked: needs approval" in line


@pytest.mark.parametrize("status", ["todo", "ready", "running", "done", "archived",
                                   "review", "scheduled", "triage", None, "", 7])
def test_current_blocker_requires_blocked_status(status):
    assert current([event()], task_fields={"status": status}) == []
    missing = task()
    del missing["status"]
    assert not [d for d in kd.compute_task_diagnostics(missing, [event()], [], now=101)
                if d.kind == "current_blocker"]


@pytest.mark.parametrize("events", [[], [event(reason="")]])
def test_current_blocker_never_uses_failure_fields_or_runs(events):
    runs = [{"id": 7, "task_id": TASK_ID, "summary": "run summary",
             "error": "run error", "outcome": "blocked"}]
    assert current(events, runs, {"last_failure_error": "failure error",
                                 "last_spawn_error": "legacy error"}) == []


@pytest.mark.parametrize("foreign_only", [False, True])
def test_current_blocker_ignores_foreign_task_reason(foreign_only):
    foreign = event(id=2, task_id="other", reason="foreign secret")
    diags = current([foreign] if foreign_only else [event(), foreign])
    assert [d.detail for d in diags] == ([] if foreign_only else ["needs approval"])


@pytest.mark.parametrize("identity", [None, 7, "", " ", " t_demo00", "t_demo00 "])
@pytest.mark.parametrize("location", ["target", "event", "missing_target", "missing_event"])
def test_current_blocker_rejects_missing_or_malformed_task_identity(identity, location):
    target = task()
    newest = event(id=2)
    if location == "target":
        target["id"] = identity
    elif location == "event":
        newest["task_id"] = identity
    elif location == "missing_target":
        del target["id"]
    else:
        del newest["task_id"]
    assert not [d for d in kd.compute_task_diagnostics(target, [event(), newest], [], now=101)
                if d.kind == "current_blocker"]


@pytest.mark.parametrize("reference,runs", [
    (True, [{"id": 1, "task_id": TASK_ID}]),
    ("7", [{"id": 7, "task_id": TASK_ID}]),
    (7.0, [{"id": 7, "task_id": TASK_ID}]),
    (0, [{"id": 0, "task_id": TASK_ID}]),
    (-1, [{"id": -1, "task_id": TASK_ID}]),
    (7, [{"task_id": TASK_ID}]),
    (7, [{"id": True, "task_id": TASK_ID}]),
    (7, [{"id": "7", "task_id": TASK_ID}]),
    (7, [{"id": 7.0, "task_id": TASK_ID}]),
    (7, [{"id": 0, "task_id": TASK_ID}]),
    (7, [{"id": -1, "task_id": TASK_ID}]),
    (7, [{"id": 7}]),
    (7, [{"id": 7, "task_id": None}]),
    (7, [{"id": 7, "task_id": 7}]),
    (7, [{"id": 7, "task_id": ""}]),
    (7, [{"id": 7, "task_id": " t_demo00"}]),
    (7, [{"id": 7, "task_id": TASK_ID}, {"task_id": "other"}]),
])
def test_current_blocker_malformed_run_identity_fails_closed(reference, runs):
    assert current([event(), event(id=2, kind="heartbeat", run_id=reference)], runs) == []
    # Malformed supplied run IDs distrust non-null references, not task scope.
    assert current([event()], runs)[0].detail == "needs approval"


@pytest.mark.parametrize("kind", ["unblocked", "claimed", "status"])
def test_current_blocker_ignores_foreign_task_boundary(kind):
    foreign = event(kind, id=False, task_id="other", run_id="bad")
    assert current([event(), foreign])[0].detail == "needs approval"


def test_current_blocker_requires_same_task_run_binding():
    diags = current([event(run_id=7)], [{"id": 7, "task_id": TASK_ID,
                                       "summary": "not reason", "error": "not reason"}])
    assert diags[0].detail == "needs approval"


def test_current_blocker_orphan_run_clears_evidence():
    assert current([event(), event("heartbeat", id=2, run_id=7)]) == []


def test_current_blocker_cross_task_run_clears_evidence():
    assert current([event(), event("heartbeat", id=2, run_id=7)],
                   [{"id": 7, "task_id": "other"}]) == []


@pytest.mark.parametrize("owner", [TASK_ID, "other"])
def test_current_blocker_duplicate_run_ids_are_ambiguous(owner):
    assert current([event(), event(id=2, run_id=7)],
                   [{"id": 7, "task_id": TASK_ID}, {"id": 7, "task_id": owner}]) == []


def test_current_blocker_orders_events_by_id_without_mutation():
    from copy import deepcopy
    target = task()
    events = [event(id=9, reason="new", created_at=2), event(id=1, created_at=100)]
    runs = [{"id": 7, "task_id": TASK_ID}]
    before = deepcopy((target, events, runs))
    diags = kd.compute_task_diagnostics(target, events, runs, now=101)
    assert [d.detail for d in diags if d.kind == "current_blocker"] == ["new"]
    assert (target, events, runs) == before


@pytest.mark.parametrize("ids,expected", [
    ([None, None], "new"), ([2, 2], "new"), ([1, None], None),
    ([None, 1], None), ([True, 2], None), (["1", 2], None),
    ([1.0, 2], None), ([0, 2], None), ([-1, 2], None),
])
def test_current_blocker_handles_synthetic_event_ids(ids, expected):
    events = [event(id=ids[0]), event(id=ids[1], reason="new")]
    if ids == [None, None]:
        del events[0]["id"]
    assert [d.detail for d in current(events)] == ([] if expected is None else [expected])


@pytest.mark.parametrize("run_id", [None, 7])
def test_current_blocker_exposes_breaker_error(run_id):
    diags = current([event("gave_up", run_id=run_id, payload={"error": "worker unavailable"})],
                    [{"id": 7, "task_id": TASK_ID}])
    assert len(diags) == 1
    assert diags[0].detail == "worker unavailable"
    assert diags[0].data == {"source_event_kind": "gave_up", "source_event_id": 1}


@pytest.mark.parametrize("payload", [{"classified_in_place": True},
    {"classified_in_place": True, "reason": None},
    {"classified_in_place": True, "reason": " \t"}])
def test_current_blocker_blank_classification_retains_same_episode_breaker(payload):
    breaker = event("gave_up", payload={"error": "breaker"}, created_at=10)
    diags = current([breaker, event("commented", id=2), event(id=3, payload=payload)])
    assert len(diags) == 1
    assert diags[0].detail == "breaker"
    assert diags[0].data == {"source_event_kind": "gave_up", "source_event_id": 1}
    assert diags[0].first_seen_at == diags[0].last_seen_at == 10


@pytest.mark.parametrize("kind", ["blocked", "gave_up", "classification"])
@pytest.mark.parametrize("payload", [None, [], 7, "{bad", "[]", "null", '"text"',
                                    {}, {"reason": "", "error": ""},
                                    {"reason": 7, "error": 7},
                                    {"reason": " \t", "error": " \t"}])
def test_current_blocker_invalid_newest_evidence_clears_reason(kind, payload):
    if kind == "classification" and isinstance(payload, dict):
        payload = {**payload, "classified_in_place": True}
        # Blank classifications retain a breaker; use ordinary prior evidence here.
    newest = event("blocked" if kind == "classification" else kind, id=2, payload=payload)
    assert current([event(), newest]) == []
    assert current([event(), newest, event(id=3, reason="fresh")])[0].detail == "fresh"


BOUNDARIES = """created claimed promoted promoted_manual unblocked reclaimed completed
archived scheduled review_requested changes_requested review_reopened specified decomposed
dependency_wait block_loop_detected stale reconciled timed_out crashed spawn_failed
rate_limited protocol_violation status""".split()


@pytest.mark.parametrize("kind", BOUNDARIES)
def test_current_blocker_lifecycle_boundaries_clear_evidence(kind):
    history = [event("gave_up", payload={"error": "old"}), event(kind, id=2)]
    assert current(history) == []
    assert current(history + [event(id=3, reason="fresh")])[0].detail == "fresh"


@pytest.mark.parametrize("kind", [None, "", " \t", 7, [], {}])
def test_current_blocker_malformed_kind_clears_evidence(kind):
    history = [event(), event(kind, id=2)]
    assert current(history) == []
    assert current(history + [event(id=3, reason="fresh")])[0].detail == "fresh"
    missing = event(id=2)
    del missing["kind"]
    assert current([event(), missing]) == []


@pytest.mark.parametrize("timestamp", [None, True, "100", 1.5, -1, "bad"])
def test_current_blocker_invalid_timestamp_uses_zero(timestamp):
    diags = current([event(created_at=timestamp)])
    assert len(diags) == 1
    assert diags[0].first_seen_at == diags[0].last_seen_at == 0


@pytest.mark.parametrize("kind,payload", [("blocked", {"reason": "scoped"}),
                                         ("gave_up", {"error": "scoped"})])
@pytest.mark.parametrize("omitted", [False, True])
def test_current_blocker_accepts_null_task_scoped_evidence(kind, payload, omitted):
    ev = event(kind, payload=payload)
    if omitted:
        del ev["run_id"]
    assert current([ev])[0].detail == "scoped"


@pytest.mark.parametrize("kind", "commented assigned edited reprioritized linked unlinked attached attachment_removed model_override_set reasoning_effort_set heartbeat claim_extended future_info".split())
def test_current_blocker_informational_events_preserve_reason(kind):
    assert current([event(), event(kind, id=2, run_id=7)],
                   [{"id": 7, "task_id": TASK_ID}])[0].detail == "needs approval"


def test_current_blocker_classification_reason_replaces_breaker():
    history = [event("gave_up", payload={"error": "old"}),
               event(id=2, created_at=99, payload={"classified_in_place": True, "reason": "new"})]
    diag = current(history)[0]
    assert diag.detail == "new"
    assert diag.first_seen_at == diag.last_seen_at == 99
    assert diag.data == {"source_event_kind": "blocked", "source_event_id": 2}


@pytest.mark.parametrize("middle", [event("unblocked", id=2), event(id=2, reason=""),
                                    event(id=2, reason="ordinary")])
def test_current_blocker_blank_classification_does_not_revive_history(middle):
    assert current([event("gave_up", payload={"error": "old"}), middle,
                    event(id=3, payload={"classified_in_place": True, "reason": None})]) == []


@pytest.mark.parametrize("kind", ["blocked", "unblocked", "heartbeat"])
def test_current_blocker_invalid_binding_cannot_resurrect_breaker(kind):
    history = [event("gave_up", payload={"error": "old"}), event(kind, id=2, run_id=7),
               event(id=3, payload={"classified_in_place": True})]
    assert current(history) == []
    assert current(history + [event(id=4, reason="fresh")])[0].detail == "fresh"


def test_current_blocker_supports_rows_and_dataclasses():
    import json
    import sqlite3
    from dataclasses import asdict
    from hermes_cli import kanban_db as kb
    ev = event(run_id=7)
    run = dict(id=7, task_id=TASK_ID, profile=None, step_key=None, status="closed",
               claim_lock=None, claim_expires=None, worker_pid=None, max_runtime_seconds=None,
               last_heartbeat_at=None, started_at=90, ended_at=100, outcome="blocked",
               summary=None, metadata=None, error=None)
    expected = current([ev], [run])[0].to_dict()
    assert current([kb.Event(**ev)], [kb.Run(**run)])[0].to_dict() == expected
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        row_event = conn.execute(
            "SELECT 1 AS id, ? AS task_id, 7 AS run_id, 'blocked' AS kind, ? AS payload, 100 AS created_at",
            (TASK_ID, json.dumps(ev["payload"])),
        ).fetchone()
        row_run = conn.execute("SELECT 7 AS id, ? AS task_id", (TASK_ID,)).fetchone()
        assert current([row_event], [row_run])[0].to_dict() == expected
    assert asdict(kb.Event(**ev)) == ev


def assert_safe_outputs(diag, capsys, secrets):
    import json
    from hermes_cli.kanban import _print_diagnostics

    _print_diagnostics([diag], "", with_kind=True)
    rendered = capsys.readouterr().out
    outputs = [diag.title, diag.detail, json.dumps(diag.to_dict()), rendered]
    for output in outputs:
        for secret in secrets:
            assert secret not in output
    return rendered


def test_current_blocker_redacts_url_credentials(capsys):
    secrets = ["shortuser", "shortpass", "querysecret"]
    reason = "https://shortuser:shortpass@example.invalid/path?token=querysecret"
    diag = current([event(reason=reason)])[0]
    assert_safe_outputs(diag, capsys, secrets)
    assert "example.invalid" in diag.detail


def test_current_blocker_removes_terminal_controls(capsys):
    reason = "\x1b[31mneeds\x1b[0m\r\n\tapproval\x1b]8;;https://hidden.invalid\x07 now\x1b]8;;\x1b\\\x00\x85next\u200b\u202e\x9b"
    diag = current([event(reason=reason)])[0]
    assert diag.detail == "needs approval now next"
    assert diag.title == "Blocked: needs approval now next"
    rendered = assert_safe_outputs(diag, capsys, ["hidden.invalid", "\x1b", "\r", "\x00", "\x85", "\u200b", "\u202e", "\x9b"])
    assert "[warning] current_blocker: Blocked: needs approval now next" in rendered


@pytest.mark.parametrize("length", [171, 1000, 1500])
def test_current_blocker_bounds_title_and_detail(length):
    reason = "x" * length
    diag = current([event(reason=reason)])[0]
    title = "Blocked: " + reason
    assert diag.title == (title if len(title) <= 180 else title[:179] + "…")
    assert diag.detail == (reason if length <= 1000 else reason[:999] + "…")


@pytest.mark.parametrize("kind", ["blocked", "gave_up"])
def test_current_blocker_empty_sanitized_reason_does_not_revive_history(kind):
    payload = {"reason" if kind == "blocked" else "error": "\x1b[31m\x1b[0m\x00\u200b"}
    assert current([event(), event(kind, id=2, payload=payload)]) == []
    classification = event(id=3, payload={"classified_in_place": True, "reason": " "})
    assert current([event(), event(kind, id=2, payload=payload), classification]) == []


def test_current_blocker_redacts_with_ordinary_redaction_disabled(monkeypatch, capsys):
    from agent import redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    diag = current([event(reason="https://disableduser:disabledpass@example.invalid/?api_key=disabledquery")])[0]
    assert_safe_outputs(diag, capsys, ["disableduser", "disabledpass", "disabledquery"])
    assert "***" in diag.detail


def test_current_blocker_redacts_before_truncation(capsys):
    prefix = "x" * 145 + " https://"
    secret = "cutoffcredentialspanningboundary"
    raw = prefix + secret + ":cutoffpass@example.invalid/?token=cutoffquery"
    diag = current([event(reason=raw)])[0]
    assert_safe_outputs(diag, capsys, [secret, "cutoffpass", "cutoffquery"])
    # Truncating first would expose the credential prefix, not merely its full value.
    assert secret[:12] not in diag.title
    assert "***:***@example.invalid" in diag.detail
    expected = "Blocked: " + diag.detail
    assert diag.title == (expected if len(expected) <= 180 else expected[:179] + "…")


def test_current_blocker_serializes_only_safe_metadata(capsys):
    reason = "https://metauser:metapass@example.invalid/?token=metaquery"
    diag = current([event(run_id=7, reason=reason, payload={"reason": reason, "private": "payloadsecret"})],
                   [{"id": 7, "task_id": TASK_ID, "summary": "summarysecret", "error": "errorsecret"}])[0]
    assert diag.to_dict()["data"] == {"source_event_kind": "blocked", "source_event_id": 1}
    assert diag.run_id is None
    assert_safe_outputs(diag, capsys, ["metauser", "metapass", "metaquery", "payloadsecret", "summarysecret", "errorsecret"])


@pytest.fixture
def show_fixture(monkeypatch):
    import argparse
    import sqlite3
    from contextlib import contextmanager
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db as kb

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE task_runs (
        id INTEGER, task_id TEXT, profile TEXT, step_key TEXT, status TEXT,
        claim_lock TEXT, claim_expires INTEGER, worker_pid INTEGER,
        max_runtime_seconds INTEGER, last_heartbeat_at INTEGER, started_at INTEGER,
        ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT, error TEXT
    )""")
    conn.executemany("""INSERT INTO task_runs (id, task_id, profile, status,
        started_at, ended_at, outcome) VALUES (?, ?, 'fixture', 'closed', 90, 100, ?)""",
        [(1, TASK_ID, "completed"), (7, TASK_ID, "blocked")])
    target = kb.Task(id=TASK_ID, title="Fixture", body=None, assignee="fixture",
                     status="blocked", priority=0, created_by="fixture", created_at=90,
                     started_at=None, completed_at=None, workspace_kind="scratch",
                     workspace_path=None, claim_lock=None, claim_expires=None, tenant=None,
                     max_retries=2)

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(kd.time, "time", lambda: 101)
    monkeypatch.setattr(kc.kbc, "connect_closing", connection)
    monkeypatch.setattr(kb, "get_task", lambda *_: target)
    monkeypatch.setattr(kb, "list_comments", lambda *_: [])
    monkeypatch.setattr(kb, "list_events", lambda *_: [kb.Event(**event(run_id=7))])
    monkeypatch.setattr(kb, "parent_ids", lambda *_: [])
    monkeypatch.setattr(kb, "child_ids", lambda *_: [])
    monkeypatch.setattr(kb, "latest_summary", lambda *_: None)
    monkeypatch.setattr(kb, "task_graph_context", lambda *_: {"parents": [], "children": []})

    def show(**fields):
        return kc._cmd_show(argparse.Namespace(
            **{"task_id": TASK_ID, "json": False, "state_type": "outcome",
               "state_name": "completed", **fields}))

    try:
        yield show
    finally:
        conn.close()


def test_show_state_filter_preserves_current_blocker_diagnostic(show_fixture, capsys):
    assert show_fixture() == 0
    output = capsys.readouterr().out
    assert any("[warning] Blocked: needs approval" in line for line in output.splitlines())
    runs = output.split("Runs (1):")[1]
    assert "#1 " in runs
    assert "#7 " not in runs


def test_show_empty_run_filter_preserves_current_blocker_diagnostic(show_fixture, capsys):
    assert show_fixture(state_name="gave_up") == 0
    output = capsys.readouterr().out
    assert "[warning] Blocked: needs approval" in output
    assert "Runs (" not in output


def test_show_without_run_filter_keeps_full_run_history(show_fixture, capsys):
    assert show_fixture(state_type=None, state_name=None) == 0
    output = capsys.readouterr().out
    assert "[warning] Blocked: needs approval" in output
    runs = output.split("Runs (2):")[1]
    assert "#1 " in runs and "#7 " in runs


def test_show_json_state_filter_preserves_existing_payload(show_fixture, capsys, monkeypatch):
    import json

    def forbidden(*args, **kwargs):
        raise AssertionError("JSON must not compute diagnostics")

    monkeypatch.setattr(kd, "compute_task_diagnostics", forbidden)
    assert show_fixture(json=True) == 0
    output = capsys.readouterr().out
    data = json.loads(output)
    assert set(data) == {"task", "latest_summary", "parents", "children", "comments", "events", "runs"}
    assert data["task"]["id"] == TASK_ID
    assert [(r["id"], r["outcome"]) for r in data["runs"]] == [(1, "completed")]
    assert data["events"][0]["run_id"] == 7
    assert "Diagnostics" not in output


@pytest.mark.parametrize("fields", [{"state_type": None}, {"state_name": None}])
def test_show_rejects_incomplete_state_filter_before_connection(show_fixture, capsys, monkeypatch, fields):
    from hermes_cli import kanban as kc

    def forbidden():
        raise AssertionError("incomplete filter must not connect")

    monkeypatch.setattr(kc.kbc, "connect_closing", forbidden)
    assert show_fixture(**fields) == 2
    assert "kanban show: pass both --state-type and --state-name, or omit both" in capsys.readouterr().err


def test_current_blocker_exposes_ordinary_reason():
    diags = current([event()])
    assert len(diags) == 1
    assert diags[0].to_dict() == {
        "kind": "current_blocker", "severity": "warning",
        "title": "Blocked: needs approval", "detail": "needs approval",
        "actions": [{"kind": "comment", "label": "Review and discuss the blocker",
                     "payload": {}, "suggested": True}],
        "first_seen_at": 100, "last_seen_at": 100, "count": 1, "run_id": None,
        "data": {"source_event_kind": "blocked", "source_event_id": 1},
    }
