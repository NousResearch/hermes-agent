from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

_HISTORICAL_DDL = "CREATE TABLE IF NOT EXISTS tasks (\n    id                   TEXT PRIMARY KEY,\n    title                TEXT NOT NULL,\n    body                 TEXT,\n    assignee             TEXT,\n    status               TEXT NOT NULL,\n    priority             INTEGER DEFAULT 0,\n    created_by           TEXT,\n    created_at           INTEGER NOT NULL,\n    started_at           INTEGER,\n    completed_at         INTEGER,\n    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',\n    workspace_path       TEXT,\n    branch_name          TEXT,\n    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,\n    -- the task's worktree is anchored under the project's primary repo with a\n    -- deterministic branch name instead of a random wt/<task-id> fallback.\n    project_id           TEXT,\n    claim_lock           TEXT,\n    claim_expires        INTEGER,\n    tenant               TEXT,\n    result               TEXT,\n    idempotency_key      TEXT,\n    -- Unified consecutive-failure counter. Incremented on spawn\n    -- failure, timeout, or crash; reset only on successful completion.\n    -- The circuit breaker in _record_task_failure trips when this\n    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.\n    consecutive_failures INTEGER NOT NULL DEFAULT 0,\n    worker_pid           INTEGER,\n    -- Short excerpt of the most recent failure's error text.\n    last_failure_error   TEXT,\n    max_runtime_seconds  INTEGER,\n    last_heartbeat_at    INTEGER,\n    -- Pointer into task_runs for the currently-active run (NULL if no\n    -- run is in-flight). Denormalised for cheap reads.\n    current_run_id       INTEGER,\n    -- Forward-compat for v2 workflow routing. In v1 the kernel writes\n    -- these when the task is opted into a template but otherwise ignores\n    -- them; the dispatcher doesn't consult them for routing yet.\n    workflow_template_id TEXT,\n    current_step_key     TEXT,\n    -- Force-loaded skills for the worker on this task, stored as JSON.\n    -- Passed to the worker via `--skills`. NULL or empty array = no extras.\n    skills               TEXT,\n    -- Per-task model override. When set, the dispatcher passes -m <model>\n    -- to the worker, overriding the profile's default model. NULL = use\n    -- the profile default.\n    model_override       TEXT,\n    -- Provider the model override belongs to. When set (alongside\n    -- model_override), the dispatcher passes --provider <name> so the\n    -- worker resolves the model against the right backend instead of the\n    -- profile's configured provider. NULL = profile provider.\n    provider_override    TEXT,\n    -- Per-task reasoning effort for the worker (minimal|low|medium|high|\n    -- xhigh|max|ultra, or 'none' for thinking off). When set, the dispatcher\n    -- passes --reasoning <level> so the worker runs at that depth regardless\n    -- of the profile's agent.reasoning_effort. NULL = profile setting.\n    reasoning_effort     TEXT,\n    -- Per-task override for the consecutive-failure circuit breaker.\n    -- The value is the failure count at which the breaker trips — e.g.\n    -- ``max_retries=1`` blocks on the first failure. NULL (the common\n    -- case) falls through to the dispatcher-level ``kanban.failure_limit``\n    -- config and then ``DEFAULT_FAILURE_LIMIT``.\n    max_retries          INTEGER,\n    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an\n    -- auxiliary judge re-evaluates the worker's response against the\n    -- card title/body after each turn and feeds a continuation prompt\n    -- back into the SAME session until the judge agrees the work is done\n    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot\n    -- worker (the default).\n    goal_mode            INTEGER NOT NULL DEFAULT 0,\n    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the\n    -- goals-engine default.\n    goal_max_turns       INTEGER,\n    -- Originating chat/agent session id when the task was created from\n    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL\n    -- for tasks created from the CLI, dashboard, or any path that doesn't\n    -- set the env var. Indexed so per-session list queries stay cheap on\n    -- larger boards.\n    session_id           TEXT,\n    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or\n    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never\n    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go\n    -- to ``blocked`` for a human. Preserved across unblock so a re-block for\n    -- the SAME kind can be recognised as a loop.\n    block_kind           TEXT,\n    -- Unblock-loop counter. Incremented each time a task is re-blocked for the\n    -- same truly-blocked reason after having been unblocked. When it reaches\n    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of\n    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a\n    -- successful completion — NOT on unblock (resetting on unblock is exactly\n    -- the amnesia that let the loop run unbounded).\n    block_recurrences    INTEGER NOT NULL DEFAULT 0\n);\nCREATE TABLE IF NOT EXISTS task_events (\n    id         INTEGER PRIMARY KEY AUTOINCREMENT,\n    task_id    TEXT NOT NULL,\n    run_id     INTEGER,\n    kind       TEXT NOT NULL,\n    payload    TEXT,\n    created_at INTEGER NOT NULL\n);\nCREATE TABLE IF NOT EXISTS kanban_notify_subs (\n    task_id       TEXT NOT NULL,\n    platform      TEXT NOT NULL,\n    chat_id       TEXT NOT NULL,\n    thread_id     TEXT NOT NULL DEFAULT '',\n    user_id       TEXT,\n    created_at    INTEGER NOT NULL,\n    last_event_id INTEGER NOT NULL DEFAULT 0,\n    last_ping_event_id INTEGER NOT NULL DEFAULT 0,\n    PRIMARY KEY (task_id, platform, chat_id, thread_id)\n);"


def _insert_required(conn, table, values):
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    data = {}
    for col in info:
        name, typ, required, default, pk = col[1], col[2], col[3], col[4], col[5]
        if name in values:
            data[name] = values[name]
        elif pk and typ.upper() == "INTEGER":
            continue
        elif required and default is None:
            data[name] = 0 if "INT" in typ.upper() else ""
    cols = ",".join(data)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' for _ in data)})", tuple(data.values()))


def _legacy_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(_HISTORICAL_DDL)
    _insert_required(conn, "tasks", {"id":"legacy-task", "title":"legacy", "status":"done", "created_at":1, "updated_at":2})
    _insert_required(conn, "task_events", {"id":7, "task_id":"legacy-task", "kind":"completed", "created_at":2,
        "payload":'{"summary":"done"}', "actor":"worker"})
    _insert_required(conn, "kanban_notify_subs", {"task_id":"legacy-task", "platform":"telegram", "chat_id":"chat",
        "thread_id":"topic", "user_id":"user", "created_at":1, "last_event_id":6, "last_ping_event_id":7})
    conn.commit(); conn.close()


def test_pre_outbox_checkpoint_is_admitted_once_with_exact_receipt_provenance(tmp_path):
    path = tmp_path / "legacy.db"
    _legacy_db(path)
    kb.init_db(path)
    conn = kbc.connect(path)
    sub = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert sub["legacy_ping_after_event_id"] == 6
    assert sub["legacy_ping_through_event_id"] == 7
    assert sub["legacy_ping_admission_kind"] == "legacy_checkpoint_v1"
    old, new, events = kbn.claim_unseen_events_for_sub(conn, task_id="legacy-task", platform="telegram",
        chat_id="chat", thread_id="topic", kinds=["completed"], incarnation_id=sub["incarnation_id"])
    assert (old, new, [e.id for e in events]) == (6, 7, [7])
    row = dict(conn.execute("SELECT * FROM kanban_delivery_outbox").fetchone())
    assert row["incarnation_id"] == sub["incarnation_id"]
    assert row["state"] == "pending"
    assert row["ping_acceptance_provenance"] == "legacy_checkpoint_v1"
    assert row["ping_delivered_at"] is None and row["ping_receipt"] is None
    key = row["delivery_key"]
    assert conn.execute("SELECT legacy_ping_admission_kind FROM kanban_notify_subs").fetchone()[0] is None
    conn.close()
    kb.init_db(path)
    conn = kbc.connect(path)
    sub = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert sub["legacy_ping_admission_kind"] is None
    kbn.claim_unseen_events_for_sub(conn, task_id="legacy-task", platform="telegram", chat_id="chat",
        thread_id="topic", kinds=["completed"], incarnation_id=sub["incarnation_id"])
    rows = conn.execute("SELECT delivery_key,ping_acceptance_provenance FROM kanban_delivery_outbox").fetchall()
    assert [(r[0], r[1]) for r in rows] == [(key, "legacy_checkpoint_v1")]
    conn.close()


def test_existing_outbox_checkpoint_quarantines_only_missing_obligation(tmp_path):
    path = tmp_path / "outbox-before-marker.db"
    _legacy_db(path)
    conn = sqlite3.connect(path)
    conn.executescript(kbc._DELIVERY_OUTBOX_SQL)
    for event_id in (8, 9):
        _insert_required(conn, "task_events", {
            "id": event_id, "task_id": "legacy-task", "kind": "completed",
            "created_at": event_id, "payload": '{"summary":"done"}', "actor": "worker",
        })
    conn.execute("UPDATE kanban_notify_subs SET last_ping_event_id=9")
    _insert_required(conn, "kanban_delivery_outbox", {
        "delivery_key": "legacy-represented", "task_id": "legacy-task", "event_id": 7,
        "platform": "telegram", "chat_id": "chat", "thread_id": "topic",
        "incarnation_id": None, "notifier_profile": None, "payload_digest": "legacy-digest",
        "payload_json": '{"legacy":true}', "state": "delivery_unknown", "created_at": 3,
        "updated_at": 4,
    })
    _insert_required(conn, "kanban_delivery_outbox", {
        "delivery_key": "modern-represented", "task_id": "legacy-task", "event_id": 8,
        "platform": "telegram", "chat_id": "chat", "thread_id": "topic",
        "incarnation_id": "prior-incarnation", "notifier_profile": "prior-owner",
        "payload_digest": "modern-digest", "payload_json": '{"modern":true}',
        "state": "delivered", "transport_receipt": "real-receipt", "created_at": 5,
        "updated_at": 6,
    })
    conn.commit()
    conn.close()

    kb.init_db(path)
    conn = kbc.connect(path)
    sub = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert sub["legacy_ping_admission_kind"] == "legacy_checkpoint_uncertain_v1"
    assert (sub["legacy_ping_after_event_id"], sub["legacy_ping_through_event_id"]) == (6, 9)

    old, new, events = kbn.claim_unseen_events_for_sub(
        conn, task_id="legacy-task", platform="telegram", chat_id="chat",
        thread_id="topic", kinds=["completed"], incarnation_id=sub["incarnation_id"],
    )
    assert (old, new, [event.id for event in events]) == (6, 9, [7, 8, 9])
    rows = [dict(row) for row in conn.execute(
        "SELECT * FROM kanban_delivery_outbox ORDER BY event_id",
    )]
    assert len(rows) == 3
    by_key = {row["delivery_key"]: row for row in rows}
    assert by_key["legacy-represented"]["incarnation_id"] is None
    assert by_key["legacy-represented"]["payload_json"] == '{"legacy":true}'
    assert by_key["modern-represented"]["incarnation_id"] == "prior-incarnation"
    assert by_key["modern-represented"]["transport_receipt"] == "real-receipt"
    missing = next(row for row in rows if row["event_id"] == 9)
    assert missing["state"] == "delivery_unknown"
    assert missing["incarnation_id"] is None
    assert missing["ping_acceptance_provenance"] == "legacy_checkpoint_uncertain_v1"
    assert missing["ping_delivered_at"] is None
    assert missing["ping_receipt"] is None
    assert missing["transport_receipt"] is None
    conn.close()


@pytest.mark.parametrize(
    "mutation",
    ({"notifier_profile": "new-owner"}, {"delivery_mode": "notify"}),
    ids=("owner", "delivery-mode"),
)
def test_authority_mutation_discards_unconsumed_checkpoint_interval(tmp_path, mutation):
    path = tmp_path / f"mutation-{next(iter(mutation))}.db"
    _legacy_db(path)
    kb.init_db(path)
    conn = kbc.connect(path)
    before = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())

    kbn.add_notify_sub(
        conn, task_id="legacy-task", platform="telegram", chat_id="chat", thread_id="topic",
        notifier_profile=before["notifier_profile"], delivery_mode=before["delivery_mode"],
    )
    unchanged = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert unchanged["incarnation_id"] == before["incarnation_id"]
    assert unchanged["legacy_ping_admission_kind"] == "legacy_checkpoint_v1"

    kbn.add_notify_sub(
        conn, task_id="legacy-task", platform="telegram", chat_id="chat", thread_id="topic",
        **mutation,
    )
    changed = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert changed["incarnation_id"] != before["incarnation_id"]
    assert (
        changed["legacy_ping_after_event_id"], changed["legacy_ping_through_event_id"],
        changed["legacy_ping_admission_kind"],
    ) == (None, None, None)

    kbn.claim_unseen_events_for_sub(
        conn, task_id="legacy-task", platform="telegram", chat_id="chat",
        thread_id="topic", kinds=["completed"], incarnation_id=changed["incarnation_id"],
    )
    row = dict(conn.execute("SELECT * FROM kanban_delivery_outbox").fetchone())
    assert row["state"] == "pending"
    assert row["incarnation_id"] == changed["incarnation_id"]
    assert row["ping_acceptance_provenance"] is None
    conn.close()


def test_modern_checkpoint_is_not_seeded_and_prior_incarnation_does_not_suppress_admission(tmp_path):
    path = tmp_path / "modern.db"
    conn = kbc.connect(path)
    task_id = kb.create_task(conn, title="modern")
    kbn.add_notify_sub(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        notifier_profile="owner", delivery_mode="notify+wake",
    )
    sub = kbn.list_notify_subs(conn, task_id)[0]
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "completed", {"summary": "done"})
    event_id = int(conn.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id=?", (task_id,),
    ).fetchone()[0])
    kbn.claim_unseen_events_for_sub(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        kinds=["completed"], incarnation_id=sub["incarnation_id"],
    )
    first = dict(conn.execute("SELECT * FROM kanban_delivery_outbox").fetchone())

    kbn.add_notify_sub(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        delivery_mode="notify",
    )
    current = kbn.list_notify_subs(conn, task_id)[0]
    kbn.advance_notify_cursor(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        new_cursor=event_id - 1, incarnation_id=current["incarnation_id"],
    )
    kbn.record_notify_ping(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        event_id=event_id, incarnation_id=current["incarnation_id"],
    )
    conn.close()

    kb.init_db(path)
    conn = kbc.connect(path)
    current = kbn.list_notify_subs(conn, task_id)[0]
    assert current["legacy_ping_admission_kind"] is None
    kbn.claim_unseen_events_for_sub(
        conn, task_id=task_id, platform="telegram", chat_id="chat", thread_id="topic",
        kinds=["completed"], incarnation_id=current["incarnation_id"],
    )
    rows = [dict(row) for row in conn.execute(
        "SELECT * FROM kanban_delivery_outbox WHERE event_id=? ORDER BY id", (event_id,),
    )]
    assert len(rows) == 2
    assert rows[0]["delivery_key"] == first["delivery_key"]
    assert rows[0]["revoked_at"] is not None
    assert rows[1]["incarnation_id"] == current["incarnation_id"]
    assert rows[1]["ping_acceptance_provenance"] is None
    conn.close()


@pytest.mark.parametrize(
    "failed_column",
    ("legacy_ping_after_event_id", "legacy_ping_through_event_id", "legacy_ping_admission_kind"),
)
def test_interrupted_marker_migration_recovers_exact_legacy_admission(
    tmp_path, monkeypatch, failed_column,
):
    path = tmp_path / f"interrupted-{failed_column}.db"
    _legacy_db(path)
    original = kbc._add_column_if_missing

    def add_then_fail(conn, table, name, ddl):
        added = original(conn, table, name, ddl)
        if table == "kanban_notify_subs" and name == failed_column:
            raise RuntimeError("injected migration interruption")
        return added

    with monkeypatch.context() as patcher:
        patcher.setattr(kbc, "_add_column_if_missing", add_then_fail)
        with pytest.raises(RuntimeError, match="injected migration interruption"):
            kb.init_db(path)

    kb.init_db(path)
    conn = kbc.connect(path)
    sub = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    assert (
        sub["legacy_ping_after_event_id"], sub["legacy_ping_through_event_id"],
        sub["legacy_ping_admission_kind"],
    ) == (6, 7, "legacy_checkpoint_v1")
    kbn.claim_unseen_events_for_sub(
        conn, task_id="legacy-task", platform="telegram", chat_id="chat",
        thread_id="topic", kinds=["completed"], incarnation_id=sub["incarnation_id"],
    )
    obligations = [dict(row) for row in conn.execute(
        "SELECT state,ping_acceptance_provenance,ping_delivered_at "
        "FROM kanban_delivery_outbox WHERE event_id=7",
    )]
    assert obligations == [{
        "state": "pending", "ping_acceptance_provenance": "legacy_checkpoint_v1",
        "ping_delivered_at": None,
    }]
    conn.close()


@pytest.mark.parametrize("source_had_outbox", (False, True))
def test_interrupted_outbox_creation_preserves_source_provenance(
    tmp_path, monkeypatch, source_had_outbox,
):
    path = tmp_path / f"outbox-boundary-{source_had_outbox}.db"
    _legacy_db(path)
    if source_had_outbox:
        conn = sqlite3.connect(path)
        conn.executescript(kbc._DELIVERY_OUTBOX_SQL)
        conn.close()

    with monkeypatch.context() as patcher:
        patcher.setattr(
            kbc, "_DELIVERY_OUTBOX_SQL",
            kbc._DELIVERY_OUTBOX_SQL + "\nSELECT * FROM injected_missing_table;",
        )
        with pytest.raises(sqlite3.OperationalError, match="injected_missing_table"):
            kb.init_db(path)

    kb.init_db(path)
    conn = kbc.connect(path)
    sub = dict(conn.execute("SELECT * FROM kanban_notify_subs").fetchone())
    expected = (
        "legacy_checkpoint_uncertain_v1" if source_had_outbox
        else "legacy_checkpoint_v1"
    )
    assert (
        sub["legacy_ping_after_event_id"], sub["legacy_ping_through_event_id"],
        sub["legacy_ping_admission_kind"],
    ) == (6, 7, expected)
    conn.close()
