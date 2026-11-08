"""``AdmissionRepository`` on SQLite against the legacy store: per verb, the same outcome and the same rows as the
legacy function (or the legacy gateway statement the verb absorbed) on copies of one seed, one transaction per verb,
and a failure inside a write rolling every table back. Verbs that reach a conversation body are in
``test_admission_conversation.py``; the hosted member check is in ``test_admission_hosted.py``."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3

import pytest

from persistence.authority.admissions import UnheldMedia
from tests.persistence_sqlite.authority._support import assert_same, counted_transactions, dump, outcome, run_both
from tests.persistence_sqlite.authority._admission_support import (
    EPOCH, admit, assert_rolls_back, authority, begin_epoch, claim, record_dict, recover, register, settle, steer,
)


def _rows(records):
    return [record_dict(r) for r in records]


# ── seeds ──────────────────────────────────────────────────────────────────────


def _running(db, epoch):
    """s1: r1 claimed (a worker on its generation), r2 queued and steered onto the claim, r3 queued."""
    db.create_session("s1", source="cli")
    db.create_session("s2", source="cli")
    first, second, third = (admit(db, "s1", r) for r in ("r1", "r2", "r3"))
    claimed = claim(db, "s1")
    register(db, "s1", claimed["generation"])
    steer(db, second, claimed)
    return {"first": first, "second": second, "third": third, "claimed": claimed}


_RESULT = {"result": {"final_response": "done", "messages": [
    {"role": "user", "content": "r1"}, {"role": "assistant", "content": "done"}]}, "usage": {}}


# ── settle (without a boundary) ────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["completed", "failed", "stale", "bad_outcome", "stale_epoch"])
def test_settle_matches_legacy(seeded, legacy_store, case):
    from hermes_state_runtime import settle_session_input
    path, store, v = seeded(_running)
    row = v["claimed"]
    generation = row["generation"] + (1 if case == "stale" else 0)
    epoch = EPOCH + (1 if case == "stale_epoch" else 0)
    outcome_value = {"failed": "failed", "bad_outcome": "cancelled"}.get(case, "completed")
    transfers = {v["second"]["admission_id"]: case == "completed"}
    legacy, backend = run_both(
        path,
        lambda db: settle_session_input(db, epoch=epoch, admission_id=row["admission_id"], generation=generation,
                                        outcome=outcome_value, result=_RESULT, _transfers=transfers),
        lambda: store.admissions.settle(epoch=epoch, admission_id=row["admission_id"], generation=generation,
                                        outcome=outcome_value, result=_RESULT, transfers=transfers))
    assert_same(legacy, backend, record_dict)


def test_settle_is_one_write_and_rolls_back(seeded):
    path, store, v = seeded(_running)
    row = v["claimed"]
    call = lambda: store.admissions.settle(epoch=EPOCH, admission_id=row["admission_id"],
                                           generation=row["generation"], outcome="completed", result=_RESULT)
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 0}


# ── cancel, steer, revert_steer ────────────────────────────────────────────────


@pytest.mark.parametrize("which", ["third", "first", "second", "missing"])
def test_cancel_matches_legacy(seeded, legacy_store, which):
    from hermes_state_runtime import cancel_session_input
    path, store, v = seeded(_running)
    admission_id = v[which]["admission_id"] if which != "missing" else "absent"
    legacy, backend = run_both(path, lambda db: cancel_session_input(db, epoch=EPOCH, admission_id=admission_id),
                               lambda: store.admissions.cancel(epoch=EPOCH, admission_id=admission_id))
    assert_same(legacy, backend, record_dict)


def test_cancel_unknown_matches_legacy(seeded, legacy_store):
    from hermes_state_runtime import cancel_session_input

    def seed(db, epoch):
        values = _running(db, epoch)
        recover(db, begin_epoch(db, "next"))
        return values

    path, store, v = seeded(seed)
    admission_id = v["first"]["admission_id"]
    legacy, backend = run_both(path, lambda db: cancel_session_input(db, epoch=EPOCH + 1, admission_id=admission_id),
                               lambda: store.admissions.cancel(epoch=EPOCH + 1, admission_id=admission_id))
    assert_same(legacy, backend, record_dict)
    assert backend[1].reason == "unknown_execution"


@pytest.mark.parametrize("which,offset", [("third", 0), ("second", 0), ("first", 0), ("third", 1)])
def test_steer_matches_legacy(seeded, legacy_store, which, offset):
    from hermes_state_runtime import steer_session_input
    path, store, v = seeded(_running)
    admission_id, generation = v[which]["admission_id"], v["claimed"]["generation"] + offset
    legacy, backend = run_both(
        path, lambda db: steer_session_input(db, epoch=EPOCH, admission_id=admission_id, generation=generation),
        lambda: store.admissions.steer(epoch=EPOCH, admission_id=admission_id, generation=generation))
    assert_same(legacy, backend, lambda pair: (record_dict(pair[0]), pair[1]))


@pytest.mark.parametrize("which", ["second", "third"])
def test_revert_steer_matches_legacy(seeded, legacy_store, which):
    from hermes_state_runtime import revert_steer_input
    path, store, v = seeded(_running)
    admission_id, generation = v[which]["admission_id"], v["claimed"]["generation"]
    legacy, backend = run_both(
        path, lambda db: revert_steer_input(db, epoch=EPOCH, admission_id=admission_id, generation=generation),
        lambda: store.admissions.revert_steer(epoch=EPOCH, admission_id=admission_id, generation=generation))
    assert_same(legacy, backend)


@pytest.mark.parametrize("verb", ["cancel", "steer", "revert_steer"])
def test_lifecycle_writes_are_one_write_and_roll_back(seeded, verb):
    path, store, v = seeded(_running)
    generation = v["claimed"]["generation"]
    calls = {
        "cancel": lambda: store.admissions.cancel(epoch=EPOCH, admission_id=v["third"]["admission_id"]),
        "steer": lambda: store.admissions.steer(epoch=EPOCH, admission_id=v["third"]["admission_id"],
                                                generation=generation),
        "revert_steer": lambda: store.admissions.revert_steer(epoch=EPOCH, admission_id=v["second"]["admission_id"],
                                                              generation=generation),
    }
    assert_rolls_back(path, store.holder(), calls[verb])
    with counted_transactions(store.holder()) as counts:
        calls[verb]()
    assert counts == {"write": 1, "read": 0}


# ── mark_unknown ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["started", "delivered", "stale", "terminal"])
def test_mark_unknown_matches_legacy(seeded, legacy_store, monkeypatch, case):
    from tests.persistence_sqlite._main_callers import session_settlement_recovery as recovery
    path, store, v = seeded(_running)
    row = dict(v["claimed"])
    if case == "stale":
        row["generation"] += 1
    transfers = {v["second"]["admission_id"]: case == "delivered"} if case in ("started", "delivered") else None
    monkeypatch.setattr(recovery, "_transfers", lambda _authority, _live, _row: transfers)
    if case == "terminal":
        row = dict(v["second"], generation=v["claimed"]["generation"], owner_epoch=EPOCH)
    legacy, backend = run_both(
        path, lambda db: recovery._mark_unknown(authority(db), row),
        lambda: store.admissions.mark_unknown(epoch=EPOCH, admission_id=row["admission_id"],
                                              generation=row["generation"], transfers=transfers))
    assert_same(legacy, backend)


def test_mark_unknown_is_one_write_and_rolls_back(seeded):
    path, store, v = seeded(_running)
    row = v["claimed"]
    call = lambda: store.admissions.mark_unknown(epoch=EPOCH, admission_id=row["admission_id"],
                                                 generation=row["generation"])
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        assert call() == "unknown"
    assert counts == {"write": 1, "read": 0}


# ── rehome, retarget, recover ──────────────────────────────────────────────────


def _rehome_seed(db, epoch):
    db.create_session("old", source="cli")
    db.create_session("new", source="cli")
    rows = [admit(db, "old", "r1", principal_id="local:a"), admit(db, "old", "r2", principal_id="other"),
            admit(db, "old", "r3", principal_id="local:b")]
    return rows


@pytest.mark.parametrize("target,prefix", [("new", "local:"), ("missing", "local:"), ("new", "nobody")])
def test_rehome_queued_matches_legacy(seeded, legacy_store, target, prefix):
    from hermes_state_runtime import rehome_queued_inputs
    path, store, _ = seeded(_rehome_seed)
    legacy, backend = run_both(
        path, lambda db: rehome_queued_inputs(db, epoch=EPOCH, old_session_id="old", new_session_id=target,
                                              principal_prefix=prefix),
        lambda: store.admissions.rehome_queued(epoch=EPOCH, old_session_id="old", new_session_id=target,
                                               principal_prefix=prefix))
    assert_same(legacy, backend)


def _retarget_seed(db, epoch):
    db.create_session("owner", source="cli")
    db.create_session("tip", source="cli")
    return [admit(db, "owner", f"r{i}", principal_id="peer",
                  payload={"text": f"t{i}", "local_automation_v1": {"target": target}})
            for i, target in enumerate(("stale-a", "kept", "stale-b"))]


@pytest.mark.parametrize("new_target", ["tip", "missing"])
def test_retarget_local_automation_matches_legacy(seeded, legacy_store, new_target):
    from hermes_state_runtime import retarget_local_automation
    path, store, _ = seeded(_retarget_seed)
    legacy, backend = run_both(
        path, lambda db: retarget_local_automation(db, epoch=EPOCH, session_id="owner", principal_id="peer",
                                                   old_targets=["stale-a", "stale-b"], new_target=new_target),
        lambda: store.admissions.retarget_local_automation(epoch=EPOCH, session_id="owner", principal_id="peer",
                                                           old_targets=("stale-a", "stale-b"), new_target=new_target))
    assert_same(legacy, backend)


def _next_owner(db, epoch):
    values = _running(db, epoch)
    values["next"] = begin_epoch(db, "next")
    return values


@pytest.mark.parametrize("epoch_offset", [1, 0, 2])
def test_recover_matches_legacy(seeded, legacy_store, epoch_offset):
    from hermes_state_runtime import recover_session_inputs
    path, store, _v = seeded(_next_owner)
    epoch = EPOCH + epoch_offset
    legacy, backend = run_both(path, lambda db: recover_session_inputs(db, epoch=epoch),
                               lambda: store.admissions.recover(epoch=epoch))
    assert_same(legacy, backend)


@pytest.mark.parametrize("verb", ["rehome_queued", "retarget_local_automation", "recover"])
def test_rehoming_writes_are_one_write_and_roll_back(seeded, verb):
    seeds = {"rehome_queued": _rehome_seed, "retarget_local_automation": _retarget_seed, "recover": _next_owner}
    path, store, _ = seeded(seeds[verb])
    calls = {
        "rehome_queued": lambda: store.admissions.rehome_queued(epoch=EPOCH, old_session_id="old",
                                                                new_session_id="new", principal_prefix="local:"),
        "retarget_local_automation": lambda: store.admissions.retarget_local_automation(
            epoch=EPOCH, session_id="owner", principal_id="peer", old_targets=["stale-a"], new_target="tip"),
        "recover": lambda: store.admissions.recover(epoch=EPOCH + 1),
    }
    assert_rolls_back(path, store.holder(), calls[verb])
    with counted_transactions(store.holder()) as counts:
        calls[verb]()
    assert counts == {"write": 1, "read": 0}


# ── reads of the legacy runtime module ─────────────────────────────────────────


def _reads_seed(db, epoch):
    values = _running(db, epoch)
    db.create_session("gone", source="cli")
    done = admit(db, "gone", "rg", principal_id="api", payload={"text": "gone"})
    settle(db, claim(db, "gone"), result=_RESULT)
    db.delete_session("gone")
    values["retired"] = done
    return values


@pytest.mark.parametrize("pending_only", [True, False])
def test_list_matches_legacy(seeded, legacy_store, pending_only):
    from hermes_state_runtime import list_session_admissions
    path, store, _ = seeded(_reads_seed)
    legacy, backend = run_both(path, lambda db: list_session_admissions(db, session_id="s1", pending_only=pending_only),
                               lambda: store.admissions.list(session_id="s1", pending_only=pending_only))
    assert_same(legacy, backend, _rows)


@pytest.mark.parametrize("which", ["first", "retired", "missing"])
def test_get_matches_legacy(seeded, legacy_store, which):
    from hermes_state_runtime import get_session_admission
    path, store, v = seeded(_reads_seed)
    admission_id = v[which]["admission_id"] if which != "missing" else "absent"
    legacy, backend = run_both(path, lambda db: get_session_admission(db, admission_id=admission_id),
                               lambda: store.admissions.get(admission_id=admission_id))
    assert_same(legacy, backend, record_dict)
    if which == "retired":
        assert backend[1] is not None and backend[1].status == "terminal"


@pytest.mark.parametrize("key", [("p", "s1", "r2"), ("p", "s1", "nope"), ("api", "gone", "rg")])
def test_find_matches_legacy(seeded, legacy_store, key):
    from hermes_state_runtime import find_session_admission
    path, store, _ = seeded(_reads_seed)
    principal_id, session_id, request_id = key
    legacy, backend = run_both(
        path, lambda db: find_session_admission(db, principal_id=principal_id, session_id=session_id,
                                                request_id=request_id),
        lambda: store.admissions.find(principal_id=principal_id, session_id=session_id, request_id=request_id))
    assert_same(legacy, backend, record_dict)


# ── reads the gateway ran as its own SQL ───────────────────────────────────────


def _request_admission_rows(db, request_id):
    """Main's admission read in gateway/platforms/api_server_response_admissions.py."""
    from hermes_state_terminal import terminal_api_admissions
    with db._read_ctx() as conn:
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM session_admissions WHERE principal_id='api' AND request_id=?", (request_id,))]
        rows.extend(terminal_api_admissions(conn, request_id))
    return rows


def test_match_matches_the_gateway_lookups(seeded, legacy_store):
    path, store, v = seeded(_reads_seed)
    ids = lambda rows: [row["admission_id"] for row in rows]
    legacy, backend = run_both(path, lambda db: ids(db._read_all(  # main's lookup in gateway/platforms/webhook_ingress.py
        'SELECT admission_id FROM session_admissions WHERE principal_id=? AND request_id=?', ("p", "r2"))),
                               lambda: [r.admission_id for r in store.admissions.match(principal_id="p", request_id="r2")])
    assert_same(legacy, backend)
    legacy, backend = run_both(path, lambda db: ids(_request_admission_rows(db, "rg")),
                               lambda: [r.admission_id for r in store.admissions.match(
                                   principal_id="api", request_id="rg", include_retired=True)])
    assert_same(legacy, backend)
    assert backend[1] == [v["retired"]["admission_id"]]
    # The (target, request) lookups of main's gateway/session_automation.py and hermes_cli/kanban_owner_recovery.py.
    assert [r.admission_id for r in store.admissions.match(session_id="s1", request_id="r1")] == [v["first"]["admission_id"]]
    assert store.admissions.match(session_id="s2", request_id="r1") == []


def test_settled_result_matches_legacy(seeded, legacy_store):
    from tests.persistence_sqlite._main_callers.session_results import admission_result, finish_result

    def seed(db, epoch):
        values = _running(db, epoch)
        settle(db, values["claimed"], result=_RESULT)
        return values

    path, store, v = seeded(seed)
    row = v["claimed"]
    for admission_id in (row["admission_id"], v["third"]["admission_id"], "absent"):
        legacy, backend = run_both(path, lambda db: admission_result(db, admission_id),
                                   lambda: store.admissions.settled_result(admission_id=admission_id))
        assert_same(legacy, backend, lambda found: None if found is None else found.result)
    from hermes_state_runtime import _row

    def precheck(db, epoch, generation):
        current, response = finish_result(db, epoch=epoch, row=dict(row, generation=generation), response="",
                                          outcome="completed")
        return _row(current), response

    for epoch, generation in ((EPOCH, row["generation"]), (EPOCH, row["generation"] + 1), (EPOCH + 1, row["generation"])):
        legacy, backend = run_both(
            path, lambda db: precheck(db, epoch, generation),
            lambda: store.admissions.settled_result(admission_id=row["admission_id"], epoch=epoch, generation=generation))
        assert_same(legacy, backend, lambda found: (record_dict(found.record), found.result["result"]["final_response"]))


def test_settled_result_answers_a_live_claim_without_a_result(seeded):
    _path, store, v = seeded(_running)
    row = v["claimed"]
    found = store.admissions.settled_result(admission_id=row["admission_id"], epoch=EPOCH, generation=row["generation"])
    assert found.result is None and found.record.status == "started"
    with pytest.raises(ValueError):
        store.admissions.settled_result(admission_id=row["admission_id"], epoch=EPOCH)


# The pending-target reads of main's gateway/run_runtime.py and gateway/session_api_turn.py.
_PENDING_TARGETS = {
    (frozenset({"unknown"}), None): "SELECT DISTINCT target_session_id FROM session_admissions WHERE status='unknown'",
    (frozenset({"queued", "unknown"}), None):
        "SELECT DISTINCT target_session_id FROM session_admissions WHERE status IN ('queued','unknown')",
    (frozenset({"queued"}), "api"):
        "SELECT DISTINCT target_session_id FROM session_admissions WHERE principal_id='api' AND status='queued'",
}


def _mixed_seed(db, epoch):
    values = _reads_seed(db, epoch)
    admit(db, "s2", "api-1", principal_id="api", payload={"text": "a"})
    recover(db, begin_epoch(db, "next"))
    return values


@pytest.mark.parametrize("shape", sorted(_PENDING_TARGETS, key=str))
def test_pending_targets_match_the_gateway_reads(seeded, legacy_store, shape):
    path, store, _ = seeded(_mixed_seed)
    statuses, principal_id = shape
    legacy, backend = run_both(path, lambda db: frozenset(row[0] for row in db._read_all(_PENDING_TARGETS[shape])),
                               lambda: store.admissions.pending_targets(statuses=statuses, principal_id=principal_id))
    assert_same(legacy, backend)
    assert backend[1]


def test_list_unknown_matches_the_boot_read(seeded, legacy_store):
    from hermes_state_runtime import _row
    path, store, _ = seeded(_mixed_seed)

    def legacy(db):  # main's read in gateway/session_unattended_unknown.py
        with db._read_ctx() as conn:
            return [_row(r) for r in conn.execute(
                "SELECT * FROM session_admissions WHERE status='unknown' ORDER BY seq").fetchall()]

    legacy_out, backend = run_both(path, legacy, lambda: store.admissions.list_unknown())
    assert_same(legacy_out, backend, _rows)
    assert backend[1]


def test_has_pending_matches_the_gateway_reads(seeded, legacy_store):
    path, store, _ = seeded(_mixed_seed)
    # main's idle-exit check in gateway/run_idle_exit.py
    legacy, backend = run_both(
        path, lambda db: bool(db._read_all("SELECT 1 FROM session_admissions WHERE status IN ('queued','started') LIMIT 1")),
        lambda: store.admissions.has_pending(statuses=frozenset({"queued", "started"})))
    assert_same(legacy, backend)
    # main's check in gateway/session_authorities.py
    for pair in (("s1", "s2"), ("absent", "nowhere")):
        legacy, backend = run_both(
            path, lambda db: bool(db._read_one('SELECT 1 FROM session_admissions WHERE target_session_id IN (?,?) LIMIT 1', pair)),
            lambda: store.admissions.has_pending(session_ids=pair))
        assert_same(legacy, backend)


@pytest.mark.parametrize("session_id,target", [("s1", "s1"), ("s2", "s2"), ("s2", "s1"), ("none", "none")])
def test_ledger_busy_matches_the_viewer_close_read(seeded, legacy_store, session_id, target):
    from tests.persistence_sqlite._main_callers.session_viewer_close import _busy
    path, store, _ = seeded(_running)
    legacy, backend = run_both(path, lambda db: _busy(authority(db), session_id, target),
                               lambda: store.admissions.ledger_busy(session_id, target))
    assert_same(legacy, backend)


def _webhook_seed(db, epoch):
    db.create_session("w1", source="webhook")
    db.create_session("w2", source="webhook")
    source = {"platform": "webhook", "chat_id": "chat-1"}
    first = admit(db, "w1", "d1", payload={"text": "hook", "native_text_v1": {"source": source}})
    admit(db, "w1", "d2", payload={"text": "auto", "native_text_v1": {"source": source, "automation": {"k": 1}}})
    settle(db, claim(db, "w1"))
    admit(db, "w2", "d3", payload={"text": "other", "native_text_v1": {"source": dict(source, chat_id="chat-2")}})
    return first


@pytest.mark.parametrize("chat_id", ["chat-1", "chat-2", "chat-3"])
def test_webhook_ingress_payloads_match_the_delivery_read(seeded, legacy_store, chat_id):
    path, store, _ = seeded(_webhook_seed)
    query = """SELECT payload_json FROM session_admissions
        WHERE json_extract(payload_json, '$.native_text_v1.source.chat_id')=?
          AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'
          AND json_extract(payload_json, '$.native_text_v1.automation') IS NULL"""  # main's gateway/platforms/webhook_delivery.py
    legacy, backend = run_both(path, lambda db: [json.loads(r["payload_json"]) for r in db._read_all(query, (chat_id,))],
                               lambda: store.admissions.webhook_ingress_payloads(chat_id=chat_id))
    assert_same(legacy, backend)


def test_unfinalized_webhook_admissions_match_the_recovery_read(seeded, legacy_store):
    path, store, first = seeded(_webhook_seed)
    query = """
        SELECT a.admission_id FROM sessions AS s
        JOIN session_admissions AS a ON a.target_session_id=s.id
        WHERE s.source='webhook' AND s.ended_at IS NULL
          AND a.status='terminal' AND a.generation IS NOT NULL
          AND json_extract(a.payload_json, '$.native_text_v1.source.platform')='webhook'
          AND json_extract(a.payload_json, '$.native_text_v1.automation') IS NULL
        """  # main's gateway/platforms/webhook_ingress.py
    legacy, backend = run_both(path, lambda db: [r["admission_id"] for r in db._read_all(query)],
                               lambda: store.admissions.unfinalized_webhook_admissions())
    assert_same(legacy, backend)
    assert backend[1] == [first["admission_id"]]


def test_terminal_ids_match_the_response_store_read(seeded, legacy_store):
    from hermes_state_terminal import ADMISSION_PREFIX
    path, store, v = seeded(_reads_seed)
    wanted = [v["first"]["admission_id"], v["second"]["admission_id"], v["retired"]["admission_id"], "absent"]

    def legacy(db):  # main's lookup in gateway/platforms/api_server_response_store.py
        terminal = []
        with db._read_ctx() as canon:
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                marks = ','.join('?' * len(chunk))
                terminal += [row[0] for row in canon.execute(
                    f"SELECT admission_id FROM session_admissions WHERE status='terminal' AND admission_id IN ({marks})", chunk)]
                terminal += [row[0][len(ADMISSION_PREFIX):] for row in canon.execute(
                    f'SELECT key FROM state_meta WHERE key IN ({marks})', [ADMISSION_PREFIX + i for i in chunk])]
        return frozenset(terminal)

    legacy_out, backend = run_both(path, legacy, lambda: store.admissions.terminal_ids(wanted))
    assert_same(legacy_out, backend)
    assert backend[1] == {v["second"]["admission_id"], v["retired"]["admission_id"]}


def _media_seed(db, epoch):
    db.create_session("m1", source="cli")
    db.create_session("m2", source="cli")
    reference = {"path": "/media/" + "a" * 64 + "/one.png", "sha256": "a" * 64}
    live = {"path": "/media/" + "b" * 64 + "/two.png", "sha256": "b" * 64}
    admit(db, "m1", "x1", payload={"text": "x", "attachments_v1": {"media": [reference]}})
    settle(db, claim(db, "m1"))
    admit(db, "m2", "x2", payload={"text": "y", "native_text_v1": {"media": [live]}})
    db.append_message("m1", role="assistant", content="see " + "a" * 64 + "/one.png")
    return {"reference": reference, "live": live}


def test_settled_media_matches_the_sweep_read(seeded, legacy_store):
    path, store, v = seeded(_media_seed)

    def legacy(db):  # main's sweep read in gateway/session_ingress_media.py
        with db._read_ctx() as conn:
            rows = conn.execute("""SELECT json_extract(payload_json, '$.attachments_v1.media', '$.native_text_v1.media')
                FROM session_admissions WHERE status='terminal' AND (json_type(payload_json, '$.attachments_v1.media')
                IS NOT NULL OR json_type(payload_json, '$.native_text_v1.media') IS NOT NULL)""").fetchall()
        return [(a or []) + (n or []) for a, n in (json.loads(r[0]) for r in rows)]

    legacy_out, backend = run_both(path, legacy, lambda: store.admissions.settled_media())
    assert_same(legacy_out, backend)
    assert backend[1] == [[v["reference"]]]


def test_history_mentions_match_legacy(seeded, legacy_store):
    from tests.persistence_sqlite._main_callers.session_ingress_media import _history_mentions
    path, store, v = seeded(_media_seed)
    references = [v["reference"], v["live"]]

    def legacy(db):
        with db._read_ctx() as conn:
            return conn.execute('SELECT coalesce(max(id), 0) FROM messages').fetchone()[0], frozenset(_history_mentions(conn, references))

    legacy_out, backend = run_both(path, legacy, lambda: store.admissions.history_mentions(references))
    assert_same(legacy_out, backend)
    assert backend[1][1] == {v["reference"]["path"]}


@pytest.mark.parametrize("retain_history,after_id", [(True, 0), (False, 0), (True, 10_000)])
def test_unheld_media_matches_legacy(seeded, legacy_store, monkeypatch, retain_history, after_id):
    from tests.persistence_sqlite._main_callers import session_ingress_media as ingress
    from pathlib import Path
    path, store, v = seeded(_media_seed)
    monkeypatch.setattr(ingress, "_media_root", lambda: Path("/media"))
    references = [v["reference"], v["live"], {"path": "/media/" + "c" * 64 + "/free.png", "sha256": "c" * 64}]

    def legacy(db):  # main's collection in gateway/session_ingress_media.py, without the owner's file checks and unlinks
        def collect(conn):
            held = ingress._held_media_paths(conn)
            if retain_history:
                held |= ingress._history_mentions(conn, references, after_id=after_id)
            return tuple(r for r in references if r["path"] not in held), frozenset(held)
        return db._execute_write(collect)

    legacy_out, backend = run_both(path, legacy, lambda: store.admissions.unheld_media(
        references, after_id=after_id, retain_history=retain_history, media_root="/media"))
    assert_same(legacy_out, backend, lambda found: (found.candidates, found.held))
    assert isinstance(backend[1], UnheldMedia)


def test_unheld_media_is_one_write_that_writes_nothing(seeded):
    path, store, v = seeded(_media_seed)
    before = dump(path)
    with counted_transactions(store.holder()) as counts:
        store.admissions.unheld_media([v["reference"]], after_id=0, retain_history=True, media_root="/media")
    assert counts == {"write": 1, "read": 0}
    assert dump(path) == before


def _candidate_seed(db, epoch):
    keys = ["gateway.media.retired.v1." + c * 64 for c in "ab"] + ["gateway.spillover.retired.v1.x"]
    db._execute_write(lambda conn: conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?)',
                                                    [(k, json.dumps({"path": k})) for k in keys]))
    return keys


def test_drop_retired_candidates_matches_legacy(seeded, legacy_store):
    path, store, keys = seeded(_candidate_seed)
    done = [(keys[0],), (keys[2],), ("absent",)]
    legacy, backend = run_both(  # main's deletes in hermes_state_media.py
        path, lambda db: db._execute_write(lambda conn: conn.executemany('DELETE FROM state_meta WHERE key=?', done)) and None,
        lambda: store.admissions.drop_retired_candidates([k for (k,) in done]))
    assert_same(legacy, backend)


def test_drop_retired_candidates_is_one_write_and_rolls_back(seeded):
    path, store, keys = seeded(_candidate_seed)
    call = lambda: store.admissions.drop_retired_candidates(keys)
    assert_rolls_back(path, store.holder(), call)
    with counted_transactions(store.holder()) as counts:
        call()
    assert counts == {"write": 1, "read": 0}


# ── API payload compaction ─────────────────────────────────────────────────────


_PIXELS = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _api_seed(db, epoch):
    digest = hashlib.sha256(_PIXELS).hexdigest()
    url = "data:image/png;base64," + base64.b64encode(_PIXELS).decode()
    content = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}]
    media = [{"path": f"/media/{digest}/api_{digest[:32]}.png", "sha256": digest}]
    ids = []
    for sid, request_id, history in (("a1", "req-1", [{"role": "user", "content": "x"}]), ("a2", "responses:r", [1]),
                                     ("a3", "req-3", None)):
        db.create_session(sid, source="api_server")
        row = admit(db, sid, request_id, principal_id="api", payload={
            "text": content, "api_turn_v1": {"media": media, "history": history, "settings": {}}})
        if sid != "a3":
            settle(db, claim(db, sid))
        ids.append(row["admission_id"])
    return ids


def test_api_payload_compaction_matches_the_startup_sweep(seeded, legacy_store):
    from tests.persistence_sqlite._main_callers.session_api_media import compact_settled_api_payloads
    path, store, _ids = seeded(_api_seed)

    def backend():
        return store.admissions.compact_api_payloads(admission_ids=store.admissions.api_compaction_candidates())

    legacy, after = run_both(path, compact_settled_api_payloads, backend)
    assert_same(legacy, after)
    assert after[1] == 2


@pytest.mark.parametrize("index", [0, 1, 2])
def test_api_payload_compaction_matches_the_settlement_pass(seeded, legacy_store, index):
    from tests.persistence_sqlite._main_callers.session_api_media import compact_settled_api_payloads
    path, store, ids = seeded(_api_seed)
    legacy, after = run_both(path, lambda db: compact_settled_api_payloads(db, ids[index]),
                             lambda: store.admissions.compact_api_payloads(admission_ids=[ids[index]]))
    assert_same(legacy, after)


def test_api_compaction_rewrites_the_row_the_write_reads(seeded, legacy_store):
    """A row that changed after the candidate scan is compacted from what it holds inside the write."""
    from tests.persistence_sqlite._main_callers.session_api_media import compact_settled_api_payloads
    path, store, ids = seeded(_api_seed)
    candidates = store.admissions.api_compaction_candidates()
    assert candidates == ids[:2]
    conn = sqlite3.connect(path)
    conn.execute("UPDATE session_admissions SET payload_json=json_set(payload_json, '$.text[0].text', 'changed') "
                 "WHERE admission_id=?", (ids[0],))
    conn.commit()
    conn.close()
    legacy, after = run_both(path, compact_settled_api_payloads,
                             lambda: store.admissions.compact_api_payloads(admission_ids=candidates))
    assert_same(legacy, after)
    assert after[1] == 2
    assert store.admissions.get(admission_id=ids[0]).payload["text"][0] == {"type": "text", "text": "changed"}
    assert store.admissions.compact_api_payloads(admission_ids=candidates) == 0
    assert store.admissions.api_compaction_candidates() == []


def test_compact_api_payloads_is_one_write_and_rolls_back(seeded):
    path, store, ids = seeded(_api_seed)
    assert_rolls_back(path, store.holder(), lambda: store.admissions.compact_api_payloads(admission_ids=ids))
    with counted_transactions(store.holder()) as counts:
        store.admissions.compact_api_payloads(admission_ids=ids)
    assert counts == {"write": 1, "read": 0}


# ── every read is one read context ─────────────────────────────────────────────


_READS = {
    "find": lambda a, v: a.find(principal_id="p", session_id="s1", request_id="r1"),
    "match": lambda a, v: a.match(principal_id="api", request_id="rg", include_retired=True),
    "list": lambda a, v: a.list(session_id="s1"),
    "get": lambda a, v: a.get(admission_id=v["retired"]["admission_id"]),
    "settled_result": lambda a, v: a.settled_result(admission_id=v["claimed"]["admission_id"], epoch=EPOCH,
                                                    generation=v["claimed"]["generation"]),
    "pending_targets": lambda a, v: a.pending_targets(statuses=frozenset({"queued"})),
    "list_unknown": lambda a, v: a.list_unknown(),
    "has_pending": lambda a, v: a.has_pending(session_ids=("s1",)),
    "ledger_busy": lambda a, v: a.ledger_busy("s1", "s1"),
    "webhook_ingress_payloads": lambda a, v: a.webhook_ingress_payloads(chat_id="c"),
    "unfinalized_webhook_admissions": lambda a, v: a.unfinalized_webhook_admissions(),
    "terminal_ids": lambda a, v: a.terminal_ids([v["first"]["admission_id"]] * 600),
    "api_compaction_candidates": lambda a, v: a.api_compaction_candidates(),
    "history_mentions": lambda a, v: a.history_mentions([{"path": "/m/" + "a" * 64 + "/f", "sha256": "a" * 64}]),
    "settled_media": lambda a, v: a.settled_media(),
}


@pytest.mark.parametrize("verb", sorted(_READS))
def test_every_read_is_one_read_context(seeded, verb):
    path, store, v = seeded(_reads_seed)
    before = dump(path)
    with counted_transactions(store.holder()) as counts:
        _READS[verb](store.admissions, v)
    assert counts == {"write": 0, "read": 1}
    assert dump(path) == before


def test_reads_keep_the_legacy_binding_refusals(seeded, legacy_store):
    from persistence import PermanentStoreError
    path, store, _ = seeded(_reads_seed)
    legacy, backend = run_both(path, lambda db: __import__("hermes_state_runtime").get_session_admission(db, admission_id=[]),
                               lambda: store.admissions.get(admission_id=[]))
    assert_same(legacy, backend)
    assert isinstance(backend[1], PermanentStoreError)
    assert outcome(lambda: store.admissions.list(session_id=memoryview(b"123")[::2]))[1].__class__ is BufferError
