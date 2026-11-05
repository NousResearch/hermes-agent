"""``AdmissionRepository`` on SQLite: each verb is one transaction, a failure inside a write rolls every table back, and
each read opens one read context. Verbs that reach a conversation body are in ``test_admission_conversation.py``; the
hosted member check is in ``test_admission_hosted.py``."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

from tests.persistence_sqlite.authority._support import counted_transactions, dump
from tests.persistence_sqlite.authority._admission_support import (
    EPOCH, admit, assert_rolls_back, begin_epoch, claim, register, settle, steer,
)


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


def _retarget_seed(db, epoch):
    db.create_session("owner", source="cli")
    db.create_session("tip", source="cli")
    return [admit(db, "owner", f"r{i}", principal_id="peer",
                  payload={"text": f"t{i}", "local_automation_v1": {"target": target}})
            for i, target in enumerate(("stale-a", "kept", "stale-b"))]


def _next_owner(db, epoch):
    values = _running(db, epoch)
    values["next"] = begin_epoch(db, "next")
    return values


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


# ── the reads' seed ────────────────────────────────────────────────────────────


def _reads_seed(db, epoch):
    values = _running(db, epoch)
    db.create_session("gone", source="cli")
    done = admit(db, "gone", "rg", principal_id="api", payload={"text": "gone"})
    settle(db, claim(db, "gone"), result=_RESULT)
    db.delete_session("gone")
    values["retired"] = done
    return values


# ── settled results and media retirement ───────────────────────────────────────


def test_settled_result_answers_a_live_claim_without_a_result(seeded):
    _path, store, v = seeded(_running)
    row = v["claimed"]
    found = store.admissions.settled_result(admission_id=row["admission_id"], epoch=EPOCH, generation=row["generation"])
    assert found.result is None and found.record.status == "started"
    with pytest.raises(ValueError):
        store.admissions.settled_result(admission_id=row["admission_id"], epoch=EPOCH)


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
