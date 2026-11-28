"""A settled result sheds its transcript copies once it can no longer answer an exact retry.

The settlement writes one result per settled turn under ``gateway.admission.result.v1.<id>`` and used
to keep every one of them, transcript copies included, for the life of its session: only a *retired*
session's tombstones were reclaimed, so one long-lived chat carried one full result per turn forever
(751 rows / 127 MB on the reference install, 96% of it the per-turn ``messages`` copies).

``RETRY_HORIZON_S`` already bounds which retries get a byte-exact reply. Past it the row is trimmed,
**not deleted**: ``finish_result_precheck`` and the hosted-task recovery path read "terminal admission
with no stored result" as a storage fault, and the bot receipt path degrades a settled turn to
``ambiguous`` without a ``final_response``. These tests pin the stamp, the trim, and the invariants
those readers depend on.
"""
import json
import time

import pytest

from gateway.session_authority import admission_row
from gateway.session_persistence import authority_persistence
from gateway.session_results import admission_result, finish_result
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.sqlite.authority._admission_owner import finish_result_precheck
from persistence.sqlite.conversation._store.terminal import settled_key
from session_retention import run_settled_result_trim
from tests.fixtures.storage_holder import admission_result_key, raw_holder
from tests.fixtures.storage_seed import seed_session


def _settle(owner, request_id="r1", history=None):
    seed_session(owner.persistence, "s", "api_server")
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id="api", session_id="s",
                                      request_id=request_id, payload={"text": "ask"})
    row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id="s"))
    messages = history or [{"role": "assistant", "content": "done"}]
    finish_result(authority_persistence(owner), epoch=owner.epoch, row=row, response="done", outcome="completed",
                  result={"result": {"final_response": "done", "messages": messages,
                                     "completed": True, "model": "m", "estimated_cost_usd": .25,
                                     "_turn_user_row": {"role": "user", "content": "ask"}},
                          "usage": {"input_tokens": 3}})
    return row


def _raw(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key=?",
                           (admission_result_key(admission_id),)).fetchone()
    return None if row is None else row[0]


def _stamp_of(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key=?",
                           (settled_key(admission_id),)).fetchone()
    return None if row is None else int(row[0])


def _backdate(owner, admission_id, seconds):
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "UPDATE state_meta SET value=? WHERE key=?",
        (str(int(time.time()) - seconds), settled_key(admission_id))))


def _admission_status(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        return conn.execute("SELECT status FROM session_admissions WHERE admission_id=?",
                            (admission_id,)).fetchone()[0]


def test_settlement_stamps_beside_the_result_without_touching_it(owner):
    row = _settle(owner)
    assert _stamp_of(owner, row["admission_id"]) == pytest.approx(time.time(), abs=60)
    # The stamp is a sibling key: the stored-result payload keeps exactly what it stored.
    raw = json.loads(_raw(owner, row["admission_id"]))
    assert "settled_at" not in raw
    assert admission_result(authority_persistence(owner), row["admission_id"])["result"]["messages"] == [
        {"role": "assistant", "content": "done"}]


def test_past_horizon_trims_transcript_copies_and_keeps_every_reader_field(owner):
    row = _settle(owner, history=[{"role": "assistant", "content": "done"},
                                  {"role": "user", "content": "x" * 4000}])
    admission_id = row["admission_id"]
    _backdate(owner, admission_id, RETRY_HORIZON_S + 60)
    before = len(_raw(owner, admission_id))
    assert owner.persistence.maintenance.trim_settled_results() == 1
    after_raw = _raw(owner, admission_id)
    assert before > len(after_raw)                          # the copies were the weight
    after = json.loads(after_raw)
    assert after["result"]["messages"] == []                 # transcript copies gone
    assert "_turn_user_row" not in after["result"]
    assert after["result"]["final_response"] == "done"       # what receipts report
    assert after["usage"] == {"input_tokens": 3}             # what runs/ledger report
    assert after["result"]["completed"] is True
    assert after["result"]["model"] == "m"
    # The row still exists: "terminal admission, no stored result" is a storage fault to two readers.
    assert admission_result(authority_persistence(owner), admission_id) is not None
    assert _admission_status(owner, admission_id) == "terminal"


def test_a_trimmed_admission_still_passes_the_settlement_precheck(owner):
    """``finish_result`` on an already-settled admission must not turn into ``storage_unavailable``."""
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1
    stored, result = finish_result_precheck(raw_holder(owner.persistence), epoch=owner.epoch,
                                            admission_id=row["admission_id"],
                                            generation=row["generation"])
    assert stored["status"] == "terminal" and result["result"]["final_response"] == "done"


def test_result_inside_horizon_is_not_touched(owner):
    row = _settle(owner)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert admission_result(authority_persistence(owner), row["admission_id"])["result"]["messages"] == [
        {"role": "assistant", "content": "done"}]


def test_legacy_row_without_a_stamp_is_stamped_not_trimmed(owner):
    row = _settle(owner)
    raw_holder(owner.persistence)._execute_write(
        lambda conn: conn.execute("DELETE FROM state_meta WHERE key=?", (settled_key(row["admission_id"]),)))
    assert _stamp_of(owner, row["admission_id"]) is None
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert _stamp_of(owner, row["admission_id"]) is not None
    assert json.loads(_raw(owner, row["admission_id"]))["result"]["messages"]


def test_a_settled_but_empty_reply_keeps_its_transcript(owner):
    """The Bot delivery fallback rebuilds an empty reply from ``messages``; keep that copy."""
    seed_session(owner.persistence, "s", "api_server")
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id="api", session_id="s",
                                      request_id="empty", payload={"text": "ask"})
    row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id="s"))
    finish_result(authority_persistence(owner), epoch=owner.epoch, row=row, response="", outcome="completed",
                  result={"result": {"final_response": "", "messages": [
                      {"role": "assistant", "content": "the only copy of the reply"}]},
                          "usage": {"input_tokens": 3}})
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    saved = admission_result(authority_persistence(owner), row["admission_id"])
    assert saved["result"]["messages"] == [{"role": "assistant", "content": "the only copy of the reply"}]


def test_an_unreadable_row_does_not_block_the_pass(owner):
    row = _settle(owner)
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "INSERT INTO state_meta(key,value) VALUES(?,?)", ("gateway.admission.result.v1.broken", "{not json")))
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "INSERT INTO state_meta(key,value) VALUES(?,?)",
        (settled_key("broken"), str(int(time.time()) - RETRY_HORIZON_S - 60))))
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1


def test_trim_is_idempotent(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1
    assert owner.persistence.maintenance.trim_settled_results() == 0


def test_retirement_compaction_does_not_extend_a_row_life(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    stamped = _stamp_of(owner, row["admission_id"])
    # A retirement's compaction re-encodes the result row; the stamp must not move with it.
    from persistence.sqlite.conversation._store.retirement import _compact_result
    raw_holder(owner.persistence)._execute_write(lambda conn: _compact_result(conn, row["admission_id"]))
    assert _stamp_of(owner, row["admission_id"]) == stamped


def test_trim_throttles_and_needs_no_auto_prune(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    first = run_settled_result_trim(owner.persistence, None, vacuum=False)
    assert first["trimmed"] == 1 and not first["skipped"]
    again = run_settled_result_trim(owner.persistence, None, vacuum=False)
    assert again["skipped"] and again["trimmed"] == 0
    assert json.loads(_raw(owner, row["admission_id"]))["result"]["messages"] == []
