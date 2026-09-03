from __future__ import annotations

from claude_selfimprove.store import CandidateStore


def _occ(store, session_id="s1", confidence=0.6, canonical_key="always-run-tests", source_hash_value=None):
    return store.upsert_occurrence(
        canonical_key=canonical_key,
        category="repeated_fix",
        scope="global",
        target_kind="rule",
        target_path=None,
        title="Always run tests before committing",
        body="Always run the test suite before committing a change.",
        confidence=confidence,
        session_id=session_id,
        source_hash_value=source_hash_value or f"hash-{session_id}",
    )


def test_first_occurrence_inserts_pending_row(sandbox):
    with CandidateStore() as store:
        cid = _occ(store)
        row = store.get(cid)
        assert row["status"] == "pending"
        assert row["evidence_count"] == 1
        assert row["occurrence_count"] == 1
        assert row["session_ids"] == ["s1"]


def test_repeated_occurrence_with_distinct_evidence_merges_and_counts_each(sandbox):
    with CandidateStore() as store:
        # Three genuinely distinct occurrences: different source hashes,
        # the second two sharing a session (one session can legitimately
        # contribute more than one distinct task occurrence).
        cid1 = _occ(store, session_id="s1", source_hash_value="hash-a")
        cid2 = _occ(store, session_id="s2", source_hash_value="hash-b")
        cid3 = _occ(store, session_id="s2", source_hash_value="hash-c")

        assert cid1 == cid2 == cid3
        row = store.get(cid1)
        assert row["evidence_count"] == 3
        assert row["occurrence_count"] == 3
        assert row["session_ids"] == ["s1", "s2"]


def test_replaying_the_identical_source_hash_does_not_double_count(sandbox):
    # This is the crash/retry scenario: a scan that dies after this exact
    # write commits, but before the transcript checkpoint advances, re-reads
    # the same bytes next run and calls upsert_occurrence again with the
    # SAME source_hash_value for the SAME session. That must not inflate
    # evidence_count/occurrence_count - it is one real event, not two.
    with CandidateStore() as store:
        cid1 = _occ(store, session_id="s1", source_hash_value="hash-a")
        cid2 = _occ(store, session_id="s1", source_hash_value="hash-a")  # replay
        cid3 = _occ(store, session_id="s1", source_hash_value="hash-a")  # replay again

        assert cid1 == cid2 == cid3
        row = store.get(cid1)
        assert row["evidence_count"] == 1
        assert row["occurrence_count"] == 1
        assert row["session_ids"] == ["s1"]


def test_replay_does_not_block_a_later_genuinely_new_occurrence(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1", source_hash_value="hash-a")
        _occ(store, session_id="s1", source_hash_value="hash-a")  # replay, no-op on counters
        _occ(store, session_id="s2", source_hash_value="hash-b")  # genuinely new

        row = store.get(cid)
        assert row["evidence_count"] == 2
        assert row["occurrence_count"] == 2
        assert row["session_ids"] == ["s1", "s2"]


def test_replay_still_refreshes_confidence_and_timestamp(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1", source_hash_value="hash-a", confidence=0.3)
        row_before = store.get(cid)
        _occ(store, session_id="s1", source_hash_value="hash-a", confidence=0.95)  # replay, higher confidence
        row_after = store.get(cid)

        assert row_after["confidence"] == 0.95  # still refreshed
        assert row_after["evidence_count"] == 1  # but not double-counted
        assert row_after["updated_at"] >= row_before["updated_at"]


def test_confidence_takes_the_max_seen(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1", confidence=0.4)
        _occ(store, session_id="s2", confidence=0.9)
        row = store.get(cid)
        assert row["confidence"] == 0.9


def test_applied_candidate_ignores_further_occurrences(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1")
        store.set_status(cid, "applied")
        _occ(store, session_id="s2")
        row = store.get(cid)
        assert row["status"] == "applied"
        assert row["evidence_count"] == 1  # unchanged


def test_rejected_candidate_ignores_further_occurrences(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1")
        store.set_status(cid, "rejected", reason="not actually a rule George wants")
        _occ(store, session_id="s2")
        row = store.get(cid)
        assert row["status"] == "rejected"
        assert row["evidence_count"] == 1
        assert row["reject_reason"] == "not actually a rule George wants"


def test_clear_rejection_reopens_for_future_occurrences(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1")
        store.set_status(cid, "rejected", reason="too soon")
        store.clear_rejection(cid)
        _occ(store, session_id="s2")
        row = store.get(cid)
        assert row["status"] == "pending"
        assert row["evidence_count"] == 2


def test_conflict_blocked_still_merges_evidence(sandbox):
    with CandidateStore() as store:
        cid = _occ(store, session_id="s1")
        store.set_status(cid, "conflict_blocked", reason="contradicts existing rule X")
        _occ(store, session_id="s2")
        row = store.get(cid)
        assert row["status"] == "conflict_blocked"
        assert row["evidence_count"] == 2


def test_distinct_canonical_keys_do_not_collide(sandbox):
    with CandidateStore() as store:
        cid1 = _occ(store, canonical_key="rule-a")
        cid2 = _occ(store, canonical_key="rule-b")
        assert cid1 != cid2
        assert len(store.all()) == 2


def test_list_pending_global_excludes_repo_scope_and_other_statuses(sandbox):
    with CandidateStore() as store:
        global_cid = store.upsert_occurrence(
            canonical_key="global-rule", category="explicit_instruction", scope="global",
            target_kind="rule", target_path=None, title="t", body="b",
            confidence=0.9, session_id="s1", source_hash_value="h1",
        )
        store.upsert_occurrence(
            canonical_key="repo-rule", category="explicit_instruction", scope="repo",
            target_kind="rule", target_path=None, title="t2", body="b2",
            confidence=0.9, session_id="s1", source_hash_value="h2",
        )
        applied_cid = store.upsert_occurrence(
            canonical_key="already-applied", category="explicit_instruction", scope="global",
            target_kind="rule", target_path=None, title="t3", body="b3",
            confidence=0.9, session_id="s1", source_hash_value="h3",
        )
        store.set_status(applied_cid, "applied")

        pending_global = store.list_pending_global()
        assert [r["id"] for r in pending_global] == [global_cid]


def test_counts_by_status(sandbox):
    with CandidateStore() as store:
        cid = _occ(store)
        store.set_status(cid, "applied")
        _occ(store, canonical_key="other-rule")
        counts = store.counts_by_status()
        assert counts == {"applied": 1, "pending": 1}


def test_candidate_id_is_deterministic_and_scope_sensitive():
    from claude_selfimprove.store import candidate_id

    a = candidate_id("same-key", "global")
    b = candidate_id("same-key", "global")
    c = candidate_id("same-key", "repo")
    assert a == b
    assert a != c


def test_db_file_has_owner_only_permissions_on_creation(sandbox):
    import stat

    with CandidateStore() as store:
        _occ(store)
    mode = stat.S_IMODE(store.db_path.stat().st_mode)
    assert mode == 0o600


def test_db_file_permissions_are_repaired_on_reopen(sandbox):
    import os
    import stat

    with CandidateStore() as store:
        db_path = store.db_path
    os.chmod(db_path, 0o644)  # simulate a file created before this fix

    with CandidateStore(db_path) as store2:
        _occ(store2)
    mode = stat.S_IMODE(db_path.stat().st_mode)
    assert mode == 0o600
