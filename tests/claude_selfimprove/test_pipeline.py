from __future__ import annotations

import json
import time

from claude_selfimprove import notify, paths, pipeline, scanner, store, writers


def _classify_runner(canonical_key="never-force-push-main", scope="global", target_kind="claude_md_block", confidence=0.9):
    def runner(prompt, *, model, provider, timeout):
        n = prompt.count("[category:")
        items = [
            {
                "index": i, "is_real_lesson": True, "canonical_key": canonical_key,
                "scope": scope, "target_kind": target_kind,
                "title": "Never force push main", "body": "Never force push to the main branch.",
                "confidence": confidence,
            }
            for i in range(1, n + 1)
        ]
        return True, json.dumps(items)

    return runner


def _no_conflict_runner():
    def runner(prompt, *, model, provider, timeout):
        n = prompt.count("\n") and prompt.count(". ")  # unused, just build from eligible count
        import re
        indices = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
        # crude: answer "no conflict" for every numbered "NEW RULES" entry found
        items = [{"index": int(i), "conflicts": False, "reason": ""} for i in indices]
        return True, json.dumps(items)

    return runner


def _seed_transcript(sandbox, session_id, text):
    sandbox.write_transcript(
        "claude", "proj-a", session_id,
        [{"type": "user", "sessionId": session_id, "message": {"role": "user", "content": text}}],
    )


def test_scan_flags_and_classifies_a_candidate(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    result = pipeline.scan(llm_runner=_classify_runner())
    assert result.error is None
    assert result.turns_scanned == 1
    assert result.candidates_flagged == 1
    assert result.candidates_classified == 1

    with store.CandidateStore() as db:
        rows = db.all()
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["evidence_count"] == 1


def test_scan_dry_run_does_not_advance_checkpoint_or_store(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    result = pipeline.scan(llm_runner=_classify_runner(), dry_run=True)
    assert result.candidates_flagged == 1

    with store.CandidateStore() as db:
        assert db.all() == []

    # Running again (not dry-run) must still see the same turn, proving the
    # checkpoint never advanced during the dry run.
    result2 = pipeline.scan(llm_runner=_classify_runner())
    assert result2.turns_scanned == 1


def test_scan_second_run_is_incremental(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    pipeline.scan(llm_runner=_classify_runner())
    result2 = pipeline.scan(llm_runner=_classify_runner())
    assert result2.turns_scanned == 0
    assert result2.candidates_flagged == 0


def test_explicit_instruction_applies_after_a_single_occurrence(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    scan_result = pipeline.scan(llm_runner=_classify_runner(confidence=0.9))
    assert scan_result.candidates_classified == 1

    consolidate_result = pipeline.consolidate(conflict_runner=_no_conflict_runner())
    assert consolidate_result.error is None
    assert len(consolidate_result.applied) == 1

    content = paths.claude_md_path().read_text()
    assert "Never force push to the main branch." in content

    with store.CandidateStore() as db:
        row = db.all()[0]
    assert row["status"] == "applied"
    assert row["target_kind"] == "claude_md_block"

    queue = paths.self_improvement_queue_path().read_text().splitlines()
    events = [json.loads(l) for l in queue]
    assert any(e["action"] == "apply" for e in events)


def test_repeated_fix_needs_three_sessions_and_two_tasks(sandbox):
    runner = _classify_runner(canonical_key="always-check-logs", target_kind="rule")

    def mixed_runner(prompt, *, model, provider, timeout):
        n = prompt.count("[category:")
        items = [
            {
                "index": i, "is_real_lesson": True, "canonical_key": "always-check-logs",
                "scope": "global", "target_kind": "rule",
                "title": "Always check logs first", "body": "Always check the logs before guessing at a fix.",
                "confidence": 0.9,
            }
            for i in range(1, n + 1)
        ]
        return True, json.dumps(items)

    # Session 1: one occurrence.
    sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [
            {"type": "assistant", "sessionId": "sess-1", "message": {"role": "assistant", "content": "Checked the logs and found the root cause."}},
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "That fixed it, works now."}},
        ],
    )
    pipeline.scan(llm_runner=mixed_runner)
    r1 = pipeline.consolidate(conflict_runner=_no_conflict_runner())
    assert r1.applied == []
    assert r1.not_yet_eligible == 1

    # Session 2: second occurrence, different session.
    sandbox.write_transcript(
        "claude", "proj-a", "sess-2",
        [
            {"type": "assistant", "sessionId": "sess-2", "message": {"role": "assistant", "content": "Checked the logs and found the root cause."}},
            {"type": "user", "sessionId": "sess-2", "message": {"role": "user", "content": "That fixed it, works now."}},
        ],
    )
    pipeline.scan(llm_runner=mixed_runner)
    r2 = pipeline.consolidate(conflict_runner=_no_conflict_runner())
    assert r2.applied == []  # 2 sessions, 2 tasks - still short of 3 sessions

    # Session 3: third occurrence, different session -> now eligible.
    sandbox.write_transcript(
        "claude", "proj-a", "sess-3",
        [
            {"type": "assistant", "sessionId": "sess-3", "message": {"role": "assistant", "content": "Checked the logs and found the root cause."}},
            {"type": "user", "sessionId": "sess-3", "message": {"role": "user", "content": "That fixed it, works now."}},
        ],
    )
    pipeline.scan(llm_runner=mixed_runner)
    r3 = pipeline.consolidate(conflict_runner=_no_conflict_runner())
    assert len(r3.applied) == 1

    rule_path = paths.rules_dir() / "always-check-logs.md"
    assert rule_path.exists()
    assert "Always check the logs" in rule_path.read_text()


def test_conflict_blocks_application(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    pipeline.scan(llm_runner=_classify_runner())

    def conflicting_runner(prompt, *, model, provider, timeout):
        import re
        indices = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
        items = [{"index": int(i), "conflicts": True, "reason": "contradicts an existing rule"} for i in indices]
        return True, json.dumps(items)

    result = pipeline.consolidate(conflict_runner=conflicting_runner)
    assert result.applied == []
    assert len(result.conflict_blocked) == 1

    with store.CandidateStore() as db:
        row = db.all()[0]
    assert row["status"] == "conflict_blocked"
    assert not paths.claude_md_path().exists()

    events = [json.loads(l) for l in paths.self_improvement_queue_path().read_text().splitlines()]
    assert any(e["action"] == "conflict_blocked" for e in events)


def test_repo_scope_is_excluded_and_never_applied(sandbox):
    def repo_runner(prompt, *, model, provider, timeout):
        n = prompt.count("[category:")
        items = [
            {
                "index": i, "is_real_lesson": True, "canonical_key": "this-repos-weird-quirk",
                "scope": "repo", "target_kind": "rule", "title": "Repo quirk", "body": "Only this repo needs this.",
                "confidence": 0.95,
            }
            for i in range(1, n + 1)
        ]
        return True, json.dumps(items)

    _seed_transcript(sandbox, "sess-1", "Never do the weird repo-specific thing here.")
    pipeline.scan(llm_runner=repo_runner)
    result = pipeline.consolidate(conflict_runner=_no_conflict_runner())
    assert result.applied == []
    assert len(result.excluded_repo_scope) == 1

    with store.CandidateStore() as db:
        row = db.all()[0]
    assert row["status"] == "excluded_repo_scope"
    assert not (paths.rules_dir() / "this-repos-weird-quirk.md").exists()


def test_consolidate_dry_run_does_not_mutate_anything(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    pipeline.scan(llm_runner=_classify_runner())
    result = pipeline.consolidate(conflict_runner=_no_conflict_runner(), dry_run=True)
    assert len(result.applied) == 1  # "would apply"

    with store.CandidateStore() as db:
        row = db.all()[0]
    assert row["status"] == "pending"  # unchanged
    assert not paths.claude_md_path().exists()


def test_scan_skips_cleanly_when_lock_is_held(sandbox):
    paths.ensure_state_dirs()
    paths.lock_dir_path().mkdir(parents=True)
    result = pipeline.scan()
    assert result.skipped_busy is True
    assert result.error is None


def test_consolidate_skips_cleanly_when_lock_is_held(sandbox):
    paths.ensure_state_dirs()
    paths.lock_dir_path().mkdir(parents=True)
    result = pipeline.consolidate()
    assert result.skipped_busy is True


def test_stale_lock_is_reclaimed(sandbox):
    paths.ensure_state_dirs()
    lock = paths.lock_dir_path()
    lock.mkdir(parents=True)
    old_time = time.time() - 1000
    import os
    os.utime(lock, (old_time, old_time))

    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    result = pipeline.scan(llm_runner=_classify_runner())
    assert result.skipped_busy is False
    assert result.error is None


def test_scan_never_raises_on_llm_exception(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")

    def broken_runner(prompt, *, model, provider, timeout):
        raise RuntimeError("simulated crash")

    # classify_batch calls the runner directly - a raising runner is a bug
    # in the runner, not something llm.py guards. scan() itself must not
    # propagate it uncaught.
    result = pipeline.scan(llm_runner=broken_runner)
    assert result.error is not None
    with store.CandidateStore() as db:
        assert db.all() == []  # nothing corrupted


def test_scan_crash_after_db_write_before_checkpoint_save_does_not_double_count(sandbox, monkeypatch):
    # Real failure mode for an unattended nightly job: the database writes
    # for this run commit (each upsert_occurrence commits immediately), then
    # the process is killed - or, as simulated here, checkpoints.save()
    # itself raises - before the transcript checkpoint advances. The next
    # scan re-reads the exact same unconsumed bytes and re-classifies them.
    # That replay must land as the SAME evidence, not new evidence, or a
    # crash alone could push a candidate over its threshold.
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")

    real_save = scanner.CheckpointStore.save
    call_count = {"n": 0}

    def flaky_save(self):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated crash between db commit and checkpoint save")
        return real_save(self)

    monkeypatch.setattr(scanner.CheckpointStore, "save", flaky_save)

    first = pipeline.scan(llm_runner=_classify_runner())
    assert first.error is not None  # the simulated crash surfaced

    with store.CandidateStore() as db:
        rows = db.all()
    assert len(rows) == 1
    assert rows[0]["evidence_count"] == 1  # the pre-crash db write did land

    # Retry: the checkpoint never advanced, so the exact same turn is read
    # and reclassified again.
    monkeypatch.setattr(scanner.CheckpointStore, "save", real_save)
    second = pipeline.scan(llm_runner=_classify_runner())
    assert second.error is None
    assert second.turns_scanned == 1  # proves the same bytes were re-read

    with store.CandidateStore() as db:
        rows = db.all()
    assert len(rows) == 1
    assert rows[0]["evidence_count"] == 1  # NOT 2 - replay must not double-count
    assert rows[0]["occurrence_count"] == 1

    # A third, truly independent occurrence (different session) must still
    # count normally - the fix must not have broken real accumulation.
    sandbox.write_transcript(
        "claude", "proj-a", "sess-2",
        [{"type": "user", "sessionId": "sess-2", "message": {"role": "user", "content": "Never force push to main, that's a hard rule."}}],
    )
    pipeline.scan(llm_runner=_classify_runner())
    with store.CandidateStore() as db:
        rows = db.all()
    assert rows[0]["evidence_count"] == 2
    assert rows[0]["session_ids"] == ["sess-1", "sess-2"]
