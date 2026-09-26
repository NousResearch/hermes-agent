from __future__ import annotations

from claude_selfimprove import paths, pipeline, store


def _seed_transcript(sandbox, session_id, text):
    sandbox.write_transcript(
        "claude", "proj-a", session_id,
        [{"type": "user", "sessionId": session_id, "message": {"role": "user", "content": text}}],
    )


def test_seed_marks_existing_files_fully_read_without_mining(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    result = pipeline.seed()
    assert result.error is None
    assert result.files_seeded == 1
    assert result.already_seeded is False

    with store.CandidateStore() as db:
        assert db.all() == []  # nothing mined from the seeded history


def test_scan_after_seed_only_sees_new_content(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    pipeline.seed()

    def runner(prompt, *, model, provider, timeout):
        raise AssertionError("classifier must not be called - nothing new was scanned")

    scan_result = pipeline.scan(llm_runner=runner)
    assert scan_result.turns_scanned == 0
    assert scan_result.candidates_flagged == 0


def test_scan_after_seed_sees_content_appended_later(sandbox):
    path = _path = sandbox.write_transcript(
        "claude", "proj-a", "sess-1",
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "old content, pre-seed"}}],
    )
    pipeline.seed()

    import json
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "Never force push to main again."}}) + "\n")

    scan_result = pipeline.scan()
    assert scan_result.turns_scanned == 1
    assert scan_result.candidates_flagged == 1


def test_seed_refuses_to_run_twice_without_force(sandbox):
    _seed_transcript(sandbox, "sess-1", "hello")
    r1 = pipeline.seed()
    assert r1.already_seeded is False

    r2 = pipeline.seed()
    assert r2.already_seeded is True
    assert r2.files_seeded == 0


def test_seed_force_reseeds(sandbox):
    _seed_transcript(sandbox, "sess-1", "hello")
    pipeline.seed()
    result = pipeline.seed(force=True)
    assert result.already_seeded is False
    assert result.files_seeded == 1


def test_seed_dry_run_does_not_persist(sandbox):
    _seed_transcript(sandbox, "sess-1", "Never force push to main, that's a hard rule.")
    result = pipeline.seed(dry_run=True)
    assert result.files_seeded == 1
    assert not paths.checkpoints_path().exists()


def test_seed_skips_cleanly_when_lock_held(sandbox):
    paths.ensure_state_dirs()
    paths.lock_dir_path().mkdir(parents=True)
    result = pipeline.seed()
    assert result.skipped_busy is True
