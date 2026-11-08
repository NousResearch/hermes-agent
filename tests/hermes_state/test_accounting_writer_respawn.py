"""A respawned writer must wait for the caller-owned accounting receipt to settle."""
import threading

from hermes_state import SessionDB


def test_respawn_waits_for_caller_drain_and_preserves_accounting_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", "cli")
    original = db._apply_token_batch
    # Model a proven unapplied refusal: accepted work stays queued as the writer exits.
    monkeypatch.setattr(db, "_apply_token_batch", lambda batch: batch)
    db.queue_token_counts("s", input_tokens=3, api_call_count=1, cost_status="estimated",
                          billing_provider="first", model="first-model")
    with db._token_queue_cond:
        writer = db._token_writer_thread
    if writer is not None:
        writer.join(5)
        assert not writer.is_alive()
    entered, release, later = threading.Event(), threading.Event(), threading.Event()
    writer_observed = threading.Event()
    errors = []
    original_wait = db._token_queue_cond.wait

    def observed_wait(timeout=None):
        if threading.current_thread().name == "session-db-token-writer":
            writer_observed.set()
        return original_wait(timeout)

    def paused(batch):
        if batch[0][1]["input_tokens"] == 3:
            entered.set()
            assert release.wait(10), "caller drain was not released"
        else:
            later.set()
            writer_observed.set()
        return original(batch)

    def drain():
        try:
            db.flush_token_counts(timeout=10)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(db, "_apply_token_batch", paused)
    monkeypatch.setattr(db._token_queue_cond, "wait", observed_wait)
    caller = threading.Thread(target=drain)
    caller.start()
    try:
        assert entered.wait(5), "caller did not claim the retained delta"
        db.queue_token_counts("s", input_tokens=7, api_call_count=1, cost_status="confirmed",
                              billing_provider="second", model="second-model")
        assert writer_observed.wait(5), "respawned writer neither waited nor applied"
        assert not later.is_set(), "respawned writer overtook the caller-owned batch"
        assert db.has_pending_token_counts()
        assert db._token_pending_marker_path.exists()
    finally:
        release.set()
        caller.join(10)
        try:
            assert not caller.is_alive()
            assert not errors, errors
            assert db.flush_token_counts(timeout=10)
            row = db.get_session("s")
            assert row["input_tokens"] == 10
            assert row["api_call_count"] == 2
            assert row["billing_provider"] == "first"
            assert row["cost_status"] == "confirmed"
            assert not db.has_pending_token_counts()
        finally:
            db.close()
