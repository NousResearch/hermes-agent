"""An agent built with an outside ``session_db=SessionDB(...)`` keeps the compressor's conflict handling.

SessionDB raises its own conflict classes (``hermes_state_errors``); the compaction and prune commit paths catch
the storage interface's. A lost compare-and-swap still discards the micro-compaction pass, and a prune row the
store cannot name still falls back to the full rewrite.
"""
import hermes_state
import hermes_state_errors as legacy
from agent.context_compressor import _DB_PERSISTED_MARKER, ContextCompressor


def _bound_compressor(tmp_path):
    db = hermes_state.SessionDB(tmp_path / "state.db")
    db.create_session("s1", "cli")
    cc = ContextCompressor(model="test-model", threshold_percent=0.75, protect_first_n=1, protect_last_n=2,
                           quiet_mode=True, config_context_length=40960, provider="test")
    cc.bind_session_state(db, "s1")
    return db, cc


def test_a_lost_held_row_cas_discards_the_micro_compaction_pass(tmp_path, monkeypatch):
    db, cc = _bound_compressor(tmp_path)

    def rewritten(*_args, **_kwargs):
        raise legacy.HeldRowRewrittenError("held row 7 rewritten in place before the commit")
    monkeypatch.setattr(db, "archive_and_compact", rewritten)
    try:
        compacted = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "[micro summary]"}]
        assert cc._sync_micro_compact_to_db(compacted) is False
    finally:
        db.close()


def test_an_unresolved_prune_row_falls_back_to_the_full_rewrite(tmp_path, monkeypatch):
    db, cc = _bound_compressor(tmp_path)

    def unresolved(*_args, **_kwargs):
        raise legacy.PruneRowUnresolvedError("2 live rows match a pruned tool message")
    monkeypatch.setattr(db, "rewrite_pruned_rows", unresolved)
    try:
        held = [{"role": "tool", "content": "x" * 10, "tool_call_id": "t", _DB_PERSISTED_MARKER: True}]
        pruned = [{"role": "tool", "content": "[pruned]", "tool_call_id": "t", _DB_PERSISTED_MARKER: True}]
        assert cc._rewrite_pruned_rows_in_place(db, "s1", held, pruned, {}) is None
    finally:
        db.close()
