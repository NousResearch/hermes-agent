"""Dup-on-replay fix: post-compaction restart must not re-ingest stored tail.

Bug: after a compaction the replayed active context is
[scaffold/summary head] + [fresh tail of already-stored rows]. The scaffold
head breaks prefix-based replay proof, so the ingest cursor only skips the
head and the already-stored fresh tail is re-appended as duplicates (the
349->5962 store_id pattern observed live). Fix:
_compacted_replay_stored_tail_overlap drops the leading run of new rows that
already exists as the stored tail, gated on scaffold evidence so a genuinely
new tail-only delta is never dropped (dup-over-loss preserved).
"""
import os
import sqlite3
import tempfile

import pytest

from plugins.context_engine.lcm.engine import LCMEngine
from plugins.context_engine.lcm.config import LCMConfig

SCAFFOLD = (
    "[Note: This conversation uses Lossless Context Management (LCM). "
    "Earlier turns have been compacted into hierarchical summaries below.]"
)
SUMMARY = "[Recent Summary (d0, node 1)] foo [Expand for details: bar]"


def _engine(tmp_path):
    eng = LCMEngine(config=LCMConfig())
    eng._bind_storage(str(tmp_path / "lcm.db"))
    eng._session_id = "S"
    return eng, str(tmp_path / "lcm.db")


def _base(n=400):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(n)
    ]


def _count(db, content):
    return sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM messages WHERE content=?", (content,)
    ).fetchone()[0]


class TestDupOnReplayFix:
    def test_compacted_replay_does_not_duplicate_stored_tail(self, tmp_path):
        eng, db = _engine(tmp_path)
        base = _base()
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        comp = (
            [{"role": "system", "content": SCAFFOLD},
             {"role": "assistant", "content": SUMMARY}]
            + [dict(m) for m in base[-40:]]
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(comp)
        assert eng._store.get_session_count("S") == n0  # nothing re-ingested
        assert all(_count(db, m["content"]) == 1 for m in base[-40:])

    def test_compacted_replay_with_genuine_new_tail_ingests_only_new(self, tmp_path):
        eng, db = _engine(tmp_path)
        base = _base()
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        comp = (
            [{"role": "system", "content": SCAFFOLD},
             {"role": "assistant", "content": SUMMARY}]
            + [dict(m) for m in base[-40:]]
            + [{"role": "user", "content": "GENUINELY_NEW_TURN"}]
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(comp)
        assert eng._store.get_session_count("S") == n0 + 1
        assert _count(db, "GENUINELY_NEW_TURN") == 1
        assert all(_count(db, m["content"]) == 1 for m in base[-40:])

    def test_tail_only_no_scaffold_still_ingests_dup_over_loss(self, tmp_path):
        """SACRED no-loss: an anchorless tail delta (even if it repeats the
        durable tail) is still ingested — losing a real new message is worse
        than a duplicate."""
        eng, db = _engine(tmp_path)
        base = _base()
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages([dict(m) for m in base[-10:]])  # no scaffold head
        assert eng._store.get_session_count("S") > n0  # ingested, not dropped

    def test_full_identical_replay_is_clean(self, tmp_path):
        eng, db = _engine(tmp_path)
        base = _base()
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages([dict(m) for m in base])
        assert eng._store.get_session_count("S") == n0

    def test_scaffold_with_genuinely_new_content_ingests(self, tmp_path):
        """Scaffold head but the tail does NOT match the store -> must ingest."""
        eng, db = _engine(tmp_path)
        base = _base()
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        comp = [
            {"role": "system", "content": SCAFFOLD},
            {"role": "user", "content": "totally new A"},
            {"role": "assistant", "content": "new B"},
        ]
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(comp)
        assert eng._store.get_session_count("S") == n0 + 2

    @pytest.mark.parametrize("n", [40, 600, 9400])
    def test_no_dup_across_session_sizes(self, tmp_path, n):
        eng, db = _engine(tmp_path)
        base = _base(n)
        eng._store.append_batch("S", base)
        n0 = eng._store.get_session_count("S")
        tail = min(40, n)
        comp = (
            [{"role": "system", "content": SCAFFOLD},
             {"role": "assistant", "content": SUMMARY}]
            + [dict(m) for m in base[-tail:]]
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(comp)
        assert eng._store.get_session_count("S") == n0

    def test_repeating_tail_pattern_preserves_genuine_new_turns(self, tmp_path):
        """Greptile #107 P1 (dup-over-loss): when the durable store ends with a
        REPEATING identity pattern and the post-compaction replay is the kept
        fresh tail [A,B,A,B] followed by GENUINELY-NEW [A,B], the overlap skip
        must NOT consume the new turns. The longest-match version lost them;
        smallest-match never skips past the real fresh-tail boundary."""
        eng, db = _engine(tmp_path)
        A = {"role": "user", "content": "AAA"}
        B = {"role": "assistant", "content": "BBB"}
        base = _base(20)
        stored = base + [dict(A), dict(B), dict(A), dict(B), dict(A), dict(B)]
        eng._store.append_batch("S", stored)
        n0 = eng._store.get_session_count("S")
        replay = (
            [{"role": "system", "content": SCAFFOLD},
             {"role": "assistant", "content": SUMMARY}]
            + [dict(A), dict(B), dict(A), dict(B)]
            + [dict(A), dict(B)]
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(replay)
        # SACRED no-loss: the 2 genuinely-new turns must be ingested (dup-over-loss).
        assert eng._store.get_session_count("S") >= n0 + 2

    def test_preserved_objective_outside_compaction_head_does_not_skip(self, tmp_path):
        """Greptile #107 P1 (2nd): a genuinely-new replay (NO real compaction
        scaffold) whose turns include a message starting with the weak
        '[Current user objective preserved from compacted history]' prefix —
        which _is_replayed_context_scaffold_message also matches — must NOT
        trigger the overlap skip, even when the trailing turns coincidentally
        match the stored tail. The skip is gated on the STRONG compaction
        signal (LCM system note / summary-node), not the weak prefix."""
        import sqlite3
        eng, db = _engine(tmp_path)
        A = {"role": "user", "content": "AAA"}
        B = {"role": "assistant", "content": "BBB"}
        PREFIX = "[Current user objective preserved from compacted history]"
        eng._store.append_batch("S", _base(20) + [dict(A), dict(B)])
        # replay: genuinely-new turns; one trips the WEAK scaffold prefix, then
        # [A,B] that coincidentally match the stored tail — must be ingested.
        replay = [{"role": "user", "content": PREFIX + " do next"}, dict(A), dict(B)]
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(replay)
        # the genuinely-new A and B must each now appear twice (stored + new),
        # i.e. the overlap skip did NOT consume them.
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM messages WHERE content='AAA'").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM messages WHERE content='BBB'").fetchone()[0] == 2

    def test_strong_vs_weak_scaffold_signal(self, tmp_path):
        """The strong-scaffold gate recognizes the LCM system note + summary-node
        rows, but NOT the weak preserved-objective prefix alone."""
        eng, _ = _engine(tmp_path)
        assert eng._is_strong_compaction_scaffold(
            {"role": "system", "content": SCAFFOLD}
        )
        assert eng._is_strong_compaction_scaffold(
            {"role": "assistant", "content": SUMMARY}
        )
        assert not eng._is_strong_compaction_scaffold(
            {"role": "user",
             "content": "[Current user objective preserved from compacted history] x"}
        )
        assert not eng._is_strong_compaction_scaffold(
            {"role": "user", "content": "an ordinary turn"}
        )

    def test_reconciled_durable_tail_does_not_double_count_new_repeat(self, tmp_path):
        """Greptile #107 P1 (3rd): when the cursor reconcile ALREADY advances
        past the replayed durable tail, the stored-tail overlap guard (the
        fallback for a scaffold-ONLY cursor advance) must not run again — else
        it double-counts and strips a genuinely-new row that coincidentally
        repeats the last stored identity.

        Scenario: store ends with a 'go'. On restart the active context is
        [LCM system note] + [full fresh tail incl. 'go' (already stored)] +
        [a genuinely-new 'go']. The reconcile consumes the durable tail
        (cursor past 'go'), leaving new_messages == [new 'go']. The overlap
        guard would otherwise match run=1 against the stored tail and drop it.
        The new 'go' must survive (count 'go' == 2: 1 stored + 1 new)."""
        eng, db = _engine(tmp_path)
        stored = _base(8) + [{"role": "user", "content": "go"}]
        eng._store.append_batch("S", stored)
        n0 = eng._store.get_session_count("S")
        replay = (
            [{"role": "system", "content": SCAFFOLD}]
            + _base(8)
            + [{"role": "user", "content": "go"}]   # already-stored fresh tail
            + [{"role": "user", "content": "go"}]   # genuinely NEW, same identity
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(replay)
        assert eng._store.get_session_count("S") == n0 + 1
        assert _count(db, "go") == 2

    def test_scaffold_only_cursor_advance_still_dedups_stored_tail(self, tmp_path):
        """Counterpart guard: when the reconcile only skips the scaffold head
        (consumes ZERO durable rows), the overlap guard MUST still run and
        de-dup the already-stored fresh tail — the canonical dup-on-replay
        fix is unchanged by the Greptile #3 gate."""
        eng, db = _engine(tmp_path)
        stored = _base(10)
        eng._store.append_batch("S", stored)
        n0 = eng._store.get_session_count("S")
        replay = (
            [{"role": "system", "content": SCAFFOLD},
             {"role": "assistant", "content": SUMMARY}]
            + _base(10)  # already-stored fresh tail, no genuinely-new rows
        )
        eng._ingest_cursor_needs_reconcile = True
        eng._ingest_messages(replay)
        assert eng._store.get_session_count("S") == n0
