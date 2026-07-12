"""
Regression tests for the finetune pipeline review fixes.

Covers: import_external persistence, session-lineage merging (chronological +
grandchildren + late children), watermark re-extraction, UTC timestamps,
multipart-content scoring, scored-snapshot dedup, min_turn_score plumbing,
bad-label exclusion, stale train/eval truncation, retro queue filtering and
label precedence, scoring weight renormalization, cluster-ID collision
handling, and secret redaction.

All tests run against synthetic data (no real state.db).
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Add skill scripts to path (mirrors tests/test_finetune.py)
_scripts_dir = str(Path(__file__).resolve().parent.parent / "optional-skills" / "mlops" / "finetune" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# The pipeline modules are imported lazily (inside the pipeline_env fixture,
# after HERMES_HOME points at a tmp dir) so that collection-time imports can
# never bind the user's real ~/.hermes paths into the module globals —
# tests/test_finetune.py imports these modules lazily for the same reason.
common = None
extract = None
format_mod = None
retro = None
score = None


def _import_pipeline_modules():
    global common, extract, format_mod, retro, score
    if common is None:
        import common as _common
        import extract as _extract
        import format as _format
        import retro as _retro
        import score as _score
        common, extract, format_mod, retro, score = (
            _common, _extract, _format, _retro, _score,
        )


# ============================================================================
# Fixtures
# ============================================================================

_PATH_ATTRS = (
    "HERMES_HOME", "FINETUNE_DIR", "DATA_DIR", "EXTRACTED_DIR", "SCORED_DIR",
    "CLUSTERS_DIR", "IMPORTED_DIR", "ADAPTERS_DIR", "MODELS_DIR", "LOGS_DIR",
    "BENCH_DIR", "FEEDBACK_PATH", "REGISTRY_PATH", "CLUSTER_STATE_PATH",
    "STATE_DB_PATH", "EXTRACT_STATE_PATH",
)


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with path constants patched on EVERY module.

    The scripts do `from common import X` at module top, so patching only
    common.X does not rebind the name in extract/score/format/retro — each
    module object gets patched individually.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _import_pipeline_modules()

    values = {
        "HERMES_HOME": hermes_home,
        "FINETUNE_DIR": hermes_home / "finetune",
        "DATA_DIR": hermes_home / "finetune" / "data",
        "EXTRACTED_DIR": hermes_home / "finetune" / "data" / "extracted",
        "SCORED_DIR": hermes_home / "finetune" / "data" / "scored",
        "CLUSTERS_DIR": hermes_home / "finetune" / "data" / "clusters",
        "IMPORTED_DIR": hermes_home / "finetune" / "data" / "imported",
        "ADAPTERS_DIR": hermes_home / "finetune" / "adapters",
        "MODELS_DIR": hermes_home / "finetune" / "models" / "merged",
        "LOGS_DIR": hermes_home / "finetune" / "logs",
        "BENCH_DIR": hermes_home / "finetune" / "bench",
        "FEEDBACK_PATH": hermes_home / "finetune" / "feedback.jsonl",
        "REGISTRY_PATH": hermes_home / "finetune" / "adapters" / "registry.json",
        "CLUSTER_STATE_PATH": hermes_home / "finetune" / "adapters" / "cluster_state.json",
        "STATE_DB_PATH": hermes_home / "state.db",
        "EXTRACT_STATE_PATH": hermes_home / "finetune" / "extract_state.json",
    }
    for mod in (common, extract, score, format_mod, retro):
        for attr in _PATH_ATTRS:
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, values[attr])

    common.ensure_dirs()
    return hermes_home


def _make_db(db_path: Path):
    """Create an empty mock state.db (schema mirrors tests/test_finetune.py)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT,
            model TEXT, model_config TEXT, system_prompt TEXT,
            parent_session_id TEXT, started_at REAL NOT NULL, ended_at REAL,
            end_reason TEXT, message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, title TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
            tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
            timestamp REAL NOT NULL, token_count INTEGER, finish_reason TEXT
        )
    """)
    conn.commit()
    return conn


def _add_session(conn, sid, started_at, messages, parent=None, source="cli"):
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, source, None, "test-model", None, None, parent,
         started_at, started_at + 100, None, len(messages), 0, 10, 20, None),
    )
    for i, (role, content) in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (sid, role, content, started_at + i),
        )
    conn.commit()


def _set_messages(conn, sid, started_at, messages):
    """Replace a session's messages and update its message_count."""
    conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    for i, (role, content) in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (sid, role, content, started_at + i),
        )
    conn.execute(
        "UPDATE sessions SET message_count = ? WHERE id = ?",
        (len(messages), sid),
    )
    conn.commit()


def _scored_session(sid, turns, composite=0.8, turn_scores=None, bad_turns=None,
                    started_at="2026-01-01T00:00:00+00:00"):
    scoring = {
        "composite_score": composite,
        "bucket": "good" if composite >= 0.7 else (
            "neutral" if composite >= 0.4 else "bad"),
        "turn_scores": turn_scores or [],
    }
    if bad_turns is not None:
        scoring["bad_turn_indices"] = bad_turns
    return {
        "session_id": sid,
        "started_at": started_at,
        "turns": turns,
        "metadata": {"source": "cli", "model": "test", "tool_call_count": 0,
                     "total_tokens": 100},
        "scoring": scoring,
    }


# ============================================================================
# Finding 1: import_external persistence
# ============================================================================

class TestImportExternal:
    def test_imported_records_persisted_to_extracted_dir(self, pipeline_env):
        record = {
            "session_id": "ext-1",
            "started_at": "2026-01-01T00:00:00+00:00",
            "turns": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}],
            "metadata": {"source": "import"},
        }
        src = extract.IMPORTED_DIR / "claude_export.jsonl"
        src.write_text(json.dumps(record) + "\n", encoding="utf-8")

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        imported = extractor.import_external()

        assert len(imported) == 1
        # Record must land in EXTRACTED_DIR so scoring/clustering sees it
        batches = list(extract.EXTRACTED_DIR.glob("extract_*.jsonl"))
        assert len(batches) == 1
        persisted = common.read_jsonl(batches[0])
        assert persisted == [record]
        # Source file moved to processed/
        assert not src.exists()
        assert (extract.IMPORTED_DIR / "processed" / "claude_export.jsonl").exists()

    def test_unparseable_file_not_moved_to_processed(self, pipeline_env):
        src = extract.IMPORTED_DIR / "garbage.jsonl"
        src.write_text("this is { not json\nnor this ]\n", encoding="utf-8")

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        imported = extractor.import_external()

        assert imported == []
        # File must stay in place — the failure is visible, data not lost
        assert src.exists()
        assert not (extract.IMPORTED_DIR / "processed" / "garbage.jsonl").exists()
        assert list(extract.EXTRACTED_DIR.glob("extract_*.jsonl")) == []


# ============================================================================
# Finding 2: session lineage (chronological order, grandchildren, late child)
# ============================================================================

class TestLineage:
    def test_children_merge_chronologically_not_lexicographically(self, pipeline_env):
        conn = _make_db(extract.STATE_DB_PATH)
        _add_session(conn, "root", 1000, [("user", "start"), ("assistant", "first")])
        # Lexicographic order (a-child, z-child) is the REVERSE of
        # chronological order here.
        _add_session(conn, "z-child", 2000,
                     [("user", "second-q"), ("assistant", "second-a")], parent="root")
        _add_session(conn, "a-child", 3000,
                     [("user", "third-q"), ("assistant", "third-a")], parent="root")
        conn.close()

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        sessions = extractor.extract(full=True)

        assert len(sessions) == 1
        merged = sessions[0]
        assert merged["session_id"] == "root"
        contents = [t["content"] for t in merged["turns"]]
        assert contents == ["start", "first", "second-q", "second-a",
                            "third-q", "third-a"]

    def test_grandchildren_are_merged(self, pipeline_env):
        conn = _make_db(extract.STATE_DB_PATH)
        _add_session(conn, "root", 1000, [("user", "gen0"), ("assistant", "a0")])
        _add_session(conn, "child", 2000, [("user", "gen1"), ("assistant", "a1")],
                     parent="root")
        _add_session(conn, "grandchild", 3000, [("user", "gen2"), ("assistant", "a2")],
                     parent="child")
        conn.close()

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        sessions = extractor.extract(full=True)

        assert len(sessions) == 1
        contents = [t["content"] for t in sessions[0]["turns"]]
        # Grandchild messages must not vanish
        assert "gen2" in contents and "a2" in contents
        assert contents.index("gen1") < contents.index("gen2")

    def test_child_arriving_after_parent_extracted_is_not_lost(self, pipeline_env):
        conn = _make_db(extract.STATE_DB_PATH)
        _add_session(conn, "root", 1000, [("user", "orig"), ("assistant", "resp")])
        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        first = extractor.extract(full=True)
        assert [s["session_id"] for s in first] == ["root"]

        # A compression-split child appears in a later incremental window
        _add_session(conn, "late-child", 2000,
                     [("user", "late-q"), ("assistant", "late-a")], parent="root")
        conn.close()

        second = extractor.extract(full=False)
        assert len(second) == 1
        merged = second[0]
        assert merged["session_id"] == "root"
        contents = [t["content"] for t in merged["turns"]]
        assert contents == ["orig", "resp", "late-q", "late-a"]

        # Deduped load keeps the newest (merged) copy — exactly one record
        all_extracted = extractor.get_all_extracted()
        assert len(all_extracted) == 1
        assert len(all_extracted[0]["turns"]) == 4


# ============================================================================
# Finding 3: UTC timestamps + watermark re-extraction of grown sessions
# ============================================================================

class TestWatermark:
    def test_started_at_is_timezone_aware_utc(self, pipeline_env):
        from datetime import datetime, timezone
        conn = _make_db(extract.STATE_DB_PATH)
        _add_session(conn, "s1", 1700000000, [("user", "q"), ("assistant", "a")])
        conn.close()

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        sessions = extractor.extract(full=True)

        ts = datetime.fromisoformat(sessions[0]["started_at"])
        assert ts.tzinfo is not None
        assert ts.timestamp() == 1700000000
        assert ts.astimezone(timezone.utc) == datetime.fromtimestamp(
            1700000000, tz=timezone.utc)

    def test_in_progress_session_reextracted_after_growth(self, pipeline_env):
        conn = _make_db(extract.STATE_DB_PATH)
        _add_session(conn, "grow", 1000, [("user", "q1"), ("assistant", "a1")])

        extractor = extract.SessionExtractor(db_path=extract.STATE_DB_PATH)
        assert len(extractor.extract(full=True)) == 1

        # No change → nothing re-extracted
        assert extractor.extract(full=False) == []

        # Session gains messages (was still in progress at extraction time)
        _set_messages(conn, "grow", 1000,
                      [("user", "q1"), ("assistant", "a1"),
                       ("user", "q2"), ("assistant", "a2")])
        conn.close()

        again = extractor.extract(full=False)
        assert len(again) == 1
        assert len(again[0]["turns"]) == 4
        # And it settles: a further run with no changes extracts nothing
        assert extractor.extract(full=False) == []


# ============================================================================
# Finding 4: multipart content must not abort scoring
# ============================================================================

class TestMultipartScoring:
    def _multipart_session(self):
        return {
            "session_id": "multipart",
            "started_at": "2026-01-01T00:00:00+00:00",
            "turns": [
                {"role": "user", "content": [
                    {"type": "text", "text": "please look at this screenshot"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "It shows a traceback in main.py, here is the fix."},
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "thanks, that's exactly right"},
                ]},
            ],
            "metadata": {"source": "cli", "model": "test", "tool_call_count": 0,
                         "total_tokens": 60},
        }

    def test_score_session_handles_multipart_content(self, pipeline_env):
        scorer = score.QualityScorer()
        scored = scorer.score_session(self._multipart_session())
        assert "composite_score" in scored["scoring"]
        assert scored["scoring"]["turn_scores"]  # assistant turn was scored

    def test_score_session_handles_missing_role(self, pipeline_env):
        scorer = score.QualityScorer()
        session = {
            "session_id": "no-role",
            "turns": [
                {"content": "orphan message with no role"},
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "hi, how can I help you today?"},
            ],
            "metadata": {},
        }
        scored = scorer.score_session(session)
        assert "composite_score" in scored["scoring"]

    def test_score_all_skips_broken_session_instead_of_aborting(self, pipeline_env):
        scorer = score.QualityScorer()
        good = self._multipart_session()
        broken = {"session_id": "broken", "turns": None, "metadata": {}}
        scored = scorer.score_all([good, broken])
        assert [s["session_id"] for s in scored] == ["multipart"]
        # The good session's snapshot was still written
        assert list(score.SCORED_DIR.glob("scored_*.jsonl"))

    def test_content_to_text_drops_non_text_parts(self, pipeline_env):
        assert common.content_to_text(
            [{"type": "text", "text": "hello"},
             {"type": "image_url", "image_url": {"url": "data:..."}}]
        ) == "hello"
        assert common.content_to_text("plain") == "plain"
        assert common.content_to_text(None) == ""


# ============================================================================
# Finding 5: scored-snapshot dedup (cluster.py / format.py load paths)
# ============================================================================

class TestScoredDedup:
    def _write_snapshots(self):
        turns = [{"role": "user", "content": "Q"},
                 {"role": "assistant", "content": "A"}]
        old = _scored_session("dup", turns, turn_scores=[(1, 0.9)])
        new = _scored_session("dup", turns, turn_scores=[(1, 0.95)])
        common.append_jsonl(common.SCORED_DIR / "scored_20260101_000000.jsonl", [old])
        common.append_jsonl(common.SCORED_DIR / "scored_20260102_000000.jsonl", [new])

    def test_load_records_dedup_keeps_newest(self, pipeline_env):
        self._write_snapshots()
        records = common.load_records_dedup(common.SCORED_DIR, "scored_*.jsonl")
        assert len(records) == 1
        assert records[0]["scoring"]["turn_scores"] == [[1, 0.95]]

    def test_format_all_scored_produces_no_duplicate_records(self, pipeline_env):
        self._write_snapshots()
        formatter = format_mod.TrainingFormatter()
        counts = formatter.format_for_cluster(
            common.load_records_dedup(format_mod.SCORED_DIR, "scored_*.jsonl"),
            "_general", min_score=0.7,
        )
        assert counts["train"] + counts["eval"] == 1

        counts2 = formatter.format_all_scored()
        assert counts2["train"] + counts2["eval"] == 1

    def test_cluster_load_path_dedupes(self, pipeline_env, monkeypatch):
        cluster = pytest.importorskip("cluster")
        self._write_snapshots()
        monkeypatch.setattr(cluster, "SCORED_DIR", common.SCORED_DIR)
        # Same dedup helper, driven with cluster's own module constant
        sessions = common.load_records_dedup(cluster.SCORED_DIR, "scored_*.jsonl")
        assert len(sessions) == 1


# ============================================================================
# Findings 7+8: min_turn_score plumbing in cluster → format
# ============================================================================

class TestMinTurnScorePlumbing:
    def test_formatter_default_threshold_is_config_min_turn_score(self, pipeline_env):
        turns = [{"role": "user", "content": "Q"},
                 {"role": "assistant", "content": "A"}]
        # 0.5 clears the old neutral threshold (0.4) but NOT min_turn_score (0.7)
        mediocre = _scored_session("med", turns, turn_scores=[(1, 0.5)])
        good = _scored_session("ok", turns, turn_scores=[(1, 0.9)])

        formatter = format_mod.TrainingFormatter()
        assert formatter.min_turn_score == 0.7
        counts = formatter.format_for_cluster([mediocre, good], "_general")
        assert counts["train"] + counts["eval"] == 1

    def test_count_trainable_turns_uses_per_turn_scores(self, pipeline_env):
        cluster = pytest.importorskip("cluster")
        turns = [{"role": "user", "content": "Q"},
                 {"role": "assistant", "content": "A1"},
                 {"role": "user", "content": "Q2"},
                 {"role": "assistant", "content": "A2"}]
        # good bucket, but only ONE turn is trainable at 0.7
        s = _scored_session("s", turns, composite=0.8,
                            turn_scores=[(1, 0.9), (3, 0.5)])
        assert cluster._count_trainable_turns([s], 0.7) == 1
        # bad-labeled turns never count
        s2 = _scored_session("s2", turns, composite=0.8,
                             turn_scores=[(1, 0.9), (3, 0.9)], bad_turns=[3])
        assert cluster._count_trainable_turns([s2], 0.7) == 1

    def test_fallback_to_general_maturity_and_formatting(self, pipeline_env, monkeypatch):
        cluster = pytest.importorskip("cluster")
        monkeypatch.setattr(cluster, "SCORED_DIR", common.SCORED_DIR)
        monkeypatch.setattr(cluster, "CLUSTERS_DIR", common.CLUSTERS_DIR)
        monkeypatch.setattr(cluster, "CLUSTER_STATE_PATH", common.CLUSTER_STATE_PATH)

        turns = [{"role": "user", "content": "Q"},
                 {"role": "assistant", "content": "A"}]
        # Good-bucket session whose only turn is below min_turn_score:
        # old code counted its assistant turn AND formatted it (0.4 cutoff).
        mediocre = _scored_session("m", turns, composite=0.8,
                                   turn_scores=[(1, 0.5)])
        state = cluster.DomainClusterer(config={}) \
            ._fallback_to_general([mediocre])

        assert state["clusters"]["_general"]["good_turns"] == 0
        train = common.CLUSTERS_DIR / "_general" / "train.jsonl"
        eval_ = common.CLUSTERS_DIR / "_general" / "eval.jsonl"
        assert train.exists() and train.read_text() == ""
        assert eval_.exists() and eval_.read_text() == ""


# ============================================================================
# Finding 10: bad-label exclusion + stale train/eval truncation
# ============================================================================

class TestFormatExclusions:
    def test_bad_labeled_turn_excluded_even_at_zero_threshold(self, pipeline_env):
        turns = [{"role": "user", "content": "Q1"},
                 {"role": "assistant", "content": "A1"},
                 {"role": "user", "content": "Q2"},
                 {"role": "assistant", "content": "A2"}]
        s = _scored_session("bad-turn", turns,
                            turn_scores=[(1, 0.0), (3, 0.9)], bad_turns=[1])
        examples = format_mod.extract_training_turns(s, min_turn_score=0.0)
        targets = [ex["conversations"][-1]["value"] for ex in examples]
        assert "A1" not in targets
        assert "A2" in targets

    def test_retro_bad_label_flows_through_scoring_to_format(self, pipeline_env):
        # retro bad label on assistant turn 1 → score → format excludes it
        retro.write_label("flow", "bad", 0.0, turn_index=1)
        scorer = score.QualityScorer()
        session = {
            "session_id": "flow",
            "turns": [{"role": "user", "content": "Q"},
                      {"role": "assistant", "content": "A"}],
            "metadata": {},
        }
        scored = scorer.score_session(session)
        assert scored["scoring"]["bad_turn_indices"] == [1]
        assert format_mod.extract_training_turns(scored, min_turn_score=0.0) == []

    def test_rerun_with_zero_records_truncates_stale_files(self, pipeline_env):
        turns = [{"role": "user", "content": "Q"},
                 {"role": "assistant", "content": "A"}]
        good = _scored_session("g", turns, turn_scores=[(1, 0.9)])
        formatter = format_mod.TrainingFormatter()
        counts = formatter.format_for_cluster([good], "c-x", min_score=0.7)
        assert counts["train"] + counts["eval"] == 1

        train = common.CLUSTERS_DIR / "c-x" / "train.jsonl"
        eval_ = common.CLUSTERS_DIR / "c-x" / "eval.jsonl"
        assert train.read_text() or eval_.read_text()

        # Re-run where everything is filtered out — stale data must not survive
        bad = _scored_session("g", turns, turn_scores=[(1, 0.1)])
        counts = formatter.format_for_cluster([bad], "c-x", min_score=0.7)
        assert counts == {"train": 0, "eval": 0}
        assert train.read_text() == ""
        assert eval_.read_text() == ""


# ============================================================================
# Finding 11: retro queue filtering + label precedence
# ============================================================================

class TestRetroQueueAndPrecedence:
    def _seed_scored(self, *sids):
        sessions = [
            _scored_session(sid, [{"role": "user", "content": f"question {sid}"},
                                  {"role": "assistant", "content": f"answer {sid}"}],
                            composite=0.5)
            for sid in sids
        ]
        common.append_jsonl(common.SCORED_DIR / "scored_20260101_000000.jsonl", sessions)
        return sessions

    def test_turn_labeled_sessions_leave_the_queue(self, pipeline_env, capsys):
        self._seed_scored("turn-labeled", "untouched")
        retro.write_label("turn-labeled", "good", 1.0, turn_index=1)

        class Args:
            limit = 10
        retro.cmd_list(Args())
        out = capsys.readouterr().out
        assert "untouched" in out
        assert "turn-labeled" not in out
        assert "1 unlabeled / 2 total" in out

    def test_session_level_label_writes_single_session_scope_record(self, pipeline_env):
        self._seed_scored("whole")
        retro._apply_label("whole", "bad", None)
        feedback = retro.load_feedback()
        assert len(feedback) == 1
        assert feedback[0]["session_id"] == "whole"
        assert feedback[0]["signal"] == "bad"
        assert "turn_index" not in feedback[0]

    def test_turn_label_wins_over_session_label_regardless_of_order(self, pipeline_env):
        turns = [{"role": "user", "content": "Q1"},
                 {"role": "assistant", "content": "A1"},
                 {"role": "user", "content": "Q2"},
                 {"role": "assistant", "content": "A2"}]

        # Order 1: turn-level good FIRST, session-level bad SECOND.
        # The old expansion clobbered the turn label (last-record-wins).
        retro.write_label("p1", "good", 1.0, turn_index=1)
        retro.write_label("p1", "bad", 0.0)
        # Order 2: session-level bad first, then turn-level good.
        retro.write_label("p2", "bad", 0.0)
        retro.write_label("p2", "good", 1.0, turn_index=1)

        scorer = score.QualityScorer()
        for sid in ("p1", "p2"):
            scored = scorer.score_session({
                "session_id": sid, "turns": list(turns), "metadata": {},
            })
            scoring = scored["scoring"]
            assert scoring["manual_override"] is True
            ts = dict((int(i), s) for i, s in scoring["turn_scores"])
            assert ts[1] == 1.0, f"{sid}: turn override must win"
            assert ts[3] == 0.0, f"{sid}: unlabeled turn takes session score"
            # msg 3 is bad (session-level), msg 1 is NOT (turn-level good wins)
            assert scoring["bad_turn_indices"] == [3]

            # Format keeps the good turn, drops the bad one — at any threshold
            examples = format_mod.extract_training_turns(scored, min_turn_score=0.0)
            targets = [ex["conversations"][-1]["value"] for ex in examples]
            assert targets == ["A1"]


# ============================================================================
# Finding 6: composite weight renormalization
# ============================================================================

class TestWeightRenormalization:
    def test_default_positive_weights_sum_to_one(self, pipeline_env):
        scorer = score.QualityScorer()
        total = scorer.w_conv + scorer.w_negative + scorer.w_positive + scorer.w_sent
        assert total == pytest.approx(1.0)
        # Relative proportions preserved (0.15 : 0.25 : 0.35 : 0.05)
        assert scorer.w_positive / scorer.w_negative == pytest.approx(0.35 / 0.25)
        assert scorer.w_conv / scorer.w_sent == pytest.approx(0.15 / 0.05)

    def test_perfect_signals_can_reach_good_bucket(self, pipeline_env):
        scorer = score.QualityScorer()
        # All component signals at 1.0 (sentiment maxes at +0.2 → 0.7 term)
        composite = (
            scorer.w_conv * 1.0 + scorer.w_negative * 1.0
            + scorer.w_positive * 1.0 + scorer.w_sent * (0.5 + 0.2)
        )
        assert composite > scorer.good_threshold  # was capped at ~0.785 before

    def test_legacy_mode_weights_unchanged(self, pipeline_env):
        scorer = score.QualityScorer(config={"mode": "legacy", "weights": {},
                                             "thresholds": {}})
        assert scorer.w_conv == 0.3
        assert scorer.w_turn == 0.4


# ============================================================================
# Finding 9: cluster ID uniqueness on previous-cluster matching
# ============================================================================

class TestClusterIdCollision:
    def test_two_clusters_cannot_claim_same_previous_id(self, pipeline_env):
        cluster = pytest.importorskip("cluster")
        np = pytest.importorskip("numpy")

        base = np.zeros(8)
        base[0] = 1.0
        near1 = base.copy()
        near2 = base * 0.999  # both are >0.9 cosine-similar to "c-old"
        near2[1] = 0.01

        clusterer = cluster.DomainClusterer(config={})
        mapping = clusterer._match_previous_clusters(
            {0: near1, 1: near2},
            {"centroids": {"c-old": base.tolist()}},
        )

        assert set(mapping.keys()) == {0, 1}
        # No key collision: both clusters keep distinct IDs
        assert len(set(mapping.values())) == 2
        # The better match gets the previous ID
        assert mapping[0] == "c-old"
        assert mapping[1] != "c-old"


# ============================================================================
# Finding 13: failure-pattern tightening + link-4 word boundaries
# ============================================================================

class TestFailurePatterns:
    def test_benign_mentions_are_not_hard_failures(self, pipeline_env):
        ok = ("The deprecated helper was not found in the new docs; "
              "we also fixed a syntax error in the old release notes.")
        assert score._tool_result_status(ok) == "success"

    def test_real_failures_still_detected(self, pipeline_env):
        assert score._tool_result_status("bash: foo: command not found") == "hard_failure"
        assert score._tool_result_status("cat: x.txt: No such file or directory") == "hard_failure"
        assert score._tool_result_status("Error: connection refused") == "hard_failure"
        assert score._tool_result_status(
            'File "x.py", line 1\nSyntaxError: invalid syntax') == "hard_failure"
        assert score._tool_result_status(
            "Traceback (most recent call last):") == "hard_failure"

    def test_link4_requires_word_boundary(self, pipeline_env):
        def turns_with_user_reply(reply):
            return [
                {"role": "user", "content": "check it"},
                {"role": "assistant", "content": "ok", "tool_calls": [
                    {"function": {"name": "terminal", "arguments": "{}"}}]},
                {"role": "tool", "content": "name", "tool_name": "terminal"},
                {"role": "user", "content": reply},
            ]

        # 'name' inside 'rename' must NOT count as referencing the output
        embedded = score.positive_tool_success_chain(
            turns_with_user_reply(
                "we should rename everything else here soon please and thanks"), 1)
        assert embedded == 0.9
        # A real standalone reference still upgrades to 1.0
        exact = score.positive_tool_success_chain(
            turns_with_user_reply(
                "the name value looks correct to me after checking it twice"), 1)
        assert exact == 1.0


# ============================================================================
# Finding 14: common.py config defaults + env override + read_jsonl warning
# ============================================================================

class TestCommonConfig:
    def test_default_base_model_is_hf_id(self, pipeline_env, monkeypatch):
        monkeypatch.delenv("FINETUNE_BASE_MODEL", raising=False)
        config = common.load_config()
        assert config["training"]["base_model"] == "kai-os/Carnice-9b"
        assert config["serving"]["converter"] == ""
        assert config["serving"]["server_pid_file"] == ""
        assert config["serving"]["server_log_path"] == ""

    def test_finetune_base_model_env_override(self, pipeline_env, monkeypatch):
        monkeypatch.setenv("FINETUNE_BASE_MODEL", "myorg/my-model")
        config = common.load_config()
        assert config["training"]["base_model"] == "myorg/my-model"

    def test_read_jsonl_warns_on_malformed_lines(self, pipeline_env, caplog):
        import logging as _logging
        p = common.FINETUNE_DIR / "mixed.jsonl"
        p.write_text('{"ok": 1}\nnot json at all\n{"ok": 2}\n', encoding="utf-8")
        with caplog.at_level(_logging.WARNING, logger="hermes.finetune"):
            records = common.read_jsonl(p)
        assert len(records) == 2
        assert any("malformed" in r.message.lower() for r in caplog.records)


# ============================================================================
# Finding 16: secret redaction in training records
# ============================================================================

class TestSecretRedaction:
    def test_obvious_secrets_redacted(self, pipeline_env):
        cases = [
            "my key is AKIAIOSFODNN7EXAMPLE ok",
            "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx1234",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----",
            'password = "hunter2butlonger"',
            "token=ghp_abcdefghijklmnopqrstuvwxyz012345",
        ]
        for text in cases:
            redacted = format_mod.redact_secrets(text)
            assert "[REDACTED]" in redacted, text
        assert "AKIAIOSFODNN7EXAMPLE" not in format_mod.redact_secrets(cases[0])
        assert "sk-abcdefghijklmnopqrstuvwx1234" not in format_mod.redact_secrets(cases[1])

    def test_normal_code_not_mangled(self, pipeline_env):
        code = (
            "def get_token():\n"
            "    token = get_token_from_env()\n"
            "    return token or None\n"
            "result = client.request(path)\n"
        )
        assert format_mod.redact_secrets(code) == code

    def test_redaction_applied_to_training_records(self, pipeline_env):
        turns = [
            {"role": "user", "content": "here is my key sk-abcdefghijklmnopqrstuvwx9876"},
            {"role": "assistant", "content": "Got it, I'll use the key you provided."},
        ]
        s = _scored_session("secret", turns, turn_scores=[(1, 0.9)])
        examples = format_mod.extract_training_turns(s, min_turn_score=0.7)
        assert examples
        dumped = json.dumps(examples)
        assert "sk-abcdefghijklmnopqrstuvwx9876" not in dumped
        assert "[REDACTED]" in dumped


# ============================================================================
# Finding 12: recency decay handles aware and legacy-naive timestamps
# ============================================================================

class TestRecencyDecay:
    def test_aware_and_naive_local_agree(self, pipeline_env):
        from datetime import datetime, timedelta, timezone
        aware = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        naive_local = (datetime.now() - timedelta(days=7)).isoformat()
        d_aware = retro._recency_decay(aware)
        d_naive = retro._recency_decay(naive_local)
        # 7 days at a 14-day half-life → 0.5 ** 0.5 ≈ 0.707
        assert d_aware == pytest.approx(0.5 ** 0.5, abs=0.02)
        # Legacy naive local values decay identically (no UTC-offset skew)
        assert d_naive == pytest.approx(d_aware, abs=0.02)

    def test_garbage_timestamp_returns_zero(self, pipeline_env):
        assert retro._recency_decay("not-a-date") == 0.0
        assert retro._recency_decay(None) == 0.0
