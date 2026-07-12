"""
Unit tests for the finetune pipeline.

Tests extraction, scoring, formatting, clustering logic, registry management,
and routing — all against synthetic data (no real state.db needed).
"""

import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

# Add skill scripts to path
_scripts_dir = str(Path(__file__).resolve().parent.parent / "optional-skills" / "mlops" / "finetune" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_hermes(tmp_path, monkeypatch):
    """Set up a temporary ~/.hermes with finetune dirs."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Patch common.py paths
    import common
    monkeypatch.setattr(common, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(common, "FINETUNE_DIR", hermes_home / "finetune")
    monkeypatch.setattr(common, "DATA_DIR", hermes_home / "finetune" / "data")
    monkeypatch.setattr(common, "EXTRACTED_DIR", hermes_home / "finetune" / "data" / "extracted")
    monkeypatch.setattr(common, "SCORED_DIR", hermes_home / "finetune" / "data" / "scored")
    monkeypatch.setattr(common, "CLUSTERS_DIR", hermes_home / "finetune" / "data" / "clusters")
    monkeypatch.setattr(common, "IMPORTED_DIR", hermes_home / "finetune" / "data" / "imported")
    monkeypatch.setattr(common, "ADAPTERS_DIR", hermes_home / "finetune" / "adapters")
    monkeypatch.setattr(common, "MODELS_DIR", hermes_home / "finetune" / "models" / "merged")
    monkeypatch.setattr(common, "LOGS_DIR", hermes_home / "finetune" / "logs")
    monkeypatch.setattr(common, "BENCH_DIR", hermes_home / "finetune" / "bench")
    monkeypatch.setattr(common, "FEEDBACK_PATH", hermes_home / "finetune" / "feedback.jsonl")
    monkeypatch.setattr(common, "REGISTRY_PATH", hermes_home / "finetune" / "adapters" / "registry.json")
    monkeypatch.setattr(common, "CLUSTER_STATE_PATH", hermes_home / "finetune" / "adapters" / "cluster_state.json")
    monkeypatch.setattr(common, "EXTRACT_STATE_PATH", hermes_home / "finetune" / "extract_state.json")

    common.ensure_dirs()
    return hermes_home


@pytest.fixture
def mock_state_db(tmp_path):
    """Create a mock state.db with sample sessions."""
    db_path = tmp_path / "state.db"
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

    # Session 1: Good conversation (user thanks at end)
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("s1", "cli", None, "test-model", None, None, None,
         1700000000, 1700001000, None, 4, 0, 100, 200, None),
    )
    msgs_s1 = [
        ("s1", "system", "You are a helpful assistant.", 1700000000),
        ("s1", "user", "How do I sort a list in Python?", 1700000001),
        ("s1", "assistant", "You can use the sorted() function or list.sort() method.", 1700000002),
        ("s1", "user", "Thanks, that's exactly what I needed!", 1700000003),
    ]
    for sid, role, content, ts in msgs_s1:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (sid, role, content, ts),
        )

    # Session 2: Bad conversation (user corrects)
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("s2", "cli", None, "test-model", None, None, None,
         1700002000, 1700003000, None, 4, 0, 100, 200, None),
    )
    msgs_s2 = [
        ("s2", "system", "You are a helpful assistant.", 1700002000),
        ("s2", "user", "Write a Fibonacci function in Python", 1700002001),
        ("s2", "assistant", "Here's a Fibonacci function:\ndef fib(n): return n", 1700002002),
        ("s2", "user", "No, that's wrong. That's not Fibonacci at all.", 1700002003),
    ]
    for sid, role, content, ts in msgs_s2:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (sid, role, content, ts),
        )

    # Session 3: Short session (below min_turns=2 when counting user turns)
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("s3", "cli", None, "test-model", None, None, None,
         1700004000, 1700004100, None, 1, 0, 10, 20, None),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        ("s3", "user", "hi", 1700004000),
    )

    conn.commit()
    conn.close()
    return db_path


def _make_session(session_id, turns, score=None):
    """Helper to create a scored session dict."""
    session = {
        "session_id": session_id,
        "started_at": "2026-01-01T00:00:00",
        "turns": turns,
        "metadata": {"source": "cli", "model": "test", "parent_session_id": None,
                      "tool_call_count": 0, "total_tokens": 100},
    }
    if score is not None:
        session["scoring"] = {
            "composite_score": score,
            "bucket": "good" if score >= 0.7 else ("neutral" if score >= 0.4 else "bad"),
        }
    return session


# ============================================================================
# Extract tests
# ============================================================================

class TestExtractor:
    def test_extract_from_db(self, tmp_hermes, mock_state_db, monkeypatch):
        import common
        monkeypatch.setattr(common, "STATE_DB_PATH", mock_state_db)

        from extract import SessionExtractor
        extractor = SessionExtractor(db_path=mock_state_db)
        sessions = extractor.extract(full=True)

        # s1 and s2 have message_count >= 2, s3 has only 1
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s1"
        assert sessions[1]["session_id"] == "s2"

    def test_extract_turns_structure(self, tmp_hermes, mock_state_db, monkeypatch):
        import common
        monkeypatch.setattr(common, "STATE_DB_PATH", mock_state_db)

        from extract import SessionExtractor
        extractor = SessionExtractor(db_path=mock_state_db)
        sessions = extractor.extract(full=True)

        s1 = sessions[0]
        assert len(s1["turns"]) == 4
        assert s1["turns"][0]["role"] == "system"
        assert s1["turns"][1]["role"] == "user"
        assert s1["metadata"]["source"] == "cli"

    def test_incremental_extraction(self, tmp_hermes, mock_state_db, monkeypatch):
        import common
        monkeypatch.setattr(common, "STATE_DB_PATH", mock_state_db)

        from extract import SessionExtractor
        extractor = SessionExtractor(db_path=mock_state_db)

        # First extraction
        s1 = extractor.extract(full=True)
        assert len(s1) == 2

        # Second extraction (no new data)
        s2 = extractor.extract(full=False)
        assert len(s2) == 0


# ============================================================================
# Score tests
# ============================================================================

class TestScorer:
    def test_good_session_scores_higher(self, tmp_hermes):
        from score import QualityScorer
        scorer = QualityScorer()

        good = _make_session("good", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "How do I sort a list?"},
            {"role": "assistant", "content": "Use sorted() or list.sort()."},
            {"role": "user", "content": "Thanks, that's exactly what I needed!"},
        ])
        bad = _make_session("bad", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write fibonacci"},
            {"role": "assistant", "content": "def fib(n): return n"},
            {"role": "user", "content": "No, that's wrong. That's not right at all."},
        ])

        scored_good = scorer.score_session(good)
        scored_bad = scorer.score_session(bad)

        assert scored_good["scoring"]["composite_score"] > scored_bad["scoring"]["composite_score"]

    def test_manual_override(self, tmp_hermes):
        from score import QualityScorer
        import common

        # Write feedback
        common.append_jsonl(common.FEEDBACK_PATH, [
            {"session_id": "override-test", "score": 1.0}
        ])

        scorer = QualityScorer()
        session = _make_session("override-test", [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ])
        scored = scorer.score_session(session)

        assert scored["scoring"]["composite_score"] == 1.0
        assert scored["scoring"]["manual_override"] is True
        assert scored["scoring"]["bucket"] == "good"

    def test_correction_detection(self, tmp_hermes):
        from score import QualityScorer
        scorer = QualityScorer()

        session = _make_session("correction", [
            {"role": "user", "content": "Do X"},
            {"role": "assistant", "content": "Here's how to do Y"},
            {"role": "user", "content": "No, I meant X not Y. Actually, I want X."},
        ])
        scored = scorer.score_session(session)
        # The negative_turn_signal carries the correction-induced low score.
        # In positive_signals mode, turn_scores exposes positive signals
        # (which default to 0.5 neutral when nothing fires), so the
        # discriminating signal lives in negative_turn_signal.
        assert scored["scoring"]["negative_turn_signal"] < 0.3
        # And the composite should land in the bad bucket given the correction
        assert scored["scoring"]["bucket"] in ("bad", "neutral")


# ============================================================================
# Format tests
# ============================================================================

class TestFormatter:
    def test_chatml_format(self, tmp_hermes):
        from format import format_session_chatml

        session = _make_session("fmt-test", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ], score=0.8)

        result = format_session_chatml(session)
        assert result is not None
        assert len(result["conversations"]) == 3
        assert result["conversations"][0]["from"] == "system"
        assert result["conversations"][1]["from"] == "human"
        assert result["conversations"][2]["from"] == "gpt"

    def test_tool_calls_in_format(self, tmp_hermes):
        from format import format_session_chatml

        session = _make_session("tool-test", [
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
            ]},
            {"role": "tool", "content": "file1.py\nfile2.py", "tool_name": "terminal"},
            {"role": "assistant", "content": "Here are the files."},
        ], score=0.8)

        result = format_session_chatml(session)
        assert result is not None
        # Should have system + human + gpt(tool) + tool + gpt
        assert any("tool_call" in c["value"] for c in result["conversations"]
                    if c["from"] == "gpt")

    def test_train_eval_split(self, tmp_hermes):
        from format import TrainingFormatter

        sessions = [
            _make_session(f"split-{i}", [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ], score=0.8)
            for i in range(100)
        ]

        formatter = TrainingFormatter(eval_ratio=0.1)
        counts = formatter.format_for_cluster(sessions, "_general")

        # With 100 sessions and 10% eval, expect ~90 train, ~10 eval
        assert counts["train"] > 0
        assert counts["eval"] > 0
        assert counts["train"] + counts["eval"] == 100

    def test_empty_session_skipped(self, tmp_hermes):
        from format import format_session_chatml

        session = _make_session("empty", [], score=0.5)
        result = format_session_chatml(session)
        assert result is None

    def test_reasoning_scratchpad_converted(self, tmp_hermes):
        from format import format_session_chatml

        session = _make_session("reasoning", [
            {"role": "user", "content": "Think about X"},
            {"role": "assistant", "content": "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>Answer"},
        ], score=0.8)

        result = format_session_chatml(session)
        gpt_turn = [c for c in result["conversations"] if c["from"] == "gpt"][0]
        assert "<think>" in gpt_turn["value"]
        assert "<REASONING_SCRATCHPAD>" not in gpt_turn["value"]


# ============================================================================
# Registry tests
# ============================================================================

class TestRegistry:
    def test_register_and_promote(self, tmp_hermes):
        from manage import AdapterRegistry
        import common

        # Create adapter directory
        adapter_dir = common.ADAPTERS_DIR / "c-test" / "v1"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "config.yml").write_text("test: true")

        registry = AdapterRegistry()
        entry = registry.register_adapter("c-test", "v1", "established", 100)

        assert entry["status"] == "trained"
        assert entry["cluster_id"] == "c-test"

        # Promote
        assert registry.promote("c-test", "v1") is True

        # Reload and check
        registry2 = AdapterRegistry()
        active = registry2._find_active("c-test")
        assert active is not None
        assert active["version"] == "v1"
        assert active["status"] == "active"

    def test_rollback(self, tmp_hermes):
        from manage import AdapterRegistry
        import common

        for v in ["v1", "v2"]:
            d = common.ADAPTERS_DIR / "c-rb" / v
            d.mkdir(parents=True)
            (d / "config.yml").write_text("test: true")

        registry = AdapterRegistry()
        registry.register_adapter("c-rb", "v1", "established", 50)
        registry.promote("c-rb", "v1")
        registry.register_adapter("c-rb", "v2", "established", 80)
        registry.promote("c-rb", "v2")

        # v2 is active, v1 is previous
        assert registry._find_active("c-rb")["version"] == "v2"

        # Rollback
        assert registry.rollback("c-rb") is True
        assert registry._find_active("c-rb")["version"] == "v1"

    def test_status_report(self, tmp_hermes):
        from manage import AdapterRegistry

        registry = AdapterRegistry()
        report = registry.status()

        assert "FINETUNE PIPELINE STATUS" in report
        assert "Extracted sessions:" in report


# ============================================================================
# Eval gate tests
# ============================================================================

class TestEvalGate:
    def test_verdict_pass(self):
        from eval import compare_metrics, verdict

        baseline = {
            "tool_selection_accuracy": 0.80,
            "tool_execution_success": 0.70,
            "task_completion_rate": 0.60,
            "format_compliance": 0.98,
            "hallucination_rate": 0.0,
            "canary_pass_rate": 0.95,
        }
        candidate = {
            "tool_selection_accuracy": 0.85,
            "tool_execution_success": 0.72,
            "task_completion_rate": 0.62,
            "format_compliance": 0.97,
            "hallucination_rate": 0.0,
            "canary_pass_rate": 0.95,
        }

        comp = compare_metrics(candidate, baseline)
        checks = verdict(comp)

        assert checks["overall"] is True
        assert checks["tool_selection"] is True
        assert checks["no_hallucinations"] is True

    def test_verdict_fail_regression(self):
        from eval import compare_metrics, verdict

        baseline = {"tool_selection_accuracy": 0.85, "tool_execution_success": 0.75,
                     "task_completion_rate": 0.65, "format_compliance": 0.98,
                     "hallucination_rate": 0.0, "canary_pass_rate": 0.95}
        candidate = {"tool_selection_accuracy": 0.70, "tool_execution_success": 0.60,
                      "task_completion_rate": 0.50, "format_compliance": 0.90,
                      "hallucination_rate": 0.05, "canary_pass_rate": 0.80}

        comp = compare_metrics(candidate, baseline)
        checks = verdict(comp)

        assert checks["overall"] is False
        assert checks["no_hallucinations"] is False

    def test_format_report(self, tmp_hermes):
        from eval import format_report

        metrics = {"tool_selection_accuracy": 0.85, "format_compliance": 0.97,
                    "total_cases": 50}
        report = format_report(metrics, cluster_id="c-test", version="v1")
        assert "FINETUNE BENCH" in report


# ============================================================================
# Routing tests
# ============================================================================

class TestRouting:
    def test_routing_disabled(self, tmp_hermes):
        from route import AdapterRouter

        router = AdapterRouter(config={
            "clustering": {"embedding_model": "test", "confidence_threshold": 0.6},
            "routing": {"enabled": False, "providers": []},
        })
        result = router.route("test prompt")
        assert result["fallback"] is True
        assert result["cluster_id"] is None

    def test_should_route_local(self, tmp_hermes):
        from route import AdapterRouter

        router = AdapterRouter(config={
            "clustering": {"embedding_model": "test", "confidence_threshold": 0.6},
            "routing": {"enabled": True, "providers": ["local", "llama-cpp"]},
        })
        assert router.should_route("local") is True
        assert router.should_route("openrouter") is False
        assert router.should_route("llama-cpp") is True


# ============================================================================
# Common utilities tests
# ============================================================================

class TestCommon:
    def test_ensure_dirs(self, tmp_hermes):
        import common
        common.ensure_dirs()
        assert common.EXTRACTED_DIR.exists()
        assert common.SCORED_DIR.exists()
        assert common.ADAPTERS_DIR.exists()

    def test_json_roundtrip(self, tmp_hermes):
        import common
        test_path = common.FINETUNE_DIR / "test.json"
        data = {"key": "value", "nested": {"a": 1}}
        common.save_json(test_path, data)
        loaded = common.load_json(test_path)
        assert loaded == data

    def test_jsonl_roundtrip(self, tmp_hermes):
        import common
        test_path = common.FINETUNE_DIR / "test.jsonl"
        records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        common.append_jsonl(test_path, records)
        loaded = common.read_jsonl(test_path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1

    def test_load_config_defaults(self, tmp_hermes):
        import common
        config = common.load_config()
        assert config["enabled"] is True
        assert config["extract"]["min_turns"] == 2
        assert config["scoring"]["weights"]["turn_signal"] == 0.4


# ============================================================================
# Retro tests
# ============================================================================

class TestRetro:
    def _make_scored(self, sid, turns, composite, tool_calls=0, days_old=1):
        """Helper to build a scored session for retro tests."""
        from datetime import datetime, timedelta, timezone
        started = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        return {
            "session_id": sid,
            "started_at": started,
            "turns": turns,
            "metadata": {
                "source": "cli",
                "model": "test",
                "tool_call_count": tool_calls,
                "total_tokens": 100,
            },
            "scoring": {
                "composite_score": composite,
                "bucket": "good" if composite >= 0.7 else (
                    "neutral" if composite >= 0.4 else "bad"
                ),
            },
        }

    def test_priority_uncertainty_peaks_at_neutral(self, tmp_hermes):
        from retro import compute_priority

        neutral = self._make_scored("n", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.5)
        confident_good = self._make_scored("g", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.95)

        # Neutral session should have higher priority than a confident one
        # because the human label resolves more ambiguity.
        assert compute_priority(neutral) > compute_priority(confident_good)

    def test_priority_recency_decay(self, tmp_hermes):
        from retro import compute_priority

        recent = self._make_scored("r", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.5, days_old=1)
        old = self._make_scored("o", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.5, days_old=60)

        # Recent should outrank old at the same uncertainty
        assert compute_priority(recent) > compute_priority(old)

    def test_priority_tool_density_boost(self, tmp_hermes):
        from retro import compute_priority

        no_tools = self._make_scored("nt", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.5, tool_calls=0)
        many_tools = self._make_scored("mt", [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ], composite=0.5, tool_calls=10)

        assert compute_priority(many_tools) > compute_priority(no_tools)

    def test_parse_turn_spec_simple(self, tmp_hermes):
        from retro import parse_turn_spec
        assert parse_turn_spec("1,3,5", 10) == [1, 3, 5]
        assert parse_turn_spec("2", 10) == [2]
        assert parse_turn_spec("", 10) == list(range(1, 11))

    def test_parse_turn_spec_range(self, tmp_hermes):
        from retro import parse_turn_spec
        assert parse_turn_spec("2-5", 10) == [2, 3, 4, 5]
        assert parse_turn_spec("1,3-5,8", 10) == [1, 3, 4, 5, 8]

    def test_parse_turn_spec_clamps_to_max(self, tmp_hermes, capsys):
        from retro import parse_turn_spec
        # Out-of-range turns are dropped with a warning
        result = parse_turn_spec("1,3,99", 5)
        assert result == [1, 3]

    def test_label_writes_session_marker_and_per_turn(self, tmp_hermes):
        """Session-level label expands to per-turn entries plus a session marker."""
        from retro import write_label, load_feedback
        write_label("test-sid", "good", 1.0)  # session marker
        write_label("test-sid", "good", 1.0, turn_index=1)
        write_label("test-sid", "good", 1.0, turn_index=2)

        feedback = load_feedback()
        assert len(feedback) == 3
        # One session-level marker + two turn-level
        session_level = [r for r in feedback if r.get("turn_index") is None]
        turn_level = [r for r in feedback if r.get("turn_index") is not None]
        assert len(session_level) == 1
        assert len(turn_level) == 2

    def test_labeled_session_ids_excludes_skip(self, tmp_hermes):
        """A skip record should not count as a label."""
        from retro import write_label, load_feedback, labeled_session_ids, skipped_session_ids
        write_label("good-sid", "good", 1.0)
        write_label("skip-sid", "skip", 0.5)

        feedback = load_feedback()
        labeled = labeled_session_ids(feedback)
        skipped = skipped_session_ids(feedback)

        assert "good-sid" in labeled
        assert "skip-sid" not in labeled
        assert "skip-sid" in skipped

    def test_score_session_honors_per_turn_overrides(self, tmp_hermes):
        """The scorer should pick up retro turn-level labels via feedback.jsonl."""
        from retro import write_label
        from score import QualityScorer

        # Write a turn-level label for assistant turn 1 of session 'override-test'
        write_label("override-test", "good", 1.0, turn_index=1)

        scorer = QualityScorer()
        session = {
            "session_id": "override-test",
            "started_at": "2026-01-01T00:00:00",
            "turns": [
                {"role": "user", "content": "What is X?"},
                {"role": "assistant", "content": "X is a thing."},
                {"role": "user", "content": "tell me more"},
            ],
            "metadata": {"source": "cli", "model": "test", "tool_call_count": 0, "total_tokens": 50},
        }
        scored = scorer.score_session(session)

        # Turn 1 (the assistant) should have the override score 1.0
        turn_scores = scored["scoring"]["turn_scores"]
        # turn_scores is List[Tuple[msg_idx, score]] — find the assistant entry
        assistant_score = next(s for idx, s in turn_scores if idx == 1)
        assert assistant_score == 1.0

    def test_load_all_scored_dedupes_by_id(self, tmp_hermes):
        """Re-scoring a session shouldn't duplicate it in the queue."""
        from retro import load_all_scored
        import common

        s1 = {"session_id": "dup", "started_at": "2026-01-01T00:00:00",
              "turns": [], "metadata": {}, "scoring": {"composite_score": 0.4}}
        s2 = {"session_id": "dup", "started_at": "2026-01-01T00:00:00",
              "turns": [], "metadata": {}, "scoring": {"composite_score": 0.7}}

        common.append_jsonl(common.SCORED_DIR / "scored_a.jsonl", [s1])
        common.append_jsonl(common.SCORED_DIR / "scored_b.jsonl", [s2])

        result = load_all_scored()
        ids = [s["session_id"] for s in result]
        assert ids.count("dup") == 1


# ============================================================================
# Turn-based extraction tests (replaces session-as-atomic-unit model)
# ============================================================================

class TestTurnExtraction:
    def _session_with_turn_scores(self, sid, turns, turn_scores=None):
        """Build a scored session with explicit per-turn scores."""
        return {
            "session_id": sid,
            "started_at": "2026-01-01T00:00:00",
            "turns": turns,
            "metadata": {"source": "cli", "model": "test", "tool_call_count": 0, "total_tokens": 100},
            "scoring": {
                "composite_score": 0.5,
                "bucket": "neutral",
                "turn_scores": turn_scores or [],
            },
        }

    def test_one_assistant_turn_one_example(self, tmp_hermes):
        from format import extract_training_turns

        session = self._session_with_turn_scores("s1", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ], turn_scores=[(2, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        assert len(examples) == 1
        # The single example should have system + user + assistant
        convs = examples[0]["conversations"]
        assert convs[0]["from"] == "system"
        assert convs[1]["from"] == "human"
        assert convs[2]["from"] == "gpt"
        assert convs[2]["value"] == "Hi there"

    def test_multiple_assistant_turns_multiple_examples(self, tmp_hermes):
        from format import extract_training_turns

        # 3 user/assistant exchanges, all scored good
        session = self._session_with_turn_scores("s2", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
        ], turn_scores=[(2, 0.9), (4, 0.9), (6, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        # Should produce 3 training examples, one per assistant turn
        assert len(examples) == 3

        # Each example must end on a gpt turn
        for ex in examples:
            assert ex["conversations"][-1]["from"] == "gpt"

        # The targets should be A1, A2, A3 in order
        targets = [ex["conversations"][-1]["value"] for ex in examples]
        assert targets == ["A1", "A2", "A3"]

    def test_low_score_turns_filtered_out(self, tmp_hermes):
        from format import extract_training_turns

        # Two assistant turns, only one is good
        session = self._session_with_turn_scores("s3", [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1 (bad)"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2 (good)"},
        ], turn_scores=[(2, 0.2), (4, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        assert len(examples) == 1
        assert examples[0]["conversations"][-1]["value"] == "A2 (good)"

    def test_context_window_truncation(self, tmp_hermes):
        from format import extract_training_turns

        # Long session — 10 user/assistant exchanges
        turns = [{"role": "system", "content": "sys"}]
        for i in range(10):
            turns.append({"role": "user", "content": f"Q{i}"})
            turns.append({"role": "assistant", "content": f"A{i}"})

        scores = [(i, 0.9) for i, t in enumerate(turns) if t["role"] == "assistant"]
        session = self._session_with_turn_scores("s4", turns, turn_scores=scores)

        # Use a small window
        examples = extract_training_turns(session, context_window_turns=4, min_turn_score=0.7)
        assert len(examples) == 10

        # The LAST example should have system + last few turns + target
        last = examples[-1]
        # Should always have system at position 0
        assert last["conversations"][0]["from"] == "system"
        # Window of 4 means at most 4 preceding turns + target = 5 + system = 6 total
        assert len(last["conversations"]) <= 6
        # The target should be the final assistant turn
        assert last["conversations"][-1]["value"] == "A9"

    def test_target_must_have_content(self, tmp_hermes):
        from format import extract_training_turns

        # Empty assistant turn — not trainable
        session = self._session_with_turn_scores("s5", [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": ""},
        ], turn_scores=[(2, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        assert len(examples) == 0

    def test_must_have_user_in_context(self, tmp_hermes):
        from format import extract_training_turns

        # Assistant turn with no preceding user — not a valid example
        session = self._session_with_turn_scores("s6", [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "Spontaneous"},
        ], turn_scores=[(1, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        assert len(examples) == 0

    def test_train_eval_split_keeps_session_together(self, tmp_hermes):
        """All turns from the same session should land in the same split."""
        from format import TrainingFormatter

        # 20 sessions, each with 3 trainable turns
        sessions = []
        for i in range(20):
            turns = [{"role": "system", "content": "sys"}]
            for j in range(3):
                turns.append({"role": "user", "content": f"Q{i}.{j}"})
                turns.append({"role": "assistant", "content": f"A{i}.{j}"})
            sessions.append(self._session_with_turn_scores(
                f"sess-{i}",
                turns,
                turn_scores=[(idx, 0.9) for idx, t in enumerate(turns) if t["role"] == "assistant"],
            ))

        formatter = TrainingFormatter(eval_ratio=0.2)
        counts = formatter.format_for_cluster(sessions, "_general", min_score=0.7)

        # Each session produces 3 examples, total 60 records
        assert counts["train"] + counts["eval"] == 60

        # Verify session_id grouping: read back the JSONL files and check
        # no session_id appears in both train and eval
        from common import CLUSTERS_DIR, read_jsonl
        train_sids = {r["session_id"] for r in read_jsonl(CLUSTERS_DIR / "_general" / "train.jsonl")}
        eval_sids = {r["session_id"] for r in read_jsonl(CLUSTERS_DIR / "_general" / "eval.jsonl")}
        assert not (train_sids & eval_sids), "Session IDs leaked across train/eval splits"

    def test_per_session_count_in_record(self, tmp_hermes):
        """Each record should know which session and turn it came from."""
        from format import extract_training_turns

        session = self._session_with_turn_scores("trace-test", [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ], turn_scores=[(1, 0.9)])

        examples = extract_training_turns(session, min_turn_score=0.7)
        assert examples[0]["session_id"] == "trace-test"
        assert "turn_index_in_session" in examples[0]
        assert examples[0]["score"] == 0.9


# ============================================================================
# Positive signal tests (Phase 1 — see hermes-finetune-positive-signals-spec)
# ============================================================================

class TestPositiveSignals:
    def _turn(self, role, content="", tool_calls=None, tool_name=None):
        t = {"role": role, "content": content}
        if tool_calls:
            t["tool_calls"] = tool_calls
        if tool_name:
            t["tool_name"] = tool_name
        return t

    # ---- Tool result classification ----

    def test_tool_result_status_success(self, tmp_hermes):
        from score import _tool_result_status
        assert _tool_result_status("file content here") == "success"
        assert _tool_result_status('{"total_count": 5, "files": ["a.py"]}') == "success"

    def test_tool_result_status_soft_failure(self, tmp_hermes):
        from score import _tool_result_status
        assert _tool_result_status('{"total_count": 0}') == "soft_failure"
        assert _tool_result_status('{"results": []}') == "soft_failure"
        assert _tool_result_status("No matches found") == "soft_failure"

    def test_tool_result_status_hard_failure(self, tmp_hermes):
        from score import _tool_result_status
        assert _tool_result_status('{"error": "permission denied"}') == "hard_failure"
        assert _tool_result_status("Traceback (most recent call last):") == "hard_failure"
        assert _tool_result_status("command not found") == "hard_failure"

    # ---- Tool success chain ----

    def test_tool_success_chain_no_tools(self, tmp_hermes):
        from score import positive_tool_success_chain
        turns = [
            self._turn("user", "hello"),
            self._turn("assistant", "hi there"),
        ]
        assert positive_tool_success_chain(turns, 1) == 0.0

    def test_tool_success_chain_full_chain(self, tmp_hermes):
        from score import positive_tool_success_chain
        turns = [
            self._turn("user", "list files in /tmp"),
            self._turn("assistant", "Looking now", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{"cmd": "ls /tmp"}'}}
            ]),
            self._turn("tool", "file1.txt\nfile2.txt", tool_name="terminal"),
            self._turn("user", "great, now show me file1.txt"),
        ]
        score = positive_tool_success_chain(turns, 1)
        # Tool succeeded (0.5) + user did not correct (0.7) + user advanced
        # topic with >8 words (0.9). The reference check requires the
        # extracted artifact tokens to appear in the user message — file1.txt
        # is in the user msg, so we expect 1.0.
        assert score >= 0.9

    def test_tool_success_chain_failure(self, tmp_hermes):
        from score import positive_tool_success_chain
        turns = [
            self._turn("user", "do thing"),
            self._turn("assistant", "trying", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{}'}}
            ]),
            self._turn("tool", '{"error": "command not found"}'),
        ]
        assert positive_tool_success_chain(turns, 1) == 0.0

    def test_tool_success_chain_soft_failure(self, tmp_hermes):
        """Empty search results count as failure even though tool returned cleanly."""
        from score import positive_tool_success_chain
        turns = [
            self._turn("user", "find foo"),
            self._turn("assistant", "searching", tool_calls=[
                {"function": {"name": "search_files", "arguments": '{"query": "foo"}'}}
            ]),
            self._turn("tool", '{"total_count": 0}'),
        ]
        assert positive_tool_success_chain(turns, 1) == 0.0

    # ---- Artifact longevity ----

    def test_artifact_longevity_path_referenced(self, tmp_hermes):
        from score import positive_artifact_longevity
        turns = [
            self._turn("user", "create a script"),
            self._turn("assistant", "I'll write src/main.py for you"),
            self._turn("user", "thanks"),
            self._turn("assistant", "now editing src/main.py"),
        ]
        score = positive_artifact_longevity(turns, 1)
        # span = 3 turns from intro to last reference, in the 3-5 bucket → 0.7
        assert score >= 0.5

    def test_artifact_longevity_no_reference(self, tmp_hermes):
        from score import positive_artifact_longevity
        turns = [
            self._turn("user", "what is python"),
            self._turn("assistant", "Python is a programming language"),
        ]
        # No artifacts at all
        assert positive_artifact_longevity(turns, 1) == 0.0

    def test_artifact_longevity_modified_bonus(self, tmp_hermes):
        from score import positive_artifact_longevity
        turns = [
            self._turn("user", "make a config"),
            self._turn("assistant", "I created config.yaml"),
            self._turn("user", "update it"),
            self._turn("assistant", "updating", tool_calls=[
                {"function": {"name": "patch", "arguments": '{"path": "config.yaml"}'}}
            ]),
        ]
        score = positive_artifact_longevity(turns, 1)
        # Span 3 turns + modified bonus → at least 0.6
        assert score >= 0.6

    # ---- Self-correction ----

    def test_self_correction_after_tool_failure(self, tmp_hermes):
        from score import positive_self_correction
        turns = [
            self._turn("user", "list files matching foo"),
            self._turn("assistant", "trying", tool_calls=[
                {"function": {"name": "search_files", "arguments": '{}'}}
            ]),
            self._turn("tool", '{"total_count": 0}'),
            self._turn("user", "search for the pattern foo* in src/ instead"),
            self._turn("assistant", "retrying", tool_calls=[
                {"function": {"name": "search_files", "arguments": '{}'}}
            ]),
            self._turn("tool", '{"total_count": 3, "files": ["a.py", "b.py", "c.py"]}'),
            self._turn("user", "perfect, now read a.py"),
        ]
        # Index 4 is the corrected response
        score = positive_self_correction(turns, 4)
        assert score == 0.9

    def test_self_correction_not_a_correction(self, tmp_hermes):
        """When previous turn succeeded, this isn't a correction context."""
        from score import positive_self_correction
        turns = [
            self._turn("user", "list files"),
            self._turn("assistant", "ok", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{}'}}
            ]),
            self._turn("tool", "a.py b.py"),
            self._turn("user", "read a.py"),
            self._turn("assistant", "ok", tool_calls=[
                {"function": {"name": "read_file", "arguments": '{}'}}
            ]),
        ]
        assert positive_self_correction(turns, 4) is None

    # ---- Resolution detection ----

    def test_detect_resolution_tool_chain(self, tmp_hermes):
        from score import detect_resolution
        turns = [
            self._turn("user", "create the file"),
            self._turn("assistant", "ok", tool_calls=[
                {"function": {"name": "write_file", "arguments": '{}'}}
            ]),
            self._turn("tool", "file written"),
            self._turn("user", "and now run it"),
            self._turn("assistant", "running", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{}'}}
            ]),
            self._turn("tool", "output: hello"),
        ]
        # Last assistant turn had a successful tool call, no user response after
        idx = detect_resolution(turns)
        assert idx == 4

    def test_detect_resolution_unresolved(self, tmp_hermes):
        from score import detect_resolution
        turns = [
            self._turn("user", "do thing"),
            self._turn("assistant", "trying", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{}'}}
            ]),
            self._turn("tool", '{"error": "failed"}'),
            self._turn("user", "that didn't work"),
        ]
        # User responded after the tool failed → not resolved
        assert detect_resolution(turns) is None

    def test_detect_resolution_too_short(self, tmp_hermes):
        from score import detect_resolution
        turns = [
            self._turn("user", "hi"),
            self._turn("assistant", "hello"),
        ]
        assert detect_resolution(turns) is None

    # ---- No-tool response (Signal 6) ----

    def test_no_tool_response_user_accepts(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "what's the difference between a list and a tuple in python"),
            self._turn("assistant", "Lists are mutable, tuples are immutable. Tuples are also "
                                    "hashable so they can be used as dict keys."),
            self._turn("user", "thanks, perfect"),
        ]
        score = positive_no_tool_response(turns, 1)
        assert score is not None and score >= 0.85

    def test_no_tool_response_user_demands_tool(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "what does main.py do"),
            self._turn("assistant", "It probably defines a main function and prints something."),
            self._turn("user", "actually read the file and tell me"),
        ]
        score = positive_no_tool_response(turns, 1)
        assert score == 0.0

    def test_no_tool_response_user_corrects(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "what's 2 + 2"),
            self._turn("assistant", "The answer is 5, here's a longer explanation about it"),
            self._turn("user", "no, that's wrong"),
        ]
        score = positive_no_tool_response(turns, 1)
        assert score == 0.0

    def test_no_tool_response_skipped_for_tool_call(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "list files"),
            self._turn("assistant", "Looking", tool_calls=[
                {"function": {"name": "terminal", "arguments": '{"cmd": "ls"}'}}
            ]),
            self._turn("user", "thanks"),
        ]
        # Tool turn — let other signals handle it
        assert positive_no_tool_response(turns, 1) is None

    def test_no_tool_response_user_adds_constraints(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "explain decorators"),
            self._turn("assistant", "A decorator is a function that wraps another function "
                                    "to extend its behavior without modifying it."),
            self._turn("user", "okay but I specifically need to know how to write one that "
                               "takes arguments and preserves the wrapped function's signature "
                               "and also works with async functions"),
        ]
        score = positive_no_tool_response(turns, 1)
        # New constraints → ambiguous (0.4), not full credit
        assert score == 0.4

    def test_no_tool_response_final_turn(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "what's the capital of france"),
            self._turn("assistant", "The capital of France is Paris."),
        ]
        score = positive_no_tool_response(turns, 1)
        assert score == 0.6

    def test_no_tool_response_skips_trivial_content(self, tmp_hermes):
        from score import positive_no_tool_response
        turns = [
            self._turn("user", "ok"),
            self._turn("assistant", "ok"),
            self._turn("user", "thanks"),
        ]
        # Below 20-char threshold — not a meaningful no-tool decision
        assert positive_no_tool_response(turns, 1) is None

    # ---- End-to-end scoring with the new mode ----

    def test_score_session_positive_mode_with_tool_success(self, tmp_hermes):
        """A session with successful tool chains should score in good bucket."""
        from score import QualityScorer
        scorer = QualityScorer()
        session = {
            "session_id": "tool-success-session",
            "started_at": "2026-01-01T00:00:00",
            "turns": [
                {"role": "user", "content": "list the python files in src/"},
                {"role": "assistant", "content": "Looking", "tool_calls": [
                    {"function": {"name": "terminal", "arguments": '{"cmd": "ls src/*.py"}'}}
                ]},
                {"role": "tool", "content": "src/main.py\nsrc/utils.py", "tool_name": "terminal"},
                {"role": "user", "content": "now read main.py and tell me what it does"},
                {"role": "assistant", "content": "Reading", "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'}}
                ]},
                {"role": "tool", "content": "def main(): print('hello')", "tool_name": "read_file"},
                {"role": "assistant", "content": "It defines a main function that prints hello"},
            ],
            "metadata": {"source": "cli", "model": "test", "tool_call_count": 2, "total_tokens": 200},
        }
        scored = scorer.score_session(session)
        assert scored["scoring"]["scoring_mode"] == "positive_signals"
        # At least one assistant turn should score in the good range
        turn_scores = scored["scoring"]["turn_scores"]
        assert any(s >= 0.7 for _, s in turn_scores), \
            f"expected at least one good-scored turn, got {turn_scores}"

    def test_score_session_legacy_mode_still_works(self, tmp_hermes):
        from score import QualityScorer
        scorer = QualityScorer(config={"mode": "legacy", "weights": {}, "thresholds": {}})
        session = {
            "session_id": "legacy-session",
            "started_at": "2026-01-01T00:00:00",
            "turns": [
                {"role": "user", "content": "what is X"},
                {"role": "assistant", "content": "X is a thing"},
                {"role": "user", "content": "thanks, that helps a lot"},
            ],
            "metadata": {"source": "cli", "model": "test"},
        }
        scored = scorer.score_session(session)
        assert scored["scoring"]["scoring_mode"] == "legacy"
        assert "composite_score" in scored["scoring"]


# ============================================================================
# Auto-redeploy tests (HF snapshot detection, conversion, server lifecycle)
# ============================================================================

class TestRedeploy:
    def test_find_base_snapshot_returns_none_for_unknown(self, tmp_path, monkeypatch):
        from manage import find_base_snapshot
        # Empty cache → None
        monkeypatch.setenv("HOME", str(tmp_path))
        result = find_base_snapshot("nonexistent/repo")
        assert result is None

    def test_find_base_snapshot_locates_real_dir(self, tmp_path, monkeypatch):
        from manage import find_base_snapshot
        # Create a fake HF cache layout
        cache = tmp_path / ".cache" / "huggingface" / "hub"
        snap_dir = cache / "models--myorg--mymodel" / "snapshots" / "abc123"
        snap_dir.mkdir(parents=True)
        (snap_dir / "config.json").write_text("{}")

        monkeypatch.setenv("HOME", str(tmp_path))
        result = find_base_snapshot("myorg/mymodel")
        assert result == snap_dir

    def test_find_base_snapshot_picks_most_recent(self, tmp_path, monkeypatch):
        import time
        from manage import find_base_snapshot
        cache = tmp_path / ".cache" / "huggingface" / "hub"
        snap_root = cache / "models--myorg--mymodel" / "snapshots"
        snap_root.mkdir(parents=True)

        old = snap_root / "old_hash"
        new = snap_root / "new_hash"
        old.mkdir()
        time.sleep(0.05)
        new.mkdir()

        monkeypatch.setenv("HOME", str(tmp_path))
        result = find_base_snapshot("myorg/mymodel")
        assert result == new

    def test_find_base_snapshot_rejects_bad_id(self, tmp_path, monkeypatch):
        from manage import find_base_snapshot
        monkeypatch.setenv("HOME", str(tmp_path))
        assert find_base_snapshot("") is None
        assert find_base_snapshot("no_slash_in_this_id") is None

    def test_convert_adapter_to_gguf_caches(self, tmp_path):
        """If the GGUF already exists, convert returns it without re-running."""
        from manage import convert_adapter_to_gguf
        adapter_dir = tmp_path / "adapters" / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)
        existing = adapter_dir / "adapter.gguf"
        existing.write_bytes(b"fake gguf")

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        converter = tmp_path / "converter.py"
        converter.write_text("#!/usr/bin/env python")

        result = convert_adapter_to_gguf(adapter_dir, snapshot, converter, force=False)
        assert result == existing
        # File contents are unchanged (we didn't actually run anything)
        assert existing.read_bytes() == b"fake gguf"

    def test_convert_adapter_to_gguf_missing_converter(self, tmp_path):
        from manage import convert_adapter_to_gguf
        adapter_dir = tmp_path / "adapters" / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()

        with pytest.raises(RuntimeError, match="Converter not found"):
            convert_adapter_to_gguf(adapter_dir, snapshot, tmp_path / "missing.py")

    def test_convert_adapter_to_gguf_missing_snapshot(self, tmp_path):
        from manage import convert_adapter_to_gguf
        adapter_dir = tmp_path / "adapters" / "v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)

        converter = tmp_path / "converter.py"
        converter.write_text("#!/usr/bin/env python")

        with pytest.raises(RuntimeError, match="Base snapshot not found"):
            convert_adapter_to_gguf(adapter_dir, tmp_path / "missing", converter)

    def test_convert_adapter_to_gguf_missing_adapter_model(self, tmp_path):
        from manage import convert_adapter_to_gguf
        adapter_dir = tmp_path / "adapters" / "v1"
        adapter_dir.mkdir(parents=True)
        # Note: no adapter_model subdir

        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        converter = tmp_path / "converter.py"
        converter.write_text("#!/usr/bin/env python")

        with pytest.raises(RuntimeError, match="Adapter model dir not found"):
            convert_adapter_to_gguf(adapter_dir, snapshot, converter)

    def test_stop_llama_server_no_pid_file(self, tmp_path):
        """Stopping when there's no PID file and no running server is a no-op."""
        from manage import stop_llama_server
        # This should not raise even though there's no llama-server to kill
        result = stop_llama_server(pid_file=tmp_path / "nonexistent.pid")
        assert result in (True, False)  # either is acceptable

    def test_stop_llama_server_stale_pid(self, tmp_path):
        """A stale PID file should be cleaned up without crashing."""
        from manage import stop_llama_server
        pid_file = tmp_path / "stale.pid"
        pid_file.write_text("99999999")  # almost certainly not a real PID
        stop_llama_server(pid_file=pid_file)
        assert not pid_file.exists()  # cleaned up

    def test_health_check_llama_server_failure(self):
        """Health check on a non-existent server returns False quickly."""
        from manage import health_check_llama_server
        result = health_check_llama_server("http://localhost:1/nonexistent", timeout=2)
        assert result is False

    def test_redeploy_no_active_adapter(self, tmp_hermes):
        """redeploy() with no active adapter and no explicit dir returns False."""
        from manage import redeploy
        result = redeploy()
        assert result is False

    def test_redeploy_missing_adapter_dir(self, tmp_hermes, tmp_path):
        """redeploy() with an adapter_dir that lacks adapter_model returns False."""
        from manage import redeploy
        bogus = tmp_path / "bogus_adapter"
        bogus.mkdir()
        result = redeploy(adapter_dir=bogus)
        assert result is False

    def test_redeploy_no_snapshot_no_converter(self, tmp_hermes, tmp_path, monkeypatch):
        """redeploy() with auto snapshot lookup but empty HF cache returns False."""
        from manage import redeploy
        # Set up an adapter dir
        adapter_dir = tmp_path / "adapter_v1"
        (adapter_dir / "adapter_model").mkdir(parents=True)
        # Point HOME at an empty cache
        monkeypatch.setenv("HOME", str(tmp_path))

        result = redeploy(adapter_dir=adapter_dir)
        assert result is False  # snapshot detection should fail

    def test_bench_passes_skips_smoke_baseline(self, tmp_hermes):
        """bench_passes() should not pick a 2-case smoke test as a baseline
        for a 243-case full bench. The case count must be within 10% of the
        candidate's case count."""
        from manage import bench_passes
        import common
        results = common.BENCH_DIR / "results"
        results.mkdir(parents=True, exist_ok=True)

        # Smoke test with 2 cases
        smoke = results / "bench_20260101_000001.json"
        common.save_json(smoke, {
            "metrics": {"total_cases": 2, "tool_selection_accuracy": 1.0,
                        "tool_execution_success": 0.0, "task_completion_rate": 0.0,
                        "format_compliance": 1.0, "no_tool_accuracy": 1.0,
                        "hallucination_rate": 0.0, "canary_pass_rate": 0.0},
            "cases": [],
        })

        # Real prior baseline with 243 cases
        import time
        time.sleep(0.05)
        real_baseline = results / "bench_20260101_000002.json"
        common.save_json(real_baseline, {
            "metrics": {"total_cases": 243, "tool_selection_accuracy": 0.80,
                        "tool_execution_success": 0.90, "task_completion_rate": 0.62,
                        "format_compliance": 1.0, "no_tool_accuracy": 0.97,
                        "hallucination_rate": 0.0, "canary_pass_rate": 0.89},
            "cases": [],
        })

        # Newer smoke test (most recent — would be picked by old logic)
        time.sleep(0.05)
        newer_smoke = results / "bench_20260101_000003.json"
        common.save_json(newer_smoke, {
            "metrics": {"total_cases": 5, "tool_selection_accuracy": 1.0,
                        "tool_execution_success": 1.0, "task_completion_rate": 0.0,
                        "format_compliance": 1.0, "no_tool_accuracy": 1.0,
                        "hallucination_rate": 0.0, "canary_pass_rate": 0.0},
            "cases": [],
        })

        # Candidate is a fresh 243-case run
        time.sleep(0.05)
        candidate = results / "bench_20260101_000004.json"
        common.save_json(candidate, {
            "metrics": {"total_cases": 243, "tool_selection_accuracy": 0.79,
                        "tool_execution_success": 0.97, "task_completion_rate": 0.67,
                        "format_compliance": 1.0, "no_tool_accuracy": 0.79,
                        "hallucination_rate": 0.0, "canary_pass_rate": 0.71},
            "cases": [],
        })

        passed, report = bench_passes(candidate)
        # The report should reference the real 243-case baseline, not either smoke test
        assert "bench_20260101_000002.json" in report
        # Smoke tests should NOT appear as the baseline filename
        assert "bench_20260101_000001.json" not in report
        assert "bench_20260101_000003.json" not in report

    def test_bench_passes_no_comparable_baseline(self, tmp_hermes):
        """When no prior result has a similar case count, treat as new baseline."""
        from manage import bench_passes
        import common
        results = common.BENCH_DIR / "results"
        results.mkdir(parents=True, exist_ok=True)

        # Only smoke tests exist
        smoke = results / "bench_smoke.json"
        common.save_json(smoke, {
            "metrics": {"total_cases": 2, "tool_selection_accuracy": 1.0,
                        "tool_execution_success": 0.0, "task_completion_rate": 0.0,
                        "format_compliance": 1.0, "no_tool_accuracy": 1.0,
                        "hallucination_rate": 0.0, "canary_pass_rate": 0.0},
            "cases": [],
        })

        import time
        time.sleep(0.05)
        # Candidate is a 243-case run
        candidate = results / "bench_full.json"
        common.save_json(candidate, {
            "metrics": {"total_cases": 243, "tool_selection_accuracy": 0.80},
            "cases": [],
        })

        passed, report = bench_passes(candidate)
        # No comparable baseline → pass with a "new baseline" message
        assert passed is True
        assert "new baseline" in report.lower()


# ============================================================================
# Installed-skill integration
# ============================================================================

class TestInstalledSkill:
    """The official install path copies only the skill bundle (see
    tools/skills_hub.py OptionalSkillSource.fetch). Everything /finetune
    needs — including the bench env — must live inside the bundle and
    resolve correctly from the installed location."""

    SKILL_SRC = Path(__file__).resolve().parent.parent / "optional-skills" / "mlops" / "finetune"

    def _install_skill(self, tmp_path):
        """Mirror OptionalSkillSource.fetch() + install_from_quarantine():
        copy every non-hidden, non-pyc file under the skill dir."""
        import shutil

        dest = tmp_path / "skills" / "mlops" / "finetune"
        for f in self.SKILL_SRC.rglob("*"):
            if (
                f.is_file()
                and not f.name.startswith(".")
                and "__pycache__" not in f.parts
                and f.suffix != ".pyc"
            ):
                rel = f.relative_to(self.SKILL_SRC)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
        return dest

    def test_bench_assets_ship_with_the_bundle(self, tmp_path):
        """The bench env, config, and prompt bank survive installation."""
        dest = self._install_skill(tmp_path)
        assert (dest / "bench" / "finetune_bench_env.py").exists()
        assert (dest / "bench" / "default.yaml").exists()
        assert (dest / "bench" / "prompt_bank.yaml").exists()
        assert (dest / "scripts" / "manage.py").exists()

    def test_manage_resolves_bench_from_installed_skill(self, tmp_path):
        """manage.py finds the bench env relative to the installed skill,
        not the repo checkout. Runs in a subprocess so the repo-path
        `manage` module already imported by other tests isn't reused."""
        import subprocess

        dest = self._install_skill(tmp_path)
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        code = (
            "import sys, json; "
            f"sys.path.insert(0, {str(dest / 'scripts')!r}); "
            "import manage; "
            "print(json.dumps({"
            "'env': str(manage.BENCH_ENV_SCRIPT), "
            "'env_exists': manage.BENCH_ENV_SCRIPT.exists(), "
            "'cfg_exists': manage.BENCH_DEFAULT_CONFIG.exists()}))"
        )
        env = dict(os.environ, HERMES_HOME=str(hermes_home))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        assert info["env_exists"] is True, info
        assert info["cfg_exists"] is True, info
        assert str(dest) in info["env"], info

    def test_bench_config_loads_from_installed_skill(self, tmp_path):
        """The bench env parses its default config and resolves the prompt
        bank from the installed location (no repo, no atroposlib)."""
        import subprocess

        dest = self._install_skill(tmp_path)
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(exist_ok=True)

        bench_dir = dest / "bench"
        code = (
            "import sys, json, importlib.util; "
            f"spec = importlib.util.spec_from_file_location('finetune_bench_env', {str(bench_dir / 'finetune_bench_env.py')!r}); "
            "mod = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(mod); "
            "from pathlib import Path; "
            f"cfg = mod.FinetuneBenchConfig.load(Path({str(bench_dir / 'default.yaml')!r})); "
            "bank = Path(cfg.prompt_bank_path); "
            "bank = bank if bank.is_absolute() else mod.BENCH_DIR / bank; "
            "import yaml; cases = yaml.safe_load(bank.read_text())['cases']; "
            "print(json.dumps({'n_cases': len(cases), 'backend': cfg.terminal_backend}))"
        )
        env = dict(os.environ, HERMES_HOME=str(hermes_home))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert proc.returncode == 0, proc.stderr
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        assert info["n_cases"] > 200, info
        assert info["backend"] == "docker", info
