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
        # Turn score for the assistant turn should be low
        turn_scores = scored["scoring"]["turn_scores"]
        assert any(s < 0.3 for _, s in turn_scores)


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
        from common import SCORED_DIR, append_jsonl

        s1 = {"session_id": "dup", "started_at": "2026-01-01T00:00:00",
              "turns": [], "metadata": {}, "scoring": {"composite_score": 0.4}}
        s2 = {"session_id": "dup", "started_at": "2026-01-01T00:00:00",
              "turns": [], "metadata": {}, "scoring": {"composite_score": 0.7}}

        append_jsonl(SCORED_DIR / "scored_a.jsonl", [s1])
        append_jsonl(SCORED_DIR / "scored_b.jsonl", [s2])

        result = load_all_scored()
        ids = [s["session_id"] for s in result]
        assert ids.count("dup") == 1
