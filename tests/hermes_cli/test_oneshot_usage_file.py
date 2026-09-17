"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json

import pytest

from hermes_cli.oneshot import _write_usage_file


def _result(**overrides):
    base = {
        "estimated_cost_usd": 0.1234,
        "cost_status": "estimated",
        "cost_source": "pricing-table",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 0,
        "reasoning_tokens": 50,
        "total_tokens": 1250,
        "api_calls": 3,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "abc123",
        "completed": True,
        "failed": False,
    }
    base.update(overrides)
    return base


class TestWriteUsageFile:
    def test_writes_report_with_cost_and_tokens(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["estimated_cost_usd"] == 0.1234
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 200
        assert report["model"] == "openai/gpt-5.5"
        assert report["api_calls"] == 3
        assert report["failed"] is False
        assert "failure" not in report

    def test_none_path_is_noop(self, tmp_path):
        # Must not raise and must not create a report file.
        _write_usage_file(None, _result())
        assert not (tmp_path / "usage.json").exists()

    def test_failure_marks_failed_and_records_message(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), {}, failure="boom")
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure"] == "boom"
        # Missing result fields serialize as null, not KeyError.
        assert report["estimated_cost_usd"] is None




class TestAuxiliaryLedger:
    """#112848: auxiliary LLM spend (title generation, vision, ...) recorded in session_model_usage
    belongs in the pipeline ledger, additively — the main-loop keys stay main-loop-only."""

    def test_aux_usage_is_a_separate_breakdown_and_main_keys_unchanged(self, tmp_path):
        from hermes_cli.oneshot import _auxiliary_usage, _attach_auxiliary_usage
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("root", "cli", model="main")
            # Rows from an earlier run of a resumed session must not count toward this run.
            db.record_auxiliary_usage("root", "vision", model="v", input_tokens=500, output_tokens=5,
                                      estimated_cost_usd=0.5)
            before = _auxiliary_usage(db, "root")
            # This run: aux billed to the id the turn started with; compression mints a child id.
            db.record_auxiliary_usage("root", "title_generation", model="t", input_tokens=40, output_tokens=8,
                                      estimated_cost_usd=0.001)
            db.create_session("child", "cli", model="main", parent_session_id="root")
            result = _result(session_id="child")
            _attach_auxiliary_usage(result, db, before)
        finally:
            db.close()

        path = tmp_path / "usage.json"
        _write_usage_file(str(path), result)
        report = json.loads(path.read_text())
        assert report["api_calls"] == 3 and report["total_tokens"] == 1250  # main loop untouched
        assert report["auxiliary"]["by_task"] == {
            "title_generation": {"api_calls": 1, "input_tokens": 40, "output_tokens": 8, "cache_read_tokens": 0,
                                 "cache_write_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.001},
        }
        assert report["auxiliary"]["total_tokens"] == 48
        assert report["total_including_auxiliary"] == {
            "api_calls": 4, "total_tokens": 1298, "estimated_cost_usd": pytest.approx(0.1244),
        }

    def test_waits_for_in_flight_title_thread(self, tmp_path):
        import threading
        import time

        from agent import title_generator
        from hermes_cli.oneshot import _attach_auxiliary_usage
        from hermes_state import SessionDB

        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("s", "cli", model="main")

            def late_row():
                time.sleep(0.2)
                db.record_auxiliary_usage("s", "title_generation", model="t", input_tokens=7, output_tokens=1)

            thread = threading.Thread(target=late_row, name="auto-title")
            title_generator._UPGRADE_THREADS.add(thread)
            thread.start()
            result = _result(session_id="s")
            _attach_auxiliary_usage(result, db, {})
        finally:
            db.close()
        assert result["auxiliary_usage"]["title_generation"]["api_calls"] == 1
