"""Tests for the morning brief composer script (issue #6).

Each test class is anchored to a specific Acceptance Criterion:
  AC-a  staleness detection for each of the three contracts
  AC-b  todo dedup logic
  AC-c  confidence threshold filtering
  AC-d  dev-category approval-routing annotation
  AC-e  all three individual degradation paths (file missing, file stale)
  AC-f  all-contracts-unavailable producing a valid (all-unavailable) brief
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT = REPO_ROOT / "scripts" / "morning_brief_composer.py"

sys.path.insert(0, str(REPO_ROOT))
from scripts.morning_brief_composer import (
    compose_brief,
    filter_todos,
    get_today_bangkok,
    load_contract,
    normalize_todo_text,
    render_dev_report_section,
    render_journal_section,
    render_todo_section,
    render_training_section,
    validate_freshness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return get_today_bangkok()


def _yesterday() -> str:
    from zoneinfo import ZoneInfo
    from datetime import datetime
    tz = ZoneInfo("Asia/Bangkok")
    now = datetime.now(tz)
    return (now.date() - timedelta(days=1)).isoformat()


def _journal_data(reflection="Morning thoughts.", todos=None, for_date=None):
    return {
        "for_date": for_date or _today(),
        "reflection": {"markdown": reflection},
        "todos": todos or [],
    }


def _perfcoach_data(advisories=None, for_date=None, **extra):
    d = {"for_date": for_date or _today()}
    if advisories is not None:
        d["advisories"] = advisories
    d.update(extra)
    return d


def _commander_data(completed=None, needs_review=None, dead_letter=None, cost="$0.12", for_date=None):
    return {
        "for_date": for_date or _today(),
        "completed": completed or [],
        "needs_review": needs_review or [],
        "dead_letter": dead_letter or [],
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# AC-a  staleness detection (validate_freshness)
# ---------------------------------------------------------------------------

class TestStalenessDetection:
    """AC-a: each contract's for_date is validated against today (Asia/Bangkok)."""

    def test_fresh_contract_passes(self):
        data = {"for_date": _today()}
        assert validate_freshness(data, _today()) is True

    def test_stale_contract_fails(self):
        data = {"for_date": _yesterday()}
        assert validate_freshness(data, _today()) is False

    def test_missing_for_date_fails(self):
        data = {}
        assert validate_freshness(data, _today()) is False

    def test_future_date_fails(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tomorrow = (datetime.now(ZoneInfo("Asia/Bangkok")).date() + timedelta(days=1)).isoformat()
        data = {"for_date": tomorrow}
        assert validate_freshness(data, _today()) is False

    def test_journal_staleness_independent(self, tmp_path):
        """Journal staleness does not affect perfcoach or commander rendering."""
        journal_path = tmp_path / "journal.json"
        journal_path.write_text(
            json.dumps({"for_date": _yesterday(), "reflection": {"markdown": "old"}, "todos": []}),
            encoding="utf-8",
        )
        perfcoach_path = tmp_path / "perf.json"
        perfcoach_path.write_text(json.dumps(_perfcoach_data(advisories=["Rest well."])), encoding="utf-8")
        commander_path = tmp_path / "cmd.json"
        commander_path.write_text(json.dumps(_commander_data(completed=["task-A"])), encoding="utf-8")

        j_data, j_reason = load_contract(journal_path)
        p_data, _ = load_contract(perfcoach_path)
        c_data, _ = load_contract(commander_path)

        brief = compose_brief(j_data, j_reason, p_data, "", c_data, "")
        assert "⚠️ unavailable" in brief       # journal section degraded
        assert "Rest well." in brief            # perfcoach section still renders
        assert "task-A" in brief                # commander section still renders

    def test_perfcoach_staleness_independent(self, tmp_path):
        """Perfcoach staleness does not affect journal or commander."""
        journal_path = tmp_path / "journal.json"
        journal_path.write_text(json.dumps(_journal_data(reflection="Fresh reflection.")), encoding="utf-8")
        perfcoach_path = tmp_path / "perf.json"
        perfcoach_path.write_text(
            json.dumps({"for_date": _yesterday(), "advisories": ["old advice"]}),
            encoding="utf-8",
        )
        commander_path = tmp_path / "cmd.json"
        commander_path.write_text(json.dumps(_commander_data(completed=["task-B"])), encoding="utf-8")

        j_data, j_reason = load_contract(journal_path)
        p_data, p_reason = load_contract(perfcoach_path)
        c_data, c_reason = load_contract(commander_path)

        brief = compose_brief(j_data, j_reason, p_data, p_reason, c_data, c_reason)
        assert "Fresh reflection." in brief
        assert "⚠️ unavailable" in brief        # perfcoach degraded
        assert "task-B" in brief

    def test_commander_staleness_independent(self, tmp_path):
        """Commander staleness does not affect journal or perfcoach."""
        journal_path = tmp_path / "journal.json"
        journal_path.write_text(json.dumps(_journal_data(reflection="Good morning.")), encoding="utf-8")
        perfcoach_path = tmp_path / "perf.json"
        perfcoach_path.write_text(json.dumps(_perfcoach_data(advisories=["Stay hydrated."])), encoding="utf-8")
        commander_path = tmp_path / "cmd.json"
        commander_path.write_text(
            json.dumps({"for_date": _yesterday(), "completed": [], "needs_review": [], "dead_letter": [], "cost": "$0"}),
            encoding="utf-8",
        )

        j_data, j_reason = load_contract(journal_path)
        p_data, p_reason = load_contract(perfcoach_path)
        c_data, c_reason = load_contract(commander_path)

        brief = compose_brief(j_data, j_reason, p_data, p_reason, c_data, c_reason)
        assert "Good morning." in brief
        assert "Stay hydrated." in brief
        assert "⚠️ unavailable" in brief        # commander degraded


# ---------------------------------------------------------------------------
# AC-b  todo dedup logic (normalize_todo_text + filter_todos)
# ---------------------------------------------------------------------------

class TestTodoDedup:
    """AC-b: todos are deduplicated by normalising text (case-insensitive, strip punctuation)."""

    def test_exact_duplicates_removed(self):
        todos = [
            {"text": "Fix the bug", "confidence": 0.9, "priority": 1, "category": "dev"},
            {"text": "Fix the bug", "confidence": 0.8, "priority": 2, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        todos = [
            {"text": "fix the bug", "confidence": 0.9, "priority": 1, "category": "dev"},
            {"text": "Fix The Bug", "confidence": 0.8, "priority": 2, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 1

    def test_punctuation_stripped_for_dedup(self):
        todos = [
            {"text": "Review PR!", "confidence": 0.9, "priority": 1, "category": "dev"},
            {"text": "Review PR", "confidence": 0.8, "priority": 2, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 1

    def test_different_todos_kept(self):
        todos = [
            {"text": "Buy groceries", "confidence": 0.9, "priority": 2, "category": "personal"},
            {"text": "Write tests", "confidence": 0.8, "priority": 1, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 2

    def test_normalize_strips_leading_trailing_punctuation(self):
        assert normalize_todo_text("  Fix bug!  ") == normalize_todo_text("fix bug")

    def test_normalize_case_fold(self):
        assert normalize_todo_text("UPPER") == normalize_todo_text("upper")

    def test_normalize_internal_punctuation_stripped(self):
        normalized = normalize_todo_text("do it, now!")
        assert "," not in normalized
        assert "!" not in normalized


# ---------------------------------------------------------------------------
# AC-c  confidence threshold filtering
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    """AC-c: only todos with confidence >= 0.6 are kept."""

    def test_high_confidence_kept(self):
        todos = [{"text": "Task A", "confidence": 0.9, "priority": 1, "category": "dev"}]
        assert len(filter_todos(todos)) == 1

    def test_exactly_0_6_kept(self):
        todos = [{"text": "Task B", "confidence": 0.6, "priority": 1, "category": "dev"}]
        assert len(filter_todos(todos)) == 1

    def test_below_threshold_excluded(self):
        todos = [{"text": "Task C", "confidence": 0.59, "priority": 1, "category": "dev"}]
        assert len(filter_todos(todos)) == 0

    def test_zero_confidence_excluded(self):
        todos = [{"text": "Task D", "confidence": 0.0, "priority": 1, "category": "dev"}]
        assert len(filter_todos(todos)) == 0

    def test_missing_confidence_excluded(self):
        todos = [{"text": "Task E", "priority": 1, "category": "dev"}]
        assert len(filter_todos(todos)) == 0

    def test_mixed_confidence_filtered_correctly(self):
        todos = [
            {"text": "Keep A", "confidence": 0.8, "priority": 2, "category": "personal"},
            {"text": "Drop B", "confidence": 0.4, "priority": 1, "category": "personal"},
            {"text": "Keep C", "confidence": 0.6, "priority": 3, "category": "personal"},
        ]
        result = filter_todos(todos)
        texts = [t["text"] for t in result]
        assert "Keep A" in texts
        assert "Keep C" in texts
        assert "Drop B" not in texts


# ---------------------------------------------------------------------------
# AC-c (sort)  priority sort descending
# ---------------------------------------------------------------------------

class TestTodoPrioritySort:
    """Part of AC-c: todos are sorted by priority field descending."""

    def test_sorted_descending(self):
        todos = [
            {"text": "Low", "confidence": 0.9, "priority": 1, "category": "dev"},
            {"text": "High", "confidence": 0.9, "priority": 10, "category": "dev"},
            {"text": "Mid", "confidence": 0.9, "priority": 5, "category": "dev"},
        ]
        result = filter_todos(todos)
        priorities = [t["priority"] for t in result]
        assert priorities == sorted(priorities, reverse=True)

    def test_items_with_equal_priority_both_present(self):
        todos = [
            {"text": "Alpha", "confidence": 0.9, "priority": 5, "category": "dev"},
            {"text": "Beta", "confidence": 0.9, "priority": 5, "category": "personal"},
        ]
        result = filter_todos(todos)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# AC-d  dev-category approval-routing annotation
# ---------------------------------------------------------------------------

class TestDevCategoryAnnotation:
    """AC-d: items with category='dev' get <!-- route: approval --> annotation."""

    def test_dev_category_annotated(self):
        todos = [{"text": "Deploy feature", "confidence": 0.9, "priority": 5, "category": "dev"}]
        result = filter_todos(todos)
        assert result[0].get("_approval_route") is True

    def test_non_dev_not_annotated(self):
        todos = [{"text": "Go for a run", "confidence": 0.9, "priority": 5, "category": "health"}]
        result = filter_todos(todos)
        assert not result[0].get("_approval_route")

    def test_annotation_in_rendered_output(self):
        todos = [
            {"text": "Ship it", "confidence": 0.9, "priority": 5, "category": "dev"},
            {"text": "Meditate", "confidence": 0.9, "priority": 3, "category": "wellness"},
        ]
        data = _journal_data(todos=todos)
        section = render_todo_section(data, "")
        assert "<!-- route: approval -->" in section

    def test_non_dev_no_approval_comment_in_output(self):
        todos = [{"text": "Read a book", "confidence": 0.9, "priority": 5, "category": "learning"}]
        data = _journal_data(todos=todos)
        section = render_todo_section(data, "")
        assert "<!-- route: approval -->" not in section

    def test_mixed_categories_only_dev_annotated(self):
        todos = [
            {"text": "Code review", "confidence": 0.9, "priority": 8, "category": "dev"},
            {"text": "Buy coffee", "confidence": 0.9, "priority": 7, "category": "personal"},
        ]
        data = _journal_data(todos=todos)
        section = render_todo_section(data, "")
        lines = section.splitlines()
        dev_line = next((l for l in lines if "Code review" in l), "")
        personal_line = next((l for l in lines if "Buy coffee" in l), "")
        assert "<!-- route: approval -->" in dev_line or any(
            "<!-- route: approval -->" in lines[i]
            for i, l in enumerate(lines) if "Code review" in l
        )
        assert "<!-- route: approval -->" not in personal_line


# ---------------------------------------------------------------------------
# AC-e  individual degradation paths
# ---------------------------------------------------------------------------

class TestDegradationPaths:
    """AC-e: each contract degrades independently — file missing or stale."""

    def test_journal_file_missing_shows_unavailable(self, tmp_path):
        missing = tmp_path / "no_file.json"
        data, reason = load_contract(missing)
        section = render_journal_section(data, reason)
        assert "⚠️ unavailable" in section
        assert reason  # reason string is non-empty

    def test_perfcoach_file_missing_shows_unavailable(self, tmp_path):
        missing = tmp_path / "no_perf.json"
        data, reason = load_contract(missing)
        section = render_training_section(data, reason)
        assert "⚠️ unavailable" in section

    def test_commander_file_missing_shows_unavailable(self, tmp_path):
        missing = tmp_path / "no_cmd.json"
        data, reason = load_contract(missing)
        section = render_dev_report_section(data, reason)
        assert "⚠️ unavailable" in section

    def test_journal_stale_shows_unavailable_with_date(self, tmp_path):
        path = tmp_path / "journal.json"
        stale_date = _yesterday()
        path.write_text(
            json.dumps({"for_date": stale_date, "reflection": {"markdown": "old"}, "todos": []}),
            encoding="utf-8",
        )
        data, reason = load_contract(path)
        section = render_journal_section(data, reason)
        assert "⚠️ unavailable" in section
        assert stale_date in section or "stale" in section.lower()

    def test_perfcoach_stale_shows_unavailable_with_date(self, tmp_path):
        path = tmp_path / "perf.json"
        stale_date = _yesterday()
        path.write_text(
            json.dumps({"for_date": stale_date, "advisories": ["old"]}),
            encoding="utf-8",
        )
        data, reason = load_contract(path)
        section = render_training_section(data, reason)
        assert "⚠️ unavailable" in section
        assert stale_date in section or "stale" in section.lower()

    def test_commander_stale_shows_unavailable_with_date(self, tmp_path):
        path = tmp_path / "cmd.json"
        stale_date = _yesterday()
        path.write_text(
            json.dumps({
                "for_date": stale_date,
                "completed": [], "needs_review": [], "dead_letter": [], "cost": "$0",
            }),
            encoding="utf-8",
        )
        data, reason = load_contract(path)
        section = render_dev_report_section(data, reason)
        assert "⚠️ unavailable" in section
        assert stale_date in section or "stale" in section.lower()

    def test_missing_file_exit_code_still_zero(self, tmp_path, monkeypatch):
        """Script never exits non-zero due to missing contract files alone."""
        import subprocess
        env_overrides = {
            "JOURNAL_BRIEF_PATH": str(tmp_path / "nope1.json"),
            "PERFCOACH_BRIEF_PATH": str(tmp_path / "nope2.json"),
            "COMMANDER_REPORT_PATH": str(tmp_path / "nope3.json"),
        }
        import os
        env = {**os.environ, **env_overrides}
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"exit non-zero on missing contracts: {result.stderr}"


# ---------------------------------------------------------------------------
# AC-f  all-contracts-unavailable → valid brief printed
# ---------------------------------------------------------------------------

class TestAllContractsUnavailable:
    """AC-f: when all contracts are unavailable, compose_brief still returns a valid markdown brief."""

    def test_all_unavailable_brief_has_four_sections(self):
        brief = compose_brief(None, "file not found", None, "file not found", None, "file not found")
        assert "Journal Reflection" in brief or "Section 1" in brief or "## " in brief
        assert brief.count("⚠️ unavailable") == 4

    def test_all_unavailable_brief_is_string(self):
        brief = compose_brief(None, "missing", None, "missing", None, "missing")
        assert isinstance(brief, str)
        assert len(brief) > 0

    def test_all_unavailable_brief_exit_zero(self, tmp_path):
        import subprocess
        import os
        env = {
            **os.environ,
            "JOURNAL_BRIEF_PATH": str(tmp_path / "x1.json"),
            "PERFCOACH_BRIEF_PATH": str(tmp_path / "x2.json"),
            "COMMANDER_REPORT_PATH": str(tmp_path / "x3.json"),
        }
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "⚠️ unavailable" in result.stdout


# ---------------------------------------------------------------------------
# Section rendering — happy-path smoke tests (not exhaustive; anchored to AC3-7)
# ---------------------------------------------------------------------------

class TestSectionRendering:
    """Happy-path rendering for all four sections."""

    def test_journal_reflection_rendered_verbatim(self):
        data = _journal_data(reflection="## My morning thoughts\n\nFeel great today.")
        section = render_journal_section(data, "")
        assert "## My morning thoughts" in section
        assert "Feel great today." in section

    def test_training_uses_advisories_primary(self):
        data = _perfcoach_data(advisories=["Easy 30-min run.", "Focus on form."])
        section = render_training_section(data, "")
        assert "Easy 30-min run." in section
        assert "Focus on form." in section

    def test_training_falls_back_when_advisories_absent(self):
        data = _perfcoach_data(today="Rest day.", tomorrow="Long run.")
        section = render_training_section(data, "")
        assert "Rest day." in section
        assert "Long run." in section

    def test_training_falls_back_when_advisories_empty(self):
        data = _perfcoach_data(advisories=[], today="Easy jog.", form="Neutral posture.")
        section = render_training_section(data, "")
        assert "Easy jog." in section or "Neutral posture." in section

    def test_dev_report_renders_completed(self):
        data = _commander_data(completed=["feature-X shipped", "bug-Y fixed"])
        section = render_dev_report_section(data, "")
        assert "feature-X shipped" in section
        assert "bug-Y fixed" in section

    def test_dev_report_renders_needs_review(self):
        data = _commander_data(needs_review=["PR #42"])
        section = render_dev_report_section(data, "")
        assert "PR #42" in section

    def test_dev_report_renders_dead_letter(self):
        data = _commander_data(dead_letter=["stalled-task-Z"])
        section = render_dev_report_section(data, "")
        assert "stalled-task-Z" in section

    def test_dev_report_renders_cost(self):
        data = _commander_data(cost="$1.23")
        section = render_dev_report_section(data, "")
        assert "Cost: $1.23" in section

    def test_dry_run_prints_to_stdout(self, tmp_path):
        import subprocess
        import os
        journal = tmp_path / "j.json"
        journal.write_text(json.dumps(_journal_data(reflection="Hello.")), encoding="utf-8")
        perf = tmp_path / "p.json"
        perf.write_text(json.dumps(_perfcoach_data(advisories=["Run easy."])), encoding="utf-8")
        cmd = tmp_path / "c.json"
        cmd.write_text(json.dumps(_commander_data(completed=["task done"])), encoding="utf-8")

        env = {
            **os.environ,
            "JOURNAL_BRIEF_PATH": str(journal),
            "PERFCOACH_BRIEF_PATH": str(perf),
            "COMMANDER_REPORT_PATH": str(cmd),
        }
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Hello." in result.stdout
        assert "Run easy." in result.stdout
        assert "task done" in result.stdout
