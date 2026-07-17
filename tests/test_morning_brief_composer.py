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
SCRIPT = REPO_ROOT / "plugins" / "life_ops" / "scripts" / "morning_brief_composer.py"

sys.path.insert(0, str(REPO_ROOT))
from plugins.life_ops.scripts.morning_brief_composer import (
    compose_brief,
    filter_todos,
    get_today_bangkok,
    get_todo_text,
    load_contract,
    normalize_todo_text,
    render_advisory,
    render_dev_report_section,
    render_journal_section,
    render_session_value,
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
# get_todo_text — "content" is canonical, "text" is a fallback
# ---------------------------------------------------------------------------

class TestTodoContentField:
    """The journal contract's canonical todo field is "content"; "text" is
    kept only as a fallback for older/alternate producers."""

    def test_content_key_used_when_present(self):
        assert get_todo_text({"content": "Ship the release"}) == "Ship the release"

    def test_falls_back_to_text_when_content_absent(self):
        assert get_todo_text({"text": "Legacy shaped todo"}) == "Legacy shaped todo"

    def test_content_preferred_over_text_when_both_present(self):
        item = {"content": "New shape", "text": "Old shape"}
        assert get_todo_text(item) == "New shape"

    def test_empty_content_falls_back_to_text(self):
        # "" is falsy, so get_todo_text() should fall back to "text" per the
        # `item.get("content") or item.get("text", "")` implementation.
        item = {"content": "", "text": "Fallback text"}
        assert get_todo_text(item) == "Fallback text"

    def test_missing_both_returns_empty_string(self):
        assert get_todo_text({}) == ""

    def test_filter_todos_dedups_content_key_items(self):
        todos = [
            {"content": "Fix the bug", "confidence": 0.9, "priority": 1, "category": "dev"},
            {"content": "Fix the bug", "confidence": 0.8, "priority": 2, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 1

    def test_filter_todos_dedups_across_content_and_text_keys(self):
        """A "content"-shaped item and a "text"-shaped item with the same
        normalized text are still recognized as duplicates."""
        todos = [
            {"content": "Fix the bug", "confidence": 0.9, "priority": 2, "category": "dev"},
            {"text": "Fix the bug", "confidence": 0.8, "priority": 1, "category": "dev"},
        ]
        result = filter_todos(todos)
        assert len(result) == 1

    # NOTE: render_todo_section() no longer sources from the raw journal
    # contract's todos[] at all — it reads plugins.life_ops.todo_store's
    # persistent, stable-key store instead (see TestRenderTodoSectionFromStore
    # below). The two tests formerly here fed contract-shaped todos straight
    # into render_todo_section() and asserted the text/annotation showed up
    # in the output; that assumption is now false regardless of the "content"
    # vs "text" field question this class is actually about, so they were
    # removed rather than patched — get_todo_text()'s own behavior (the
    # thing this class is meant to cover) is still fully exercised by the
    # tests above.


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

    # test_annotation_in_rendered_output and test_mixed_categories_only_dev_
    # annotated formerly asserted render_todo_section() emits
    # "<!-- route: approval -->" for category="dev" contract items.
    # render_todo_section() now renders exclusively from
    # plugins.life_ops.todo_store.get_open_todos() (see
    # TestRenderTodoSectionFromStore below), whose rows carry no "category"
    # field at all — that annotation path is dead in the new renderer, so
    # those two tests were removed. test_non_dev_no_approval_comment_in_output
    # (kept below) still documents that the string never appears, which
    # remains true (trivially, now) and doubles as a forbidden-field guard.
    def test_non_dev_no_approval_comment_in_output(self):
        todos = [{"text": "Read a book", "confidence": 0.9, "priority": 5, "category": "learning"}]
        data = _journal_data(todos=todos)
        section = render_todo_section(data, "")
        assert "<!-- route: approval -->" not in section


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
        # Only 3 of the 4 sections show "⚠️ unavailable" — the todo section
        # (Section 2) no longer depends on the journal contract at all; an
        # unavailable/missing contract just means "today" falls back to the
        # system date for recency math, while the row content itself comes
        # from todo_store.get_open_todos() (empty here -> "(no open todos)").
        assert brief.count("⚠️ unavailable") == 3
        assert "(no open todos)" in brief

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


# ---------------------------------------------------------------------------
# render_advisory — dict advisories (warn / info) and legacy string advisories
# ---------------------------------------------------------------------------

class TestAdvisoryRendering:
    """Advisories are {key, severity: "info"|"warn", text} dicts; plain
    strings are still accepted for legacy producers."""

    def test_warn_severity_gets_warning_prefix(self):
        advisory = {"key": "overtraining", "severity": "warn", "text": "Back off intensity today."}
        assert render_advisory(advisory) == "- ⚠️ Back off intensity today."

    def test_info_severity_has_no_prefix(self):
        advisory = {"key": "hydration", "severity": "info", "text": "Drink more water."}
        assert render_advisory(advisory) == "- Drink more water."

    def test_missing_severity_has_no_prefix(self):
        advisory = {"key": "misc", "text": "Just a note."}
        assert render_advisory(advisory) == "- Just a note."

    def test_legacy_string_advisory_rendered_verbatim(self):
        assert render_advisory("Easy 30-min run.") == "- Easy 30-min run."

    def test_training_section_renders_dict_advisories_warn(self):
        data = _perfcoach_data(advisories=[{"key": "load", "severity": "warn", "text": "High load this week."}])
        section = render_training_section(data, "")
        assert "- ⚠️ High load this week." in section

    def test_training_section_renders_dict_advisories_info(self):
        data = _perfcoach_data(advisories=[{"key": "note", "severity": "info", "text": "Sleep was solid."}])
        section = render_training_section(data, "")
        assert "- Sleep was solid." in section
        assert "⚠️" not in section

    def test_training_section_renders_mixed_warn_and_info_advisories(self):
        data = _perfcoach_data(
            advisories=[
                {"key": "load", "severity": "warn", "text": "Reduce volume."},
                {"key": "note", "severity": "info", "text": "HRV trending up."},
            ]
        )
        section = render_training_section(data, "")
        assert "- ⚠️ Reduce volume." in section
        assert "- HRV trending up." in section


# ---------------------------------------------------------------------------
# render_session_value — today/tomorrow/recent_wrap dicts, including planned=False
# ---------------------------------------------------------------------------

class TestSessionRendering:
    """Sessions are dicts with session_type/intensity/duration_min/notes and
    a "planned" flag; planned=False renders as a rest/nothing-planned line."""

    def test_planned_false_renders_rest_message(self):
        value = {"planned": False}
        assert render_session_value(value) == "Rest / nothing planned"

    def test_planned_false_ignores_other_fields(self):
        """Even if other fields are present, planned=False short-circuits to
        the rest message — a rest day shouldn't show stale session details."""
        value = {"planned": False, "session_type": "easy run", "notes": "stale leftover"}
        assert render_session_value(value) == "Rest / nothing planned"

    def test_full_session_renders_all_present_fields(self):
        value = {
            "planned": True,
            "session_type": "long run",
            "intensity": "moderate",
            "duration_min": 60,
            "notes": "negative split",
        }
        rendered = render_session_value(value)
        assert "session type: long run" in rendered
        assert "intensity: moderate" in rendered
        assert "duration min: 60" in rendered
        assert "notes: negative split" in rendered

    def test_partial_session_skips_missing_fields(self):
        value = {"planned": True, "session_type": "easy run"}
        rendered = render_session_value(value)
        assert rendered == "session type: easy run"

    def test_empty_dict_renders_em_dash(self):
        assert render_session_value({}) == "—"

    def test_legacy_string_value_rendered_verbatim(self):
        assert render_session_value("Long run.") == "Long run."

    def test_training_section_fallback_renders_planned_false_session(self):
        data = _perfcoach_data(today={"planned": False}, tomorrow={"planned": True, "session_type": "intervals"})
        section = render_training_section(data, "")
        assert "**today:** Rest / nothing planned" in section
        assert "**tomorrow:** session type: intervals" in section

    def test_training_section_fallback_dict_session_shape_contract(self):
        """End-to-end: perfcoach contract shaped with dict sessions (the
        canonical shape) renders readable lines, not raw dict reprs."""
        data = _perfcoach_data(
            today={"planned": True, "session_type": "tempo run", "intensity": "hard", "duration_min": 45},
            recent_wrap={"planned": False},
        )
        section = render_training_section(data, "")
        assert "{" not in section  # no raw dict repr leaking into output
        assert "session type: tempo run" in section
        assert "**recent_wrap:** Rest / nothing planned" in section


# ---------------------------------------------------------------------------
# render_todo_section — sources from plugins.life_ops.todo_store.get_open_todos(),
# NOT the raw journal contract. HERMES_HOME is isolated per test by
# tests/conftest.py's autouse fixture, so every test gets a fresh todos.db.
# ---------------------------------------------------------------------------

def _seed_todo(key, text="fake task text", priority="medium", for_date="2026-07-14",
                source_dates=None, recurring=False):
    from plugins.life_ops import todo_store as ts
    ts.upsert_from_contract(
        [{
            "key": key,
            "text": text,
            "priority": priority,
            "recurring": recurring,
            "source_dates": source_dates if source_dates is not None else [for_date],
        }],
        for_date,
    )


class TestRenderTodoSectionFromStore:
    """render_todo_section() renders todo_store's open rows, independent of
    the journal contract's own todos[] array (see the module's own docstring
    on render_todo_section for the "store is authoritative" rationale)."""

    def test_ignores_raw_contract_todos_entirely(self):
        data = _journal_data(
            todos=[{"content": "Should never appear", "confidence": 0.99, "priority": 10, "category": "dev"}]
        )
        section = render_todo_section(data, "")
        assert "Should never appear" not in section
        assert "(no open todos)" in section

    def test_only_open_rows_render_snoozed_done_dismissed_excluded(self):
        from plugins.life_ops import todo_store as ts
        ts.upsert_from_contract(
            [
                {"key": "open-1", "text": "Open task alpha", "priority": "high", "source_dates": ["2026-07-14"]},
                {"key": "snooze-1", "text": "Snoozed task", "priority": "medium", "source_dates": ["2026-07-14"]},
                {"key": "done-1", "text": "Done task", "priority": "low", "source_dates": ["2026-07-14"]},
                {"key": "dismiss-1", "text": "Dismissed task", "priority": "low", "source_dates": ["2026-07-14"]},
            ],
            "2026-07-14",
        )
        ts.close_todo("snooze-1", "snooze", "test", snooze_until="2026-08-01")
        ts.close_todo("done-1", "done", "test")
        ts.close_todo("dismiss-1", "dismiss", "test")

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")

        assert "Open task alpha" in section
        assert "Snoozed task" not in section
        assert "Done task" not in section
        assert "Dismissed task" not in section
        assert "To-do · 1 open" in section

    def test_header_shows_correct_open_count(self):
        for i in range(3):
            _seed_todo(f"k{i}", text=f"task {i}")
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "To-do · 3 open" in section

    def test_output_wrapped_in_fenced_code_block(self):
        _seed_todo("k1", text="task")
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert section.count("```") == 2
        assert section.index("```") < section.rindex("```")

    def test_empty_store_still_wrapped_in_fenced_code_block(self):
        section = render_todo_section(_journal_data(), "")
        assert section.count("```") == 2
        assert "(no open todos)" in section

    def test_high_priority_gets_bang_glyph(self):
        _seed_todo("hi", text="high prio item", priority="high")
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        line = next(l for l in section.splitlines() if "high prio item" in l)
        assert line.strip().startswith("!")

    def test_non_high_priority_gets_dot_glyph(self):
        _seed_todo("med", text="medium prio item", priority="medium")
        _seed_todo("lo", text="low prio item", priority="low")
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        med_line = next(l for l in section.splitlines() if "medium prio item" in l)
        lo_line = next(l for l in section.splitlines() if "low prio item" in l)
        assert med_line.strip().startswith("·")
        assert lo_line.strip().startswith("·")

    def test_row_contains_the_stable_key(self):
        _seed_todo("stable-key-42", text="task text")
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "stable-key-42" in section

    def test_text_truncated_to_configured_max_chars(self, monkeypatch, tmp_path):
        long_text = "x" * 100
        _seed_todo("k1", text=long_text)

        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("todo_section:\n  text_max_chars: 10\n", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert long_text not in section
        assert ("x" * 9 + "…") in section


class TestRecencyFormatting:
    """↻ {N}d for recurring todos (days since first_seen), else the latest
    source_dates entry as MM-DD."""

    def test_recurring_shows_days_since_first_seen(self):
        _seed_todo("rk", text="recurring task", for_date="2026-07-01", recurring=True)
        section = render_todo_section(_journal_data(for_date="2026-07-15"), "")
        assert "↻ 14d" in section

    def test_recurring_zero_days_on_the_first_day(self):
        _seed_todo("rk", text="brand new recurring task", for_date="2026-07-15", recurring=True)
        section = render_todo_section(_journal_data(for_date="2026-07-15"), "")
        assert "↻ 0d" in section

    def test_non_recurring_shows_max_source_date_as_mm_dd(self):
        _seed_todo("nk", text="non recurring task", for_date="2026-07-14",
                    source_dates=["2026-07-10", "2026-07-12"])
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "07-12" in section

    def test_non_recurring_no_source_dates_shows_question_mark(self):
        # Insert directly via upsert with an empty source_dates list.
        from plugins.life_ops import todo_store as ts
        ts.upsert_from_contract(
            [{"key": "nk", "text": "no dates task", "priority": "medium", "source_dates": []}],
            "2026-07-14",
        )
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        line = next(l for l in section.splitlines() if "no dates task" in l)
        assert line.rstrip().endswith("?")


class TestBriefRenderConfigFieldToggles:
    """config/brief_render.yaml's todo_section.fields toggles which columns
    render; text_max_chars/header_format are also configurable. Missing or
    unreadable config falls back to code defaults without crashing."""

    def test_key_false_omits_key_from_row(self, monkeypatch, tmp_path):
        _seed_todo("should-not-appear-42", text="task")

        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("todo_section:\n  fields:\n    key: false\n", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "should-not-appear-42" not in section

    def test_recency_false_omits_recency(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="task", source_dates=["2026-01-01"])

        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("todo_section:\n  fields:\n    recency: false\n", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "01-01" not in section

    def test_glyph_false_omits_glyph(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="high prio task", priority="high")

        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("todo_section:\n  fields:\n    glyph: false\n", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        line = next(l for l in section.splitlines() if "high prio task" in l)
        assert not line.strip().startswith("!")

    def test_text_false_omits_text(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="unique-omitted-text-marker")

        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("todo_section:\n  fields:\n    text: false\n", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "unique-omitted-text-marker" not in section

    def test_missing_config_file_falls_back_to_defaults_no_crash(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="task", priority="high")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(tmp_path / "does-not-exist.yaml"))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "k1" in section
        assert "!" in section  # glyph field defaults to on

    def test_unparsable_config_file_falls_back_to_defaults_no_crash(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="task", priority="high")
        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text("not: valid: yaml: [[[", encoding="utf-8")
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "k1" in section

    def test_custom_header_format_applied(self, monkeypatch, tmp_path):
        _seed_todo("k1", text="task")
        cfg_path = tmp_path / "brief_render.yaml"
        cfg_path.write_text(
            "todo_section:\n  header_format: 'Custom header ({count})'\n", encoding="utf-8"
        )
        monkeypatch.setenv("BRIEF_RENDER_CONFIG", str(cfg_path))

        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "Custom header (1)" in section


class TestAwayMarker:
    """A one-line away-mode marker appears near the top of the brief when
    away_mode.is_away() is true; absent entirely otherwise."""

    def test_marker_present_and_mentions_until_date_when_away(self):
        from plugins.life_ops import away_mode
        away_mode.set_away(until="2026-07-20")

        brief = compose_brief(_journal_data(for_date="2026-07-15"), "", None, "x", None, "x")

        assert "🌙" in brief
        assert "2026-07-20" in brief

    def test_marker_absent_when_not_away(self):
        brief = compose_brief(_journal_data(for_date="2026-07-15"), "", None, "x", None, "x")
        assert "🌙" not in brief

    def test_marker_absent_after_away_mode_cleared(self):
        from plugins.life_ops import away_mode
        away_mode.set_away(until="2026-07-20")
        away_mode.clear_away()

        brief = compose_brief(_journal_data(for_date="2026-07-15"), "", None, "x", None, "x")
        assert "🌙" not in brief

    def test_marker_appears_before_section_1(self):
        from plugins.life_ops import away_mode
        away_mode.set_away(until="2026-07-20")

        brief = compose_brief(_journal_data(for_date="2026-07-15"), "", None, "x", None, "x")

        assert brief.index("🌙") < brief.index("Section 1")


class TestNoForbiddenFieldsInTodoSection:
    """category/status/id/confidence/origin are intentionally never rendered
    in the todo section, regardless of what the store row carries."""

    def test_forbidden_fields_never_appear(self):
        from plugins.life_ops import todo_store as ts
        ts.upsert_from_contract(
            [{
                "key": "abc123",
                "text": "a task about something unrelated",
                "priority": "high",
                "category": "dev",
                "confidence": 0.95,
                "source_dates": ["2026-07-14"],
            }],
            "2026-07-14",
        )
        section = render_todo_section(_journal_data(for_date="2026-07-14"), "")
        assert "category" not in section
        assert "confidence" not in section
        assert "origin" not in section
        assert "0.95" not in section
        # "status" as a rendered field/value, not as an incidental substring
        # of some other word.
        assert "status" not in section.lower()


# ---------------------------------------------------------------------------
# v3 perf-coach contract fixtures (issue #59)
# ---------------------------------------------------------------------------

def _v3_form(**overrides):
    base = {
        "ctl": 45.2,
        "atl": 43.1,
        "tsb": 2.1,
        "acwr": 1.05,
        "acwr_state": "optimal",
        "interpretation": "Load balance looks good",
    }
    base.update(overrides)
    return base


def _v3_weight(**overrides):
    base = {
        "current_kg": 70.5,
        "trend_7d": "+0.2kg",
        "target_kg": 68.0,
        "target_date": "2026-10-01",
        "on_track": True,
    }
    base.update(overrides)
    return base


def _v3_week_plan():
    return [
        {"day": "Mon", "planned": True, "session_type": "easy run", "duration_min": 45},
        {"day": "Tue", "planned": False},
        {"day": "Wed", "planned": True, "session_type": "intervals", "duration_min": 60},
    ]


def _v3_advisories():
    return [
        {"key": "load", "severity": "warn", "text": "High load this week."},
        {"key": "note", "severity": "info", "text": "Recovery recommended."},
    ]


def _full_v3_data():
    return _perfcoach_data(
        form=_v3_form(),
        weight=_v3_weight(),
        week_plan=_v3_week_plan(),
        advisories=_v3_advisories(),
    )


# ---------------------------------------------------------------------------
# AC-1  Form line
# ---------------------------------------------------------------------------

class TestV3FormLine:
    """AC-1: When data.form is a dict, renders CTL/ATL/TSB line with optional
    ACWR segment; line omitted when form is missing or not a dict."""

    def test_full_form_line_rendered(self):
        data = _perfcoach_data(advisories=["adv"], form=_v3_form())
        section = render_training_section(data, "")
        assert (
            "**Form:** CTL 45.2 · ATL 43.1 · TSB 2.1 · ACWR 1.05 (optimal) — Load balance looks good"
            in section
        )

    def test_acwr_segment_present_when_acwr_in_form(self):
        data = _perfcoach_data(advisories=["adv"], form=_v3_form())
        section = render_training_section(data, "")
        assert "ACWR 1.05 (optimal)" in section

    def test_acwr_segment_omitted_when_acwr_absent(self):
        form = {k: v for k, v in _v3_form().items() if k not in ("acwr", "acwr_state")}
        data = _perfcoach_data(advisories=["adv"], form=form)
        section = render_training_section(data, "")
        assert "**Form:** CTL 45.2 · ATL 43.1 · TSB 2.1 — Load balance looks good" in section
        assert "ACWR" not in section

    def test_form_line_omitted_when_form_key_missing(self):
        data = _perfcoach_data(advisories=["adv"])
        section = render_training_section(data, "")
        assert "**Form:**" not in section

    def test_form_line_omitted_when_form_is_not_dict(self):
        data = _perfcoach_data(advisories=["adv"], form="Easy run.")
        section = render_training_section(data, "")
        assert "**Form:** CTL" not in section


# ---------------------------------------------------------------------------
# AC-2  Weight line
# ---------------------------------------------------------------------------

class TestV3WeightLine:
    """AC-2: When data.weight is present, renders weight tracking line;
    omitted when data.weight is absent."""

    def test_weight_line_on_track_rendered(self):
        data = _perfcoach_data(advisories=["adv"], weight=_v3_weight(on_track=True))
        section = render_training_section(data, "")
        assert (
            "**Weight:** 70.5kg · 7d +0.2kg · target 68.0 by 2026-10-01 (on track)"
            in section
        )

    def test_weight_line_off_pace_rendered(self):
        data = _perfcoach_data(advisories=["adv"], weight=_v3_weight(on_track=False))
        section = render_training_section(data, "")
        assert "(off pace)" in section
        assert "(on track)" not in section

    def test_weight_line_omitted_when_weight_absent(self):
        data = _perfcoach_data(advisories=["adv"])
        section = render_training_section(data, "")
        assert "**Weight:**" not in section


# ---------------------------------------------------------------------------
# AC-3  Week plan block
# ---------------------------------------------------------------------------

class TestV3WeekPlanBlock:
    """AC-3: When data.week_plan is a non-empty list, renders heading + one
    row per day; block omitted when week_plan is missing or empty."""

    def test_week_plan_heading_rendered(self):
        data = _perfcoach_data(advisories=["adv"], week_plan=_v3_week_plan())
        section = render_training_section(data, "")
        assert "**Week plan:**" in section

    def test_planned_true_day_rendered_with_session(self):
        data = _perfcoach_data(advisories=["adv"], week_plan=_v3_week_plan())
        section = render_training_section(data, "")
        assert "· Mon  easy run 45min" in section
        assert "· Wed  intervals 60min" in section

    def test_planned_false_day_rendered_as_rest(self):
        data = _perfcoach_data(advisories=["adv"], week_plan=_v3_week_plan())
        section = render_training_section(data, "")
        assert "· Tue  rest" in section

    def test_week_plan_omitted_when_key_missing(self):
        data = _perfcoach_data(advisories=["adv"])
        section = render_training_section(data, "")
        assert "**Week plan:**" not in section

    def test_week_plan_omitted_when_empty_list(self):
        data = _perfcoach_data(advisories=["adv"], week_plan=[])
        section = render_training_section(data, "")
        assert "**Week plan:**" not in section


# ---------------------------------------------------------------------------
# AC-4  Advisories heading
# ---------------------------------------------------------------------------

class TestV3AdvisoriesHeading:
    """AC-4: **Advisories:** heading present iff at least one new block
    (Form, Weight, Week plan) precedes the advisories; absent otherwise."""

    def test_heading_present_when_form_precedes_advisories(self):
        data = _perfcoach_data(advisories=_v3_advisories(), form=_v3_form())
        section = render_training_section(data, "")
        assert "**Advisories:**" in section

    def test_heading_present_when_weight_precedes_advisories(self):
        data = _perfcoach_data(advisories=_v3_advisories(), weight=_v3_weight())
        section = render_training_section(data, "")
        assert "**Advisories:**" in section

    def test_heading_present_when_week_plan_precedes_advisories(self):
        data = _perfcoach_data(advisories=_v3_advisories(), week_plan=_v3_week_plan())
        section = render_training_section(data, "")
        assert "**Advisories:**" in section

    def test_heading_absent_when_no_new_blocks_present(self):
        """v2 contract: advisories only, no form/weight/week_plan."""
        data = _perfcoach_data(advisories=_v3_advisories())
        section = render_training_section(data, "")
        assert "**Advisories:**" not in section
        assert "High load this week." in section

    def test_advisories_still_rendered_without_heading_for_v2(self):
        data = _perfcoach_data(advisories=[{"key": "k", "severity": "info", "text": "Nice one."}])
        section = render_training_section(data, "")
        assert "- Nice one." in section
        assert "**Advisories:**" not in section


# ---------------------------------------------------------------------------
# AC-5  Legacy fallback preserved
# ---------------------------------------------------------------------------

class TestV3LegacyFallbackPreserved:
    """AC-5: today/tomorrow/form/recent_wrap rendered via render_session_value
    when advisories is absent; path is untouched."""

    def test_today_field_still_renders_when_no_advisories(self):
        data = _perfcoach_data(today={"planned": True, "session_type": "tempo run", "duration_min": 40})
        section = render_training_section(data, "")
        assert "**today:** session type: tempo run" in section

    def test_tomorrow_field_still_renders_when_no_advisories(self):
        data = _perfcoach_data(tomorrow="Long run tomorrow.")
        section = render_training_section(data, "")
        assert "**tomorrow:** Long run tomorrow." in section

    def test_recent_wrap_field_still_renders_when_no_advisories(self):
        data = _perfcoach_data(recent_wrap={"planned": False})
        section = render_training_section(data, "")
        assert "**recent_wrap:** Rest / nothing planned" in section

    def test_no_data_at_all_shows_no_training_message(self):
        data = _perfcoach_data()
        section = render_training_section(data, "")
        assert "(no training data today)" in section


# ---------------------------------------------------------------------------
# AC-6  v2 snapshot (byte-identical)
# ---------------------------------------------------------------------------

class TestV2Snapshot:
    """AC-6: A v2 contract (advisories only, no form/weight/week_plan keys)
    produces byte-identical output to the pre-change implementation."""

    _V2_ADVISORIES = [
        {"key": "run", "severity": "info", "text": "Easy 30-min run."},
        {"key": "load", "severity": "warn", "text": "High load this week."},
    ]

    def _expected_v2_output(self):
        return "\n".join([
            "## Section 3 — Training\n",
            "- Easy 30-min run.",
            "- ⚠️ High load this week.",
        ])

    def test_v2_contract_output_byte_identical(self):
        data = _perfcoach_data(advisories=self._V2_ADVISORIES)
        actual = render_training_section(data, "")
        assert actual == self._expected_v2_output()

    def test_v2_contract_has_no_advisories_heading(self):
        data = _perfcoach_data(advisories=self._V2_ADVISORIES)
        section = render_training_section(data, "")
        assert "**Advisories:**" not in section

    def test_v2_contract_has_no_form_weight_weekplan_lines(self):
        data = _perfcoach_data(advisories=self._V2_ADVISORIES)
        section = render_training_section(data, "")
        assert "**Form:**" not in section
        assert "**Weight:**" not in section
        assert "**Week plan:**" not in section


# ---------------------------------------------------------------------------
# AC-7  v3 integration — full v3 renders Form → Weight → Week plan → Advisories
# ---------------------------------------------------------------------------

class TestV3Integration:
    """AC-7: A full v3 fixture with all four fields renders Form → Weight →
    Week plan → Advisories in that order."""

    def test_all_four_sections_present(self):
        section = render_training_section(_full_v3_data(), "")
        assert "**Form:**" in section
        assert "**Weight:**" in section
        assert "**Week plan:**" in section
        assert "**Advisories:**" in section

    def test_order_form_weight_weekplan_advisories(self):
        section = render_training_section(_full_v3_data(), "")
        form_pos = section.index("**Form:**")
        weight_pos = section.index("**Weight:**")
        weekplan_pos = section.index("**Week plan:**")
        advisories_pos = section.index("**Advisories:**")
        assert form_pos < weight_pos < weekplan_pos < advisories_pos

    def test_advisory_content_present(self):
        section = render_training_section(_full_v3_data(), "")
        assert "- ⚠️ High load this week." in section
        assert "- Recovery recommended." in section

    def test_week_plan_rows_present(self):
        section = render_training_section(_full_v3_data(), "")
        assert "· Mon  easy run 45min" in section
        assert "· Tue  rest" in section
        assert "· Wed  intervals 60min" in section


# ---------------------------------------------------------------------------
# AC-8  Per-block omission (parameterised)
# ---------------------------------------------------------------------------

class TestV3PerBlockOmission:
    """AC-8: Each of Form, Weight, and Week plan is independently omitted when
    its top-level key is missing; the remaining blocks are unaffected."""

    def _data_without(self, *omit_keys):
        d = _full_v3_data()
        for key in omit_keys:
            d.pop(key, None)
        return d

    def test_form_omitted_weight_and_week_plan_present(self):
        data = self._data_without("form")
        section = render_training_section(data, "")
        assert "**Form:**" not in section
        assert "**Weight:**" in section
        assert "**Week plan:**" in section
        assert "**Advisories:**" in section

    def test_weight_omitted_form_and_week_plan_present(self):
        data = self._data_without("weight")
        section = render_training_section(data, "")
        assert "**Weight:**" not in section
        assert "**Form:**" in section
        assert "**Week plan:**" in section
        assert "**Advisories:**" in section

    def test_week_plan_omitted_form_and_weight_present(self):
        data = self._data_without("week_plan")
        section = render_training_section(data, "")
        assert "**Week plan:**" not in section
        assert "**Form:**" in section
        assert "**Weight:**" in section
        assert "**Advisories:**" in section

    def test_form_omitted_advisories_heading_still_present(self):
        """Weight + week_plan still trigger the Advisories heading even without form."""
        data = self._data_without("form")
        section = render_training_section(data, "")
        assert "**Advisories:**" in section

    def test_all_three_omitted_no_heading_v2_behaviour(self):
        """When form/weight/week_plan all absent → no Advisories heading."""
        data = self._data_without("form", "weight", "week_plan")
        section = render_training_section(data, "")
        assert "**Advisories:**" not in section
        assert "- ⚠️ High load this week." in section


# ---------------------------------------------------------------------------
# Issue #60: Structured commander contract — render_dev_report_section
# ---------------------------------------------------------------------------

# Helpers for structured project contract fixtures

def _project(name="project-alpha", status="in_progress", **kwargs):
    """Build a minimal project dict."""
    p = {"name": name, "status": status}
    p.update(kwargs)
    return p


def _in_progress_sub(sprint_label="S5", percent=60, ticket="PROJ-12"):
    return {"sprint_label": sprint_label, "percent": percent, "ticket": ticket}


def _shipped_item(label="v1.2", goal="Ship auth", done=3, pr_number=42):
    return {"label": label, "goal": goal, "done": done, "pr_number": pr_number}


def _fixed_item(issue_number=7, title="Fix login crash"):
    return {"issue_number": issue_number, "title": title}


def _stale_blocked(issue_number=9, age_days=5, type="review", title="Auth PR"):
    return {"kind": "blocked", "issue_number": issue_number, "age_days": age_days,
            "type": type, "title": title}


def _stale_waiting_signoff(label="v1.1", age_days=3):
    return {"kind": "waiting_signoff", "label": label, "age_days": age_days}


def _stale_backlog(label="backlog-A", age_days=10, ticket_count=5):
    return {"kind": "backlog", "label": label, "age_days": age_days, "ticket_count": ticket_count}


def _waiting_item(label="v1.3", ticket_count=4, estimated_hours=8):
    return {"label": label, "ticket_count": ticket_count, "estimated_hours": estimated_hours}


def _commander_projects_data(projects, cost="$0.12", for_date=None):
    return {
        "for_date": for_date or _today(),
        "projects": projects,
        "cost": cost,
    }


# ---------------------------------------------------------------------------
# AC-1: projects present → header format with glyph per status
# ---------------------------------------------------------------------------

class TestProjectsHeaderLine:
    """AC-1: each project renders **{name}** — {glyph} {status} with correct glyph."""

    _GLYPH_MAP = {
        "shipped": "🚀",
        "in_progress": "⏳",
        "blocked": "⛔",
        "waiting_signoff": "📋",
        "idle": "💤",
    }

    def test_shipped_glyph(self):
        data = _commander_projects_data([_project("proj", "shipped")])
        section = render_dev_report_section(data, "")
        assert "**proj** — 🚀 shipped" in section

    def test_in_progress_glyph(self):
        data = _commander_projects_data([_project("proj", "in_progress")])
        section = render_dev_report_section(data, "")
        assert "**proj** — ⏳ in_progress" in section

    def test_blocked_glyph(self):
        data = _commander_projects_data([_project("proj", "blocked")])
        section = render_dev_report_section(data, "")
        assert "**proj** — ⛔ blocked" in section

    def test_waiting_signoff_glyph(self):
        data = _commander_projects_data([_project("proj", "waiting_signoff")])
        section = render_dev_report_section(data, "")
        assert "**proj** — 📋 waiting_signoff" in section

    def test_idle_glyph(self):
        data = _commander_projects_data([_project("proj", "idle")])
        section = render_dev_report_section(data, "")
        assert "**proj** — 💤 idle" in section

    def test_multiple_projects_each_get_header(self):
        data = _commander_projects_data([
            _project("alpha", "in_progress"),
            _project("beta", "idle"),
        ])
        section = render_dev_report_section(data, "")
        assert "**alpha** — ⏳ in_progress" in section
        assert "**beta** — 💤 idle" in section


# ---------------------------------------------------------------------------
# AC-2: in_progress header suffix
# ---------------------------------------------------------------------------

class TestInProgressHeaderSuffix:
    """AC-2: in_progress header appends (sprint_label, percent% — ticket) when
    in_progress sub-key present; omitted when sub-key absent."""

    def test_in_progress_suffix_rendered(self):
        p = _project("proj", "in_progress", in_progress=_in_progress_sub("S5", 60, "PROJ-12"))
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "(S5, 60% — PROJ-12)" in section

    def test_in_progress_suffix_absent_when_sub_key_missing(self):
        p = _project("proj", "in_progress")
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "**proj** — ⏳ in_progress" in section
        assert "%" not in section


# ---------------------------------------------------------------------------
# AC-3: compact counts suffix when any bucket non-empty
# ---------------------------------------------------------------------------

class TestCountsSuffix:
    """AC-3: compact counts suffix on header line when any bucket non-empty."""

    def test_counts_suffix_present_when_shipped_non_empty(self):
        p = _project("proj", "in_progress", shipped=[_shipped_item()])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        # Header line contains counts bracket
        header_line = next(l for l in section.splitlines() if "**proj**" in l)
        assert "[" in header_line and "]" in header_line

    def test_counts_suffix_absent_for_idle_project(self):
        p = _project("proj", "idle")
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        header_line = next(l for l in section.splitlines() if "**proj**" in l)
        assert "[" not in header_line

    def test_counts_suffix_reflects_shipped_count(self):
        p = _project("proj", "shipped", shipped=[_shipped_item(), _shipped_item("v1.3", "Other", 1, 43)])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        header_line = next(l for l in section.splitlines() if "**proj**" in l)
        assert "2" in header_line


# ---------------------------------------------------------------------------
# AC-4: shipped bucket sub-bullets
# ---------------------------------------------------------------------------

class TestShippedBucket:
    """AC-4: non-empty shipped renders - Shipped: {label} "{goal}" ({done} done, PR #{pr})."""

    def test_shipped_sub_bullet_format(self):
        p = _project("proj", "shipped", shipped=[_shipped_item("v1.2", "Ship auth", 3, 42)])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert '- Shipped: v1.2 "Ship auth" (3 done, PR #42)' in section

    def test_shipped_string_item_rendered_verbatim(self):
        p = _project("proj", "shipped", shipped=["verbatim shipped text"])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- verbatim shipped text" in section

    def test_shipped_missing_pr_number_no_keyerror(self):
        item = {"label": "v1.0", "goal": "Launch", "done": 1}
        p = _project("proj", "shipped", shipped=[item])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "v1.0" in section
        assert "Launch" in section


# ---------------------------------------------------------------------------
# AC-5: fixed bucket sub-bullets
# ---------------------------------------------------------------------------

class TestFixedBucket:
    """AC-5: non-empty fixed renders - Fixed: #{issue_number} {title}."""

    def test_fixed_sub_bullet_format(self):
        p = _project("proj", "shipped", fixed=[_fixed_item(7, "Fix login crash")])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- Fixed: #7 Fix login crash" in section

    def test_fixed_string_item_rendered_verbatim(self):
        p = _project("proj", "shipped", fixed=["verbatim fixed text"])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- verbatim fixed text" in section


# ---------------------------------------------------------------------------
# AC-6: stale bucket — all three kinds
# ---------------------------------------------------------------------------

class TestStaleBucket:
    """AC-6: stale renders blocked/waiting_signoff/backlog sub-bullets."""

    def test_stale_blocked_format(self):
        item = _stale_blocked(9, 5, "review", "Auth PR")
        p = _project("proj", "blocked", stale=[item])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "#9 blocked 5d (review) — Auth PR" in section

    def test_stale_waiting_signoff_format(self):
        item = _stale_waiting_signoff("v1.1", 3)
        p = _project("proj", "waiting_signoff", stale=[item])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "v1.1 awaiting sign-off 3d" in section

    def test_stale_backlog_format(self):
        item = _stale_backlog("backlog-A", 10, 5)
        p = _project("proj", "in_progress", stale=[item])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "backlog-A backlog untouched 10d (5 tickets)" in section

    def test_stale_string_item_rendered_verbatim(self):
        p = _project("proj", "blocked", stale=["stale string item"])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- stale string item" in section

    def test_stale_mixed_kinds_all_render(self):
        stale = [
            _stale_blocked(1, 2, "ci", "CI job"),
            _stale_waiting_signoff("v0.9", 4),
            _stale_backlog("tech-debt", 7, 3),
        ]
        p = _project("proj", "blocked", stale=stale)
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "#1 blocked 2d (ci) — CI job" in section
        assert "v0.9 awaiting sign-off 4d" in section
        assert "tech-debt backlog untouched 7d (3 tickets)" in section


# ---------------------------------------------------------------------------
# AC-7: waiting bucket sub-bullets
# ---------------------------------------------------------------------------

class TestWaitingBucket:
    """AC-7: non-empty waiting renders - Waiting: {label} sign-off ({count} tickets, ~{h}h)."""

    def test_waiting_sub_bullet_format(self):
        p = _project("proj", "waiting_signoff", waiting=[_waiting_item("v1.3", 4, 8)])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- Waiting: v1.3 sign-off (4 tickets, ~8h)" in section

    def test_waiting_string_item_rendered_verbatim(self):
        p = _project("proj", "waiting_signoff", waiting=["verbatim waiting text"])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "- verbatim waiting text" in section

    def test_waiting_missing_estimated_hours_no_keyerror(self):
        item = {"label": "v2.0", "ticket_count": 2}
        p = _project("proj", "waiting_signoff", waiting=[item])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "v2.0" in section
        assert "2 tickets" in section


# ---------------------------------------------------------------------------
# AC-8: idle projects collapse to header line only
# ---------------------------------------------------------------------------

class TestIdleProjectCollapse:
    """AC-8: idle projects (all buckets empty) render only the header line."""

    def test_idle_project_no_sub_bullets(self):
        p = _project("proj", "idle")
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        lines = [l for l in section.splitlines() if l.strip().startswith("-")]
        assert not lines

    def test_idle_project_with_explicitly_empty_buckets_no_sub_bullets(self):
        p = _project("proj", "idle", shipped=[], fixed=[], stale=[], waiting=[])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        lines = [l for l in section.splitlines() if l.strip().startswith("-")]
        assert not lines


# ---------------------------------------------------------------------------
# AC-9: dict/string tolerance (render_advisory pattern)
# ---------------------------------------------------------------------------

class TestBucketItemTolerance:
    """AC-9: every bucket item accepts dict or plain string."""

    def test_shipped_string_and_dict_both_accepted(self):
        p = _project("p", "shipped",
                     shipped=["string item", _shipped_item("v1.0", "Goal", 1, 10)])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "string item" in section
        assert "v1.0" in section

    def test_stale_string_and_dict_both_accepted(self):
        p = _project("p", "blocked",
                     stale=["plain stale", _stale_blocked(3, 1, "review", "Bug")])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "plain stale" in section
        assert "#3 blocked 1d (review) — Bug" in section


# ---------------------------------------------------------------------------
# AC-10: missing sub-keys default cleanly
# ---------------------------------------------------------------------------

class TestMissingSubKeysSafe:
    """AC-10: missing optional sub-keys (in_progress, pr_number, estimated_hours)
    produce no KeyError."""

    def test_in_progress_sub_key_absent_no_crash(self):
        p = _project("proj", "in_progress")
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "**proj**" in section

    def test_shipped_missing_pr_number_no_crash(self):
        p = _project("proj", "shipped", shipped=[{"label": "v1", "goal": "G", "done": 1}])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "v1" in section

    def test_waiting_missing_estimated_hours_no_crash(self):
        p = _project("proj", "waiting_signoff", waiting=[{"label": "v2", "ticket_count": 3}])
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "v2" in section

    def test_all_project_buckets_absent_no_crash(self):
        p = {"name": "minimal", "status": "idle"}
        data = _commander_projects_data([p])
        section = render_dev_report_section(data, "")
        assert "**minimal**" in section


# ---------------------------------------------------------------------------
# AC-11: Cost line rendered after all project blocks
# ---------------------------------------------------------------------------

class TestCostLinePosition:
    """AC-11: Cost: {cost} line is unchanged and appears after all project blocks."""

    def test_cost_line_present(self):
        data = _commander_projects_data([_project("p", "idle")], cost="$1.23")
        section = render_dev_report_section(data, "")
        assert "Cost: $1.23" in section

    def test_cost_line_after_project_blocks(self):
        p = _project("proj", "shipped", shipped=[_shipped_item()])
        data = _commander_projects_data([p], cost="$0.50")
        section = render_dev_report_section(data, "")
        proj_pos = section.index("**proj**")
        cost_pos = section.index("Cost: $0.50")
        assert proj_pos < cost_pos


# ---------------------------------------------------------------------------
# AC-12: projects absent or empty → flat fallthrough unchanged
# ---------------------------------------------------------------------------

class TestFlatFallthrough:
    """AC-12: when data.projects is absent or [], falls through to flat path."""

    def test_no_projects_key_uses_flat_path(self):
        data = _commander_data(completed=["task-X"], cost="$0.10")
        section = render_dev_report_section(data, "")
        assert "task-X" in section
        assert "**Completed:**" in section

    def test_empty_projects_uses_flat_path(self):
        data = {
            "for_date": _today(),
            "projects": [],
            "completed": ["task-Y"],
            "needs_review": [],
            "dead_letter": [],
            "cost": "$0.05",
        }
        section = render_dev_report_section(data, "")
        assert "task-Y" in section
        assert "**Completed:**" in section

    def test_projects_non_empty_ignores_flat_keys(self):
        """AC-13: both projects and legacy flat keys → projects wins."""
        data = {
            "for_date": _today(),
            "projects": [_project("structured-proj", "idle")],
            "completed": ["should-be-ignored"],
            "needs_review": ["also-ignored"],
            "dead_letter": [],
            "cost": "$0.01",
        }
        section = render_dev_report_section(data, "")
        assert "**structured-proj**" in section
        assert "should-be-ignored" not in section
        assert "also-ignored" not in section


# ---------------------------------------------------------------------------
# AC-14: snapshot test — legacy flat output byte-identical before/after
# ---------------------------------------------------------------------------

class TestLegacyFlatSnapshot:
    """AC-14: legacy flat contract produces the same output byte-for-byte."""

    def test_flat_contract_snapshot(self):
        data = _commander_data(
            completed=["feature-X shipped", "bug-Y fixed"],
            needs_review=["PR #42"],
            dead_letter=["stalled-task-Z"],
            cost="$1.23",
        )
        section = render_dev_report_section(data, "")
        expected = "\n".join([
            "## Section 4 — Overnight Dev Report\n",
            "**Completed:**",
            "- feature-X shipped",
            "- bug-Y fixed",
            "\n**Needs Review:**",
            "- PR #42",
            "\n**Dead Letter:**",
            "- stalled-task-Z",
            "\nCost: $1.23",
        ])
        assert section == expected
