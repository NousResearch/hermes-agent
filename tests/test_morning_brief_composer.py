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
