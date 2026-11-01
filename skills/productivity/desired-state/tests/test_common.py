"""Tests for _common.py — frontmatter parse/serialize, GoalDoc, helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

from _common import (
    GoalDoc,
    _as_number,
    add_milestone,
    dump_frontmatter,
    iso_now,
    milestones_in,
    parse_date,
    parse_frontmatter,
    set_milestone,
    slugify,
)


# =============================================================================
# slugify
# =============================================================================

class TestSlugify:
    def test_basic(self):
        assert slugify("Savings Rate 2026!") == "savings-rate-2026"

    def test_collapses_and_trims(self):
        assert slugify("  --Hello__World--  ") == "hello-world"

    def test_empty_falls_back(self):
        assert slugify("!!!") == "untitled"


# =============================================================================
# Frontmatter parsing
# =============================================================================

class TestParseFrontmatter:
    def test_scalars_and_coercion(self):
        text = '---\ndomain: finance\ntarget_value: 15\ncurrent_value: 12.4\nunit: "%"\n---\nbody here\n'
        data, body = parse_frontmatter(text)
        assert data["domain"] == "finance"
        assert data["target_value"] == 15 and isinstance(data["target_value"], int)
        assert data["current_value"] == 12.4
        assert data["unit"] == "%"
        assert body.strip() == "body here"

    def test_block_list(self):
        text = "---\ntags:\n  - money\n  - 2026\nlinked_people:\n  - ryan\n---\n"
        data, _ = parse_frontmatter(text)
        assert data["tags"] == ["money", "2026"]
        assert data["linked_people"] == ["ryan"]

    def test_inline_list(self):
        data, _ = parse_frontmatter("---\ntags: [a, b, c]\n---\n")
        assert data["tags"] == ["a", "b", "c"]

    def test_empty_list_key(self):
        data, _ = parse_frontmatter("---\nlinked_todos:\n---\n")
        assert data["linked_todos"] == []

    def test_null_and_bool(self):
        data, _ = parse_frontmatter("---\nunit: ~\ndirection:\nflag: true\n---\n")
        assert data["unit"] is None
        assert data["direction"] is None
        assert data["flag"] is True

    def test_schema_string_scalars_are_not_coerced(self):
        data, _ = parse_frontmatter("---\ndomain: 2026\ngoal: true\nunit: false\n---\n")
        assert data["domain"] == "2026"
        assert data["goal"] == "true"
        assert data["unit"] == "false"

    def test_missing_frontmatter_raises(self):
        import pytest

        with pytest.raises(ValueError):
            parse_frontmatter("no frontmatter here")

    def test_unterminated_raises(self):
        import pytest

        with pytest.raises(ValueError):
            parse_frontmatter("---\ndomain: x\nstill going")


# =============================================================================
# Round-trip stability
# =============================================================================

class TestRoundTrip:
    def test_dump_then_parse_is_stable(self):
        data = {
            "domain": "finance",
            "goal": "Hit 15% savings rate",
            "unit": "%",
            "tags": ["money", "2026"],
            "linked_todos": [],
            "target_value": 15,
            "current_value": 12.4,
        }
        text = "---\n" + dump_frontmatter(data) + "\n---\n\nbody\n"
        parsed, _ = parse_frontmatter(text)
        for key, value in data.items():
            assert parsed[key] == value

    def test_quoted_value_with_embedded_quotes_round_trips(self):
        # codex P3: a scalar that starts ambiguously AND contains " must survive
        # dump/parse without accumulating backslash escapes.
        doc = GoalDoc(domain="learning", goal='"Rust" mastery')
        again = GoalDoc.from_text(doc.to_text())
        assert again.goal == '"Rust" mastery'
        assert "\\" not in again.goal

    def test_string_scalars_that_look_typed_round_trip_as_strings(self):
        doc = GoalDoc(domain="2026", goal="true", unit="null")
        again = GoalDoc.from_text(doc.to_text())
        assert again.domain == "2026"
        assert again.goal == "true"
        assert again.unit == "null"

    def test_inline_list_respects_quoted_commas(self):
        data, _ = parse_frontmatter('---\ntags: ["alpha, beta", gamma]\n---\n')
        assert data["tags"] == ["alpha, beta", "gamma"]

    def test_goaldoc_to_text_from_text(self):
        doc = GoalDoc(domain="health", goal="Cut resting HR", unit="bpm",
                      direction="decrease", target_value=60, current_value=70,
                      tags=["fitness"], body="## Milestones\n- [ ] Sleep 8h\n")
        again = GoalDoc.from_text(doc.to_text())
        assert again.domain == "health"
        assert again.target_value == 60 and again.current_value == 70
        assert again.direction == "decrease"
        assert again.tags == ["fitness"]
        assert "Sleep 8h" in again.body


# =============================================================================
# GoalDoc validation + typing helpers
# =============================================================================

class TestGoalDocValidation:
    def test_clean_goal_valid(self):
        doc = GoalDoc(domain="finance", goal="Save more")
        assert doc.validate() == []

    def test_missing_domain(self):
        doc = GoalDoc(domain="", goal="Save more")
        assert any("domain" in p for p in doc.validate())

    def test_bad_enum_values(self):
        doc = GoalDoc(domain="d", goal="g", horizon="huge", status="ongoing", direction="sideways")
        problems = " ".join(doc.validate())
        assert "horizon" in problems and "status" in problems and "direction" in problems

    def test_bad_target_date(self):
        doc = GoalDoc(domain="d", goal="g", target_date="not-a-date")
        assert any("target_date" in p for p in doc.validate())

    def test_bad_start_date(self):
        # codex P2 (round 4): start_date validated symmetrically with target_date.
        doc = GoalDoc(domain="d", goal="g", start_date="not-a-date")
        assert any("start_date" in p for p in doc.validate())

    def test_is_quantifiable(self):
        assert GoalDoc(domain="d", goal="g", target_value=10, current_value=1).is_quantifiable()
        assert not GoalDoc(domain="d", goal="g", target_value=10).is_quantifiable()


class TestEffectiveDirection:
    def test_explicit_wins(self):
        assert GoalDoc(domain="d", goal="g", direction="maintain").effective_direction() == "maintain"

    def test_inferred_increase(self):
        assert GoalDoc(domain="d", goal="g", current_value=9, target_value=15).effective_direction() == "increase"

    def test_inferred_from_baseline_not_current(self):
        # codex P2: baseline 72, target 60 -> decrease, even though current (58)
        # is already below target. Inferring from current alone would flip it.
        doc = GoalDoc(domain="d", goal="g", baseline_value=72, current_value=58, target_value=60)
        assert doc.effective_direction() == "decrease"


class TestHelpers:
    def test_as_number(self):
        assert _as_number(5) == 5.0
        assert _as_number("12.4") == 12.4
        assert _as_number("nope") is None
        assert _as_number(True) is None  # bools are not numbers here

    def test_parse_date(self):
        assert parse_date("2026-12-31") == date(2026, 12, 31)
        assert parse_date("2026-12-31T09:00:00Z") == date(2026, 12, 31)
        assert parse_date("") is None
        assert parse_date("garbage") is None

    def test_iso_now_pinned(self):
        fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        assert iso_now(fixed) == "2026-01-02T03:04:05Z"


# =============================================================================
# HERMES_HOME resolution (codex P2: Windows default must be %LOCALAPPDATA%)
# =============================================================================

class TestMilestoneEditing:
    def test_list(self):
        assert milestones_in("- [x] a\n- [ ] b\n* [X] c\n") == [(True, "a"), (False, "b"), (True, "c")]

    def test_add_creates_section(self):
        body = add_milestone("## Why\ncontext\n", "Do the thing")
        assert "## Milestones" in body and "- [ ] Do the thing" in body

    def test_add_appends_under_existing(self):
        out = add_milestone("## Milestones\n- [ ] one\n", "two")
        assert milestones_in(out) == [(False, "one"), (False, "two")]

    def test_add_empty_raises(self):
        import pytest

        with pytest.raises(ValueError):
            add_milestone("body", "   ")

    def test_check_and_uncheck(self):
        checked = set_milestone("- [ ] a\n- [ ] b\n", 2, True)
        assert milestones_in(checked) == [(False, "a"), (True, "b")]
        back = set_milestone(checked, 2, False)
        assert milestones_in(back) == [(False, "a"), (False, "b")]

    def test_out_of_range_raises(self):
        import pytest

        with pytest.raises(IndexError):
            set_milestone("- [ ] a\n", 5, True)


class TestHermesHome:
    def test_env_var_wins(self, monkeypatch, tmp_path):
        import _common

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert _common.get_hermes_home() == tmp_path

    def test_windows_uses_localappdata(self, monkeypatch, tmp_path):
        import _common

        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(_common.sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
        assert _common.get_hermes_home() == tmp_path / "AppData" / "Local" / "hermes"

    def test_posix_uses_dot_hermes(self, monkeypatch):
        import _common
        from pathlib import Path

        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(_common.sys, "platform", "linux")
        assert _common.get_hermes_home() == Path.home() / ".hermes"
