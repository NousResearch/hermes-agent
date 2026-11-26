"""Tests for ds_store.py — CRUD, errors, timestamps, HERMES_HOME."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from _common import GoalDoc, desired_root
from ds_store import (
    GoalExistsError,
    GoalNotFoundError,
    archive_goal,
    create_goal,
    get_goal,
    list_goals,
    set_current,
    update_goal,
)

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)


def _make(store, **kw):
    base = dict(domain="finance", goal="Hit 15% savings rate", target_value=15, current_value=9)
    base.update(kw)
    return create_goal(GoalDoc(**base), root=store, now=NOW)


class TestCreateRead:
    def test_create_stamps_and_persists(self, store):
        saved = _make(store)
        assert saved.path.exists()
        assert saved.created_at == "2026-07-07T12:00:00Z"
        assert saved.updated_at == "2026-07-07T12:00:00Z"
        back = get_goal("finance", "hit-15-savings-rate", root=store)
        assert back.target_value == 15 and back.current_value == 9

    def test_create_invalid_raises_before_write(self, store):
        with pytest.raises(ValueError):
            create_goal(GoalDoc(domain="", goal="bad"), root=store, now=NOW)
        assert not (store).exists() or not any(store.rglob("*.md"))

    def test_create_existing_raises(self, store):
        _make(store)
        with pytest.raises(GoalExistsError):
            _make(store)

    def test_overwrite_allowed(self, store):
        _make(store)
        create_goal(
            GoalDoc(domain="finance", goal="Hit 15% savings rate", target_value=15, current_value=13),
            root=store, overwrite=True, now=NOW,
        )
        assert get_goal("finance", "hit-15-savings-rate", root=store).current_value == 13

    def test_get_missing_raises(self, store):
        with pytest.raises(GoalNotFoundError):
            get_goal("finance", "nope", root=store)


class TestListing:
    def test_filter_and_sort(self, store):
        _make(store)
        create_goal(GoalDoc(domain="health", goal="Cut HR", target_value=60, current_value=70),
                    root=store, now=NOW)
        assert [d.domain for d in list_goals(root=store)] == ["finance", "health"]
        assert [d.domain for d in list_goals(root=store, domain="health")] == ["health"]

    def test_empty_store(self, store):
        assert list_goals(root=store) == []

    def test_malformed_file_skipped(self, store):
        _make(store)
        bad = store / "finance" / "broken.md"
        bad.write_text("not valid frontmatter", encoding="utf-8")
        # listing still returns the good one, does not raise
        slugs = [d.slug for d in list_goals(root=store)]
        assert "hit-15-savings-rate" in slugs and "broken" not in slugs

    def test_valid_frontmatter_missing_required_field_skipped(self, store):
        # Regression (codex P2): valid frontmatter but no `goal` raised
        # TypeError and aborted list/report/gap instead of being skipped.
        _make(store)
        halfbaked = store / "finance" / "halfbaked.md"
        halfbaked.write_text("---\ndomain: finance\nstatus: active\n---\n\nno goal field\n", encoding="utf-8")
        slugs = [d.slug for d in list_goals(root=store)]
        assert "hit-15-savings-rate" in slugs and "halfbaked" not in slugs

    def test_bare_numeric_domain_does_not_break_listing_sort(self, store):
        # Hand-edited YAML-like scalars in schema string fields must stay strings;
        # otherwise sorting mixed domains can compare int vs str and crash.
        (store / "2026").mkdir(parents=True)
        (store / "finance").mkdir(parents=True)
        (store / "2026" / "annual-plan.md").write_text("---\ndomain: 2026\ngoal: Annual plan\n---\n\n", encoding="utf-8")
        (store / "finance" / "save.md").write_text("---\ndomain: finance\ngoal: Save\n---\n\n", encoding="utf-8")
        assert [d.domain for d in list_goals(root=store)] == ["2026", "finance"]

    def test_get_missing_required_field_raises_value_error(self, store):
        _make(store)
        bad = store / "finance" / "halfbaked.md"
        bad.write_text("---\ndomain: finance\n---\n\nbody\n", encoding="utf-8")
        with pytest.raises(ValueError):
            get_goal("finance", "halfbaked", root=store)


class TestUpdate:
    def test_set_current(self, store):
        _make(store)
        later = datetime(2026, 7, 8, tzinfo=timezone.utc)
        doc = set_current("finance", "hit-15-savings-rate", 12.4, root=store, now=later)
        assert doc.current_value == 12.4
        assert doc.updated_at == "2026-07-08T00:00:00Z"
        assert doc.created_at == "2026-07-07T12:00:00Z"  # unchanged

    def test_unknown_field_raises(self, store):
        _make(store)
        with pytest.raises(KeyError):
            update_goal("finance", "hit-15-savings-rate", {"nonsense": 1}, root=store, now=NOW)

    def test_edit_body(self, store):
        _make(store)
        doc = update_goal("finance", "hit-15-savings-rate", {"body": "## new\n- [ ] x\n"}, root=store, now=NOW)
        assert "new" in doc.body


class TestArchive:
    def test_archive_sets_status_keeps_file(self, store):
        saved = _make(store)
        doc = archive_goal("finance", "hit-15-savings-rate", status="achieved", root=store, now=NOW)
        assert doc.status == "achieved"
        assert saved.path.exists()  # never deleted

    def test_bad_status_rejected(self, store):
        _make(store)
        with pytest.raises(ValueError):
            archive_goal("finance", "hit-15-savings-rate", status="deleted", root=store, now=NOW)


class TestDirectionPersistence:
    def test_inferred_direction_persisted_on_create(self, store):
        # codex P2: lock in inferred direction at define time.
        doc = create_goal(
            GoalDoc(domain="health", goal="Cut HR", current_value=70, target_value=60),
            root=store, now=NOW,
        )
        assert doc.direction == "decrease"
        assert get_goal("health", "cut-hr", root=store).direction == "decrease"

    def test_direction_stable_after_crossing_target(self, store):
        from gap import compute_gap

        create_goal(GoalDoc(domain="health", goal="Cut HR", current_value=70, target_value=60),
                    root=store, now=NOW)
        set_current("health", "cut-hr", 58, root=store, now=NOW)  # cross below target
        res = compute_gap(get_goal("health", "cut-hr", root=store), now=NOW)
        assert res.direction == "decrease"  # did NOT flip to increase
        assert res.pace == "met"

    def test_target_only_does_not_lock_wrong_direction(self, store):
        # codex P2 (round 4): a target with no reference must NOT persist a
        # guessed "increase"; leave it unset so a later value infers correctly.
        doc = create_goal(GoalDoc(domain="health", goal="Resting HR under 60", target_value=60),
                          root=store, now=NOW)
        assert doc.direction is None

    def test_target_only_then_track_infers_decrease(self, store):
        from gap import compute_gap

        create_goal(GoalDoc(domain="health", goal="Resting HR under 60", target_value=60),
                    root=store, now=NOW)
        set_current("health", "resting-hr-under-60", 72, root=store, now=NOW)
        res = compute_gap(get_goal("health", "resting-hr-under-60", root=store), now=NOW)
        assert res.direction == "decrease"   # not a false "increase"/met
        assert res.pace != "met"

    def test_target_only_direction_locks_on_first_track(self, store):
        # codex P2 (round 6): the first tracked value freezes direction, so a
        # target-only decrease goal stays stable after crossing the target.
        from gap import compute_gap

        create_goal(GoalDoc(domain="health", goal="Resting HR under 60", target_value=60),
                    root=store, now=NOW)
        set_current("health", "resting-hr-under-60", 72, root=store, now=NOW)  # first ref locks it
        assert get_goal("health", "resting-hr-under-60", root=store).direction == "decrease"
        set_current("health", "resting-hr-under-60", 58, root=store, now=NOW)  # cross target
        res = compute_gap(get_goal("health", "resting-hr-under-60", root=store), now=NOW)
        assert res.direction == "decrease"  # stable, did not flip
        assert res.pace == "met"


class TestHermesHome:
    def test_desired_root_honors_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert desired_root() == tmp_path / "state" / "desired"
