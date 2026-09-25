"""Blueprint-to-skill relationships, independent of the generic catalog/renderer tests."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from croniter import croniter

from cron.blueprint_catalog import CATALOG, fill_blueprint, get_blueprint

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("key,skill", [
    ("morning-brief", "google-workspace"), ("important-mail", "email-inbox-triage"),
    ("weekly-review", "weekly-review-planning"), ("price-watch", "product-price-monitor"),
])
def test_task_blueprint_loads_procedure(key, skill):
    blueprint = get_blueprint(key)
    assert blueprint is not None, key
    assert skill in blueprint.skills
    if key == "weekly-review":
        assert skill in fill_blueprint(blueprint, {})["prompt"]


def test_every_blueprint_skill_is_bundled():
    bundled = {path.parent.name for path in (REPO / "skills").rglob("SKILL.md")}
    for blueprint in CATALOG:
        assert set(blueprint.skills) <= bundled, blueprint.key


@pytest.mark.parametrize("hours", next(
    slot.options for bp in CATALOG if bp.key == "price-watch"
    for slot in bp.slots if slot.name == "interval_h"
))
def test_price_watch_fills_and_persists_with_chosen_cadence(hours, tmp_path, monkeypatch):
    from cron import jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "jobs.json")
    # Re-pointing two of the three store constants selects the "live constants" store, whose
    # OUTPUT_DIR would still be the import-time (real-home) path.
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "output")
    blueprint = get_blueprint("price-watch")
    assert blueprint is not None
    spec = fill_blueprint(blueprint, {
        "item": "https://example.test/widget", "condition": "below 42 credits",
        "interval_h": hours, "deliver": "local",
    })
    created = jobs.create_job(**spec)
    saved = jobs.get_job(created["id"])
    assert saved is not None
    assert "product-price-monitor" in saved["skills"]
    assert saved["deliver"] == "local"
    for text in ("https://example.test/widget", "below 42 credits", "[SILENT]"):
        assert text in saved["prompt"], text
    ticks = croniter(saved["schedule"]["expr"], datetime(2026, 1, 1, tzinfo=timezone.utc))
    # Span a day boundary: wrong-field steps can otherwise look valid once.
    times = [ticks.get_next(datetime) for _ in range(26)]
    assert {b - a for a, b in zip(times, times[1:])} == {timedelta(hours=int(hours))}
