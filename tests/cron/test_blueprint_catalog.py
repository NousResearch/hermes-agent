"""Tests for Automation Blueprints — the parameterized automation template system.

Covers the core catalog/slot schema/renderers/fill (cron/blueprint_catalog.py),
the shared /blueprint command handler (hermes_cli/blueprint_cmd.py), and
the docs generator. Uses an isolated HERMES_HOME for anything that touches the
cron job store.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cron.blueprint_catalog import (
    CATALOG,
    BlueprintFillError,
    BlueprintSlot,
    fill_blueprint,
    get_blueprint,
    blueprint_catalog_entry,
    blueprint_deeplink,
    blueprint_form_schema,
    blueprint_slash_command,
)


class TestCatalog:
    def test_catalog_nonempty_and_keyed(self):
        assert len(CATALOG) >= 1
        for r in CATALOG:
            assert get_blueprint(r.key) is r

    def test_every_slot_has_known_type(self):
        for r in CATALOG:
            for s in r.slots:
                assert s.type in {"time", "enum", "text", "weekdays"}

    def test_bad_slot_type_rejected(self):
        with pytest.raises(ValueError):
            BlueprintSlot(name="x", type="bogus", label="X")


class TestScheduleResolution:
    def test_time_to_cron(self):
        spec = fill_blueprint(get_blueprint("morning-brief"), {"time": "08:30"})
        assert spec["schedule"] == "30 8 * * *"

    def test_interval_schedule(self):
        spec = fill_blueprint(
            get_blueprint("important-mail"),
            {"interval_min": "15", "criteria": "x", "deliver": "origin"},
        )
        assert spec["schedule"] == "*/15 * * * *"

    def test_day_to_dow(self):
        spec = fill_blueprint(
            get_blueprint("weekly-review"),
            {"time": "18:00", "day": "sunday", "deliver": "origin"},
        )
        assert spec["schedule"] == "0 18 * * 0"

    def test_weekday_preset_to_dow(self):
        spec = fill_blueprint(
            get_blueprint("custom-reminder"),
            {"what": "stretch", "time": "14:00", "recurrence": "weekdays", "deliver": "origin"},
        )
        assert spec["schedule"] == "0 14 * * 1-5"

    def test_defaults_fill_when_omitted(self):
        spec = fill_blueprint(get_blueprint("morning-brief"), {})
        assert spec["schedule"] == "0 8 * * *"


class TestValidation:
    def test_invalid_time_rejected(self):
        with pytest.raises(BlueprintFillError, match="invalid time"):
            fill_blueprint(get_blueprint("morning-brief"), {"time": "25:99"})

    def test_bad_enum_rejected_and_names_slot(self):
        with pytest.raises(BlueprintFillError, match="not allowed"):
            fill_blueprint(get_blueprint("news-digest"), {"count": "42"})

    def test_deliver_slot_accepts_any_platform(self):
        # deliver is a non-strict enum: its options are suggestions, the real
        # set of valid platforms depends on the user's configured gateways and
        # is validated downstream by the cron scheduler.
        spec = fill_blueprint(get_blueprint("morning-brief"), {"time": "08:00", "deliver": "slack"})
        assert spec["deliver"] == "slack"

    def test_unknown_slot_name_rejected(self):
        # A typo'd slot must NOT silently create a job with the default value.
        with pytest.raises(BlueprintFillError, match="unknown slot"):
            fill_blueprint(get_blueprint("morning-brief"), {"tiem": "07:15"})

    def test_hydration_hourly_step_actually_fires_at_chosen_cadence(self):
        # Regression: a minute-field step (*/90) silently wraps to hourly.
        # The hour-field step form must produce the cadence the user picked.
        croniter = pytest.importorskip("croniter").croniter
        from datetime import datetime

        spec = fill_blueprint(get_blueprint("hydration-move"), {"interval_hours": "2"})
        it = croniter(spec["schedule"], datetime(2026, 6, 10, 8, 0))
        first_three = [it.get_next(datetime) for _ in range(3)]
        gaps = {
            (b - a).total_seconds()
            for a, b in zip(first_three, first_three[1:])
        }
        assert gaps == {7200.0}, f"expected 2h gaps, got {spec['schedule']} -> {first_three}"

    def test_text_slot_renders_into_prompt(self):
        spec = fill_blueprint(
            get_blueprint("important-mail"),
            {"interval_min": "30", "criteria": "from my CEO", "deliver": "origin"},
        )
        assert "from my CEO" in spec["prompt"]

    def test_origin_threads_through(self):
        spec = fill_blueprint(
            get_blueprint("morning-brief"), {"time": "08:00"}, origin={"platform": "telegram", "chat_id": "9"}
        )
        assert spec["origin"] == {"platform": "telegram", "chat_id": "9"}


class TestRenderers:
    def test_form_schema_fields(self):
        schema = blueprint_form_schema(get_blueprint("morning-brief"))
        names = [f["name"] for f in schema["fields"]]
        assert names == ["time", "deliver"]
        assert schema["key"] == "morning-brief"

    def test_slash_command_defaults(self):
        cmd = blueprint_slash_command(get_blueprint("morning-brief"))
        assert cmd.startswith("/blueprint morning-brief")
        assert "time=08:00" in cmd

    def test_slash_command_quotes_freetext(self):
        cmd = blueprint_slash_command(
            get_blueprint("custom-reminder"), {"what": "drink water", "time": "10:00"}
        )
        assert '"drink water"' in cmd

    def test_deeplink_shape(self):
        url = blueprint_deeplink(get_blueprint("morning-brief"), {"time": "07:15"})
        assert url.startswith("hermes://blueprint/morning-brief?")
        assert "time=07" in url

    def test_catalog_entry_has_all_surfaces(self):
        entry = blueprint_catalog_entry(get_blueprint("morning-brief"))
        assert entry["command"].startswith("/blueprint")
        assert entry["appUrl"].startswith("hermes://")
        assert entry["scheduleHuman"]
        assert "fields" in entry


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs as jobs
    importlib.reload(jobs)
    return jobs


class TestCommandHandler:
    def test_bare_lists_catalog(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        res = handle_blueprint_command("")
        assert "morning-brief" in res.text and "Automation Blueprints" in res.text
        assert res.agent_seed is None

    def test_name_seeds_agent(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        # `/blueprint <name>` (no inline slots) now seeds the agent to ask
        # the user for each value conversationally instead of dumping fields.
        res = handle_blueprint_command("morning-brief")
        assert res.agent_seed is not None
        assert "morning-brief" in res.agent_seed
        assert "cronjob tool" in res.agent_seed
        # the schedule template is handed to the agent to build the cron expr
        assert "* * *" in res.agent_seed

    def test_name_match_is_forgiving(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command, match_blueprint

        # prefix match
        r, cands = match_blueprint("morning")
        assert r is not None and r.key == "morning-brief"
        # fuzzy / typo
        r2, _ = match_blueprint("mornning-brief")
        assert r2 is not None and r2.key == "morning-brief"
        # a forgiving name still seeds the agent
        res = handle_blueprint_command("morning")
        assert res.agent_seed is not None

    def test_fill_creates_job(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        res = handle_blueprint_command("morning-brief time=07:30 deliver=telegram")
        assert "Scheduled" in res.text
        assert res.agent_seed is None
        jobs = isolated_home.load_jobs()
        assert len(jobs) == 1
        assert (jobs[0].get("schedule_display") or jobs[0].get("schedule")) == "30 7 * * *"
        assert jobs[0].get("deliver") == "telegram"

    def test_unknown_blueprint(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        res = handle_blueprint_command("zzz-nope-nothing")
        assert "No automation blueprint" in res.text
        assert res.agent_seed is None

    def test_bad_value_names_slot(self, isolated_home):
        from hermes_cli.blueprint_cmd import handle_blueprint_command

        res = handle_blueprint_command("morning-brief time=99:99")
        assert "Can't set up" in res.text and "time" in res.text
        assert res.agent_seed is None


class TestDocsGenerator:
    def test_generator_emits_valid_index(self, tmp_path):
        # The generator imports the catalog and writes a flat JSON array.
        import importlib.util

        script = (
            Path(__file__).resolve().parents[2]
            / "website" / "scripts" / "extract-automation-blueprints.py"
        )
        spec = importlib.util.spec_from_file_location("extract_cron_blueprints", script)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        index = mod.build_index()
        assert isinstance(index, list) and len(index) == len(CATALOG)
        # Each entry must round-trip through json and carry the surfaces.
        json.dumps(index)
        assert all("command" in e and "appUrl" in e for e in index)
