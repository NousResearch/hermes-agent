"""Tests for per-job ``max_turns`` support in cron jobs.

A per-job ``max_turns`` caps how many agent tool-use iterations a single
scheduled job may run, independently of the global ``agent.max_turns`` default
(90). This bounds a runaway job's cost — a one-shot reminder that starts looping
stops early — without lowering the ceiling for the tenant's interactive turns.

Covers:
  - jobs._normalize_job_max_turns: positive int / string / zero / negative /
    junk / bool / None
  - jobs.create_job: param plumbing, byte-identical when unset
  - jobs.update_job: set, clear (0), re-normalize
  - tools.cronjob_tools.cronjob: create + update JSON round-trip, schema
    advertises max_turns, _format_job exposes it only when set
  - scheduler._resolve_max_iterations: per-job cap wins, config + default
    fallback, junk ignored
  - tools.blueprints: max_turns parses, validates, and reaches the job spec
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# jobs._normalize_job_max_turns
# ---------------------------------------------------------------------------

class TestNormalizeJobMaxTurns:
    def test_none_returns_none(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns(None) is None

    def test_positive_int_passthrough(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns(3) == 3
        assert _normalize_job_max_turns(90) == 90

    def test_numeric_string_coerced(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns("15") == 15

    def test_zero_returns_none(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns(0) is None

    def test_negative_returns_none(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns(-5) is None

    def test_junk_returns_none(self):
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns("nope") is None
        assert _normalize_job_max_turns([3]) is None

    def test_bool_returns_none(self):
        # A YAML ``max_turns: true`` must NOT become a cap of 1.
        from cron.jobs import _normalize_job_max_turns
        assert _normalize_job_max_turns(True) is None
        assert _normalize_job_max_turns(False) is None


# ---------------------------------------------------------------------------
# jobs.create_job and update_job
# ---------------------------------------------------------------------------

class TestCreateJobMaxTurns:
    def test_max_turns_stored_when_set(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h", max_turns=3)
        assert get_job(job["id"])["max_turns"] == 3

    def test_unset_stays_byte_identical(self, tmp_cron_dir):
        # Like attach_to_session, an unset max_turns leaves NO key on the stored
        # job so existing jobs / the common case stay byte-identical.
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h")
        assert "max_turns" not in get_job(job["id"])

    def test_nonpositive_not_stored(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h", max_turns=0)
        assert "max_turns" not in get_job(job["id"])

    def test_numeric_string_coerced_on_create(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job
        job = create_job(prompt="hello", schedule="every 1h", max_turns="7")
        assert get_job(job["id"])["max_turns"] == 7


class TestUpdateJobMaxTurns:
    def test_set_via_update(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h")
        update_job(job["id"], {"max_turns": 12})
        assert get_job(job["id"])["max_turns"] == 12

    def test_clear_with_zero(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h", max_turns=5)
        update_job(job["id"], {"max_turns": 0})
        # Cleared back to None → scheduler falls back to the config/global cap.
        assert get_job(job["id"]).get("max_turns") is None

    def test_junk_update_clears(self, tmp_cron_dir):
        from cron.jobs import create_job, get_job, update_job
        job = create_job(prompt="x", schedule="every 1h", max_turns=5)
        update_job(job["id"], {"max_turns": "bogus"})
        assert get_job(job["id"]).get("max_turns") is None


# ---------------------------------------------------------------------------
# tools.cronjob_tools: end-to-end JSON round-trip
# ---------------------------------------------------------------------------

class TestCronjobToolMaxTurns:
    def test_create_with_max_turns_json_roundtrip(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob
        result = json.loads(
            cronjob(action="create", prompt="hi", schedule="every 1h", max_turns=3)
        )
        assert result["success"] is True
        assert result["job"]["max_turns"] == 3

    def test_create_without_max_turns_hides_field(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob
        result = json.loads(
            cronjob(action="create", prompt="hi", schedule="every 1h")
        )
        assert result["success"] is True
        assert "max_turns" not in result["job"]

    def test_update_sets_and_clears_max_turns(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob
        created = json.loads(
            cronjob(action="create", prompt="hi", schedule="every 1h")
        )
        job_id = created["job_id"]

        updated = json.loads(cronjob(action="update", job_id=job_id, max_turns=8))
        assert updated["success"] is True
        assert updated["job"]["max_turns"] == 8

        cleared = json.loads(cronjob(action="update", job_id=job_id, max_turns=0))
        assert cleared["success"] is True
        assert "max_turns" not in cleared["job"]

    def test_schema_advertises_max_turns(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA
        props = CRONJOB_SCHEMA["parameters"]["properties"]
        assert "max_turns" in props
        assert props["max_turns"]["type"] == "integer"


# ---------------------------------------------------------------------------
# scheduler._resolve_max_iterations: precedence
# ---------------------------------------------------------------------------

class TestResolveMaxIterations:
    def test_per_job_cap_wins(self):
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({"max_turns": 3}, {"agent": {"max_turns": 50}}) == 3

    def test_config_agent_cap_used_when_no_job_cap(self):
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({}, {"agent": {"max_turns": 25}}) == 25

    def test_legacy_top_level_max_turns(self):
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({}, {"max_turns": 40}) == 40

    def test_default_90(self):
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({}, {}) == 90

    def test_junk_job_cap_ignored_falls_back(self):
        # A hand-edited jobs.json with a non-positive / bool value must never
        # silently disable the cap — fall back to config, then the default.
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({"max_turns": 0}, {"agent": {"max_turns": 30}}) == 30
        assert _resolve_max_iterations({"max_turns": -1}, {}) == 90
        assert _resolve_max_iterations({"max_turns": True}, {}) == 90

    def test_job_cap_can_exceed_config(self):
        # A per-job cap is authoritative in BOTH directions — a long weekly job
        # may legitimately declare more headroom than the tenant default.
        from cron.scheduler import _resolve_max_iterations
        assert _resolve_max_iterations({"max_turns": 120}, {"agent": {"max_turns": 30}}) == 120


# ---------------------------------------------------------------------------
# tools.blueprints: max_turns round-trip
# ---------------------------------------------------------------------------

class TestBlueprintMaxTurns:
    _SKILL_MD = (
        "---\n"
        "name: demo-skill\n"
        "metadata:\n"
        "  hermes:\n"
        "    blueprint:\n"
        "      schedule: '0 9 * * *'\n"
        "      prompt: do the thing\n"
        "      max_turns: 5\n"
        "---\n"
        "# Demo\n"
    )

    def test_parse_reads_max_turns(self):
        from tools.blueprints import parse_blueprint
        spec = parse_blueprint(self._SKILL_MD)
        assert spec is not None
        assert spec.max_turns == 5

    def test_job_spec_carries_max_turns(self):
        from tools.blueprints import parse_blueprint, blueprint_to_job_spec
        spec = parse_blueprint(self._SKILL_MD)
        assert blueprint_to_job_spec(spec)["max_turns"] == 5

    def test_invalid_max_turns_rejected(self):
        from tools.blueprints import parse_blueprint, BlueprintError
        bad = self._SKILL_MD.replace("max_turns: 5", "max_turns: -3")
        with pytest.raises(BlueprintError, match="max_turns"):
            parse_blueprint(bad)
