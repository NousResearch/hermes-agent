"""Per-job reasoning_effort override: store contract + scheduler precedence.

A cron job may pin its own reasoning effort (Coatue request, NS-696).
Contract under test:

- Job store (cron/jobs.py): the field is validated at the storage choke
  point against the canonical Hermes effort grammar (parse_reasoning_effort
  in hermes_constants — the SAME parser every other effort surface uses).
  Garbage never persists; absent field keeps the job record byte-identical
  to pre-feature behavior. Capability clamping (xhigh on a model that caps
  at high, etc.) is intentionally NOT validated here — that is owned by the
  provider transports at send time, same as config-set effort.
- Scheduler resolution (cron/scheduler.py::_resolve_job_reasoning_config):
  a job-pinned effort wins outright over BOTH the global
  agent.reasoning_effort and per-model agent.reasoning_overrides; an absent
  field yields a result byte-identical to resolve_reasoning_config(cfg,
  model); a garbage value in a hand-edited store warns and falls back to
  config resolution instead of killing the tick.
"""

import pytest

from cron.jobs import create_job, load_jobs, update_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


def _create(**kw):
    kw.setdefault("prompt", "say hi")
    kw.setdefault("schedule", "every 1h")
    return create_job(**kw)


class TestJobStoreReasoningEffort:
    def test_absent_field_stores_none_and_shape_unchanged(self, tmp_cron_dir):
        """No reasoning_effort arg => None in the record; the rest of the job
        dict keeps exactly the keys pre-feature jobs had (plus the new field),
        so existing consumers see no shape drift."""
        job = _create()
        assert job.get("reasoning_effort") is None
        # The new field must not perturb sibling inference axes.
        assert job["model"] is None
        assert job["provider"] is None

    @pytest.mark.parametrize(
        "level",
        ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
    )
    def test_valid_levels_stored_normalized(self, tmp_cron_dir, level):
        job = _create(reasoning_effort=level)
        assert job["reasoning_effort"] == level
        # And it round-trips through the store.
        assert load_jobs()[0]["reasoning_effort"] == level

    @pytest.mark.parametrize("raw,expected", [("HIGH", "high"), ("  high ", "high"), ("None", "none")])
    def test_spelling_normalized_lowercase_stripped(self, tmp_cron_dir, raw, expected):
        job = _create(reasoning_effort=raw)
        assert job["reasoning_effort"] == expected

    @pytest.mark.parametrize("garbage", ["turbo", "11", "hgih", "medium-plus"])
    def test_garbage_rejected_nothing_persisted(self, tmp_cron_dir, garbage):
        with pytest.raises(ValueError) as exc:
            _create(reasoning_effort=garbage)
        # Actionable message: names the bad value and the valid grammar.
        msg = str(exc.value)
        assert garbage in msg
        assert "minimal" in msg and "ultra" in msg
        assert load_jobs() == []

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_means_unset(self, tmp_cron_dir, empty):
        job = _create(reasoning_effort=empty)
        assert job.get("reasoning_effort") is None

    def test_update_sets_field(self, tmp_cron_dir):
        job = _create()
        updated = update_job(job["id"], {"reasoning_effort": "xhigh"})
        assert updated["reasoning_effort"] == "xhigh"
        assert load_jobs()[0]["reasoning_effort"] == "xhigh"

    def test_update_empty_string_clears(self, tmp_cron_dir):
        job = _create(reasoning_effort="high")
        updated = update_job(job["id"], {"reasoning_effort": ""})
        assert updated["reasoning_effort"] is None

    def test_update_garbage_rejected_stored_value_untouched(self, tmp_cron_dir):
        job = _create(reasoning_effort="high")
        with pytest.raises(ValueError):
            update_job(job["id"], {"reasoning_effort": "warp9"})
        assert load_jobs()[0]["reasoning_effort"] == "high"

    def test_effort_change_does_not_trigger_snapshot_recompute(self, tmp_cron_dir):
        """Effort is NOT a drift-guard axis (#44585): updating it alone must
        not touch provider_snapshot/model_snapshot."""
        job = _create()
        before = (job.get("provider_snapshot"), job.get("model_snapshot"))
        updated = update_job(job["id"], {"reasoning_effort": "low"})
        assert (updated.get("provider_snapshot"), updated.get("model_snapshot")) == before


class TestSchedulerJobReasoningPrecedence:
    """Contract for cron/scheduler.py::_resolve_job_reasoning_config."""

    CFG = {
        "model": {"default": "anthropic/claude-opus-4.5"},
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"anthropic/claude-opus-4.5": "xhigh"},
        },
    }

    def test_job_effort_beats_global_and_per_model_override(self):
        from cron.scheduler import _resolve_job_reasoning_config

        job = {"reasoning_effort": "high"}
        result = _resolve_job_reasoning_config(job, self.CFG, "anthropic/claude-opus-4.5")
        assert result == {"enabled": True, "effort": "high"}

    def test_job_none_disables_thinking_never_reenabled_by_config(self):
        from cron.scheduler import _resolve_job_reasoning_config

        job = {"reasoning_effort": "none"}
        result = _resolve_job_reasoning_config(job, self.CFG, "anthropic/claude-opus-4.5")
        assert result == {"enabled": False}

    def test_absent_field_byte_identical_to_config_resolution(self):
        from hermes_constants import resolve_reasoning_config
        from cron.scheduler import _resolve_job_reasoning_config

        for model in ("anthropic/claude-opus-4.5", "gpt-5", ""):
            expected = resolve_reasoning_config(self.CFG, model)
            assert _resolve_job_reasoning_config({}, self.CFG, model) == expected
            assert _resolve_job_reasoning_config({"reasoning_effort": None}, self.CFG, model) == expected

    def test_garbage_in_store_warns_and_falls_back(self, caplog):
        """A hand-edited jobs.json with an invalid level must not kill the
        tick: warn, then resolve from config exactly as if unset."""
        import logging

        from hermes_constants import resolve_reasoning_config
        from cron.scheduler import _resolve_job_reasoning_config

        job = {"id": "abc123", "reasoning_effort": "turbo"}
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            result = _resolve_job_reasoning_config(job, self.CFG, "gpt-5")
        assert result == resolve_reasoning_config(self.CFG, "gpt-5")
        assert any("turbo" in r.message for r in caplog.records)

    def test_job_effort_is_model_independent(self):
        """Pinned effort governs whichever model actually runs (auth fallback
        can swap the model after resolution) — the job pins intent, the
        transport clamps capability."""
        from cron.scheduler import _resolve_job_reasoning_config

        job = {"reasoning_effort": "ultra"}
        for model in ("gpt-5.6-sol", "x-ai/grok-4", "gemini-3-pro", ""):
            assert _resolve_job_reasoning_config(job, self.CFG, model) == {
                "enabled": True,
                "effort": "ultra",
            }


class TestCronjobToolReasoningEffort:
    """The model tool covers BOTH mutation verbs (create AND update) and
    surfaces the field in job listings — the mutation-verb symmetry rule."""

    def test_create_via_tool_persists_pin(self, tmp_cron_dir, monkeypatch):
        import json

        from tools.cronjob_tools import cronjob

        out = json.loads(
            cronjob(
                action="create",
                prompt="daily digest",
                schedule="every 1h",
                reasoning_effort="high",
            )
        )
        assert out["success"] is True
        assert out["job"]["reasoning_effort"] == "high"
        assert load_jobs()[0]["reasoning_effort"] == "high"

    def test_update_via_tool_sets_and_clears_pin(self, tmp_cron_dir):
        import json

        from tools.cronjob_tools import cronjob

        job = _create()
        set_out = json.loads(
            cronjob(action="update", job_id=job["id"], reasoning_effort="XHIGH")
        )
        assert set_out["success"] is True
        assert load_jobs()[0]["reasoning_effort"] == "xhigh"

        clear_out = json.loads(
            cronjob(action="update", job_id=job["id"], reasoning_effort="")
        )
        assert clear_out["success"] is True
        assert load_jobs()[0].get("reasoning_effort") is None

    def test_update_via_tool_garbage_is_clean_tool_error(self, tmp_cron_dir):
        import json

        from tools.cronjob_tools import cronjob

        job = _create(reasoning_effort="low")
        out = json.loads(
            cronjob(action="update", job_id=job["id"], reasoning_effort="turbo")
        )
        assert out["success"] is False
        assert "turbo" in out["error"]
        # Stored value untouched by the failed update.
        assert load_jobs()[0]["reasoning_effort"] == "low"

    def test_format_job_omits_field_when_unset(self, tmp_cron_dir):
        import json

        from tools.cronjob_tools import cronjob

        _create()
        listed = json.loads(cronjob(action="list"))["jobs"][0]
        assert "reasoning_effort" not in listed

    def test_schema_exposes_reasoning_effort(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        prop = CRONJOB_SCHEMA["parameters"]["properties"]["reasoning_effort"]
        desc = prop["description"]
        # Prompt-surface honesty: the description must state the full level
        # grammar, precedence, transport clamping, and the clear semantics.
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
            assert level in desc
        assert "clamp" in desc
        assert "empty string" in desc
