"""Per-job reasoning_effort override: store contract + scheduler precedence.

A cron job may pin its own reasoning effort, independent of global config.
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
        """Effort is NOT a drift-guard axis (#44585 guard unchanged): updating
        it alone must not touch provider_snapshot/model_snapshot."""
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
    """The model tool READS the field (list surfacing) but must never WRITE
    it: models don't make model-configuration decisions (standing policy —
    the only exception is user-defined profile selection). The pin is set
    via `hermes cron create/edit --reasoning-effort` only."""

    def test_format_job_surfaces_pin_when_set(self, tmp_cron_dir):
        import json

        from tools.cronjob_tools import cronjob

        _create(reasoning_effort="high")
        listed = json.loads(cronjob(action="list"))["jobs"][0]
        assert listed["reasoning_effort"] == "high"

    def test_format_job_omits_field_when_unset(self, tmp_cron_dir):
        import json

        from tools.cronjob_tools import cronjob

        _create()
        listed = json.loads(cronjob(action="list"))["jobs"][0]
        assert "reasoning_effort" not in listed

    def _tool_handler(self):
        import tools.cronjob_tools as mod

        return mod.registry._tools["cronjob"].handler

    def test_schema_does_not_expose_reasoning_effort(self):
        """Policy pin: the model-facing surface must NOT offer the
        reasoning_effort knob. Models never choose model config; the CLI is
        the only mutation surface for this field. The cronjob() function
        keeps the parameter for the CLI lane (hermes_cli/cron.py), but the
        tool schema and the registry dispatch drop it — same pattern as
        model/provider/base_url."""
        import inspect

        import tools.cronjob_tools as mod

        assert "reasoning_effort" not in mod.CRONJOB_SCHEMA["parameters"]["properties"]
        # The registry handler lambda must not forward the agent's args to
        # the parameter (mirrors the intentional model/provider omission).
        source = inspect.getsource(self._tool_handler())
        assert 'args.get("reasoning_effort")' not in source

    def test_tool_dispatch_drops_reasoning_effort_arg(self, tmp_cron_dir):
        """Even if a model hallucinates the argument, dispatch ignores it:
        the created job must carry NO pin."""
        import json

        out = json.loads(
            self._tool_handler()(
                {
                    "action": "create",
                    "prompt": "daily digest",
                    "schedule": "every 1h",
                    "reasoning_effort": "max",
                }
            )
        )
        assert out["success"] is True
        assert load_jobs()[0].get("reasoning_effort") is None
