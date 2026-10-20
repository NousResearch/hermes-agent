"""Per-job api_max_retries override: producer, store contract, and the E2E
path from a created job to the constructed agent.

A cron job may pin its own API retry budget — how many attempts each model
API call gets on transient errors before the fallback-provider chain engages
— independent of the global ``agent.api_max_retries``. Contract under test:

- Producer (``tools/cronjob_tools.py::cronjob`` create/update, reached from
  ``hermes cron create/edit --api-max-retries``): the value is accepted,
  validated and persisted. The model-facing surface deliberately does NOT
  expose it (same standing policy as model/provider/reasoning_effort — models
  do not make unattended-spend decisions), so the tool schema omits it and
  the registry dispatch drops a hallucinated argument.
- Job store (``cron/jobs.py``): validated at the storage choke point
  (``_normalize_api_max_retries``) — integers clamp to >= 1, garbage raises
  so nothing invalid ever persists for a fire-and-forget job, and an absent
  field keeps the record byte-identical to pre-feature behavior.
- Scheduler consumption (``cron/scheduler.py::_apply_job_api_max_retries``):
  a pinned budget overrides the agent's config-resolved ``_api_max_retries``;
  an absent pin is a no-op; a garbage value in a hand-edited store warns and
  keeps the agent default instead of killing the tick.
- End to end: a job created through the supported ``cronjob(action='create')``
  producer, reloaded from the store, reaches the constructed agent with the
  pinned budget on it — the wiring a raw-dict scheduler test cannot see.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable (mirrors tests/cron/test_cron_provider_pin.py).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.jobs import create_job, get_job, load_jobs, update_job


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


class TestJobStoreApiMaxRetries:
    def test_absent_field_not_persisted(self, tmp_cron_dir):
        """No api_max_retries arg => the key is absent, so an existing job's
        record shape is unchanged and the job follows agent.api_max_retries."""
        job = _create()
        assert "api_max_retries" not in job
        assert load_jobs()[0].get("api_max_retries") is None

    @pytest.mark.parametrize("value", [1, 2, 6, 25])
    def test_valid_counts_stored(self, tmp_cron_dir, value):
        job = _create(api_max_retries=value)
        assert job["api_max_retries"] == value
        # And it round-trips through the store.
        assert load_jobs()[0]["api_max_retries"] == value

    @pytest.mark.parametrize("raw,expected", [("6", 6), (" 4 ", 4)])
    def test_numeric_strings_coerced(self, tmp_cron_dir, raw, expected):
        """argparse hands the CLI a string; the store owns the coercion."""
        assert _create(api_max_retries=raw)["api_max_retries"] == expected

    @pytest.mark.parametrize("value", [0, -3])
    def test_below_floor_clamps_to_one(self, tmp_cron_dir, value):
        """Clamped at the store, not at fire time, so the persisted record
        reads exactly as it behaves (1 = single attempt, no retry)."""
        assert _create(api_max_retries=value)["api_max_retries"] == 1

    @pytest.mark.parametrize("garbage", ["six", "3.5", [4], {"n": 4}])
    def test_garbage_rejected_at_create(self, tmp_cron_dir, garbage):
        """A fire-and-forget job must never persist a pin that silently
        degrades to the default at 3am."""
        with pytest.raises(ValueError):
            _create(api_max_retries=garbage)
        assert load_jobs() == []

    @pytest.mark.parametrize("value", [3.5, 2.999, 1.0])
    def test_floats_rejected_not_truncated(self, tmp_cron_dir, value):
        """A float must raise, not silently truncate.

        ``int(3.5)`` is ``3``, so without an explicit guard a float pin
        persists a DIFFERENT budget than the caller asked for — the exact
        silent-degradation failure this normalizer exists to prevent, and
        which the string form ``"3.5"`` is already rejected for. Reported by
        @Enough1122: the garbage parametrization above covers only the string.

        ``1.0`` is included deliberately: it round-trips through ``int()``
        without changing value, so a truncation-only check would let it pass
        and leave the type contract half-enforced.
        """
        with pytest.raises(ValueError):
            _create(api_max_retries=value)
        assert load_jobs() == []

    def test_floats_rejected_at_update(self, tmp_cron_dir):
        """Same guard on the update path — the stored pin stays untouched."""
        job = _create(api_max_retries=6)
        with pytest.raises(ValueError):
            update_job(job["id"], {"api_max_retries": 3.5})
        assert load_jobs()[0]["api_max_retries"] == 6

    @pytest.mark.parametrize("flag", [True, False])
    def test_booleans_rejected_not_coerced(self, tmp_cron_dir, flag):
        """int(True) == 1 would silently turn a YAML `true` into a retry
        budget of 1 — i.e. disable retries. Reject instead."""
        with pytest.raises(ValueError):
            _create(api_max_retries=flag)

    def test_empty_string_is_unset(self, tmp_cron_dir):
        assert "api_max_retries" not in _create(api_max_retries="")

    def test_update_sets_field(self, tmp_cron_dir):
        job = _create()
        updated = update_job(job["id"], {"api_max_retries": 8})
        assert updated["api_max_retries"] == 8
        assert load_jobs()[0]["api_max_retries"] == 8

    def test_update_empty_string_clears(self, tmp_cron_dir):
        job = _create(api_max_retries=6)
        assert update_job(job["id"], {"api_max_retries": ""})["api_max_retries"] is None

    def test_update_garbage_rejected_stored_value_untouched(self, tmp_cron_dir):
        job = _create(api_max_retries=6)
        with pytest.raises(ValueError):
            update_job(job["id"], {"api_max_retries": "lots"})
        assert load_jobs()[0]["api_max_retries"] == 6

    def test_update_clamps_below_floor(self, tmp_cron_dir):
        job = _create(api_max_retries=6)
        assert update_job(job["id"], {"api_max_retries": 0})["api_max_retries"] == 1

    def test_retries_change_does_not_trigger_snapshot_recompute(self, tmp_cron_dir):
        """The retry budget is NOT a drift-guard axis (#44585 guard
        unchanged): updating it alone must not touch the snapshots."""
        job = _create()
        before = (job.get("provider_snapshot"), job.get("model_snapshot"))
        updated = update_job(job["id"], {"api_max_retries": 6})
        assert (updated.get("provider_snapshot"), updated.get("model_snapshot")) == before


class TestSchedulerAppliesJobApiMaxRetries:
    """Contract for cron/scheduler.py::_apply_job_api_max_retries."""

    def _agent(self, default=3):
        agent = MagicMock()
        agent._api_max_retries = default
        return agent

    def test_pin_overrides_agent_default(self):
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=3)
        _apply_job_api_max_retries(agent, {"id": "j1", "api_max_retries": 6})
        assert agent._api_max_retries == 6

    def test_absent_pin_is_a_noop(self):
        """An unpinned job must be byte-identical to pre-feature behavior."""
        from cron.scheduler import _apply_job_api_max_retries

        for job in ({}, {"api_max_retries": None}):
            agent = self._agent(default=3)
            _apply_job_api_max_retries(agent, job)
            assert agent._api_max_retries == 3

    def test_pin_can_lower_the_budget(self):
        """A job on a hopeless endpoint can fail over faster than global."""
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=5)
        _apply_job_api_max_retries(agent, {"id": "j2", "api_max_retries": 1})
        assert agent._api_max_retries == 1

    def test_hand_edited_garbage_warns_and_keeps_default(self, caplog):
        """A bad pin in a hand-edited jobs.json must degrade the retry
        budget, never kill the tick."""
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=3)
        job = {"id": "abc123", "api_max_retries": "loads"}
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            _apply_job_api_max_retries(agent, job)
        assert agent._api_max_retries == 3
        assert any("loads" in r.message for r in caplog.records)

    def test_hand_edited_boolean_warns_and_keeps_default(self, caplog):
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=3)
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            _apply_job_api_max_retries(agent, {"id": "b", "api_max_retries": True})
        assert agent._api_max_retries == 3

    def test_hand_edited_float_warns_and_keeps_default(self, caplog):
        """JSON has no integer type distinction, so a hand-edited jobs.json
        can legally carry ``3.5``. The store choke point rejects floats, but
        this consumer reads the file directly and must not silently truncate
        to a budget the operator never wrote — degrade to the agent default
        and say so, matching the boolean/garbage paths.
        """
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=3)
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            _apply_job_api_max_retries(agent, {"id": "f", "api_max_retries": 3.5})
        assert agent._api_max_retries == 3
        assert any("3.5" in (r.getMessage()) for r in caplog.records)

    def test_hand_edited_below_floor_clamps(self):
        """The store clamps, but a hand-edited 0 must not disable the loop."""
        from cron.scheduler import _apply_job_api_max_retries

        agent = self._agent(default=3)
        _apply_job_api_max_retries(agent, {"id": "z", "api_max_retries": 0})
        assert agent._api_max_retries == 1


def _run_persisted_job(job_id, tmp_path):
    """Drive the real run_job path with the job as it was PERSISTED.

    Returns (success, error, effective_api_max_retries) where the third value
    is read off the agent the scheduler actually constructed — the wiring a
    raw-dict scheduler test cannot see.
    """
    from cron.scheduler import run_job

    job = get_job(job_id)
    assert job is not None, "job was not persisted"

    fake_db = MagicMock()
    seen = {}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        agent = MagicMock()
        # The agent resolves the global default from config at construction
        # (agent/agent_init.py); the scheduler override lands on top of it.
        agent._api_max_retries = 3
        agent.run_conversation.side_effect = lambda *a, **kw: (
            seen.update(api_max_retries=agent._api_max_retries),
            {"final_response": "ok"},
        )[1]
        mock_agent_cls.return_value = agent

        success, _output, _final, error = run_job(job)

    return success, error, seen.get("api_max_retries")


class TestCronjobProducerEndToEnd:
    """The reviewer-visible contract: a job created through the supported
    producer reaches the constructed agent with its pinned retry budget."""

    def test_created_job_budget_reaches_the_agent(self, tmp_cron_dir, tmp_path):
        from tools.cronjob_tools import cronjob

        out = json.loads(
            cronjob(
                action="create",
                prompt="morning digest",
                schedule="every 1h",
                model="gpt-5-codex",
                provider="openrouter",
                api_max_retries=6,
            )
        )
        assert out["success"] is True
        assert out["job"]["api_max_retries"] == 6

        success, error, effective = _run_persisted_job(out["job_id"], tmp_path)
        assert success is True, error
        assert effective == 6

    def test_unpinned_created_job_keeps_agent_default(self, tmp_cron_dir, tmp_path):
        """No pin => the agent's config-resolved budget is untouched."""
        from tools.cronjob_tools import cronjob

        out = json.loads(
            cronjob(
                action="create",
                prompt="morning digest",
                schedule="every 1h",
                model="gpt-5-codex",
                provider="openrouter",
            )
        )
        assert out["success"] is True
        assert "api_max_retries" not in out["job"]

        success, error, effective = _run_persisted_job(out["job_id"], tmp_path)
        assert success is True, error
        assert effective == 3

    def test_updated_job_budget_reaches_the_agent(self, tmp_cron_dir, tmp_path):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(
                action="create",
                prompt="morning digest",
                schedule="every 1h",
                model="gpt-5-codex",
                provider="openrouter",
            )
        )
        updated = json.loads(
            cronjob(
                action="update",
                job_id=created["job_id"],
                api_max_retries=9,
            )
        )
        assert updated["success"] is True

        success, error, effective = _run_persisted_job(created["job_id"], tmp_path)
        assert success is True, error
        assert effective == 9

    def test_producer_rejects_garbage_and_persists_nothing(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        out = json.loads(
            cronjob(
                action="create",
                prompt="morning digest",
                schedule="every 1h",
                api_max_retries="six",
            )
        )
        assert out.get("success") is False
        assert "api_max_retries" in out.get("error", "")
        assert load_jobs() == []


class TestCronjobToolApiMaxRetriesPolicy:
    """The model tool READS the field (list surfacing) but must never WRITE
    it: models do not make unattended-spend decisions (same standing policy
    as model/provider/base_url/reasoning_effort). The pin is set via
    `hermes cron create/edit --api-max-retries` only."""

    def _tool_handler(self):
        import tools.cronjob_tools as mod

        return mod.registry._tools["cronjob_manage"].handler

    def test_format_job_surfaces_pin_when_set(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        _create(api_max_retries=6)
        assert json.loads(cronjob(action="list"))["jobs"][0]["api_max_retries"] == 6

    def test_format_job_surfaces_floor_pin(self, tmp_cron_dir):
        """1 is falsy-adjacent in the sense that reviewers reach for
        `if job.get(...)`; it must still surface."""
        _create(api_max_retries=1)

        from tools.cronjob_tools import cronjob

        assert json.loads(cronjob(action="list"))["jobs"][0]["api_max_retries"] == 1

    def test_format_job_omits_field_when_unset(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        _create()
        assert "api_max_retries" not in json.loads(cronjob(action="list"))["jobs"][0]

    def test_schema_does_not_expose_api_max_retries(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        assert "api_max_retries" not in CRONJOB_SCHEMA["parameters"]["properties"]

    def test_tool_dispatch_drops_api_max_retries_arg(self, tmp_cron_dir):
        """Even if a model hallucinates the argument, dispatch ignores it:
        the created job must carry NO pin."""
        out = json.loads(
            self._tool_handler()(
                {
                    "action": "create",
                    "prompt": "daily digest",
                    "schedule": "every 1h",
                    "api_max_retries": 50,
                }
            )
        )
        assert out["success"] is True
        assert load_jobs()[0].get("api_max_retries") is None

    def test_tool_dispatch_cannot_raise_an_existing_pin(self, tmp_cron_dir):
        """The update door is closed to the model too: the argument is
        dropped before it becomes an update, so the pin is untouched."""
        job = _create(api_max_retries=2)
        out = json.loads(
            self._tool_handler()(
                {"action": "update", "job_id": job["id"], "api_max_retries": 99}
            )
        )
        # Nothing the tool accepts was supplied, so there is no update to make.
        assert out["success"] is False
        assert load_jobs()[0]["api_max_retries"] == 2


class TestCliProducerWiring:
    """`hermes cron create/edit --api-max-retries` is the supported producer:
    the flag must parse and reach cronjob()."""

    def _parse(self, argv):
        import argparse

        from hermes_cli.subcommands.cron import build_cron_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        build_cron_parser(sub, cmd_cron=lambda args: 0)
        return parser.parse_args(argv)

    def test_create_flag_parses(self):
        args = self._parse(["cron", "create", "every 1h", "ping", "--api-max-retries", "6"])
        assert args.api_max_retries == "6"

    def test_edit_flag_parses(self):
        args = self._parse(["cron", "edit", "abc123", "--api-max-retries", "6"])
        assert args.api_max_retries == "6"

    def test_create_forwards_to_cronjob(self, tmp_cron_dir):
        from hermes_cli.cron import cron_create

        args = self._parse(["cron", "create", "every 1h", "ping", "--api-max-retries", "6"])
        assert cron_create(args) == 0
        assert load_jobs()[0]["api_max_retries"] == 6

    def test_edit_forwards_to_cronjob(self, tmp_cron_dir):
        from hermes_cli.cron import cron_edit

        job = _create()
        args = self._parse(["cron", "edit", job["id"], "--api-max-retries", "6"])
        assert cron_edit(args) == 0
        assert load_jobs()[0]["api_max_retries"] == 6

    def test_edit_empty_string_clears(self, tmp_cron_dir):
        from hermes_cli.cron import cron_edit

        job = _create(api_max_retries=6)
        args = self._parse(["cron", "edit", job["id"], "--api-max-retries", ""])
        assert cron_edit(args) == 0
        assert load_jobs()[0].get("api_max_retries") is None
