"""Unpinned cron jobs run on the main agent model at fire time; ``pinned`` locks it.

Contract:
  - run_job() resolves per-job pin > cron.model / cron.model_provider > the main agent model
    (config ``model:``). There is no creation-time snapshot axis any more: a record that still
    carries legacy ``provider_snapshot`` / ``model_snapshot`` keys follows the main model.
  - create_job(pinned=True) / update_job({"pinned": True}) lock the CURRENT main provider+model
    onto the job as an ordinary per-job pin; ``pinned=False`` releases both.

These tests exercise the full run_job path (real imports, mocked AIAgent +
resolve_runtime_provider against a temp HERMES_HOME) and the job-store pin helpers.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _run_owned_job(job, tmp_path):
    """Exercise the production owner bridge with an isolated canonical store."""
    from gateway.session_contract import SessionRef
    from gateway.session_cron import current_execution, execute
    from hermes_state_registry import acquire, release

    ref = SessionRef("test-profile", "cron-owner-session")
    admission_id = "cron-owner-admission"
    request_id = "cron-owner-request"
    db = acquire(tmp_path / "state.db")
    db.create_session(ref.session_id, source="cron")
    authority = SimpleNamespace(
        db=db,
        sessions={ref.session_id: SimpleNamespace(source=SimpleNamespace(user_id="cron-owner"))},
        pending_results={},
    )
    row = {
        "admission_id": admission_id, "request_id": request_id,
        "principal_id": "cron-owner", "payload": {"text": ""},
    }
    policy = SimpleNamespace(request_json=json.dumps({
        "cron_job": job, "extra_prompt": None, "request_id": request_id,
    }))

    async def run():
        previous = current_execution()
        try:
            await execute(authority, ref, row, policy)
        except RuntimeError as exc:
            # Failed scheduler tuples are published before execute raises.
            saved = authority.pending_results.get(admission_id)
            if saved is None:
                raise
            assert str(exc) == saved["result"]["cron_result"][3]
        assert current_execution() is previous
        assert authority._cron_cancellations == {}
        return tuple(authority.pending_results[admission_id]["result"]["cron_result"])

    try:
        return asyncio.run(run())
    finally:
        release(db)


def _base_job(**overrides):
    job = {
        "id": "pin-test",
        "name": "pin test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, tmp_path, *, current_provider="openrouter", current_model=None, cron_model=None,
         cron_model_provider=None):
    """Drive run_job against a temp config.yaml whose ``model.default`` / ``model.provider`` are
    the CURRENT global defaults. Returns ``(success, error, agent_kwargs, resolve_kwargs)`` where
    the last two are the kwargs AIAgent / resolve_runtime_provider were called with (None when
    never called)."""
    config_yaml = ""
    if current_model or current_provider:
        config_yaml += "model:\n"
        if current_model:
            config_yaml += f"  default: {current_model}\n"
        if current_provider:
            config_yaml += f"  provider: {current_provider}\n"
    cron_lines = []
    if cron_model is not None:
        cron_lines.append(f"  model: {cron_model}")
    if cron_model_provider is not None:
        cron_lines.append(f"  model_provider: {cron_model_provider}")
    if cron_lines:
        config_yaml += "cron:\n" + "\n".join(cron_lines) + "\n"
    (tmp_path / "config.yaml").write_text(config_yaml)

    resolve_kwargs = {}

    def _resolve(**kwargs):
        resolve_kwargs.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or current_provider,
            "api_mode": "chat_completions",
        }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = _run_owned_job(job, tmp_path)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None
    return success, error, agent_kwargs, (resolve_kwargs or None)


class TestUnpinnedJobsFollowTheMainModel:
    def test_legacy_snapshot_record_follows_the_main_model(self, tmp_path):
        """A record created under the old snapshot design keeps running, on the CURRENT main
        provider/model, never on what it was created under."""
        job = _base_job(provider_snapshot="old-provider", model_snapshot="old-model")
        success, error, agent_kwargs, resolve_kwargs = _run(
            job, tmp_path, current_provider="new-provider", current_model="new-model")

        assert success is True, error
        assert agent_kwargs["model"] == "new-model"
        assert resolve_kwargs["requested"] is None
        assert resolve_kwargs["target_model"] == "new-model"

    def test_explicit_pin_then_fleet_default_beat_the_main_model(self, tmp_path):
        pinned = _base_job(provider="pinned-provider", model="pinned-model")
        success, error, agent_kwargs, resolve_kwargs = _run(
            pinned, tmp_path, current_provider="new-provider", current_model="new-model",
            cron_model="fleet-model", cron_model_provider="fleet-provider")
        assert success is True, error
        assert (agent_kwargs["model"], resolve_kwargs["requested"]) == ("pinned-model", "pinned-provider")

        success, error, agent_kwargs, resolve_kwargs = _run(
            _base_job(), tmp_path, current_provider="new-provider", current_model="new-model",
            cron_model="fleet-model", cron_model_provider="fleet-provider")
        assert success is True, error
        assert (agent_kwargs["model"], resolve_kwargs["requested"]) == ("fleet-model", "fleet-provider")

    def test_missing_model_guides_to_user_owned_cli(self, tmp_path, monkeypatch):
        """A missing-model failure cannot advertise agent-owned pinning."""
        monkeypatch.delenv("HERMES_MODEL", raising=False)
        success, error, agent_kwargs, _ = _run(
            _base_job(), tmp_path, current_provider="openrouter", current_model=None)

        assert success is False
        assert agent_kwargs is None
        assert error


class TestPinnedLocksTheMainModel:
    """``pinned`` is a lock on the main model at the time it is set, stored as a plain pin."""

    @staticmethod
    def _store(monkeypatch, tmp_path, main_model="main-model", main_provider="openrouter"):
        from cron import jobs
        (tmp_path / "config.yaml").write_text(f"model:\n  default: {main_model}\n")
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
        state = {"jobs": []}
        monkeypatch.setattr(jobs, "load_jobs", lambda: list(state["jobs"]), raising=True)
        monkeypatch.setattr(jobs, "save_jobs", lambda j: state.__setitem__("jobs", list(j)), raising=True)
        monkeypatch.setattr(jobs, "resolve_job_ref", lambda ref: next(
            (j for j in state["jobs"] if j["id"] == ref), None), raising=True)
        resolver = MagicMock(return_value={"provider": main_provider})
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
        return jobs, resolver

    def test_pinned_true_locks_then_pinned_false_releases(self, monkeypatch, tmp_path):
        jobs, _ = self._store(monkeypatch, tmp_path)

        from tools.cronjob_job_args import _format_job

        unpinned = jobs.create_job(prompt="do a thing", schedule="every 1 hour")
        assert (unpinned["model"], unpinned["provider"]) == (None, None)
        assert _format_job(unpinned)["pinned"] is False

        locked = jobs.update_job(unpinned["id"], {"pinned": True})
        assert (locked["model"], locked["provider"]) == ("main-model", "openrouter")
        assert _format_job(locked)["pinned"] is True
        assert "pinned" not in jobs.load_jobs()[0]  # derived, never stored

        # The main model moves on; the locked job does not.
        (tmp_path / "config.yaml").write_text("model:\n  default: newer-model\n")
        assert jobs.update_job(locked["id"], {"name": "renamed"})["model"] == "main-model"

        released = jobs.update_job(locked["id"], {"pinned": False})
        assert (released["model"], released["provider"]) == (None, None)

    def test_pinned_never_overrides_an_explicit_model(self, monkeypatch, tmp_path):
        jobs, resolver = self._store(monkeypatch, tmp_path)

        job = jobs.create_job(prompt="do a thing", schedule="every 1 hour", model="my-model",
                              provider="nous", pinned=True)
        assert (job["model"], job["provider"]) == ("my-model", "nous")
        resolver.assert_not_called()

        still = jobs.update_job(job["id"], {"pinned": True, "model": "other-model"})
        assert still["model"] == "other-model"


class TestRuntimeResolutionTargetModel:
    """run_job must resolve the primary provider against the model the job will actually run
    (per-job pin > cron.model > the main agent model), so providers with model-specific
    api_mode routing pick the mode for that model instead of the stale persisted default."""

    def test_primary_resolution_passes_effective_model(self, tmp_path):
        job = _base_job(model="my-pinned-model", provider="openrouter")
        success, error, _agent_kwargs, resolve_kwargs = _run(
            job, tmp_path, current_provider="openrouter", current_model="other-model")

        assert success is True, error
        assert resolve_kwargs["target_model"] == "my-pinned-model"
        assert resolve_kwargs["requested"] == "openrouter"


class TestNamedCustomPin:
    @pytest.mark.parametrize("same_endpoint", [False, True])
    def test_named_route_survives_default_change_and_unpin(self, tmp_path, monkeypatch, same_endpoint):
        """Create and edit must retain route B, even when A shares its model/endpoint."""
        from cron import jobs
        from cron.scheduler import _CronJobConfig, _resolve_job_runtime
        from hermes_cli import runtime_provider as rp

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path)
        cfg = {
            "model": {"provider": "route-b", "default": "shared-model"},
            "custom_providers": [
                {"name": "route-a", "base_url": ("https://models.example.invalid/v1" if same_endpoint
                                                  else "https://other.example.invalid/v1"),
                 "api_key": "synthetic-key-a"},
                {"name": "route-b", "base_url": "https://models.example.invalid/v1",
                 "api_key": "synthetic-key-b"},
            ],
        }
        monkeypatch.setattr(rp, "load_config", lambda: cfg)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **kw: None)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(json.dumps(cfg))
        jobs.create_job(prompt="hello", schedule="every 1 hour", pinned=True)
        edited = jobs.create_job(prompt="hello", schedule="every 1 hour")
        jobs.update_job(edited["id"], {"pinned": True})
        # The persisted rows, not the in-memory create response, reach execution.
        cfg["model"] = {"provider": "route-a", "default": "other-model"}
        config_path.write_text(json.dumps(cfg))
        jc = _CronJobConfig(cfg=cfg, model="shared-model", model_cfg=cfg["model"], cron_default_provider="")
        for saved in jobs.load_jobs():
            assert saved["provider"] == "route-b"
            assert saved["base_url"] == "https://models.example.invalid/v1"
            runtime, model = _resolve_job_runtime(saved, saved["id"], jc)
            assert (model, runtime["base_url"], runtime["api_key"]) == (
                "shared-model", "https://models.example.invalid/v1", "synthetic-key-b")
            released = jobs.update_job(saved["id"], {"pinned": False})
            assert (released["provider"], released["model"], released["base_url"]) == (None, None, None)
