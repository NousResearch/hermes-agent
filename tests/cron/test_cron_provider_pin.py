"""Provider-drift fail-closed guard for cron jobs (#44585).

Background: an UNPINNED cron job follows the global default provider. If that
global state is changed (e.g. a temporary switch to a paid provider like
nous/claude-fable-5), the job would silently inherit it on its next tick and
spend real money — the $7.73 incident.

The fix has two halves:
  - create_job() snapshots the provider resolution WOULD pick at creation into
    job["provider_snapshot"] (only for unpinned, agent-backed jobs).
  - run_job() fails closed when an unpinned job's CURRENTLY-resolved provider
    differs from that snapshot: it skips the run, makes no paid call, and
    delivers a loud actionable error.

These tests exercise the full run_job path (real imports, mocked AIAgent +
resolve_runtime_provider against a temp HERMES_HOME) and the create_job
snapshot capture. They are load-bearing: without the guard, cases (b) call the
agent and "succeed" instead of failing closed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "pin-test",
        "name": "pin test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run_with_current_provider(job, current_provider, tmp_path):
    """Drive run_job with resolve_runtime_provider pinned to ``current_provider``.

    Returns (success, output, final_response, error, agent_constructed).
    """
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": current_provider,
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        success, output, final_response, error = run_job(job)
        agent_constructed = mock_agent_cls.called

    return success, output, final_response, error, agent_constructed


class TestProviderDriftGuard:
    def test_a_unpinned_snapshot_matches_runs_normally(self, tmp_path):
        """(a) Unpinned job whose snapshot == current provider → runs normally."""
        job = _base_job(provider_snapshot="openrouter")
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider(job, "openrouter", tmp_path)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert agent_constructed is True

    def test_b_unpinned_snapshot_differs_fails_closed(self, tmp_path):
        """(b) Unpinned job whose snapshot != current provider → fail closed.

        The paid call must NOT be made (AIAgent never constructed) and the
        delivered error must name both providers and tell the user to pin.
        """
        job = _base_job(provider_snapshot="openrouter")
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider(job, "nous", tmp_path)

        # Fail closed: no agent constructed, no inference call.
        assert agent_constructed is False
        assert success is False
        assert error is not None

        # Loud + actionable: names both providers, mentions spend + pinning.
        blob = f"{error}\n{output}".lower()
        assert "openrouter" in blob
        assert "nous" in blob
        assert "spend" in blob
        assert "cronjob action=update" in blob
        assert "44585" in blob

    def test_c_no_snapshot_runs_backcompat(self, tmp_path):
        """(c) Pre-existing job with NO provider_snapshot → runs (back-compat).

        Even though the current provider differs from anything, a job without a
        snapshot must behave exactly as before this fix: the guard never engages.
        """
        # A job dict that predates the field entirely (key absent, not None).
        job = _base_job()
        job.pop("provider_snapshot", None)
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider(job, "nous", tmp_path)

        assert success is True
        assert error is None
        assert agent_constructed is True

    def test_c2_snapshot_none_runs_backcompat(self, tmp_path):
        """(c') Job with provider_snapshot explicitly None → runs (back-compat)."""
        job = _base_job(provider_snapshot=None)
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider(job, "nous", tmp_path)

        assert success is True
        assert error is None
        assert agent_constructed is True

    def test_d_explicitly_pinned_runs_regardless_of_drift(self, tmp_path):
        """(d) Explicitly-pinned job (job["provider"] set) → runs regardless.

        A pinned job does not follow global state, so even a snapshot/current
        mismatch must not skip it. (Snapshot would normally be None for pinned
        jobs, but we set a mismatching one to prove the pin wins.)
        """
        job = _base_job(provider="openrouter", provider_snapshot="anthropic")
        # Current resolution differs from the (stale) snapshot, but the job is
        # pinned, so the guard must not engage.
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider(job, "nous", tmp_path)

        assert success is True
        assert error is None
        assert agent_constructed is True


class TestCreateJobSnapshot:
    """create_job captures provider_snapshot for unpinned agent jobs only."""

    @staticmethod
    def _isolate_storage(monkeypatch):
        """Patch cron.jobs storage so create_job never touches the real store."""
        import contextlib
        import cron.jobs as jobs

        @contextlib.contextmanager
        def _noop_lock():
            yield

        monkeypatch.setattr(jobs, "_jobs_lock", _noop_lock, raising=True)
        monkeypatch.setattr(jobs, "load_jobs", lambda: [], raising=True)
        monkeypatch.setattr(jobs, "save_jobs", lambda j: None, raising=True)
        return jobs

    def test_unpinned_job_captures_snapshot(self, monkeypatch):
        jobs = self._isolate_storage(monkeypatch)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "openrouter"},
        ):
            job = jobs.create_job(prompt="do a thing", schedule="every 1 hour")

        assert job["provider"] is None
        assert job["provider_snapshot"] == "openrouter"

    def test_pinned_job_skips_snapshot(self, monkeypatch):
        jobs = self._isolate_storage(monkeypatch)

        resolver = MagicMock(return_value={"provider": "openrouter"})
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", resolver):
            job = jobs.create_job(
                prompt="do a thing", schedule="every 1 hour", provider="nous"
            )

        # Explicit provider → pinned → no snapshot needed, and resolution skipped.
        assert job["provider"] == "nous"
        assert job["provider_snapshot"] is None
        resolver.assert_not_called()

    def test_snapshot_resolution_error_fails_open_to_none(self, monkeypatch):
        """If resolution raises at creation, snapshot is None — creation never breaks."""
        jobs = self._isolate_storage(monkeypatch)

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=RuntimeError("no creds"),
        ):
            job = jobs.create_job(prompt="do a thing", schedule="every 1 hour")

        assert job["provider_snapshot"] is None

    def test_unpinned_model_captures_model_snapshot(self, monkeypatch, tmp_path):
        """An unpinned model captures config.yaml model.default into model_snapshot."""
        jobs = self._isolate_storage(monkeypatch)
        (tmp_path / "config.yaml").write_text("model:\n  default: llama-3.3-70b:free\n")
        monkeypatch.setattr(
            "cron.jobs.get_hermes_home", lambda: tmp_path, raising=True
        )
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "openrouter"},
        ):
            job = jobs.create_job(prompt="do a thing", schedule="every 1 hour")
        assert job["model"] is None
        assert job["model_snapshot"] == "llama-3.3-70b:free"

    def test_pinned_model_skips_model_snapshot(self, monkeypatch, tmp_path):
        """An explicit model → pinned → no model_snapshot captured."""
        jobs = self._isolate_storage(monkeypatch)
        (tmp_path / "config.yaml").write_text("model:\n  default: llama-3.3-70b:free\n")
        monkeypatch.setattr(
            "cron.jobs.get_hermes_home", lambda: tmp_path, raising=True
        )
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "openrouter"},
        ):
            job = jobs.create_job(
                prompt="do a thing", schedule="every 1 hour", model="my-model"
            )
        assert job["model"] == "my-model"
        assert job["model_snapshot"] is None


class TestUpdateJobRecomputesSnapshots:
    """`update_job` must rewrite provider/model snapshots whenever the caller
    explicitly passes any inference-routing field, even when the new value
    equals the old. Re-applying the same pin is a common UI pattern (form
    auto-loads current values, user saves without changing) and must not
    leave stale snapshots in place.
    """

    @staticmethod
    def _isolate_storage(monkeypatch):
        import contextlib
        import cron.jobs as jobs

        @contextlib.contextmanager
        def _noop_lock():
            yield

        monkeypatch.setattr(jobs, "_jobs_lock", _noop_lock, raising=True)

        store = []

        def _load():
            return list(store)

        def _save(jobs_list):
            store.clear()
            store.extend(jobs_list)

        monkeypatch.setattr(jobs, "load_jobs", _load, raising=True)
        monkeypatch.setattr(jobs, "save_jobs", _save, raising=True)
        return jobs, store

    def test_update_explicit_pin_clears_snapshot(self, monkeypatch, tmp_path):
        """A pinned job's snapshot must be None; re-applying the same pin must
        still rewrite the (previously stale) snapshot to None."""
        jobs, store = self._isolate_storage(monkeypatch)
        (tmp_path / "config.yaml").write_text("model:\n  default: old-global\n")
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)

        # Job created when it was still following global defaults.
        store.append(
            {
                "id": "upd-1",
                "name": "stale pin",
                "prompt": "x",
                "provider": "openrouter",
                "model": "old-model",
                "provider_snapshot": "openrouter",
                "model_snapshot": "old-global",
                "base_url": None,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 1 hour"},
                "enabled": True,
                "state": "scheduled",
            }
        )

        # Re-apply the same explicit pin values. Without the fix this leaves
        # the stale snapshots in place; with the fix, the re-applied explicit
        # pin (provider != None, model != None) must clear both snapshots
        # because pinned axes have no snapshot.
        result = jobs.update_job(
            "upd-1",
            {"provider": "openrouter", "model": "old-model"},
        )

        assert result["provider"] == "openrouter"
        assert result["model"] == "old-model"
        assert result["provider_snapshot"] is None
        assert result["model_snapshot"] is None

    def test_update_unpin_captures_fresh_snapshot(self, monkeypatch, tmp_path):
        """Clearing a pinned field must capture the current global default
        into the snapshot, even if the surrounding pinned fields are unchanged.
        """
        jobs, store = self._isolate_storage(monkeypatch)
        (tmp_path / "config.yaml").write_text("model:\n  default: fresh-global\n")
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)

        store.append(
            {
                "id": "upd-2",
                "name": "pinning provider only",
                "prompt": "x",
                "provider": "openrouter",
                "model": None,
                "provider_snapshot": None,
                "model_snapshot": None,
                "base_url": None,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 1 hour"},
                "enabled": True,
                "state": "scheduled",
            }
        )

        # Caller un-pins the provider by passing None. model stays None too,
        # so both are unpinned and the call must capture the fresh global
        # values into snapshots.
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "openrouter"},
        ):
            result = jobs.update_job(
                "upd-2",
                {"provider": None},
            )

        # provider remains effectively unpinned (None), snapshot is the
        # currently-resolved provider, and model_snapshot matches config.yaml.
        assert result["provider"] is None
        assert result["provider_snapshot"] == "openrouter"
        assert result["model_snapshot"] == "fresh-global"

    def test_update_without_inference_fields_leaves_snapshots(self, monkeypatch, tmp_path):
        """When the caller doesn't touch provider/model/base_url/no_agent at
        all, snapshots must be untouched (no churn, no surprise behavior)."""
        jobs, store = self._isolate_storage(monkeypatch)
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)

        store.append(
            {
                "id": "upd-3",
                "name": "no-op update",
                "prompt": "x",
                "provider": "openrouter",
                "model": None,
                "provider_snapshot": "openrouter",
                "model_snapshot": "old-global",
                "base_url": None,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 60, "display": "every 1 hour"},
                "enabled": True,
                "state": "scheduled",
            }
        )

        result = jobs.update_job("upd-3", {"name": "renamed"})

        assert result["name"] == "renamed"
        # snapshots untouched because the caller didn't ask about inference.
        assert result["provider_snapshot"] == "openrouter"
        assert result["model_snapshot"] == "old-global"


def _run_with_current_provider_and_model(job, current_provider, current_model, tmp_path):
    """Drive run_job with resolved provider pinned and config.yaml model.default
    set to ``current_model`` (the unpinned-model fire-time source)."""
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {current_model}\n"
    )
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": current_provider,
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(job)
        agent_constructed = mock_agent_cls.called
    return success, output, final_response, error, agent_constructed


class TestModelDriftGuard:
    """#44585 C1: model drift on the SAME provider must also fail closed —
    the incident named a model (claude-fable-5), and an unpinned job reads
    config.yaml model.default fresh every tick independently of provider."""

    def test_model_drift_same_provider_fails_closed(self, tmp_path):
        # Provider unchanged (openrouter==openrouter), but the global default
        # MODEL swapped to a premium model since creation → must fail closed.
        job = _base_job(
            provider_snapshot="openrouter",
            model_snapshot="llama-3.3-70b-instruct:free",
        )
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider_and_model(
                job, "openrouter", "claude-fable-5", tmp_path
            )
        assert agent_constructed is False, "paid call must not be made on model drift"
        assert success is False
        blob = f"{error}\n{output}".lower()
        assert "claude-fable-5" in blob
        assert "llama-3.3-70b-instruct:free" in blob
        assert "44585" in blob

    def test_model_snapshot_matches_runs(self, tmp_path):
        # Default model unchanged → runs normally.
        job = _base_job(
            provider_snapshot="openrouter",
            model_snapshot="llama-3.3-70b-instruct:free",
        )
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider_and_model(
                job, "openrouter", "llama-3.3-70b-instruct:free", tmp_path
            )
        assert agent_constructed is True
        assert success is True

    def test_pinned_model_bypasses_guard(self, tmp_path):
        # Explicit job["model"] → not unpinned → no model-drift skip even if the
        # global default differs from any snapshot.
        job = _base_job(
            provider_snapshot="openrouter",
            model_snapshot="old-model",
            model="my-pinned-model",
        )
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider_and_model(
                job, "openrouter", "claude-fable-5", tmp_path
            )
        assert agent_constructed is True
        assert success is True

    def test_no_model_snapshot_backcompat(self, tmp_path):
        # Pre-existing job without model_snapshot → no model-drift skip.
        job = _base_job(provider_snapshot="openrouter")  # no model_snapshot key set to a value
        success, output, final_response, error, agent_constructed = \
            _run_with_current_provider_and_model(
                job, "openrouter", "claude-fable-5", tmp_path
            )
        assert agent_constructed is True
        assert success is True


class TestRuntimeResolutionTargetModel:
    """run_job must resolve the primary provider against the model the job
    will actually run (per-job pin > env > config default), so providers with
    model-specific api_mode routing (e.g. OpenCode Zen/Go) pick the mode for
    the pinned model instead of the stale persisted default."""

    def test_primary_resolution_passes_effective_model(self, tmp_path):
        job = _base_job(model="my-pinned-model", provider="openrouter")
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            }

        fake_db = MagicMock()
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 side_effect=_capture,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            run_job(job)

        assert captured.get("target_model") == "my-pinned-model"
        assert captured.get("requested") == "openrouter"
