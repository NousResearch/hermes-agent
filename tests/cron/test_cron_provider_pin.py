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

from unittest.mock import MagicMock, patch

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
