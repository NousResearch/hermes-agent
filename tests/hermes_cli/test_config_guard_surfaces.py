"""Fail-closed corrupt-config guards on gateway, serve, and cron surfaces.

Companion to test_noninteractive_config_guard.py (PR #81988): issue #81952
extended to every non-interactive startup surface.
"""

from __future__ import annotations

import os

import pytest

@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    yield
    os.environ.pop("HERMES_IGNORE_USER_CONFIG", None)

def _write_corrupt_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: [unterminated\n", encoding="utf-8")
    return path

class TestGatewayGuard:
    def test_gateway_refuses_corrupt_config(self, tmp_path, capsys):
        from gateway.run import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            _guard_corrupt_user_config()

        assert exc_info.value.code == 2
        assert "Hermes stopped because your settings file" in capsys.readouterr().err

    def test_gateway_allows_valid_config(self, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        (tmp_path / "config.yaml").write_text("model:\n  default: local/test\n")
        _guard_corrupt_user_config()  # must not raise

    def test_gateway_allows_missing_config(self, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        _guard_corrupt_user_config()  # first-run state: must not raise

    def test_gateway_escape_hatch(self, monkeypatch, tmp_path):
        from gateway.run import _guard_corrupt_user_config

        _write_corrupt_config(tmp_path)
        monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
        _guard_corrupt_user_config()  # must not raise

class TestCronRunJobGuard:
    def _job(self, **overrides):
        job = {"id": "job-test-1", "name": "guard test", "prompt": "hi"}
        job.update(overrides)
        return job

    @pytest.mark.asyncio
    async def test_run_job_fails_closed_on_corrupt_config(self, tmp_path):
        import json
        from types import SimpleNamespace

        from gateway.session_contract import SessionRef
        from gateway.session_cron import current_execution, execute
        from hermes_state_registry import acquire, release

        # Agent-backed run_job is now a client. Exercise the scheduler's guard
        # through its owner entry, without launching a daemon or bypassing it.
        # The store is opened before the config is corrupted: store construction
        # now resolves the profile database and fails OPEN on a corrupted config
        # (see test_store_construction_recovers_sqlite_on_corrupt_config), so the
        # guard under test here is reached with an already-open store; the guard's
        # own fail-closed config check still refuses the corrupt file.
        db = acquire(tmp_path / "state.db")
        _write_corrupt_config(tmp_path)
        ref = SessionRef("default", "cron-guard")
        db.create_session(ref.session_id, source="cron")
        owner = SimpleNamespace(db=db, pending_results={}, sessions={
            ref.session_id: SimpleNamespace(source=SimpleNamespace(user_id="cron-owner"))})
        admission = {"admission_id": "guard-fire", "request_id": "guard-fire",
                     "principal_id": "cron-owner", "owner_epoch": 1, "generation": 1, "payload": {"text": ""}}
        policy = SimpleNamespace(request_json=json.dumps({
            "cron_job": self._job(), "extra_prompt": None, "request_id": "guard-fire"}))
        previous = current_execution()
        try:
            with pytest.raises(RuntimeError, match="Hermes stopped because your settings file"):
                await execute(owner, ref, admission, policy)
            success, _output_doc, final_response, error = owner.pending_results["guard-fire"]["result"]["cron_result"]
            assert current_execution() is previous
            assert not owner._cron_cancellations
        finally:
            release(db)

        assert success is False
        assert error is not None
        assert "Hermes stopped because your settings file" in error
        assert "config.yaml" in error
        assert final_response == ""

    def test_store_construction_recovers_sqlite_on_corrupt_config(self, tmp_path):
        """Opening a store resolves the profile database config and fails open.

        The dispatch seam must not abort a reader surface (``hermes sessions repair``
        is the documented recovery path for a damaged store) when the owning
        profile's config.yaml is torn mid-edit: the last-good user layer is served,
        and with no last-good layer at all the construction falls to the SQLite
        default. Readable-but-invalid configs (non-mapping ``database:`` section,
        unknown backend, missing DSN) keep failing closed in the resolver.
        """
        from hermes_state_registry import acquire, release

        _write_corrupt_config(tmp_path)
        db = acquire(tmp_path / "state.db")
        try:
            assert db is not None
        finally:
            release(db)

    def test_run_job_no_agent_exempt(self, tmp_path):
        from cron.scheduler import run_job

        _write_corrupt_config(tmp_path)

        _success, _output_doc, _final_response, error = run_job(
            self._job(no_agent=True, script="true", deliver="none")
        )
        assert "Hermes stopped because your settings file" not in (error or "")

class TestServeGuard:
    def test_serve_headless_refuses_corrupt_config(self, tmp_path, capsys):
        """The `hermes serve` headless path fails closed before startup."""
        from argparse import Namespace

        from hermes_cli import main as main_mod

        _write_corrupt_config(tmp_path)
        args = Namespace(
            headless_backend=True,
            ignore_user_config=False,
            ssh_session_token_file=None,
            ssh_owner_nonce=None,
            status=False,
            stop=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            main_mod.cmd_dashboard(args)

        assert exc_info.value.code == 2
        assert "Hermes stopped because your settings file" in capsys.readouterr().err
