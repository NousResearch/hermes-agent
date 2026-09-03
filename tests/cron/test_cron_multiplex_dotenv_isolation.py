"""Regression tests for #87575 — cron .env reloads must not leak profile secrets.

Main scopes every multiplex cron fire to the firing profile: the ticker
thread wraps the tick in that profile's ``HERMES_HOME`` override, and
``_run_one_job_body`` installs the profile's secret and terminal scope around
run and delivery. The remaining cross-profile leak was ``run_job`` itself:
both of its ``load_hermes_dotenv`` reloads pushed the firing profile's .env
values into the shared process-global ``os.environ``, where every OTHER
profile's subprocesses would inherit them after the fire. Both reload sites
are now skipped while profile isolation is active; the firing profile's
secrets stay authoritative through the secret scope.

Ensures:
1. The ``no_agent`` script path skips ``load_hermes_dotenv`` under multiplex
   isolation and still reloads for single-profile gateways.
2. The agent path skips the pre-run reload under multiplex isolation and
   still reloads for single-profile gateways.
"""

from __future__ import annotations

import pytest

import cron.scheduler as cron_scheduler
from agent.secret_scope import is_multiplex_active, set_multiplex_active


@pytest.fixture(autouse=True)
def _restore_multiplex_flag():
    original = is_multiplex_active()
    yield
    set_multiplex_active(original)


@pytest.fixture()
def dotenv_recorder(monkeypatch):
    """Replace ``load_hermes_dotenv`` with a call recorder (no env mutation).

    ``run_agent`` is imported up front because its module-level dotenv load
    (run_agent.py startup behavior) would otherwise be attributed to
    ``run_job`` on the first import inside a test.
    """
    import run_agent  # noqa: F401  (module import side effects land here)

    calls: list[str] = []

    def _record(hermes_home=None, **_kwargs):
        calls.append(str(hermes_home))

    import hermes_cli.env_loader as env_loader

    monkeypatch.setattr(env_loader, "load_hermes_dotenv", _record)
    return calls


NO_AGENT_JOB = {
    "id": "multiplex-dotenv-test",
    "name": "Multiplex Dotenv Test",
    "no_agent": True,
    "script": "/does/not/matter.py",
}


def test_no_agent_dotenv_reload_skipped_under_multiplex(
    tmp_path, monkeypatch, dotenv_recorder
):
    """A multiplex cron fire must not push the firing profile's .env into os.environ."""
    set_multiplex_active(True)
    monkeypatch.setattr(
        cron_scheduler,
        "_run_job_script_with_claim_heartbeat",
        lambda job, script_path, workdir=None, cancel_event=None: (True, ""),
    )

    success, _doc, response, error = cron_scheduler.run_job(dict(NO_AGENT_JOB))

    assert success is True
    assert response == cron_scheduler.SILENT_MARKER
    assert error is None
    assert dotenv_recorder == []


def test_no_agent_dotenv_reload_still_runs_single_profile(
    tmp_path, monkeypatch, dotenv_recorder
):
    """Single-profile gateways keep the no_agent .env reload (delivery targets)."""
    set_multiplex_active(False)
    monkeypatch.setattr(
        cron_scheduler,
        "_run_job_script_with_claim_heartbeat",
        lambda job, script_path, workdir=None, cancel_event=None: (True, ""),
    )

    success, _doc, _response, _error = cron_scheduler.run_job(dict(NO_AGENT_JOB))

    assert success is True
    assert len(dotenv_recorder) == 1


AGENT_JOB = {
    "id": "multiplex-dotenv-agent-test",
    "name": "Multiplex Dotenv Agent Test",
    "prompt": "Say done",
}


def _install_agent_mocks(monkeypatch):
    """Drive run_job's agent path with fakes so no provider/session I/O happens."""

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.session_db = kwargs.get("session_db")

        def run_conversation(self, prompt, task_id=None):
            return {
                "completed": True,
                "failed": False,
                "final_response": "done",
                "turn_exit_reason": "",
            }

        def close(self):
            pass

    class _FakeSessionDB:
        def __init__(self, db_path=None, read_only=False):
            pass

        def set_session_title(self, *args, **kwargs):
            pass

        def end_session(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr("hermes_state.SessionDB", _FakeSessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )


def test_agent_dotenv_reload_skipped_under_multiplex(monkeypatch, dotenv_recorder):
    """The agent path's pre-run reload must be skipped under multiplex isolation."""
    set_multiplex_active(True)
    _install_agent_mocks(monkeypatch)

    success, _doc, response, _error = cron_scheduler.run_job(dict(AGENT_JOB))

    assert success is True
    assert response == "done"
    assert dotenv_recorder == []


def test_agent_dotenv_reload_still_runs_single_profile(monkeypatch, dotenv_recorder):
    """Single-profile gateways keep the pre-run .env/config reload (#33465)."""
    set_multiplex_active(False)
    _install_agent_mocks(monkeypatch)

    success, _doc, response, _error = cron_scheduler.run_job(dict(AGENT_JOB))

    assert success is True
    assert response == "done"
    assert len(dotenv_recorder) == 1
