"""andrexibiza P1-B: ``ConfigBackendUnavailable`` is a ``SystemExit`` on purpose (a fail-open reader
must never serve defaults instead of the selected backend's config; boot fails closed), so an
``except Exception`` / ``suppress(Exception)`` around a config read does not catch it. Each
long-lived consumer below handles it explicitly with ITS outcome, and its process keeps running.

Every test raises the real ``ConfigBackendUnavailable`` from the config-read seam the consumer
actually calls and asserts the contained outcome. Each one fails (the exit propagates) with its
fix reverted.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from hermes_cli.config_backend import ConfigBackendUnavailable

OUTAGE = "Remote Config: cannot load the config for profile 'default' from https://plane: HTTP 503"


def _outage(*_a, **_kw):
    raise ConfigBackendUnavailable(OUTAGE)


# --- the six named sites ------------------------------------------------------------------------

def test_readiness_probe_reports_config_backend_unavailable(tmp_path, monkeypatch):
    """The probe whose job is to report config health reports the outage, never crashes."""
    from gateway import readiness
    monkeypatch.setattr("hermes_cli.config_backend.config_exists", _outage)
    assert readiness._probe_config(tmp_path) == {"status": "degraded", "detail": "config backend unavailable"}

    monkeypatch.setattr("hermes_cli.config_backend.config_exists", lambda _p: True)
    monkeypatch.setattr("hermes_cli.config_backend.read_config_doc_readonly", _outage)
    assert readiness._probe_config(tmp_path)["status"] == "degraded"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = readiness.collect_runtime_readiness(configured_model="m", runtime_status={})
    assert result["status"] == "degraded" and result["checks"]["config"]["status"] == "degraded"


def test_cron_tick_fails_only_that_fire_and_never_runs_on_defaults(tmp_path, monkeypatch):
    """``_load_cron_job_config`` keeps refusing defaults (the SystemExit passes its ``except
    Exception``); ``run_job`` — the fire's owner — turns it into this fire's failure result."""
    import run_agent  # run_job imports it; its import-time dotenv load reads config
    from cron import scheduler
    monkeypatch.setattr("hermes_cli.config_backend.config_exists", _outage)
    with pytest.raises(ConfigBackendUnavailable):
        scheduler._load_cron_job_config({"id": "j", "model": "m"}, "j", "job")

    # Through the real run_job as the fire's owner: the pre-agent gates pass, then the read refuses.
    monkeypatch.setattr("gateway.session_cron.current_execution", lambda: (None, "s", "j", "fire"))
    monkeypatch.setattr(scheduler, "_prepare_job_prompt", lambda *a, **k: (None, "do the thing"))
    monkeypatch.setattr(scheduler, "_reload_dotenv_and_publish_delivery_target", lambda job: None)
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    success, doc, final, error = scheduler.run_job({"id": "j", "name": "job", "prompt": "p"},
                                                   execution_id="fire")
    assert success is False and final == ""
    assert error and "ConfigBackendUnavailable" in error and "HTTP 503" in error
    assert "FAILED" in doc


def test_cron_job_body_records_the_outage_and_keeps_the_ticker(monkeypatch, tmp_path):
    """The body owner (``_run_one_job_body``) records a fire that escaped with the outage as failed
    and returns, instead of re-raising it into the ticker thread like a real SystemExit."""
    from cron import executions, jobs, scheduler
    monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: _outage())
    marked = []
    real_mark = scheduler.mark_job_run
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **kw: (marked.append((a, kw)), real_mark(*a, **kw)))
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
        fire = executions.create_execution(job["id"], source="test")["id"]
        assert scheduler._run_one_job_body(dict(jobs.get_job(job["id"]), execution_id=fire)) is False
        assert executions.get_execution(fire)["status"] == "failed"
        assert marked and marked[0][0][1] is False and "HTTP 503" in marked[0][0][2]
        assert jobs.get_job(job["id"])["last_status"] == "error"


def test_cron_pin_refuses_instead_of_dropping_the_lock(monkeypatch, tmp_path):
    """``pinned=True`` reads the main model; with the plane down that is an invalid request
    (ValueError, the create/update path's refusal), never ``(None, None)`` = silently unpinned
    and never a SystemExit out of the cronjob tool or RPC."""
    from cron import jobs
    from cron.jobs_model_pin import _main_model_pin
    monkeypatch.setattr("hermes_cli.config_backend.config_exists", _outage)
    with pytest.raises(ValueError, match="cannot pin the job to the main model.*HTTP 503"):
        _main_model_pin()
    with jobs.use_cron_store(tmp_path / "cron"):
        with pytest.raises(ValueError, match="cannot pin"):
            jobs.create_job(prompt="report", schedule="every 1h", deliver="local", pinned=True)
        assert jobs.load_jobs() == []


def test_tui_profile_cwd_returns_none(monkeypatch, tmp_path):
    from tui_gateway import server
    monkeypatch.setattr(server, "config_exists", _outage)
    assert server._profile_configured_cwd(tmp_path) is None
    assert server._default_session_cwd()  # the session-create tail still answers a cwd


def test_tui_raw_config_read_is_a_failed_read_that_refuses_to_save(monkeypatch, tmp_path):
    """``_load_cfg_raw`` answers its designed degraded value (``FailedConfigRead``) and the
    write-back primitive refuses to persist it, so the RPC answers an error instead of dying
    or saving a ``{}`` over the plane's document."""
    from hermes_cli.config_read_errors import FailedConfigRead, _refuse_failed_read
    from tui_gateway import server
    server._drop_cfg_cache()
    monkeypatch.setattr(server, "config_version", _outage)
    monkeypatch.setattr(server, "config_exists", lambda _p: True)
    data = server._load_cfg_raw()
    assert isinstance(data, FailedConfigRead) and data == {}
    assert isinstance(data.read_error, ConfigBackendUnavailable)
    with pytest.raises(RuntimeError) as refused:
        _refuse_failed_read(tmp_path / "config.yaml", data)
    assert "could not be read" in str(refused.value) and "HTTP 503" in str(refused.value)
    assert "formatting error" not in str(refused.value)  # not a YAML problem; nothing to edit


def test_prompt_attachments_degrade_to_the_profile_home_store(monkeypatch, tmp_path):
    """``file.attach`` keeps working: storage falls back to ``""`` (the profile-home store the
    sandbox mounts and the vision allowlist always cover)."""
    from tui_gateway import server
    monkeypatch.setattr("hermes_cli.config_backend.config_exists", _outage)
    assert server._profile_attachments_storage(tmp_path) == ""
    session = {"profile_home": str(tmp_path), "cwd": str(tmp_path)}
    assert server._session_home_dir(session, "attachments") == tmp_path / "attachments"


# --- the owners above those sites (the rest of the class) --------------------------------------

def test_tui_pooled_rpc_answers_once_with_the_outage(monkeypatch):
    """The request owner: a long handler that hits the outage still writes one error frame."""
    from tui_gateway import server
    monkeypatch.setitem(server._methods, "cc.refused", lambda rid, params: _outage())
    monkeypatch.setattr(server, "_LONG_HANDLERS", server._LONG_HANDLERS | {"cc.refused"})
    written = threading.Event()
    frames = []

    class Recorder:
        def write(self, obj):
            frames.append(obj)
            written.set()
            return True

        def close(self):
            pass

    assert server.dispatch({"jsonrpc": "2.0", "id": "r1", "method": "cc.refused", "params": {}}, Recorder()) is None
    assert written.wait(10)
    assert frames[0]["id"] == "r1" and "HTTP 503" in frames[0]["error"]["message"]


def test_tui_agent_build_thread_reports_the_outage(monkeypatch):
    """The deferred agent-build thread records ``agent_error`` and releases its waiters (a
    SystemExit would have ended the thread with ``agent_ready`` never set)."""
    from tui_gateway import server
    sid = "build-outage"
    session = {"session_key": "k", "agent": None, "agent_ready": threading.Event(),
               "history_lock": threading.Lock(), "profile_home": None}
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_await_resume_history", lambda sid, current: True)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda home: None)
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: _outage())
    monkeypatch.setattr(server, "_deferred_build_agent_kwargs", lambda current, db: {})
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr("tui_gateway.entry.ensure_mcp_discovery_started", lambda: None, raising=False)
    server._start_agent_build(sid, session)
    session["_agent_build_thread"].join(10)
    assert session["agent_ready"].is_set()
    assert "HTTP 503" in str(session.get("agent_error"))


def test_gateway_turn_owner_contains_the_outage_and_notifies():
    """A platform turn that hits the outage notifies the user and ends that turn only: re-raised,
    a SystemExit out of the background task stops the gateway's event loop."""
    from tests.gateway.test_baseexception_turn_notify import _event, _make_adapter, _raising_handler
    from gateway.session import build_session_key
    adapter = _make_adapter(_raising_handler(ConfigBackendUnavailable(OUTAGE)))
    event = _event()
    asyncio.run(adapter._process_message_background(event, build_session_key(event.source)))
    assert adapter.sent and "error" in adapter.sent[0].lower()


def test_gateway_api_server_answers_503():
    pytest.importorskip("aiohttp")
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms.api_server_config_guard import config_unavailable_middleware

    async def handler(_request):
        _outage()

    async def main():
        app = web.Application(middlewares=[config_unavailable_middleware])
        app.router.add_get("/v1/x", handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/v1/x")
            return resp.status, await resp.json()

    status, body = asyncio.run(main())
    assert status == 503 and body["error"]["code"] == "config_backend_unavailable"


def test_gateway_profile_reconcile_parks_the_profile(monkeypatch, tmp_path):
    """Serving a new multiplexed profile whose first config fetch fails parks that profile (the
    rescan retries it) instead of ending the reconcile watcher and the loop."""
    from gateway import run_profile_reconcile as rpr
    released = []

    class Runner(rpr.GatewayProfileReconcileMixin):
        session_authorities = type("A", (), {"for_home": staticmethod(lambda home: None)})()

    async def serve(runner, name, home):
        _outage()

    monkeypatch.setattr("gateway.run_runtime.reserve_profile_home", lambda *a: None)
    monkeypatch.setattr("gateway.run_runtime.release_profile_home", lambda runner, home: released.append(home))
    monkeypatch.setattr("gateway.run_runtime.serve_profile_runtime", serve)
    verdict = asyncio.run(Runner()._serve_profile_runtime("work", Path(tmp_path)))
    assert verdict and "HTTP 503" in verdict and released == [Path(tmp_path)]


def test_compute_host_turn_answers_turn_error(monkeypatch):
    """The dashboard compute host's turn worker answers ``turn.error`` and keeps serving."""
    import io
    import json as _json
    from tui_gateway.compute_host import ComputeHost
    out = io.StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    monkeypatch.setattr(host, "_ensure_server_session", lambda server, frame: _outage())
    try:
        host._run_real_turn({"sid": "s1", "request_id": "q1", "text": "hi"})
    finally:
        host.close()
    frames = [_json.loads(line) for line in out.getvalue().splitlines()]
    assert [f["type"] for f in frames] == ["turn.error"]
    assert frames[0]["request_id"] == "q1" and "HTTP 503" in frames[0]["message"]
