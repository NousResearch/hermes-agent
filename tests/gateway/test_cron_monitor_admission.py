"""Regression for #140222: unchanged monitor ticks must not create chat sessions."""
import asyncio
from types import SimpleNamespace


def test_unchanged_monitor_fire_does_not_create_session(tmp_path, monkeypatch):
    _probe_unchanged_monitor(tmp_path, monkeypatch, require_legacy_receipt=False)


def test_legacy_submit_retains_receipt_contract(tmp_path, monkeypatch):
    _probe_unchanged_monitor(tmp_path, monkeypatch, require_legacy_receipt=True)


def _probe_unchanged_monitor(tmp_path, monkeypatch, *, require_legacy_receipt):
    from cron import jobs, monitor
    from cron.scheduler import SILENT_MARKER
    from gateway import run, session_cron
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {
        "id": "monitor-job", "prompt": "Report changes", "model": "fixture-model",
        "monitor_url": "https://example.invalid/status",
        "monitor_state": {"last_output_hash": monitor.hash_monitor_output("stable")},
    }
    samples = []

    def sample(_job):
        samples.append(_job["id"])
        return True, "stable"

    monkeypatch.setattr(monitor, "_run_monitor_source", sample)
    monkeypatch.setattr(jobs, "get_job", lambda jid: dict(job) if jid == job["id"] else None)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {}, "platform_toolsets": {"cli": []}})
    db = SessionDB(tmp_path / "state.db")
    runner = SimpleNamespace(
        _draining=False, adapters={},
        session_store=SessionStore(tmp_path / "sessions", GatewayConfig()),
    )
    authority = SessionAuthority(
        runner, profile_id=str(tmp_path), instance_id="test", db=db,
        epoch=begin_runtime_epoch(db, instance_id="test"),
    )
    runner.session_authority = authority
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    async def probe():
        from gateway.session_controls import AuthorityConnection
        connection = AuthorityConnection(authority, object(), {'user_id': 'tester'})
        async def call(name, params):
            response = await connection.dispatch({'id': 1, 'method': 'cron.' + name, 'params': params})
            assert 'error' not in response, response
            return response['result']
        for fire in ("first-unchanged", "second-unchanged"):
            params = {
                "job_id": job["id"], "request_id": fire, "extra_prompt": None,
            }
            receipt = await call('submit' if require_legacy_receipt else 'submit_v2', params)
            if not require_legacy_receipt:
                assert receipt['status'] == 'preparing'
                from gateway.session_runtime_workers import mutation_tasks
                await asyncio.gather(*mutation_tasks(authority))
                receipt = await call('recover_v2', params)
            # Existing schedulers require this receipt even when no conversation is
            # allocated. Exercise the declared wire shape, not only the owner helper.
            from tui_gateway.contracts.canonical import CronSubmitResult
            if require_legacy_receipt:
                CronSubmitResult.model_validate(receipt)
            if "session_id" not in receipt:
                assert receipt["status"] == "terminal"
                assert receipt["result"][2] == SILENT_MARKER
                assert await call('submit_v2', params) == receipt
                assert await call('recover_v2', params) == receipt
                continue
            sid = receipt["session_id"]
            if db.get_session(sid) is not None:
                row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
                policy = runner.adapters[Platform.LOCAL].policies[sid]
                assert await session_cron.execute(authority, SessionRef(str(tmp_path), sid), row, policy) == SILENT_MARKER
                saved = authority.pending_results.pop(row["admission_id"])
                settle_session_input(
                    db, epoch=authority.epoch, admission_id=row["admission_id"],
                    generation=row["generation"], outcome="completed", result=saved,
                )
            state = await session_cron.operation(authority, "status", receipt)
            assert state["status"] == "terminal"
            assert state["result"][2] == SILENT_MARKER
            recovered = await session_cron.operation(authority, "recover", {
                "job_id": job["id"], "request_id": fire, "extra_prompt": None,
            })
            assert recovered["session_id"] == sid
            assert recovered["admission_id"] == receipt["admission_id"]
        assert samples == [job["id"], job["id"]]
        if not require_legacy_receipt:
            assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    try:
        asyncio.run(probe())
    finally:
        db.close()


def test_prepared_monitor_without_admission_is_not_missing(tmp_path, monkeypatch):
    """A sampled change survives a lost handoff as unknown, never a safe non-admission."""
    from cron import jobs, monitor
    from gateway import session_cron, session_cron_monitor
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "changed-job", "prompt": "Report changes",
           "monitor_url": "https://example.invalid/status"}
    monkeypatch.setattr(jobs, "get_job", lambda jid: dict(job))
    calls = []

    def sample(_job):
        calls.append(_job["id"])
        return True, "new observation"

    monkeypatch.setattr(monitor, "_run_monitor_source", sample)
    monkeypatch.setattr(monitor, "_persist_monitor_state", lambda *args: None)
    db = SessionDB(tmp_path / "state.db")
    runner = SimpleNamespace(_draining=False)
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="test"))
    runner.session_authority = authority
    params = {"job_id": job["id"], "request_id": "changed-fire", "extra_prompt": None}

    async def probe():
        actor = session_cron._actor(authority, None)
        prepared = await session_cron_monitor.prepare(authority, actor, params)
        assert prepared["status"] == "prepared"
        # Simulate the caller vanishing between source observation and admission.
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        recovered = await session_cron.operation(authority, "recover", params)
        assert recovered["status"] == "unknown"
        assert calls == [job["id"]]

    try:
        asyncio.run(probe())
    finally:
        db.close()
