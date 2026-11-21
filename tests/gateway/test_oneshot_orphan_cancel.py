"""Regression tests for gateway-side cancellation when a source='oneshot' viewer disconnects."""

import asyncio
from types import MethodType, SimpleNamespace

from gateway import run
from tests.waiting import async_wait_until


_CONFIG = "platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n"


async def _owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run_shutdown_session_end import GatewaySessionEndMixin
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    (tmp_path / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    calls = []

    async def finalize(**kwargs):
        calls.append(("finalize", kwargs))
        return []

    async def housekeeping(func, *args):
        return func(*args)

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kwargs: calls.append((name, kwargs)))
    runner = SimpleNamespace(
        session_store=store,
        _session_db=store._db,
        adapters={},
        _draining=False,
        _cached_agent_for=lambda route: None,
        _evict_cached_agent_at_boundary=lambda key: calls.append(("commit", key)),
        _finalize_session_off_loop=finalize,
        _FINALIZE_TIMEOUT_S=10.0,
        _run_housekeeping_in_executor=housekeeping,
        _run_release_in_profile_scope=lambda target, args, key: target(*args),
    )
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._reset_session_off_loop = MethodType(GatewaySessionEndMixin._reset_session_off_loop, runner)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id="owner")
    return store.persistence, authority, calls


def _viewer(authority, tmp_path):
    from gateway.session_controls import AuthorityConnection

    return AuthorityConnection(
        authority,
        SimpleNamespace(write=lambda frame: None),
        {
            "provider": "local",
            "user_id": "native",
            "profile_id": str(tmp_path),
            "instance_id": "owner",
            "native_bootstrap": True,
        },
    )


async def _call(conn, method, **params):
    reply = await conn.dispatch({"id": 1, "method": method, "params": params})
    assert "result" in reply, reply
    return reply["result"]


def test_oneshot_viewer_disconnect_immediately_cancels_running_and_queued_turns(tmp_path, monkeypatch):
    from gateway import session_finite

    async def scenario():
        persistence, authority, calls = await _owner(tmp_path, monkeypatch)
        viewer = _viewer(authority, tmp_path)
        created = await _call(viewer, "session.create", request_id="req-oneshot", source="oneshot", cwd=str(tmp_path))
        sid = created["stored_session_id"]

        started = asyncio.Event()
        interrupted = asyncio.Event()
        interrupt_calls = []

        fake_agent = SimpleNamespace()

        def fake_hard_interrupt(agent, *args, **kwargs):
            interrupt_calls.append((agent, args, kwargs))
            interrupted.set()
            return True

        monkeypatch.setattr("agent.interrupt_compat.request_hard_interrupt", fake_hard_interrupt)

        async def slow_execute(authority_arg, ref, row):
            with authority_arg.sessions[sid].event_stream.lock:
                authority_arg.adopted[sid] = (row["generation"], fake_agent)
            started.set()
            try:
                await asyncio.wait_for(interrupted.wait(), timeout=5)
                return {"final_response": "", "completed": False, "interrupted": True}
            finally:
                authority_arg.adopted.pop(sid, None)

        monkeypatch.setattr(session_finite, "execute_finite_admission", slow_execute)

        r1 = await _call(
            viewer, "prompt.submit", session_id=sid, input_id="in-1", text="run slow tool", finite=True, unattended=True
        )
        r2 = await _call(
            viewer, "prompt.submit", session_id=sid, input_id="in-2", text="queued follow-up", finite=True, unattended=True
        )
        await asyncio.wait_for(started.wait(), timeout=5)

        # Simulate timeout / SIGKILL dropping the WebSocket without sending prompt.cancel or end_reason.
        await viewer.close()

        await asyncio.wait_for(interrupted.wait(), timeout=5)
        assert len(interrupt_calls) == 1
        assert interrupt_calls[0][0] is fake_agent
        assert sid not in authority.orphan_closes

        # Queued turn was cancelled immediately without ever running.
        adm2 = persistence.admissions.get(admission_id=r2["admission_id"])
        assert (adm2.status, adm2.outcome) == ("terminal", "cancelled")

        # Once the interrupted turn unwinds, the oneshot session closes cleanly with oneshot_disconnect.
        await async_wait_until(lambda: persistence.sessions.get(sid).ended_at is not None, timeout=5)
        assert persistence.sessions.get(sid).end_reason == "oneshot_disconnect"
        adm1 = persistence.admissions.get(admission_id=r1["admission_id"])
        assert adm1.status == "terminal"
        assert [name for name, _ in calls] == ["commit", "finalize"]

    asyncio.run(scenario())


def test_interactive_cli_viewer_disconnect_keeps_running_turn_alive(tmp_path, monkeypatch):
    from gateway import session_finite

    async def scenario():
        _db, authority, _calls = await _owner(tmp_path, monkeypatch)
        viewer = _viewer(authority, tmp_path)
        created = await _call(viewer, "session.create", request_id="req-cli", source="cli", cwd=str(tmp_path))
        sid = created["stored_session_id"]

        started = asyncio.Event()
        release = asyncio.Event()
        interrupt_calls = []

        monkeypatch.setattr(
            "agent.interrupt_compat.request_hard_interrupt",
            lambda *args, **kwargs: interrupt_calls.append((args, kwargs)) or True,
        )

        async def slow_execute(authority_arg, ref, row):
            started.set()
            await release.wait()
            return {"final_response": "finished", "completed": True}

        monkeypatch.setattr(session_finite, "execute_finite_admission", slow_execute)
        await _call(viewer, "prompt.submit", session_id=sid, input_id="in-cli", text="interactive turn")
        await asyncio.wait_for(started.wait(), timeout=5)

        await viewer.close()
        assert interrupt_calls == []
        release.set()
        if sid in authority.orphan_closes:
            authority.orphan_closes.pop(sid).cancel()

    asyncio.run(scenario())
