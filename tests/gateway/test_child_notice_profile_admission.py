"""Child-process notices across profiles: every consumer follows the owning profile's policy and live ownership.

A delegate child's process runs under profile A, then B, then A again; each leg checks one consumer against the
owning profile's settings, a handoff to the parent, and the stale child pin.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner

OWNERS = ["child", "handoff", "foreground", "live_handoff"]


def _build_rig(monkeypatch, tmp_path, request, surface_a, mode):
    from typing import cast

    from gateway.platforms.base import BasePlatformAdapter
    from gateway.run import _async_profile_runtime_scope
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    import hermes_state
    from tools import process_registry as pr_module
    from tools.process_registry import ProcessRegistry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BASE_HOME", str(tmp_path))
    # The global fixture pins the default DB; restore call-time profile resolution
    # within these explicitly private homes instead of sharing its root handle.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    import gateway.run as gw
    monkeypatch.setattr(gw, "_hermes_home", tmp_path)
    registry = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=tmp_path / "sessions"))
    # Match shutdown ordering: async wrappers can borrow the store's profile handles.
    request.addfinalizer(runner.close_all_session_db_handles)
    request.addfinalizer(runner.session_store.close_all_db_handles)
    runner._completion_notification_batch_window = 0
    admitted = []
    homes = {}
    for name, surface in (("a", surface_a), ("b", not surface_a)):
        home = get_profile_dir(name)
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"delegation:\n  surface_child_process_notifications: {str(surface).lower()}\n"
            f"display:\n  background_process_notifications: {mode}\n",
        )
        homes[name] = home

    async def accept(event):
        async with _async_profile_runtime_scope(homes[event.source.profile]):
            key = event.metadata["gateway_session_key"]
            entry = runner.session_store.lookup_by_session_key(key)
            assert entry is not None
            pinned = event.metadata.get("gateway_session_id")
            if pinned:
                entry = await runner._resolve_async_delegation_session(entry, pinned)
            assert entry is not None
            admitted.append((event, entry.session_id, get_hermes_home()))
        event._gateway_accepted = True

    adapter = SimpleNamespace(send=AsyncMock(), handle_message=accept)
    transport = cast(BasePlatformAdapter, adapter)
    runner.adapters[Platform.TELEGRAM] = transport
    runner._profile_adapters = {name: {Platform.TELEGRAM: transport} for name in homes}
    runner._profile_configs = {name: GatewayConfig() for name in homes}
    return SimpleNamespace(runner=runner, registry=registry, adapter=adapter, admitted=admitted, homes=homes)


def _transfer_to_parent(rig, leg):
    assert rig.registry.transfer_ownership(
        leg.proc.id, from_owner="sa-child", to_owner="parent-owner", to_task_id="parent-owner",
        to_session_key=leg.entry.session_key, note="finish build",
    ) is leg.proc


async def _for_each_leg(rig, owner, surface_a, path, consume):
    """Run ``consume`` on legs A, B, A; afterwards no route, row or pin may have moved."""
    from gateway.run import _async_profile_runtime_scope, _profile_runtime_scope
    from gateway.session import SessionSource
    from hermes_constants import get_hermes_home
    from tools.process_registry import ProcessSession

    runner, registry, homes = rig.runner, rig.registry, rig.homes
    handed_off = owner in {"handoff", "live_handoff"}
    for leg_index, name in enumerate(("a", "b", "a")):
        async with _async_profile_runtime_scope(homes[name]):
            source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", profile=name)
            entry = runner.session_store.get_or_create_session(source, force_new=True)
            db = runner.session_store._db_for_key(entry.session_key)
            assert db is not None
            assert db.db_path == homes[name] / "state.db"
            assert db.get_session(entry.session_id) is not None
            child = f"worker-{name}-{leg_index}"
            db.create_session(child, source="subagent", model_config={"_delegate_from": entry.session_id})
            before = {sid: db.get_session(sid) for sid in (entry.session_id, child)}
        proc = ProcessSession(
            id=f"proc_{name}{leg_index}abcd", command="build",
            owner_task_id="parent-owner" if owner == "foreground" else "sa-child",
            task_id="container-parent", session_key=entry.session_key,
            parent_session_id=child, notify_on_complete=True, output_buffer="READY\n", started_at=1.0,
        )
        registry._running[proc.id] = proc
        watcher = {
            "session_id": proc.id, "check_interval": 0, "platform": "telegram", "chat_id": "123",
            "session_key": entry.session_key, "chat_type": "dm",
            "parent_session_id": child, "owner_task_id": "sa-child",
            "notify_on_complete": path == "agent",
        }
        async with runner._completion_event_scope(watcher):
            assert get_hermes_home() == homes[name], runner._primary_profile_name
            assert await runner._session_db.get_session(child) is not None
        leg = SimpleNamespace(
            name=name, index=leg_index, entry=entry, db=db, child=child, proc=proc, watcher=watcher,
            surfaced=handed_off or owner == "foreground" or (surface_a if name == "a" else not surface_a),
        )
        if owner == "handoff":
            _transfer_to_parent(rig, leg)
            assert registry.running_owned_by("sa-child") == []
            assert registry.kill_all("sa-child") == 0
        rig.adapter.send.reset_mock()
        rig.admitted.clear()

        await consume(leg)

        for event, resolved_id, home in rig.admitted:
            assert event.source.profile == name and home == homes[name]
            assert resolved_id == entry.session_id
            if path == "agent":
                assert event.metadata["gateway_session_id"] == entry.session_id
        completion = runner._build_process_completion_event(watcher, proc, proc.id)
        assert completion["owner_task_id"] == proc.owner_task_id
        assert completion["parent_session_id"] == child
        assert proc.notify_on_complete is True
        if not proc.exited:
            proc.mark_exited(2)
        registry._running.pop(proc.id)
        registry._finished[proc.id] = proc
        assert registry.unread_completions_owned_by(proc.owner_task_id) == [proc]
        registry._finished.pop(proc.id)
        with _profile_runtime_scope(homes[name]):
            current = runner.session_store.lookup_by_session_key(entry.session_key)
            assert current is not None
            assert current.session_id == entry.session_id
            for sid, row in before.items():
                after = db.get_session(sid)
                for field in ("ended_at", "end_reason", "source", "session_key", "model_config"):
                    assert after[field] == row[field], (path, name, sid, field)


def _consumer_event(rig, leg, owner, path):
    if owner == "live_handoff":
        _transfer_to_parent(rig, leg)
    return {
        **leg.watcher, "type": path, "owner_task_id": leg.proc.owner_task_id, "task_id": leg.proc.task_id,
        "pattern": "READY", "command": "build", "output": "READY\n",
        "elapsed_seconds": 10, "heartbeat_seq": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_a", [True, False])
@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("path,mode", [
    ("agent", "all"), ("agent", "off"), ("raw", "all"),
    ("raw", "result"), ("raw", "error"), ("raw", "concise"), ("raw", "off"),
])
async def test_process_watcher_follows_the_owning_profile(
    monkeypatch, tmp_path, request, surface_a, owner, path, mode, private_db_probe_cleanup,
):
    rig = _build_rig(monkeypatch, tmp_path, request, surface_a, mode)

    async def consume(leg):
        ticks = 0

        async def tick(*args, **kwargs):
            nonlocal ticks
            ticks += 1
            assert ticks <= 10, (path, mode, leg.name, leg.proc.exited, leg.proc.owner_task_id)
            # all-mode raw consumers see both running output and an exit.
            if owner == "live_handoff" and ticks == 2:
                _transfer_to_parent(rig, leg)
            if ticks >= 2 or (path == "agent" and owner != "live_handoff"):
                leg.proc.mark_exited(2)

        with monkeypatch.context() as clock:
            clock.setattr(asyncio, "sleep", tick)
            await rig.runner._run_process_watcher(leg.watcher)
        own_profile_surfaces = surface_a if leg.name == "a" else not surface_a
        expected_admissions = int(leg.surfaced and path == "agent")
        expected_sends = 0
        if leg.surfaced and path == "raw":
            if mode == "all":
                expected_sends = 1 if owner == "live_handoff" and not own_profile_surfaces else 2
            else:
                expected_sends = int(mode != "off")
        assert len(rig.admitted) == expected_admissions
        assert rig.adapter.send.await_count == expected_sends

    await _for_each_leg(rig, owner, surface_a, path, consume)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_a", [True, False])
@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("path", ["watch_match", "heartbeat"])
async def test_watch_events_follow_the_owning_profile(
    monkeypatch, tmp_path, request, surface_a, owner, path, private_db_probe_cleanup,
):
    rig = _build_rig(monkeypatch, tmp_path, request, surface_a, "all")

    async def consume(leg):
        evt = _consumer_event(rig, leg, owner, path)
        rig.registry.completion_queue.put(evt)
        await rig.runner._drain_watch_notifications(rig.registry.completion_queue)
        assert len(rig.admitted) == int(leg.surfaced)
        assert rig.registry.completion_queue.empty()

    await _for_each_leg(rig, owner, surface_a, path, consume)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_a", [True, False])
@pytest.mark.parametrize("owner", OWNERS)
async def test_registry_drain_follows_the_owning_profile(
    monkeypatch, tmp_path, request, surface_a, owner, private_db_probe_cleanup,
):
    from gateway.run import _profile_runtime_scope

    rig = _build_rig(monkeypatch, tmp_path, request, surface_a, "all")

    async def consume(leg):
        evt = _consumer_event(rig, leg, owner, "registry")
        evt["type"] = "completion"
        evt["exit_code"] = 0
        rig.registry.completion_queue.put(evt)
        with _profile_runtime_scope(rig.homes[leg.name]):
            drained = rig.registry.drain_notifications(session_key=leg.entry.session_key)
        assert len(drained) == int(leg.surfaced)

    await _for_each_leg(rig, owner, surface_a, "registry", consume)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_a", [True, False])
@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("path", ["async_delegation", "async_delegation_work_closeout"])
async def test_delegation_results_are_never_process_noise(
    monkeypatch, tmp_path, request, surface_a, owner, path, private_db_probe_cleanup,
):
    rig = _build_rig(monkeypatch, tmp_path, request, surface_a, "all")

    async def consume(leg):
        evt = _consumer_event(rig, leg, owner, path)
        evt.pop("parent_session_id")
        async with rig.runner._completion_event_scope(evt):
            assert await rig.runner._deliver_completion_notification_scoped("result", evt) is True
        assert len(rig.admitted) == 1

    await _for_each_leg(rig, owner, surface_a, path, consume)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface_a", [True, False])
@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("path", [
    "batch_child_first", "batch_foreground_first", "batch_boundary_first", "batch_boundary_last",
])
async def test_batched_completions_follow_the_owning_profile(
    monkeypatch, tmp_path, request, surface_a, owner, path, private_db_probe_cleanup,
):
    from gateway.run import _async_profile_runtime_scope

    rig = _build_rig(monkeypatch, tmp_path, request, surface_a, "all")
    runner = rig.runner

    async def consume(leg):
        runner._completion_notification_batch_window = 0.01
        if owner == "live_handoff":
            _transfer_to_parent(rig, leg)
        leg.proc.output_buffer = "PROCESS_RESULT"
        evt = runner._build_process_completion_event(leg.watcher, leg.proc, leg.proc.id)
        companion = {**evt, "session_id": f"companion-{leg.name}-{leg.index}",
                     "owner_task_id": "parent-owner", "output": "COMPANION_RESULT"}
        boundary = path in {"batch_boundary_first", "batch_boundary_last"}
        if boundary:
            async with _async_profile_runtime_scope(rig.homes[leg.name]):
                stale = f"stale-{leg.name}-{leg.index}"
                leg.db.create_session(stale, source="telegram")
                leg.db.end_session(stale, end_reason="session_reset")
            companion["parent_session_id"] = stale
        pair = [("PROCESS_RESULT", evt), ("COMPANION_RESULT", companion)]
        if path in {"batch_foreground_first", "batch_boundary_first"}:
            pair.reverse()
        outcomes = await asyncio.gather(*(
            runner._enqueue_process_completion_notification(text, event)
            for text, event in pair
        ))
        assert outcomes[pair.index(("PROCESS_RESULT", evt))] is (True if leg.surfaced else None)
        assert outcomes[pair.index(("COMPANION_RESULT", companion))] is (None if boundary else True)
        texts = "\n".join(event.text for event, _sid, _home in rig.admitted)
        assert ("PROCESS_RESULT" in texts) is leg.surfaced
        assert ("COMPANION_RESULT" in texts) is (not boundary)
        assert len(rig.admitted) == (int(leg.surfaced) if boundary else 1)

    await _for_each_leg(rig, owner, surface_a, path, consume)
