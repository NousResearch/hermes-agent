"""Desktop-local /loop wakeups behind the default-profile multiplexer (#144966).

A /loop armed on a desktop-local session of a served secondary profile rides a ``local``
route. On the multiplexer this path was silent when it failed (the CLI kept reporting
"active, due" while nothing was ever delivered): the served profile has no LOCAL adapter
until one of its sessions restores (one DEBUG line), a route without ``chat_type``/``user_id``
never resolves a synthetic source (no log at all), scan exceptions logged at DEBUG, and an
authority refusal fell back to an adapter with no message handler (one ERROR, then silence —
the claimed tick sat "wakeup running" forever). Real GatewayRunner, real per-profile
authorities, SessionStore and LocalSessionAdapter: the working delivery as the positive
control, and every quiet failure pinned visible.
"""
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.waiting import async_wait_until

# run_goals.py keeps the origin module's logger (see the "parity" comment there).
_GOALS_LOGGER = "gateway.run"


def _reserve_homes(tmp_path, monkeypatch, names=("lead",)):
    """Bare multiplex homes; the canonical rig of test_secondary_local_pick_multiplex.py."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [("default", root)]
    for name in names:
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}")
        homes.append((name, home))
    return root, homes


@pytest_asyncio.fixture
async def lead_mux(tmp_path, monkeypatch):
    """A real multiplexing runner: default + served ``lead`` profile, each with its authority.

    LOCAL adapter maps stay empty at boot (they fill on a first local session); the tests place
    one through ``local_adapter`` or leave the profile bare.
    """
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from hermes_cli import model_switch

    monkeypatch.setattr(run, "_load_gateway_config", lambda *a: {
        "platform_toolsets": {"cli": [], "tui": []},
        "model": {"provider": "custom", "base_url": "http://127.0.0.1:9/v1", "api_key": "loopback-only"}})
    monkeypatch.setattr(model_switch, "switch_model", lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k["raw_input"], target_provider="custom",
        base_url="http://127.0.0.1:9/v1"))
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        yield SimpleNamespace(root=root, runner=runner, lead=registry.for_home(homes[1][1]),
                              lead_home=homes[1][1], homes=homes)
    finally:
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


def _loop_stub(monkeypatch, fired, abandoned=None):
    """Stand-in ``hermes_cli.loops.LoopManager``: ``is_due`` always True, ``fire_tick`` claims and
    yields its prompt, ``abandon_tick`` records (or refuses, when admission is expected)."""

    class Loop:
        state = SimpleNamespace(ticks_fired=1)

        def __init__(self, session_id=None):
            pass

        def is_due(self, now):
            return True

        def fire_tick(self):
            fired.append(True)
            return "loop tick prompt"

        def abandon_tick(self):
            if abandoned is None:
                raise AssertionError("the tick was not admitted")
            abandoned.append(True)
            return True

    monkeypatch.setattr(__import__("hermes_cli.loops", fromlist=["LoopManager"]), "LoopManager", Loop)
    monkeypatch.setattr(
        __import__("hermes_cli.loops", fromlist=["goal_blocks_loop_tick"]), "goal_blocks_loop_tick",
        lambda sid: False)
    return Loop


@pytest.mark.asyncio
async def test_desktop_local_loop_tick_runs_as_a_fifo_turn(lead_mux, monkeypatch):
    """Positive control, the local mirror of test_idle_loop_tick_runs_as_a_fifo_turn_not_beside_it:
    a due tick of a desktop-local session on a served profile reaches the session authority and
    runs as its own FIFO admission turn. (Passes before and after the logging fix; it guards the
    path the fix must not degrade.)"""
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_authorities import owner_scope
    from gateway.session_ingress import executing_admission
    from gateway.session_local import create_local_session
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions

    rig, runner = lead_mux, lead_mux.runner
    connection = None
    try:
        from gateway.session_controls import AuthorityConnection
        connection = AuthorityConnection(rig.lead, object(), {"user_id": "human"})
        with owner_scope(rig.lead):
            ref = create_local_session(rig.lead, connection.actor, dict(
                request_id="loop-seed", source="cli", cwd=str(rig.root),
                model="primary-model", toolsets=[]))
        sid = ref.session_id
        # The created session is live in the registry with its route bound to the LOCAL adapter map.
        assert rig.lead.sessions[sid].route == f"agent:lead:local:dm:{sid}"
        from gateway.session_local import LocalSessionAdapter
        adapter = next(iter(rig.runner._profile_adapters["lead"].values()))
        assert isinstance(adapter, LocalSessionAdapter)

        started = []

        async def turn(event, source, quick_key, run_generation):
            started.append((event.text, executing_admission.get()))
            captured = execution_result.get()  # None outside the drain
            if captured is not None:
                captured["result"] = {"final_response": "ok", "messages": []}
            return "ok"
        monkeypatch.setattr(runner, "_handle_message_with_agent", turn)

        _loop_stub(monkeypatch, [], None)
        # The route a real /loop captures from the session's own source (slash_commands_goals).
        route = {"platform": "local", "chat_id": sid, "chat_type": "dm", "user_id": "human",
                 "profile": "lead"}
        state = SimpleNamespace(awaiting_response=False, next_due_at=0.0, route=route)
        async with _async_profile_runtime_scope(rig.lead_home):
            await runner._loop_wakeup_fire_one(sid, state, 1e12, set(), "lead")

        await async_wait_until(lambda: len(started) >= 1, what="the local loop tick turn to start")
        assert started == [("loop tick prompt", True)], started

        def rows():
            return list_session_admissions(rig.lead.db, session_id=sid, pending_only=False)
        await async_wait_until(lambda: [r["status"] for r in rows()] == ["terminal"],
                               what="the admitted tick to settle")
        assert [(r["principal_id"].split(":")[0], r["outcome"]) for r in rows()] == [
            ("automation", "completed")]
    finally:
        if connection is not None:
            await connection.close()


@pytest.mark.asyncio
async def test_probe_route_without_chat_type_stays_due_and_reports(lead_mux, monkeypatch, caplog):
    """The programmatic route from the issue — ``{'platform': 'local', 'chat_id': sid,
    'profile': 'lead'}`` with no chat_type/user_id — cannot resolve a synthetic source: the fire
    used to return with no run_goals log at all. It must report INFO once per session naming the
    missing route fields, never crash, and leave the tick due (unclaimed)."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_local_recovery import local_adapter
    from hermes_state_runtime import list_session_admissions

    rig, runner = lead_mux, lead_mux.runner
    local_adapter(rig.lead)  # the LOCAL adapter exists: an unresolvable source is the failing step
    fired = []
    _loop_stub(monkeypatch, fired)
    route = {"platform": "local", "chat_id": "ghost-session", "profile": "lead"}
    warned: set = set()
    with caplog.at_level(logging.INFO):
        async with _async_profile_runtime_scope(rig.lead_home):
            for _ in range(2):
                await runner._loop_wakeup_fire_one(
                    "ghost-session", SimpleNamespace(awaiting_response=False, next_due_at=0.0,
                                                     route=route), 1e12, warned, "lead")
    records = [r for r in caplog.records
               if r.name == _GOALS_LOGGER and "no messaging source" in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "chat_type" in message and "user_id" in message, message
    assert "ghost-session" in message, message
    assert fired == [], "the tick was claimed although its source never resolved"
    assert list_session_admissions(rig.lead.db, session_id="ghost-session", pending_only=False) == []


@pytest.mark.asyncio
async def test_served_profile_without_local_adapter_reports_at_info(lead_mux, monkeypatch, caplog):
    """A served secondary's LOCAL session is unreachable while no LOCAL adapter exists in its map
    (nothing restored it yet): the watcher used to note this once per session at DEBUG, invisible
    in agent.log. It must report INFO with the platform, the session and the profile."""
    from gateway.run import _async_profile_runtime_scope

    rig, runner = lead_mux, lead_mux.runner
    runner._profile_adapters["lead"] = {}
    fired = []
    _loop_stub(monkeypatch, fired)
    route = {"platform": "local", "chat_id": "ghost-session", "chat_type": "dm",
             "user_id": "u1", "profile": "lead"}
    warned: set = set()
    with caplog.at_level(logging.INFO):
        async with _async_profile_runtime_scope(rig.lead_home):
            for _ in range(2):
                await runner._loop_wakeup_fire_one(
                    "ghost-session", SimpleNamespace(awaiting_response=False, next_due_at=0.0,
                                                     route=route), 1e12, warned, "lead")
    records = [r for r in caplog.records
               if r.name == _GOALS_LOGGER and "no adapter for platform" in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "'local'" in message and "lead" in message, message
    assert fired == [], "the tick was claimed although no adapter could deliver it"


@pytest.mark.asyncio
async def test_authority_refusal_on_an_adapter_without_message_handler_warns(lead_mux, monkeypatch,
                                                                            caplog):
    """A ghost local session (a closed Desktop conversation with no re-openable receipt) makes the
    authority refuse the wake, and the fire fell back to ``handle_message`` on a LOCAL adapter —
    which has no message handler and silently dropped the claim one ERROR at a time, leaving the
    loop "wakeup running" forever. The fire must refuse with a visible WARNING and release the
    claim so the tick stays due."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.session_results import execution_result
    from gateway.session_local_recovery import local_adapter
    from hermes_state_runtime import list_session_admissions

    rig, runner = lead_mux, lead_mux.runner
    adapter = local_adapter(rig.lead)
    assert adapter._message_handler is None  # a LOCAL adapter's delivery is the authority's
    started = []

    async def turn(event, source, quick_key, run_generation):
        started.append(event.text)
        captured = execution_result.get()  # None outside the drain
        if captured is not None:
            captured["result"] = {"final_response": "ok", "messages": []}
        return "ok"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    fired, abandoned = [], []
    _loop_stub(monkeypatch, fired, abandoned)
    route = {"platform": "local", "chat_id": "ghost-session", "chat_type": "dm",
             "user_id": "u1", "profile": "lead"}
    warned: set = set()
    with caplog.at_level(logging.INFO):
        async with _async_profile_runtime_scope(rig.lead_home):
            for _ in range(2):
                await runner._loop_wakeup_fire_one(
                    "ghost-session", SimpleNamespace(awaiting_response=False, next_due_at=0.0,
                                                     route=route), 1e12, warned, "lead")
    records = [r for r in caplog.records
               if r.name == _GOALS_LOGGER and "no message handler" in r.getMessage()]
    assert len(records) == 1, [r.getMessage() for r in caplog.records]
    assert records[0].levelno == logging.WARNING, records
    assert "ghost-session" in records[0].getMessage()
    assert fired == [True, True], "an unadmittable tick must not stay claimed"
    assert abandoned == [True, True], "the refused fallback must release the tick's claim"
    assert started == [], "the synthetic event was injected into a handler-less adapter"
    assert list_session_admissions(rig.lead.db, session_id="ghost-session", pending_only=False) == []


def test_loop_scan_error_report_is_rate_limited_but_never_silent(monkeypatch, caplog):
    """A store that keeps failing its scan must surface at WARNING once per scope per day and
    stay DEBUG-quiet (with the traceback) in between — never invisible like before (#144966)."""
    from gateway import run_goals
    monkeypatch.setattr(run_goals, "_loop_scan_warned_at", {})
    with caplog.at_level(logging.INFO):
        assert run_goals._warn_loop_scan_error("lead", RuntimeError("boom"))
        assert not run_goals._warn_loop_scan_error("lead", RuntimeError("boom"))
        monkeypatch.setattr(run_goals, "_loop_scan_warned_at", {"lead": time.time() - 86401.0})
        assert run_goals._warn_loop_scan_error("lead", RuntimeError("boom again"))
    warns = [r for r in caplog.records
             if r.name == run_goals.logger.name and r.levelno == logging.WARNING]
    assert len(warns) == 2, [r.getMessage() for r in caplog.records]
    assert "lead" in warns[0].getMessage() and "boom" in warns[0].getMessage()
