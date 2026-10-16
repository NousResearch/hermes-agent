"""Which turns of a local session skip the whole-script ``execute_code`` approval (T2).

Only a turn the session's viewer submitted (``session.submit``) follows the interactive-CLI rule. A
plugin ``inject_message`` admission, a /loop tick or any other producer admission into the same
session keeps the whole-script gate: a script's ``shutil``/``subprocess`` calls never pass through a
``terminal()`` prompt. The turns run through the real FIFO drain (``execute_admission``); the handler
checks the gate the way the turn's agent thread does.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_only_the_viewers_own_turn_skips_the_whole_script_gate(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_automation import admit_route_wake
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_contract import Principal, Submission
    from gateway.session_local import create_local_session
    from gateway.session_policy import _ACTIVE_POLICY, policy_for_source
    from hermes_state import SessionDB
    from tools import approval as A
    from tools import approval_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")  # start_gateway() sets it process-wide
    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner", db=db)
    gates = {}

    def gate_in_turn_thread(live):
        # What TurnRunner binds for the agent thread: the frozen policy, the session key, its notifier.
        prompts = []

        def nobody_approves(data):
            prompts.append(data["pattern_key"])
            with A._lock:
                entry = A._gateway_queues[live.route][-1]
                entry.result = "deny"
                entry.event.set()

        vars_tokens = set_session_vars(platform="local", session_key=live.route, session_id=live.route)
        key_token = approval_context.set_current_session_key(live.route)
        policy_token = _ACTIVE_POLICY.set(policy_for_source(runner, live.source))
        A.register_gateway_notify(live.route, nobody_approves)
        try:
            return A.check_execute_code_guard("import shutil\nshutil.rmtree('victim')", "local")["approved"], prompts
        finally:
            A.unregister_gateway_notify(live.route)
            _ACTIVE_POLICY.reset(policy_token)
            approval_context.reset_current_session_key(key_token)
            clear_session_vars(vars_tokens)
            with A._lock:
                A._gateway_queues.pop(live.route, None)
                A._pending.pop(live.route, None)

    async def answer(event):
        gates[event.text] = await asyncio.to_thread(gate_in_turn_thread, authority.sessions[ref.session_id])
        return "ok"

    monkeypatch.setattr(runner, "_handle_message", answer)
    try:
        viewer = Principal("test-owner", "default", frozenset(
            {"session:create", "session:submit", "session:read", "session:approve"}), "viewer")
        ref = create_local_session(authority, viewer, dict(request_id="cli", source="cli", model="fixture",
                                                          toolsets=[]))
        live = authority.sessions[ref.session_id]
        await authority.attach(viewer, ref)
        entry = runner.session_store.lookup_by_session_key(live.route)
        adapter = runner._adapter_for_source(live.source)

        await authority.submit(viewer, Submission("typed", ref, {"text": "typed by the viewer"}, "queue"))
        await async_wait_until(lambda: "typed by the viewer" in gates, timeout=10, what="the viewer's turn")
        assert await runner._admit_plugin_injection(
            adapter, live.source, session_key=live.route, session_id=entry.session_id,
            content="plugin injection", plugin_id="desk") is True
        await async_wait_until(lambda: "plugin injection" in gates, timeout=10, what="the injected turn")
        assert await admit_route_wake(runner, adapter, live.source, "loop tick", "loop-tick") is True
        await async_wait_until(lambda: "loop tick" in gates, timeout=10, what="the /loop tick")
    finally:
        db.close()

    assert gates == {"typed by the viewer": (True, []),
                     "plugin injection": (False, ["execute_code"]),
                     "loop tick": (False, ["execute_code"])}


@pytest.mark.parametrize("request_id,payload,viewer", [
    ("typed", {"text": "hi", "cli_turn_v1": {"api_text": "[voice] hi"}, "finite": True}, True),
    ("typed", {"text": "hi", "local_operator_v1": {"principal_id": "op"}}, True),
    ('["plugin","desk","route","x"]', {"text": "hi", "local_automation_v1": {"identity": "x"}}, False),
    ('hosted:[{"task":1},0]', {"text": "room task"}, False),
    ("native", {"text": "hi", "native_text_v1": {"source": {}}}, False),
], ids=["cli-submit", "operator-submit", "automation", "hosted-room-task", "messaging"])
def test_viewer_input_is_only_a_session_submit_row(request_id, payload, viewer):
    from gateway.session_operator import is_viewer_input
    assert is_viewer_input({"request_id": request_id, "payload": payload}) is viewer
