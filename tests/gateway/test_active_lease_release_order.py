"""A turn end's queued lease release is already in effect for the next claim in this process.

The durable registry drop runs on a worker (gateway/active_lease_io.py), but the finished turn's
slot must be free the moment the turn ends: the next session's claim at max_concurrent_sessions,
and a Desktop writer taking over the same session, run on the loop straight after. The worker is
held busy with an earlier release (registry contention) so its drop provably has not landed.
"""

import threading

import gateway.run as run_mod
from gateway.active_lease_io import release_soon
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_cli import active_sessions


class _ReleaseStuckOnTheRegistry:
    """An earlier queued release that holds the single release worker until told to finish."""

    enabled = False
    released = False

    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()

    def release(self):
        self.started.set()
        self.finish.wait(10)


def _runner(cfg):
    runner = run_mod.GatewayRunner.__new__(run_mod.GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner.config = cfg
    return runner


def _end_gateway_turn(runner, session_key, source):
    lease, refusal = runner._claim_active_session_slot(session_key, source)
    assert refusal is None and lease is not None, refusal
    runner._session_state(session_key).turn.lease = lease
    runner._release_running_agent_state(session_key)
    return lease


def _registry_session_ids(home):
    return sorted(e["session_id"] for e in active_sessions._read_entries(
        home / "runtime" / "active_sessions.json", strict=True))


def _drain(lease, stuck):
    stuck.finish.set()
    done = threading.Event()
    release_soon(type("Marker", (), {"enabled": False, "released": False, "release": lambda self: done.set()})())
    assert done.wait(10), "the release worker never drained"
    assert lease.released


def test_the_next_sessions_claim_at_the_cap_sees_the_ended_turns_slot_free(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _runner({"max_concurrent_sessions": 1})
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
    stuck = _ReleaseStuckOnTheRegistry()
    release_soon(stuck)
    assert stuck.started.wait(10)

    lease_a = _end_gateway_turn(runner, "agent:main:telegram:dm:a", src)
    lease_b, refusal = runner._claim_active_session_slot("agent:main:telegram:dm:b", src)
    assert refusal is None, f"the next session was refused with nothing running: {refusal}"

    _drain(lease_a, stuck)
    assert _registry_session_ids(tmp_path) == ["agent:main:telegram:dm:b"]
    lease_b.release()


def test_a_desktop_claim_of_the_session_a_gateway_turn_just_ended_is_not_already_owned(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tui_gateway import server

    runner = _runner({})
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
    key = "agent:main:telegram:dm:shared"
    stuck = _ReleaseStuckOnTheRegistry()
    release_soon(stuck)
    assert stuck.started.wait(10)

    lease_gw = _end_gateway_turn(runner, key, src)
    lease_desk, refusal = server._claim_active_session_slot(key, live_session_id="desktop-view", surface="desktop", profile_home=tmp_path)
    assert refusal is None, f"Desktop saw the ended gateway turn as the session's owner: {refusal}"

    _drain(lease_gw, stuck)
    entries = active_sessions._read_entries(tmp_path / "runtime" / "active_sessions.json", strict=True)
    assert [(e["session_id"], e["surface"]) for e in entries] == [(key, "desktop")]
    lease_desk.release()
