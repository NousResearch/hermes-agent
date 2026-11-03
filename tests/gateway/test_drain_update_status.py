"""A gateway draining for an update says so: reason, target and the work it is finishing (#142254)."""

from __future__ import annotations

from types import SimpleNamespace

from gateway.drain_control import drain_status, mark_update_drain, open_restart_wait, restart_wait_open
from hermes_cli.gateway_runtime import drain_cause, drain_detail


def _runner(units):
    return SimpleNamespace(session_runtime_descriptor={"state": "ready"}, _draining=False, _running=True,
                           _describe_active_work=lambda: units)


def test_an_update_drain_publishes_why_where_and_what_it_finishes_without_breaking_the_restart_wait():
    runner = _runner([{"kind": "chat"}, {"kind": "chat"}, {"kind": "cron"}])
    assert drain_status(runner) is None  # not draining: nothing to announce
    mark_update_drain(runner, "v0.23.1")
    runner._draining = True
    open_restart_wait(runner)  # what request_restart does next
    assert restart_wait_open(runner)  # clients still attach to answer a parked approval (#139041)
    status = drain_status(runner)
    assert status == {"reason": "update", "target": "v0.23.1", "waiting_for": {"chat": 2, "cron": 1}}
    assert drain_detail(status) == "updating to v0.23.1, finishing 2 turns, 1 cron run"


def test_peer_text_never_reaches_the_client_message():
    runner = _runner([])
    mark_update_drain(runner, "v1; rm -rf ~ \x1b[2J")
    runner._draining = True
    assert drain_status(runner)["target"] is None
    assert drain_detail({"reason": "update", "target": "\x1b[31mx", "waiting_for": {"chat": "9"}}) == "updating"
    assert drain_detail({"reason": "restart_wait"}) is None


def test_discovery_hands_clients_only_the_validated_update_cause():
    assert drain_cause({"reason": "update", "target": "v0.23.1",
                        "waiting_for": {"chat": 2, "Bad Kind": 3, "cron": "1", "delegation": 0}}) == {
        "reason": "update", "target": "v0.23.1", "waiting_for": {"chat": 2}}
    assert drain_cause({"reason": "update", "target": "\x1b[31mx"})["target"] is None
    assert drain_cause({"reason": "restart_wait"}) is None and drain_cause(None) is None
