import json
import sys
import threading
from types import SimpleNamespace

import pytest

from tools.delegate_tool_deadline import ReviewedDeadline
from tools import delegate_tool_registry as registry
from plugins.subagent_progress.supervision import Supervision
from test_progress import setup, report


@pytest.fixture
def supervised(setup, monkeypatch):
    plugin, ctx, parent, child = setup
    clock = SimpleNamespace(now=100.0)
    child._delegate_reviewed_deadline = ReviewedDeadline(600, clock=lambda: clock.now)
    child.get_activity_summary = lambda: {"current_tool": "terminal", "api_call_count": 2, "last_activity_ts": clock.now}
    child.steer = lambda text: True
    monkeypatch.setitem(registry._active_subagents, child._subagent_id, {
        "agent": child, "owner_agent_session_id": parent.session_id, "accepting_steer": True})
    supervisor = Supervision(plugin)
    return plugin, ctx, parent, child, supervisor, clock


def review(plugin, supervisor, parent, checkpoint, decision="approve"):
    plugin.current_child = lambda: parent
    return json.loads(supervisor.review({"checkpoint_id": checkpoint, "decision": decision,
        "reason": "Source evidence checked against commissioned goal", "evidence_checked": ["evidence.json"]}))


def test_report_requests_parent_review_but_does_not_renew(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    clock.now = 650
    checkpoint = report(plugin)["checkpoint_id"]
    assert len(ctx.wakes) == 1 and "SUBAGENT CHECKPOINT ID:" in ctx.wakes[0][0]
    assert child._delegate_reviewed_deadline.remaining() == 50
    result = review(plugin, sup, parent, checkpoint)
    assert result['success'] and result['remaining_seconds'] == 600
    assert not review(plugin, sup, parent, checkpoint)['success']


def test_child_and_unrelated_parent_cannot_approve(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    assert not review(plugin, sup, child, checkpoint)['success']
    assert not review(plugin, sup, SimpleNamespace(session_id="other"), checkpoint)['success']
    assert child._delegate_reviewed_deadline.snapshot()['renewals'] == 0


def test_steer_denies_renewal_requires_a_new_report(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    clock.now = 650
    result = review(plugin, sup, parent, checkpoint, 'steer')
    assert result['success'] and result['control_requested'] and result['remaining_seconds'] == 50
    assert not review(plugin, sup, parent, checkpoint)['success']
    clock.now = 700
    assert child._delegate_reviewed_deadline.remaining() == 0


def test_independent_timer_detects_silence_without_renewing(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    from agent.periodic_scheduler import schedule
    clock.now = 450  # no report at all; model activity timestamp still advances
    event = threading.Event()
    original = plugin.notify
    def noticed(*args):
        result = original(*args)
        event.set()
        return result
    plugin.notify = noticed
    handle = schedule(sup.tick, 0.02)
    try:
        assert event.wait(2)
    finally:
        handle.cancel(wait=2)
    assert child._delegate_reviewed_deadline.remaining() == 250
    sup.tick()
    assert len(ctx.wakes) == 1
    with plugin.db() as db:
        assert db.execute('SELECT COUNT(*) FROM supervision_checks').fetchone()[0] == 1
        alert = db.execute('SELECT id FROM reports').fetchone()[0]
    assert not review(plugin, sup, parent, alert)['success']


def test_stop_preserves_real_checkpoint_not_supervisor_notice(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    clock.now = 450
    sup.tick()
    plugin.stop(child_session_id=child.session_id, child_subagent_id=child._subagent_id, child_status='timeout')
    with plugin.db() as db:
        payload = json.loads(db.execute('SELECT payload FROM reports ORDER BY id DESC LIMIT 1').fetchone()[0])
    assert payload['kind'] == 'terminal_checkpoint'
    assert payload['completed'] == 'Input checked' and payload['status'] == 'timeout'
    assert not payload['review_required']


def test_timeout_while_parent_ignores_report_rejects_late_review(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)["checkpoint_id"]
    clock.now = 700
    assert not review(plugin, sup, parent, checkpoint)['success']
    with plugin.db() as db:
        assert db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0] == 0
