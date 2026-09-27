import json
from types import SimpleNamespace

from test_supervision import supervised, review
from test_progress import setup, report
from tools.delegate_tool_registry import _active_subagents


def test_newer_checkpoint_supersedes_old_and_child_cannot_choose_window(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    first = report(plugin)['checkpoint_id']
    second = report(plugin, completed='Calculation checked')['checkpoint_id']
    assert not review(plugin, sup, parent, first)['success']
    assert not json.loads(sup.review({'checkpoint_id':second,'decision':'approve','reason':'checked',
        'evidence_checked':['proof'], 'timeout_seconds':99999}))['success']
    result = review(plugin, sup, parent, second)
    assert result['success'] and result['timeout_seconds']==600


def test_stop_does_not_renew_and_stops_actual_child(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)['checkpoint_id']
    seen = []
    child.interrupt = lambda *a, **kw: seen.append('stopped')
    clock.now = 650
    result = review(plugin, sup, parent, checkpoint,'stop')
    assert result['success'] and result['remaining_seconds']==50
    assert result['control_requested'] and seen
    assert child._delegate_reviewed_deadline.snapshot()['renewals']==0


def test_no_evidence_cannot_be_approved(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    checkpoint = report(plugin)['checkpoint_id']
    plugin.current_child = lambda: parent
    result = json.loads(sup.review({'checkpoint_id':checkpoint,'decision':'approve','reason':'looks okay'}))
    assert not result['success']
    assert child._delegate_reviewed_deadline.snapshot()['renewals']==0


def test_expired_report_rolls_back_and_preserves_prior_evidence(supervised):
    plugin, ctx, parent, child, sup, clock = supervised
    report(plugin)
    clock.now = 700
    assert not report(plugin,completed='too late')['success']
    with plugin.db() as db:
        rows=db.execute('SELECT payload FROM reports').fetchall()
    assert len(rows)==1 and json.loads(rows[0][0])['completed']=='Input checked'
