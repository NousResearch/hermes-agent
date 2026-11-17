"""The public work CLI records evidence and exposes an explicit delegation contract."""
import argparse
import json

import pytest

from hermes_cli.task_delivery.cli import configure, execute
from hermes_cli.task_delivery.store import DeliveryError
from hermes_state import SessionDB


def call(db, *arguments):
    return execute(configure(argparse.ArgumentParser()).parse_args(arguments), db)


def test_cli_rejects_unverified_completion_and_preserves_delivery_across_reopen(tmp_path):
    database = tmp_path / 'state.db'
    with SessionDB(db_path=database) as db:
        db.create_session('parent', source='gui')
        call(db, 'init', '--session', 'parent', '--task', 'release', '--title', 'Fix request',
             '--acceptance', 'A real customer request succeeds')
        context = call(db, 'context', '--session', 'parent', '--task', 'release')
        assert context.startswith('[HERMES_WORK_ITEM_V1]\n')
        assert json.loads(context.splitlines()[1])['session_id'] == 'parent'
        claim = call(db, 'claim', '--session', 'parent', '--task', 'release', '--owner', 'worker',
                     '--request-id', 'attempt')
        artifact = tmp_path / 'candidate.patch'
        artifact.write_text('patch contents')
        item = call(db, 'deliver', '--session', 'parent', '--task', 'release', '--owner', 'worker',
                    '--generation', str(claim['generation']), '--file', str(artifact), '--summary', 'Implemented')
        digest = item['candidate']['digest']
        review = tmp_path / 'review.json'
        review.write_text(json.dumps({'request_id': 'review', 'reviewer': 'reviewer', 'candidate_digest': digest,
                                      'passed': True, 'findings': []}))
        call(db, 'review', '--session', 'parent', '--task', 'release', '--file', str(review))
        evidence = tmp_path / 'evidence.json'
        evidence.write_text(json.dumps({'candidate_digest': digest, 'passed': False, 'scenario': 'customer flow'}))
        with pytest.raises(DeliveryError, match='Failed or unknown'):
            call(db, 'advance', '--session', 'parent', '--task', 'release', '--stage', 'acceptance',
                 '--candidate', digest, '--evidence-file', str(evidence))
        for stage, report in [('integration', {'revision': 'tested-revision'}),
                              ('verification', {'passed': True, 'command': 'test-service', 'exit_code': 0}),
                              ('acceptance', {'passed': True, 'scenario': 'real request'})]:
            evidence.write_text(json.dumps({'candidate_digest': digest, **report}))
            call(db, 'advance', '--session', 'parent', '--task', 'release', '--stage', stage,
                 '--candidate', digest, '--evidence-file', str(evidence))
    with SessionDB(db_path=database) as resumed:
        state = call(resumed, 'show', '--session', 'parent', '--task', 'release')
        assert state['acceptance']['candidate_digest'] == digest
        assert state['review']['status'] == 'passed'


def test_reports_accept_utf8_bom_from_windows_editors(tmp_path):
    from hermes_cli.task_delivery.cli import _read_json
    report = tmp_path / 'report.json'
    report.write_text('{"passed": true}', encoding='utf-8-sig')
    assert _read_json(report) == {'passed': True}
