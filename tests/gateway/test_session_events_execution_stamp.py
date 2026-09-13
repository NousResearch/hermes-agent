"""The execution stamp on published events marks exactly one claimed, unsettled execution."""
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[2] / 'apps' / 'desktop' / 'src' / 'lib' / 'execution-authority.frames.json'


def _projection(frames):
    return [(f['params']['type'], f['params'].get('authority_epoch'), f['params'].get('execution_generation'),
             f['params']['payload'].get('running'), f['params']['seq']) for f in frames]


@pytest.mark.asyncio
async def test_settled_execution_stops_stamping_idle_mutations_and_desktop_fixture_matches(tmp_path, monkeypatch):
    from tests.gateway.fixtures.execution_frames_capture import capture
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    captured = await capture(tmp_path)
    for label in ('epoch1', 'epoch2'):
        epoch = captured[label]['authority_epoch']
        frames = captured[label]['frames']
        assert frames[0]['params']['type'] == 'session.info' and 'authority_epoch' not in frames[0]['params'], \
            'queued pending was stamped before any claim'
        claimed = frames[1:]
        assert [f['params']['type'] for f in claimed][:1] == ['message.start']
        assert claimed[-1]['params']['type'] == 'message.complete'
        assert all(f['params']['authority_epoch'] == epoch for f in claimed)
        assert len({f['params']['execution_generation'] for f in claimed}) == 1
    idle = captured['epoch2']['idle_mutation_frames']
    assert [f['params']['type'] for f in idle] == ['session.updated']
    assert not {'authority_epoch', 'execution_generation', 'admission_id'} & set(idle[0]['params']), \
        'a settled execution kept stamping later idle mutations'
    # The Desktop fence tests replay this exact server output; keep the committed copy honest.
    committed = json.loads(FIXTURE.read_text(encoding='utf-8'))
    for label in ('epoch1', 'epoch2'):
        assert _projection(committed[label]['frames']) == _projection(captured[label]['frames'])
    assert _projection(committed['epoch2']['idle_mutation_frames']) == _projection(idle)
