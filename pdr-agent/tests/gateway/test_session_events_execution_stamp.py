"""The execution stamp on published events marks exactly one claimed, unsettled execution."""
import pytest


@pytest.mark.asyncio
async def test_settled_execution_stops_stamping_idle_mutations(tmp_path, monkeypatch):
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
        assert [(f['params']['type'], f['params']['payload'].get('running')) for f in claimed[-2:]] == [
            ('message.complete', None), ('session.info', False)], 'the idle snapshot preceded the completion'
        assert all(f['params']['authority_epoch'] == epoch for f in claimed)
        assert len({f['params']['execution_generation'] for f in claimed}) == 1
    idle = captured['epoch2']['idle_mutation_frames']
    assert [f['params']['type'] for f in idle] == ['session.updated']
    assert not {'authority_epoch', 'execution_generation', 'admission_id'} & set(idle[0]['params']), \
        'a settled execution kept stamping later idle mutations'
