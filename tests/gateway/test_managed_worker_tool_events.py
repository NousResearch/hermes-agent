"""A managed worker's tool events reach every viewer in the shape an in-process turn publishes.

Ink reads ``tool_id``/``context``; ACP reads ``args``/``result``/``is_error`` to show paths, output and
failures; API observers get the real arguments and result. The worker forwards its callback arguments
and the owner builds the event with the ordinary turn's payload builder (real authority over SQLite)."""
from types import SimpleNamespace

from agent.managed_worker import encode_frame, read_frame, tool_frame
from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input


def _wire(frame):
    import io
    return read_frame(io.BytesIO(encode_frame(frame)))


def test_worker_tool_frames_publish_the_in_process_payload(tmp_path):
    from gateway.run_turn_progress import _tool_complete_payload, _tool_lifecycle_payload, publish_worker_tool_event
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False))
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db, epoch=epoch)
        live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='s', request_id='r', payload={'text': 'go'})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        live.event_stream.execution = {'admission_id': row['admission_id']}
        frames, observed = [], []
        live.event_stream.observers.add(frames.append)
        authority.api_observers = {row['admission_id']: [{
            'tool_start_callback': lambda *a: observed.append(('start', a)),
            'tool_complete_callback': lambda *a: observed.append(('complete', a))}]}

        args = {'command': 'echo OUT; exit 3', 'timeout': 10}
        result = '{"output": "OUT", "exit_code": 3, "error": null}'
        start = _wire({'type': 'tool.start', **tool_frame('call-1', 'terminal', args)})
        done = _wire({'type': 'tool.complete', **tool_frame('call-1', 'terminal', args, result)})
        assert publish_worker_tool_event(authority, 's', row['generation'], start)
        assert publish_worker_tool_event(authority, 's', row['generation'], done)

        published = [(f['params']['type'], f['params']['payload']) for f in frames]
        assert published == [
            ('tool.start', _tool_lifecycle_payload('call-1', 'terminal', args)),
            ('tool.complete', _tool_complete_payload('call-1', 'terminal', args, result, is_error=True, verbose=False))]
        complete = published[1][1]
        assert complete['tool_id'] == 'call-1' and complete['context'] and complete['is_error'] is True
        assert complete['result'] == result and complete['args'] == args
        assert observed == [('start', ('call-1', 'terminal', args)), ('complete', ('call-1', 'terminal', args, result))]
        # Anything else (the retired id/name-only shape included) is not a tool frame.
        assert not publish_worker_tool_event(authority, 's', row['generation'],
                                             {'type': 'tool.start', 'tool_call_id': 'x', 'name': 'terminal'})
    finally:
        db.close()
