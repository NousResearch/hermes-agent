"""Real authenticated gateway transport for durable local operation notices."""
import asyncio

from tests.gateway.fixtures import tui_desktop_routes_peer as peer
from tests.gateway.fixtures.local_recovery_probe import rpc


async def notice(authority, ws, reader, sid, marker, receipt):
    events = []
    authority.sessions[sid].event_stream.observers.add(lambda frame: events.append(frame['params']))
    session = authority.db.get_session(sid)
    params = dict(session_id=sid, request_id='operation-final', expected_revision=session['runtime_revision'],
                  expected_generation=session['runtime_generation'], operation='notice',
                  payload={'text': '[Delivery test] operation passed'})
    denied = await peer._call(reader, 'session.mutate', **params)
    assert peer._error(denied) == 'permission_denied', denied
    reply = await rpc(ws, 'session.mutate', **params)
    assert 'result' in reply, reply
    notices = [event for event in events if event['type'] == 'notification.show']
    assert len(notices) == 1, events
    assert notices[0]['payload']['text'] == params['payload']['text']
    retry = await rpc(ws, 'session.mutate', **params)
    assert retry == reply, (reply, retry)
    assert len([event for event in events if event['type'] == 'notification.show']) == 1
    unknown = await rpc(ws, 'session.mutate', **(params | {'session_id': 'does-not-exist'}))
    assert 'error' in unknown, unknown
    receipt['notified'] = True


if __name__ == '__main__':
    peer.MODES['notice'] = notice
    asyncio.run(peer.probe('notice'))
