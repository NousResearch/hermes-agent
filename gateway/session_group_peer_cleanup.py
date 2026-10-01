"""Durable exact-grant cleanup owned by the canonical room home.

Obligations live independently of routes: replacing or deleting a route cannot
forget the bearer still needing retirement. A lost reply is retried idempotently.
The state database has the same private custody boundary as hosted_room_links.
"""
import hashlib
import json
import time

from gateway import hosted_room_links as links, hosted_rooms
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

_PREFIX = 'gateway.peer.cleanup.v1:'


def obligation_key(grant, mode='exact'):
    return _PREFIX + hashlib.sha256((mode + '\0' + grant).encode()).hexdigest()


def retain(db_path, link, *, mode='exact', conn=None):
    """Journal before a route write or remote effect, optionally in its transaction."""
    key = obligation_key(link.grant, mode)
    value = json.dumps({'link': link.as_record(), 'mode': mode, 'attempts': 0, 'next_at': 0})
    def write(writer):
        writer.execute('INSERT OR IGNORE INTO hosted_room_peer_cleanup(key,value) VALUES (?,?)', (key, value))
    if conn is not None:
        write(conn)
    else:
        with hosted_rooms._transaction(db_path) as writer:
            write(writer)
    return key


def release(conn, grant):
    """Publication and removing its provisional cleanup obligation are one write."""
    conn.execute('DELETE FROM hosted_room_peer_cleanup WHERE key=?', (obligation_key(grant),))


def obligations(db_path):
    with hosted_rooms._transaction(db_path) as conn:
        rows = conn.execute('SELECT key,value FROM hosted_room_peer_cleanup WHERE key LIKE ?', (_PREFIX + '%',)).fetchall()
    result = []
    for row in rows:
        try:
            value = json.loads(row['value'])
            links.StoredRoomLink.from_record(value['link'])
            if value['mode'] not in {'exact', 'scope'} or type(value['attempts']) is not int:
                raise ValueError('invalid cleanup record')
            float(value['next_at'])
        except Exception:
            value = {'corrupt': True}
        result.append((row['key'], value))
    return result


def status(db_path, room_id=None):
    result = []
    for _, value in obligations(db_path):
        if value.get('corrupt'):
            result.append({'status': 'unreadable'})
        elif room_id is None or value['link']['room_id'] == room_id:
            result.append({'room_id': value['link']['room_id'], 'member_id': value['link']['member_id'],
                           'mode': value['mode'], 'status': 'pending', 'attempts': value['attempts']})
    return result


def drain(service, *, force=False, room_id=None):
    """Retry a bounded batch even when no live room remains, including after restart."""
    from gateway.session_group_peer_routes import _retire
    from tui_gateway.hosted_room_peer_http import room_grant_request_budget, room_grant_request_budget_remaining
    from tui_gateway.hosted_room_service import _grant_revoke_is_terminal
    now = time.time()
    # This is the same publication lock used by registration/renewal/Disband.
    # Provisional grants cannot be retired while their publication is in flight.
    with service.peer_route_lock, room_grant_request_budget(2.0):
        due = [(key, value) for key, value in obligations(service.db_path)
               if not value.get('corrupt')
               and (room_id is None or value['link']['room_id'] == room_id)
               and (force or value['next_at'] <= now)]
        for key, value in sorted(due, key=lambda item: (item[1]['next_at'], item[0]))[:32]:
            if room_grant_request_budget_remaining() <= 0:
                break
            try:
                link = links.StoredRoomLink.from_record(value['link'])
                client = PeerRunsHTTPClient(base_url=link.target_url, api_key='', timeout_seconds=2,
                                            proof_install_id=link.catalog.installation_id)
                if value['mode'] == 'exact':
                    _retire(client, link.grant)
                else:
                    try:
                        client.revoke_grant(grant=link.grant)
                    except PeerRunsHTTPError as exc:
                        if not _grant_revoke_is_terminal(exc):
                            raise
            except Exception:
                value['attempts'] += 1
                value['next_at'] = now + min(120, 2 ** min(value['attempts'], 7))
                with hosted_rooms._transaction(service.db_path) as conn:
                    conn.execute('UPDATE hosted_room_peer_cleanup SET value=? WHERE key=?', (json.dumps(value), key))
            else:
                with hosted_rooms._transaction(service.db_path) as conn:
                    conn.execute('DELETE FROM hosted_room_peer_cleanup WHERE key=?', (key,))
