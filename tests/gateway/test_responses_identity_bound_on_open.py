"""A reopened Responses store enforces its settled-identity bound before serving a request."""
from gateway.platforms.api_server_response_store import ResponseStore


def test_reopen_prunes_settled_identities_to_the_bound(tmp_path):
    path = str(tmp_path / 'response_store.db')
    store = ResponseStore(db_path=path, max_size=10, max_identities=10)
    for index in range(4):
        key = f'idem:scope:k{index}'
        assert store.bind_request_key(key, 'fp')
        store.put(key, {'fingerprint': 'fp', 'response': {'id': f'resp_{index}'}})
    assert store._conn.execute('SELECT COUNT(*) FROM response_keys WHERE settled_at IS NOT NULL').fetchone()[0] == 4
    store.close()
    reopened = ResponseStore(db_path=path, max_size=10, max_identities=1)
    try:
        kept = [row[0] for row in reopened._conn.execute('SELECT request_key FROM response_keys')]
        assert kept == ['idem:scope:k3'], kept
    finally:
        reopened.close()
