"""The fences a generation walk passed: the bind's write pins them, and reports a walk one of them no longer holds."""


def test_keep_passed_fences_reports_a_walk_it_could_not_pin(store):
    """``keep_passed_fences`` pins every fence a walk passed and says whether the walk still holds:
    a passed generation without a marker means the walk would now stop there."""
    import json
    from functools import partial
    from gateway.session_api import api_generation_id
    from persistence.sqlite.authority._runtime_retirement import keep_passed_fences
    from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, retire_routes
    derive = partial(api_generation_id, 'root-x')
    passed = ['root-x', derive(1)]
    store.holder()._execute_write(lambda conn: retire_routes(conn, passed, expiring=passed))
    assert store.holder()._execute_write(lambda conn: keep_passed_fences(conn, derive, derive(2))) is True
    assert all('expires' not in json.loads(store.state_meta.get(RETIRED_PREFIX + sid)) for sid in passed)
    store.holder()._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (RETIRED_PREFIX + passed[1],)))
    assert store.holder()._execute_write(lambda conn: keep_passed_fences(conn, derive, derive(2))) is False
