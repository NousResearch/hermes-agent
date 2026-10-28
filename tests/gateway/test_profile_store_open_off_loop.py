"""A multiplexed boot opens each secondary profile's state.db off the event loop, under its own scope.

A first open runs the schema / FTS setup (about a second per store on a loaded host). Run on the
loop, an 11-profile boot froze every connected client for ~10 s.
"""
import threading

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_secondary_store_opens_run_off_the_loop_in_their_own_scope(tmp_path, monkeypatch):
    import hermes_state_registry
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from hermes_constants import get_hermes_home

    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    opens = []
    real_open = hermes_state_registry._open_session_db

    def spy(path):
        opens.append((path.parent.resolve(), threading.get_ident(), get_hermes_home().resolve()))
        return real_open(path)

    monkeypatch.setattr(hermes_state_registry, "_open_session_db", spy)
    loop_thread = threading.get_ident()
    runner = None
    try:
        runner = _runner(root, homes)  # opens the launch store (start_gateway prewarms it off-loop)
        opens.clear()
        await initialize_gateway_runtime(runner)
        secondaries = {home.resolve() for _name, home in homes[1:]}
        opened = {parent for parent, _tid, _scope in opens}
        assert secondaries <= opened
        for parent, tid, scope in opens:
            assert tid != loop_thread, f"{parent}/state.db was opened on the event loop"
            assert scope == parent, "the worker opened a store outside that profile's scope"
        for _name, home in homes:
            assert runner.session_authorities.for_home(home).db.db_path.parent.resolve() == home.resolve()
    finally:
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
