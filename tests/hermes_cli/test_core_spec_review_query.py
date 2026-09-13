import sys
import time
from types import SimpleNamespace


def test_persistent_query_terminalizes_delegation(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import cli
    from tools import async_delegation as ad
    ad._reset_for_tests()
    rec = {'delegation_id': 'review-query', 'goal': 'offline', 'dispatched_at': time.time(),
           'status': 'running', 'interrupt_fn': lambda: None}
    ad._persist_dispatch(rec)
    ad._records[rec['delegation_id']] = rec
    for name in ('_wait_for_oneshot_background_completions', '_flush_one_shot_session_store', '_notify_single_query_session_finalize', '_shutdown_agent_memory_provider'):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    for name in ('_arm_exit_watchdog', '_reset_terminal_input_modes_on_exit'):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    monkeypatch.setattr(cli, '_CLEANUP_STEPS', (('_interrupt_async_delegations', Exception),))
    monkeypatch.setattr(cli, '_cleanup_done', False)
    try:
        cli._finalize_single_query(SimpleNamespace(_release_active_session=lambda: None))
        assert ad.get_durable_delegation(rec['delegation_id'])['state'] == 'unknown'
    finally:
        ad._reset_for_tests()
