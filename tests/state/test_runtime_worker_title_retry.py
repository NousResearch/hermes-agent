"""Expected title conflicts are committed outcomes, not failed persistence."""
import pytest

from agent.title_generator import _persist_session_title
from hermes_state_runtime import RuntimeStoreError, mutate_worker_execution
from tests.state.test_runtime_worker_lifecycle import setup_store


def test_worker_title_collision_can_dedupe_without_poisoning_persistence(tmp_path):
    db, store, scope = setup_store(tmp_path)
    try:
        db.set_session_title('foreign', 'shared-title')
        # Call the real consumer: it catches ValueError, gets a suffix, then retries.
        assert _persist_session_title(store, 'owned', 'shared-title', source='llm') == 'shared-title #2'
        assert db.get_session('owned')['title_source'] == 'llm'
        assert db.get_session('foreign')['title'] == 'shared-title'
        with pytest.raises(ValueError):
            store.set_session_title('owned', 'x' * (db.MAX_TITLE_LENGTH + 1))
        with pytest.raises(ValueError):
            store.set_session_title_source('owned', 'invalid')
        assert store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'still-running'}]) == 1
        # An unrelated title query is not an execution capability.
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            mutate_worker_execution(db, **scope, sequence=store.journal['next_sequence'],
                                    operation='session.next_title', payload={'base_title': 'unrelated'})
        assert store.finish()['status'] == 'terminal'
    finally:
        store.close()
        db.close()
