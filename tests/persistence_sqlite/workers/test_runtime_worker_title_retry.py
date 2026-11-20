"""Expected title conflicts are committed outcomes, not failed persistence."""
import pytest

from agent.title_generator import _persist_session_title
from persistence.conversation.sessions import MAX_TITLE_LENGTH
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.conversation._store.sessions import get_session
from persistence.sqlite.conversation._store.titles import set_session_title
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import setup_store


def test_worker_title_collision_can_dedupe_without_poisoning_persistence(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        set_session_title(holder, 'foreign', 'shared-title')
        # Call the real consumer: it catches ValueError, gets a suffix, then retries.
        assert _persist_session_title(store, 'owned', 'shared-title', source='llm') == 'shared-title #2'
        assert get_session(holder, 'owned')['title_source'] == 'llm'
        assert get_session(holder, 'foreign')['title'] == 'shared-title'
        with pytest.raises(ValueError):
            store.set_session_title('owned', 'x' * (MAX_TITLE_LENGTH + 1))
        with pytest.raises(ValueError):
            store.set_session_title_source('owned', 'invalid')
        assert store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'still-running'}]) == 1
        # An unrelated title query is not an execution capability.
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            mutate_worker_execution(holder, **scope, sequence=store.journal['next_sequence'],
                                    operation='session.next_title', payload={'base_title': 'unrelated'})
        assert store.finish()['status'] == 'terminal'
    finally:
        store.close()
