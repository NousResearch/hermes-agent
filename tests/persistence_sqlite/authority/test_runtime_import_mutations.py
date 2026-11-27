"""Import rows and retry receipt either commit together or not at all."""
import sqlite3
import pytest
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, list_session_admissions
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import get_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def test_import_rolls_back_rows_with_receipt_failure(store):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    args = dict(epoch=epoch, principal_id='human', session_id='first', request_id='import',
                expected_revision=0, operation='import', payload={'sessions': [
                    {'id': 'first', 'source': 'cli', 'messages': [{'role': 'user', 'content': 'keep'}]},
                    {'id': 'second', 'source': 'cli', 'parent_session_id': 'first', 'messages': []}]})
    holder._execute_write(lambda c: c.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON state_meta WHEN NEW.key LIKE 'gateway.mutation.%' BEGIN SELECT RAISE(ABORT, 'receipt failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='receipt failed'):
        mutate_runtime_session(holder, **args)
    assert get_session(holder, 'first') is None and get_session(holder, 'second') is None
    holder._execute_write(lambda c: c.execute('DROP TRIGGER fail_receipt'))
    result, _, _ = mutate_runtime_session(holder, **args)
    assert result['imported_ids'] == ['first', 'second']
    assert get_session(holder, 'second')['parent_session_id'] == 'first'
    assert get_messages(holder, 'first')[0]['content'] == 'keep'
    assert mutate_runtime_session(holder, **args)[0] == result
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        mutate_runtime_session(holder, **(args | {'payload': {'sessions': []}}))


def test_import_never_overwrites_existing_live_rows(store):
    holder = store.holder()
    create_session(holder, 'live', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    admit_session_input(holder, epoch=epoch, principal_id='human', session_id='live', request_id='queued', payload={'text': 'keep'})
    before = get_session(holder, 'live')
    result, _, _ = mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='new',
        request_id='import', expected_revision=0, operation='import', payload={'sessions': [
            {'id': 'new', 'source': 'cli', 'messages': []},
            {'id': 'live', 'source': 'cli', 'title': 'Overwrite', 'messages': []}]})
    assert result['skipped_ids'] == ['live']
    assert get_session(holder, 'live') == before
    assert list_session_admissions(holder, session_id='live')[0]['status'] == 'queued'
