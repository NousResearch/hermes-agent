"""Worker rows use the same stored-content interpretation as transcripts: a literal string with the reserved prefix reads back as written."""

import pytest

from tests.contracts.workers._support import SESSION, assistant, ident, worker


@pytest.mark.parametrize("content, expected", [("before\ud800after", "before\ufffdafter"), ("\x00json:[1,2]", "\x00json:[1,2]"), ("\x00json:bad", "\x00json:bad"), ("body", "body")])
def test_worker_append_uses_stored_content(backend, scope_a, content, expected):
    store, epoch = worker(backend, scope_a)
    result = store.worker_executions.append_transcript(**ident(epoch), messages=[assistant(content)])
    rows = store.messages.read_transcript(SESSION)
    assert len(rows) == 1
    assert rows[0].id == result.annotations[0].row_id
    assert rows[0].content == expected
