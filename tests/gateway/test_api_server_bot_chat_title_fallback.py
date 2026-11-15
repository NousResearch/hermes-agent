"""The peer-dm title fallback for a reset LOCAL Bot Chat honours the listing's own filters.

``GET /api/sessions?title=Bot+Chat`` lists a reset lineage as its latest, untitled segment, so a
fallback returns the titled root. It must not resurrect a deliberately archived Bot Chat into
``hermes peer dm`` routing, nor answer a ``source=`` query the row does not match."""
import pytest

from gateway.platforms import api_server_bot_chat as bot_chat


class FakeDB:
    def __init__(self, **row):
        self.row = dict({'id': 'local-root', 'title': 'Bot Chat', 'end_reason': 'session_reset',
                         'hidden': 0, 'archived': 0, 'source': 'local'}, **row)

    def get_session_by_title(self, title):
        return dict(self.row) if title == self.row['title'] else None


@pytest.fixture(autouse=True)
def reset_lineage(monkeypatch):
    monkeypatch.setattr('hermes_state_local.local_receipt', lambda db, sid: {'lineage': [sid, 'fresh']})


def test_the_reset_bot_chat_is_found():
    assert [r['id'] for r in bot_chat._reset_local_title_owner(FakeDB(), 'Bot Chat', True)] == ['local-root']
    assert bot_chat._reset_local_title_owner(FakeDB(), 'Bot Chat', True, source='local') != []


def test_a_deliberately_archived_bot_chat_is_not_returned():
    assert bot_chat._reset_local_title_owner(FakeDB(archived=1), 'Bot Chat', True) == []


def test_a_source_filter_the_row_does_not_match_excludes_it():
    assert bot_chat._reset_local_title_owner(FakeDB(), 'Bot Chat', True, source='telegram') == []


def test_hidden_needs_include_hidden():
    assert bot_chat._reset_local_title_owner(FakeDB(hidden=1), 'Bot Chat', False) == []
