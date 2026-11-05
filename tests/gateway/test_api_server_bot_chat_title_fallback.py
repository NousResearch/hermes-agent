"""The peer-dm title fallback for a reset LOCAL Bot Chat honours the listing's own filters.

``GET /api/sessions?title=Bot+Chat`` lists a reset lineage as its latest, untitled segment, so a
fallback returns the titled root. It must not resurrect a deliberately archived Bot Chat into
``hermes peer dm`` routing, nor answer a ``source=`` query the row does not match."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server_bot_chat as bot_chat
from tests.fixtures.storage_seed import seed_session

_ROW_FIELDS = (
    'source', 'user_id', 'model', 'title', 'started_at', 'ended_at', 'end_reason', 'message_count', 'tool_call_count',
    'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens',
    'estimated_cost_usd', 'actual_cost_usd', 'api_call_count', 'parent_session_id', 'pinned', 'archived', 'hidden',
    'system_prompt')


class FakeStore:
    """A storage interface handle holding one session row and a creation receipt whose lineage was reset."""

    def __init__(self, **row):
        fields = dict.fromkeys(_ROW_FIELDS) | {'title': 'Bot Chat', 'end_reason': 'session_reset', 'hidden': False,
                                               'archived': False, 'source': 'local'} | row
        self.row = SimpleNamespace(session_id='local-root', model_config={}, **fields)
        self.sessions = SimpleNamespace(get_by_title=lambda title: self.row if title == self.row.title else None)
        self.bindings = SimpleNamespace(receipt=lambda sid: SimpleNamespace(lineage=(sid, 'fresh')))


def test_the_reset_bot_chat_is_found():
    assert [r['id'] for r in bot_chat._reset_local_title_owner(FakeStore(), 'Bot Chat', True)] == ['local-root']
    assert bot_chat._reset_local_title_owner(FakeStore(), 'Bot Chat', True, source='local') != []


def test_a_deliberately_archived_bot_chat_is_not_returned():
    assert bot_chat._reset_local_title_owner(FakeStore(archived=True), 'Bot Chat', True) == []


def test_a_source_filter_the_row_does_not_match_excludes_it():
    assert bot_chat._reset_local_title_owner(FakeStore(), 'Bot Chat', True, source='telegram') == []


def test_hidden_needs_include_hidden():
    assert bot_chat._reset_local_title_owner(FakeStore(hidden=True), 'Bot Chat', False) == []


_IDENTITY = {"profile_id": "p", "principal_id": "local", "request_id": "r", "session_id": "local-root",
             "route": "local:local-root", "entry": {"session_id": "fresh"}, "policy": {}}


# (receipt as stored, what main's ``local_receipt(...).get('lineage') or []`` made of it)
@pytest.mark.parametrize("stored,found", [
    (json.dumps({**_IDENTITY, "lineage": ["local-root", "fresh"]}), True),
    (json.dumps(_IDENTITY), False),
    (json.dumps({**_IDENTITY, "lineage": None}), False),
    (json.dumps({"lineage": ["local-root", "fresh"]}), True),
    (json.dumps({**{k: v for k, v in _IDENTITY.items() if k != "route"}, "lineage": ["local-root", "fresh"]}), True),
    (json.dumps({"lineage": None}), False),
    (json.dumps([1, 2]), False),
    ("{", False),
    (None, False),
], ids=["receipt", "no-lineage", "lineage-null", "no-identity", "no-route", "no-identity-lineage-null",
        "not-an-object", "undecodable", "no-receipt"])
def test_the_stored_lineage_decides_on_the_store(tmp_path, stored, found):
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "local-root", "local")
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("UPDATE sessions SET title='Bot Chat', end_reason='session_reset', ended_at=1 WHERE id='local-root'")
        if stored is not None:
            conn.execute("INSERT INTO state_meta (key, value) VALUES ('gateway.local_policy.v1:local-root', ?)", (stored,))
    with open_persistence(profile_scope(home=tmp_path)) as store:
        rows = bot_chat._reset_local_title_owner(store, "Bot Chat", True)
    assert [row["id"] for row in rows] == (["local-root"] if found else [])
