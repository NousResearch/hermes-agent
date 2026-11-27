"""``session/list`` withholds hidden sessions; the side panel's catalog reads them, flagged.

A tab in the strip is an *unhidden* session, so a client that draws tabs needs both halves of the
listing and the flag that tells them apart (#160752).
"""
from acp_adapter.catalog import catalog_sessions, catalog_with_hidden
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _seed(home):
    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, "open-tab", "acp", cwd=str(home))
        seed_messages(store, "open-tab", {"role": "user", "content": "open input"})
        seed_session(store, "closed-tab", "acp", cwd=str(home), hidden=True)
        seed_messages(store, "closed-tab", {"role": "user", "content": "closed input"})


def test_catalog_with_hidden_returns_hidden_rows_with_their_flag(tmp_path):
    _seed(tmp_path)
    rows = {row["session_id"]: row for row in catalog_with_hidden(tmp_path)}
    assert set(rows) == {"open-tab", "closed-tab"}
    assert rows["open-tab"]["hidden"] is False
    assert rows["closed-tab"]["hidden"] is True
    assert rows["closed-tab"]["message_count"] == 1


def test_catalog_sessions_still_withholds_hidden_rows(tmp_path):
    """``session/list``'s projection is unchanged: two readers of one store, two rules."""
    _seed(tmp_path)
    assert [row["session_id"] for row in catalog_sessions(tmp_path)] == ["open-tab"]


def test_catalog_with_hidden_keeps_the_cwd_filter(tmp_path):
    _seed(tmp_path)
    assert [row["session_id"] for row in catalog_with_hidden(tmp_path, "/elsewhere")] == []
    assert {row["session_id"] for row in catalog_with_hidden(tmp_path, str(tmp_path))} == {
        "open-tab", "closed-tab"}
