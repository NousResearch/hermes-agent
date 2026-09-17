"""``search_messages(exclude_sources=...)`` must honour the delegate-child marker.

Regression for #51855.

A ``delegate_task`` child normally lands on ``source='subagent'``, but under a
gateway turn it inherits the gateway's source (``HERMES_SESSION_SOURCE``) and is
then recognisable *only* by ``model_config['_delegate_from']``. The listing
boundary (``hermes_state_sessions``) and the trigram index boundary
(``fts_trigram_session_sql``) are marker-aware; the shared search filter was
source-column-only, so a gateway-sourced child leaked into ``session_search``
while its CLI-sourced sibling stayed hidden.

The v30 decision keeps children in the standard word index, so the marker may
only join the boundary when the caller actually asked to exclude ``subagent`` —
bare searches still return child transcripts (see
``test_fts_trigram_subagent_exclusion.py``).
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_state_common import fts_trigram_session_sql

# Exactly what tools/session_search_tool.py passes (_HIDDEN_SESSION_SOURCES).
SESSION_SEARCH_EXCLUDES = ["kanban", "subagent", "tool"]


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _seed(db: SessionDB) -> None:
    db.create_session("root", source="telegram")
    # Control: an ordinary session on a source that is not excluded.
    db.create_session("plain", source="telegram")
    # Child spawned under a gateway turn: gateway source + the delegate marker.
    db.create_session(
        "gw-kid", source="telegram", parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    # Child spawned from the CLI: source alone already identifies it.
    db.create_session(
        "cli-kid", source="subagent", parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    db.append_message("root", role="user", content="root token rootword")
    db.append_message("plain", role="user", content="plain control token plainword")
    db.append_message("gw-kid", role="assistant", content="gateway child token gwkidword")
    db.append_message("cli-kid", role="assistant", content="cli child token clikidword")


def _hits(db: SessionDB, query: str, **kwargs) -> list[str]:
    return sorted({row["session_id"] for row in db.search_messages(query, **kwargs)})


def test_marker_child_is_hidden_by_source_scoped_search(db: SessionDB):
    """The marker is part of the boundary the caller asked for, not just the source column."""
    _seed(db)
    excluded = ["kanban", "subagent", "tool"]
    # A gateway-sourced child carries a source the caller excluded *in spirit* but
    # not literally; only the marker can hide it.
    assert _hits(db, "gwkidword", exclude_sources=excluded) == []
    assert _hits(db, "clikidword", exclude_sources=excluded) == []
    # Control: an ordinary session on the very same source is still found.
    assert _hits(db, "plainword", exclude_sources=excluded) == ["plain"]


def test_visibility_surfaces_agree_on_marker_child(db: SessionDB):
    """Listings, the trigram index boundary, and source-scoped search all hide the
    same sessions — and all keep showing an ordinary session."""
    _seed(db)
    excluded = ["kanban", "subagent", "tool"]

    def in_listing(session_id: str) -> bool:
        return session_id in {
            row["id"] for row in db.list_sessions_rich(exclude_sources=excluded, limit=100)
        }

    def in_trigram_boundary(session_id: str) -> bool:
        rows = db._conn.execute(
            f"SELECT 1 FROM sessions s WHERE s.id = ? AND {fts_trigram_session_sql('s')}",
            (session_id,),
        ).fetchall()
        return bool(rows)

    def in_scoped_search(session_id: str, token: str) -> bool:
        return session_id in _hits(db, token, exclude_sources=excluded)

    # The v30 contract that stays true: children remain in the standard word index,
    # so a bare search (no exclude_sources) still reaches them.
    assert _hits(db, "gwkidword") == ["gw-kid"]
    assert _hits(db, "clikidword") == ["cli-kid"]

    for session_id, token in (("gw-kid", "gwkidword"), ("cli-kid", "clikidword")):
        assert not in_listing(session_id), session_id
        assert not in_trigram_boundary(session_id), session_id
        assert not in_scoped_search(session_id, token), session_id

    for session_id, token in (("root", "rootword"), ("plain", "plainword")):
        assert in_listing(session_id), session_id
        assert in_trigram_boundary(session_id), session_id
        assert in_scoped_search(session_id, token), session_id


# ── The CJK LIKE-fallback route ───────────────────────────────────────────
#
# ``_search_filter_clauses()`` is shared by every search route, but the FTS
# arms above only prove the word-MATCH route. A 2-char CJK query is below the
# trigram index's >=3-char-per-token gate (and the cjk-bigram index needs an
# optional tokenizer extension), so it answers via a canonical LIKE scan built
# by the same clause helper — the route
# ``test_fts_trigram_subagent_exclusion.py`` drives with ``source_filter``.
# The marker boundary must hold there too.

CJK_QUERY = "运行"  # shared by both contents; short enough to miss the trigram gate
CJK_CHILD_EXTRA = "网关子任务"
CJK_CONTROL_EXTRA = "普通会话"


def _seed_cjk(db: SessionDB) -> None:
    db.create_session("plain", source="telegram")
    # Gateway-sourced delegate child: only the marker identifies it.
    db.create_session(
        "gw-kid", source="telegram", parent_session_id="plain",
        model_config={"_delegate_from": "plain"},
    )
    db.append_message("gw-kid", role="assistant", content=f"{CJK_CHILD_EXTRA}{CJK_QUERY}正常")
    db.append_message("plain", role="assistant", content=f"{CJK_CONTROL_EXTRA}{CJK_QUERY}正常")


def test_cjk_like_fallback_honours_marker_boundary(db: SessionDB):
    """The CJK fallback route shares the boundary: scoped CJK search hides the
    marker child and keeps the control; bare CJK search still reaches the child."""
    _seed_cjk(db)
    # v30 parity first: the same CJK query with no exclude list reaches BOTH —
    # so the scoped zero below can only come from the filter, not tokenisation.
    assert _hits(db, CJK_QUERY) == ["gw-kid", "plain"]
    # Behaviour contract, not a snapshot: exactly the control survives scoping.
    assert _hits(db, CJK_QUERY, exclude_sources=SESSION_SEARCH_EXCLUDES) == ["plain"]
