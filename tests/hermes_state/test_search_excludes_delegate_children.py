"""``search_messages(exclude_sources=[..., 'subagent', ...])`` honours the delegate-child marker.

A ``delegate_task`` child spawned under a gateway turn inherits the gateway's
``source`` (``HERMES_SESSION_SOURCE``) and is recognisable only by
``model_config['_delegate_from']``. Listings and the trigram index boundary are
marker-aware; the shared search filter was source-column-only, so such a child
leaked into ``session_search`` while its CLI-sourced sibling stayed hidden
(#111365, umbrella #51855).

The marker joins the boundary only when the caller excludes ``subagent``: bare
searches keep children word-searchable (v30, see
``test_fts_trigram_subagent_exclusion.py``).
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB

# Exactly what tools/session_search_tool.py passes (_HIDDEN_SESSION_SOURCES).
SESSION_SEARCH_EXCLUDES = ["kanban", "subagent", "tool"]
CJK_QUERY = "运行"  # 2 chars: below the trigram gate, answered by the LIKE fallback route


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _seed(db: SessionDB) -> None:
    db.create_session("root", source="telegram")
    db.create_session("plain", source="telegram")
    # Gateway-spawned child: gateway source + marker (only the marker identifies it).
    db.create_session("gw-kid", source="telegram", parent_session_id="root",
                      model_config={"_delegate_from": "root"})
    # CLI-spawned child: the source alone already identifies it.
    db.create_session("cli-kid", source="subagent", parent_session_id="root",
                      model_config={"_delegate_from": "root"})
    db.append_message("root", role="user", content="rootword")
    db.append_message("plain", role="assistant", content=f"plainword 普通会话{CJK_QUERY}正常")
    db.append_message("gw-kid", role="assistant", content=f"gwkidword 网关子任务{CJK_QUERY}正常")
    db.append_message("cli-kid", role="assistant", content="clikidword cli child transcript")


def _hits(db: SessionDB, query: str, **kwargs) -> list[str]:
    return sorted({row["session_id"] for row in db.search_messages(query, **kwargs)})


def test_source_scoped_search_hides_marker_child_on_every_route(db: SessionDB):
    """Excluding 'subagent' hides the gateway-sourced child too — on the FTS word route
    and on the CJK/LIKE fallback route — while an ordinary session on the same source
    and the CLI-sourced child behave as before."""
    _seed(db)
    assert _hits(db, "gwkidword", exclude_sources=SESSION_SEARCH_EXCLUDES) == []
    assert _hits(db, "clikidword", exclude_sources=SESSION_SEARCH_EXCLUDES) == []
    assert _hits(db, "plainword", exclude_sources=SESSION_SEARCH_EXCLUDES) == ["plain"]
    assert _hits(db, CJK_QUERY, exclude_sources=SESSION_SEARCH_EXCLUDES) == ["plain"]


def test_bare_search_still_reaches_marker_child(db: SessionDB):
    """v30 contract: with no exclude list, children remain word-searchable on both routes."""
    _seed(db)
    assert _hits(db, "gwkidword") == ["gw-kid"]
    assert _hits(db, CJK_QUERY) == ["gw-kid", "plain"]
