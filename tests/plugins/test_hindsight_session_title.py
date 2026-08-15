"""Hindsight session-title threading (#86824).

sync_turn accepts a `session_title` kwarg and surfaces it as a `title:`
lineage tag at retain time so banks stop listing sessions under raw ids.
"""

from __future__ import annotations

from plugins.memory.hindsight import HindsightMemoryProvider


def _make_provider() -> HindsightMemoryProvider:
    p = HindsightMemoryProvider.__new__(HindsightMemoryProvider)
    p._bank_id = "bank"
    p._session_id = ""
    p._parent_session_id = ""
    p._retain_tags = []
    p._observation_scopes = None
    return p


def test_lineage_tags_carry_title():
    """The title lands in the lineage tags the retain path ships (#86824)."""
    p = _make_provider()
    p._session_id = "sess-1"
    p._parent_session_id = ""

    tags = p._lineage_tags("API Key Update Verified")

    assert "title:API Key Update Verified" in tags
    assert "session:sess-1" in tags
    assert p._session_title == "API Key Update Verified"


def test_lineage_tags_omit_empty_title():
    p = _make_provider()
    p._session_id = "sess-1"
    p._parent_session_id = ""

    tags = p._lineage_tags("")

    assert "title:" not in " ".join(tags)
    assert "session:sess-1" in tags
