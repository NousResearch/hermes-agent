from contextlib import closing

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("needle", ["root", "middle", "tip", "literal_%"])
def test_lineage_ranking_preserves_filters_and_literal_matching(tmp_path, needle):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        middle = "literal_%" if needle == "literal_%" else "middle"
        assert db.import_sessions([
            {"id": "root", "source": "cli", "started_at": 100, "end_reason": "compression"},
            {"id": middle, "source": "cli", "started_at": 110,
             "parent_session_id": "root", "end_reason": "compression"},
            {"id": "tip", "source": "cli", "started_at": 120, "parent_session_id": middle},
            *[{"id": f"new-{needle}-{i}", "source": "cli", "started_at": 200 + i}
              for i in range(8)],
            {"id": f"{needle}-other", "source": "cron", "started_at": 300},
        ])["ok"]
        for filters in ({"source": "cli"}, {"sources": ["cli"]}, {"exclude_sources": ["cron"]}):
            rows = db.search_sessions_by_id(needle.upper(), limit=1, **filters)
            assert [row["id"] for row in rows] == ["tip"]
            assert rows[0]["_lineage_root_id"] == "root"
        db.set_session_archived("root", True)
        assert db.search_sessions_by_id(needle, limit=1, source="cli")[0]["id"] == "tip"
        assert db.search_sessions_by_id(
            needle, limit=1, source="cli", include_archived=False,
        )[0]["id"] == f"new-{needle}-7"
