import asyncio
from contextlib import closing

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("target", ["needle", "needle-prefix"])
def test_id_match_quality_precedes_recency_limit(tmp_path, monkeypatch, target):
    from hermes_cli.web_routers.sessions import search_sessions
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = set_hermes_home_override(tmp_path)
    try:
        with closing(SessionDB()) as db:
            assert db.import_sessions([
                {"id": target, "source": "cli", "started_at": 100, "archived": True},
                *[{"id": f"recent-needle-{i}", "source": "cli", "started_at": 200 + i}
                  for i in range(12)],
            ])["ok"]
        for limit in (1, 2):
            rows = asyncio.run(search_sessions(q="NEEDLE", limit=limit, source="cli"))["results"]
            assert [row["session_id"] for row in rows] == [target, "recent-needle-11"][:limit]
            assert rows[0]["archived"] is True
    finally:
        reset_hermes_home_override(token)
