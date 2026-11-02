from contextlib import closing
import subprocess

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("heartbeat,new_message", [(200, 300), (300, 200)])
def test_project_recency_tracks_latest_live_activity(tmp_path, heartbeat, new_message):
    from tui_gateway import git_probe
    from tui_gateway.server import _discover_repos_payload

    ongoing = tmp_path / "ongoing-project"
    idle = tmp_path / "idle-project"
    for root in (ongoing, idle):
        subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    git_probe.invalidate()
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        result = db.import_sessions([
            {"id": "long-lived", "source": "cli", "cwd": str(ongoing), "started_at": 100},
            {"id": "idle", "source": "cli", "cwd": str(idle), "started_at": 250},
        ])
        assert result["ok"]
        db.append_message("long-lived", "user", "ongoing work", timestamp=new_message)
        db.touch_session_activity("long-lived", ts=heartbeat)
        repos = _discover_repos_payload(db, backfill=False, include_cached=False)
        assert [repo["label"] for repo in repos] == [ongoing.name, idle.name]
        assert repos[0]["last_active"] == max(heartbeat, new_message)
        assert [repo["sessions"] for repo in repos] == [1, 1]


def test_project_recency_keeps_end_time_and_archive_filter(tmp_path):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        db.import_sessions([
            {"id": "ended", "cwd": "/project", "started_at": 100, "ended_at": 300,
             "messages": [{"role": "user", "content": "done", "timestamp": 200}]},
            {"id": "archived", "cwd": "/project", "started_at": 500, "archived": True},
        ])
        assert db.distinct_session_cwds() == [{"cwd": "/project", "sessions": 1, "last_active": 300}]
        assert db.distinct_session_cwds(include_archived=True) == [
            {"cwd": "/project", "sessions": 2, "last_active": 500}
        ]
