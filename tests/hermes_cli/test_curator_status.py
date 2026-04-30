"""Tests for the curator CLI status renderer."""

from types import SimpleNamespace


def test_status_uses_last_activity_not_only_last_used(monkeypatch, capsys):
    import agent.curator as curator_state
    import hermes_cli.curator as curator_cli
    import tools.skill_usage as skill_usage

    monkeypatch.setattr(curator_state, "load_state", lambda: {
        "paused": False,
        "last_run_at": None,
        "last_run_summary": "(none)",
        "run_count": 0,
    })
    monkeypatch.setattr(curator_state, "is_enabled", lambda: True)
    monkeypatch.setattr(curator_state, "get_interval_hours", lambda: 168)
    monkeypatch.setattr(curator_state, "get_stale_after_days", lambda: 30)
    monkeypatch.setattr(curator_state, "get_archive_after_days", lambda: 90)
    monkeypatch.setattr(skill_usage, "agent_created_report", lambda: [
        {
            "name": "recently-viewed",
            "state": "active",
            "pinned": False,
            "use_count": 0,
            "view_count": 3,
            "patch_count": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_used_at": None,
            "last_viewed_at": "2026-04-30T10:00:00+00:00",
            "last_patched_at": "2026-04-30T11:00:00+00:00",
            "last_activity_at": "2026-04-30T11:00:00+00:00",
            "activity_count": 4,
        }
    ])

    assert curator_cli._cmd_status(SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "least recently active" in out
    assert "activity=  4" in out
    assert "last_activity=never" not in out
    assert "last_used=never" not in out
