"""``sessions.min_interval_hours: 0`` (or negative) must not reach ``maybe_auto_archive`` /
``maybe_auto_prune_and_vacuum`` un-floored from the gateway's own housekeeping chore.

``now - last < min_interval_hours * 3600`` is false on every call once ``min_interval_hours`` is
<= 0, so the sweep this throttle gates would run on every housekeeping tick (60s here) instead of
at most once per interval — the exact class of bug ``agent.curator._bounded_count`` already
guards against for ``curator.interval_hours``.
"""

from pathlib import Path

import gateway.run as gateway_run


def test_state_db_maintenance_floors_a_non_positive_min_interval_hours(tmp_path: Path, monkeypatch):
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "sessions:\n"
        "  auto_archive: true\n"
        "  auto_archive_days: 3\n"
        "  auto_prune: true\n"
        "  retention_days: 90\n"
        "  min_interval_hours: 0\n"
        "  vacuum_after_prune: false\n",
        encoding="utf-8")

    archive_calls: list = []
    prune_calls: list = []
    monkeypatch.setattr(
        SessionDB, "maybe_auto_archive",
        lambda self, **kw: archive_calls.append(kw) or {"skipped": False, "archived": 0})
    monkeypatch.setattr(
        SessionDB, "maybe_auto_prune_and_vacuum",
        lambda self, **kw: prune_calls.append(kw) or {"skipped": False, "pruned": 0, "closed": 0, "vacuumed": False})

    gateway_run._housekeeping_state_db_maintenance(launch=None)

    assert archive_calls == [{"idle_days": 3.0, "min_interval_hours": 24}]
    assert prune_calls == [{
        "retention_days": 90, "min_interval_hours": 24,
        "min_vacuum_interval_days": 30, "vacuum": False,
        "sessions_dir": tmp_path / "sessions",
    }]
