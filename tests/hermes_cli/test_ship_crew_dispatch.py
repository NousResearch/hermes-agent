from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.ship_crew_dispatch import (
    quarantine_task,
    validate_task_for_dispatch,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_task_routing_round_trips_and_rejects_running(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="route me",
            assignee="engineer",
            complexity_tier="T2",
            route_id="local-t2",
            reasoning_effort="medium",
            executor="hermes-native",
            executor_mode="medium",
            quota_domain="local",
            routing_metadata={"contract_version": "1.0", "governance_class": "standard"},
        )
        task = kb.get_task(conn, task_id)
        assert task and task.route_id == "local-t2"
        assert validate_task_for_dispatch(task).valid
        assert kb.claim_task(conn, task_id) is not None
        assert not kb.update_task_routing(conn, task_id, route_id="changed")


def test_invalid_route_is_quarantined_before_dispatch(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="bad route", assignee="engineer", route_id="r")
        task = kb.get_task(conn, task_id)
        assert task is not None
        validation = validate_task_for_dispatch(task)
        assert not validation.valid
        assert quarantine_task(conn, task_id, validation)
        quarantined = kb.get_task(conn, task_id)
        assert quarantined and quarantined.status == "blocked"
        assert quarantined.block_kind == "contract_rejected"
