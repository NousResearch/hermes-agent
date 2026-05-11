from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.inbox_outbox import (
    archive_task,
    claim_task,
    complete_task,
    create_inbox_task,
    fail_task,
    list_claimed_items,
    list_completed_items,
    list_pending_items,
    resolve_task,
)


def _set_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> Path:
    home = tmp_path / name
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_create_task_writes_atomically_and_lists_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _set_home(monkeypatch, tmp_path, "queue-home-create")

    record = create_inbox_task(
        {"subject": "ship milestone 1", "priority": "high"},
        title="Ship milestone 1",
    )

    path = Path(record["current_path"])
    assert path.exists()
    assert path.parent == home / "inbox" / "pending"

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["state"] == "pending"
    assert on_disk["payload"]["subject"] == "ship milestone 1"
    assert on_disk["history"][-1]["state"] == "pending"

    pending = list_pending_items()
    assert len(pending) == 1
    assert pending[0]["task_id"] == record["task_id"]
    assert list_claimed_items() == []
    assert list_completed_items() == []


def test_claim_task_moves_item_into_claimed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _set_home(monkeypatch, tmp_path, "queue-home-claim")

    created = create_inbox_task({"subject": "triage inbox"}, title="Triage inbox")
    claimed = claim_task(created["task_id"], note="picked up by worker")

    claimed_path = Path(claimed["current_path"])
    assert claimed["state"] == "claimed"
    assert claimed_path.parent == home / "inbox" / "claimed"
    assert not Path(created["current_path"]).exists()
    assert list_pending_items() == []
    assert len(list_claimed_items()) == 1

    history_states = [item["state"] for item in claimed["history"]]
    assert history_states == ["pending", "claimed"]
    assert claimed["history"][-1]["note"] == "picked up by worker"


@pytest.mark.parametrize(
    ("finisher", "expected_state", "expected_subdir"),
    [
        (complete_task, "completed", "completed"),
        (fail_task, "failed", "failed"),
        (archive_task, "archived", "archived"),
    ],
)
def test_finalize_moves_claimed_tasks_to_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finisher,
    expected_state: str,
    expected_subdir: str,
):
    home = _set_home(monkeypatch, tmp_path, f"queue-home-{expected_state}")

    created = create_inbox_task({"subject": f"finish {expected_state}"}, title=f"Finish {expected_state}")
    claim_task(created["task_id"])

    finished = finisher(
        created["task_id"],
        result={"summary": f"{expected_state} ok"},
        note=f"{expected_state} note",
    )

    finished_path = Path(finished["current_path"])
    assert finished["state"] == expected_state
    assert finished_path.parent == home / "outbox" / expected_subdir
    assert finished["metadata"]["summary"] == f"{expected_state} ok"
    assert list_completed_items(expected_state)[0]["task_id"] == created["task_id"]
    assert len(list_completed_items()) == 1

    history_states = [item["state"] for item in finished["history"]]
    assert history_states == ["pending", "claimed", expected_state]


def test_resolve_task_uses_hermes_home_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _set_home(monkeypatch, tmp_path, "custom-hermes-home")

    record = create_inbox_task({"subject": "home override"}, title="Home override")
    resolved = resolve_task(record["task_id"])

    assert Path(record["current_path"]).is_relative_to(home)
    assert resolved["task_id"] == record["task_id"]
    assert resolved["current_path"] == record["current_path"]