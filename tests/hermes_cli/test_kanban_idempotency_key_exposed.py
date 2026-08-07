"""``idempotency_key`` must survive the trip back out through the CLI.

Automation that files a task under a dedup key needs to recognise that task
later — to notice it is already open, to comment on it, to decide whether to
file again. ``create`` returns an id but never says whether it deduped, and
every other lookup is by id, so without the key on the way out the only
available handle is ``title``. That is not a key: a human editing the title
detaches the automation from its own task, silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import _task_to_dict


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def test_idempotency_key_is_serialized(board):
    tid = kb.create_task(board, title="nightly import failed",
                         idempotency_key="import:nightly")
    payload = _task_to_dict(kb.get_task(board, tid))
    assert payload["idempotency_key"] == "import:nightly"


def test_absent_key_serializes_as_none(board):
    """A hand-created task has no key, and must not blow up serialization."""
    tid = kb.create_task(board, title="written by a human")
    payload = _task_to_dict(kb.get_task(board, tid))
    assert payload["idempotency_key"] is None


def test_key_survives_a_title_edit(board):
    """The reason title is not a substitute for the key."""
    tid = kb.create_task(board, title="original title",
                         idempotency_key="alert:disk-full")
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET title = ? WHERE id = ?",
            ("disk full on prod-3 (renamed by hand)", tid),
        )

    payload = _task_to_dict(kb.get_task(board, tid))
    assert payload["title"] != "original title"
    assert payload["idempotency_key"] == "alert:disk-full"
