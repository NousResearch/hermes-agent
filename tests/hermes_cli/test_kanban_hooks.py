from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from kanban_hooks import HookRegistry, emit_kanban_event, wait_for_pending_events


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def reset_hook_registry():
    HookRegistry.reset_for_tests()
    yield
    HookRegistry.reset_for_tests()


def _create_hook(home: Path, hook_name: str, events: list[str], sink_path: Path) -> None:
    hook_dir = home / "kanban-hooks" / hook_name
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(
        "name: {name}\n"
        "description: test hook\n"
        "events:\n{events_block}".format(
            name=hook_name,
            events_block="".join(f"  - {event}\n" for event in events),
        ),
        encoding="utf-8",
    )
    (hook_dir / "handler.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        f"SINK = Path({str(sink_path)!r})\n\n"
        "def handle(event_type, context):\n"
        "    SINK.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with SINK.open('a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps({'event_type': event_type, 'context': context}, sort_keys=True) + '\\n')\n",
        encoding="utf-8",
    )


def _read_events(sink_path: Path) -> list[dict]:
    if not sink_path.exists():
        return []
    return [json.loads(line) for line in sink_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_emit_kanban_event_dispatches_only_matching_hooks(kanban_home, tmp_path):
    sink = tmp_path / "events.jsonl"
    _create_hook(kanban_home, "blocked-only", ["task:blocked"], sink)
    _create_hook(kanban_home, "completed-only", ["task:completed"], sink)

    emit_kanban_event("task:blocked", {"task_id": "t_demo", "board": "default"})
    wait_for_pending_events(timeout=2.0)

    events = _read_events(sink)
    assert len(events) == 1
    assert events[0]["event_type"] == "task:blocked"
    assert events[0]["context"]["task_id"] == "t_demo"


@pytest.mark.parametrize(
    ("event_name", "mutate", "assert_context"),
    [
        (
            "task:created",
            lambda conn: kb.create_task(conn, title="created", assignee="alice"),
            lambda event, task_id: (
                event["context"]["task_id"] == task_id
                and event["context"]["board"] == "default"
                and event["context"]["new_status"] == "ready"
                and event["context"]["assignee"] == "alice"
            ),
        ),
        (
            "task:blocked",
            lambda conn: (lambda tid: (kb.block_task(conn, tid, reason="need help"), tid)[1])(
                kb.create_task(conn, title="blocked")
            ),
            lambda event, task_id: (
                event["context"]["task_id"] == task_id
                and event["context"]["previous_status"] == "ready"
                and event["context"]["new_status"] == "blocked"
                and event["context"]["reason"] == "need help"
            ),
        ),
        (
            "task:completed",
            lambda conn: (lambda tid: (kb.complete_task(conn, tid, summary="done summary"), tid)[1])(
                kb.create_task(conn, title="completed")
            ),
            lambda event, task_id: (
                event["context"]["task_id"] == task_id
                and event["context"]["previous_status"] == "ready"
                and event["context"]["new_status"] == "done"
                and event["context"]["summary"] == "done summary"
            ),
        ),
        (
            "task:unblocked",
            lambda conn: (lambda tid: (kb.block_task(conn, tid, reason="wait"), kb.unblock_task(conn, tid), tid)[2])(
                kb.create_task(conn, title="unblocked")
            ),
            lambda event, task_id: (
                event["context"]["task_id"] == task_id
                and event["context"]["previous_status"] == "blocked"
                and event["context"]["new_status"] == "ready"
            ),
        ),
    ],
)
def test_kanban_db_transitions_emit_hook_events(
    kanban_home,
    tmp_path,
    event_name,
    mutate,
    assert_context,
):
    sink = tmp_path / f"{event_name.replace(':', '_')}.jsonl"
    _create_hook(kanban_home, event_name.replace(":", "-"), [event_name], sink)

    with kb.connect() as conn:
        task_id = mutate(conn)

    wait_for_pending_events(timeout=2.0)
    events = _read_events(sink)

    assert len(events) == 1
    assert events[0]["event_type"] == event_name
    assert assert_context(events[0], task_id)
