"""Mandatory terminal-call rule in the shared kanban worker guidance.

Workers that end a turn with only prose (no terminal board tool) hang the
card: nothing on the board moves, the dispatcher re-queues with a corrective
reminder, and repeated violations burn the retry budget. The rule must be
present in every rendered worker prompt (``KANBAN_GUIDANCE`` is the single
shared template injected via ``system_prompt`` / ``agent_init``), and the
dispatcher must fail closed if a refactor or fork strips the block.

Superset of the partial fix proposed in PR #50386 (a 6-line prose rule with
no dispatch guard and no presence tests).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Presence: the block lives in the shared template
# ---------------------------------------------------------------------------


def test_kanban_guidance_carries_terminal_call_heading():
    """KANBAN_GUIDANCE is the single shared template — the heading must be in it."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert "## TERMINAL CALL REQUIRED" in KANBAN_GUIDANCE
    # The block must name every terminal tool the worker can legally end on
    # (agent/kanban_stop.py::_TERMINAL_KANBAN_TOOLS).
    for tool in (
        "kanban_complete",
        "kanban_block",
        "kanban_schedule",
        "kanban_request_review",
        "kanban_request_changes",
    ):
        assert f"`{tool}`" in KANBAN_GUIDANCE, f"terminal tool {tool} missing from block"


def test_kanban_guidance_block_promises_violation_consequence():
    """The rule states what happens when a turn ends without a terminal call —
    that consequence (re-queue / reminder / retry budget) is why models obey it."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    head = KANBAN_GUIDANCE.find("## TERMINAL CALL REQUIRED")
    tail = KANBAN_GUIDANCE.find("## Lifecycle")
    assert 0 < head < tail
    block = KANBAN_GUIDANCE[head:tail]
    assert "protocol violation" in block
    assert "retry budget" in block


def test_rendered_worker_prompt_contains_terminal_call_block(monkeypatch):
    """A dispatcher-owned worker's system prompt must contain the block.

    Mirrors the injection path: ``agent_init`` stamps ``_kanban_worker_guidance``
    and ``system_prompt._tool_guidance_block`` appends it when the kanban
    toolset is loaded.
    """
    from agent.prompt_builder import KANBAN_GUIDANCE
    from agent.system_prompt import _tool_guidance_block

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_render_test")
    agent = type("A", (), {
        "valid_tool_names": {"kanban_show", "kanban_complete"},
        "_kanban_worker_guidance": KANBAN_GUIDANCE,
        "_memory_enabled": False,
        "_user_profile_enabled": False,
    })()
    guidance = _tool_guidance_block(agent)
    assert guidance is not None
    assert "## TERMINAL CALL REQUIRED" in guidance


def test_interactive_session_does_not_get_the_block(monkeypatch):
    """Non-worker sessions (no owned task) must not see the worker-only block."""
    from contextlib import nullcontext

    from agent.delegation_context import non_dispatcher_owned_context
    from agent.prompt_builder import KANBAN_GUIDANCE
    from agent.system_prompt import _tool_guidance_block

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = type("A", (), {
        "valid_tool_names": {"kanban_show"},
        "_kanban_worker_guidance": "",
        "_memory_enabled": False,
        "_user_profile_enabled": False,
    })()
    with non_dispatcher_owned_context():
        assert _tool_guidance_block(agent) in (None, "")
    # Sanity: with ownership the same stub DOES get the block.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_owned")
    agent2 = type("A", (), {
        "valid_tool_names": {"kanban_show"},
        "_kanban_worker_guidance": KANBAN_GUIDANCE,
        "_memory_enabled": False,
        "_user_profile_enabled": False,
    })()
    with nullcontext():
        assert "## TERMINAL CALL REQUIRED" in (_tool_guidance_block(agent2) or "")


# ---------------------------------------------------------------------------
# Dispatch-time guard: stripped template refuses to spawn
# ---------------------------------------------------------------------------


def test_terminal_call_rule_present_detects_stripped_template():
    """The pure predicate: heading in template -> True; stripped -> False."""
    good = "# Kanban\n## TERMINAL CALL REQUIRED\nblah"
    stripped = "# Kanban\n## Lifecycle\nblah"
    assert kbd.terminal_call_rule_present(good) is True
    assert kbd.terminal_call_rule_present(stripped) is False
    assert kbd.terminal_call_rule_present("") is False
    # ``None`` loads the live KANBAN_GUIDANCE; with the block present it is True.
    assert kbd.terminal_call_rule_present(None) is True


def test_live_guidance_passes_the_guard():
    """The real KANBAN_GUIDANCE on this tree must pass the guard (regression)."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert kbd.terminal_call_rule_present(KANBAN_GUIDANCE) is True


def test_dispatch_refuses_to_spawn_when_guidance_stripped(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Fail-closed: a stripped KANBAN_GUIDANCE must not produce a worker.

    The card stays ``ready`` (never claimed), lands in
    ``result.guidance_stripped``, and gets a one-shot ``guidance_stripped``
    event so ``hermes kanban tail`` explains the stall.
    """
    from agent import prompt_builder

    monkeypatch.setattr(
        prompt_builder, "KANBAN_GUIDANCE",
        "# Kanban task execution protocol\nYou have been assigned ONE task.\n",
    )
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="would-spawn", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]

    assert spawns == [], "spawn_fn must not run against a stripped template"
    assert res.guidance_stripped == [tid]
    assert res.spawned == []
    assert task is not None and task.status == "ready"
    assert task.claim_lock in (None, ""), f"card was claimed: {task.claim_lock!r}"
    assert "guidance_stripped" in kinds


def test_dispatch_guard_event_is_written_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Repeated ticks against a stripped template must not append one event each."""
    from agent import prompt_builder

    monkeypatch.setattr(prompt_builder, "KANBAN_GUIDANCE", "no heading here")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="sticky", assignee="alice")
        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert kinds.count("guidance_stripped") == 1
    # Explicit cleanup in case future tests read from the same process env
    from agent import prompt_builder
    try:
        del prompt_builder.KANBAN_GUIDANCE
    except AttributeError:
        pass


def test_dispatch_dry_run_reports_stripped_without_writing(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """dry_run surfaces the refusal in the result but writes no event."""
    from agent import prompt_builder

    monkeypatch.setattr(prompt_builder, "KANBAN_GUIDANCE", "no heading here")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="dry", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 1, dry_run=True)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert res.guidance_stripped == [tid]
    assert res.spawned == []
    assert "guidance_stripped" not in kinds


def test_dispatch_spawns_normally_with_intact_guidance(
    kanban_home, all_assignees_spawnable,
):
    """Control: the real template spawns as before — the guard is not a brake."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fine", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]

    assert spawns == [tid]
    assert res.spawned and res.spawned[0][0] == tid
    assert res.guidance_stripped == []
    assert task is not None and task.status == "running"
    assert "guidance_stripped" not in kinds


def test_describe_suppression_counts_guidance_stripped():
    """Telemetry surface: a stripped template shows up in the one-line summary."""
    res = kbd.DispatchResult(guidance_stripped=["t_a", "t_b"])
    assert "guidance_stripped=2" in kbd.describe_suppression([res])
