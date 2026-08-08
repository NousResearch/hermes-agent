"""_touch_activity's live comment-injection bridge must not run for an
in-process execution that merely inherited a kanban worker's
HERMES_KANBAN_TASK env var (e.g. a cron job fired via cronjob(action="run")
inside a worker's own process).

Sibling of 80f37e36e (#79657/#78961), which gated kanban_tools' own
toolset/task-id decisions on is_dispatcher_owned_worker_context() so a cron
agent sharing a worker's process isn't misidentified as that worker.
_touch_activity's call to inject_new_comments_from_env(self) was missed:
unlike heartbeat_current_worker_from_env() (which only extends the claim TTL
for whichever task id env identifies -- safe regardless of caller identity),
inject_new_comments_from_env steers ``self``, i.e. the CALLER's own live
conversation. An unrelated cron agent sharing the worker's process would
otherwise pull the worker's operator comments into its own turn and, since
the watermark is keyed by task id rather than by agent, silently consume the
notification the real worker was supposed to receive.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

import run_agent
from agent.session_activity import ActivityProvenance
from hermes_cli import kanban_db as kb
import tools.kanban_tools as kt


@pytest.fixture
def worker_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    kt._comment_watermark.clear()
    kt._comment_poll_last_attempt = 0.0
    yield home
    kt._comment_watermark.clear()
    kt._comment_poll_last_attempt = 0.0


def _agent_with_steer(session_id: str = "sess-1"):
    """Minimal AIAgent-like stub exercising the real _touch_activity method."""
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=None,
        _last_activity_ts=0.0,
        _last_activity_desc="",
        _last_activity_provenance=ActivityProvenance.UNKNOWN,
        _session_activity_last_persist_mono=0.0,
        _current_tool=None,
        _api_call_count=0,
        max_iterations=10,
        iteration_budget=SimpleNamespace(used=0, max_total=10),
        steers=[],
    )
    agent.steer = lambda text: (agent.steers.append(text), True)[1]
    agent._touch_activity = run_agent.AIAgent._touch_activity.__get__(agent, SimpleNamespace)
    agent._persist_session_activity_if_due = (
        run_agent.AIAgent._persist_session_activity_if_due.__get__(agent, SimpleNamespace)
    )
    return agent


def _seed_task_with_pending_comment(monkeypatch) -> str:
    """Create a kanban task with a comment already waiting, and set
    HERMES_KANBAN_TASK to it. Returns the task id."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="live task")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")

    # Seed the watermark past the empty thread (mirrors inject_new_comments_
    # from_env's own "first poll only seeds" contract), then add the comment
    # that a subsequent poll should find.
    kt.inject_new_comments_from_env(_agent_with_steer())
    kt._comment_poll_last_attempt = 0.0

    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="desktop", body="actually use the v2 API")
    finally:
        conn.close()
    kt._comment_poll_last_attempt = 0.0
    return tid


def test_dispatcher_owned_worker_still_gets_live_comments(worker_home, monkeypatch):
    """Control: a real dispatcher-spawned worker (the common case) must keep
    receiving live operator notes through _touch_activity."""
    _seed_task_with_pending_comment(monkeypatch)
    agent = _agent_with_steer()

    agent._touch_activity("running a tool")

    assert len(agent.steers) == 1
    assert "v2 API" in agent.steers[0]


def test_non_dispatcher_owned_context_does_not_steer_the_caller(worker_home, monkeypatch):
    """A cron job fired in-process from a kanban worker (HERMES_KANBAN_TASK
    legitimately still in os.environ, but this execution does not own that
    task) must not have the worker's operator comments injected into its own
    conversation."""
    from agent.delegation_context import non_dispatcher_owned_context

    _seed_task_with_pending_comment(monkeypatch)
    agent = _agent_with_steer()

    with non_dispatcher_owned_context():
        agent._touch_activity("running a tool")

    assert agent.steers == [], (
        "an unrelated cron agent sharing the worker's process must not "
        "receive the worker's operator comments as a steer into its own turn"
    )


def test_non_dispatcher_owned_context_does_not_consume_the_watermark(worker_home, monkeypatch):
    """A skipped injection must not advance the poll watermark, or the real
    worker's next _touch_activity would silently miss the comment the cron
    job's skipped call would otherwise have consumed."""
    from agent.delegation_context import non_dispatcher_owned_context

    _seed_task_with_pending_comment(monkeypatch)
    cron_agent = _agent_with_steer()

    with non_dispatcher_owned_context():
        cron_agent._touch_activity("running a tool")
    assert cron_agent.steers == []

    kt._comment_poll_last_attempt = 0.0
    worker_agent = _agent_with_steer()
    worker_agent._touch_activity("running a tool")

    assert len(worker_agent.steers) == 1
    assert "v2 API" in worker_agent.steers[0]


def test_delegated_child_context_also_does_not_steer_the_caller(worker_home, monkeypatch):
    """delegate_task children share the same is_dispatcher_owned_worker_context
    gate; a delegated child must not receive the worker's live comments."""
    from agent.delegation_context import delegated_child_context

    _seed_task_with_pending_comment(monkeypatch)
    agent = _agent_with_steer()

    with delegated_child_context():
        agent._touch_activity("running a tool")

    assert agent.steers == []
