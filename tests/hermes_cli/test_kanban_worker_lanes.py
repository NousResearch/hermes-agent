"""Plugin Kanban worker lanes: routing by assignee/board/predicate, fallback, lane kind, the
in-process lifecycle and multi-profile visibility, on real boards through ``dispatch_once``.

Refs #70547, #146497, #125759.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_lane_registry as registry
from hermes_cli.kanban_worker_lanes import NOT_HANDLED, KanbanWorkerLane
from hermes_constants import get_hermes_home, hermes_home_key


class _Lane(KanbanWorkerLane):
    def __init__(self, name="bridge", *, assignees=(), boards=(), accept=True, result=None):
        self._name, self.assignees, self.boards = name, tuple(assignees), tuple(boards)
        self._accept, self._result = accept, result
        self.claims, self.homes = [], []

    @property
    def name(self):
        return self._name

    def accepts(self, task, *, board, lane):
        return self._accept(task) if callable(self._accept) else self._accept

    def spawn(self, claim):
        self.claims.append(claim)
        self.homes.append(get_hermes_home())
        return self._result


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "alice")
    kb.init_db()
    kb.create_board("ops")
    registry._reset_for_tests()
    yield home
    registry._reset_for_tests()


def _register(lane, scope=None):
    registry.register_provider(lane, scope=scope or hermes_home_key())
    return lane


def _native():
    calls = []

    def spawn(task, workspace, *, board=None):
        calls.append((task.id, board))
    return spawn, calls


def _dispatch(board, **kwargs):
    with kbc.connect_closing(board=board) as conn:
        return kbd.dispatch_once(conn, board=board, **kwargs)


def _create(board, **kwargs):
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title="t", **kwargs)


def _task(board, task_id):
    with kbc.connect_closing(board=board) as conn:
        return kb.get_task(conn, task_id)


def test_lane_scoped_to_a_board_leaves_the_profile_native_elsewhere(board_home):
    lane = _register(_Lane(assignees=("alice",), boards=("ops",)))
    native, calls = _native()
    ops_task = _create("ops", assignee="alice")
    default_task = _create("default", assignee="alice")

    _dispatch("ops", spawn_fn=native)
    _dispatch("default", spawn_fn=native)

    assert [c.task_id for c in lane.claims] == [ops_task]
    assert calls == [(default_task, "default")]


def test_non_profile_assignee_runs_only_through_a_matching_lane(board_home):
    native, calls = _native()
    task_id = _create("default", assignee="codex-1")
    assert _dispatch("default", spawn_fn=native).skipped_nonspawnable == [task_id]

    lane = _register(_Lane(assignees=("codex-*",)))
    with kbc.connect_closing(board="default") as conn:
        assert kbd.has_spawnable_ready(conn, board="default")
    result = _dispatch("default", spawn_fn=native)

    assert [c.task_id for c in lane.claims] == [task_id]
    assert result.spawned and calls == []


def test_accepts_predicate_filters_before_the_claim(board_home):
    _register(_Lane(assignees=("codex-*",), accept=lambda task: task.tenant == "acme"))
    native, _calls = _native()
    other = _create("default", assignee="codex-1", tenant="other")

    result = _dispatch("default", spawn_fn=native)

    assert result.skipped_nonspawnable == [other]
    assert _task("default", other).status == "ready"


def test_a_declined_spawn_falls_back_to_the_native_spawn(board_home):
    lane = _register(_Lane(assignees=("alice",), result=NOT_HANDLED))
    native, calls = _native()
    task_id = _create("default", assignee="alice")

    _dispatch("default", spawn_fn=native)

    assert [c.task_id for c in lane.claims] == [task_id]
    assert calls == [(task_id, "default")]


def test_the_claim_carries_the_lane_kind_and_run_identity(board_home):
    lane = _register(_Lane(assignees=("alice",)))
    task_id = _create("default", assignee="alice")
    _dispatch("default")
    first = lane.claims[0]
    assert (first.lane, first.board, first.run_id) == ("ready", "default", _task("default", task_id).current_run_id)
    assert first.claim_lock and first.workspace

    assert first.request_review("ready for review")
    _dispatch("default")

    review = lane.claims[1]
    assert review.lane == "review" and review.run_id != first.run_id


def test_in_process_lane_reports_through_its_claim(board_home):
    lane = _register(_Lane(assignees=("alice",)))
    done_id = _create("default", assignee="alice")
    failed_id = _create("default", assignee="alice")
    _dispatch("default")
    done, failed = sorted(lane.claims, key=lambda c: c.task_id != done_id)

    with kbc.connect_closing(board="default") as conn:
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) + 5, done_id))
    assert done.heartbeat("halfway")
    assert _task("default", done_id).claim_expires > int(time.time()) + 60
    assert done.complete("finished")
    assert _task("default", done_id).status == "done"
    assert not done.heartbeat()

    assert failed.fail("turn crashed")
    after = _task("default", failed_id)
    assert (after.status, after.consecutive_failures, after.claim_lock) == ("ready", 1, None)
    assert not failed.fail("again")


def test_a_lane_run_that_stops_heartbeating_is_reclaimed_by_the_claim_ttl(board_home):
    lane = _register(_Lane(assignees=("alice",)))
    task_id = _create("default", assignee="alice")
    _dispatch("default", ttl_seconds=1)
    claim = lane.claims[0]

    with kbc.connect_closing(board="default") as conn:
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 5, task_id))
        assert kb.release_stale_claims(conn) == 1

    assert _task("default", task_id).status == "ready"
    assert not claim.heartbeat()
    assert not claim.complete("too late")


def test_lanes_of_every_served_profile_run_under_their_own_home(board_home, tmp_path):
    second_home = tmp_path / "profiles" / "second"
    second_home.mkdir(parents=True)
    lane = _register(_Lane(assignees=("codex-*",)), scope=hermes_home_key(second_home))
    task_id = _create("default", assignee="codex-1")

    _dispatch("default")

    assert [c.task_id for c in lane.claims] == [task_id]
    assert lane.homes == [second_home]


def test_without_lanes_the_dispatch_is_unchanged(board_home):
    native, calls = _native()
    task_id = _create("default", assignee="alice")
    result = _dispatch("default", spawn_fn=native)
    assert calls == [(task_id, "default")] and [s[0] for s in result.spawned] == [task_id]


_PLUGIN_LANE = (
    "from hermes_cli.kanban_worker_lanes import KanbanWorkerLane\n"
    "    class Lane(KanbanWorkerLane):\n"
    "        assignees = {assignees!r}\n"
    "        name = 'plugin-lane'\n"
    "        def spawn(self, claim): return None\n"
    "    ctx.register_kanban_worker_lane(Lane())"
)


def _load_plugin(home, assignees):
    import hermes_yaml as yaml
    from hermes_cli.plugins import PluginManager
    plugin_dir = home / "plugins" / "lane-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "lane-plugin", "version": "0.1.0"}))
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n    " + _PLUGIN_LANE.format(assignees=assignees) + "\n")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["lane-plugin"]}}))
    manager = PluginManager()
    manager.discover_and_load()
    assert manager._plugins["lane-plugin"].enabled, manager._plugins["lane-plugin"].error
    return manager


def test_a_plugin_registers_a_lane_through_its_context(board_home):
    _load_plugin(board_home, ("codex-*",))
    assert [lane.name for _scope, lane in registry.matching_lanes("codex-1", "default")] == ["plugin-lane"]
    assert registry.matching_lanes("alice", "default") == []


def test_a_lane_that_names_no_assignee_or_board_is_refused(board_home):
    _load_plugin(board_home, ())
    assert registry.matching_lanes("codex-1", "default") == []
    lane = _Lane(boards=("ops",))
    lane.assignees = "alice"
    with pytest.raises(ValueError):
        registry.check_lane(lane)
