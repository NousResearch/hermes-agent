"""Model-lane back-pressure guard for the kanban dispatcher.

The concurrency caps bound WORKERS, but nothing bounded the inference lane
those workers feed. Where one local model serves requests serially (oMLX
``scheduler.max_concurrent_requests: 1``), an extra card cannot raise
throughput — it only deepens the queue, and a turn stuck behind a long prefill
is likelier to be interrupted (``rc=130 waiting for model response``) than to
finish.

Covers the guard added to ``_tick_spawn_budget``: at/over
``kanban.lane_queue_limit`` queued requests, spawns are held for the tick
(deferred, not dropped), and an unreadable lane fails open.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _lane(monkeypatch, *, limit, depth):
    """Pin the guard's config and lane reading for one test."""
    monkeypatch.setattr(kbd, "configured_lane_queue_limit", lambda: limit)
    monkeypatch.setattr(kbd, "_lane_queue_depth", lambda *a, **k: depth)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self._payload


def _stub_config(monkeypatch, cfg):
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config_readonly", lambda *a, **k: cfg)


# ---------------------------------------------------------------------------
# lane status URL / queue depth
# ---------------------------------------------------------------------------


def test_lane_status_url_derives_from_loopback_base_url(monkeypatch):
    _stub_config(monkeypatch, {"model": {"base_url": "http://127.0.0.1:8080/v1"}})
    assert kbd._lane_status_url() == "http://127.0.0.1:8080/api/status"


def test_lane_status_url_skips_hosted_base_url(monkeypatch):
    """A hosted provider must never be probed once per dispatch tick."""
    _stub_config(monkeypatch, {"model": {"base_url": "https://api.openai.com/v1"}})
    assert kbd._lane_status_url() is None


def test_lane_status_url_explicit_override_wins(monkeypatch):
    _stub_config(
        monkeypatch,
        {
            "kanban": {"lane_status_url": "http://127.0.0.1:9999/x"},
            "model": {"base_url": "https://api.openai.com/v1"},
        },
    )
    assert kbd._lane_status_url() == "http://127.0.0.1:9999/x"


def test_lane_queue_depth_reads_waiting_requests(monkeypatch):
    monkeypatch.setattr(
        kbd.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b'{"waiting_requests": 4}'),
    )
    assert kbd._lane_queue_depth("http://127.0.0.1:8080/api/status") == 4


def test_lane_queue_depth_unknown_on_malformed_payload(monkeypatch):
    monkeypatch.setattr(
        kbd.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse(b'{"waiting_requests": "4"}'),
    )
    assert kbd._lane_queue_depth("http://127.0.0.1:8080/api/status") is None


def test_lane_queue_depth_unknown_when_unreachable(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(kbd.urllib.request, "urlopen", boom)
    assert kbd._lane_queue_depth("http://127.0.0.1:8080/api/status") is None


# ---------------------------------------------------------------------------
# dispatch_once under a saturated lane
# ---------------------------------------------------------------------------


def test_lane_guard_holds_spawns_when_queue_at_limit(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    _lane(monkeypatch, limit=2, depth=2)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert not res.spawned
    assert res.lane_queue_depth == 2


def test_lane_guard_defers_not_drops(kanban_home, all_assignees_spawnable, monkeypatch):
    """A held card stays 'ready' and spawns once the lane drains."""
    depth = {"value": 5}
    monkeypatch.setattr(kbd, "configured_lane_queue_limit", lambda: 2)
    monkeypatch.setattr(kbd, "_lane_queue_depth", lambda *a, **k: depth["value"])
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert not spawns
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        depth["value"] = 0
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [task]
    assert res.lane_queue_depth is None


def test_lane_guard_fails_open_when_lane_unknown(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    _lane(monkeypatch, limit=1, depth=None)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1


def test_lane_guard_disabled_when_unset(kanban_home, all_assignees_spawnable, monkeypatch):
    """Unset limit spawns normally and never probes the lane."""
    def unexpected(*a, **k):
        raise AssertionError("lane must not be probed when the guard is unset")

    monkeypatch.setattr(kbd, "configured_lane_queue_limit", lambda: None)
    monkeypatch.setattr(kbd, "_lane_queue_depth", unexpected)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
