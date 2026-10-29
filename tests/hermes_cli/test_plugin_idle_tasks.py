"""ctx.register_idle_task: the curator's idle scheduler, for plugins."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_idle_tasks import cmd_idle_tasks, load_state, run_plugin_idle_tasks

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
IDLE = 3 * 3600.0


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _ctx(name="wisdom"):
    manager = PluginManager()
    return manager, PluginContext(PluginManifest(name=name, source="user"), manager)


def _poll(manager, now, idle=IDLE):
    return run_plugin_idle_tasks(idle_for_seconds=idle, manager=manager, now=now)


def _recorder(calls, result="checked 3 skills"):
    def fn(run):
        calls.append(run)
        return result
    return fn


def test_first_poll_seeds_and_defers_one_interval_then_runs_on_the_interval(home):
    manager, ctx = _ctx()
    calls = []
    ctx.register_idle_task("suggest", _recorder(calls), interval_hours=24)

    assert _poll(manager, T0) == {}
    state = load_state(ctx.state.data_dir / "idle-tasks" / "suggest.json")
    assert state["last_run_at"] == T0.isoformat() and state["run_count"] == 0 and not calls

    assert _poll(manager, T0 + timedelta(hours=23)) == {}
    assert _poll(manager, T0 + timedelta(hours=24)) == {"wisdom:suggest": "ok"}
    assert len(calls) == 1
    run = calls[0]
    assert (run.name, run.idle_for_seconds, run.last_run_at, run.run_count) == ("suggest", IDLE, T0.isoformat(), 0)

    state = load_state(ctx.state.data_dir / "idle-tasks" / "suggest.json")
    assert (state["run_count"], state["last_status"], state["last_summary"]) == (1, "ok", "checked 3 skills")
    assert _poll(manager, T0 + timedelta(hours=25)) == {}


def test_immediate_first_run_and_min_idle_gate(home):
    manager, ctx = _ctx()
    calls = []
    ctx.register_idle_task("suggest", _recorder(calls), interval_hours=24, min_idle_hours=2, first_run="immediate")

    assert _poll(manager, T0, idle=3600.0) == {}  # host active one hour ago: not idle long enough
    assert _poll(manager, T0) == {"wisdom:suggest": "ok"}
    assert calls[0].last_run_at is None


def test_registration_validates_and_floors_the_interval(home):
    manager, ctx = _ctx()
    for bad in ({"name": "Bad Name"}, {"first_run": "now"}, {"fn": "not callable"}, {"interval_hours": "weekly"},
                {"interval_hours": float("inf")}, {"interval_hours": float("nan")},
                {"min_idle_hours": float("inf")}, {"min_idle_hours": float("nan")}):
        kwargs = {"name": "ok", "fn": _recorder([]), "interval_hours": 24} | bad
        with pytest.raises(ValueError):
            ctx.register_idle_task(kwargs.pop("name"), kwargs.pop("fn"), **kwargs)

    calls = []
    ctx.register_idle_task("tight", _recorder(calls), interval_hours=0.001, first_run="immediate")
    assert _poll(manager, T0) == {"wisdom:tight": "ok"}
    assert _poll(manager, T0 + timedelta(minutes=59)) == {}  # floored to one hour
    assert _poll(manager, T0 + timedelta(hours=1)) == {"wisdom:tight": "ok"}


def test_pause_and_resume_from_the_cli(home, capsys):
    manager, ctx = _ctx()
    calls = []
    ctx.register_idle_task("suggest", _recorder(calls), interval_hours=1, first_run="immediate")
    _poll(manager, T0)

    cmd_idle_tasks("wisdom", pause="suggest")
    assert _poll(manager, T0 + timedelta(hours=5)) == {}
    cmd_idle_tasks()
    assert "wisdom:suggest  paused  runs=1" in capsys.readouterr().out

    cmd_idle_tasks("wisdom", resume="suggest")
    assert _poll(manager, T0 + timedelta(hours=5)) == {"wisdom:suggest": "ok"}
    with pytest.raises(SystemExit):
        cmd_idle_tasks("wisdom", pause="missing")


def test_a_run_never_overlaps_itself(home):
    manager, ctx = _ctx()
    nested = []

    def fn(run):
        nested.append(_poll(manager, T0 + timedelta(hours=5)))  # a second poller while this run holds the claim

    ctx.register_idle_task("suggest", fn, interval_hours=1, first_run="immediate")
    assert _poll(manager, T0) == {"wisdom:suggest": "ok"}
    assert nested == [{}]

    lock = ctx.state.data_dir / "idle-tasks" / ".suggest.lock"
    lock.write_text("4242")  # another process holds it
    assert _poll(manager, T0 + timedelta(hours=5)) == {}
    lock.unlink()
    assert _poll(manager, T0 + timedelta(hours=5)) == {"wisdom:suggest": "ok"}


def test_a_failing_task_is_contained_and_recorded(home):
    manager, ctx = _ctx()
    other = []

    def boom(run):
        raise RuntimeError("model unreachable")

    ctx.register_idle_task("boom", boom, interval_hours=1, first_run="immediate")
    ctx.register_idle_task("fine", _recorder(other), interval_hours=1, first_run="immediate")

    assert _poll(manager, T0) == {"wisdom:boom": "error", "wisdom:fine": "ok"}
    state = load_state(ctx.state.data_dir / "idle-tasks" / "boom.json")
    assert state["last_status"] == "error" and "model unreachable" in state["last_error"]
    assert state["run_count"] == 1 and len(other) == 1


def test_profile_scoping_and_disposal(home, tmp_path):
    manager, ctx = _ctx()
    calls = []
    registration = ctx.register_idle_task("suggest", _recorder(calls), interval_hours=1, first_run="immediate")

    elsewhere = PluginManager(scope_key=str(tmp_path / "profiles" / "coder"))
    PluginContext(PluginManifest(name="wisdom", source="user"), elsewhere).register_idle_task(
        "suggest", _recorder(calls), interval_hours=1, first_run="immediate")
    assert _poll(elsewhere, T0) == {}  # another profile's manager runs nothing under this home
    assert not calls

    assert _poll(manager, T0) == {"wisdom:suggest": "ok"}
    assert (home / "plugin-data").is_dir() and ctx.state.data_dir.is_relative_to(home)

    registration.dispose()
    assert _poll(manager, T0 + timedelta(hours=5)) == {}


def test_the_gateway_idle_poll_runs_the_curator_and_the_plugin_tasks(home, monkeypatch):
    import agent.curator
    from gateway import run as gateway_run
    from hermes_cli import plugins_idle_tasks

    seen = []
    monkeypatch.setattr(agent.curator, "maybe_run_curator", lambda **kw: seen.append(("curator", kw["idle_for_seconds"])))
    monkeypatch.setattr(plugins_idle_tasks, "run_plugin_idle_tasks", lambda **kw: seen.append(("plugins", kw["idle_for_seconds"])))

    gateway_run._housekeeping_curator()
    assert seen == [("curator", float("inf")), ("plugins", float("inf"))]
