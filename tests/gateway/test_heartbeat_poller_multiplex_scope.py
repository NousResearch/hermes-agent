"""Regression: the gateway-wide heartbeat poller must scope each watch entry
to the profile that owns it, in a multiplex (gateway.multiplex_profiles)
gateway.

_start_heartbeat_poller() creates exactly one asyncio.Task, the first time
any profile registers a heartbeat (idempotent — later registrations from
OTHER profiles reuse it). asyncio freezes contextvars (HERMES_HOME override,
secret scope) at task-creation time via copy_context(), so without
re-entering _profile_runtime_scope per watch entry inside the poll loop,
every profile OTHER than the one that happened to register first is polled
under the wrong HERMES_HOME: HeartbeatManager opens the wrong SessionDB,
has_heartbeat() reads back False, and the watch entry is silently popped —
indistinguishable from the user having cleared it themselves.

Tests are module-level functions rather than grouped in a class: nesting
these under a class allowed state to leak between tests in a way plain
functions don't reproduce (observed while writing this file).

Note on DB isolation: tests/conftest.py's autouse fixture repoints
hermes_state.DEFAULT_DB_PATH to one fixed per-test path so an argless
SessionDB() never touches a developer's real state.db. That's exactly right
for hygiene, but it means _default_db_path() no longer varies with
get_hermes_home() inside a test process — an argless SessionDB() (what
HeartbeatManager's _get_session_db() constructs on a cache miss) always
resolves to that one fixed file regardless of which profile is scoped in,
which would make two different "profile homes" in a naive test collapse
onto the same underlying database and silently defeat the very isolation
this test needs to observe. _seed_db_for() below sidesteps that by
pre-populating hermes_cli.goals._DB_CACHE (keyed exactly like
_get_session_db() keys it: str(get_hermes_home())) with an explicitly
pathed SessionDB per simulated home, so each "profile" genuinely gets its
own file — matching real multi-profile deployments, where HERMES_HOME
differs by process/profile and _default_db_path() is never pinned.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_cli import goals
from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
from hermes_state import SessionDB


class _FakeAdapter:
    pass


@pytest.fixture(autouse=True)
def _clear_db_cache():
    goals._DB_CACHE.clear()
    yield
    for db in goals._DB_CACHE.values():
        try:
            db.close()
        except Exception:
            pass
    goals._DB_CACHE.clear()


def _make_runner(*, multiplex: bool) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=multiplex)
    runner._running_agents = {}
    runner._heartbeat_watch = {}
    return runner


def _seed_db_for(home) -> None:
    """Register a real, distinctly-pathed SessionDB for ``home`` in the same
    cache _get_session_db() reads, so it's picked up instead of falling
    through to an argless SessionDB() (see module docstring)."""
    key = str(home)
    if key not in goals._DB_CACHE:
        goals._DB_CACHE[key] = SessionDB(db_path=home / "state.db")


def _seed_due_heartbeat(profile_home, session_id: str) -> None:
    _seed_db_for(profile_home)
    with _profile_runtime_scope(profile_home):
        save_heartbeat(
            session_id,
            HeartbeatState(
                prompt="check CI",
                interval_seconds=60,
                status="active",
                created_at=0.0,  # far in the past -> always due
            ),
        )


# --- Unit-level: the per-entry check itself only finds a profile's
# heartbeat under that profile's own HERMES_HOME scope. -----------------


@pytest.mark.asyncio
async def test_poll_entry_wrong_scope_misses_the_due_heartbeat(tmp_path, monkeypatch):
    ambient_home = tmp_path / "ambient"
    ambient_home.mkdir()
    profile_home = tmp_path / "profiles" / "planner"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))
    _seed_db_for(ambient_home)

    session_id = "sess-planner-hb"
    _seed_due_heartbeat(profile_home, session_id)

    runner = _make_runner(multiplex=True)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", profile="planner",
    )
    runner._heartbeat_watch["qk"] = (source, session_id)

    fired = []
    runner._adapter_for_source = lambda src: _FakeAdapter()
    runner._enqueue_fifo = lambda qk, event, adapter: fired.append((qk, event.text))

    # No scope entered — this is what the pre-fix _poll_loop did for every
    # entry, regardless of which profile owns it.
    await runner._poll_heartbeat_watch_entry("qk", source, session_id)

    assert fired == []
    assert "qk" not in runner._heartbeat_watch, (
        "has_heartbeat() reading False under the wrong home must not "
        "silently pop a still-active watch entry"
    )


@pytest.mark.asyncio
async def test_poll_entry_correct_scope_fires_the_due_heartbeat(tmp_path, monkeypatch):
    ambient_home = tmp_path / "ambient"
    ambient_home.mkdir()
    profile_home = tmp_path / "profiles" / "planner"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))
    _seed_db_for(ambient_home)

    session_id = "sess-planner-hb"
    _seed_due_heartbeat(profile_home, session_id)

    runner = _make_runner(multiplex=True)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", profile="planner",
    )
    runner._heartbeat_watch["qk"] = (source, session_id)

    fired = []
    runner._adapter_for_source = lambda src: _FakeAdapter()
    runner._enqueue_fifo = lambda qk, event, adapter: fired.append((qk, event.text))

    with _profile_runtime_scope(profile_home):
        await runner._poll_heartbeat_watch_entry("qk", source, session_id)

    assert len(fired) == 1
    assert fired[0][0] == "qk"
    assert "check CI" in fired[0][1]
    assert "qk" in runner._heartbeat_watch


# --- End-to-end: _start_heartbeat_poller's actual _poll_loop must enter
# _profile_runtime_scope per entry using _resolve_profile_home_for_source,
# not rely on whatever context was ambient when the task was created. ---


@pytest.mark.asyncio
async def test_poller_task_created_under_one_profile_still_serves_another(tmp_path, monkeypatch):
    ambient_home = tmp_path / "ambient"
    ambient_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))
    _seed_db_for(ambient_home)

    # get_profile_dir()/_resolve_profile_home_for_source resolve against the
    # un-overridden base home (HERMES_HOME), so this must live under
    # ambient_home/profiles/<name> for the real (unmocked) resolution path
    # to find it.
    profile_home = ambient_home / "profiles" / "planner"
    profile_home.mkdir(parents=True)

    session_id = "sess-planner-hb"
    _seed_due_heartbeat(profile_home, session_id)

    runner = _make_runner(multiplex=True)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", profile="planner",
    )
    runner._heartbeat_watch["qk"] = (source, session_id)
    runner._background_tasks = set()

    fired = []
    runner._adapter_for_source = lambda src: _FakeAdapter()
    runner._enqueue_fifo = lambda qk, event, adapter: fired.append((qk, event.text))

    # Speed up the poll cadence so the test doesn't wait POLL_SECONDS (5s)
    # for the first real tick.
    monkeypatch.setattr("hermes_cli.heartbeat.POLL_SECONDS", 0.01)

    # The task is created here, at ambient scope (no _profile_runtime_scope
    # active) — exactly like production, where _start_heartbeat_poller is
    # idempotent and only the FIRST profile to register ever creates it.
    runner._start_heartbeat_poller()
    try:
        for _ in range(50):
            if fired:
                break
            await asyncio.sleep(0.02)
    finally:
        runner._heartbeat_poll_task.cancel()
        try:
            await runner._heartbeat_poll_task
        except asyncio.CancelledError:
            pass

    assert fired, (
        "the poll loop must enter the owning profile's scope per entry, "
        "not inherit whatever context was ambient at task-creation time"
    )
    assert fired[0][0] == "qk"
