"""Values apply at process start; a change restarts the process.

The remote store installs the plane's view once. Its own writes apply in-process as a local save
does and never restart it; a change made anywhere else (an editor, another process, a revocation),
a write conflict, or a write reply that carries someone else's change asks for a graceful restart
and changes nothing in the running process. A failed poll keeps it running.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

import pytest

import hermes_secret_routing as routing
from tests.waiting import wait_until

from .stub_secrets_plane import INSTANCE

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

MARK = "restart-marker-91c4"


def _boot(home):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)


def _poll():
    routing.routed_provider().poll_all()


def _readers(home, name):
    """What the running process serves for *name*: os.environ, the profile read and a scope."""
    from agent.secret_scope import build_profile_secret_scope
    return {"environ": os.environ.get(name), "profile": routing.read_profile_env(home / ".env").get(name),
            "scope": build_profile_secret_scope(home).get(name)}


def test_an_own_write_applies_like_a_local_save_and_never_restarts(plane, monkeypatch):
    monkeypatch.delenv("OWN_KEY", raising=False)
    _boot(plane.home)
    from hermes_cli.config import remove_env_value, save_env_value
    save_env_value("OWN_KEY", MARK)
    assert set(_readers(plane.home, "OWN_KEY").values()) == {MARK}
    _poll()
    assert plane.restarts == []
    assert plane.requests[-1]["method"] == "GET" and plane.requests[-1]["if_none_match"]  # a 304
    assert remove_env_value("OWN_KEY") is True
    assert set(_readers(plane.home, "OWN_KEY").values()) == {None}
    from hermes_cli import auth
    with auth._auth_store_lock():
        store = auth._load_auth_store()
        store["providers"]["openai-codex"] = {"tokens": {"access_token": MARK}}
        auth._save_auth_store(store)
    _poll()
    assert plane.restarts == []
    assert len(plane.patches()) == 3


def test_a_change_made_elsewhere_restarts_and_changes_nothing_in_the_running_process(plane, monkeypatch, caplog):
    monkeypatch.delenv("ROTATED_KEY", raising=False)
    plane.set_level(plane.instances[INSTANCE]["level"], env={"ROTATED_KEY": "boot-value"})
    _boot(plane.home)
    _poll()
    assert plane.restarts == []
    plane.set_level(plane.instances[INSTANCE]["level"], env={"ROTATED_KEY": MARK})
    _poll()
    assert plane.restarts == ["restart"]
    assert set(_readers(plane.home, "ROTATED_KEY").values()) == {"boot-value"}
    _boot(plane.home)  # an explicit reload still loads the view the process started with
    assert set(_readers(plane.home, "ROTATED_KEY").values()) == {"boot-value"}
    assert "ROTATED_KEY" in caplog.text and MARK not in caplog.text
    gets = len([r for r in plane.requests if r["method"] == "GET"])
    _poll()  # asked once: the poller stops asking the plane
    assert plane.restarts == ["restart"] and len([r for r in plane.requests if r["method"] == "GET"]) == gets


def test_a_revocation_elsewhere_restarts(plane, monkeypatch):
    monkeypatch.delenv("REVOKED_KEY", raising=False)
    plane.set_level(plane.profile("default"), env={"REVOKED_KEY": MARK})
    _boot(plane.home)
    plane.set_level(plane.profile("default"), unset=("REVOKED_KEY",))
    _poll()
    assert plane.restarts == ["restart"]
    assert os.environ["REVOKED_KEY"] == MARK  # until the restart: nothing is withdrawn live


def test_a_write_conflict_restarts_instead_of_merging(plane, caplog):
    _boot(plane.home)
    from hermes_cli.config import save_env_value
    from hermes_secret_port import SecretWriteFailed
    plane.conflict_next_patch = 1   # another writer changed the profile level since the boot
    with pytest.raises(SecretWriteFailed, match="secret_version_conflict") as failed:
        save_env_value("CONFLICT_KEY", MARK)
    assert MARK not in str(failed.value) and MARK not in caplog.text
    assert plane.restarts == ["restart"] and len(plane.patches()) == 1
    assert "CONFLICT_KEY" not in plane.profile("default")["env"]


def test_an_own_write_whose_reply_carries_another_writers_change_restarts(plane, monkeypatch, caplog):
    monkeypatch.delenv("TENANT_KEY", raising=False)
    monkeypatch.delenv("WRITTEN_KEY", raising=False)
    _boot(plane.home)
    plane.set_level(plane.tenant, env={"TENANT_KEY": MARK})  # no profile-level conflict
    from hermes_cli.config import save_env_value
    save_env_value("WRITTEN_KEY", "written")
    assert plane.profile("default")["env"]["WRITTEN_KEY"][0] == "written"
    assert plane.restarts == ["restart"]
    assert set(_readers(plane.home, "WRITTEN_KEY").values()) == {"written"}   # the process's own write
    _boot(plane.home)
    assert set(_readers(plane.home, "TENANT_KEY").values()) == {None}         # the other one waits
    assert "TENANT_KEY" in caplog.text and "WRITTEN_KEY" not in caplog.text.split("changed elsewhere")[-1]
    assert MARK not in caplog.text


def test_a_failed_poll_keeps_the_process_running_on_its_view(plane, monkeypatch, caplog):
    monkeypatch.delenv("TENANT_KEY", raising=False)
    plane.set_level(plane.tenant, env={"TENANT_KEY": MARK})
    _boot(plane.home)
    for status in (503, 401):
        plane.fail_status = status
        _poll()
    assert plane.restarts == [] and os.environ["TENANT_KEY"] == MARK
    assert "last poll failed" in routing.routed_provider().describe(plane.home)
    plane.fail_status = None
    _poll()
    assert plane.restarts == [] and MARK not in caplog.text
    assert "last poll failed" not in routing.routed_provider().describe(plane.home)


def test_multiplexed_scopes_are_built_from_the_view_each_profile_booted_with(plane, monkeypatch):
    from agent.secret_scope import build_profile_secret_scope, get_secret, set_secret_scope
    work = plane.home.parent / "work"
    work.mkdir()
    plane.set_level(plane.profile("work"), env={"WORK_KEY": "first-bind"})
    _boot(plane.home)
    scope = build_profile_secret_scope(work)   # the satellite's first bind fetches its view
    plane.set_level(plane.profile("work"), env={"WORK_KEY": MARK})
    _poll()
    assert plane.restarts == ["restart"]       # a change to any profile restarts the process
    with set_secret_scope(build_profile_secret_scope(work), profile_home=work):
        assert get_secret("WORK_KEY") == "first-bind"
    assert scope["WORK_KEY"] == "first-bind" and "WORK_KEY" not in os.environ


def test_a_forked_child_re_arms_its_poller(plane):
    _boot(plane.home)
    provider = routing.routed_provider()
    provider._poller_pid = -1   # as a forked child sees its parent's (absent) poller thread
    _boot(plane.home)
    assert provider._poller_pid == os.getpid() and provider._poller.is_alive()


def test_without_a_restart_mechanism_the_change_waits_for_the_next_start(plane, caplog):
    routing.set_restart_handler(None)   # an interactive CLI registers none
    plane.set_level(plane.profile("default"), env={"CLI_KEY": MARK})
    _boot(plane.home)
    plane.set_level(plane.profile("default"), env={"CLI_KEY": "changed"})
    with caplog.at_level(logging.WARNING):
        _poll()
    assert "they apply when Hermes next starts" in caplog.text and "CLI_KEY" in caplog.text
    assert os.environ["CLI_KEY"] == MARK


class _Runner:
    """A gateway runner as the restart path sees it: the real ``request_restart`` and idle wait;
    only ``stop`` (the final exit) is replaced."""

    def __init__(self):
        self._restart_task_started = False
        self._draining = False
        self.stopped: list[dict] = []

    def request_restart(self, *, detached=False, via_service=False):
        from gateway.restart_when_idle import request_restart
        return request_restart(self, detached=detached, via_service=via_service)

    def _awaitable_work_count(self):
        return 0

    def _describe_active_work(self):
        return []

    def _scale_to_zero_status(self, *_args):
        return None

    async def _launch_detached_restart_command(self):
        self.stopped.append({"detached_helper": True})

    async def stop(self, **kwargs):
        self.stopped.append(kwargs)


@pytest.mark.parametrize("supervised", [True, False], ids=["supervised", "unsupervised"])
def test_a_change_elsewhere_takes_the_gateways_planned_restart(plane, monkeypatch, supervised):
    import gateway.restart as restart_mod
    from gateway.restart_when_idle import install_secret_store_restart
    monkeypatch.setattr(restart_mod, "is_gateway_supervisor_process", lambda: supervised)
    monkeypatch.setattr(restart_mod, "is_container_restart_context", lambda: False)
    plane.set_level(plane.profile("default"), env={"GATEWAY_KEY": "boot-value"})
    _boot(plane.home)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = _Runner()
    try:
        install_secret_store_restart(loop, runner)
        plane.set_level(plane.profile("default"), env={"GATEWAY_KEY": MARK})
        _poll()   # the poller's own thread in production: marshalled onto the loop
        wait_until(lambda: any("restart" in s for s in runner.stopped), timeout=10)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    stop = next(s for s in runner.stopped if "restart" in s)
    assert stop["restart"] is True and stop["service_restart"] is supervised
    assert stop["detached_restart"] is (not supervised)
    assert ({"detached_helper": True} in runner.stopped) is (not supervised)


def test_the_gateway_installs_its_planned_restart_at_start():
    """``start_gateway`` registers the restart beside its signal handlers (the line this pins is the
    only caller; without it a running gateway would only log that a change waits for a restart)."""
    import ast
    import inspect

    import gateway.run_bootstrap as bootstrap
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(bootstrap)))
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "install_secret_store_restart"]
    assert [[ast.unparse(a) for a in call.args] for call in calls] == [["loop", "runner"]]


def test_a_restart_asked_for_before_any_handler_is_replayed_once_on_registration():
    """A gateway's poller starts before its runner registers the restart: the request waits."""
    routing.reset_for_tests()
    routing.set_restart_handler(None)
    calls = []
    try:
        assert routing.request_restart("profile 'default' changed elsewhere") is False
        routing.set_restart_handler(lambda: calls.append("first"))
        assert calls == ["first"]
        routing.set_restart_handler(lambda: calls.append("second"))   # taken once, not per handler
        assert calls == ["first"]
        assert routing.request_restart("again") is True and calls == ["first"]   # already restarting
    finally:
        routing.set_restart_handler(None)
        routing.reset_for_tests()


def test_a_change_during_a_slow_gateway_start_restarts_once_the_gateway_registers(plane, monkeypatch):
    import gateway.restart as restart_mod
    from gateway.restart_when_idle import install_secret_store_restart
    monkeypatch.setattr(restart_mod, "is_gateway_supervisor_process", lambda: True)
    monkeypatch.setattr(restart_mod, "is_container_restart_context", lambda: False)
    routing.set_restart_handler(None)   # booted, runner not registered yet
    plane.set_level(plane.profile("default"), env={"EARLY_KEY": "boot-value"})
    _boot(plane.home)
    plane.set_level(plane.profile("default"), env={"EARLY_KEY": MARK})
    _poll()   # a due tick while the gateway is still starting
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = _Runner()
    try:
        install_secret_store_restart(loop, runner)
        wait_until(lambda: bool(runner.stopped), timeout=10)
        _poll()
        install_secret_store_restart(loop, runner)   # a second registration does not ask again
        for _ in range(3):   # let anything it scheduled run
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    assert runner.stopped == [{"restart": True, "detached_restart": False, "service_restart": True}]
    assert os.environ["EARLY_KEY"] == "boot-value"


class _ClosedLoop:
    """An old gateway loop that has closed: ``call_soon_threadsafe`` raises ``RuntimeError`` as a
    closed asyncio loop does, after *entered*/*release* when given (to hold it mid-delivery)."""

    def __init__(self, entered=None, release=None):
        self.entered, self.release = entered, release

    def call_soon_threadsafe(self, *_args):
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(10)
        loop = asyncio.new_event_loop()
        loop.close()
        return loop.call_soon_threadsafe(lambda: None)


class _GatewayLoop:
    """A running gateway loop on its own thread."""

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        return self.loop

    def settle(self):
        for _ in range(3):   # let anything scheduled on the loop run
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self.loop).result(5)

    def __exit__(self, *_exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


@pytest.fixture
def supervised(monkeypatch):
    import gateway.restart as restart_mod
    monkeypatch.setattr(restart_mod, "is_gateway_supervisor_process", lambda: True)
    monkeypatch.setattr(restart_mod, "is_container_restart_context", lambda: False)


_PLANNED = [{"restart": True, "detached_restart": False, "service_restart": True}]


def test_a_new_gateway_registered_while_delivery_to_a_closed_old_loop_is_in_flight_restarts(plane, supervised):
    """The poll hands the restart to the old handler, whose loop has closed; while that delivery is
    in flight the new gateway registers. Its registration takes the restart (it is still wanted):
    the old delivery failing afterwards must not leave the process running on its boot view."""
    from gateway.restart_when_idle import install_secret_store_restart
    plane.set_level(plane.profile("default"), env={"CLOSED_RACE": "before"})
    _boot(plane.home)
    entered, release = threading.Event(), threading.Event()
    install_secret_store_restart(_ClosedLoop(entered, release), _Runner())
    plane.set_level(plane.profile("default"), env={"CLOSED_RACE": "after"})
    poll = threading.Thread(target=_poll)
    poll.start()
    runner = _Runner()
    gateway = _GatewayLoop()
    with gateway as loop:
        try:
            assert entered.wait(10)
            install_secret_store_restart(loop, runner)
        finally:
            release.set()
            poll.join(10)
        assert not poll.is_alive()
        wait_until(lambda: bool(runner.stopped), timeout=10)
        _poll()
        gateway.settle()
    assert runner.stopped == _PLANNED
    assert os.environ["CLOSED_RACE"] == "before"


def test_a_closed_loop_then_a_working_gateway_restarts_once(plane, supervised):
    from gateway.restart_when_idle import install_secret_store_restart
    plane.set_level(plane.profile("default"), env={"CLOSED_KEY": "before"})
    _boot(plane.home)
    install_secret_store_restart(_ClosedLoop(), _Runner())
    plane.set_level(plane.profile("default"), env={"CLOSED_KEY": "after"})
    _poll()
    _poll()   # still wanted: the tick asks the closed loop again, which still cannot take it
    runner = _Runner()
    gateway = _GatewayLoop()
    with gateway as loop:
        install_secret_store_restart(loop, runner)
        wait_until(lambda: bool(runner.stopped), timeout=10)
        _poll()
        install_secret_store_restart(loop, runner)
        gateway.settle()
    assert runner.stopped == _PLANNED


def test_a_closed_loop_never_says_the_change_waits_for_the_next_start(plane, supervised, caplog):
    """The old gateway's closed loop cannot take the restart, but a handler is registered: nothing
    says the change waits for the next start, and the new gateway's restart is the one log line.
    Only a process with no handler at all says it, once."""
    from gateway.restart_when_idle import install_secret_store_restart
    plane.set_level(plane.profile("default"), env={"CLOSED_LOG": "before"})
    _boot(plane.home)
    install_secret_store_restart(_ClosedLoop(), _Runner())
    plane.set_level(plane.profile("default"), env={"CLOSED_LOG": "after"})
    runner = _Runner()
    with caplog.at_level(logging.WARNING), _GatewayLoop() as loop:
        _poll()
        _poll()
        install_secret_store_restart(loop, runner)
        wait_until(lambda: bool(runner.stopped), timeout=10)
    assert "they apply when Hermes next starts" not in caplog.text
    assert caplog.text.count("restarting gracefully to apply them") == 1
    assert runner.stopped == _PLANNED


def test_with_no_handler_the_next_start_is_said_once(plane, caplog):
    routing.set_restart_handler(None)
    plane.set_level(plane.profile("default"), env={"CLI_ONCE": MARK})
    _boot(plane.home)
    plane.set_level(plane.profile("default"), env={"CLI_ONCE": "changed"})
    with caplog.at_level(logging.WARNING):
        _poll()
        _poll()
        routing.request_restart("again")
    assert caplog.text.count("they apply when Hermes next starts") == 1


def test_the_poller_asks_again_on_every_tick_until_the_handler_takes_the_restart(plane):
    """A handler that cannot take the restart the first time (no new registration follows): only
    the poller asking again on its next tick delivers it, and once taken it is never asked again."""
    calls = []

    def handler():
        calls.append("asked")
        if len(calls) == 1:
            raise RuntimeError("Event loop is closed")

    plane.set_level(plane.profile("default"), env={"TICK_KEY": "before"})
    _boot(plane.home)
    routing.set_restart_handler(handler)
    plane.set_level(plane.profile("default"), env={"TICK_KEY": "after"})
    _poll()
    assert calls == ["asked"]
    gets = len([r for r in plane.requests if r["method"] == "GET"])
    _poll()
    assert calls == ["asked", "asked"]
    for _ in range(3):
        _poll()
    assert calls == ["asked", "asked"]
    assert len([r for r in plane.requests if r["method"] == "GET"]) == gets   # the plane is not asked


def test_the_poller_thread_keeps_running_while_the_restart_is_wanted(plane, monkeypatch):
    """The real poll loop, not one tick: a restart no handler took keeps the thread asking; the
    handler registered later takes it and the thread then ends."""
    from plugins.secret_store.remote import backend
    monkeypatch.setattr(backend, "_jittered", lambda: 0.05)
    routing.set_restart_handler(None)
    plane.set_level(plane.profile("default"), env={"LOOP_KEY": "before"})
    _boot(plane.home)
    provider = routing.routed_provider()
    asked = []
    original = routing.request_restart

    def counting(reason):
        asked.append(reason)
        return original(reason)

    monkeypatch.setattr(routing, "request_restart", counting)
    plane.set_level(plane.profile("default"), env={"LOOP_KEY": "after"})
    for st in provider._roster():
        st.next_poll = 0.0
    provider._stop.set()   # the boot's poller (waiting a whole interval) ends; start a fast one
    provider._poller.join(10)
    provider._stop.clear()
    provider._poller_pid = -1
    provider._ensure_poller()
    wait_until(lambda: len(asked) >= 3, timeout=10)
    assert provider._poller.is_alive()
    calls = []
    monkeypatch.setattr(routing, "request_restart", original)
    routing.set_restart_handler(lambda: calls.append("taken"))
    wait_until(lambda: not provider._poller.is_alive(), timeout=10)
    assert calls == ["taken"]


@pytest.mark.parametrize("order", ["request_first", "handler_first"])
def test_a_handler_registered_during_the_poll_takes_the_restart_once(plane, monkeypatch, order):
    """Control: the ordinary no-handler poll racing the first registration, in both orders."""
    routing.set_restart_handler(None)
    plane.set_level(plane.profile("default"), env={"RACE_VALUE": "old"})
    _boot(plane.home)
    plane.set_level(plane.profile("default"), env={"RACE_VALUE": "new"})
    entered, release = threading.Event(), threading.Event()
    original = routing.request_restart

    def request(reason):
        if order == "handler_first":
            entered.set()
            assert release.wait(10)
            return original(reason)
        result = original(reason)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(routing, "request_restart", request)
    poll = threading.Thread(target=_poll)
    poll.start()
    calls = []
    try:
        assert entered.wait(10)
        routing.set_restart_handler(lambda: calls.append("first"))
    finally:
        release.set()
        poll.join(10)
    assert not poll.is_alive()
    _poll()
    routing.set_restart_handler(lambda: calls.append("second"))
    _poll()
    assert calls == ["first"]
    assert routing.routed_provider().env_view(plane.home)["RACE_VALUE"] == "old"


def test_repeated_ticks_and_registrations_plan_exactly_one_gateway_restart(plane, supervised):
    from gateway.restart_when_idle import install_secret_store_restart
    plane.set_level(plane.profile("default"), env={"ONCE_KEY": "before"})
    _boot(plane.home)
    runner = _Runner()
    gateway = _GatewayLoop()
    with gateway as loop:
        install_secret_store_restart(loop, runner)
        plane.set_level(plane.profile("default"), env={"ONCE_KEY": "after"})
        for _ in range(5):
            _poll()
            install_secret_store_restart(loop, runner)
        wait_until(lambda: bool(runner.stopped), timeout=10)
        gateway.settle()
    assert runner.stopped == _PLANNED
