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
