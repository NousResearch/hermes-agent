"""On a gateway serving two profiles, a session prompt reaches only the OWNING profile's plugins.

Two real ``SessionAuthority`` objects, one per profile home, each over its own real ``SessionDB``; each
home's real ``PluginManager`` carries its own recording callbacks. A prompt is opened from a thread
bound to the OTHER profile (or to no profile), the way a loop-side registration or a thread hop can be,
and the call must still land in the owner's plugins, running inside the owner's scope.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.run import _profile_runtime_scope
from gateway.session_authority import LiveSession, SessionAuthority
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tests.gateway.test_session_prompt_hooks import OPENED, SETTLED, Recorder, _Peer, register
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


class _HomeRecorder(Recorder):
    """Also records the Hermes home the callback ran under."""

    def opened(self, **kwargs):
        super().opened(**kwargs, home=get_hermes_home())

    def settled(self, **kwargs):
        super().settled(**kwargs, home=get_hermes_home())


@pytest.fixture()
def profiles(tmp_path):
    from hermes_cli.plugins import get_plugin_manager
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True), _draining=False)
    served, dbs = {}, []
    for name in ("alpha", "beta"):
        home = (tmp_path / name).resolve()
        home.mkdir()
        db = SessionDB(db_path=home / "state.db")
        dbs.append(db)
        db.create_session("s", source="test")  # the same session id in both profiles
        epoch = begin_runtime_epoch(db, instance_id=name)
        authority = SessionAuthority(runner, profile_id=str(home), instance_id=name, db=db, epoch=epoch)
        authority.sessions["s"] = LiveSession(None, f"route-{name}")
        admit_session_input(db, epoch=epoch, principal_id="u", session_id="s", request_id="t", payload={"text": "go"})
        generation = claim_session_input(db, epoch=epoch, session_id="s")["generation"]
        recorder = _HomeRecorder()
        with _profile_runtime_scope(home):
            manager = get_plugin_manager()
            register(manager, OPENED, recorder.opened)
            register(manager, SETTLED, recorder.settled)
        served[name] = SimpleNamespace(home=home, authority=authority, generation=generation, plugin=recorder,
                                       live=authority.sessions["s"])
    runner.session_authorities = [served["alpha"].authority, served["beta"].authority]
    yield SimpleNamespace(runner=runner, **served)
    for db in dbs:
        db.close()
    for name in ("alpha", "beta"):
        approval._gateway_queues.pop(f"route-{name}", None)


def _in_scope(home, fn):
    """Run *fn* on a fresh thread bound to *home* (None: unbound, the launch profile's ambient state)."""
    out = {}

    def run():
        if home is None:
            out["value"] = fn()
            return
        with _profile_runtime_scope(home):
            out["value"] = fn()
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(30)
    return out.get("value")


@pytest.mark.parametrize("asked_from", ["unbound", "other profile"])
def test_an_approval_reaches_only_the_owning_profiles_plugins(profiles, asked_from):
    beta = profiles.beta
    entry = _ApprovalEntry({"command": "rm -rf build"})
    approval._gateway_queues.setdefault("route-beta", []).append(entry)
    home = None if asked_from == "unbound" else profiles.alpha.home
    _in_scope(home, lambda: beta.authority.register_approval("s", beta.generation, "route-beta", entry.data))
    [opened] = beta.plugin.wait(OPENED)
    assert opened["prompt_id"] == entry.data["request_id"] and opened["home"] == beta.home
    _in_scope(home, lambda: beta.live.controls.respond("s", beta.generation, opened["prompt_id"], {"choice": "once"}))
    assert entry.event.is_set()
    [settled] = beta.plugin.wait(SETTLED)
    assert settled["home"] == beta.home
    # alpha's plugins saw nothing: deliver a prompt of alpha's own, in order behind beta's.
    alpha_entry = _ApprovalEntry({"command": "ls"})
    approval._gateway_queues.setdefault("route-alpha", []).append(alpha_entry)
    profiles.alpha.authority.register_approval("s", profiles.alpha.generation, "route-alpha", alpha_entry.data)
    [alpha_opened] = profiles.alpha.plugin.wait(OPENED)
    assert alpha_opened["prompt_id"] == alpha_entry.data["request_id"] and alpha_opened["home"] == profiles.alpha.home
    assert [kw["prompt_id"] for _hook, kw in profiles.alpha.plugin.calls] == [alpha_entry.data["request_id"]]
    assert all(kw["prompt_id"] == opened["prompt_id"] for _hook, kw in beta.plugin.calls)


def test_a_request_frame_reaches_the_profile_whose_stream_it_went_out_on(profiles, monkeypatch):
    from gateway import run_api
    from gateway.session_server_requests import ask_viewer, resolve
    monkeypatch.setattr(run_api, "_ACTIVE_RUNNER", profiles.runner)  # the gateway API is up
    beta = profiles.beta
    beta.live.event_stream.fanout.attach(_Peer())
    out = {}
    asker = threading.Thread(target=lambda: out.setdefault("value", _in_scope(
        profiles.alpha.home, lambda: ask_viewer(beta.live.event_stream, "s", "sudo", {"command": "sudo true"}))))
    asker.start()
    [opened] = beta.plugin.wait(OPENED)
    assert opened["kind"] == "sudo" and opened["home"] == beta.home
    assert resolve({"jsonrpc": "2.0", "id": opened["prompt_id"], "result": {"value": "pw"}})
    asker.join(10)
    assert out["value"] == "pw"
    [settled] = beta.plugin.wait(SETTLED)
    assert settled["home"] == beta.home and settled["reason"] == "resolved"
    assert profiles.alpha.plugin.calls == []  # one call goes to one manager: beta's got it


def test_without_a_gateway_api_a_request_frame_runs_in_the_askers_scope(profiles):
    from gateway.session_server_requests import ask_viewer, resolve
    beta = profiles.beta
    beta.live.event_stream.fanout.attach(_Peer())
    out = {}
    asker = threading.Thread(target=lambda: out.setdefault("value", _in_scope(
        beta.home, lambda: ask_viewer(beta.live.event_stream, "s", "secret", {"env_var": "K"}))))
    asker.start()
    [opened] = beta.plugin.wait(OPENED)
    assert opened["home"] == beta.home
    resolve({"jsonrpc": "2.0", "id": opened["prompt_id"], "result": {"value": "v"}})
    asker.join(10)
    beta.plugin.wait(SETTLED)
    assert profiles.alpha.plugin.calls == []
