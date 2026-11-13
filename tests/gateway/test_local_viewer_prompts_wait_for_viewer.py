"""A local viewer's sudo, secret and vault prompts wait for a viewer that attaches later.

A phone that was backgrounded or a Desktop window that reconnects a second later has no transport
attached at the instant the turn asks. The prompt used to be skipped at once (sudo failed headless, a
secret was "skipped", a vault prompt answered empty) although ``session.resume`` hands a reconnecting
viewer every open request. Main's sidecar (``tui_gateway/server_requests.py``) registers the request and
waits out its timeout with nobody attached; so do these three prompts now. A finite single-query turn
(``chat -q``) still answers at once, and every other request (batch clarify, a plugin's choice, the
Desktop window requests) still returns at once with no viewer.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.waiting import wait_until


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


def _detached():
    """A live session's stream with no viewer attached; *published* records what it publishes."""
    from gateway.session_events import SessionEvents
    events, published = SessionEvents(), []
    events.observers.add(published.append)  # an observer is not a transport: still nobody attached
    assert not events.fanout.transports()
    return events, published


def _turn(events, session_id="s1"):
    authority = SimpleNamespace(sessions={session_id: SimpleNamespace(event_stream=events)})
    return SimpleNamespace(_approval_owner=(authority, session_id, 1),
                           _ctx=SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL)))


def _cancels(published):
    return [f["params"]["payload"] for f in published if f["params"].get("type") == "request.cancel"]


def _replayed(events, method, session_id="s1"):
    """The request a reconnecting viewer is handed (``session.resume`` / ``session.events.since``)."""
    from gateway.session_server_requests import open_requests
    return wait_until(lambda: next((e for e in open_requests(events, session_id) if e["method"] == method), None),
                      what=f"{method} in open_requests")


def _answer(entry, value):
    """The response frame a viewer that just attached sends for a replayed request."""
    from gateway.session_server_requests import is_response_frame, resolve
    frame = {"jsonrpc": "2.0", "id": entry["id"], "result": {"value": value}}
    assert is_response_frame(frame) and resolve(frame)


class _Call:
    """Run *fn* on its own thread (the agent / tool worker) and keep what it returned."""

    def __init__(self, fn):
        self.out, self.done = {}, threading.Event()

        def run():
            try:
                self.out["value"] = fn()
            finally:
                self.done.set()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def result(self, timeout=10):
        assert self.done.wait(timeout), "the prompt never ended"
        self.elapsed = time.monotonic() - self.started
        return self.out["value"]


def _sudo(events, agent=None, finite=None):
    from gateway.session_finite import finite_turn_scope
    from gateway.session_sudo_prompt import sudo_prompt_scope
    from tools.terminal_tool_sudo import _prompt_for_sudo_password

    def run():
        with finite_turn_scope(finite), sudo_prompt_scope(_turn(events), agent or SimpleNamespace()):
            return _prompt_for_sudo_password(command="sudo true")  # the terminal tool's own caller
    return _Call(run)


def _secret(events, monkeypatch, agent=None, finite=None):
    from gateway.session_finite import finite_turn_scope
    from gateway.session_secret_prompt import secret_prompt_scope
    from tools.skills_tool_setup import _capture_required_environment_variables
    from tools.thread_context import propagate_context_to_thread
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setenv("HAB_WAIT_KEY", "")
    monkeypatch.delenv("HAB_WAIT_KEY")  # restored (unset) after the test, whatever the save sets
    out = {}

    def run():
        with finite_turn_scope(finite), secret_prompt_scope(_turn(events), agent or SimpleNamespace()):
            worker = threading.Thread(target=propagate_context_to_thread(lambda: out.update(
                _capture_required_environment_variables("hab-skill", [{"name": "HAB_WAIT_KEY", "prompt": "Key"}]))))
            worker.start()
            worker.join(30)
        return out
    return _Call(run)


def _vault_unlock(events, agent=None, finite=None):
    from agent.vault_backends import unlock
    from gateway.session_finite import finite_turn_scope
    from gateway.session_vault_prompt import vault_prompt_scope

    def run():
        with finite_turn_scope(finite), vault_prompt_scope(_turn(events), agent or SimpleNamespace()):
            return unlock.get_unlock_prompt_callback()("bitwarden", "Bitwarden")
    return _Call(run)


def _held_open(events, method):
    from gateway.session_server_requests import waiting
    wait_until(lambda: waiting(events, "s1", method), what=f"{method} held open")


# 1. A viewer that attaches while the prompt waits gets it and its answer reaches the turn.

def test_sudo_waits_for_a_viewer_that_attaches_and_returns_its_password():
    events, published = _detached()
    call = _sudo(events)
    entry = _replayed(events, "sudo")
    assert entry["params"] == {"session_id": "s1", "command": "sudo true"}
    assert not call.done.is_set()
    _answer(entry, "hunter2")
    assert call.result() == "hunter2"
    assert [c["reason"] for c in _cancels(published)] == ["resolved"]
    assert not any("hunter2" in repr(f) for f in published)


def test_secret_waits_for_a_viewer_that_attaches_and_is_saved_in_the_profile_env(monkeypatch):
    from hermes_constants import get_hermes_home
    events, published = _detached()
    call = _secret(events, monkeypatch)
    entry = _replayed(events, "secret")
    assert entry["params"]["env_var"] == "HAB_WAIT_KEY"
    _answer(entry, "sekrit")
    out = call.result()
    assert out["missing_names"] == [] and not out.get("setup_skipped")
    assert "HAB_WAIT_KEY=sekrit" in (get_hermes_home() / ".env").read_text(encoding="utf-8")
    assert not any("sekrit" in repr(f) for f in published)


def test_vault_unlock_waits_for_a_viewer_that_attaches_and_returns_the_master_password():
    events, published = _detached()
    call = _vault_unlock(events)
    entry = _replayed(events, "vault.unlock_prompt")
    assert entry["params"]["display_name"] == "Bitwarden"
    _answer(entry, "correct horse")
    assert call.result() == "correct horse"
    assert not any("correct horse" in repr(f) for f in published)


# 2. Nobody ever attaches: the existing timeout and result, and the existing request.cancel.

def test_sudo_with_nobody_attaching_ends_at_its_timeout(monkeypatch):
    from gateway import session_sudo_prompt
    monkeypatch.setattr(session_sudo_prompt, "SUDO_TIMEOUT_S", 0.5)
    events, published = _detached()
    call = _sudo(events)
    assert call.result() == ""
    assert call.elapsed >= 0.5
    assert [(c["method"], c["reason"]) for c in _cancels(published)] == [("sudo", "timeout")]


def test_secret_with_nobody_attaching_ends_at_its_timeout_and_is_skipped(monkeypatch):
    from gateway import session_secret_prompt
    monkeypatch.setattr(session_secret_prompt, "SECRET_TIMEOUT_S", 0.5)
    events, published = _detached()
    call = _secret(events, monkeypatch)
    out = call.result()
    assert call.elapsed >= 0.5
    assert out["missing_names"] == ["HAB_WAIT_KEY"] and out["setup_skipped"] is True
    assert [(c["method"], c["reason"]) for c in _cancels(published)] == [("secret", "timeout")]


def test_vault_unlock_with_nobody_attaching_ends_at_its_timeout(monkeypatch):
    from gateway import session_vault_prompt
    monkeypatch.setattr(session_vault_prompt, "UNLOCK_TIMEOUT_S", 0.5)
    events, published = _detached()
    call = _vault_unlock(events)
    assert call.result() == ""
    assert call.elapsed >= 0.5
    assert [(c["method"], c["reason"]) for c in _cancels(published)] == [("vault.unlock_prompt", "timeout")]


# 3. An interrupt during the wait ends it at once.

@pytest.mark.parametrize(("start", "method", "ended"), [
    (lambda events, agent, mp: _sudo(events, agent), "sudo", lambda out: out == ""),
    (lambda events, agent, mp: _secret(events, mp, agent), "secret", lambda out: out["setup_skipped"] is True),
    (lambda events, agent, mp: _vault_unlock(events, agent), "vault.unlock_prompt", lambda out: out == ""),
], ids=["sudo", "secret", "vault"])
def test_an_interrupt_ends_the_wait_at_once(monkeypatch, start, method, ended):
    events, published = _detached()
    agent = SimpleNamespace(_interrupt_requested=False)
    call = start(events, agent, monkeypatch)
    _held_open(events, method)
    agent._interrupt_requested = True
    assert ended(call.result(timeout=10))
    assert [(c["method"], c["reason"]) for c in _cancels(published)] == [(method, "interrupted")]


# 4. Preserved behavior (passes on the base too).

def test_an_attached_viewer_gets_the_frame_and_its_answer_settles_the_request():
    from gateway.session_events import SessionEvents
    events, peer = SessionEvents(), _Peer()
    events.fanout.attach(peer)
    call = _sudo(events)
    request = wait_until(lambda: next((f for f in peer.frames if f.get("method") == "sudo"), None), what="sudo frame")
    _answer(request, "hunter2")
    assert call.result() == "hunter2"


@pytest.mark.parametrize("start", [
    lambda events, mp: _sudo(events, finite=True),
    lambda events, mp: _secret(events, mp, finite=True),
    lambda events, mp: _vault_unlock(events, finite=True),
], ids=["sudo", "secret", "vault"])
def test_a_finite_turn_with_no_viewer_still_answers_at_once(monkeypatch, start):
    """``chat -q``: its client detaches on purpose and nobody can answer, so nothing is held open."""
    events, published = _detached()
    call = start(events, monkeypatch)
    out = call.result(timeout=5)
    assert out == "" or out["setup_skipped"] is True
    assert call.elapsed < 2
    assert published == []


def test_batch_clarify_with_no_viewer_still_returns_at_once():
    from gateway.session_clarify_batch import ask_clarify_batch
    events, published = _detached()
    turn = _turn(events)
    turn._ctx.agent_holder = [SimpleNamespace(_interrupt_requested=False)]
    started = time.monotonic()
    out = ask_clarify_batch(turn, [{"qid": "q0", "question": "colour?", "choices": ["red"], "multi_select": False}])
    assert time.monotonic() - started < 2
    assert out == {"answers": {}, "outcome": "timed_out"} and published == []


def test_a_plugin_command_choice_with_no_viewer_still_returns_at_once():
    from gateway.plugin_command_choice import viewer_choices
    from hermes_cli.plugin_choices import ask_choice
    events, published = _detached()
    started = time.monotonic()
    with viewer_choices(SimpleNamespace(event_stream=events), "s1"):
        assert ask_choice("Pick one", ["alpha", "beta"]) is None
    assert time.monotonic() - started < 2 and published == []


def test_every_desktop_window_request_with_no_viewer_still_returns_at_once():
    from gateway import local_desktop_ui
    from gateway.session_server_requests import open_requests
    events, published = _detached()
    turn, agent = _turn(events), SimpleNamespace(_interrupt_requested=False)
    started = time.monotonic()
    for method in ("terminal.read", "preview.read", "window.read", "preview.act", "tour", "theme.apply"):
        assert local_desktop_ui._ask(turn, agent, method, {}, 60) == ""
    for method in ("theme.pick", "setup_beat"):
        assert local_desktop_ui._ask_result(turn, agent, method, {}, 60) is None
    assert time.monotonic() - started < 2
    assert open_requests(events, "s1") == [] and published == []
