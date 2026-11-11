"""``on_session_prompt`` / ``on_session_prompt_settled``: the prompts a gateway session owner opens for a person.

Real owner pieces drive every prompt: ``SessionAuthority`` over a real ``SessionDB`` with a real claim, the
real approval queue and clarify registry, ``PendingControls`` and the request registry
(``gateway.session_server_requests``) with a fake viewer socket. Callbacks sit on the active home's real
``PluginManager`` exactly as ``ctx.register_hook`` stores them, so delivery goes through the real plugin
dispatch. Each prompt is answered through its real path with the id the hook delivered.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tests.waiting import wait_until
from tools import approval, clarify_gateway
from tools.approval_gateway_wait import _ApprovalEntry

OPENED, SETTLED = "on_session_prompt", "on_session_prompt_settled"
TOKEN = "ghp_" + "Z" * 36
PERSON = ("sudo", "secret", "clarify", "vault.unlock_prompt", "vault.code", "vault.save_login")


class Recorder:
    """A plugin's two callbacks; ``calls`` is ``[(hook, kwargs), ...]`` in delivery order, without the
    ``telemetry_schema_version`` the plugin manager stamps on every observer call."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def opened(self, **kwargs):
        kwargs.pop("telemetry_schema_version", None)
        with self.lock:
            self.calls.append((OPENED, kwargs))

    def settled(self, **kwargs):
        kwargs.pop("telemetry_schema_version", None)
        with self.lock:
            self.calls.append((SETTLED, kwargs))

    def of(self, prompt_id):
        with self.lock:
            return [(hook, kw) for hook, kw in self.calls if kw.get("prompt_id") == prompt_id]

    def wait(self, hook, *, prompt_id=None, count=1):
        def found():
            with self.lock:
                hits = [kw for name, kw in self.calls
                        if name == hook and (prompt_id is None or kw.get("prompt_id") == prompt_id)]
            return hits if len(hits) >= count else None
        return wait_until(found, timeout=10, what=f"{count} {hook} for {prompt_id or 'any prompt'}")


def register(manager, hook, callback):
    manager._discovered = True  # no on-disk plugins in the test home
    manager._hooks.setdefault(hook, []).append(callback)


@pytest.fixture()
def plugin():
    from hermes_cli.plugins import get_plugin_manager
    manager, recorder = get_plugin_manager(), Recorder()
    register(manager, OPENED, recorder.opened)
    register(manager, SETTLED, recorder.settled)
    yield recorder
    for hook, callback in ((OPENED, recorder.opened), (SETTLED, recorder.settled)):
        manager._hooks[hook] = [cb for cb in manager._hooks.get(hook, []) if cb is not callback]


@pytest.fixture()
def owner(tmp_path):
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("s", source="test")
        epoch = begin_runtime_epoch(db, instance_id="current")
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="bot",
                                     instance_id="current", db=db, epoch=epoch)
        authority.sessions["s"] = LiveSession(None, "route")
        admit_session_input(db, epoch=epoch, principal_id="bot:alpha", session_id="s",
                            request_id="turn", payload={"text": "go"})
        generation = claim_session_input(db, epoch=epoch, session_id="s")["generation"]
        yield SimpleNamespace(authority=authority, live=authority.sessions["s"], generation=generation)
    approval._gateway_queues.pop("route", None)
    clarify_gateway.clear_session("route")


def _approval(owner, command="rm -rf build"):
    entry = _ApprovalEntry({"command": command, "description": "dangerous"})
    approval._gateway_queues.setdefault("route", []).append(entry)
    owner.authority.register_approval("s", owner.generation, "route", entry.data)
    return entry


def _clarify(owner, clarify_id="q-1"):
    entry = clarify_gateway.register(clarify_id, "route", "Red or blue?", ["red", "blue"])
    owner.authority.register_clarify("s", owner.generation, entry)
    return entry


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


def _viewer_stream():
    from gateway.session_events import SessionEvents
    events, peer = SessionEvents(), _Peer()
    events.fanout.attach(peer)
    return events, peer


def _ask_in_thread(events, method, params, **kwargs):
    """Run one owner request on its own (agent) thread; ``out['value']`` is what the asker got back."""
    from gateway.session_server_requests import ask_viewer, ask_viewer_result
    out = {}
    structured = method == "clarify"
    ask = ask_viewer_result if structured else ask_viewer
    if structured:
        kwargs.setdefault("qids", ["q0"])
    worker = threading.Thread(target=lambda: out.setdefault("value", ask(events, "s", method, params, **kwargs)))
    worker.start()
    return worker, out


# -- approval and clarify events ------------------------------------------------------------------------

def test_approval_fires_with_the_id_that_answers_it(owner, plugin):
    entry = _approval(owner)
    [opened] = plugin.wait(OPENED)
    assert set(opened) == {"session_id", "kind", "prompt_id", "execution_generation", "params"}
    assert (opened["session_id"], opened["kind"], opened["execution_generation"]) == ("s", "approval", owner.generation)
    assert opened["prompt_id"] == entry.data["request_id"]
    assert opened["params"]["choices"] == ["once", "deny", "session", "always"]
    reply = owner.live.controls.respond("s", owner.generation, opened["prompt_id"], {"choice": "once"})
    assert reply["status"] == "resolved"
    assert entry.event.wait(5) and entry.result == "once"  # the agent's waiter unblocked
    [settled] = plugin.wait(SETTLED)
    assert settled == {"session_id": "s", "kind": "approval", "prompt_id": opened["prompt_id"],
                       "execution_generation": owner.generation, "reason": "settled"}
    owner.live.controls.snapshot("s", owner.generation)
    _approval(owner)  # a later prompt is delivered after anything still queued for the first
    plugin.wait(OPENED, count=2)
    assert [hook for hook, _ in plugin.of(opened["prompt_id"])] == [OPENED, SETTLED]


def test_clarify_fires_with_the_id_that_answers_it(owner, plugin):
    entry = _clarify(owner)
    [opened] = plugin.wait(OPENED)
    assert (opened["kind"], opened["prompt_id"]) == ("clarify", "q-1")
    assert opened["params"]["question"] == "Red or blue?" and opened["params"]["choices"] == ["red", "blue"]
    reply = owner.live.controls.respond("s", owner.generation, opened["prompt_id"], {"answer": "blue"},
                                        kind="clarify")
    assert reply["status"] == "resolved"
    assert entry.event.is_set() and entry.response == "blue"
    [settled] = plugin.wait(SETTLED)
    assert (settled["kind"], settled["prompt_id"], settled["reason"]) == ("clarify", "q-1", "settled")


def test_a_prompt_the_snapshot_settles_lazily_settles_once(owner, plugin):
    entry = _clarify(owner)
    plugin.wait(OPENED)
    clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "red")  # answered natively, e.g. a chat button
    owner.live.controls.snapshot("s", owner.generation)
    owner.live.controls.snapshot("s", owner.generation)
    _approval(owner)
    owner.live.controls.snapshot("s", None)  # the turn ended: every prompt of it settles
    plugin.wait(SETTLED, count=2)
    _clarify(owner, "q-2")
    plugin.wait(OPENED, prompt_id="q-2")
    assert [hook for hook, _ in plugin.of("q-1")] == [OPENED, SETTLED]
    settled = [kw for hook, kw in plugin.calls if hook == SETTLED]
    assert len({kw["prompt_id"] for kw in settled}) == len(settled) == 2


def test_a_remote_responder_answer_settles(owner, plugin):
    answers = []
    owner.live.controls.remote_responders["remote-1"] = lambda *answer: answers.append(answer)
    owner.authority.register_approval("s", owner.generation, "route", {"request_id": "remote-1", "command": "x"})
    [opened] = plugin.wait(OPENED)
    assert opened["prompt_id"] == "remote-1"
    reply = owner.live.controls.respond("s", owner.generation, "remote-1", {"choice": "once"})
    assert reply["status"] == "resolved" and answers == [("approval", "remote-1", "once")]
    [settled] = plugin.wait(SETTLED)
    assert settled["prompt_id"] == "remote-1" and settled["reason"] == "settled"


def test_a_managed_worker_prompt_settles_when_the_worker_says_so(owner, plugin):
    from gateway.session_managed_worker import _prompt_frame
    worker = SimpleNamespace(respond=lambda *_a: None)
    ref, row = SimpleNamespace(session_id="s"), {"generation": owner.generation}
    _prompt_frame(owner.authority, ref, row, worker, {"type": "clarify", "prompt_id": "w-1", "question": "Go?",
                                                     "choices": ["yes"], "multi_select": False})
    plugin.wait(OPENED, prompt_id="w-1")
    _prompt_frame(owner.authority, ref, row, worker, {"type": "prompt_settled", "prompt_id": "w-1"})
    plugin.wait(SETTLED, prompt_id="w-1")
    _prompt_frame(owner.authority, ref, row, worker, {"type": "prompt_settled", "prompt_id": "w-1"})
    _clarify(owner, "after")
    plugin.wait(OPENED, prompt_id="after")
    assert [hook for hook, _ in plugin.of("w-1")] == [OPENED, SETTLED]


# -- request frames ------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", PERSON)
def test_each_person_request_fires_with_the_frame_id_that_answers_it(plugin, method):
    from gateway.session_server_requests import open_requests, resolve
    events, peer = _viewer_stream()
    params = {"questions": [{"qid": "q0", "question": "Which?"}]} if method == "clarify" else {"site": "example"}
    worker, out = _ask_in_thread(events, method, params)
    [opened] = plugin.wait(OPENED)
    frame = next(f for f in peer.frames if f.get("method") == method)
    assert opened == {"session_id": "s", "kind": method, "prompt_id": frame["id"], "execution_generation": None,
                      "params": frame["params"]}
    assert [r["id"] for r in open_requests(events, "s")] == [opened["prompt_id"]]
    result = {"answers": {"q0": "this"}} if method == "clarify" else {"value": "typed-answer"}
    assert resolve({"jsonrpc": "2.0", "id": opened["prompt_id"], "result": result})
    worker.join(5)
    assert not worker.is_alive()
    assert out["value"] == ({"answers": {"q0": "this"}, "outcome": "submitted"} if method == "clarify"
                            else "typed-answer")
    [settled] = plugin.wait(SETTLED)
    assert settled == {"session_id": "s", "kind": method, "prompt_id": opened["prompt_id"],
                       "execution_generation": None, "reason": "resolved"}
    assert "typed-answer" not in repr(plugin.calls)  # never what the person typed


@pytest.mark.parametrize("ending", ["timeout", "interrupted", "withdrawn"])
def test_a_request_that_nobody_answers_settles_once_with_its_reason(plugin, ending):
    from gateway.session_server_requests import withdraw
    events, _peer = _viewer_stream()
    stop = threading.Event()
    worker, out = _ask_in_thread(events, "sudo", {"command": "sudo true"}, cancelled=stop.is_set,
                                 timeout=0.25 if ending == "timeout" else 30)
    [opened] = plugin.wait(OPENED)
    if ending == "interrupted":
        stop.set()
    elif ending == "withdrawn":
        assert withdraw(events, "s", "sudo") == 1
    worker.join(10)
    assert out["value"] == ""
    [settled] = plugin.wait(SETTLED)
    assert settled["prompt_id"] == opened["prompt_id"]
    assert settled["reason"] == {"timeout": "timeout", "interrupted": "interrupted", "withdrawn": "resolved"}[ending]


def test_window_requests_and_unasked_prompts_never_fire(plugin):
    from gateway.session_events import SessionEvents
    from gateway.session_server_requests import ask_viewer
    events, peer = _viewer_stream()
    for method in ("theme.apply", "theme.pick", "tour", "preview.act", "preview.read", "window.read",
                   "terminal.read", "setup_beat"):
        worker, _out = _ask_in_thread(events, method, {}, timeout=0.25)
        worker.join(10)
    worker, _out = _ask_in_thread(events, "sudo", {"command": "x"}, timeout=0.25, replay=False)
    worker.join(10)
    assert ask_viewer(SessionEvents(), "s", "sudo", {"command": "x"}, timeout=0.25) == ""  # nobody to ask
    window_ids = {f["id"] for f in peer.frames if "id" in f}
    worker, _out = _ask_in_thread(events, "secret", {"env_var": "K"}, timeout=0.25)
    [sentinel] = plugin.wait(SETTLED)  # delivered in order: anything earlier would be here by now
    worker.join(10)
    assert [kw["prompt_id"] for _hook, kw in plugin.calls] == [sentinel["prompt_id"]] * 2
    assert sentinel["prompt_id"] not in window_ids


# -- failure isolation, locks, redaction -------------------------------------------------------------------

def test_a_raising_plugin_changes_nothing_about_the_prompt(owner, plugin):
    from hermes_cli.plugins import get_plugin_manager
    from gateway.session_server_requests import open_requests, resolve

    def boom(**_kwargs):
        raise RuntimeError("plugin bug")
    manager = get_plugin_manager()
    for hook in (OPENED, SETTLED):
        manager._hooks[hook].insert(0, boom)
    entry = _approval(owner)
    [opened] = plugin.wait(OPENED)  # the next plugin still gets the call
    assert [p["prompt_id"] for p in owner.live.controls.snapshot("s", owner.generation)] == [opened["prompt_id"]]
    owner.live.controls.respond("s", owner.generation, opened["prompt_id"], {"choice": "once"})
    assert entry.event.is_set() and entry.result == "once"
    plugin.wait(SETTLED)
    events, peer = _viewer_stream()
    worker, out = _ask_in_thread(events, "sudo", {"command": "sudo true"})
    [_, frame_opened] = plugin.wait(OPENED, count=2)
    assert [r["id"] for r in open_requests(events, "s")] == [frame_opened["prompt_id"]]
    assert resolve({"jsonrpc": "2.0", "id": frame_opened["prompt_id"], "result": {"value": "pw"}})
    worker.join(5)
    assert out["value"] == "pw"
    assert [f["params"]["type"] for f in peer.frames if f.get("method") == "event"] == ["request.cancel"]


def test_plugin_code_runs_off_the_prompt_path_and_outside_every_lock(owner, plugin):
    """A plugin that blocks does not hold up the registration (made under ``event_stream.lock``): it
    returns while the callback still waits, and the callback can take both locks itself."""
    from gateway import session_server_requests
    from hermes_cli.plugins import get_plugin_manager
    released, seen = threading.Event(), []

    def slow(**kwargs):
        if kwargs["kind"] == "approval":
            seen.append(released.wait(10))
            seen.append(owner.live.event_stream.lock.acquire(timeout=5))
            owner.live.event_stream.lock.release()
            seen.append(session_server_requests._lock.acquire(timeout=5))
            session_server_requests._lock.release()
    get_plugin_manager()._hooks[OPENED].insert(0, slow)
    entry = _approval(owner)
    assert [p["prompt_id"] for p in owner.live.controls.snapshot("s", owner.generation)] == [entry.data["request_id"]]
    released.set()  # only now, after registration returned: a callback run inline would have timed out
    plugin.wait(OPENED)
    assert seen == [True, True, True]


def test_params_are_force_redacted_and_a_copy(plugin, monkeypatch):
    from agent import redact
    from gateway.session_server_requests import open_requests, resolve
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)  # security.redact_secrets: false
    command = f"sudo env GITHUB_TOKEN={TOKEN} apt-get install -y jq"
    assert TOKEN in redact.redact_approval_display(command)  # only force=True can hide it now
    events, peer = _viewer_stream()
    worker, _out = _ask_in_thread(events, "sudo", {"command": command, "nested": [{"prompt": command}]})
    [opened] = plugin.wait(OPENED)
    assert TOKEN in repr(peer.frames)  # the viewer's own frame is untouched
    assert TOKEN not in repr(opened)
    assert opened["params"]["command"] == redact.redact_approval_display(command, force=True)
    opened["params"]["command"] = "mutated"
    assert open_requests(events, "s")[0]["params"]["command"] == command
    resolve({"jsonrpc": "2.0", "id": opened["prompt_id"], "result": {"value": ""}})
    worker.join(5)


def test_a_plugin_declaring_both_hooks_validates(tmp_path):
    from hermes_cli.plugin_dev import doctor_plugin
    root = tmp_path / "prompt-watch"
    root.mkdir()
    (root / "plugin.yaml").write_text(
        "name: prompt-watch\nversion: 0.1.0\nprovides_hooks:\n  - on_session_prompt\n  - on_session_prompt_settled\n",
        encoding="utf-8")
    (root / "__init__.py").write_text(
        "def seen(**kwargs):\n    return None\n\n"
        "def register(ctx):\n"
        "    ctx.register_hook('on_session_prompt', seen)\n"
        "    ctx.register_hook('on_session_prompt_settled', seen)\n", encoding="utf-8")
    report = doctor_plugin(root)
    assert report.ok, report.format_text()
