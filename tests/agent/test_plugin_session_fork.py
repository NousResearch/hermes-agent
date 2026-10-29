"""Plugin session forks (``ctx.fork_session``, ``agent/plugin_session_fork.py``).

Plugins load through real discovery from a temp ``HERMES_HOME``. The lease, gating and
lifecycle rules run against a scripted stand-in fork (``build_cache_parity_fork`` patched);
the end-to-end tests run a real turn whose plugin forks the real session agent from
``post_llm_call`` and lets the fork call the plugin's own tool.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

import hermes_cli.plugins as plugins_mod
from agent import background_review, plugin_session_fork, review_idle_queue, side_question_lease
from agent.side_question_lease import issue_side_question_token
from hermes_cli.plugins import PluginContext

CAP = "llm.session_fork"
PLUGIN = "wisdom"


def _install_plugin(name: str, *, declares: bool, body: str = "def register(ctx):\n    pass\n") -> None:
    plugin_dir = Path(os.environ["HERMES_HOME"]) / "plugins" / name
    plugin_dir.mkdir(parents=True)
    manifest = {"name": name, **({"capabilities": [CAP]} if declares else {})}
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")


def _load(enabled: list[str], granted: list[str]) -> plugins_mod.PluginManager:
    entries = {name: {"granted_capabilities": [CAP]} for name in granted}
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": enabled, "entries": entries}}), encoding="utf-8")
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)
    return plugins_mod.get_plugin_manager()


def _ctx(manager, name: str = PLUGIN) -> PluginContext:
    return PluginContext(manager._plugins[name].manifest, manager)


class _Agent(SimpleNamespace):
    """A stand-in parent (leases and fork runs hold their parent weakly)."""


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {
        "type": "object", "properties": {}}}}


def _agent(turn_id="turn-1", **overrides):
    return _Agent(**{
        "_persist_disabled": False, "_cached_system_prompt": "sys", "_cached_system_prompt_static": None,
        "tools": [_tool("cw_offer"), _tool("terminal")], "session_id": "sess", "_current_turn_id": turn_id,
        "model": "m", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions", "reasoning_config": None, **overrides,
    })


class _FakeFork:
    """What ``build_cache_parity_fork`` hands the worker: runs ``script`` as its turn."""

    def __init__(self, script):
        self._script = script
        self._review_input_token_budget = 1000
        self.started = threading.Event()
        self.interrupted = threading.Event()
        self.seen: dict = {}

    def interrupt(self, message=None):
        self.interrupted.set()

    def run_conversation(self, user_message, conversation_history=None):
        from agent.plugin_session_fork import current_turn_origin
        from tools.terminal_tool import _get_approval_callback

        self.seen = {"prompt": user_message, "origin": current_turn_origin(),
                     "approval": _get_approval_callback(), "budget": self._review_input_token_budget}
        self.started.set()
        return self._script(self)


@pytest.fixture
def forks(monkeypatch):
    """Patch the fork builder; ``forks.script`` decides what the next fork's turn does."""
    state = SimpleNamespace(built=[], released=[], script=lambda fork: {"final_response": "done", "messages": []})

    def _build(parent, task_cfg, *, max_iterations, write_origin):
        fork = _FakeFork(state.script)
        state.built.append({"fork": fork, "task_cfg": task_cfg, "max_iterations": max_iterations,
                            "write_origin": write_origin})
        return fork, {}, False

    def _release(parent, fork, *, task, session_id=None):
        state.released.append({"task": task, "session_id": session_id})
        return {}

    monkeypatch.setattr(background_review, "build_cache_parity_fork", _build)
    monkeypatch.setattr(background_review, "release_cache_parity_fork", _release)
    yield state
    side_question_lease._leases.clear()


@pytest.fixture
def consented():
    """``wisdom`` declares and is granted ``llm.session_fork``; ``bystander`` declares it only."""
    _install_plugin(PLUGIN, declares=True)
    _install_plugin("bystander", declares=True)
    return _load([PLUGIN, "bystander"], granted=[PLUGIN])


def _wait_built(forks, count=1):
    from tests.waiting import wait_until

    wait_until(lambda: len(forks.built) >= count, timeout=10)
    return forks.built[count - 1]["fork"]


def _settled(handle, timeout=10.0):
    assert handle is not None and handle.wait(timeout), handle
    return handle


# ── 1. capability and lease rules ────────────────────────────────────────────────────────


def test_session_fork_capability_alone_earns_a_token(forks, consented):
    """A turn is captured when a plugin holds ``llm.session_fork`` (not only side_question)."""
    parent = _agent()
    assert isinstance(issue_side_question_token(parent, [], "turn-1"), str)


def test_without_the_capability_fork_session_is_none_and_the_token_unspent(forks, consented):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _ctx(consented, "bystander").fork_session(token, "q") is None
    assert forks.built == []
    assert _settled(_ctx(consented).fork_session(token, "q")).outcome == "ok"


def test_one_redemption_per_token(forks, consented):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _settled(_ctx(consented).fork_session(token, "q")).outcome == "ok"
    assert _ctx(consented).fork_session(token, "q") is None
    assert side_question_lease.redeem_side_question(token, "q", plugin_id=PLUGIN) is None
    assert len(forks.built) == 1


def test_expired_and_moved_on_leases_are_none(forks, consented, monkeypatch):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    parent._current_turn_id = "turn-2"
    assert _ctx(consented).fork_session(token, "q") is None

    token = issue_side_question_token(parent, [], "turn-2")
    real = side_question_lease.time.monotonic
    monkeypatch.setattr(side_question_lease.time, "monotonic",
                        lambda: real() + side_question_lease.LEASE_TTL_SECONDS + 1)
    assert _ctx(consented).fork_session(token, "q") is None
    assert forks.built == []


def test_unattended_turns_get_no_token(forks, consented):
    assert issue_side_question_token(_agent(platform="cron"), [], "turn-1") is None
    assert issue_side_question_token(_agent(_delegate_depth=1), [], "turn-1") is None


def test_another_profile_cannot_fork_the_session(forks, consented, tmp_path):
    other_home = tmp_path / "profiles" / "other"
    other_home.mkdir(parents=True)
    (other_home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": [PLUGIN], "entries": {PLUGIN: {"granted_capabilities": [CAP]}}}}), encoding="utf-8")
    foreign = PluginContext(consented._plugins[PLUGIN].manifest, plugins_mod.PluginManager(scope_key=str(other_home)))
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")

    assert foreign.fork_session(token, "q") is None
    assert forks.built == []
    assert _settled(_ctx(consented).fork_session(token, "q")).outcome == "ok"


# ── 2. what the fork may do ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tools", [["not_advertised"], ["cw_offer", "missing"], "cw_offer", [42]])
def test_tools_must_already_be_in_the_session(forks, consented, tools):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _ctx(consented).fork_session(token, "q", tools=tools) is None
    assert forks.built == []
    assert _ctx(consented).fork_session(token, "q", tools=["cw_offer"]) is None  # lease spent


def test_task_must_be_the_plugins_own(forks, consented):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _ctx(consented).fork_session(token, "q", task="background_review") is None
    assert forks.built == []


def test_own_task_inheriting_background_review_routes_the_fork(forks):
    """``inherit_from="background_review"`` (a core block outside the picker) is kept, and the
    fork's ``task=`` lane follows ``auxiliary.background_review`` at read time."""
    _install_plugin(PLUGIN, declares=True, body=(
        "def register(ctx):\n"
        "    ctx.register_auxiliary_task('wisdom_suggest', display_name='Suggest', description='d',\n"
        "                                inherit_from='background_review')\n"))
    manager = _load([PLUGIN], granted=[PLUGIN])
    assert manager._aux_tasks["wisdom_suggest"]["inherit_from"] == "background_review"
    cfg_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["auxiliary"] = {"background_review": {"provider": "openrouter", "model": "vendor/review"}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert _settled(_ctx(manager).fork_session(token, "q", task="wisdom_suggest")).outcome == "ok"
    task_cfg = forks.built[0]["task_cfg"]
    assert (task_cfg["provider"], task_cfg["model"]) == ("openrouter", "vendor/review")


def test_fork_shape_origin_approvals_budget_and_usage(forks, consented):
    parent = _agent()
    token = issue_side_question_token(parent, [{"role": "user", "content": "hi"}], "turn-1")
    handle = _settled(_ctx(consented).fork_session(token, "Worth sharing?", tools=["cw_offer"], max_iterations=99,
                                                   input_budget=50))
    built = forks.built[0]
    fork = built["fork"]
    assert handle.outcome == "ok" and handle.result == "done" and handle.cancelled is False
    assert built["write_origin"] == f"plugin_fork:{PLUGIN}" and built["max_iterations"] == plugin_session_fork.MAX_ITERATIONS
    assert fork.seen["prompt"] == "Worth sharing?" and fork.seen["origin"] == f"plugin_fork:{PLUGIN}"
    assert fork.seen["approval"] is background_review._bg_review_auto_deny
    assert fork.seen["budget"] == 50  # a plugin can only lower the review's bound
    assert fork.tools == parent.tools and fork._cached_system_prompt == parent._cached_system_prompt
    assert forks.released == [{"task": f"plugin_fork:{PLUGIN}", "session_id": "sess"}]
    assert plugin_session_fork.current_turn_origin() is None  # never leaks to the caller


def test_input_budget_never_raises_the_review_bound(forks, consented):
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    _settled(_ctx(consented).fork_session(token, "q", input_budget=10**9))
    assert forks.built[0]["fork"].seen["budget"] == 1000


# ── 3. lifecycle ─────────────────────────────────────────────────────────────────────────


def _blocking(fork):
    fork.interrupted.wait(10)
    return {"final_response": None, "interrupted": True, "messages": []}


def test_next_live_turn_cancels_the_fork(forks, consented):
    forks.script = _blocking
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    handle = _ctx(consented).fork_session(token, "q")
    assert _wait_built(forks).started.wait(10)
    background_review.cancel_background_review_for_live_turn(parent)
    assert _settled(handle).outcome == "cancelled" and handle.cancelled


def test_handle_cancel_and_timeout(forks, consented):
    forks.script = _blocking
    parent = _agent()
    handle = _ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q", timeout=0.05)
    assert _settled(handle).outcome == "timeout"

    handle = _ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q")
    assert _wait_built(forks, 2).started.wait(10)
    handle.cancel()
    assert _settled(handle).outcome == "cancelled"


def test_budget_exhaustion_and_errors_settle_the_handle(forks, consented):
    parent = _agent()

    def _spent(fork):
        fork.session_input_tokens = fork._review_input_token_budget
        return {"failed": True, "messages": []}

    forks.script = _spent
    assert _settled(_ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q")).outcome \
        == "budget_exhausted"

    def _boom(fork):
        raise RuntimeError("provider down")

    forks.script = _boom
    assert _settled(_ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q")).outcome \
        == "error"


def test_managed_local_runtime_defers_and_a_newer_fork_supersedes(forks, consented, monkeypatch):
    queued: list = []
    monkeypatch.setattr(review_idle_queue, "review_targets_managed_local", lambda agent, cfg: True)
    monkeypatch.setattr(review_idle_queue.QUEUE, "enqueue",
                        lambda agent, key, kwargs, dispatch=None: queued.append((key, dispatch)))
    parent = _agent()
    first = _ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q1")
    second = _ctx(consented).fork_session(issue_side_question_token(parent, [], "turn-1"), "q2")
    assert first.outcome == "superseded" and first.done and second.outcome == "deferred"
    assert [key for key, _ in queued] == [f"sess:plugin_fork:{PLUGIN}"] * 2 and forks.built == []

    queued[-1][1]()  # the idle queue dispatches the newest
    assert _settled(second).outcome == "ok" and forks.built[0]["fork"].seen["prompt"] == "q2"


# ── 4. end to end through real discovery, a real turn and a real fork ────────────────────

_E2E_PLUGIN = '''
import json, sys

from tools.terminal_tool import _get_approval_callback

PROBE = sys.modules["_fork_probe"]


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


def _handler(name):
    def handle(args, turn_origin=None, **kwargs):
        PROBE.calls.append({"tool": name, "turn_origin": turn_origin,
                            "auto_deny": getattr(_get_approval_callback(), "__name__", None)})
        return json.dumps({"ok": True})
    return handle


def _log(name, **kw):
    PROBE.events.append({"event": name, "aux_task": kw.get("aux_task"), "plugin_id": kw.get("plugin_id")})


def register(ctx):
    for name in ("cw_offer", "cw_other"):
        ctx.register_tool(name, "cw", _schema(name), _handler(name))
    for name in ("pre_api_request", "post_api_request", "pre_auxiliary_call", "post_auxiliary_call"):
        ctx.register_hook(name, lambda _n=name, **kw: _log(_n, **kw))

    def _on_post_llm_call(**kwargs):
        PROBE.handles.append(ctx.fork_session(kwargs["side_question_token"], PROBE.prompt, tools=["cw_offer"],
                                              **PROBE.fork_kwargs))
    ctx.register_hook("post_llm_call", _on_post_llm_call)
'''


def _reply(content=None, tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
                           model="test/model", usage=SimpleNamespace(prompt_tokens=900, completion_tokens=7,
                                                                     total_tokens=907))


def _call(call_id, name):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments="{}"))


def _run_turn_with_fork(monkeypatch, **fork_kwargs):
    """Live turn: calls cw_offer, answers. Its plugin forks the session from post_llm_call; the
    fork calls cw_other (not whitelisted) and cw_offer, then answers."""
    from run_agent import AIAgent

    probe = SimpleNamespace(calls=[], events=[], handles=[], prompt="Worth sharing with the team?",
                            fork_kwargs=fork_kwargs)
    monkeypatch.setitem(sys.modules, "_fork_probe", probe)
    _install_plugin(PLUGIN, declares=True, body=_E2E_PLUGIN)
    _load([PLUGIN], granted=[PLUGIN])
    usage: list = []
    monkeypatch.setattr(background_review, "_record_review_usage_to_parent",
                        lambda parent, snapshot, *, task, session_id=None: usage.append((task, session_id)))

    replies = [
        _reply(tool_calls=[_call("call_live", "cw_offer")]),
        _reply("The answer is 42."),
        _reply(tool_calls=[_call("call_f1", "cw_other"), _call("call_f2", "cw_offer")]),
        _reply("Offered."),
    ]
    requests: list = []
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: (requests.append(kw), replies.pop(0))[1]
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=[_tool("cw_offer"), _tool("cw_other")]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, skip_context_files=True, skip_memory=True)
        parent._cached_system_prompt = "You are helpful."
        parent.skip_background_review = True
        result = parent.run_conversation("what is the answer?")
        handle = _settled(probe.handles[0], 30)
    side_question_lease._leases.clear()
    return SimpleNamespace(result=result, handle=handle, probe=probe, requests=requests, usage=usage, parent=parent)


def test_real_fork_keeps_cache_parity_and_calls_only_whitelisted_tools(monkeypatch):
    run = _run_turn_with_fork(monkeypatch)
    live, fork = run.requests[:2], run.requests[2:]

    assert run.result["final_response"] == "The answer is 42."
    assert run.handle.outcome == "ok" and run.handle.result == "Offered." and run.handle.actions == ("cw_offer",)
    # Cache parity: byte-identical system prompt and tools[]; the fork extends the live prefix.
    assert fork[0]["messages"][0] == live[-1]["messages"][0] and fork[0]["tools"] == live[-1]["tools"]
    assert fork[0]["messages"][-1] == {"role": "user", "content": "Worth sharing with the team?"}
    # The live call sees no origin; the fork's sees plugin_fork:<id> with approvals auto-denied;
    # cw_other is advertised but not whitelisted, so it never runs.
    assert run.probe.calls == [
        {"tool": "cw_offer", "turn_origin": None, "auto_deny": None},
        {"tool": "cw_offer", "turn_origin": f"plugin_fork:{PLUGIN}", "auto_deny": "_bg_review_auto_deny"},
    ]
    denied = [m for m in fork[1]["messages"] if m.get("role") == "tool" and m.get("tool_call_id") == "call_f1"]
    assert denied and "may only call: cw_offer" in str(denied[0]["content"])
    assert run.usage == [(f"plugin_fork:{PLUGIN}", run.parent.session_id)]
    # Turn observers see the live requests only; the fork reports as auxiliary work.
    assert [e["event"] for e in run.probe.events] == ["pre_api_request", "post_api_request"] * 2 + [
        "pre_auxiliary_call", "post_auxiliary_call"] * 2
    assert {(e["aux_task"], e["plugin_id"]) for e in run.probe.events[4:]} == {("plugin_fork", PLUGIN)}


def test_real_fork_stops_at_its_input_budget(monkeypatch):
    run = _run_turn_with_fork(monkeypatch, input_budget=1)
    assert run.handle.outcome == "budget_exhausted"
    assert len(run.requests) == 3  # the crossing request completes; no further provider call
