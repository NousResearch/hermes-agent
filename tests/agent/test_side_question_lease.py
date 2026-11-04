"""Turn leases for plugin side questions (``agent/side_question_lease.py``, ``ctx.side_question``).

Plugins load through real discovery from a temp ``HERMES_HOME``. The lease rules (issued only
for a consented plugin, one attempt per turn shared across plugins, expiry, invalid once the
next turn starts, prompt cap) are checked against a recorded executor; the last test runs a real
turn end to end and lets a plugin's ``post_llm_call`` hook redeem its token.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

import hermes_cli.plugins as plugins_mod
from agent import side_question_lease
from agent.side_question_lease import issue_side_question_token, redeem_side_question
from hermes_cli.plugins import PluginContext

CAP = "llm.side_question"


def _install_plugin(name: str, *, declares: bool, hook_body: str = "pass") -> None:
    plugin_dir = Path(os.environ["HERMES_HOME"]) / "plugins" / name
    plugin_dir.mkdir(parents=True)
    manifest = {"name": name, **({"capabilities": [CAP]} if declares else {})}
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import json, os, pathlib\n\n"
        "def register(ctx):\n"
        "    def _on_post_llm_call(**kwargs):\n"
        f"        {hook_body}\n"
        "    ctx.register_hook('post_llm_call', _on_post_llm_call)\n",
        encoding="utf-8",
    )


def _load(enabled: list[str], granted: list[str]) -> plugins_mod.PluginManager:
    entries = {name: {"granted_capabilities": [CAP]} for name in granted}
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": enabled, "entries": entries}}), encoding="utf-8")
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)
    return plugins_mod.get_plugin_manager()


def _ctx(manager, name: str) -> PluginContext:
    return PluginContext(manager._plugins[name].manifest, manager)


class _Agent(SimpleNamespace):
    """A stand-in parent; leases hold their parent weakly, which a bare SimpleNamespace refuses."""


def _agent(turn_id="turn-1", **overrides):
    return _Agent(**{
        "_persist_disabled": False, "_cached_system_prompt": "sys", "_cached_system_prompt_static": None,
        "tools": [], "session_id": "sess", "_current_turn_id": turn_id, **overrides,
    })


@pytest.fixture
def executor(monkeypatch):
    """Record every request the lease lets through (the request itself has its own tests)."""
    calls: list = []

    def _ask(capture, prompt, *, plugin_id="", timeout=15.0):
        calls.append({"plugin_id": plugin_id, "prompt": prompt, "timeout": timeout, "capture": capture})
        return f"answer for {plugin_id}"

    monkeypatch.setattr(side_question_lease, "ask_text_only", _ask)
    yield calls
    side_question_lease._leases.clear()


@pytest.fixture
def consented():
    """``next-prompt`` declares and is granted the capability; ``bystander`` does neither."""
    _install_plugin("next-prompt", declares=True)
    _install_plugin("bystander", declares=False)
    return _load(["next-prompt", "bystander"], granted=["next-prompt"])


# ── 1. issuance ──────────────────────────────────────────────────────────────────────────


def test_token_only_for_a_declared_and_granted_plugin(executor):
    _install_plugin("declares-only", declares=True)
    _install_plugin("granted-only", declares=False)
    _load(["declares-only", "granted-only"], granted=["granted-only"])
    assert issue_side_question_token(_agent(), [], "turn-1") is None

    _load(["declares-only", "granted-only"], granted=["declares-only", "granted-only"])
    assert isinstance(issue_side_question_token(_agent(), [], "turn-1"), str)


def test_no_token_when_the_consented_plugin_is_disabled(executor):
    _install_plugin("next-prompt", declares=True)
    _load([], granted=["next-prompt"])
    assert issue_side_question_token(_agent(), [], "turn-1") is None


def test_no_token_for_a_persist_disabled_fork(executor, consented):
    assert issue_side_question_token(_agent(_persist_disabled=True), [], "turn-1") is None


def test_no_token_for_a_cron_run(executor, consented):
    """A scheduled job has nobody watching: a plugin must not spend the main model on it."""
    assert issue_side_question_token(_agent(platform="cron"), [], "turn-1") is None
    assert isinstance(issue_side_question_token(_agent(platform="telegram"), [], "turn-1"), str)


def test_no_token_for_a_subagent_turn(executor, consented):
    """A delegated child's turn is unattended work inside the parent's turn."""
    assert issue_side_question_token(_agent(_delegate_depth=1), [], "turn-1") is None
    assert isinstance(issue_side_question_token(_agent(_delegate_depth=0), [], "turn-1"), str)


def test_capture_is_frozen_at_issue(executor, consented):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, messages, "turn-1")
    messages.append({"role": "user", "content": "later"})
    redeem_side_question(token, "q", plugin_id="next-prompt")
    assert executor[0]["capture"].messages == messages[:2]


# ── 2. capability denied keeps the attempt for the consented plugin ─────────────────────


def test_denied_plugin_gets_none_and_leaves_the_lease(executor, consented):
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")
    assert _ctx(consented, "bystander").side_question(token, "q") is None
    assert executor == []
    assert _ctx(consented, "next-prompt").side_question(token, "q") == "answer for next-prompt"


# ── 3. single use across plugins ─────────────────────────────────────────────────────────


def test_one_attempt_per_turn(executor, consented):
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")
    assert redeem_side_question(token, "q", plugin_id="next-prompt") == "answer for next-prompt"
    assert redeem_side_question(token, "q", plugin_id="other") is None
    assert len(executor) == 1


def test_a_new_lease_for_the_same_agent_drops_the_old_one(executor, consented):
    agent = _agent()
    old = issue_side_question_token(agent, [], "turn-1")
    agent._current_turn_id = "turn-2"
    new = issue_side_question_token(agent, [], "turn-2")
    assert redeem_side_question(old, "q", plugin_id="next-prompt") is None
    assert redeem_side_question(new, "q", plugin_id="next-prompt") == "answer for next-prompt"


# ── 4. expiry, turn movement, prompt cap ─────────────────────────────────────────────────


def test_many_live_sessions_keep_their_tokens(executor, consented):
    """A busy gateway finishing many turns at once must not revoke a live session's token."""
    parents = [_agent(session_id=f"sess-{i}") for i in range(40)]
    tokens = [issue_side_question_token(parent, [], "turn-1") for parent in parents]
    assert _ctx(consented, "next-prompt").side_question(tokens[0], "q") == "answer for next-prompt"


def test_expired_lease_is_none(executor, consented, monkeypatch):
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")
    real = side_question_lease.time.monotonic
    monkeypatch.setattr(side_question_lease.time, "monotonic",
                        lambda: real() + side_question_lease.LEASE_TTL_SECONDS + 1)
    assert redeem_side_question(token, "q", plugin_id="next-prompt") is None
    assert executor == []


def test_lease_dies_when_the_parent_starts_its_next_turn(executor, consented):
    agent = _agent()
    token = issue_side_question_token(agent, [], "turn-1")
    agent._current_turn_id = "turn-2"
    assert redeem_side_question(token, "q", plugin_id="next-prompt") is None
    assert executor == []


def test_lease_dies_with_its_parent(executor, consented):
    import gc

    agent = _agent()
    token = issue_side_question_token(agent, [], "turn-1")
    del agent
    gc.collect()
    assert redeem_side_question(token, "q", plugin_id="next-prompt") is None
    assert executor == []


@pytest.mark.parametrize("prompt", ["x" * (side_question_lease.MAX_PROMPT_CHARS + 1), "", "   ", None, 42])
def test_bad_prompt_is_none_and_consumes_the_lease(executor, consented, prompt):
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")
    assert redeem_side_question(token, prompt, plugin_id="next-prompt") is None
    assert redeem_side_question(token, "fine", plugin_id="next-prompt") is None
    assert executor == []


def test_another_profile_cannot_redeem_the_lease(executor, consented, tmp_path):
    """A lease is bound to the profile home it was issued in: a plugin context from another
    profile's manager (granted there too) gets None and leaves the lease to its owner."""
    other_home = tmp_path / "profiles" / "other"
    other_home.mkdir(parents=True)
    (other_home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["next-prompt"], "entries": {"next-prompt": {"granted_capabilities": [CAP]}}}}),
        encoding="utf-8")
    foreign = PluginContext(consented._plugins["next-prompt"].manifest,
                            plugins_mod.PluginManager(scope_key=str(other_home)))
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")

    assert foreign.side_question(token, "q") is None
    assert redeem_side_question(token, "q", plugin_id="next-prompt", home=other_home) is None
    assert executor == []
    assert _ctx(consented, "next-prompt").side_question(token, "q") == "answer for next-prompt"


@pytest.mark.parametrize("given, used", [(0.5, 0.5), (600, 60.0), (0, 15.0), (-3, 15.0), (float("nan"), 15.0), ("x", 15.0)])
def test_timeout_is_clamped(executor, consented, given, used):
    agent = _agent()  # the lease holds its parent weakly
    token = issue_side_question_token(agent, [], "turn-1")
    redeem_side_question(token, "q", plugin_id="next-prompt", timeout=given)
    assert executor[0]["timeout"] == used


# ── 5. end to end through real discovery and a real turn ─────────────────────────────────


def test_post_llm_call_hook_redeems_its_token_end_to_end(monkeypatch):
    from run_agent import AIAgent

    answers = Path(os.environ["HERMES_HOME"]) / "side_answers.json"
    _install_plugin("next-prompt", declares=True, hook_body=(
        "answer = ctx.side_question(kwargs['side_question_token'], 'Suggest the next prompt.')\n"
        f"        pathlib.Path({str(answers)!r}).write_text(json.dumps(answer))"
    ))
    _load(["next-prompt"], granted=["next-prompt"])

    client = MagicMock()
    requests: list = []
    replies = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None),
                                                 finish_reason="stop")], model="test/model", usage=None)
        for text in ("The answer is 42.", "Ask why it is 42.")
    ]
    client.chat.completions.create.side_effect = lambda **kw: (requests.append(kw), replies.pop(0))[1]
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, skip_context_files=True, skip_memory=True)
        parent._cached_system_prompt = "You are helpful."
        parent.skip_background_review = True
        result = parent.run_conversation("what is the answer?")

    assert result["final_response"] == "The answer is 42."
    assert json.loads(answers.read_text()) == "Ask why it is 42."
    assert requests[-1]["messages"][-1] == {"role": "user", "content": "Suggest the next prompt."}
    side_question_lease._leases.clear()


def _observed_turn_with_side_question(monkeypatch, side_reply) -> list[dict]:
    """Run a real turn whose granted ``observer`` plugin asks a side question and records every
    request event; ``side_reply`` is what the provider returns to the side request."""
    from run_agent import AIAgent

    events = Path(os.environ["HERMES_HOME"]) / "events.jsonl"
    plugin_dir = Path(os.environ["HERMES_HOME"]) / "plugins" / "observer"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "observer", "capabilities": [CAP]}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import json, pathlib\n\n"
        f"EVENTS = pathlib.Path({str(events)!r})\n\n"
        "def _log(name, **kw):\n"
        "    row = {'event': name, 'aux_task': kw.get('aux_task'), 'usage': kw.get('usage'),\n"
        "           'error_type': kw.get('error_type'), 'plugin_id': kw.get('plugin_id')}\n"
        "    with EVENTS.open('a') as fh:\n"
        "        fh.write(json.dumps(row, default=str) + '\\n')\n\n"
        "def register(ctx):\n"
        "    for name in ('pre_api_request', 'post_api_request', 'pre_auxiliary_call', 'post_auxiliary_call'):\n"
        "        ctx.register_hook(name, lambda _n=name, **kw: _log(_n, **kw))\n"
        "    def _on_post_llm_call(**kwargs):\n"
        "        ctx.side_question(kwargs['side_question_token'], 'Suggest the next prompt.')\n"
        "    ctx.register_hook('post_llm_call', _on_post_llm_call)\n",
        encoding="utf-8",
    )
    _load(["observer"], granted=["observer"])

    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=7, total_tokens=907)
    replies = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="The answer is 42.", tool_calls=None),
                                                 finish_reason="stop")], model="test/model", usage=usage),
        side_reply,
    ]
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: replies.pop(0)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        parent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                         quiet_mode=True, skip_context_files=True, skip_memory=True)
        parent._cached_system_prompt = "You are helpful."
        parent.skip_background_review = True
        parent.run_conversation("what is the answer?")
    side_question_lease._leases.clear()
    assert replies == []  # the side request was sent
    return [json.loads(line) for line in events.read_text().splitlines()]


def test_side_request_fires_auxiliary_hooks_not_turn_request_hooks(monkeypatch):
    """Turn-scoped observers (``pre/post_api_request``) see only the turn's own request; the
    side request reports through ``pre/post_auxiliary_call`` as ``plugin_side_question`` (not
    ``/btw``'s ``side_question``), naming the plugin that asked."""
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=7, total_tokens=907)
    rows = _observed_turn_with_side_question(monkeypatch, SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Ask why it is 42.", tool_calls=None),
                                 finish_reason="stop")], model="test/model", usage=usage))
    assert [r["event"] for r in rows] == [
        "pre_api_request", "post_api_request", "pre_auxiliary_call", "post_auxiliary_call"]
    assert rows[2]["aux_task"] == rows[3]["aux_task"] == "plugin_side_question"
    assert rows[2]["plugin_id"] == rows[3]["plugin_id"] == "observer"
    assert rows[0]["plugin_id"] is None  # the turn's own request names no plugin
    assert rows[3]["usage"]["output_tokens"] == 7 and rows[3]["error_type"] is None


def test_unreadable_side_reply_still_closes_the_auxiliary_pair(monkeypatch):
    """A reply that fails after the provider returned (no choices) reports its error in
    ``post_auxiliary_call`` instead of leaving the observer's request open."""
    rows = _observed_turn_with_side_question(monkeypatch, SimpleNamespace(choices=[], model="test/model", usage=None))
    assert [r["event"] for r in rows][-2:] == ["pre_auxiliary_call", "post_auxiliary_call"]
    assert rows[-1]["error_type"]
