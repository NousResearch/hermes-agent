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
