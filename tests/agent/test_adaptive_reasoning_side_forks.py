"""Adaptive reasoning and the plugin side-question / session-fork leases.

The lease is captured at turn finalization, inside the adaptive turn, and redeemed after it, once the
turn's end has restored the baseline. The adjusted level is the turn's own, not a runtime change: the
lease still matches (``runtime_key``), and the fork replays the level the captured request went out on.
A real change of the session's level since the turn still refuses, as upstream's guard intends.
"""

from __future__ import annotations

import contextvars
import copy
import json
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import background_review, plugin_session_fork, side_question_lease
from agent.adaptive_reasoning import adaptive_reasoning_turn, turn_baseline_reasoning
from agent.side_question_lease import redeem_side_question
from agent.side_question_request import SideQuestionCapture, runtime_key
from run_agent import AIAgent

BASELINE = {"enabled": True, "effort": "medium"}
POLICY = {"enabled": True, "max_effort": "xhigh"}
DEBUG_MSG = "Why does the gateway keep failing after I restart it?"  # code work: high
SIDE_PROMPT = "Suggest the user's next prompt in one line."


def _response(content):
    msg = SimpleNamespace(content=content, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0))
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="m", usage=usage)


class _Provider:
    """One fake client for the parent and its side fork; records every request."""

    def __init__(self):
        self.client = MagicMock()
        self.requests: list[dict] = []
        self.replies: list = []
        self.client.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


@pytest.fixture
def provider(monkeypatch):
    fake = _Provider()
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    patches = [
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=fake.client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        patch.object(side_question_lease, "_consented_plugin_loaded", return_value=True),
    ]
    for p in patches:
        p.start()
    yield fake
    for p in reversed(patches):
        p.stop()
    side_question_lease._leases.clear()


@pytest.fixture
def issued(monkeypatch):
    tokens: list = []
    real = side_question_lease.issue_side_question_token

    def _spy(*args, **kwargs):
        tokens.append(real(*args, **kwargs))
        return tokens[-1]

    monkeypatch.setattr(side_question_lease, "issue_side_question_token", _spy)
    return tokens


def _parent() -> AIAgent:
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="anthropic/claude-sonnet-4",
                    quiet_mode=True, skip_context_files=True, skip_memory=True,
                    reasoning_config=dict(BASELINE), adaptive_reasoning=dict(POLICY))
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.skip_background_review = True
    return agent


def _sans_messages(request: dict) -> str:
    return json.dumps({k: v for k, v in request.items() if k != "messages"}, sort_keys=True, default=str)


def _adjusted_turn(parent, provider):
    provider.replies.append(_response("Restart the socket listener first."))
    baseline = parent.reasoning_config
    parent.run_conversation(DEBUG_MSG)
    assert parent.reasoning_config is baseline, "the turn's end restored the baseline"
    assert '"high"' in _sans_messages(provider.requests[-1]), provider.requests[-1]
    return provider.requests[-1]


def test_a_side_question_after_an_adjusted_turn_replays_the_turns_level(provider, issued):
    parent = _parent()
    parent_last = _adjusted_turn(parent, provider)
    provider.replies.append(_response("Ask: which port?"))
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") == "Ask: which port?"
    # Same request parameters as the turn's last call, the adjusted level included (cache parity).
    assert _sans_messages(provider.requests[-1]) == _sans_messages(parent_last)
    assert parent.reasoning_config == BASELINE


def test_a_session_level_change_after_an_adjusted_turn_still_refuses(provider, issued):
    parent = _parent()
    _adjusted_turn(parent, provider)
    parent.reasoning_config = {"enabled": True, "effort": "low"}  # /reasoning low since the turn
    sent = len(provider.requests)
    assert redeem_side_question(issued[-1], SIDE_PROMPT, plugin_id="next-prompt") is None
    assert len(provider.requests) == sent


class _Agent(SimpleNamespace):
    """A stand-in parent (leases and fork runs hold their parent weakly)."""


def _stand_in(**overrides):
    return _Agent(**{
        "_persist_disabled": False, "_cached_system_prompt": "sys", "_cached_system_prompt_static": None,
        "tools": [], "session_id": "sess", "_current_turn_id": "turn-1", "model": "m", "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions", "platform": "cli",
        "reasoning_config": dict(BASELINE), "adaptive_reasoning": dict(POLICY), "reasoning_user_override": False,
        "_adaptive_prev_effort": None, "_adaptive_last_notified_effort": None, "_adaptive_cache_warned": False,
        "notice_callback": None, **overrides,
    })


def test_runtime_key_holds_the_baseline_only_while_this_turns_adjustment_is_installed():
    agent = _stand_in()
    with adaptive_reasoning_turn(agent, DEBUG_MSG):
        assert agent.reasoning_config["effort"] == "high"
        assert turn_baseline_reasoning(agent) == BASELINE
        inside = runtime_key(agent)
        # Something else replaced the level mid-turn (a voice route, a fallback): that IS the runtime.
        replaced = {"enabled": True, "effort": "low"}
        agent.reasoning_config, held = replaced, agent.reasoning_config
        assert turn_baseline_reasoning(agent) is replaced
        agent.reasoning_config = held
    assert runtime_key(agent) == inside


def test_a_plugin_fork_after_an_adjusted_turn_runs_on_the_captured_level(monkeypatch):
    agent = _stand_in(_plugin_fork_jobs=set())
    monkeypatch.setattr(side_question_lease, "_consented_plugin_loaded", lambda: True)
    with adaptive_reasoning_turn(agent, DEBUG_MSG):
        adjusted = agent.reasoning_config
        token = side_question_lease.issue_side_question_token(agent, [], "turn-1")
    capture = side_question_lease._leases.pop(token).capture
    assert agent.reasoning_config == BASELINE and capture.reasoning_config == adjusted

    built: list = []

    class _Fork:
        _review_input_token_budget = None

        def interrupt(self, message=None):
            pass

        def run_conversation(self, user_message, conversation_history=None):
            return {"final_response": json.dumps(self.reasoning_config), "messages": []}

    def _build(parent, task_cfg, *, max_iterations, write_origin):
        built.append(_Fork())
        built[-1].reasoning_config = parent.reasoning_config  # what the parity kwargs copy: the live level
        return built[-1], {}, False

    monkeypatch.setattr(background_review, "build_cache_parity_fork", _build)
    monkeypatch.setattr(background_review, "release_cache_parity_fork", lambda *a, **k: {})
    job = plugin_session_fork._ForkJob("probe", capture, "q", frozenset(), 1, None, {}, None,
                                       background_review._BackgroundReviewRun())
    job.handle = plugin_session_fork.PluginForkHandle("probe", lambda: None)
    plugin_session_fork._launch(job)
    assert job.handle.wait(10), job.handle
    assert job.handle.outcome == "ok", job.handle.outcome
    assert json.loads(job.handle.result) == adjusted


def test_a_capture_without_a_level_keeps_the_forks_own():
    """Captures built without ``reasoning_config`` (older callers) leave the fork's parity level alone."""
    agent = _stand_in()
    capture = SideQuestionCapture(parent_ref=weakref.ref(agent), messages=[], system_prompt="sys",
                                  system_prompt_static=None, tools=[], session_id="sess", turn_id="turn-1",
                                  runtime=runtime_key(agent), context=contextvars.copy_context())
    fork = SimpleNamespace(reasoning_config=BASELINE)
    plugin_session_fork._pin_capture(fork, capture)
    assert fork.reasoning_config is BASELINE
