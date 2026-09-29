"""Behavior contract for the post-turn review trigger (agent/review_trigger.py).

The turn-count clock decides; a ``request_background_review`` subscriber may ADD a review the clock
would not fire yet, never suppress one, and never bypass a kind's kill switch. Without a subscriber
the trigger is exactly the clock. Exercised through ``finalize_turn`` — the production call path —
with the judgment thread run inline.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent.review_trigger as review_trigger
import hermes_cli.plugins as plugins_mod
from agent.review_trigger import parse_review_request
from agent.turn_finalizer import finalize_turn
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _inline_judgment(monkeypatch):
    monkeypatch.setattr(review_trigger, "_start_thread", lambda target: target())


def _agent(*, skip_background_review=False, memory_interval=10, skill_interval=10) -> AIAgent:
    agent = AIAgent(
        model="openai/gpt-4o-mini", provider="openrouter", api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True,
        skip_memory=True, skip_background_review=skip_background_review, platform="cli",
    )
    agent._spawn_background_review = MagicMock()
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._session_messages = []
    agent._file_mutation_verifier_enabled = lambda: False
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = None
    agent._sync_external_memory_for_turn = MagicMock()
    agent.iteration_budget = MagicMock(remaining=100, used=5, max_total=100)
    agent.max_iterations = 50
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._apply_persist_user_message_override = MagicMock()
    agent.context_compressor = None
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    agent.model = "test-model"
    agent.session_id = "test-session"
    agent._turn_failed_file_mutations = {}
    agent._db_flush_scan_prefix = None
    # Both review kinds enabled, neither clock due.
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent._memory_store = object()
    agent._memory_nudge_interval = memory_interval
    agent._skill_nudge_interval = skill_interval
    agent._turns_since_memory = 3
    agent._iters_since_skill = 2
    return agent


_TRANSCRIPT = [
    {"role": "user", "content": "set up the eval"},
    {"role": "assistant", "content": "I'll fine-tune first, then build the eval."},
    {"role": "user", "content": "no - we always research before building. eval first."},
    {"role": "assistant", "content": "Understood, eval first."},
]


def _finalize(agent: AIAgent, *, clock_memory=False) -> None:
    finalize_turn(
        agent, final_response="Understood, eval first.", api_call_count=1, interrupted=False,
        failed=False, messages=list(_TRANSCRIPT), conversation_history=[], effective_task_id="t",
        turn_id="turn-1", user_message=_TRANSCRIPT[2]["content"],
        original_user_message=_TRANSCRIPT[2]["content"], _should_review_memory=clock_memory,
        _turn_exit_reason="text_response(1)",
    )


def _subscriber(monkeypatch, answer):
    """Install a request_background_review subscriber at the call site; returns the payload log."""
    calls = []

    def _invoke(name, **kwargs):
        if name != "request_background_review":
            return []
        calls.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return [answer] if answer is not None else []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "request_background_review")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _invoke)
    return calls


def _spawned(agent):
    """(review_memory, review_skills) of each spawned review."""
    return [(c.kwargs["review_memory"], c.kwargs["review_skills"]) for c in agent._spawn_background_review.call_args_list]


def test_without_a_subscriber_the_trigger_is_the_clock(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: False)
    quiet, due = _agent(), _agent()
    _finalize(quiet, clock_memory=False)
    _finalize(due, clock_memory=True)
    assert _spawned(quiet) == []
    assert _spawned(due) == [(True, False)]


def test_a_judged_lesson_reviews_before_the_clock_and_resets_it(monkeypatch):
    calls = _subscriber(monkeypatch, {"review": "memory"})
    agent = _agent()
    _finalize(agent)
    assert _spawned(agent) == [(True, False)]
    assert agent._turns_since_memory == 0  # the next clock review measures from this one
    assert agent._iters_since_skill == 2  # the kind not asked for keeps its own clock
    # The subscriber judged the actual exchange: the reply and what it was replying to.
    assert calls[0]["user_message"] == _TRANSCRIPT[2]["content"]
    assert calls[0]["previous_assistant"] == _TRANSCRIPT[1]["content"]
    assert calls[0]["clock_memory"] is False


@pytest.mark.parametrize("answer", [None, {"review": []}, {"review": "nothing"}, "yes", RuntimeError("down"),
                                    {"skip": "memory"}])
def test_a_subscriber_cannot_suppress_the_clock(monkeypatch, answer):
    _subscriber(monkeypatch, answer)
    agent = _agent()
    _finalize(agent, clock_memory=True)
    assert _spawned(agent) == [(True, False)]


def test_a_kind_that_is_switched_off_never_runs_whoever_asks(monkeypatch):
    # skills.creation_nudge_interval: 0 is the documented kill switch (#82708).
    _subscriber(monkeypatch, {"review": ["memory", "skills"]})
    agent = _agent(skill_interval=0)
    _finalize(agent)
    assert _spawned(agent) == [(True, False)]


def test_no_review_when_every_asked_kind_is_off(monkeypatch):
    _subscriber(monkeypatch, {"review": "skills"})
    agent = _agent(skill_interval=0)
    _finalize(agent)
    assert _spawned(agent) == []


def test_subscriber_is_not_consulted_when_it_could_add_nothing(monkeypatch):
    calls = _subscriber(monkeypatch, {"review": "memory"})
    agent = _agent(skill_interval=0)
    _finalize(agent, clock_memory=True)  # memory already due; skills switched off
    assert calls == []
    assert _spawned(agent) == [(True, False)]


def test_subagents_and_unattended_turns_never_consult_the_subscriber(monkeypatch):
    calls = _subscriber(monkeypatch, {"review": "memory"})
    child = _agent()
    child._delegate_depth = 1
    cron = _agent(skip_background_review=True)
    _finalize(child)
    _finalize(cron)
    assert calls == []
    assert _spawned(child) == [] and _spawned(cron) == []


def test_real_plugin_through_discovery_triggers_a_review(monkeypatch):
    """E2E: a plugin loaded from HERMES_HOME by the real discovery path asks for a review."""
    import hermes_yaml as yaml

    home = Path(os.environ["HERMES_HOME"])
    plugin_dir = home / "plugins" / "lesson_gate_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: lesson_gate_probe\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def judge(user_message='', **kw):\n"
        "        return {'review': 'memory'} if 'always research' in user_message else None\n"
        "    ctx.register_hook('request_background_review', judge)\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["lesson_gate_probe"]}}), encoding="utf-8")
    monkeypatch.setattr(plugins_mod, "_plugin_manager", plugins_mod.PluginManager())
    plugins_mod.discover_plugins()

    agent = _agent()
    _finalize(agent)
    assert _spawned(agent) == [(True, False)]


@pytest.mark.parametrize("results, expected", [
    ([{"review": "memory"}], {"memory"}),
    ([{"review": ["skills", "memory"]}], {"memory", "skills"}),
    ([{"review": "memory"}, {"review": "skills"}], {"memory", "skills"}),
    ([None, "memory", {"review": ["bogus"]}, {"other": 1}], set()),
])
def test_parse_review_request(results, expected):
    assert parse_review_request(results) == expected


def test_shell_hook_can_request_a_review():
    from agent.shell_hooks import _parse_response

    assert _parse_response("request_background_review", '{"review": ["skills", "bogus"]}') == {"review": ["skills"]}
    assert _parse_response("request_background_review", '{"review": []}') is None


# --- Opt-in: auxiliary.background_review.judgment_can_skip ------------------------------------------


def _can_skip(monkeypatch, value=True):
    monkeypatch.setattr(review_trigger, "_review_settings", lambda: (True, value))


def test_with_skip_enabled_a_judged_empty_window_drops_the_clock_review(monkeypatch):
    _can_skip(monkeypatch)
    calls = _subscriber(monkeypatch, {"skip": ["memory", "skills"]})
    agent = _agent()
    _finalize(agent, clock_memory=True)
    assert _spawned(agent) == []
    assert calls[0]["clock_can_be_skipped"] is True


def test_skip_only_drops_the_kinds_it_names(monkeypatch):
    _can_skip(monkeypatch)
    _subscriber(monkeypatch, {"skip": "skills"})
    agent = _agent()
    agent._iters_since_skill = 20  # skills clock due too
    _finalize(agent, clock_memory=True)
    assert _spawned(agent) == [(True, False)]


def test_a_request_beats_a_skip_from_another_subscriber(monkeypatch):
    _can_skip(monkeypatch)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "request_background_review")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook",
                        lambda name, **kw: [{"skip": "memory"}, {"review": "memory"}] if name == "request_background_review" else [])
    agent = _agent()
    _finalize(agent, clock_memory=True)
    assert _spawned(agent) == [(True, False)]


def test_skip_enabled_but_subscriber_fails_keeps_the_clock_review(monkeypatch):
    _can_skip(monkeypatch)
    _subscriber(monkeypatch, RuntimeError("judge down"))
    agent = _agent()
    _finalize(agent, clock_memory=True)
    assert _spawned(agent) == [(True, False)]


def test_skip_from_real_config_and_real_plugin(monkeypatch):
    """E2E: judgment_can_skip read from the real config.yaml; the plugin loaded by real discovery."""
    import hermes_yaml as yaml

    home = Path(os.environ["HERMES_HOME"])
    plugin_dir = home / "plugins" / "lesson_gate_skip_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: lesson_gate_skip_probe\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_hook('request_background_review', lambda **kw: {'skip': ['memory', 'skills']})\n",
        encoding="utf-8",
    )
    cfg = {"plugins": {"enabled": ["lesson_gate_skip_probe"]}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(plugins_mod, "_plugin_manager", plugins_mod.PluginManager())
    plugins_mod.discover_plugins()

    default_agent = _agent()
    _finalize(default_agent, clock_memory=True)
    assert _spawned(default_agent) == [(True, False)]  # off by default: the clock review runs

    cfg["auxiliary"] = {"background_review": {"judgment_can_skip": True}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    opted_in = _agent()
    _finalize(opted_in, clock_memory=True)
    assert _spawned(opted_in) == []


def test_shell_hook_can_request_a_skip():
    from agent.shell_hooks import _parse_response

    assert _parse_response("request_background_review", '{"skip": "skills"}') == {"skip": ["skills"]}
