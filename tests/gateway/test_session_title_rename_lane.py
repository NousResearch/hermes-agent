"""Which title stage is allowed to spend a platform rename.

Titling is two-stage: a derived slice of the user's own words lands inline, and
the model's version replaces it a moment later. A local sidebar wants both. A
Discord thread or a Telegram topic wants only the second — renaming twice lands
on the same name at twice the cost, and Discord allows two channel renames per
ten minutes, so the throwaway can be the one that survives.
"""

from __future__ import annotations

import types

import pytest

from gateway.config import Platform
from gateway.run import TurnRunner


def _attach(lane):
    """Attach the title callback for *lane* and return (callback, renames)."""
    renames: list = []
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="chan-1")

    runner = types.SimpleNamespace(
        _is_telegram_topic_lane=lambda src: lane == "telegram",
        _is_discord_auto_thread_lane=lambda src: lane == "discord",
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_telegram_topic_title_rename=(
            lambda src, sid, title: renames.append(title)
        ),
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append(title)
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _attach_session_title_callback=TurnRunner._attach_session_title_callback,
    )
    agent = types.SimpleNamespace(session_id="sess-1")
    holder._attach_session_title_callback(
        holder, agent, types.SimpleNamespace(source=source)
    )
    return getattr(agent, "_on_session_title", None), renames


def test_telegram_rename_waits_for_the_model_title():
    callback, renames = _attach("telegram")

    assert callable(callback)
    callback("fix the flaky auth test in log", "derived")
    assert renames == []

    callback("Fix flaky auth test", "llm")
    assert renames == ["Fix flaky auth test"]


def test_discord_does_not_schedule_rename_before_response_completion():
    callback, renames = _attach("discord")

    assert callback is None
    assert renames == []


def _discord_dispatch_fixture(monkeypatch):
    source = types.SimpleNamespace(platform=Platform.DISCORD, chat_id="thread-1")
    renames = []
    contextual_calls = []
    runner = types.SimpleNamespace(
        _is_discord_auto_thread_lane=lambda src: True,
        _is_relay_discord_channel_lane=lambda src: False,
        _schedule_discord_semantic_thread_rename=(
            lambda src, sid, title: renames.append((sid, title))
        ),
    )
    holder = types.SimpleNamespace(
        _runner=runner,
        _dispatch_discord_contextual_title=TurnRunner._dispatch_discord_contextual_title,
    )
    agent = types.SimpleNamespace(
        model="test-model",
        provider="test-provider",
        base_url=None,
        api_key=None,
        api_mode=None,
        _session_db=None,
    )
    ctx = types.SimpleNamespace(source=source, message="Summarize the recording")

    def contextual_title(session_db, session_id, opening_message, context, **kwargs):
        contextual_calls.append((opening_message, context))
        kwargs["title_callback"]("Agent workflow takeaways", "llm")

    monkeypatch.setattr(
        "agent.title_generator.maybe_generate_contextual_title", contextual_title
    )
    return holder, agent, ctx, renames, contextual_calls


def test_discord_schedules_contextual_rename_after_completed_response(monkeypatch):
    holder, agent, ctx, renames, contextual_calls = _discord_dispatch_fixture(monkeypatch)
    history = [{"role": "assistant", "content": "old answer"}]
    current_turn = [
        {"role": "user", "content": "Summarize the recording"},
        {"role": "tool", "content": "Transcript about reusable agent skills"},
        {"role": "assistant", "content": "The recording focuses on agent workflows"},
    ]
    result = {
        "completed": True,
        "final_response": "The recording focuses on agent workflows",
        "messages": history + current_turn,
    }

    holder._dispatch_discord_contextual_title(
        holder, agent, ctx, result, history, "sess-1"
    )

    assert contextual_calls == [("Summarize the recording", current_turn)]
    assert renames == [("sess-1", "Agent workflow takeaways")]


def test_discord_contextual_rename_is_one_shot_on_subsequent_turns(monkeypatch):
    holder, agent, ctx, renames, contextual_calls = _discord_dispatch_fixture(monkeypatch)
    result = {
        "completed": True,
        "final_response": "First answer",
        "messages": [
            {"role": "user", "content": "First request"},
            {"role": "assistant", "content": "First answer"},
        ],
    }

    holder._dispatch_discord_contextual_title(holder, agent, ctx, result, [], "sess-1")
    holder._dispatch_discord_contextual_title(holder, agent, ctx, result, [], "sess-1")

    assert len(contextual_calls) == 1
    assert renames == [("sess-1", "Agent workflow takeaways")]
