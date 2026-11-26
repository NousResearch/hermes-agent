"""``hermes sessions refresh-prompts`` reaches an agent that writes through a storage interface handle, as the
messaging gateway's agents do: the next turn rebuilds the stored prompt once and persists it with the epoch."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.agent.test_session_prompt_refresh import (
    _assert_stale, _assert_updated, _refresh_prompts, _system, _update_profile, _user_texts, _write_profile,
)
from tests.fakes.fake_llm_provider import FakeLLMServer


@pytest.fixture
def profile(monkeypatch, tmp_path):
    home = Path(os.environ["HERMES_HOME"])
    project = tmp_path / "project"
    with FakeLLMServer() as srv:
        _write_profile(home, srv.base_url, project)
        monkeypatch.setenv("TERMINAL_CWD", str(project))
        yield srv, home, project


def _turn(srv, home, session_id, text, history):
    """One gateway turn on a fresh AIAgent holding a writer's storage handle. Returns (history, system)."""
    from persistence.conversation.sessions import NewSession
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent

    store = open_persistence(profile_scope(home=home), eager=True)
    try:
        store.sessions.create(NewSession(session_id=session_id, source="whatsapp"))
        agent = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                        persistence=store, session_id=session_id, quiet_mode=True, platform="whatsapp")
        try:
            messages = agent.run_conversation(text, conversation_history=history, task_id="t")["messages"]
        finally:
            agent.close()
    finally:
        store.close()
    return messages, _system(srv.main_requests()[-1])


def _stored(home, session_id):
    from persistence_assembly import open_persistence, profile_scope

    with open_persistence(profile_scope(home=home), eager=True) as store:
        record = store.sessions.get(session_id)
        return record.system_prompt, dict(record.model_config)


def test_the_verb_makes_the_next_turn_adopt_the_updated_sources(profile, monkeypatch, capsys):
    srv, home, project = profile
    history, first = _turn(srv, home, "chat-a", "hello", [])
    _assert_stale(first)
    _update_profile(home, project)

    _refresh_prompts(monkeypatch, home, "--json")
    request = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert request["sessions"] == 1 and request["epoch"] >= 1

    history, refreshed = _turn(srv, home, "chat-a", "anything new?", history)
    _assert_updated(refreshed)
    assert _user_texts(srv.main_requests()[-1])[:2] == ["hello", "anything new?"]
    prompt, model_config = _stored(home, "chat-a")
    assert prompt == refreshed and model_config.get("prompt_refresh_epoch") == request["epoch"]


def test_the_refreshed_prompt_is_restored_without_another_rebuild(profile, monkeypatch):
    from run_agent import AIAgent

    srv, home, project = profile
    history, _ = _turn(srv, home, "chat-a", "hello", [])
    _update_profile(home, project)
    _refresh_prompts(monkeypatch, home)
    history, refreshed = _turn(srv, home, "chat-a", "anything new?", history)
    _assert_updated(refreshed)

    builds = []
    original = AIAgent._build_system_prompt
    monkeypatch.setattr(AIAgent, "_build_system_prompt",
                        lambda self, *a, **k: builds.append(1) or original(self, *a, **k))
    _, after = _turn(srv, home, "chat-a", "and now?", history)
    assert after == refreshed and builds == []


def test_session_flag_records_a_request_for_that_session_only(profile, monkeypatch):
    srv, home, project = profile
    history_a, _ = _turn(srv, home, "chat-a", "hello", [])
    history_b, _ = _turn(srv, home, "chat-b", "hello", [])
    _update_profile(home, project)

    _refresh_prompts(monkeypatch, home, "--session", "chat-a")

    _, system_a = _turn(srv, home, "chat-a", "next", history_a)
    _, system_b = _turn(srv, home, "chat-b", "next", history_b)
    _assert_updated(system_a)
    _assert_stale(system_b)
