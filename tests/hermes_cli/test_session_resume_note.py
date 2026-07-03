"""Production-path coverage for cross-process session resume context."""

from __future__ import annotations

import types
from contextlib import nullcontext
from typing import Any, cast

import pytest

from agent.turn_context import _merge_gateway_notes, consume_session_resume_note
from hermes_cli import session_resume
from hermes_cli.session_resume import SESSION_RESUME_NOTE, history_predates_process


def _history(timestamp: float) -> list[dict]:
    return [
        {"role": "user", "content": "before", "timestamp": timestamp - 1},
        {"role": "assistant", "content": "done", "timestamp": timestamp},
    ]


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        (_history(999.0), True),
        (_history(1001.0), False),
        ([{"role": "assistant", "content": "no timestamp"}], False),
        (
            [
                {"role": "assistant", "content": "old", "timestamp": 999.0},
                {"role": "session_meta", "content": "synthetic"},
            ],
            True,
        ),
    ],
)
def test_restart_detection_uses_newest_trustworthy_history_timestamp(history, expected):
    assert history_predates_process(history, process_start=1000.0) is expected


def test_resume_note_uses_one_shot_user_sidecar_channel():
    """No synthetic role or system-prompt mutation; the note is consumed once."""
    cached_prompt = object()
    agent = types.SimpleNamespace(
        _cached_system_prompt=cached_prompt,
        _session_resume_note=SESSION_RESUME_NOTE,
    )
    messages = [
        {"role": "assistant", "content": "restored answer"},
        {"role": "user", "content": "continue"},
    ]
    roles_before = [message["role"] for message in messages]

    plugin_context = _merge_gateway_notes(agent, messages, 1, "")

    assert plugin_context == SESSION_RESUME_NOTE
    assert consume_session_resume_note(agent) == ""
    assert _merge_gateway_notes(agent, messages, 1, "") == ""
    assert [message["role"] for message in messages] == roles_before
    assert agent._cached_system_prompt is cached_prompt


def test_cli_in_process_resume_owner_stages_after_session_reset(monkeypatch):
    # Upstream moved the switch to hermes_cli.cli_session_switch.py and moved the
    # turn into the gateway process: the CLI-side agent is a GatewayAgent RPC
    # proxy, so the resume note is staged where the real agent lives (gateway
    # session.resume -> _resume_eager / _attach_built_agent, covered above).
    # Here: the switch still repoints and resets, and never stages client-side.
    from hermes_cli.cli_session_switch import _sync_agent_to_session

    monkeypatch.setattr(session_resume, "PROCESS_START", 1000.0)

    class Agent:
        session_id = "old"
        _last_flushed_db_idx = 0
        _memory_manager = None
        _session_resume_note = "stale"

        def reset_session_state(self):
            self._session_resume_note = ""

        def _invalidate_system_prompt(self):
            pass

    cli = types.SimpleNamespace(agent=Agent(), conversation_history=_history(999.0))
    _sync_agent_to_session(
        cli, "resumed", parent_session_id="old", reason="resume"
    )

    assert cli.agent.session_id == "resumed"
    assert cli.agent._last_flushed_db_idx == len(cli.conversation_history)
    # The boundary cleared the stale note and staged nothing: the gateway's
    # live session owns note delivery for the resumed conversation.
    assert cli.agent._session_resume_note == ""


def test_tui_deferred_build_owner_stages_note_from_hydrated_history(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(session_resume, "PROCESS_START", 1000.0)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _current: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda _current: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("model", "provider"))

    current = {
        "history": _history(999.0),
        "pending_title": None,
        "resume_session_id": "stored-session",
    }
    agent = types.SimpleNamespace()
    sid = "resume-note-deferred"
    monkeypatch.setitem(server._sessions, sid, current)
    assert server._attach_built_agent(sid, current, agent)

    assert current["agent"] is agent
    assert agent._session_resume_note == SESSION_RESUME_NOTE


def test_tui_deferred_build_ignores_same_process_history(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(session_resume, "PROCESS_START", 1000.0)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _current: None)
    monkeypatch.setattr(server, "_session_todo_state", lambda _current: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("model", "provider"))

    current = {
        "history": _history(1001.0),
        "pending_title": None,
        "resume_session_id": "stored-session",
    }
    agent = types.SimpleNamespace()
    sid = "resume-note-in-process"
    monkeypatch.setitem(server._sessions, sid, current)
    assert server._attach_built_agent(sid, current, agent)

    assert not hasattr(agent, "_session_resume_note")


def test_tui_eager_resume_owner_stages_note(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(session_resume, "PROCESS_START", 1000.0)
    server._sessions.clear()
    agent = types.SimpleNamespace()
    history = _history(999.0)

    class ResumeContext:
        profile_home = None
        profile_resume_cwd = ""
        target = "stored-session"
        db = object()
        found = {}
        cols = 80
        owns_db = False

        @staticmethod
        def mint():
            return "runtime-session", "tui", ""

        @staticmethod
        def restore():
            return history, history, history

        @staticmethod
        def display_prefix():
            return []

    monkeypatch.setattr(server, "_profile_build_scope", lambda _home: nullcontext())
    monkeypatch.setattr(server, "_make_agent_in_context", lambda *_a, **_k: agent)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_transport_auth_user_id", lambda _transport: None)
    monkeypatch.setattr(server, "current_transport", lambda: None)
    monkeypatch.setattr(server, "_stored_session_runtime_overrides", lambda _row: {})

    def _init_session(sid, key, built_agent, restored, **_kwargs):
        server._sessions[sid] = {
            "agent": built_agent,
            "history": restored,
            "created_at": 1001.0,
        }

    monkeypatch.setattr(server, "_init_session", _init_session)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_resume_response", lambda *_a, **_k: {"ok": True})

    assert getattr(server, "_resume_eager")(ResumeContext()) == {"ok": True}
    assert agent._session_resume_note == SESSION_RESUME_NOTE


def test_session_boundary_clears_unconsumed_note():
    from run_agent import AIAgent

    agent = types.SimpleNamespace(
        _session_resume_note=SESSION_RESUME_NOTE,
        context_compressor=None,
        session_id="next",
        _transition_context_engine_session=lambda **_kwargs: None,
    )

    AIAgent.reset_session_state(cast(Any, agent))

    assert agent._session_resume_note == ""
