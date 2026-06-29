"""Tests for automatic Kanban worker activity journaling.

Dispatcher-spawned workers run as detached CLI processes. They should persist
observable callbacks into task_events so Kanban UIs can show an Agent0-style
execution story without exposing private chain-of-thought.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task():
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="journal me", assignee="worker")
        # create_task defaults to running for programmatic callers; dispatcher
        # workers need a current run, so start from ready and claim it.
        conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?", (tid,))
        conn.commit()
        task = kb.claim_task(conn, tid, claimer="test-lock")
        run = kb.latest_run(conn, tid)
        assert task is not None
        assert run is not None
        return tid, run.id
    finally:
        conn.close()


def _events_for(task_id: str):
    conn = kb.connect()
    try:
        return kb.list_events(conn, task_id)
    finally:
        conn.close()


def test_worker_journal_persists_observable_callbacks_from_env(kanban_home, monkeypatch):
    task_id, run_id = _running_task()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    from hermes_cli.kanban_worker_journal import KanbanWorkerJournal

    journal = KanbanWorkerJournal.from_env()
    assert journal is not None

    journal.tool_start("tc-1", "terminal", {"command": "echo hello", "api_key": "sk-secret"})
    journal.tool_complete("tc-1", "terminal", {"command": "echo hello"}, "hello\n")
    journal.tool_progress("status", name="terminal", preview="running tests")
    journal.interim_assistant("Tests OK, build en cours")

    events = [e for e in _events_for(task_id) if e.kind in {"tool_start", "tool_end", "progress_note", "assistant_text"}]
    assert [e.kind for e in events] == ["tool_start", "tool_end", "progress_note", "assistant_text"]
    assert all(e.run_id == run_id for e in events)

    assert events[0].payload["tool"] == "terminal"
    assert events[0].payload["tool_call_id"] == "tc-1"
    assert events[0].payload["args"]["command"] == "echo hello"
    assert "sk-secret" not in str(events[0].payload)

    assert events[1].payload["tool"] == "terminal"
    assert "hello" in events[1].payload["result_preview"]

    assert events[2].payload == {"note": "running tests", "source": "tool_progress", "name": "terminal"}
    assert events[3].payload == {"text": "Tests OK, build en cours", "source": "interim_assistant"}


def test_worker_journal_is_noop_without_kanban_task(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    from hermes_cli.kanban_worker_journal import KanbanWorkerJournal

    assert KanbanWorkerJournal.from_env() is None


def test_worker_journal_swallow_errors_so_callbacks_cannot_break_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_missing")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "not-an-int")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/dev/null/kanban.db")

    from hermes_cli.kanban_worker_journal import KanbanWorkerJournal

    journal = KanbanWorkerJournal.from_env()
    assert journal is not None

    # Missing/corrupt board context must not raise into the agent callback path.
    journal.tool_start("tc", "terminal", {"command": "pwd"})
    journal.tool_complete("tc", "terminal", {}, "ok")
    journal.tool_progress("status", preview="still working")
    journal.interim_assistant("visible worker text")


def test_install_on_agent_composes_existing_callbacks(kanban_home, monkeypatch):
    task_id, run_id = _running_task()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    from hermes_cli.kanban_worker_journal import install_on_agent

    calls = []

    class Agent:
        def tool_start_callback(self, tool_call_id, name, args):
            calls.append(("start", tool_call_id, name, args))

        def tool_complete_callback(self, tool_call_id, name, args, result):
            calls.append(("complete", tool_call_id, name, args, result))

        def tool_progress_callback(self, event_type, name=None, preview=None, args=None, **kwargs):
            calls.append(("progress", event_type, name, preview, args, kwargs))

        def interim_assistant_callback(self, text, **kwargs):
            calls.append(("assistant", text, kwargs))

        def stream_delta_callback(self, delta):
            calls.append(("stream", delta))

    agent = Agent()
    install_on_agent(agent)

    agent.tool_start_callback("tc-2", "read_file", {"path": "AGENTS.md"})
    agent.tool_complete_callback("tc-2", "read_file", {"path": "AGENTS.md"}, "content")
    agent.tool_progress_callback("status", name="read_file", preview="reading")
    agent.interim_assistant_callback("J’ai lu le contexte", already_streamed=False)
    agent.stream_delta_callback("Réponse ")
    agent.stream_delta_callback("streamée")
    agent.stream_delta_callback(None)

    assert calls == [
        ("start", "tc-2", "read_file", {"path": "AGENTS.md"}),
        ("complete", "tc-2", "read_file", {"path": "AGENTS.md"}, "content"),
        ("progress", "status", "read_file", "reading", None, {}),
        ("assistant", "J’ai lu le contexte", {"already_streamed": False}),
        ("stream", "Réponse "),
        ("stream", "streamée"),
        ("stream", None),
    ]

    events = _events_for(task_id)
    kinds = [e.kind for e in events]
    assert "tool_start" in kinds
    assert "tool_end" in kinds
    assert "progress_note" in kinds
    assert "assistant_text" in kinds
    assistant_payloads = [e.payload for e in events if e.kind == "assistant_text"]
    assert {p["source"] for p in assistant_payloads} == {"interim_assistant", "stream_delta"}


def test_record_final_assistant_flushes_stream_and_deduplicates(kanban_home, monkeypatch):
    task_id, run_id = _running_task()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    from hermes_cli.kanban_worker_journal import install_on_agent, record_final_assistant

    class Agent:
        def stream_delta_callback(self, delta):
            pass

    agent = Agent()
    install_on_agent(agent)

    agent.stream_delta_callback("Même réponse")
    record_final_assistant(agent, "Même réponse")
    record_final_assistant(agent, "Même réponse")

    assistant_events = [e for e in _events_for(task_id) if e.kind == "assistant_text"]
    assert len(assistant_events) == 1
    assert assistant_events[0].payload == {"text": "Même réponse", "source": "stream_delta"}


def test_cli_init_agent_installs_worker_journal(kanban_home, monkeypatch):
    task_id, run_id = _running_task()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    import cli as cli_mod
    from hermes_cli import mcp_startup

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tool_progress_callback = kwargs.get("tool_progress_callback")
            self.tool_start_callback = kwargs.get("tool_start_callback")
            self.tool_complete_callback = kwargs.get("tool_complete_callback")
            self.stream_delta_callback = kwargs.get("stream_delta_callback")
            self.interim_assistant_callback = kwargs.get("interim_assistant_callback")

    monkeypatch.setattr(cli_mod, "AIAgent", FakeAgent)
    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda timeout=0.75: None)

    shell = cli_mod.HermesCLI(compact=True, max_turns=1)
    shell._session_db = object()
    shell._resumed = False
    shell.conversation_history = []
    shell._install_tool_callbacks = lambda: None
    shell._ensure_tirith_security = lambda: None
    shell._ensure_runtime_credentials = lambda: True

    assert shell._init_agent() is True
    assert getattr(shell.agent, "_kanban_worker_journal_installed") is True

    shell.agent.tool_progress_callback("status", name="init", preview="journal installed")
    shell.agent.interim_assistant_callback("Initialisation worker visible")

    payloads = [(e.kind, e.payload) for e in _events_for(task_id)]
    assert (
        "progress_note",
        {"note": "journal installed", "source": "tool_progress", "name": "init"},
    ) in payloads
    assert (
        "assistant_text",
        {"text": "Initialisation worker visible", "source": "interim_assistant"},
    ) in payloads
