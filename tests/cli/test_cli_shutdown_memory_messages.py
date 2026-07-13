"""Regression tests for #15165 (CLI sibling site) — CLI exit cleanup must
forward the agent's conversation transcript to ``shutdown_memory_provider``
so memory providers' ``on_session_end`` hooks see the real messages.

Before the fix, ``_run_cleanup`` called
``shutdown_memory_provider(getattr(agent, 'conversation_history', None) or [])``.
``AIAgent`` has no ``conversation_history`` attribute — so the ``or []``
branch always fired and providers got an empty list on CLI exit. This
mirrors the gateway bug fixed in the same commit (gateway/run.py uses
``_session_messages``, which IS set on ``AIAgent``).

The fix reads ``_session_messages`` (same attribute the gateway path uses)
with an ``isinstance(..., list)`` guard so MagicMock-based agents in
other tests keep their existing no-arg behaviour.
"""

from __future__ import annotations

import threading
import types
from typing import Any
from unittest.mock import MagicMock, patch


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_forwards_session_messages(mock_invoke_hook):
    """_run_cleanup forwards a populated ``_session_messages`` list."""
    import cli as cli_mod

    transcript = [
        {"role": "user", "content": "remember my dog is named Biscuit"},
        {"role": "assistant", "content": "Got it — Biscuit."},
    ]

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = transcript

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with(transcript)


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_empty_list_still_forwarded(mock_invoke_hook):
    """An agent that initialised but ran no turns has an empty list.
    Forwarding it (rather than falling through) matches the gateway-side
    behaviour and is explicit to providers."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = []

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with([])


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_non_list_attribute_falls_back_to_no_arg(mock_invoke_hook):
    """A MagicMock agent auto-synthesises ``_session_messages`` as a
    nested MagicMock. ``isinstance(mock, list)`` is False, so we fall
    back to the no-arg path rather than passing a garbage value to
    providers expecting ``List[Dict]``.  This keeps existing CLI test
    suites that use bare ``MagicMock()`` agents green."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    # No explicit _session_messages — MagicMock synthesises one on access.

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with()


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_provider_exception_is_swallowed(mock_invoke_hook):
    """A raising ``shutdown_memory_provider`` must not crash CLI exit."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = [{"role": "user", "content": "x"}]
    agent.shutdown_memory_provider.side_effect = RuntimeError("boom")

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()  # must not raise
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once()


def test_cli_close_persists_agent_session_messages_before_end_session():
    """CLI shutdown flushes live agent messages before closing the session."""
    import cli as cli_mod

    transcript = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "partial answer"},
    ]
    conversation_history = [{"role": "user", "content": "long task"}]

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = conversation_history
    cli.session_id = "old-session"
    agent = MagicMock()
    agent.session_id = "live-session"
    agent._session_messages = transcript
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_called_once_with(transcript, conversation_history)
    assert cli.session_id == "live-session"


def test_cli_close_persist_falls_back_to_conversation_history():
    """Bare MagicMock agents do not provide a real _session_messages list."""
    import cli as cli_mod

    conversation_history = [{"role": "user", "content": "saved from cli"}]
    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = conversation_history
    cli.session_id = "session-id"
    agent = MagicMock()
    agent.session_id = "session-id"
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_called_once_with(conversation_history, None)


def test_cli_close_persist_skips_empty_transcripts():
    """Do not create empty session writes for idle CLI startup/shutdown."""
    import cli as cli_mod

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = []
    cli.session_id = "session-id"
    agent = MagicMock()
    agent.session_id = "session-id"
    agent._session_messages = []
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_not_called()


def test_cli_close_uses_distinct_history_as_baseline():
    """A pre-flush shutdown keeps the distinct CLI prefix as a DB baseline."""
    import cli as cli_mod

    history = [{"role": "user", "content": "resumed prompt"}]
    live_messages = history + [{"role": "assistant", "content": "partial response"}]
    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = history
    cli.session_id = "session-id"
    agent = MagicMock()
    agent.session_id = "session-id"
    agent._session_messages = live_messages
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_called_once_with(live_messages, history)


def _real_agent(db, session_id, session_messages):
    """Build the real persistence seam without the heavyweight LLM client."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent._session_messages = session_messages
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    agent._cached_system_prompt = "test system prompt"
    agent._session_init_model_config = None
    agent._parent_session_id = None
    agent._session_json_enabled = False
    agent._pending_cli_user_message = None
    agent._session_persist_lock = threading.RLock()
    return agent


def test_cli_close_persist_real_db_survives_history_alias(tmp_path, monkeypatch):
    """CLI close safety-net must persist even when history aliases messages.

    In the real CLI, ``conversation_history`` and ``agent._session_messages`` can
    point at the same live list during interrupted shutdown.  Passing that list
    as ``conversation_history`` makes ``_flush_messages_to_session_db`` treat
    every message as already durable and write zero rows.  The close safety-net
    should use marker-based dedup instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-alias"
    db.create_session(session_id=session_id, source="cli")

    transcript = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "partial answer"},
    ]

    agent = _real_agent(db, session_id, transcript)

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = transcript
    cli.session_id = "old-session"
    cli.agent = agent

    assert db.get_messages_as_conversation(session_id) == []

    cli._persist_active_session_before_close()

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == ["long task", "partial answer"]
    assert cli.session_id == session_id


def test_cli_close_preflush_resumed_prefix_is_not_duplicated(tmp_path, monkeypatch):
    """A signal during the turn-start flush preserves the old DB prefix once.

    The pause is after ``_persist_session`` records its live snapshot but before
    its normal DB flush. The close helper must retain the distinct CLI baseline.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-preflush-resume"
    db.create_session(session_id=session_id, source="cli")
    loaded = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    for message in loaded:
        db.append_message(
            session_id=session_id,
            role=message["role"],
            content=message["content"],
        )

    live_messages = list(loaded) + [{"role": "user", "content": "new prompt"}]
    agent = _real_agent(db, session_id, [])
    entered_flush = threading.Event()
    release_flush = threading.Event()
    flush_calls = 0

    def _pause_before_flush(
        messages: list[dict[str, Any]],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> None:
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 1:
            # The worker has assigned its snapshot and is now paused before its
            # regular DB write. The concurrent close call must stay live.
            agent._session_messages = messages
            entered_flush.set()
            assert release_flush.wait(timeout=5)
        from run_agent import AIAgent

        # Runtime accepts None; the stub keeps that optional contract explicit.
        return AIAgent._flush_messages_to_session_db(
            agent,
            messages,
            conversation_history if conversation_history is not None else [],
        )

    agent._flush_messages_to_session_db = _pause_before_flush
    worker = threading.Thread(
        target=lambda: agent._persist_session(live_messages, loaded),
        daemon=True,
    )
    worker.start()
    assert entered_flush.wait(timeout=5)

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = list(loaded) + [{"role": "user", "content": "ui prompt"}]
    cli.session_id = session_id
    cli.agent = agent
    close_started = threading.Event()
    close_finished = threading.Event()

    def _close_while_worker_flushes():
        close_started.set()
        cli._persist_active_session_before_close()
        close_finished.set()

    close_worker = threading.Thread(target=_close_while_worker_flushes, daemon=True)
    close_worker.start()
    assert close_started.wait(timeout=5)
    # The per-agent persistence lock holds the close flush until the normal
    # turn-start write has stamped its durable markers.
    assert not close_finished.wait(timeout=0.1)

    release_flush.set()
    worker.join(timeout=5)
    close_worker.join(timeout=5)
    assert not worker.is_alive()
    assert not close_worker.is_alive()

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]


def test_cli_close_preserves_unflushed_tail_after_prior_prefix_flush(tmp_path, monkeypatch):
    """Marker-only alias close writes only a new tail after a prior flush."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-tail"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _real_agent(db, session_id, prefix)
    agent._flush_messages_to_session_db(prefix, [])
    live_messages = prefix + [{"role": "assistant", "content": "new tail"}]
    agent._session_messages = live_messages

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = live_messages
    cli.session_id = session_id
    cli.agent = agent

    cli._persist_active_session_before_close()

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new tail",
    ]


def test_cli_close_hands_staged_user_marker_to_turn_start(tmp_path, monkeypatch):
    """A close before turn setup does not duplicate the CLI-staged user row."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-staged-user"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _real_agent(db, session_id, prefix)
    agent._flush_messages_to_session_db(prefix, [])
    staged = {"role": "user", "content": "new prompt"}
    # `chat()` copies a completed agent transcript before it stages the next
    # user input, so close initially sees the prior agent snapshot only.
    cli_history = list(prefix) + [staged]
    agent._pending_cli_user_message = staged

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = cli_history
    cli.session_id = session_id
    cli.agent = agent

    # Close appends only the pending UI dict, while treating the durable prefix
    # as its baseline. Turn setup then reuses the marked dict without re-writing.
    cli._persist_active_session_before_close()
    assert staged["_db_persisted"] is True

    worker_messages = list(prefix) + [staged]
    agent._persist_session(worker_messages, prefix)

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]


def test_cli_chat_staging_does_not_mutate_live_agent_snapshot():
    """The next CLI input must be outside the prior live agent transcript."""
    import cli as cli_mod

    previous = [{"role": "assistant", "content": "done"}]
    agent = MagicMock()
    agent._session_messages = previous
    agent._pending_cli_user_message = None

    cli = object.__new__(cli_mod.HermesCLI)
    cli.agent = agent
    cli.conversation_history = previous

    # Model the narrow staging operation in ``chat`` without starting a provider.
    if cli.conversation_history is agent._session_messages:
        cli.conversation_history = list(cli.conversation_history)
    staged = {"role": "user", "content": "next"}
    agent._pending_cli_user_message = staged
    cli.conversation_history.append(staged)

    assert agent._session_messages == [{"role": "assistant", "content": "done"}]
    assert cli.conversation_history == [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]


def test_cli_close_persists_pending_user_when_agent_snapshot_is_empty(tmp_path, monkeypatch):
    """Close before worker startup persists only the CLI-staged user input."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-before-worker"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    for message in prefix:
        db.append_message(
            session_id=session_id,
            role=message["role"],
            content=message["content"],
        )

    agent = _real_agent(db, session_id, [])
    staged = {"role": "user", "content": "new prompt"}
    agent._pending_cli_user_message = staged

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = list(prefix) + [staged]
    cli.session_id = session_id
    cli.agent = agent

    cli._persist_active_session_before_close()

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]
    assert staged["_db_persisted"] is True


def test_cli_close_preserves_clean_staged_user_across_noted_worker_turn(tmp_path, monkeypatch):
    """A noted API-only turn reuses the close-marked clean staged user row."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-noted-staged-user"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _real_agent(db, session_id, prefix)
    agent._flush_messages_to_session_db(prefix, [])
    staged = {"role": "user", "content": "new prompt"}
    agent._pending_cli_user_message = staged

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = list(prefix) + [staged]
    cli.session_id = session_id
    cli.agent = agent

    cli._persist_active_session_before_close()
    assert staged["_db_persisted"] is True

    # A queued model/skills note changes only the API message. The worker
    # reuses the marked clean dict, so the normal persistence seam cannot append
    # a second noted user row.
    from agent.turn_context import build_turn_context

    agent.quiet_mode = True
    agent.max_iterations = 1
    agent.provider = "test"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = "chat_completions"
    agent.tools = []
    agent.valid_tool_names = set()
    agent.enabled_toolsets = None
    agent.disabled_toolsets = None
    agent._skip_mcp_refresh = True
    agent.compression_enabled = False
    agent.context_compressor = types.SimpleNamespace(protect_first_n=2, protect_last_n=2)
    agent._memory_store = None
    agent._memory_manager = None
    agent._memory_nudge_interval = 0
    agent._turns_since_memory = 0
    agent._user_turn_count = 0
    agent._todo_store = types.SimpleNamespace(has_items=lambda: True)
    agent._tool_guardrails = types.SimpleNamespace(reset_for_turn=lambda: None)
    agent._compression_warning = None
    agent._interrupt_requested = False
    agent._memory_write_origin = "assistant_tool"
    agent._stream_context_scrubber = None
    agent._stream_think_scrubber = None
    agent._restore_primary_runtime = lambda: None
    agent._cleanup_dead_connections = lambda: False
    agent._emit_status = lambda _message: None
    agent._replay_compression_warning = lambda: None
    agent._hydrate_todo_store = lambda *_args: None
    agent._safe_print = lambda *_args: None

    worker = build_turn_context(
        agent,
        "[MODEL SWITCH NOTE]\n\nnew prompt",
        None,
        prefix,
        "task",
        None,
        "new prompt",
        None,
        restore_or_build_system_prompt=lambda *_args: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda value: value,
        summarize_user_message_for_log=lambda value: value,
        set_session_context=lambda _session_id: None,
        set_current_write_origin=lambda _origin: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *_args: None),
    )
    assert worker.messages[-1] is staged
    assert worker.messages[-1]["content"] == "[MODEL SWITCH NOTE]\n\nnew prompt"

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]


def test_cli_close_builds_prompt_before_creating_first_session_row(tmp_path, monkeypatch):
    """First-turn close persistence must not leave a NULL prompt snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import agent.conversation_loop as loop_mod
    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-first-turn"
    agent = _real_agent(db, session_id, [])
    agent._session_db_created = False
    agent._cached_system_prompt = None
    staged = {"role": "user", "content": "first prompt"}
    agent._pending_cli_user_message = staged

    def _build_prompt(target, _system_message, _history):
        target._cached_system_prompt = "close-built-system-prompt"

    monkeypatch.setattr(loop_mod, "_restore_or_build_system_prompt", _build_prompt)

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = [staged]
    cli.session_id = session_id
    cli.agent = agent

    cli._persist_active_session_before_close()

    session = db.get_session(session_id)
    assert session is not None
    assert session["system_prompt"] == "close-built-system-prompt"
    assert [m["content"] for m in db.get_messages_as_conversation(session_id)] == [
        "first prompt"
    ]
