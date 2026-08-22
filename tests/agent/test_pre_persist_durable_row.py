"""Integration coverage for the pre_persist_user_message hook's durable contract.

The pure composition of hook returns is unit-tested in
``test_compose_pre_persist_returns.py``. These tests drive the REAL persistence
seam (``build_turn_context`` -> ``_persist_session`` ->
``_flush_messages_to_session_db`` -> SQLite) to prove the hook's returns reach
the *durable* session row, not only this turn's live prompt:

1. The ordinary early-persistence path: a fresh staged user turn is written with
   the injected block.
2. The already-close-persisted staged-user path: the CLI close safety-net can
   persist and mark the staged user dict BEFORE this turn's hook runs. The
   append-only flush then skips that marked dict, so the durable row must be
   corrected in place (``update_active_message_content``) without inserting a
   second row — preserving the exactly-once persistence invariant.
"""

from __future__ import annotations

import threading
import types

import pytest


def _real_agent(db, session_id, session_messages):
    """Build the real persistence seam without the heavyweight LLM client.

    Mirrors the harness in tests/cli/test_cli_shutdown_memory_messages.py.
    """
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


def _configure_for_turn(agent):
    """Stub the per-turn setup collaborators build_turn_context touches."""
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


def _build_turn(agent, prefix, user_message, persist_user_message):
    from agent.turn_context import build_turn_context

    return build_turn_context(
        agent,
        user_message,
        None,
        prefix,
        "task",
        None,
        persist_user_message,
        None,
        restore_or_build_system_prompt=lambda *_args: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda value: value,
        summarize_user_message_for_log=lambda value: value,
        set_session_context=lambda _session_id: None,
        set_current_write_origin=lambda _origin: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *_args: None),
    )


def _inject_recall_hook(hook_name, **_kwargs):
    """Fake plugin dispatcher: a pre_persist plugin that appends a recall block."""
    if hook_name == "pre_persist_user_message":
        return [{"context": "[Mem] recall"}]
    return None


def test_pre_persist_returns_reach_durable_row_on_early_persist(tmp_path, monkeypatch):
    """A fresh staged user turn persists WITH the injected block (early persist)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "pre-persist-early"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _real_agent(db, session_id, prefix)
    agent._flush_messages_to_session_db(prefix, [])

    staged = {"role": "user", "content": "new prompt"}
    agent._pending_cli_user_message = staged
    _configure_for_turn(agent)

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _inject_recall_hook)

    worker = _build_turn(agent, prefix, "new prompt", "new prompt")

    # The live prompt carries the injection this turn.
    assert worker.messages[-1] is staged
    assert worker.messages[-1]["content"] == "new prompt\n\n[Mem] recall"

    # The durable row carries it too — the whole point of a *pre_persist* hook.
    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt\n\n[Mem] recall",
    ]


def test_pre_persist_corrects_close_marked_staged_row_in_place(tmp_path, monkeypatch):
    """Close-persisted staged user row is corrected in place, not duplicated.

    The CLI close safety-net writes the clean staged dict and stamps it
    ``_db_persisted`` before this turn's hook runs. The append-only flush would
    otherwise skip that marked dict, leaving the injected context out of the
    durable row. The hook path must correct that one row in place.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "pre-persist-close-marked"
    db.create_session(session_id=session_id, source="cli")
    prefix = [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _real_agent(db, session_id, prefix)
    agent._flush_messages_to_session_db(prefix, [])

    staged = {"role": "user", "content": "new prompt"}
    agent._pending_cli_user_message = staged

    # CLI close safety-net: persist + mark the clean staged dict.
    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = list(prefix) + [staged]
    cli.session_id = session_id
    cli.agent = agent
    cli._persist_active_session_before_close()
    assert staged["_db_persisted"] is True
    pre_hook_stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in pre_hook_stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]

    # Now a worker turn reuses that marked dict; the pre_persist hook injects.
    _configure_for_turn(agent)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _inject_recall_hook)

    worker = _build_turn(agent, prefix, "new prompt", "new prompt")
    assert worker.messages[-1] is staged
    assert worker.messages[-1]["content"] == "new prompt\n\n[Mem] recall"

    # The already-durable row is corrected in place: injection is now durable
    # AND there is still exactly one row for this user turn (no duplicate).
    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt\n\n[Mem] recall",
    ]


def test_close_marked_staged_row_untouched_when_hook_is_silent(tmp_path, monkeypatch):
    """No hook returns => no in-place rewrite; the clean durable row is kept.

    Guards the exactly-once/clean-row invariant the correction must not disturb
    on the ordinary (no-injection) path.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "pre-persist-close-marked-silent"
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

    _configure_for_turn(agent)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook", lambda *_a, **_k: None
    )

    _build_turn(agent, prefix, "new prompt", "new prompt")

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]
