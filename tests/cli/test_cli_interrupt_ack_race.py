"""Regression tests for the CLI interrupt-acknowledgement race.

Symptom (user report, July 2026): interrupting an active turn is
unreliable — the interrupt message is sometimes "vacuumed into the void".

Root cause: ``HermesCLI.chat()`` fires ``agent.interrupt(msg)`` from its
monitor loop, but only re-queued the message when the turn RESULT carried
``interrupted=True``. Two races defeat that:

  1. The agent thread passes its last ``_interrupt_requested`` check (or
     finishes entirely) just before the interrupt lands — the turn
     completes "normally", ``finalize_turn()`` never acknowledges the
     interrupt, and the user's message was silently dropped.
  2. Worse, when the interrupt lands *after* ``finalize_turn()``'s
     ``clear_interrupt()``, the stale ``_interrupt_requested`` flag
     survives on the agent and instantly aborts the NEXT turn at its
     first loop check.

The fix: when ``chat()`` consumed an ``interrupt_msg`` but the result
doesn't acknowledge the interrupt, re-queue the message as the next turn
and clear the stale agent flag (only when the agent thread has exited).
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch


def _make_cli():
    """Build a HermesCLI with prompt_toolkit stubbed (same pattern as
    test_cli_interrupt_drain_regression.py)."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class _StubAgent:
    """Agent whose turn completes WITHOUT acknowledging the interrupt."""

    def __init__(self, session_id, turn_seconds=0.5):
        self.session_id = session_id
        self.turn_seconds = turn_seconds
        self._interrupt_requested = False
        self._interrupt_message = None
        self._active_children = []
        self.interrupt_calls = []
        self.clear_calls = 0
        self.max_iterations = 90
        self.model = "test/model"
        self.platform = "cli"

    def run_conversation(self, **kwargs):
        # Simulate a turn that finishes normally — it never observed the
        # interrupt flag (raced past its last check).
        time.sleep(self.turn_seconds)
        return {
            "final_response": "turn finished normally",
            "messages": [
                {"role": "user", "content": "original"},
                {"role": "assistant", "content": "turn finished normally"},
            ],
            "api_calls": 1,
            "completed": True,
            # NOTE: no "interrupted" key — the race means finalize_turn
            # never saw the flag (or cleared it before it was re-set).
            "partial": True,  # skip auto-title thread in the test
            # Skip the Rich Panel rendering path (crashes under the
            # prompt_toolkit/skin mocks; irrelevant to this regression).
            "response_previewed": True,
        }

    def interrupt(self, message=None):
        self.interrupt_calls.append(message)
        self._interrupt_requested = True
        self._interrupt_message = message

    def clear_interrupt(self):
        self.clear_calls += 1
        self._interrupt_requested = False
        self._interrupt_message = None


def test_unacknowledged_interrupt_message_is_requeued_not_dropped():
    cli = _make_cli()
    agent = _StubAgent(cli.session_id)
    cli.agent = agent

    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._interrupt_queue.put("urgent new message")

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("original")

    # The interrupt fired against the agent...
    assert agent.interrupt_calls == ["urgent new message"]
    # ...the turn result never acknowledged it, so the message must be
    # re-queued as the next turn instead of dropped.
    queued = []
    while not cli._pending_input.empty():
        queued.append(cli._pending_input.get_nowait())
    assert any("urgent new message" in str(q) for q in queued), (
        f"interrupt message was dropped; pending_input={queued!r}"
    )
    # ...and the stale flag must be cleared so the NEXT turn doesn't
    # instantly self-abort at its first _interrupt_requested check.
    assert agent._interrupt_requested is False
    assert agent.clear_calls >= 1


def test_acknowledged_interrupt_still_requeues_message():
    """The pre-existing path (result carries interrupted=True) still works."""
    cli = _make_cli()

    class _AckAgent(_StubAgent):
        def run_conversation(self, **kwargs):
            # Wait until the monitor loop delivers the interrupt.
            for _ in range(100):
                if self._interrupt_requested:
                    break
                time.sleep(0.05)
            return {
                "final_response": "partial work",
                "messages": [{"role": "assistant", "content": "partial work"}],
                "api_calls": 1,
                "completed": False,
                "interrupted": True,
                "interrupt_message": self._interrupt_message,
                "partial": True,
            }

    agent = _AckAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._interrupt_queue.put("redirect please")

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("original")

    queued = []
    while not cli._pending_input.empty():
        queued.append(cli._pending_input.get_nowait())
    assert any("redirect please" in str(q) for q in queued)
    assert cli._last_turn_interrupted is True


def test_chat_persists_clean_input_when_a_queued_note_changes_api_message():
    """Queued notes remain API-local and preserve close-handoff marker identity."""
    cli = _make_cli()

    class _NoteAgent(_StubAgent):
        def __init__(self, session_id):
            super().__init__(session_id, turn_seconds=0)
            self.captured = None

        def run_conversation(self, **kwargs):
            self.captured = kwargs
            return {
                "final_response": "done",
                "messages": [{"role": "assistant", "content": "done"}],
                "api_calls": 1,
                "completed": True,
                "partial": True,
                "response_previewed": True,
            }

    agent = _NoteAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._pending_model_switch_note = "[MODEL SWITCH NOTE]"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("clean prompt")

    assert agent.captured is not None
    assert agent.captured["user_message"] == "[MODEL SWITCH NOTE]\n\nclean prompt"
    assert agent.captured["persist_user_message"] == "clean prompt"


def test_chat_preserves_clean_multimodal_input_when_note_changes_api_message():
    """A queued note forwards original native parts as the persistence override."""
    cli = _make_cli()

    class _NoteAgent(_StubAgent):
        def __init__(self, session_id):
            super().__init__(session_id, turn_seconds=0)
            self.captured = None

        def run_conversation(self, **kwargs):
            self.captured = kwargs
            return {
                "final_response": "done",
                "messages": [{"role": "assistant", "content": "done"}],
                "api_calls": 1,
                "completed": True,
                "partial": True,
                "response_previewed": True,
            }

    clean_parts = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    agent = _NoteAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._pending_model_switch_note = "[MODEL SWITCH NOTE]"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat(clean_parts)

    assert agent.captured is not None
    assert agent.captured["persist_user_message"] == clean_parts
    assert agent.captured["persist_user_message"] is not agent.captured["user_message"]
    api_parts = agent.captured["user_message"]
    assert api_parts[0]["text"] == "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"
    assert api_parts[1] == clean_parts[1]


def test_chat_multimodal_note_persists_clean_input_once(tmp_path, monkeypatch):
    """The real CLI-to-agent path stores clean image parts, never the queued note."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _make_cli()
    session_id = cli.session_id
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id=session_id, source="cli")

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent.provider = "test"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = "chat_completions"
    agent._session_messages = []
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
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._active_children = []
    agent._interrupt_requested = False

    clean_parts = [
        {"type": "text", "text": "Describe this screenshot"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured = {}

    def _realish_run(**kwargs):
        captured.update(kwargs)
        # Drive production turn setup and the real SQLite persistence seam,
        # then return a normal CLI result without starting a provider loop.
        from agent.turn_context import build_turn_context

        agent.quiet_mode = True
        agent.max_iterations = 1
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
        agent._memory_write_origin = "assistant_tool"
        agent._stream_context_scrubber = None
        agent._stream_think_scrubber = None
        agent._restore_primary_runtime = lambda: None
        agent._cleanup_dead_connections = lambda: False
        agent._emit_status = lambda _message: None
        agent._replay_compression_warning = lambda: None
        agent._hydrate_todo_store = lambda *_args: None
        agent._safe_print = lambda *_args: None

        context = build_turn_context(
            agent,
            kwargs["user_message"],
            None,
            kwargs["conversation_history"],
            kwargs["task_id"],
            None,
            kwargs["persist_user_message"],
            None,
            restore_or_build_system_prompt=lambda *_args: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda value: value,
            summarize_user_message_for_log=lambda value: (
                value if isinstance(value, str) else "[multimodal test message]"
            ),
            set_session_context=lambda _session_id: None,
            set_current_write_origin=lambda _origin: None,
            ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *_args: None),
        )
        agent._apply_persist_user_message_override(context.messages)
        agent._persist_session(context.messages, kwargs["conversation_history"])
        return {
            "final_response": "done",
            "messages": context.messages + [{"role": "assistant", "content": "done"}],
            "api_calls": 1,
            "completed": True,
            "partial": True,
            "response_previewed": True,
        }

    agent.run_conversation = _realish_run
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._pending_model_switch_note = "[MODEL SWITCH NOTE]"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat(clean_parts)

    assert captured["persist_user_message"] == clean_parts
    assert captured["user_message"][0]["text"] == "[MODEL SWITCH NOTE]\n\nDescribe this screenshot"
    assert [m["content"] for m in db.get_messages_as_conversation(session_id)] == [
        "Describe this screenshot\n[screenshot]"
    ]


def test_chat_clears_previous_turn_persistence_override_before_staging():
    """A close before the next worker starts cannot reuse a stale override."""
    cli = _make_cli()

    class _StagingAgent(_StubAgent):
        def __init__(self, session_id):
            super().__init__(session_id, turn_seconds=0)
            self.staged_override = None
            self.staged_message = None
            self._session_messages = []
            self._persist_user_message_idx = 7
            self._persist_user_message_override = "previous clean prompt"
            self._persist_user_message_timestamp = 123.0

        def run_conversation(self, **kwargs):
            self.staged_override = self._persist_user_message_override
            self.staged_message = self._pending_cli_user_message
            return {
                "final_response": "done",
                "messages": [{"role": "assistant", "content": "done"}],
                "api_calls": 1,
                "completed": True,
                "partial": True,
                "response_previewed": True,
            }

    agent = _StagingAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("new prompt")

    assert agent.staged_override is None
    assert agent._persist_user_message_idx is None
    assert agent._persist_user_message_timestamp is None
    assert agent.staged_message == {"role": "user", "content": "new prompt"}


def test_chat_close_does_not_persist_previous_turn_override(tmp_path, monkeypatch):
    """A close after input staging writes the new prompt, not old API-only text."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _make_cli()
    session_id = cli.session_id
    db = SessionDB(db_path=tmp_path / "state.db")
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

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent._session_messages = []
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
    agent._persist_user_message_idx = len(prefix)
    agent._persist_user_message_override = "previous clean prompt"
    agent._persist_user_message_timestamp = 123.0
    agent._active_children = []
    agent._interrupt_requested = False
    entered = threading.Event()
    release = threading.Event()

    def _block_run(**_kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {
            "final_response": "done",
            "messages": prefix + [{"role": "assistant", "content": "done"}],
            "api_calls": 1,
            "completed": True,
            "partial": True,
            "response_previewed": True,
        }

    agent.run_conversation = _block_run
    cli.agent = agent
    cli.conversation_history = list(prefix)
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        chat_thread = threading.Thread(target=lambda: cli.chat("new prompt"))
        chat_thread.start()
        assert entered.wait(timeout=5)
        cli._persist_active_session_before_close()
        release.set()
        chat_thread.join(timeout=10)

    assert not chat_thread.is_alive()
    assert [m["content"] for m in db.get_messages_as_conversation(session_id)] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]


def test_close_waits_for_atomic_cli_staging_before_snapshot(tmp_path, monkeypatch):
    """Close cannot retain the mutable pre-append history as its DB baseline."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _make_cli()
    session_id = cli.session_id
    db = SessionDB(db_path=tmp_path / "state.db")
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

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    # Deliberately distinct from CLI history: this is the normal pre-worker
    # state that used to let close retain the wrong mutable baseline.
    agent._session_messages = list(prefix)
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
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._active_children = []
    agent._interrupt_requested = False

    staging_entered = threading.Event()
    release_staging = threading.Event()
    run_entered = threading.Event()
    release_run = threading.Event()

    class _BlockingHistory(list):
        def __init__(self, values):
            super().__init__(values)
            self._block_next_append = True

        def append(self, value):
            if self._block_next_append:
                self._block_next_append = False
                staging_entered.set()
                assert release_staging.wait(timeout=5)
            return super().append(value)

    def _block_run(**_kwargs):
        run_entered.set()
        assert release_run.wait(timeout=5)
        return {
            "final_response": "done",
            "messages": prefix + [{"role": "assistant", "content": "done"}],
            "api_calls": 1,
            "completed": True,
            "partial": True,
            "response_previewed": True,
        }

    agent.run_conversation = _block_run
    cli.agent = agent
    cli.conversation_history = _BlockingHistory(prefix)
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        chat_thread = threading.Thread(target=lambda: cli.chat("new prompt"))
        chat_thread.start()
        assert staging_entered.wait(timeout=5)

        close_started = threading.Event()
        close_finished = threading.Event()

        def _close():
            close_started.set()
            cli._persist_active_session_before_close()
            close_finished.set()

        close_thread = threading.Thread(target=_close)
        close_thread.start()
        assert close_started.wait(timeout=5)
        # The close snapshot must wait for the locked pending-pointer/history
        # handoff; otherwise the subsequent append poisons its DB baseline.
        assert not close_finished.wait(timeout=0.1)

        release_staging.set()
        assert run_entered.wait(timeout=5)
        assert close_finished.wait(timeout=5)
        release_run.set()
        chat_thread.join(timeout=10)
        close_thread.join(timeout=10)

    assert not chat_thread.is_alive()
    assert not close_thread.is_alive()
    assert [m["content"] for m in db.get_messages_as_conversation(session_id)] == [
        "old prompt",
        "old answer",
        "new prompt",
    ]
