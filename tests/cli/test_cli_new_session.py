"""Regression tests for CLI fresh-session commands."""

from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tools.todo_tool import TodoStore


class _FakeCompressor:
    """Minimal stand-in for ContextCompressor."""

    def __init__(self):
        self.last_prompt_tokens = 500
        self.last_completion_tokens = 200
        self.last_total_tokens = 700
        self.compression_count = 3
        self._context_probed = True


class _FakeAgent:
    def __init__(self, session_id: str, session_start):
        self.session_id = session_id
        self.session_start = session_start
        self.model = "anthropic/claude-opus-4.6"
        self._last_flushed_db_idx = 7
        self._todo_store = TodoStore()
        self._todo_store.write(
            [{"id": "t1", "content": "unfinished task", "status": "in_progress"}]
        )
        self.commit_memory_session = MagicMock()
        self._invalidate_system_prompt = MagicMock()

        # Token counters (non-zero to verify reset)
        self.session_total_tokens = 1000
        self.session_input_tokens = 600
        self.session_output_tokens = 400
        self.session_prompt_tokens = 550
        self.session_completion_tokens = 350
        self.session_cache_read_tokens = 100
        self.session_cache_write_tokens = 50
        self.session_reasoning_tokens = 80
        self.session_api_calls = 5
        self.session_estimated_cost_usd = 0.42
        self.session_cost_status = "estimated"
        self.session_cost_source = "openrouter"
        self.context_compressor = _FakeCompressor()

    def reset_session_state(self):
        """Mirror the real AIAgent.reset_session_state()."""
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.last_prompt_tokens = 0
            self.context_compressor.last_completion_tokens = 0
            self.context_compressor.last_total_tokens = 0
            self.context_compressor.compression_count = 0
            self.context_compressor._context_probed = False


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
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
    if config_overrides:
        _clean_config.update(config_overrides)
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
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
            return _cli_mod.HermesCLI(**kwargs)


def _prepare_cli_with_active_session(tmp_path):
    cli = _make_cli()
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)

    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hello"}]

    old_session_start = cli.session_start - timedelta(seconds=1)
    cli.session_start = old_session_start
    cli.agent.session_start = old_session_start

    # Bypass the destructive-slash confirmation gate — these tests focus on
    # the new-session mechanics, not the confirm prompt itself (covered in
    # tests/cli/test_destructive_slash_confirm.py).
    cli._confirm_destructive_slash = lambda *_a, **_kw: "once"
    return cli


@pytest.fixture(autouse=True)
def _reset_session_id_context():
    from gateway.session_context import _UNSET, _VAR_MAP

    yield
    os.environ.pop("HERMES_SESSION_ID", None)
    _VAR_MAP["HERMES_SESSION_ID"].set(_UNSET)


def test_new_command_creates_real_fresh_session_and_resets_agent_state(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id
    old_session_start = cli.session_start

    cli.process_command("/new")

    assert cli.session_id != old_session_id

    old_session = cli._session_db.get_session(old_session_id)
    assert old_session is not None
    assert old_session["end_reason"] == "new_session"

    new_session = cli._session_db.get_session(cli.session_id)
    assert new_session is not None

    cli._session_db.append_message(cli.session_id, role="user", content="next turn")

    assert cli.agent.session_id == cli.session_id
    assert cli.agent._last_flushed_db_idx == 0
    assert cli.agent._todo_store.read() == []
    assert cli.session_start > old_session_start
    assert cli.agent.session_start == cli.session_start
    cli.agent._invalidate_system_prompt.assert_called_once()


def test_new_session_queues_boundary_commit_with_snapshot(tmp_path):
    """/new hands the OLD session's history + ids to the memory manager's
    serialized boundary task instead of blocking on extraction inline."""
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id

    mm = MagicMock()
    cli.agent._memory_manager = mm

    cli.process_command("/new")

    mm.commit_session_boundary_async.assert_called_once()
    args, kwargs = mm.commit_session_boundary_async.call_args
    assert args[0] == [{"role": "user", "content": "hello"}]
    assert kwargs["new_session_id"] == cli.session_id
    assert kwargs["parent_session_id"] == old_session_id
    assert kwargs["reason"] == "new_session"
    # The queued path replaces the inline switch — not both.
    mm.on_session_switch.assert_not_called()
    # Snapshot is consumed (a later /new without history must not re-send it).
    assert cli._session_boundary_snapshot is None


def test_new_session_without_history_switches_inline(tmp_path):
    """No old-session history → nothing to extract → plain inline switch."""
    cli = _prepare_cli_with_active_session(tmp_path)
    cli.conversation_history = []

    mm = MagicMock()
    cli.agent._memory_manager = mm

    cli.process_command("/new")

    mm.commit_session_boundary_async.assert_not_called()
    mm.on_session_switch.assert_called_once()
    _, kwargs = mm.on_session_switch.call_args
    assert kwargs["reset"] is True


def test_new_session_delivers_context_engine_boundary_synchronously(tmp_path):
    """The context-engine on_session_end must fire during /new itself.

    It is cheap local state work and ordering-sensitive: it must land before
    reset_session_state() rebinds the engine to the new session. The LLM-bound
    provider extraction is what gets deferred, not this."""
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id

    engine_calls = []
    cli.agent.context_compressor.on_session_end = (
        lambda sid, msgs: engine_calls.append((sid, list(msgs)))
    )

    cli.process_command("/new")

    assert engine_calls == [(old_session_id, [{"role": "user", "content": "hello"}])]


def test_run_cleanup_flushes_pending_memory_manager_work(tmp_path):
    """A '/new then quit' must not drop the queued old-session extraction.

    _run_cleanup gives the manager's serialized worker a bounded drain via
    flush_pending() before shutdown_all()'s short-fuse drain runs."""
    import cli as _cli_mod

    agent = MagicMock()
    mm = MagicMock()
    mm.flush_pending.return_value = True
    agent._memory_manager = mm
    agent._session_messages = []

    old_ref = _cli_mod._active_agent_ref
    _cli_mod._active_agent_ref = agent
    _cli_mod._cleanup_done = False
    try:
        _cli_mod._run_cleanup(notify_session_finalize=False)
    finally:
        _cli_mod._cleanup_done = True
        _cli_mod._active_agent_ref = old_ref

    mm.flush_pending.assert_called_once_with(timeout=10)


def test_new_command_rotates_hermes_session_id_env_and_context(tmp_path):
    from gateway.session_context import _VAR_MAP, get_session_env

    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id
    os.environ["HERMES_SESSION_ID"] = old_session_id
    _VAR_MAP["HERMES_SESSION_ID"].set(old_session_id)

    cli.process_command("/new")

    assert cli.session_id != old_session_id
    assert os.environ["HERMES_SESSION_ID"] == cli.session_id
    assert get_session_env("HERMES_SESSION_ID") == cli.session_id


def test_reset_command_is_alias_for_new_session(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id

    cli.process_command("/reset")

    assert cli.session_id != old_session_id
    assert cli._session_db.get_session(old_session_id)["end_reason"] == "new_session"
    assert cli._session_db.get_session(cli.session_id) is not None


def test_clear_command_starts_new_session_before_redrawing(tmp_path):
    cli = _prepare_cli_with_active_session(tmp_path)
    cli.console = MagicMock()
    cli.show_banner = MagicMock()

    old_session_id = cli.session_id
    cli.process_command("/clear")

    assert cli.session_id != old_session_id
    assert cli._session_db.get_session(old_session_id)["end_reason"] == "new_session"
    assert cli._session_db.get_session(cli.session_id) is not None
    cli.console.clear.assert_called_once()
    cli.show_banner.assert_called_once()
    assert cli.conversation_history == []


def test_new_session_resets_token_counters(tmp_path):
    """Regression test for #2099: /new must zero all token counters."""
    cli = _prepare_cli_with_active_session(tmp_path)

    # Verify counters are non-zero before reset
    agent = cli.agent
    assert agent.session_total_tokens > 0
    assert agent.session_api_calls > 0
    assert agent.context_compressor.compression_count > 0

    cli.process_command("/new")

    # All agent token counters must be zero
    assert agent.session_total_tokens == 0
    assert agent.session_input_tokens == 0
    assert agent.session_output_tokens == 0
    assert agent.session_prompt_tokens == 0
    assert agent.session_completion_tokens == 0
    assert agent.session_cache_read_tokens == 0
    assert agent.session_cache_write_tokens == 0
    assert agent.session_reasoning_tokens == 0
    assert agent.session_api_calls == 0
    assert agent.session_estimated_cost_usd == 0.0
    assert agent.session_cost_status == "unknown"
    assert agent.session_cost_source == "none"

    # Context compressor counters must also be zero
    comp = agent.context_compressor
    assert comp.last_prompt_tokens == 0
    assert comp.last_completion_tokens == 0
    assert comp.last_total_tokens == 0
    assert comp.compression_count == 0
    assert comp._context_probed is False


def test_new_session_with_title(capsys):
    """new_session(title=...) creates a session and sets the title."""
    cli = _make_cli()
    cli._session_db = MagicMock()
    cli.agent = _FakeAgent("old_session_id", datetime.now())
    cli.conversation_history = []

    cli.new_session(title="My Test Session")

    # Assert set_session_title was called with the new session ID and sanitized title
    cli._session_db.set_session_title.assert_called_once()
    call_args = cli._session_db.set_session_title.call_args
    assert call_args[0][0] == cli.session_id
    assert call_args[0][1] == "My Test Session"

    captured = capsys.readouterr()
    assert "My Test Session" in captured.out


def test_new_session_with_duplicate_title_surfaces_error(capsys):
    """new_session(title=...) handles ValueError from a duplicate-title conflict.

    The session is still created; the title assignment fails; the success banner
    must not claim the rejected title as the session name.
    """
    cli = _make_cli()
    cli._session_db = MagicMock()
    cli._session_db.set_session_title.side_effect = ValueError(
        "Title 'Dup' is already in use by session abc-123"
    )
    cli.agent = _FakeAgent("old_session_id", datetime.now())
    cli.conversation_history = []

    # Capture warnings printed via cli._cprint. After importlib.reload(),
    # the method's __globals__ dict is the one from the live module — patch
    # the exact dict the method will read.
    warnings: list[str] = []
    method_globals = cli.new_session.__globals__
    original = method_globals["_cprint"]
    method_globals["_cprint"] = lambda msg: warnings.append(msg)
    try:
        cli.new_session(title="Dup")
    finally:
        method_globals["_cprint"] = original

    cli._session_db.set_session_title.assert_called_once()
    joined = "\n".join(warnings)
    assert "already in use" in joined
    assert "session started untitled" in joined

    # The success banner must NOT claim the rejected title as the session name.
    captured = capsys.readouterr()
    assert "New session started: Dup" not in captured.out
    assert "New session started!" in captured.out
