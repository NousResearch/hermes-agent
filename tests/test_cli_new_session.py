"""Regression tests for CLI fresh-session commands."""

from __future__ import annotations

import importlib
import os
import sys
from datetime import timedelta
from unittest.mock import MagicMock, patch

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
        self.flush_memories = MagicMock()
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
    return cli


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
    cli.agent.flush_memories.assert_called_once_with([{"role": "user", "content": "hello"}])
    cli.agent._invalidate_system_prompt.assert_called_once()


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


# ── /resume command tests ────────────────────────────────────────────


def _prepare_cli_with_titled_session(tmp_path):
    """Create a CLI with an active session, then create a second titled session to resume."""
    cli = _make_cli()
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")

    # Create the "old" session with a title and some messages
    old_id = "20260101_120000_aaaaaa"
    cli._session_db.create_session(session_id=old_id, source="cli", model=cli.model)
    cli._session_db.set_session_title(old_id, "My Old Session")
    cli._session_db.append_message(old_id, role="user", content="hello from old session")
    cli._session_db.append_message(old_id, role="assistant", content="hi there")
    cli._session_db.end_session(old_id, "new_session")

    # Set up the current (active) session
    cli._session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "current msg"}]

    return cli, old_id


def test_resume_switches_to_titled_session(tmp_path):
    """``/resume My Old Session`` should switch to that session and load its history."""
    cli, old_id = _prepare_cli_with_titled_session(tmp_path)
    original_session_id = cli.session_id

    cli.process_command("/resume My Old Session")

    assert cli.session_id == old_id
    assert cli.session_id != original_session_id
    assert cli._resumed is True
    # History should be restored from the old session
    assert len(cli.conversation_history) >= 2
    user_msgs = [m for m in cli.conversation_history if m.get("role") == "user"]
    assert any("hello from old session" in (m.get("content") or "") for m in user_msgs)
    # Old current session should be ended
    old_current = cli._session_db.get_session(original_session_id)
    assert old_current["end_reason"] == "resume_session"
    # Agent should be pointed at the new session
    assert cli.agent.session_id == old_id
    # Memories should have been flushed for the previous session
    cli.agent.flush_memories.assert_called_once()


def _capture_cprint(cli_instance):
    """Return (printed_lines, original_cprint) for capturing _cprint output.

    Because _make_cli reloads the module, we patch via the method's __globals__
    which always points at the correct module namespace.
    """
    mod_globals = type(cli_instance).resume_session.__globals__
    original = mod_globals["_cprint"]
    lines: list = []
    mod_globals["_cprint"] = lambda t: lines.append(t)
    return lines, original, mod_globals


def test_resume_no_args_lists_sessions(tmp_path):
    """``/resume`` with no argument should list titled sessions."""
    cli, _ = _prepare_cli_with_titled_session(tmp_path)
    printed, orig, globs = _capture_cprint(cli)
    try:
        cli.process_command("/resume")
    finally:
        globs["_cprint"] = orig

    text = "\n".join(printed)
    assert "My Old Session" in text
    assert "Named Sessions" in text


def test_resume_unknown_session(tmp_path):
    """``/resume NonExistent`` should show an error."""
    cli, _ = _prepare_cli_with_titled_session(tmp_path)
    printed, orig, globs = _capture_cprint(cli)
    try:
        cli.process_command("/resume NonExistent")
    finally:
        globs["_cprint"] = orig

    text = "\n".join(printed)
    assert "No session found" in text


def test_resume_already_on_session(tmp_path):
    """``/resume`` to the current session should say already on it."""
    cli, old_id = _prepare_cli_with_titled_session(tmp_path)
    # First resume to old session
    cli.process_command("/resume My Old Session")
    printed, orig, globs = _capture_cprint(cli)
    try:
        # Then try to resume again
        cli.process_command("/resume My Old Session")
    finally:
        globs["_cprint"] = orig

    text = "\n".join(printed)
    assert "Already on session" in text
