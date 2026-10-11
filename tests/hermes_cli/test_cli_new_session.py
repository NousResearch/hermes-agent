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
    # tests/hermes_cli/test_destructive_slash_confirm.py).
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
    """Regression test for #2099: /new must zero all token counters.

    Drives the real ``AIAgent.reset_session_state`` (and the real context-engine
    ``on_session_reset``) on the fake agent's attribute bag, so this guards both the
    CLI wiring (/new must call the reset) and the reset itself.
    """
    import types

    from agent.context_engine import ContextEngine
    from run_agent import AIAgent

    cli = _prepare_cli_with_active_session(tmp_path)
    agent = cli.agent
    agent.reset_session_state = types.MethodType(AIAgent.reset_session_state, agent)
    agent._transition_context_engine_session = types.MethodType(
        AIAgent._transition_context_engine_session, agent
    )
    comp = agent.context_compressor
    comp.on_session_reset = types.MethodType(ContextEngine.on_session_reset, comp)

    assert agent.session_total_tokens > 0
    assert agent.session_api_calls > 0
    assert comp.compression_count > 0

    cli.process_command("/new")

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

    assert comp.last_prompt_tokens == 0
    assert comp.last_completion_tokens == 0
    assert comp.last_total_tokens == 0
    assert comp.compression_count == 0


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


def test_new_session_model_reset_stamps_route_on_new_session_not_outgoing(tmp_path):
    """/new must rotate the live agent onto the fresh session id BEFORE the
    config-default model reset: the reset's switch_model persists the billing
    route, and with the agent still on the outgoing session that write stamped
    the config-default provider onto a row whose accounted usage ran on a
    mid-session-switched route (frozen there: the usage-based reconciliation
    only ran while api_call_count == 0)."""
    cli = _prepare_cli_with_active_session(tmp_path)
    old_session_id = cli.session_id
    # The outgoing session's accounted usage ran on a route other than the
    # config default (a mid-session /model switch).
    cli._session_db.update_token_counts(
        old_session_id, input_tokens=100, model="claude-max-route",
        billing_provider="claude-subscription-route",
        billing_base_url="process://sub", api_call_count=1, estimated_cost_usd=0.01,
    )
    cli.model = "claude-max-route"
    cli.provider = "claude-subscription-route"

    stamped = []

    def _fake_agent_switch_model(**kwargs):
        # Mirror the real AIAgent.switch_model's route persistence: it stamps
        # the billing route of the session the agent is CURRENTLY on.
        cli._session_db.update_session_billing_route(
            cli.agent.session_id, provider=kwargs["new_provider"],
            base_url=kwargs.get("base_url") or "",
        )
        stamped.append(cli.agent.session_id)

    cli.agent.switch_model = _fake_agent_switch_model
    cli.process_command("/new")

    # The reset's route write targeted the NEW session (whose row does not
    # exist yet at reset time — a no-op there), never the outgoing row.
    assert stamped and stamped[0] != old_session_id
    assert stamped[0] == cli.session_id
    with cli._session_db._lock:
        row = cli._session_db._conn.execute(
            "SELECT model, billing_provider FROM sessions WHERE id = ?",
            (old_session_id,),
        ).fetchone()
    assert row["billing_provider"] == "claude-subscription-route"
    assert row["model"] == "claude-max-route"
    assert cli._session_db.get_session(cli.session_id) is not None


