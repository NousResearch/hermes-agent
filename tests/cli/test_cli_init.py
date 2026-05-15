"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import importlib

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
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_integer(self):
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25

    def test_none_max_turns_gets_default(self):
        cli = _make_cli(max_turns=None)
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == 90

    def test_env_var_max_turns(self):
        """Env var is used when config file doesn't set max_turns."""
        cli_obj = _make_cli(env_overrides={"HERMES_MAX_ITERATIONS": "42"})
        assert cli_obj.max_turns == 42

    def test_invalid_env_var_max_turns_falls_back_to_default(self):
        """Invalid env values should not crash CLI init."""
        cli_obj = _make_cli(env_overrides={"HERMES_MAX_ITERATIONS": "not-a-number"})
        assert cli_obj.max_turns == 90

    def test_legacy_root_max_turns_is_used_when_agent_key_exists_without_value(self):
        cli_obj = _make_cli(config_overrides={"agent": {}, "max_turns": 77})
        assert cli_obj.max_turns == 77

    def test_max_turns_never_none_for_agent(self):
        """The value passed to AIAgent must never be None (causes TypeError in run_conversation)."""
        cli = _make_cli()
        assert isinstance(cli.max_turns, int) and cli.max_turns == 90


class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in ("off", "new", "all", "verbose")


class TestBusyInputMode:
    def test_default_busy_input_mode_is_interrupt(self):
        cli = _make_cli()
        assert cli.busy_input_mode == "interrupt"

    def test_busy_input_mode_queue_is_honored(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        assert cli.busy_input_mode == "queue"

    def test_unknown_busy_input_mode_falls_back_to_interrupt(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "bogus"}})
        assert cli.busy_input_mode == "interrupt"

    def test_queue_command_works_while_busy(self):
        """When agent is running, /queue should still put the prompt in _pending_input."""
        cli = _make_cli()
        cli._agent_running = True
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_queue_command_works_while_idle(self):
        """When agent is idle, /queue should still queue (not reject)."""
        cli = _make_cli()
        cli._agent_running = False
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"

    def test_q_alias_queues_prompt(self):
        """The /q alias should resolve to /queue, not /quit."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli.process_command("/q follow up") is True
        assert cli._pending_input.get_nowait() == "follow up"

    def test_queue_mode_routes_busy_enter_to_pending(self):
        """In queue mode, Enter while busy should go to _pending_input, not _interrupt_queue."""
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        cli._agent_running = True
        # Simulate what handle_enter does for non-command input while busy
        text = "follow up"
        if cli.busy_input_mode == "queue":
            cli._pending_input.put(text)
        else:
            cli._interrupt_queue.put(text)
        assert cli._pending_input.get_nowait() == "follow up"
        assert cli._interrupt_queue.empty()

    def test_interrupt_mode_routes_busy_enter_to_interrupt(self):
        """In interrupt mode (default), Enter while busy goes to _interrupt_queue."""
        cli = _make_cli()
        cli._agent_running = True
        text = "redirect"
        if cli.busy_input_mode == "queue":
            cli._pending_input.put(text)
        else:
            cli._interrupt_queue.put(text)
        assert cli._interrupt_queue.get_nowait() == "redirect"
        assert cli._pending_input.empty()


class TestPromptToolkitTerminalCompatibility:
    def test_lf_enter_binds_to_submit_handler_posix(self):
        """Some thin PTYs deliver Enter as LF/c-j instead of CR/enter.

        On a bare local POSIX TTY (no SSH/WSL/WT) we keep c-j → submit so
        Enter works on thin PTYs (docker exec, certain ssh configurations).
        On Windows, WSL, SSH sessions, and Windows Terminal we leave c-j
        unbound here so it can be used as the Ctrl+Enter newline keystroke
        without conflicting with submit. See issue #22379.
        """
        import sys as _sys
        import os as _os
        from unittest.mock import patch as _patch
        from prompt_toolkit.key_binding import KeyBindings

        from cli import _bind_prompt_submit_keys

        def submit_handler(event):
            return None

        # Bare local POSIX (no SSH/WSL markers): both enter and c-j submit.
        with _patch.object(_sys, "platform", "linux"), \
             _patch.dict(_os.environ, {}, clear=True), \
             _patch("builtins.open", side_effect=OSError("no /proc")):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert bindings[("c-j",)] is submit_handler

        # POSIX over SSH: c-j stays free so Ctrl+Enter (sent as LF by
        # Windows Terminal / Kitty / mintty over SSH) inserts a newline.
        with _patch.object(_sys, "platform", "linux"), \
             _patch.dict(_os.environ, {"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, clear=True), \
             _patch("builtins.open", side_effect=OSError("no /proc")):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert ("c-j",) not in bindings

        # Windows: only enter submits; c-j is free for the newline binding
        # added separately in the prompt setup.
        with _patch.object(_sys, "platform", "win32"):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert ("c-j",) not in bindings

    def test_cpr_warning_callback_is_disabled(self):
        from cli import _disable_prompt_toolkit_cpr_warning

        renderer = SimpleNamespace(cpr_not_supported_callback=lambda: None)
        app = SimpleNamespace(renderer=renderer)

        _disable_prompt_toolkit_cpr_warning(app)

        assert renderer.cpr_not_supported_callback is None


class TestSingleQueryState:
    def test_voice_and_interrupt_state_initialized_before_run(self):
        """Single-query mode calls chat() without going through run()."""
        cli = _make_cli()
        assert cli._voice_tts is False
        assert cli._voice_mode is False
        assert cli._voice_tts_done.is_set()
        assert hasattr(cli, "_interrupt_queue")
        assert hasattr(cli, "_pending_input")


class TestHistoryDisplay:
    def test_history_numbers_only_visible_messages_and_summarizes_tools(self, capsys):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1"}, {"id": "call_2"}],
            },
            {"role": "tool", "content": "tool output 1"},
            {"role": "tool", "content": "tool output 2"},
            {"role": "assistant", "content": "All set."},
            {"role": "user", "content": "A" * 250},
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "[You #1]" in output
        assert "[Hermes #2]" in output
        assert "(requested 2 tool calls)" in output
        assert "[Tools]" in output
        assert "(2 tool messages hidden)" in output
        assert "[Hermes #3]" in output
        assert "[You #4]" in output
        assert "[You #5]" not in output
        assert "A" * 250 in output
        assert "A" * 250 + "..." not in output

    def test_history_shows_recent_sessions_when_current_chat_is_empty(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli.show_history()
        output = capsys.readouterr().out

        assert "No messages in the current chat yet" in output
        assert "Checking Running Hermes Agent" in output
        assert "20260401_201329_d85961" in output
        assert "/resume" in output
        assert "Current preview" not in output

    def test_resume_without_target_lists_recent_sessions(self, capsys):
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "current",
                "title": "Current",
                "preview": "Current preview",
                "last_active": 0,
            },
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli._handle_resume_command("/resume")
        output = capsys.readouterr().out

        assert "Recent sessions" in output
        assert "Checking Running Hermes Agent" in output
        assert "Use /resume <session id or title> to continue" in output

    def test_sessions_command_no_args_lists_recent_sessions(self, capsys):
        """/sessions with no args prints the recent-sessions table (TUI parity).

        Regression test: `sessions` was registered in the central command
        registry and surfaced by /help and tab-completion, but the classic
        CLI dispatcher had no elif branch for it, so the canonical name fell
        through and printed `Unknown command: sessions`.
        """
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        # Drive it through the public dispatcher to also lock in the
        # process_command wiring, not just the handler in isolation.
        cli.process_command("/sessions")
        output = capsys.readouterr().out

        assert "Unknown command" not in output
        assert "Recent sessions" in output
        assert "Checking Running Hermes Agent" in output
        assert "20260401_201329_d85961" in output

    def test_sessions_list_subcommand_lists_recent_sessions(self, capsys):
        """/sessions list is an explicit alias for the no-arg list view."""
        cli = _make_cli()
        cli.session_id = "current"
        cli._session_db = MagicMock()
        cli._session_db.list_sessions_rich.return_value = [
            {
                "id": "20260401_201329_d85961",
                "title": "Checking Running Hermes Agent",
                "preview": "check running gateways for hermes agent",
                "last_active": 0,
            },
        ]

        cli.process_command("/sessions list")
        output = capsys.readouterr().out

        assert "Unknown command" not in output
        assert "Recent sessions" in output
        assert "Checking Running Hermes Agent" in output

    def test_sessions_with_target_delegates_to_resume(self):
        """/sessions <id_or_title> behaves identically to /resume <id_or_title>.

        We intercept `_handle_resume_command` rather than the full resume
        machinery (which would otherwise require simulating an entire session
        switch). The contract under test is the dispatch wiring.
        """
        cli = _make_cli()
        with patch.object(cli, "_handle_resume_command") as mock_resume:
            cli.process_command("/sessions Checking Running Hermes Agent")

        mock_resume.assert_called_once_with(
            "/resume Checking Running Hermes Agent"
        )

    def test_sessions_command_is_dispatched(self):
        """/sessions must hit _handle_sessions_command, not fall through.

        Direct test that the process_command elif chain routes the canonical
        name to the handler. Without this wiring, /sessions printed
        `Unknown command: sessions` even though it was a registered command.
        """
        cli = _make_cli()
        cli._session_db = None  # exercise the no-db path too

        with patch.object(cli, "_handle_sessions_command") as mock_handler:
            cli.process_command("/sessions")

        mock_handler.assert_called_once()
        called_with = mock_handler.call_args.args[0]
        assert called_with.lower().startswith("/sessions")


class TestRootLevelProviderOverride:
    """Root-level provider/base_url in config.yaml must NOT override model.provider."""

    def test_model_provider_wins_over_root_provider(self, tmp_path, monkeypatch):
        """model.provider takes priority — root-level provider is only a fallback."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root-level key
            "model": {
                "default": "google/gemini-3-flash-preview",
                "provider": "openrouter",  # correct canonical key
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        assert cfg["model"]["provider"] == "openrouter"

    def test_root_provider_ignored_when_default_model_provider_exists(self, tmp_path, monkeypatch):
        """Even when model.provider is the default 'auto', root-level provider is ignored."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "provider": "opencode-go",  # stale root key
            "model": {
                "default": "google/gemini-3-flash-preview",
                # no explicit model.provider — defaults provide "auto"
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        # Root-level "opencode-go" must NOT leak through
        assert cfg["model"]["provider"] != "opencode-go"

    def test_terminal_vercel_runtime_bridged_to_env(self, tmp_path, monkeypatch):
        """Classic CLI must expose terminal.vercel_runtime to terminal_tool.py."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TERMINAL_VERCEL_RUNTIME", raising=False)

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "terminal": {
                "backend": "vercel_sandbox",
                "vercel_runtime": "python3.13",
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        assert cfg["terminal"]["vercel_runtime"] == "python3.13"
        assert os.environ["TERMINAL_VERCEL_RUNTIME"] == "python3.13"

    def test_normalize_root_model_keys_moves_to_model(self):
        """_normalize_root_model_keys migrates root keys into model section."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "opencode-go",
            "base_url": "https://example.com/v1",
            "model": {
                "default": "some-model",
            },
        }
        result = _normalize_root_model_keys(config)
        # Root keys removed
        assert "provider" not in result
        assert "base_url" not in result
        # Migrated into model section
        assert result["model"]["provider"] == "opencode-go"
        assert result["model"]["base_url"] == "https://example.com/v1"

    def test_normalize_root_model_keys_does_not_override_existing(self):
        """Existing model.provider is never overridden by root-level key."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "provider": "stale-provider",
            "model": {
                "default": "some-model",
                "provider": "correct-provider",
            },
        }
        result = _normalize_root_model_keys(config)
        assert result["model"]["provider"] == "correct-provider"
        assert "provider" not in result  # root key still cleaned up

    def test_normalize_root_context_length_migrates_to_model(self):
        """Root-level context_length is migrated into the model section."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "context_length": 128000,
            "model": {
                "default": "my-model",
            },
        }
        result = _normalize_root_model_keys(config)
        assert result["model"]["context_length"] == 128000
        assert "context_length" not in result  # root key cleaned up

    def test_normalize_root_context_length_does_not_override_existing(self):
        """Existing model.context_length is not overridden by root-level key."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "context_length": 256000,
            "model": {
                "default": "my-model",
                "context_length": 128000,
            },
        }
        result = _normalize_root_model_keys(config)
        assert result["model"]["context_length"] == 128000  # preserved
        assert "context_length" not in result  # root key still cleaned up

    def test_normalize_root_context_length_with_string_model(self):
        """Root-level context_length is migrated even when model is a string."""
        from hermes_cli.config import _normalize_root_model_keys

        config = {
            "context_length": 128000,
            "model": "my-model",
        }
        result = _normalize_root_model_keys(config)
        assert isinstance(result["model"], dict)
        assert result["model"]["default"] == "my-model"
        assert result["model"]["context_length"] == 128000
        assert "context_length" not in result


class TestProviderResolution:
    def test_api_key_is_string_or_none(self):
        cli = _make_cli()
        assert cli.api_key is None or isinstance(cli.api_key, str)

    def test_base_url_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.base_url, str)
        assert cli.base_url.startswith("http")

    def test_model_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.model, str)
        assert isinstance(cli.model, str) and '/' in cli.model
