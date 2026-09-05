"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest



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
    try:
        with patch.dict(sys.modules, prompt_toolkit_stubs), \
             patch.dict("os.environ", clean_env, clear=False):
            import cli as _cli_mod
            _cli_mod = importlib.reload(_cli_mod)
            with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
                 patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
                return _cli_mod.HermesCLI(**kwargs)
    finally:
        # The reload above re-executed cli.py while prompt_toolkit was stubbed
        # with MagicMocks, permanently rebinding cli's module globals
        # (``_pt_print``, ``_PT_ANSI``, …) to those mocks. ``patch.dict``
        # restores ``sys.modules`` on exit, but NOT the names the reloaded
        # module already bound — so ``sys.modules["cli"]`` is left with a
        # mock ``_pt_print``, and ``cli._cprint`` then silently no-ops for
        # every later test (one half of the order-dependent
        # ``test_resume_quiet_stderr`` full-suite failure; the other half is
        # the prompt_toolkit output cache reset in this dir's conftest).
        # Reload once more with the real modules visible so cli's globals
        # rebind cleanly.
        import cli as _cli_restore
        importlib.reload(_cli_restore)


class TestMaxTurnsResolution:
    """max_turns must always resolve to a positive integer, never None."""

    def test_default_max_turns_is_unlimited(self):
        # Default is now unlimited (max_turns caused more problems than it
        # solved). Still a positive int (the sys.maxsize sentinel), so loop
        # conditions like `count < max_iterations` keep working.
        import sys
        cli = _make_cli()
        assert isinstance(cli.max_turns, int)
        assert cli.max_turns == sys.maxsize

    def test_explicit_max_turns_honored(self):
        cli = _make_cli(max_turns=25)
        assert cli.max_turns == 25




    def test_legacy_root_max_turns_is_used_when_agent_key_exists_without_value(self):
        cli_obj = _make_cli(config_overrides={"agent": {}, "max_turns": 77})
        assert cli_obj.max_turns == 77



class TestVerboseAndToolProgress:
    def test_default_verbose_is_bool(self):
        cli = _make_cli()
        assert isinstance(cli.verbose, bool)

    def test_tool_progress_mode_is_string(self):
        cli = _make_cli()
        assert isinstance(cli.tool_progress_mode, str)
        assert cli.tool_progress_mode in {"off", "new", "all", "verbose"}


class TestToolsetValidationWarning:
    """Constructor-time validation must not false-positive on plugin toolsets
    that discover_plugins() would register moments later (#91757).

    ``_make_cli`` reloads ``cli``, so the patches target the underlying
    modules (``toolsets``, ``hermes_cli.plugins``) whose attributes the CLI's
    thin wrappers resolve on every call — patching ``cli`` itself would be
    wiped by the reload.
    """

    def test_plugin_toolset_false_positive_suppressed(self, capsys):
        import toolsets as toolsets_mod
        import hermes_cli.plugins as plugins_mod

        discovered = {"ran": False}

        def _fake_validate(name):
            if name == "hermes-cli":
                return True
            return discovered["ran"]  # plugin toolset known only after discovery

        with patch.object(toolsets_mod, "validate_toolset", side_effect=_fake_validate), \
             patch.object(plugins_mod, "discover_plugins", side_effect=lambda: discovered.__setitem__("ran", True)):
            _make_cli(toolsets=["hermes-cli", "cad_compare"])

        assert "Unknown toolsets" not in capsys.readouterr().out

    def test_genuinely_unknown_toolset_still_warns(self, capsys):
        import toolsets as toolsets_mod
        import hermes_cli.plugins as plugins_mod

        def _fake_validate(name):
            return name == "hermes-cli"

        with patch.object(toolsets_mod, "validate_toolset", side_effect=_fake_validate), \
             patch.object(plugins_mod, "discover_plugins", side_effect=lambda: None):
            _make_cli(toolsets=["hermes-cli", "ghost_set"])

        assert "Unknown toolsets: ghost_set" in capsys.readouterr().out

    def test_discovery_crash_does_not_propagate_and_still_warns(self, capsys):
        """A plugin raising during discovery must not escape the constructor;
        the toolset warning still prints so the failure stays user-visible."""
        import toolsets as toolsets_mod
        import hermes_cli.plugins as plugins_mod

        def _fake_validate(name):
            return name == "hermes-cli"

        with patch.object(toolsets_mod, "validate_toolset", side_effect=_fake_validate), \
             patch.object(plugins_mod, "discover_plugins", side_effect=RuntimeError("boom")):
            _make_cli(toolsets=["hermes-cli", "ghost_set"])  # must not raise

        assert "Unknown toolsets: ghost_set" in capsys.readouterr().out

    def test_multiple_invalid_names_trigger_single_discovery_pass(self, capsys):
        """Discovery runs once for the whole invalid list, not once per name."""
        import toolsets as toolsets_mod
        import hermes_cli.plugins as plugins_mod

        def _fake_validate(name):
            return name == "hermes-cli"

        with patch.object(toolsets_mod, "validate_toolset", side_effect=_fake_validate), \
             patch.object(plugins_mod, "discover_plugins", side_effect=lambda: None) as discover:
            _make_cli(toolsets=["hermes-cli", "ghost_set", "phantom_set"])

        discover.assert_called_once()
        out = capsys.readouterr().out
        assert "Unknown toolsets: ghost_set, phantom_set" in out


class TestFallbackChainInit:
    def test_merges_new_and_legacy_fallback_config(self):
        cli = _make_cli(config_overrides={
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            ],
            "fallback_model": {"provider": "nous", "model": "Hermes-4"},
        })
        assert cli._fallback_model == [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            {"provider": "nous", "model": "Hermes-4"},
        ]


class TestBusyInputMode:
    def test_default_busy_input_mode_is_interrupt(self):
        cli = _make_cli()
        assert cli.busy_input_mode == "interrupt"

    def test_busy_input_mode_queue_is_honored(self):
        cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
        assert cli.busy_input_mode == "queue"


    def test_queue_command_works_while_busy(self):
        """When agent is running, /queue should still put the prompt in _pending_input."""
        cli = _make_cli()
        cli._agent_running = True
        cli.process_command("/queue follow up")
        assert cli._pending_input.get_nowait() == "follow up"




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
    def test_lf_enter_binding_respects_multiline_shortcuts(self):
        """Ctrl+J is reserved by default, with legacy LF-submit available as an opt-out.

        Some thin POSIX PTYs deliver plain Enter as LF/c-j instead of CR/enter.
        The default keeps c-j free for multiline input; disabling multiline
        shortcuts restores c-j → submit on bare local POSIX terminals. Windows,
        WSL, SSH sessions, Windows Terminal, and Ghostty always reserve c-j for
        the Ctrl+Enter/Ctrl+J newline binding. See issue #22379.

        The native-Windows arm of this behaviour is
        ``test_windows_leaves_ctrl_j_unbound`` below — it has to run on a real
        Windows host, because ``_bind_prompt_submit_keys`` delegates to
        ``_preserve_ctrl_enter_newline()``, which short-circuits on
        ``sys.platform == "win32"``. Faking that here would assert the literal
        in the ``if`` and nothing about how prompt_toolkit actually delivers
        keys on a Windows console.
        """
        import os as _os
        from unittest.mock import patch as _patch
        from prompt_toolkit.key_binding import KeyBindings

        from cli import _bind_prompt_submit_keys

        def submit_handler(event):
            return None

        # Default: Enter submits while c-j stays free for the newline binding.
        # (Runs on the POSIX CI job; the native-Windows arm is the marked test
        # below, so no sys.platform fake is needed here.)
        with _patch.dict(_os.environ, {}, clear=True), \
             _patch("builtins.open", side_effect=OSError("no /proc")):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert ("c-j",) not in bindings

            # Legacy opt-out: bare POSIX LF/c-j submits for thin PTYs.
            kb = KeyBindings()
            _bind_prompt_submit_keys(
                kb,
                submit_handler,
                multiline_shortcuts_enabled=False,
            )
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert bindings[("c-j",)] is submit_handler

        # POSIX over SSH: c-j stays free so Ctrl+Enter (sent as LF by
        # Windows Terminal / Kitty / mintty over SSH) inserts a newline.
        with _patch.dict(_os.environ, {"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, clear=True), \
             _patch("builtins.open", side_effect=OSError("no /proc")):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert ("c-j",) not in bindings

        # Ghostty through tmux: TERM_PROGRAM is tmux, but Ghostty exports a
        # stable env marker. Keep c-j free so Ctrl+J inserts a newline.
        with _patch.dict(_os.environ, {"TERM": "tmux-256color", "TERM_PROGRAM": "tmux", "GHOSTTY_RESOURCES_DIR": "/usr/share/ghostty"}, clear=True), \
             _patch("builtins.open", side_effect=OSError("no /proc")):
            kb = KeyBindings()
            _bind_prompt_submit_keys(kb, submit_handler)
            bindings = {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}
            assert bindings[("c-m",)] is submit_handler
            assert ("c-j",) not in bindings

    @pytest.mark.windows_only
    def test_windows_leaves_ctrl_j_unbound(self):
        """On native Windows only enter submits; c-j is free for the newline
        binding added separately in the prompt setup."""
        from prompt_toolkit.key_binding import KeyBindings

        from cli import _bind_prompt_submit_keys

        def submit_handler(event):
            return None

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



    def test_cpr_gating_posix_suppresses_without_ssh(self, monkeypatch):
        """POSIX suppresses CPR without SSH.

        The native-Windows arm (``_terminal_may_leak_cpr() is False``, plus
        the ``PROMPT_TOOLKIT_NO_CPR`` override that outranks it) lives in
        ``tests/cli/test_cpr_local_leak.py`` under ``windows_only``, where it
        runs against a real Windows console.
        """
        from cli import _terminal_may_leak_cpr

        for var in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "PROMPT_TOOLKIT_NO_CPR"):
            monkeypatch.delenv(var, raising=False)

        assert _terminal_may_leak_cpr() is True

        monkeypatch.setenv("PROMPT_TOOLKIT_NO_CPR", "1")
        assert _terminal_may_leak_cpr() is True


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
        assert "Use /resume" in output
        assert "session title" in output



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


class TestNestedDictModelDefaultPairing:
    """A dict-valued ``model.default`` must keep its nested provider paired.

    ``model.default: {provider: ..., model: ...}`` canonicalizes to the string
    model AND the nested provider, so ``HermesCLI`` routes the model through
    that provider instead of discarding it and falling back to the outer
    merged ``model.provider`` (``"auto"`` — authoritative at runtime
    resolution, which would route the model through the wrong active
    provider).
    """

    def test_nested_dict_default_keeps_provider_paired(self):
        cli = _make_cli(config_overrides={
            "model": {
                "default": {"provider": "nous", "model": "nested-default-model"},
                "provider": "auto",
            },
        })
        assert cli.model == "nested-default-model"
        assert cli.requested_provider == "nous"
        assert cli.provider == "nous"

    def test_nested_dict_model_alias_keeps_provider_paired(self):
        cli = _make_cli(config_overrides={
            "model": {
                "model": {"provider": "openai", "model": "nested-alias-model"},
                "provider": "auto",
            },
        })
        assert cli.model == "nested-alias-model"
        assert cli.requested_provider == "openai"
        assert cli.provider == "openai"

    def test_flat_string_default_still_uses_outer_provider(self):
        cli = _make_cli(config_overrides={
            "model": {
                "default": "flat-default-model",
                "provider": "auto",
            },
        })
        assert cli.model == "flat-default-model"
        assert cli.requested_provider == "auto"
        assert cli.provider == "auto"

    def test_nested_provider_does_not_override_explicit_provider_arg(self):
        cli = _make_cli(
            config_overrides={
                "model": {
                    "default": {"provider": "nous", "model": "nested-default-model"},
                    "provider": "auto",
                },
            },
            provider="anthropic",
        )
        assert cli.model == "nested-default-model"
        assert cli.requested_provider == "anthropic"
        assert cli.provider == "anthropic"

    def test_whoami_command_is_dispatched_and_prints_cli_access(self, capsys):
        """/whoami is advertised in classic CLI help and must not fall through.

        Regression test: the command existed in the shared registry, so it
        appeared in /help and completion, but classic CLI dispatch lacked a
        matching branch and printed `Unknown command: /whoami`.
        """
        cli = _make_cli()

        cli.process_command("/whoami")
        output = capsys.readouterr().out

        assert "Unknown command" not in output
        assert "cli (local terminal)" in output
        assert "Tier:" in output
        assert "unrestricted" in output
        assert "Slash commands: all available" in output

    def test_provider_prefixed_startup_model_overrides_stale_provider(self):
        cli = _make_cli(
            config_overrides={
                "model": {
                    "default": "anthropic/claude-opus-4.6",
                    "provider": "anthropic",
                },
                "providers": {
                    "nous": {
                        "base_url": "https://inference-api.nousresearch.com/v1",
                    },
                },
            },
            model="nous/deepseek-v4-pro",
        )

        assert cli.model == "deepseek-v4-pro"
        assert cli.requested_provider == "nous"


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

    def test_root_provider_used_as_fallback_when_model_provider_missing(self, tmp_path, monkeypatch):
        """Legacy root-level provider still populates model.provider in the CLI loader."""
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

        assert cfg["model"]["provider"] == "opencode-go"

    def test_root_base_url_used_as_fallback_when_model_base_url_missing(self, tmp_path, monkeypatch):
        """Legacy root-level base_url still populates model.base_url in the CLI loader."""
        import yaml

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "base_url": "https://example.com/v1",
            "model": {
                "default": "google/gemini-3-flash-preview",
            },
        }))

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        cfg = cli.load_cli_config()

        assert cfg["model"]["base_url"] == "https://example.com/v1"

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






    # --- model-id alias canonicalization (issue #34500) -------------------
    # ``model.name`` / ``model.model`` must canonicalize to ``model.default``
    # so the runtime resolver (and ~14 other readers) never sends an empty
    # ``model=`` to the backend. Precedence: default > model > name.


    def test_normalize_model_alias_to_default(self):
        """model.model becomes model.default."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({"model": {"model": "via-model-key"}})
        assert result["model"]["default"] == "via-model-key"
        assert "model" not in result["model"]



    def test_normalize_model_wins_over_name(self):
        """Precedence: model > name when both are aliases and default is empty."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({"model": {"model": "m-key", "name": "n-key"}})
        assert result["model"]["default"] == "m-key"
        assert "model" not in result["model"] and "name" not in result["model"]


    # --- dict-valued model.default flattening (PR #83902 follow-up) --------
    # ``model.default: {provider: ..., model: ...}`` must flatten into a string
    # ``model.default`` plus ``model.provider`` at the load chokepoint so every
    # reader (doctor, status, fallback picker, prompt-size, context-switch
    # guard) sees plain strings instead of a nested dict that crashes
    # ``.strip()``/``.lower()`` or routes the model through the wrong provider.

    def test_nested_dict_default_flattens_model_and_provider(self):
        """dict model.default -> string default + provider, no outer provider set."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({
            "model": {
                "default": {"provider": "nous", "model": "nested-default-model"},
            },
        })
        assert result["model"]["default"] == "nested-default-model"
        assert result["model"]["provider"] == "nous"

    def test_nested_dict_default_provider_wins_over_auto(self):
        """Nested provider replaces the merged default "auto"."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({
            "model": {
                "default": {"provider": "nous", "model": "nested-default-model"},
                "provider": "auto",
            },
        })
        assert result["model"]["default"] == "nested-default-model"
        assert result["model"]["provider"] == "nous"

    def test_nested_dict_default_never_overrides_explicit_provider(self):
        """An explicitly configured model.provider beats the nested provider."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({
            "model": {
                "default": {"provider": "nous", "model": "nested-default-model"},
                "provider": "anthropic",
            },
        })
        assert result["model"]["default"] == "nested-default-model"
        assert result["model"]["provider"] == "anthropic"

    def test_nested_dict_model_alias_flattens_to_default(self):
        """dict model.model alias also flattens (default > model > name)."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({
            "model": {
                "model": {"provider": "openai", "model": "nested-alias-model"},
            },
        })
        assert result["model"]["default"] == "nested-alias-model"
        assert result["model"]["provider"] == "openai"
        assert "model" not in result["model"]

    def test_flat_string_default_untouched(self):
        """Plain string defaults keep existing behavior exactly."""
        from hermes_cli.config import _normalize_root_model_keys

        result = _normalize_root_model_keys({
            "model": {"default": "flat-default-model", "provider": "auto"},
        })
        assert result["model"]["default"] == "flat-default-model"
        assert result["model"]["provider"] == "auto"



