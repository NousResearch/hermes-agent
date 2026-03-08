"""Tests for HermesCLI initialization -- catches configuration bugs
that only manifest at runtime (not in mocked unit tests)."""

import os
import sys
import types
from contextlib import nullcontext
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _install_prompt_toolkit_stubs():
    """Provide minimal prompt_toolkit shims for non-TUI unit tests."""
    if "prompt_toolkit" in sys.modules:
        return

    class _StubBase:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class _StubStyle:
        @classmethod
        def from_dict(cls, *_args, **_kwargs):
            return cls()

    prompt_toolkit = types.ModuleType("prompt_toolkit")
    prompt_toolkit.print_formatted_text = lambda *args, **kwargs: None

    history = types.ModuleType("prompt_toolkit.history")
    history.FileHistory = _StubBase

    styles = types.ModuleType("prompt_toolkit.styles")
    styles.Style = _StubStyle

    patch_stdout = types.ModuleType("prompt_toolkit.patch_stdout")
    patch_stdout.patch_stdout = nullcontext

    application = types.ModuleType("prompt_toolkit.application")
    application.Application = _StubBase

    layout = types.ModuleType("prompt_toolkit.layout")
    layout.Layout = _StubBase
    layout.HSplit = _StubBase
    layout.Window = _StubBase
    layout.FormattedTextControl = _StubBase
    layout.ConditionalContainer = _StubBase

    processors = types.ModuleType("prompt_toolkit.layout.processors")
    processors.Processor = _StubBase
    processors.Transformation = _StubBase
    processors.PasswordProcessor = _StubBase
    processors.ConditionalProcessor = _StubBase

    filters = types.ModuleType("prompt_toolkit.filters")
    filters.Condition = lambda fn: fn

    dimension = types.ModuleType("prompt_toolkit.layout.dimension")
    dimension.Dimension = _StubBase

    menus = types.ModuleType("prompt_toolkit.layout.menus")
    menus.CompletionsMenu = _StubBase

    widgets = types.ModuleType("prompt_toolkit.widgets")
    widgets.TextArea = _StubBase

    key_binding = types.ModuleType("prompt_toolkit.key_binding")
    key_binding.KeyBindings = _StubBase

    completion = types.ModuleType("prompt_toolkit.completion")
    completion.Completer = object
    completion.Completion = _StubBase

    formatted_text = types.ModuleType("prompt_toolkit.formatted_text")
    formatted_text.ANSI = str

    sys.modules.update(
        {
            "prompt_toolkit": prompt_toolkit,
            "prompt_toolkit.history": history,
            "prompt_toolkit.styles": styles,
            "prompt_toolkit.patch_stdout": patch_stdout,
            "prompt_toolkit.application": application,
            "prompt_toolkit.layout": layout,
            "prompt_toolkit.layout.processors": processors,
            "prompt_toolkit.filters": filters,
            "prompt_toolkit.layout.dimension": dimension,
            "prompt_toolkit.layout.menus": menus,
            "prompt_toolkit.widgets": widgets,
            "prompt_toolkit.key_binding": key_binding,
            "prompt_toolkit.completion": completion,
            "prompt_toolkit.formatted_text": formatted_text,
        }
    )


def _install_rich_stubs():
    """Provide minimal rich shims for CLI unit tests."""
    if "rich" in sys.modules:
        return

    rich = types.ModuleType("rich")
    console = types.ModuleType("rich.console")
    panel = types.ModuleType("rich.panel")
    table = types.ModuleType("rich.table")

    class _RichStub:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    console.Console = _RichStub
    panel.Panel = _RichStub
    table.Table = _RichStub

    sys.modules.update(
        {
            "rich": rich,
            "rich.console": console,
            "rich.panel": panel,
            "rich.table": table,
        }
    )


def _install_cli_dependency_stubs():
    """Stub heavy runtime-only dependencies so CLI unit tests stay lightweight."""
    if "fire" not in sys.modules:
        sys.modules["fire"] = types.ModuleType("fire")

    if "run_agent" not in sys.modules:
        run_agent = types.ModuleType("run_agent")
        run_agent.AIAgent = object
        sys.modules["run_agent"] = run_agent

    if "model_tools" not in sys.modules:
        model_tools = types.ModuleType("model_tools")
        model_tools.get_tool_definitions = lambda *args, **kwargs: []
        model_tools.get_toolset_for_tool = lambda *args, **kwargs: None
        sys.modules["model_tools"] = model_tools

    if "hermes_cli.banner" not in sys.modules:
        banner = types.ModuleType("hermes_cli.banner")
        banner.cprint = lambda *args, **kwargs: None
        banner._GOLD = banner._BOLD = banner._DIM = banner._RST = ""
        banner.VERSION = "test"
        banner.HERMES_AGENT_LOGO = ""
        banner.HERMES_CADUCEUS = ""
        banner.COMPACT_BANNER = ""
        banner.get_available_skills = lambda *args, **kwargs: []
        banner.build_welcome_banner = lambda *args, **kwargs: ""
        sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli"))
        sys.modules["hermes_cli.banner"] = banner

    if "hermes_cli.commands" not in sys.modules:
        commands = types.ModuleType("hermes_cli.commands")
        commands.COMMANDS = {}
        commands.SlashCommandCompleter = object
        sys.modules["hermes_cli.commands"] = commands

    if "hermes_cli.callbacks" not in sys.modules:
        callbacks = types.ModuleType("hermes_cli.callbacks")
        callbacks.register_approval_callback = lambda *args, **kwargs: None
        callbacks.register_sudo_password_callback = lambda *args, **kwargs: None
        sys.modules["hermes_cli.callbacks"] = callbacks
        sys.modules.setdefault("hermes_cli", types.ModuleType("hermes_cli")).callbacks = callbacks

    if "toolsets" not in sys.modules:
        toolsets = types.ModuleType("toolsets")
        toolsets.get_all_toolsets = lambda *args, **kwargs: []
        toolsets.get_toolset_info = lambda *args, **kwargs: {}
        toolsets.resolve_toolset = lambda *args, **kwargs: []
        toolsets.validate_toolset = lambda *_args, **_kwargs: True
        sys.modules["toolsets"] = toolsets

    if "cron" not in sys.modules:
        cron = types.ModuleType("cron")
        cron.create_job = lambda *args, **kwargs: None
        cron.list_jobs = lambda *args, **kwargs: []
        cron.remove_job = lambda *args, **kwargs: None
        cron.get_job = lambda *args, **kwargs: None
        sys.modules["cron"] = cron

    sys.modules.setdefault("tools", types.ModuleType("tools"))

    if "tools.terminal_tool" not in sys.modules:
        terminal_tool = types.ModuleType("tools.terminal_tool")
        terminal_tool.cleanup_all_environments = lambda *args, **kwargs: None
        terminal_tool.set_sudo_password_callback = lambda *args, **kwargs: None
        terminal_tool.set_approval_callback = lambda *args, **kwargs: None
        sys.modules["tools.terminal_tool"] = terminal_tool

    if "tools.browser_tool" not in sys.modules:
        browser_tool = types.ModuleType("tools.browser_tool")
        browser_tool._emergency_cleanup_all_sessions = lambda *args, **kwargs: None
        sys.modules["tools.browser_tool"] = browser_tool


def _make_cli(env_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    _install_prompt_toolkit_stubs()
    _install_rich_stubs()
    _install_cli_dependency_stubs()
    import cli as _cli_mod
    from cli import HermesCLI
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
    if env_overrides:
        clean_env.update(env_overrides)
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch.dict("os.environ", clean_env, clear=False), \
         patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
        return HermesCLI(**kwargs)


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
