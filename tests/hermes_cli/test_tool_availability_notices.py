"""Invariants for the CLI startup 'some tools disabled' notice (tools-runtime-01 / -02)."""

from hermes_cli.tool_availability_notices import filter_to_enabled_toolsets, tool_availability_warning_lines

_TERMINAL = {"name": "terminal", "env_vars": [], "tools": ["terminal", "process_manage"]}
_WEB = {"name": "web", "env_vars": ["EXA_API_KEY", "TAVILY_API_KEY"], "tools": ["web_search"]}
_RL = {"name": "rl", "env_vars": ["TINKER_API_KEY"], "tools": ["rl_tool"]}


def test_terminal_line_names_backend_reason_and_fix_commands():
    lines = tool_availability_warning_lines(
        [_TERMINAL], terminal_reason="Docker is installed but the Docker daemon is not running",
        terminal_backend="docker")
    text = "\n".join(lines)
    assert "Terminal tool disabled" in text
    assert "'docker'" in text
    assert "Docker daemon is not running" in text
    assert "hermes doctor" in text and "hermes setup terminal" in text
    assert "TERMINAL_ENV" not in text


def test_terminal_without_reason_still_gets_a_line():
    lines = tool_availability_warning_lines([_TERMINAL], terminal_reason=None, terminal_backend="ssh")
    assert any("Terminal tool disabled" in line and "'ssh'" in line for line in lines)


def test_web_collapses_env_dump_into_one_provider_sentence():
    lines = tool_availability_warning_lines([_WEB], terminal_reason=None, terminal_backend="local")
    text = "\n".join(lines)
    assert "EXA_API_KEY" not in text
    assert "Web search is off" in text and "hermes setup tools" in text


def test_generic_toolset_lists_vars_and_setup_hint():
    lines = tool_availability_warning_lines([_RL], terminal_reason=None, terminal_backend="local")
    text = "\n".join(lines)
    assert "rl" in text and "TINKER_API_KEY" in text and "hermes setup" in text


def test_nothing_to_report_returns_no_lines():
    assert tool_availability_warning_lines([], terminal_reason=None, terminal_backend="local") == []
    # A toolset with no env vars and no special notice (e.g. lazy check_fn) stays silent.
    assert tool_availability_warning_lines([{"name": "honcho", "env_vars": [], "tools": []}],
                                           terminal_reason=None, terminal_backend="local") == []


def test_web_notice_cites_setup_tools_not_the_toggle_menu():
    # `hermes tools` with no subcommand is the enable/disable menu; providers live under `hermes setup tools`.
    lines = tool_availability_warning_lines([_WEB], terminal_reason=None, terminal_backend="local")
    text = "\n".join(lines)
    assert "hermes setup tools" in text
    assert "hermes tools[/]" not in text


class TestFilterToEnabledToolsets:
    _BUNDLE = {"hermes-cli": ["terminal", "process_manage", "web_search", "read_file"],
               "terminal": ["terminal", "process_manage"]}

    def _resolve(self, name):
        return self._BUNDLE.get(name, [])

    def test_composite_bundle_keeps_terminal_notice_on_default_install(self):
        kept = filter_to_enabled_toolsets([_TERMINAL, _WEB, _RL], ["hermes-cli"], self._resolve)
        assert [k["name"] for k in kept] == ["terminal", "web"]

    def test_direct_toolset_name_and_unknown_bundle(self):
        assert [k["name"] for k in filter_to_enabled_toolsets([_TERMINAL, _RL], ["terminal"], self._resolve)] == ["terminal"]
        assert filter_to_enabled_toolsets([_TERMINAL], ["no-such-bundle"], self._resolve) == []

    def test_empty_selection_means_everything(self):
        assert filter_to_enabled_toolsets([_TERMINAL, _RL], [], self._resolve) == [_TERMINAL, _RL]

    def test_resolver_error_is_not_fatal(self):
        def boom(name):
            raise RuntimeError("registry not ready")
        assert filter_to_enabled_toolsets([_TERMINAL], ["hermes-cli"], boom) == []
