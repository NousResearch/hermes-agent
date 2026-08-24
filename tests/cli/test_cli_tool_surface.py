"""The welcome banner projects callable tools without changing session policy."""

import os
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.tool_resolution import ToolResolutionRequest


@pytest.fixture
def cli_obj(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setattr("shutil.get_terminal_size", lambda: os.terminal_size((160, 40)))
    obj = HermesCLI.__new__(HermesCLI)
    obj.model = "test-model"
    obj.enabled_toolsets = ["file"]
    obj.disabled_toolsets = []
    obj.compact = False
    obj.console = MagicMock()
    obj.session_id = None
    obj.api_key = "test"
    obj.base_url = ""
    obj.provider = "test"
    obj._provider_source = None
    obj.agent = None
    obj._show_tool_availability_warnings = MagicMock()
    return obj


def prefetch(cli_obj, future):
    cli_obj._startup_tool_resolution = (
        ToolResolutionRequest.from_lists(cli_obj.enabled_toolsets, cli_obj.disabled_toolsets),
        future,
    )


def test_banner_waits_then_consumes_exact_surface(cli_obj):
    entered, release = threading.Event(), threading.Event()
    expected = [{"function": {"name": "read_file"}}]
    errors = []

    def resolve(**_kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return expected

    def show():
        try:
            cli_obj.show_banner()
        except Exception as exc:
            errors.append(exc)

    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions", side_effect=resolve),
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        render = threading.Thread(target=show)
        render.start()
        try:
            assert entered.wait(timeout=5)
            banner.assert_not_called()
        finally:
            release.set()
            render.join(timeout=5)
    assert not render.is_alive()
    assert errors == []
    assert banner.call_args.kwargs["tools"] == expected
    assert cli_obj._startup_tool_resolution is None
    cli_obj._show_tool_availability_warnings.assert_called_once_with()


def test_failed_prefetch_retried_once(cli_obj):
    failed = Future()
    failed.set_exception(RuntimeError("transient"))
    prefetch(cli_obj, failed)
    expected = [{"function": {"name": "read_file"}}]
    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions", return_value=expected) as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    resolve.assert_called_once()
    assert banner.call_args.kwargs["tools"] == expected
    assert cli_obj._startup_tool_resolution is None


@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("persistent")])
def test_discovery_failure_is_not_deny_all(cli_obj, error):
    failed = Future()
    failed.set_exception(error)
    prefetch(cli_obj, failed)
    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions", side_effect=error) as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    assert resolve.call_count == (0 if isinstance(error, TimeoutError) else 1)
    assert cli_obj.enabled_toolsets == ["file"]
    banner.assert_not_called()  # unknown must not be presented as a zero count
    assert any("Tool discovery unavailable" in str(call) for call in cli_obj.console.print.call_args_list)
    cli_obj._show_tool_availability_warnings.assert_not_called()


@pytest.mark.parametrize("selection", [[], (), set()])
def test_explicit_empty_banner_never_discovers(cli_obj, selection):
    cli_obj.enabled_toolsets = selection
    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions") as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    assert banner.call_args.kwargs["tools"] == []
    resolve.assert_not_called()


def test_live_agent_surface_wins_over_prefetch(cli_obj):
    cli_obj.agent = SimpleNamespace(tools=[])
    future = Future()
    future.set_result([{"function": {"name": "read_file"}}])
    prefetch(cli_obj, future)
    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions") as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    assert banner.call_args.kwargs["tools"] == []
    assert cli_obj._startup_tool_resolution is None
    resolve.assert_not_called()


def test_later_banner_uses_fresh_canonical_resolution(cli_obj):
    first, second = [], [{"function": {"name": "read_file"}}]
    future = Future()
    future.set_result(first)
    prefetch(cli_obj, future)
    with (
        patch("hermes_cli.tool_resolution.get_cli_tool_definitions", return_value=second) as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
        cli_obj.show_banner()
    resolve.assert_called_once()
    assert [call.kwargs["tools"] for call in banner.call_args_list] == [first, second]


@pytest.mark.parametrize("selection", [[], (), set()])
def test_empty_diagnostics_do_not_import_registry(cli_obj, selection):
    from cli import HermesCLI

    cli_obj.enabled_toolsets = selection
    with patch("builtins.__import__", wraps=__import__) as imports:
        HermesCLI._show_tool_availability_warnings(cli_obj)
    assert not any(call.args[0] == "model_tools" for call in imports.call_args_list)


@pytest.mark.parametrize("enabled,disabled", [(["web", "tts"], ["tts"]), (["hermes-cli"], ["hermes-cli"]), (None, ["tts"])])
def test_diagnostics_use_canonical_bundle_semantics(cli_obj, enabled, disabled):
    from cli import HermesCLI
    from model_tools import _select_tool_names, get_toolset_for_tool

    cli_obj.enabled_toolsets, cli_obj.disabled_toolsets = enabled, disabled
    expected = {owner for name in _select_tool_names(enabled, disabled, True) if (owner := get_toolset_for_tool(name))}
    with patch("model_tools.check_tool_availability", return_value=([], [])) as check:
        HermesCLI._show_tool_availability_warnings(cli_obj)
    check.assert_called_once_with(toolsets=expected)


@pytest.mark.parametrize("selection", ["", [], (), set()])
def test_main_preserves_explicit_empty(cli_obj, selection):
    from cli import main

    with patch("cli.HermesCLI", return_value=cli_obj) as constructor, patch.object(cli_obj, "run"):
        main(toolsets=selection)
    assert constructor.call_args.kwargs["toolsets"] == []


def test_one_shot_does_not_consume_banner_prefetch(cli_obj):
    from cli import main

    pending = Future()
    with (
        patch("cli.HermesCLI", return_value=cli_obj),
        patch("cli._run_single_query_mode"),
        patch.object(pending, "result", side_effect=AssertionError("banner-only work")),
        patch("hermes_cli.tool_resolution.start_tool_surface_resolution", side_effect=AssertionError("banner-only work")),
    ):
        main(query="test", oneshot=True, toolsets=["file"], _prefetched_tool_resolution=(ToolResolutionRequest.from_lists(["file"], []), pending))
    assert cli_obj.enabled_toolsets == ["file"]


@pytest.mark.parametrize("compact", [False, True])
def test_deferred_banner_never_claims_zero_tools(cli_obj, monkeypatch, compact):
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli_obj.compact = compact
    with (
        patch("hermes_cli.tool_resolution._submit_daemon") as resolve,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    resolve.assert_not_called()
    banner.assert_not_called()
    printed = str(cli_obj.console.print.call_args_list)
    assert "tools deferred" in printed
    assert "0 tools" not in printed
    assert "Tool discovery unavailable" not in printed
    cli_obj._show_tool_availability_warnings.assert_not_called()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("live", [False, True])
def test_known_surface_wins_over_deferral(cli_obj, monkeypatch, compact, live):
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli_obj.compact = compact
    tools = [{"function": {"name": "read_file"}}] if live else []
    if live:
        cli_obj.agent = SimpleNamespace(tools=tools)
    else:
        cli_obj.enabled_toolsets = []
    with (
        patch("hermes_cli.tool_resolution._submit_daemon") as start,
        patch("hermes_cli.banner.build_welcome_banner") as banner,
    ):
        cli_obj.show_banner()
    start.assert_not_called()
    if compact:
        assert f"{len(tools)} tools" in str(cli_obj.console.print.call_args_list)
    else:
        assert banner.call_args.kwargs["tools"] == tools
    assert "tools deferred" not in str(cli_obj.console.print.call_args_list)
