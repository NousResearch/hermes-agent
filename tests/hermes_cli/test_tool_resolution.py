import json
import os
import subprocess
import sys
import threading
from unittest.mock import patch

from hermes_cli.tool_resolution import (
    ToolResolutionRequest,
    get_cli_tool_definitions,
    resolve_cli_toolsets,
    resolve_startup_tools,
    start_cli_tool_resolution,
    start_tool_surface_resolution,
)


def test_cli_policy_preserves_explicit_empty_and_normalizes_values():
    assert resolve_cli_toolsets([], {}) == []
    assert resolve_cli_toolsets("", {}) == []
    assert resolve_cli_toolsets("web, terminal", {}) == ["web", "terminal"]
    assert resolve_cli_toolsets(("web,file", "skills"), {}) == [
        "web",
        "file",
        "skills",
    ]


def test_configured_empty_policy_precedes_coding_focus():
    with patch(
        "agent.coding_context.coding_selection",
        side_effect=AssertionError("empty policy must return before coding focus"),
    ) as coding:
        assert resolve_cli_toolsets(
            None, {"platform_toolsets": {"cli": []}}
        ) == []
    coding.assert_not_called()


def test_request_preserves_default_and_explicit_empty():
    assert ToolResolutionRequest.from_lists(None, []).enabled_toolsets is None
    assert ToolResolutionRequest.from_lists([], []).enabled_toolsets == frozenset()


def test_background_resolution_uses_immutable_request():
    enabled = ["web"]
    disabled = ["tts"]

    seen = []

    def resolver(**kwargs):
        seen.append((kwargs, threading.current_thread().daemon))
        return [{"function": {"name": "web_search"}}]

    with patch(
        "hermes_cli.tool_resolution.get_cli_tool_definitions", side_effect=resolver
    ):
        pending = start_tool_surface_resolution(enabled, disabled)
        assert pending is not None
        resolved_request, future = pending
        enabled.append("file")
        disabled.clear()
        result = future.result(timeout=2)

    assert resolved_request == ToolResolutionRequest.from_lists(["web"], ["tts"])
    assert result == [{"function": {"name": "web_search"}}]
    assert seen == [
        (
            {
                "enabled_toolsets": frozenset({"web"}),
                "disabled_toolsets": frozenset({"tts"}),
                "quiet_mode": True,
            },
            True,
        )
    ]


def test_cli_policy_is_available_while_definitions_resolve():
    entered = threading.Event()
    release = threading.Event()

    def resolver(**_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return [{"function": {"name": "web_search"}}]

    with patch(
        "hermes_cli.tool_resolution.get_cli_tool_definitions", side_effect=resolver
    ):
        pending = start_cli_tool_resolution(["web"], config={})
        assert pending is not None
        request, future = pending
        try:
            assert request.enabled_toolsets == {"web"}
            assert entered.wait(timeout=2)
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=2) == [
            {"function": {"name": "web_search"}}
        ]


def test_early_cli_resolution_knows_explicit_empty_without_worker_imports():
    pending = start_cli_tool_resolution([])

    assert pending is not None
    request, future = pending
    assert request.enabled_toolsets == frozenset()
    assert future.result(timeout=0) == []


def test_configured_empty_resolution_uses_the_supplied_config():
    with patch(
        "hermes_cli.tool_resolution._submit_daemon",
        side_effect=AssertionError("known-empty policy must not start a worker"),
    ):
        pending = start_cli_tool_resolution(
            None, config={"platform_toolsets": {"cli": []}}
        )

    assert pending is not None
    request, future = pending
    assert request.enabled_toolsets == frozenset()
    assert future.result(timeout=0) == []


def test_explicit_empty_resolution_imports_no_tool_runtime(tmp_path):
    script = """
import json
import sys
import threading
from hermes_cli.tool_resolution import start_cli_tool_resolution

request, future = start_cli_tool_resolution(())
print(json.dumps({
    "enabled": sorted(request.enabled_toolsets),
    "tools": future.result(timeout=0),
    "model_tools": "model_tools" in sys.modules,
    "registry": "tools.registry" in sys.modules,
    "worker": any(
        thread.name == "tool-surface-resolution"
        for thread in threading.enumerate()
    ),
}))
"""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    env.pop("HERMES_DEFER_AGENT_STARTUP", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(result.stdout) == {
        "enabled": [],
        "tools": [],
        "model_tools": False,
        "registry": False,
        "worker": False,
    }


def test_cli_definition_wrapper_short_circuits_positional_empty_policy():
    with patch(
        "hermes_cli.mcp_startup.wait_for_mcp_discovery",
        side_effect=AssertionError("empty policy must not wait for MCP"),
    ):
        assert get_cli_tool_definitions([]) == []


def test_equivalent_policy_reuses_prefetch():
    from concurrent.futures import Future

    original = ToolResolutionRequest.from_lists(["file", "web", "file"], ["tts", "vision"])
    reordered = ToolResolutionRequest.from_lists(["web", "file"], ["vision", "tts", "tts"])
    future = Future()
    expected = [{"function": {"name": "read_file"}}]
    future.set_result(expected)
    with patch("hermes_cli.tool_resolution.start_tool_surface_resolution") as start:
        assert resolve_startup_tools(reordered, (original, future)) == expected
    start.assert_not_called()


def test_changed_policy_does_not_reuse_prefetch():
    from concurrent.futures import Future

    original = ToolResolutionRequest.from_lists(["file"], [])
    future = Future()
    future.set_result([{"function": {"name": "read_file"}}])
    empty = ToolResolutionRequest.from_lists([], [])
    assert resolve_startup_tools(empty, (original, future)) == []


def test_deferred_resolution_is_unknown_not_empty(monkeypatch):
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    with patch("hermes_cli.tool_resolution._submit_daemon") as start:
        assert resolve_startup_tools(ToolResolutionRequest.from_lists(["file"], [])) is None
        assert resolve_startup_tools(ToolResolutionRequest.from_lists([], [])) == []
    start.assert_not_called()
