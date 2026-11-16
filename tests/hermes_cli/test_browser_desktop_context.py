"""Desktop context is request-local, never a daemon-wide environment update."""
import os
from types import SimpleNamespace

from hermes_cli import cli_browser_gateway
from hermes_cli.browser_connect import launch_chrome_debug


def test_local_connect_sends_only_desktop_context(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DISPLAY", ":42")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    monkeypatch.setenv("API_KEY", "never-send")
    calls = []
    connection = SimpleNamespace(call=lambda method, **params: calls.append(params) or {"connected": False})
    cli_browser_gateway.gateway_browser(SimpleNamespace(session_id="test"), connection, "connect", "")
    assert calls[0]["desktop_env"]["DISPLAY"] == ":42"
    assert calls[0]["desktop_env"]["WAYLAND_DISPLAY"] == "wayland-test"
    assert "API_KEY" not in calls[0]["desktop_env"]


def test_explicit_gateway_never_receives_desktop_context(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:9999")
    monkeypatch.setenv("DISPLAY", ":42")
    calls = []
    connection = SimpleNamespace(call=lambda method, **params: calls.append(params) or {"connected": False})
    cli_browser_gateway.gateway_browser(SimpleNamespace(session_id="test"), connection, "connect", "")
    assert "desktop_env" not in calls[0]


def test_browser_child_gets_context_without_changing_parent(tmp_path, monkeypatch):
    from hermes_cli import browser_connect
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISPLAY", ":old")
    monkeypatch.setattr(browser_connect, "get_chrome_debug_candidates", lambda system: ["fake-browser"])
    monkeypatch.setattr(browser_connect, "_wait_for_browser_debug_ready_or_exit", lambda *args: "ready")
    children = []
    monkeypatch.setattr(browser_connect.subprocess, "Popen", lambda *args, **kw: children.append(kw) or SimpleNamespace(pid=1))
    result = launch_chrome_debug(9222, "Linux", desktop_env={"DISPLAY": ":42"})
    assert result.launched
    assert children[0]["env"]["DISPLAY"] == ":42"
    assert os.environ["DISPLAY"] == ":old"


def test_invalid_context_cannot_override_executable_search(monkeypatch):
    import pytest
    with pytest.raises(ValueError, match="invalid browser desktop context"):
        launch_chrome_debug(9222, "Linux", desktop_env={"PATH": "/untrusted"})


def test_local_gateway_launch_forwards_context(monkeypatch):
    from tui_gateway import methods_browser
    from hermes_cli import browser_connect
    calls = []
    monkeypatch.setattr(browser_connect, "discover_local_cdp_url", lambda *args, **kw: None)
    monkeypatch.setattr(browser_connect, "local_port_in_use", lambda port: False)
    monkeypatch.setattr(browser_connect, "manual_chrome_debug_command", lambda *args: None)
    monkeypatch.setattr(browser_connect, "launch_chrome_debug", lambda *args, **kw: calls.append(kw) or browser_connect.ChromeDebugLaunch())
    context = {"DISPLAY": ":42"}
    methods_browser._connect_local_default(9222, "Linux", lambda *args, **kw: None, desktop_env=context)
    assert calls == [{"desktop_env": context}]


def test_real_browser_child_receives_context(tmp_path, monkeypatch):
    import json
    import sys
    import subprocess
    from hermes_cli import browser_connect
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DISPLAY", raising=False)
    output = tmp_path / "child.json"
    script = tmp_path / "browser.py"
    script.write_text("import os,json,pathlib\npathlib.Path(" + repr(str(output)) + ").write_text(json.dumps({'display':os.environ.get('DISPLAY')}))\n")
    monkeypatch.setattr(browser_connect, "get_chrome_debug_candidates", lambda system: [sys.executable])
    monkeypatch.setattr(browser_connect, "_chrome_debug_args", lambda port: [str(script)])
    popen = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(browser_connect.subprocess, "Popen", spawn)
    def finished(child, port):
        child.wait(timeout=5)
        return "exited"
    monkeypatch.setattr(browser_connect, "_wait_for_browser_debug_ready_or_exit", finished)
    launch_chrome_debug(9222, "Linux", desktop_env={"DISPLAY": ":42"})
    assert json.loads(output.read_text()) == {"display": ":42"}
    assert "DISPLAY" not in os.environ
