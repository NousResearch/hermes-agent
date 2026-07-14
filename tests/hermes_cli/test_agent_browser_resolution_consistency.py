import shutil

import hermes_constants
import tools.browser_tool as browser_tool
from hermes_cli import dep_ensure, doctor, nous_subscription


def _make_executable(path, body):
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def test_runtime_and_availability_share_windows_native_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
    monkeypatch.setattr(hermes_constants.platform, "machine", lambda: "AMD64")
    shim = _make_executable(
        tmp_path / "agent-browser.cmd",
        "#!/bin/sh\n# node agent-browser.js\nexit 1\n",
    )
    package_bin = tmp_path / "node_modules" / "agent-browser" / "bin"
    package_bin.mkdir(parents=True)
    native = _make_executable(
        package_bin / "agent-browser-win32-x64.exe",
        "#!/bin/sh\nexit 0\n",
    )

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return shim
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_has_hermes_agent_browser", lambda: False)
    monkeypatch.setattr(
        browser_tool, "_candidate_agent_browser_native_bins", lambda: []
    )
    monkeypatch.setattr(browser_tool, "_merge_browser_path", lambda path="": "")
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)

    assert browser_tool._find_agent_browser() == native
    assert dep_ensure._DEP_CHECKS["browser"]() is True
    assert nous_subscription._has_agent_browser() is True
    assert doctor._resolve_agent_browser_for_doctor(shim) == native
