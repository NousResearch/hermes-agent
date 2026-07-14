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


def test_managed_windows_native_fallback_is_consistent(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    node_root = home / "node"
    node_root.mkdir(parents=True)
    shim = _make_executable(
        node_root / "agent-browser.cmd",
        "#!/bin/sh\n# node agent-browser.js\nexit 1\n",
    )
    package_bin = node_root / "node_modules" / "agent-browser" / "bin"
    package_bin.mkdir(parents=True)
    native_path = package_bin / "agent-browser-win32-x64.exe"
    native_path.write_text("#!/bin/sh\nexit 0\n")
    native_path.chmod(0o644)
    native = str(native_path)

    monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
    monkeypatch.setattr(hermes_constants.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(browser_tool, "get_hermes_home", lambda: home)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path and str(node_root) in path:
            return shim
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(
        browser_tool, "_candidate_agent_browser_native_bins", lambda: []
    )
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)

    assert dep_ensure._DEP_CHECKS["browser"]() is True
    assert native_path.stat().st_mode & 0o111
    assert nous_subscription._has_agent_browser() is True
    assert any(
        doctor._resolve_agent_browser_for_doctor(candidate) == native
        for candidate in doctor._agent_browser_candidates_for_doctor()
    )
    assert browser_tool._find_agent_browser() == native


def test_repo_local_native_without_shim_is_consistent(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    package_bin = repo_root / "node_modules" / "agent-browser" / "bin"
    package_bin.mkdir(parents=True)
    native = _make_executable(
        package_bin / "agent-browser-linux-x64",
        "#!/bin/sh\nexit 0\n",
    )
    home = tmp_path / "hermes-home"
    home.mkdir()

    monkeypatch.setattr(hermes_constants.sys, "platform", "linux")
    monkeypatch.setattr(hermes_constants.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(browser_tool, "get_hermes_home", lambda: home)
    monkeypatch.setattr(
        browser_tool,
        "__file__",
        str(repo_root / "tools" / "browser_tool.py"),
    )
    monkeypatch.setattr(
        nous_subscription,
        "__file__",
        str(repo_root / "hermes_cli" / "nous_subscription.py"),
    )
    monkeypatch.setattr(
        dep_ensure,
        "__file__",
        str(repo_root / "hermes_cli" / "dep_ensure.py"),
    )
    monkeypatch.setattr(doctor, "PROJECT_ROOT", repo_root)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    monkeypatch.setattr(shutil, "which", lambda cmd, path=None: None)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(dep_ensure, "_has_hermes_agent_browser", lambda: False)
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)

    local_shim = repo_root / "node_modules" / ".bin" / "agent-browser"
    assert not local_shim.exists()
    assert browser_tool._find_agent_browser() == native
    assert nous_subscription._has_agent_browser() is True
    assert dep_ensure._DEP_CHECKS["browser"]() is True
    assert any(
        doctor._resolve_agent_browser_for_doctor(candidate) == native
        for candidate in doctor._agent_browser_candidates_for_doctor()
    )


def test_ambient_shim_with_managed_node_is_consistent(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    managed_bin = home / "node" / "bin"
    managed_bin.mkdir(parents=True)
    node = _make_executable(managed_bin / "node", "#!/bin/sh\nexit 0\n")
    shim = _make_executable(
        tmp_path / "agent-browser",
        "#!/bin/sh\n# node agent-browser.js\nnode --version >/dev/null\n",
    )

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(browser_tool, "get_hermes_home", lambda: home)
    monkeypatch.setattr(doctor, "HERMES_HOME", home)

    def fake_which(cmd, path=None):
        if cmd == "agent-browser" and path is None:
            return shim
        if cmd == "node" and path and str(managed_bin) in path:
            return node
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(dep_ensure, "_has_system_browser", lambda: False)
    monkeypatch.setattr(
        browser_tool, "_candidate_agent_browser_native_bins", lambda: []
    )
    monkeypatch.setattr(browser_tool, "_cached_agent_browser", None)
    monkeypatch.setattr(browser_tool, "_agent_browser_resolved", False)

    assert dep_ensure._DEP_CHECKS["browser"]() is True
    assert nous_subscription._has_agent_browser() is True
    assert doctor._resolve_agent_browser_for_doctor(shim) == shim
    assert browser_tool._find_agent_browser() == shim
