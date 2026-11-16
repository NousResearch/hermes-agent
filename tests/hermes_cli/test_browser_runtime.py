"""Browser resolution uses installed PM facts and config without provisioning a browser."""

import os

import pytest

import pm
from hermes_cli import browser_runtime
from pm import paths


def _fake_pm_chromium(store):
    lock = pm.Lockfile(paths.lockfile_path())
    target = pm.current_target()
    package = pm.get_package("chromium")
    version = lock.version("chromium")
    assert version is not None
    entry = store / package.store_entry(version, target)
    binary = entry / "browser" / ("chrome.exe" if target.startswith("win32") else "chrome")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"browser-fixture")
    binary.chmod(0o755)
    facts = pm.Facts(store / "facts.json")
    facts.record(
        "chromium", version, entry.name, {"PLAYWRIGHT_BROWSERS_PATH": str(store)}, store,
        target=target, artifacts=[a["sha256"] for a in lock.artifacts("chromium", target)],
    )
    return binary, facts


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(browser_runtime, "BAKED_EXECUTABLE_FILE", tmp_path / "etc" / "chromium-executable-path")
    return tmp_path


def _write_config(home, text):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_chromium_resolves_installed_binary_without_mutation(homes):
    binary, facts = _fake_pm_chromium(homes / "tools")
    before = facts.path.read_bytes()
    environment = dict(os.environ)

    assert browser_runtime.chromium_executable() == str(binary)
    assert facts.path.read_bytes() == before
    assert dict(os.environ) == environment


def test_configured_executable_wins_without_installing(homes, monkeypatch):
    override = homes / "external browser"
    _write_config(homes / "home", f"browser:\n  executable_path: '{override}'\n")

    def forbidden(*args, **kwargs):
        pytest.fail("browser resolution must not provision or activate packages")

    monkeypatch.setattr(pm, "ensure", forbidden)
    monkeypatch.setattr(pm, "activate", forbidden)
    with monkeypatch.context() as scoped:
        scoped.setattr(pm, "installed_package", forbidden)
        assert browser_runtime.chromium_executable() == str(override)
    # allow_override=False asks for pm's own build only.
    assert browser_runtime.chromium_executable(allow_override=False) is None
    assert not (homes / "tools").exists()


def test_agent_browser_env_is_not_an_override(homes, monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(homes / "legacy"))
    assert browser_runtime.chromium_executable() is None


def test_override_is_read_from_the_active_profile(homes):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    a, b = homes / "a", homes / "b"
    _write_config(a, "browser:\n  executable_path: /opt/a/chrome\n")
    _write_config(b, "browser:\n  executable_path: ''\n")
    for home, expected in ((a, "/opt/a/chrome"), (b, None), (a, "/opt/a/chrome")):
        token = set_hermes_home_override(home)
        try:
            assert browser_runtime.chromium_executable() == expected
        finally:
            reset_hermes_home_override(token)


def test_baked_docker_path_after_pm(homes):
    baked_binary = homes / "opt" / "chrome"
    baked_binary.parent.mkdir()
    baked_binary.write_bytes(b"x")
    browser_runtime.BAKED_EXECUTABLE_FILE.parent.mkdir()
    browser_runtime.BAKED_EXECUTABLE_FILE.write_text(f"{baked_binary}\n", encoding="utf-8")
    assert browser_runtime.chromium_executable() == str(baked_binary)

    binary, _ = _fake_pm_chromium(homes / "tools")
    assert browser_runtime.chromium_executable() == str(binary)


def test_baked_path_to_a_missing_binary_is_ignored_and_hinted(homes):
    browser_runtime.BAKED_EXECUTABLE_FILE.parent.mkdir()
    browser_runtime.BAKED_EXECUTABLE_FILE.write_text(str(homes / "gone"), encoding="utf-8")
    assert browser_runtime.chromium_executable() is None
    assert "pull the current Hermes image" in browser_runtime.chromium_missing_hint()


@pytest.mark.platforms("posix")
def test_system_chromium_only_where_pm_builds_none(homes, monkeypatch):
    bindir = homes / "bin"
    bindir.mkdir()
    exe = bindir / "chromium-browser"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    # A target pm builds Chromium for: a system chromium on PATH is not picked up.
    monkeypatch.setattr(pm, "current_target", lambda: "linux-x64")
    assert browser_runtime.chromium_executable() is None
    assert browser_runtime.chromium_missing_hint() == "Chromium is not installed: run `hermes pm install chromium`."
    # Termux (linux-arm64-bionic): pm builds no Chromium, the system one is used.
    monkeypatch.setattr(pm, "current_target", lambda: "linux-arm64-bionic")
    assert browser_runtime.chromium_executable() == str(exe)


def test_termux_hint_never_mentions_npm(homes, monkeypatch):
    from hermes_platform.host import runtime

    monkeypatch.setattr(runtime, "is_termux", lambda: True)
    hint = browser_runtime.chromium_missing_hint()
    assert "pkg install" in hint and "npm" not in hint and "npx" not in hint
