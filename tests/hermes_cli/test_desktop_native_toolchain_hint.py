"""Regression tests for the Linux native toolchain hint (#102081).

``hermes desktop`` installs the workspace's Node dependencies, and on Linux
node-pty has no prebuilds — its install script always compiles locally and
needs ``make`` plus a C++ compiler on PATH. When the compiler is missing the
npm install fails with a wall of log text and the generic "run npm ci
manually" advice cannot help, so the CLI must call out the missing toolchain
explicitly.
"""

import hermes_cli.main as main_mod


def test_hint_none_off_linux(monkeypatch):
    monkeypatch.setattr(main_mod.sys, "platform", "darwin")
    monkeypatch.setattr(main_mod.shutil, "which", lambda tool: None)
    assert main_mod._linux_native_build_toolchain_hint() is None


def test_hint_when_whole_toolchain_missing(monkeypatch):
    monkeypatch.setattr(main_mod.sys, "platform", "linux")
    monkeypatch.setattr(main_mod.shutil, "which", lambda tool: None)
    hint = main_mod._linux_native_build_toolchain_hint()
    assert hint is not None
    assert "missing from PATH: make, g++" in hint
    assert "build-essential" in hint


def test_hint_lists_only_the_missing_tools(monkeypatch):
    monkeypatch.setattr(main_mod.sys, "platform", "linux")

    def fake_which(tool: str):
        return "/usr/bin/make" if tool == "make" else None

    monkeypatch.setattr(main_mod.shutil, "which", fake_which)
    hint = main_mod._linux_native_build_toolchain_hint()
    assert hint is not None
    assert "missing from PATH: g++" in hint


def test_hint_none_when_toolchain_present(monkeypatch):
    monkeypatch.setattr(main_mod.sys, "platform", "linux")
    monkeypatch.setattr(main_mod.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    assert main_mod._linux_native_build_toolchain_hint() is None
