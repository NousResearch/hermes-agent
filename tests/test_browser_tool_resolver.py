"""Tests for cross-platform agent-browser executable resolution."""

from pathlib import Path

import pytest

try:
    import tools.browser_tool as browser_tool
except ModuleNotFoundError as exc:
    pytest.skip(f"Optional dependency missing for browser_tool import: {exc}", allow_module_level=True)


def _set_fake_repo_layout(monkeypatch, tmp_path: Path) -> Path:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    fake_file = tools_dir / "browser_tool.py"
    fake_file.write_text("# test placeholder", encoding="utf-8")
    monkeypatch.setattr(browser_tool, "__file__", str(fake_file))
    return tmp_path / "node_modules" / ".bin"


def test_find_agent_browser_windows_prefers_local_cmd(monkeypatch, tmp_path: Path):
    bin_dir = _set_fake_repo_layout(monkeypatch, tmp_path)
    bin_dir.mkdir(parents=True, exist_ok=True)
    cmd_path = bin_dir / "agent-browser.cmd"
    cmd_path.write_text("@echo off", encoding="utf-8")
    (bin_dir / "agent-browser").write_text("#!/bin/sh", encoding="utf-8")

    monkeypatch.setattr(browser_tool.os, "name", "nt", raising=False)
    monkeypatch.setattr(browser_tool.shutil, "which", lambda _name: None)

    assert browser_tool._find_agent_browser() == [str(cmd_path)]


def test_find_agent_browser_windows_uses_npx_cmd_fallback(monkeypatch, tmp_path: Path):
    _set_fake_repo_layout(monkeypatch, tmp_path)

    npx_cmd = r"C:\Program Files\nodejs\npx.cmd"

    def _fake_which(name: str):
        if name == "npx.cmd":
            return npx_cmd
        return None

    monkeypatch.setattr(browser_tool.os, "name", "nt", raising=False)
    monkeypatch.setattr(browser_tool.shutil, "which", _fake_which)

    assert browser_tool._find_agent_browser() == [npx_cmd, "agent-browser"]


def test_find_agent_browser_posix_prefers_path_binary(monkeypatch, tmp_path: Path):
    _set_fake_repo_layout(monkeypatch, tmp_path)

    def _fake_which(name: str):
        if name == "agent-browser":
            return "/usr/local/bin/agent-browser"
        return None

    monkeypatch.setattr(browser_tool.os, "name", "posix", raising=False)
    monkeypatch.setattr(browser_tool.shutil, "which", _fake_which)

    assert browser_tool._find_agent_browser() == ["/usr/local/bin/agent-browser"]
