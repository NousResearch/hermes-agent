"""Debug-browser launch must honor the existing automation executable override."""

import platform
import shlex

import pytest

from hermes_cli import browser_connect as bc


@pytest.mark.platforms("posix")
def test_override_is_first_deduplicated_and_used_by_manual_command(tmp_path, monkeypatch):
    browser = tmp_path / "custom browser"
    browser.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    browser.chmod(0o755)
    fallback = tmp_path / "fallback"
    fallback.write_text("browser", encoding="utf-8")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(browser))
    monkeypatch.setattr(bc, "_debug_candidate_paths", lambda system: iter([str(fallback), str(browser)]))

    assert bc.get_chrome_debug_candidates(platform.system()) == [str(browser), str(fallback)]
    command = bc.manual_chrome_debug_command(9222)
    assert shlex.split(command)[0] == str(browser)
    assert "--remote-debugging-port=9222" in shlex.split(command)
    assert any(arg.startswith("--user-data-dir=") for arg in shlex.split(command))


@pytest.mark.platforms("posix")
def test_unusable_override_keeps_detected_candidates(tmp_path, monkeypatch):
    fallback = tmp_path / "fallback"
    fallback.write_text("browser", encoding="utf-8")
    monkeypatch.setattr(bc, "_debug_candidate_paths", lambda system: iter([str(fallback)]))
    for override in ("", str(tmp_path / "missing"), str(tmp_path)):
        monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", override)
        assert bc.get_chrome_debug_candidates(platform.system()) == [str(fallback)]
