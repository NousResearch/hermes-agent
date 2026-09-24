"""Tests for ``_install_chromium`` post-setup verification (PR #30161).

The installer can exit 0 while leaving a build the browser gate can't see
(interrupted download, unexpected layout). A success message that
``check_browser_requirements()`` then contradicts is worse than a warning —
the post-setup hook must re-run the gate and warn loudly when the fresh
install is still undetectable (curtis921's report on PR #30161).
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    import tools.browser_tool as _bt

    _bt._cached_chromium_installed = None
    yield
    _bt._cached_chromium_installed = None


def _run_install_chromium(monkeypatch, *, gate_detects: bool, capsys) -> None:
    from hermes_cli import tools_config_post_setup as ps

    monkeypatch.setattr(ps, "_run_text", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: gate_detects)
    ps._install_chromium(["agent-browser", "install", "--with-deps"])
    return capsys.readouterr().out


class TestInstallChromiumVerifiesGate:
    def test_success_message_when_gate_detects_fresh_install(self, monkeypatch, capsys):
        out = _run_install_chromium(monkeypatch, gate_detects=True, capsys=capsys)
        assert "Chromium installed" in out
        assert "can't detect" not in out

    def test_warns_when_gate_cannot_detect_fresh_install(self, monkeypatch, capsys):
        """exit 0 but the gate still says missing — must warn, not claim success."""
        out = _run_install_chromium(monkeypatch, gate_detects=False, capsys=capsys)
        assert "can't detect" in out
        assert "AGENT_BROWSER_EXECUTABLE_PATH" in out
        assert "Chromium installed\n" not in out

    def test_cache_invalidated_before_gate_recheck(self, monkeypatch, capsys):
        """The stale cached 'missing' flag must be cleared before the re-check."""
        import tools.browser_tool as _bt
        from hermes_cli import tools_config_post_setup as ps

        _bt._cached_chromium_installed = False
        seen = {}

        def fake_gate():
            seen["cache_was"] = _bt._cached_chromium_installed
            return True

        monkeypatch.setattr(ps, "_run_text", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
        monkeypatch.setattr("tools.browser_tool_install._chromium_installed", fake_gate)
        ps._install_chromium(["agent-browser", "install", "--with-deps"])
        assert seen["cache_was"] is None

    def test_nonzero_exit_still_reports_failure(self, monkeypatch, capsys):
        from hermes_cli import tools_config_post_setup as ps

        monkeypatch.setattr(ps, "_run_text", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
        ps._install_chromium(["agent-browser", "install", "--with-deps"])
        out = capsys.readouterr().out
        assert "install failed" in out
