"""Regression: install.ps1 must honor browser.browsers_path on Chromium downloads.

Windows is where this matters most. Playwright's default cache lives under
``%LOCALAPPDATA%`` on the system drive, and the Chromium + headless-shell +
ffmpeg download is ~500MB, so moving it elsewhere is the entire purpose of the
``browser.browsers_path`` config key. ``tools/browser_tool.py`` reads that key
at runtime (``_configured_browsers_path``, consulted by
``_chromium_search_roots``); an installer that ignores it downloads to C: while
the runtime searches the configured directory and finds nothing.

These tests are source-level because CI cannot execute the PowerShell installer
-- the same constraint documented in test_install_ps1_web_server_syntax_probe.py.
The behavioural coverage for the equivalent POSIX logic is executable and lives
in test_install_sh_browser_install.py.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _text() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_helper_exists_and_reads_the_config_key() -> None:
    text = _text()

    assert "function Set-ConfiguredBrowsersPathEnv {" in text
    # It resolves the value through the installed venv's Python rather than
    # hand-parsing YAML in PowerShell.
    assert 'Join-Path $InstallDir "venv\\Scripts\\python.exe"' in text
    assert 'Join-Path $HermesHome "config.yaml"' in text
    assert 'browser.get("browsers_path")' in text
    assert "$env:PLAYWRIGHT_BROWSERS_PATH = $configured" in text


def test_ambient_env_var_wins_over_config() -> None:
    """An existing PLAYWRIGHT_BROWSERS_PATH is never relocated.

    Same precedence as the runtime's _browsers_path_env_overrides(): the Docker
    image sets this var, and an operator who exports it means it.
    """
    text = _text()
    fn = re.search(
        r"function Set-ConfiguredBrowsersPathEnv \{(.*?)\n\}", text, re.DOTALL
    )
    assert fn, "Set-ConfiguredBrowsersPathEnv not found"
    body = fn.group(1)
    assert "if ($env:PLAYWRIGHT_BROWSERS_PATH) { return }" in body, (
        "the ambient env var must short-circuit before the config lookup"
    )


def test_missing_venv_or_config_is_a_silent_no_op() -> None:
    text = _text()
    fn = re.search(
        r"function Set-ConfiguredBrowsersPathEnv \{(.*?)\n\}", text, re.DOTALL
    )
    assert fn
    body = fn.group(1)
    assert "if (-not (Test-Path $venvPython) -or -not (Test-Path $configPath)) { return }" in body
    # The Python side swallows its own failures too, so a malformed config.yaml
    # cannot abort an install.
    assert "except Exception:" in body
    assert "sys.exit(0)" in body


def test_both_chromium_download_sites_set_the_path() -> None:
    """install.ps1 downloads browsers from two places; both need the export."""
    text = _text()

    ab = text.index("Installing Chromium via agent-browser install")
    assert "Set-ConfiguredBrowsersPathEnv" in text[ab : ab + 400], (
        "the agent-browser install path must set PLAYWRIGHT_BROWSERS_PATH"
    )

    pw = text.index("playwright install chromium 2>&1 | ForEach-Object")
    assert "Set-ConfiguredBrowsersPathEnv" in text[max(0, pw - 400) : pw], (
        "the npx playwright install fallback must set PLAYWRIGHT_BROWSERS_PATH"
    )


def test_helper_is_defined_before_its_call_sites() -> None:
    """PowerShell resolves functions at call time, but keeping the definition
    first matches the file's existing layout and avoids a dot-sourcing trap."""
    text = _text()
    definition = text.index("function Set-ConfiguredBrowsersPathEnv {")
    first_call = text.index(
        "Set-ConfiguredBrowsersPathEnv", text.index("function Install-AgentBrowser {")
    )
    assert definition < first_call
