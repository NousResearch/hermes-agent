"""Regression test for install.ps1 browser setup (PR #44772 review).

agent-browser resolves lazily via npx everywhere else in the system
(tools/browser_tool.py::_find_agent_browser); Install-AgentBrowser was the
last place that still eagerly npm-installed a second, separately
version-pinned copy of it. Removed: agent-browser acquisition now happens
only via `hermes update`'s npx cache warm or an actual browser-tool call's
lazy npx resolution.

Linux CI cannot execute the PowerShell installer, so verification here is
source-text-level only, matching tests/test_install_sh_browser_install.py
and tests/test_install_ps1_ascii_only.py.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _extract_function_body(source: str, name: str) -> str:
    m = re.search(
        rf"^function {re.escape(name)} \{{.*?^\}}", source, re.MULTILINE | re.DOTALL
    )
    assert m, f"could not extract function {name} from install.ps1"
    return m.group(0)


def test_install_agent_browser_no_longer_npm_installs_agent_browser() -> None:
    body = _extract_function_body(INSTALL_PS1.read_text(), "Install-AgentBrowser")

    assert "agent-browser@" not in body
    assert "Installing Chromium via agent-browser install" not in body
    # camofox is unrelated to this change and must still be installed here.
    assert "@askjo/camofox-browser@^1.5.2" in body
    # System-browser detection is still cheap/valuable without agent-browser.
    assert "Find-SystemBrowser" in body
    assert "Write-BrowserEnv" in body


def test_install_agent_browser_drops_unused_skip_chromium_param() -> None:
    """$SkipChromium only existed to gate the now-removed Chromium-download
    branch; grep confirms it's never passed at the one real call site, so a
    lingering param would be dead code implying a control path that no
    longer exists."""
    body = _extract_function_body(INSTALL_PS1.read_text(), "Install-AgentBrowser")

    assert "SkipChromium" not in body

    # And the one real call site must not pass it either.
    text = INSTALL_PS1.read_text()
    call_site = re.search(r"^\s*Install-AgentBrowser.*$", text, re.MULTILINE)
    assert call_site, "could not find Install-AgentBrowser call site"
    assert "SkipChromium" not in call_site.group(0)


def test_install_agent_browser_still_ignore_scripts_hardened() -> None:
    """The removal of agent-browser must not have also dropped the
    supply-chain hardening that still applies to the remaining camofox
    install."""
    body = _extract_function_body(INSTALL_PS1.read_text(), "Install-AgentBrowser")

    assert "--ignore-scripts" in body


def test_install_agent_browser_no_longer_references_agent_browser_cmd_shim() -> None:
    """No dangling reference to the agent-browser.cmd shim should remain
    now that this function never installs it."""
    body = _extract_function_body(INSTALL_PS1.read_text(), "Install-AgentBrowser")

    assert "agent-browser.cmd" not in body
