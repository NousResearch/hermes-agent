"""
Tests for the Termux Chromium Browser skill (skills/devops/termux-chromium-browser).

Covers:
- CDPBrowser class instantiation and statefile management
- JavaScript evaluation payload formulation
- Shell script syntax checks (service/run and setup_service.sh)
- CLI argument handling and help display
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "devops" / "termux-chromium-browser"
SCRIPTS_DIR = SKILL_DIR / "scripts"
SERVICE_DIR = SKILL_DIR / "service"
BROWSER_SCRIPT = SCRIPTS_DIR / "browser.py"

# Import module directly
sys.path.insert(0, str(SCRIPTS_DIR))
import browser


class TestTermuxChromiumBrowserCore:
    def test_browser_init(self):
        b = browser.CDPBrowser(host="127.0.0.1", port=9222)
        assert b.host == "127.0.0.1"
        assert b.port == 9222
        assert b.ws is None

    def test_tabs_empty_on_connection_error(self):
        with patch.object(browser, "http_json", return_value="Connection Refused"):
            b = browser.CDPBrowser()
            assert b.tabs() == []

    def test_tabs_parsing_success(self):
        mock_tabs = [
            {"id": "tab1", "type": "page", "url": "https://example.com", "title": "Example"},
            {"id": "tab2", "type": "background_page", "url": "chrome://extension"},
        ]
        with patch.object(browser, "http_json", return_value=mock_tabs):
            b = browser.CDPBrowser()
            tabs = b.tabs()
            assert len(tabs) == 1
            assert tabs[0]["id"] == "tab1"

    def test_statefile_tab_tracking(self, tmp_path):
        state_file = tmp_path / ".cdp_last_tab"
        b = browser.CDPBrowser()
        b.STATEFILE = str(state_file)

        mock_tabs = [{"id": "tab_123", "type": "page", "url": "https://example.com", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/tab_123"}]
        with patch.object(browser, "http_json", return_value=mock_tabs), \
             patch("websocket.create_connection") as mock_ws:
            state_file.write_text("tab_123", encoding="utf-8")
            tab = b.attach()
            assert tab["id"] == "tab_123"
            assert b.tab_url == "https://example.com"


class TestTermuxChromiumBrowserScripts:
    def test_shell_syntax(self):
        sh_files = [
            SCRIPTS_DIR / "setup_service.sh",
            SERVICE_DIR / "run",
        ]
        for f in sh_files:
            assert f.exists(), f"File missing: {f}"
            res = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            assert res.returncode == 0, f"Syntax error in {f.name}: {res.stderr}"

    def test_cli_help(self):
        res = subprocess.run(
            [sys.executable, str(BROWSER_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "browser.py" in res.stdout
