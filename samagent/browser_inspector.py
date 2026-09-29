"""Built-in Agent Browser (`@eN` Accessibility Snapshot) + Real Chrome CDP + MCP Bridge (`samagent/browser_inspector.py`).

Unifies the 3 browser automation & testing modes in SamAgent:
1. `builtin_agent_browser` (Default): Built-in `tools/browser_tool.py` style accessibility-tree snapshot
   with `@e1`, `@e2` interactive element refs against the live local dev server (`:3000`).
2. `chrome_cdp` (Real Local Chrome): Connects to the user's real Chrome browser over Chrome DevTools
   Protocol (`tools/browser_cdp_tool.py`, `http://127.0.0.1:9222/json/version`).
3. `chrome_mcp` (MCP Server): Configures `.vscode/mcp.json` and `config.yaml` for `@playwright/mcp`
   and `chrome-devtools-mcp`.
"""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import urllib.request


class _AccessibilitySnapshotParser(HTMLParser):
    """Extracts an `agent-browser` style accessibility tree with `@e1`, `@e2` interactive refs."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: List[Dict[str, Any]] = []
        self._counter = 0
        self._current_tag = ""
        self._current_attrs: Dict[str, str] = {}
        self._text_buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        t = tag.lower()
        if t in ("button", "input", "select", "a", "h1", "h3", "section", "main"):
            self._current_tag = t
            self._current_attrs = attr_map
            self._text_buf = []
            if t in ("input", "section", "main"):
                self._emit_node(t, attr_map, "")

    def handle_data(self, data: str) -> None:
        if self._current_tag:
            self._text_buf.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == self._current_tag and t in ("button", "a", "h1", "h3", "select"):
            text = " ".join(x for x in self._text_buf if x).strip()
            self._emit_node(t, self._current_attrs, text)
            self._current_tag = ""
            self._text_buf = []

    def _emit_node(self, tag: str, attrs: Dict[str, str], text: str) -> None:
        self._counter += 1
        ref = f"@e{self._counter}"
        role_map = {
            "button": "button",
            "input": "textbox",
            "select": "combobox",
            "a": "link",
            "h1": "heading[1]",
            "h3": "heading[3]",
            "section": "region",
            "main": "main",
        }
        label = (
            text
            or attrs.get("aria-label")
            or attrs.get("placeholder")
            or attrs.get("value")
            or attrs.get("id")
            or tag
        )
        self.elements.append(
            {
                "ref": ref,
                "role": role_map.get(tag, tag),
                "tag": tag,
                "id": attrs.get("id", ""),
                "label": label[:90],
                "interactive": tag in ("button", "input", "select", "a"),
                "action_hint": attrs.get("onclick", ""),
            }
        )


def capture_agent_browser_snapshot(project_dir: Path, dev_port: int = 3000) -> Dict[str, Any]:
    """Capture an `agent-browser` accessibility-tree snapshot (`@eN` refs) from the live dev server or static HTML."""
    root = Path(project_dir).resolve()
    html_source = ""
    source_kind = "static_file"
    live_url = f"http://127.0.0.1:{dev_port}"

    try:
        with urllib.request.urlopen(live_url, timeout=0.8) as resp:
            if resp.status == 200:
                html_source = resp.read().decode("utf-8", errors="replace")
                source_kind = f"live_http ({live_url})"
    except Exception:
        idx = root / "app" / "static" / "index.html"
        if idx.exists():
            html_source = idx.read_text(encoding="utf-8", errors="replace")

    parser = _AccessibilitySnapshotParser()
    if html_source:
        parser.feed(html_source)

    interactive_refs = [e for e in parser.elements if e["interactive"]]
    tree_lines = [
        f"{e['ref']:4s} [{e['role']}] \"{e['label']}\"" + (f" (#{e['id']})" if e["id"] else "")
        for e in parser.elements
    ]

    return {
        "ok": len(parser.elements) > 0,
        "mode": "builtin_agent_browser",
        "source": source_kind,
        "url": live_url,
        "total_nodes": len(parser.elements),
        "interactive_count": len(interactive_refs),
        "elements": parser.elements,
        "snapshot_text": "\n".join(tree_lines),
        "timestamp": time.time(),
    }


def probe_chrome_cdp(cdp_port: int = 9222) -> Dict[str, Any]:
    """Check whether a local Chrome browser is listening with `--remote-debugging-port=9222`."""
    cdp_url = f"http://127.0.0.1:{cdp_port}/json/version"
    try:
        with urllib.request.urlopen(cdp_url, timeout=0.6) as resp:
            if resp.status == 200:
                info = json.loads(resp.read().decode("utf-8"))
                return {
                    "connected": True,
                    "cdp_port": cdp_port,
                    "browser": info.get("Browser", "Chrome"),
                    "ws_endpoint": info.get("webSocketDebuggerUrl", ""),
                }
    except Exception:
        pass
    return {
        "connected": False,
        "cdp_port": cdp_port,
        "launch_commands": {
            "mac": f'/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port={cdp_port}',
            "linux": f"google-chrome --remote-debugging-port={cdp_port}",
            "windows": f'start chrome.exe --remote-debugging-port={cdp_port}',
        },
    }


def configure_chrome_mcp(project_dir: Path) -> Dict[str, Any]:
    """Write `.vscode/mcp.json` so VS Code / Cursor & SamAgent can use `@playwright/mcp` & `chrome-devtools-mcp`."""
    root = Path(project_dir).resolve()
    vscode_dir = root / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    mcp_path = vscode_dir / "mcp.json"
    mcp_config = {
        "servers": {
            "playwright-browser": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"],
            },
            "chrome-devtools": {
                "command": "npx",
                "args": ["-y", "chrome-devtools-mcp@latest", "--browser-url=http://127.0.0.1:9222"],
            },
        }
    }
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "mcp_config_path": str(mcp_path),
        "servers": list(mcp_config["servers"].keys()),
        "config": mcp_config,
    }


def get_browser_capabilities_state(project_dir: Path, dev_port: int = 3000) -> Dict[str, Any]:
    root = Path(project_dir).resolve()
    snapshot = capture_agent_browser_snapshot(root, dev_port=dev_port)
    cdp = probe_chrome_cdp(9222)
    mcp_file = root / ".vscode" / "mcp.json"
    return {
        "active_mode": "chrome_cdp" if cdp["connected"] else "builtin_agent_browser",
        "builtin_agent_browser": {
            "available": True,
            "engine": "tools/browser_tool.py (@eN Accessibility Tree Snapshotter)",
            "snapshot": snapshot,
        },
        "chrome_cdp": {
            "available": True,
            "engine": "tools/browser_cdp_tool.py + tools/browser_tool_real_profile.py",
            "status": cdp,
        },
        "chrome_mcp": {
            "configured": mcp_file.exists(),
            "mcp_config_path": str(mcp_file),
            "servers": ["playwright-browser (@playwright/mcp)", "chrome-devtools (chrome-devtools-mcp)"],
        },
    }
