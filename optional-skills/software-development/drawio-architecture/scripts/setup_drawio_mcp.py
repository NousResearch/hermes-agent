#!/usr/bin/env python3
"""setup_drawio_mcp.py - Configure the official draw.io MCP server (npx @drawio/mcp)
for the current agent/IDE, and generate browser-fallback viewer URLs for .drawio files.

Usage:
  python3 setup_drawio_mcp.py --detect
  python3 setup_drawio_mcp.py --target claude-code
  python3 setup_drawio_mcp.py --target vscode --global
  python3 setup_drawio_mcp.py --target cursor --force
  python3 setup_drawio_mcp.py --viewer-url diagram.drawio
  python3 setup_drawio_mcp.py --viewer-url diagram.drawio --edit
"""
import argparse
import base64
import json
import os
import sys
import zlib

MCP_ENTRY = {"command": "npx", "args": ["-y", "@drawio/mcp"]}
SERVER_KEY = "drawio"

# Map a target to (config_path, json_key)
# json_key: "mcpServers" (claude/cursor/windsurf/opencode/devin/agy/gemini) or "servers" (vscode)
# note: printed as a caveat when the canonical setup is via the agent's UI rather than a file.


def _home():
    return os.path.expanduser("~")


def _appdata(sub):
    return os.path.join(os.environ.get("APPDATA", f"{_home()}/AppData/Roaming"), sub)


def targets():
    h = _home()
    return {
        "claude-desktop": {
            "key": "mcpServers",
            "paths": {
                "darwin": f"{h}/Library/Application Support/Claude/claude_desktop_config.json",
                "win32": _appdata("Claude/claude_desktop_config.json"),
                "linux": f"{h}/.config/Claude/claude_desktop_config.json",
            },
        },
        "claude-code": {
            "key": "mcpServers",
            "paths": {"*": f"{h}/.claude/settings.json"},
        },
        "vscode": {
            "key": "servers",
            "paths": {
                "darwin": f"{h}/Library/Application Support/Code/User/mcp.json",
                "win32": _appdata("Code/User/mcp.json"),
                "linux": f"{h}/.config/Code/User/mcp.json",
                "*": os.path.join(os.getcwd(), ".vscode", "mcp.json"),
            },
        },
        "cursor": {
            "key": "mcpServers",
            "paths": {"*": f"{h}/.cursor/mcp.json"},
        },
        "opencode": {
            "key": "mcpServers",
            "paths": {"*": f"{h}/.config/opencode/opencode.json"},
        },
        "windsurf": {
            "key": "mcpServers",
            "paths": {"*": f"{h}/.codeium/windsurf/mcp.json"},
        },
        "devin": {
            "key": "mcpServers",
            "note": "Devin Desktop typically configures MCP via its app/UI (Integrations). If your build reads a file, the conventional path is shown.",
            "paths": {
                "darwin": _appdata("Devin/mcp.json"),
                "win32": _appdata("Devin/mcp.json"),
                "linux": f"{h}/.devin/mcp.json",
            },
        },
        "devin-cli": {
            "key": "mcpServers",
            "note": "Devin CLI stores config under ~/.config/devin. If your build reads a file, the conventional path is shown; otherwise use the Devin UI.",
            "paths": {"*": f"{h}/.config/devin/mcp.json"},
        },
        "agy": {
            "key": "mcpServers",
            "note": "AGY (Antigravity CLI) config lives under ~/.gemini/antigravity-cli. Also configurable via the Antigravity UI.",
            "paths": {"*": f"{h}/.gemini/antigravity-cli/mcp.json"},
        },
        "antigravity": {
            "key": "mcpServers",
            "note": "Antigravity IDE reuses the Gemini CLI settings file.",
            "paths": {"*": f"{h}/.gemini/settings.json"},
        },
        "gemini": {
            "key": "mcpServers",
            "paths": {"*": f"{h}/.gemini/settings.json"},
        },
    }


def config_path(target, plat):
    t = targets()[target]
    paths = t["paths"]
    return paths.get(plat, paths.get("*"))


def detect():
    h = _home()
    guesses = []
    if os.path.isdir(f"{h}/.cursor"):
        guesses.append("cursor")
    if os.path.isdir(f"{h}/.claude"):
        guesses.append("claude-code")
    if os.path.isdir(f"{h}/.config/Code") or os.path.isdir(f"{h}/Library/Application Support/Code"):
        guesses.append("vscode")
    if os.path.isdir(f"{h}/.config/opencode"):
        guesses.append("opencode")
    if os.path.isdir(f"{h}/.codeium/windsurf"):
        guesses.append("windsurf")
    if os.path.isdir(f"{h}/Library/Application Support/Claude") or os.environ.get("APPDATA"):
        guesses.append("claude-desktop")
    if os.path.isdir(f"{h}/.devin") or os.path.isdir(f"{h}/.config/devin"):
        guesses.append("devin")
    if os.path.isdir(f"{h}/.gemini/antigravity-cli"):
        guesses.append("agy")
    if os.path.isdir(f"{h}/.gemini"):
        guesses.append("gemini")
    return guesses


def write_config(target, dry_run, force):
    plat = sys.platform
    path = config_path(target, plat)
    t = targets()[target]
    key = t["key"]
    if path is None:
        print(f"[!] No known config path for {target} on {plat}", file=sys.stderr)
        return 1
    if t.get("note"):
        print(f"[!] Note: {t['note']}")
    print(f"[*] Target: {target}  config: {path}  key: {key}")

    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Existing config is not valid JSON: {path}", file=sys.stderr)
            return 1
    else:
        print(f"[*] Config does not exist yet; will create it.")

    servers = existing.get(key, {})
    if SERVER_KEY in servers and not force:
        print(f"[=] '{SERVER_KEY}' already present in {key}. Use --force to overwrite.")
        print(json.dumps(existing, indent=2))
        return 0
    servers[SERVER_KEY] = MCP_ENTRY
    existing[key] = servers

    if dry_run:
        print("[dry-run] Would write:")
        print(json.dumps(existing, indent=2))
        return 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"[+] Wrote MCP server '{SERVER_KEY}' to {path}")
    print(f"[+] Restart the client, then ask it to draw a diagram.")
    return 0


def _deflate_b64(text):
    raw = text.encode("utf-8")
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    comp = co.compress(raw) + co.flush()
    return base64.b64encode(comp).decode("ascii")


def viewer_url(path, editable):
    if not os.path.exists(path):
        print(f"[!] File not found: {path}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as f:
        xml = f.read()
    # encodeURIComponent -> deflateRaw -> base64 (diagrams.net URL convention)
    import urllib.parse
    payload = urllib.parse.quote(xml, safe="-_.!~*'()")
    b64 = _deflate_b64(payload)
    if editable:
        url = "https://app.diagrams.net/#create=" + b64
    else:
        url = "https://viewer.diagrams.net/#R" + b64
    print(url)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Configure draw.io MCP server / generate viewer URLs")
    ap.add_argument("--detect", action="store_true", help="list likely MCP targets for this machine")
    ap.add_argument("--target", help="explicit target: " + ", ".join(targets().keys()) + ", raw")
    ap.add_argument("--global", dest="global_", action="store_true", help="(vscode) write user config instead of .vscode/mcp.json")
    ap.add_argument("--dry-run", action="store_true", help="print the config instead of writing it")
    ap.add_argument("--force", action="store_true", help="overwrite an existing drawio entry")
    ap.add_argument("--viewer-url", metavar="FILE", help="print a browser-fallback URL for a .drawio file")
    ap.add_argument("--edit", action="store_true", help="with --viewer-url, produce an editable editor URL")
    args = ap.parse_args()

    if args.detect:
        g = detect()
        print("Likely targets:" if g else "No known agent config dirs found.")
        for t in g:
            print(f"  - {t}  ->  {config_path(t, sys.platform)}")
        return 0

    if args.viewer_url:
        return viewer_url(args.viewer_url, args.edit)

    if args.target == "raw":
        print(json.dumps({"mcpServers": {SERVER_KEY: MCP_ENTRY}}, indent=2))
        return 0

    if not args.target:
        ap.error("specify --detect, --target TARGET, or --viewer-url FILE")

    return write_config(args.target, args.dry_run, args.force)


if __name__ == "__main__":
    sys.exit(main())
