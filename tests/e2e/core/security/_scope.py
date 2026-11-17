"""Private harness for ``test_profile_secret_scope.py``: one real ``hermes serve`` hosting two
profiles, a user dashboard plugin whose API route reads a credential through ``get_secret``, and a
stdio MCP server per profile that dumps the environment it was spawned with.

Everything here writes plain files into the fake home; no Hermes code is imported or patched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ID = "scopecanary"
SECRET_ENV = "SCOPE_CANARY_SECRET"  # same NAME in every profile's .env, distinct VALUE
MCP_ENV_KEY = "SCOPE_MCP_OWN_SECRET"  # the one variable a profile deliberately hands its MCP server
MCP_SERVER_NAME = "envdump"

# The plugin is third-party code: it only uses the documented credential API and folds every
# failure into its reply (the plugin "no data" contract), so the test sees exactly what it saw.
_PLUGIN_API = f'''
from fastapi import APIRouter

router = APIRouter()


@router.get("/secret")
def secret() -> dict:
    try:
        from agent.secret_scope import get_secret
        return {{"ok": True, "value": get_secret("{SECRET_ENV}")}}
    except Exception as exc:  # plugin contract: never raise
        return {{"ok": False, "error": type(exc).__name__}}
'''

# A stdio MCP server that records os.environ at spawn into the file named by argv[1], then serves
# one trivial tool so the client's handshake completes like any real server's.
_ENV_DUMP_SERVER = '''
import json, os, sys

out = sys.argv[1]
tmp = out + ".part"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(dict(os.environ), fh)
os.replace(tmp, out)

from mcp.server import MCPServer

server = MCPServer("envdump")


@server.tool()
def envdump_ping(nonce: str = "") -> str:
    """Echo the nonce."""
    return nonce


server.run(transport="stdio")
'''


def install_plugin(hermes_home: Path) -> None:
    """A user plugin with a backend API under the launch home's plugins dir."""
    dash = hermes_home / "plugins" / PLUGIN_ID / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    (hermes_home / "plugins" / PLUGIN_ID / "plugin.yaml").write_text(
        f"name: {PLUGIN_ID}\nversion: 0.0.1\ndescription: secret-scope canary\n", encoding="utf-8")
    (dash / "manifest.json").write_text(json.dumps({
        "name": PLUGIN_ID, "label": "Scope canary", "version": "0.0.1",
        "tab": {"path": f"/{PLUGIN_ID}", "hidden": True}, "entry": "dist/index.js", "api": "plugin_api.py",
    }), encoding="utf-8")
    (dash / "dist").mkdir(exist_ok=True)
    (dash / "dist" / "index.js").write_text("// no UI\n", encoding="utf-8")
    (dash / "plugin_api.py").write_text(_PLUGIN_API, encoding="utf-8")


def write_env_dump_server(path: Path) -> Path:
    path.write_text(_ENV_DUMP_SERVER, encoding="utf-8")
    return path


def mcp_config(server_script: Path, dump_file: Path) -> dict:
    """``mcp_servers`` entry: the profile passes exactly one secret, by reference to its own .env."""
    return {MCP_SERVER_NAME: {
        "command": sys.executable,
        "args": [str(server_script), str(dump_file)],
        "env": {MCP_ENV_KEY: "${" + SECRET_ENV + "}"},
    }}
