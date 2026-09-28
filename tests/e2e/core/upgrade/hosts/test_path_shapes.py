"""Install and run on hosts whose PATH and shell startup files are not a fresh skeleton.

Failure class: PATH shapes. Two things users report once the install "works":

* the installer's PATH wiring duplicates ``~/.local/bin`` when the distro's own startup files
  already put it on PATH with a bare ``PATH=...`` assignment (Fedora ``.bashrc``, Debian
  ``.profile``), so a login shell carries it more than once (gated on #123424); a re-run of the
  installer must not edit the startup files again;
* a node/npm the user already has, earlier on PATH, either shadows the managed toolchain where
  Hermes needs the managed one (the TUI/web builds), or the managed node is forced onto an MCP
  server the user configured with their own ``node`` (gated on #124264).

One real install through HEAD's ``scripts/install.sh`` into a HOME carrying Fedora's stock
``.bashrc`` / ``.bash_profile``, with a user-owned ``node``/``npm`` first on PATH. That node
is a stand-in (the external edge): it answers version probes as v22 and logs every call; anything
else it refuses loudly, except the user's own MCP server script, which it serves as a minimal
stdio MCP server. The managed node running the same script instead exits the way a native addon
built for the user's node does (``NODE_MODULE_VERSION`` mismatch) and records which node ran it.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.hosts import _hosts as X
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

# Fedora 44 /etc/skel, verbatim in the lines that matter: .bashrc puts ~/.local/bin on PATH with
# a bare assignment (guarded against re-sourcing), .bash_profile sources .bashrc.
FEDORA_SKEL = {
    ".bashrc": (
        "# .bashrc\n\n"
        "# Source global definitions\n"
        "if [ -f /etc/bashrc ]; then\n    . /etc/bashrc\nfi\n\n"
        "# User specific environment\n"
        "if ! [[ \"$PATH\" =~ \"$HOME/.local/bin:$HOME/bin:\" ]]; then\n"
        "    PATH=\"$HOME/.local/bin:$HOME/bin:$PATH\"\n"
        "fi\nexport PATH\n"
    ),
    ".bash_profile": (
        "# .bash_profile\n\n"
        "# Get the aliases and functions\n"
        "if [ -f ~/.bashrc ]; then\n    . ~/.bashrc\nfi\n\n"
        "# User specific environment and startup programs\n"
    ),
}
USER_NODE_VERSION = "v22.11.0"
MINIMAL_MCP_SERVER = r'''
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        continue
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "user-node-server", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "whoami", "description": "which node runs me",
                             "inputSchema": {"type": "object", "properties": {}}}]}
    else:
        result = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
    sys.stdout.flush()
'''


def _user_toolchain(root: Path) -> dict[str, Path]:
    """The user's own node/npm (first on PATH) plus their MCP server script."""
    sysbin = root / "user-node" / "bin"
    sysbin.mkdir(parents=True)
    calls = root / "user-node" / "calls.log"
    mcp_dir = root / "user-mcp"
    mcp_dir.mkdir()
    marker = mcp_dir / "ran-with.txt"
    server_py = mcp_dir / "server.py"
    server_py.write_text(MINIMAL_MCP_SERVER, encoding="utf-8")
    server_js = mcp_dir / "server.js"
    server_js.write_text(
        "require('fs').writeFileSync(" + repr(str(marker)) + ", process.execPath + ' ' + process.version);\n"
        "console.error(\"Error: The module 'better_sqlite3.node' was compiled against a different Node.js version "
        "using NODE_MODULE_VERSION 127. This version of Node.js requires NODE_MODULE_VERSION \" + process.versions.modules);\n"
        "process.exit(1);\n", encoding="utf-8")
    python = shutil.which("python3") or sys.executable
    versions = {"node": USER_NODE_VERSION, "npm": "10.9.0", "npx": "10.9.0"}
    for name, ver in versions.items():
        serve = f'  *server.js) exec "{python}" "{server_py}" ;;\n' if name == "node" else ""
        p = sysbin / name
        p.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "{name} $*" >> "{calls}"\n'
            'case "$1" in\n'
            f"  --version|-v) echo {ver}; exit 0 ;;\n"
            f"{serve}"
            "esac\n"
            f'echo "user {name} {ver}: refusing to run \'$*\' (Hermes must use its managed toolchain here)" >&2\n'
            "exit 1\n", encoding="utf-8")
        p.chmod(0o755)
    return {"bin": sysbin, "calls": calls, "server_js": server_js, "marker": marker}


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the PATH-shapes suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def world(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("path-shapes")
    origin = X.make_origin(root)
    sb = X.new_sandbox(root / "sb", origin)
    for name, text in FEDORA_SKEL.items():
        (sb.home / name).write_text(text, encoding="utf-8")
    user = _user_toolchain(root)
    # The user's node sits ahead of every system directory, as nvm/volta/distro packages do.
    parts = sb.env["PATH"].split(":")
    at = parts.index(str(sb.root / "wrap")) + 1
    sb.env["PATH"] = ":".join([*parts[:at], str(user["bin"]), *parts[at:]])
    first = I.run_installer(sb)
    return {"sb": sb, "user": user, "install": first,
            "rc_after_install": {n: (sb.home / n).read_text(encoding="utf-8") for n in FEDORA_SKEL}}


def _local_bin_count(sb: I.Sandbox) -> tuple[int, str]:
    probe = X.login_shell(sb, 'printf "PATH=%s\\n" "$PATH"')
    assert probe.returncode == 0, H.describe(probe)
    line = next((ln for ln in probe.stdout.splitlines() if ln.startswith("PATH=")), "")
    entries = line[len("PATH="):].split(":")
    return sum(1 for e in entries if e.rstrip("/") == str(sb.home / ".local" / "bin")), line


def _calls(world) -> list[str]:
    f = world["user"]["calls"]
    return f.read_text(encoding="utf-8").splitlines() if f.exists() else []


def test_user_node_first_on_path_does_not_shadow_the_managed_toolchain(world, provider):
    sb, first = world["sb"], world["install"]
    # The shim log first: when the user's node is picked up the build usually dies on it, and the
    # log names that root cause where the exit code alone would not.
    used = [c for c in _calls(world) if not re.fullmatch(r"(node|npm|npx) (--version|-v)", c)]
    assert not used, ("the installer ran the user's node/npm for real work instead of the managed toolchain:\n"
                      + "\n".join(used) + "\n" + I.describe(first))
    assert first.returncode == 0, "install.sh failed with the user's own node first on PATH:\n" + I.describe(first)
    X.ok(sb.cli("--version"))
    X.configure(sb, provider)
    X.turn(sb, provider, "turn with the user's node first on PATH")


def test_login_shell_has_local_bin_once_with_stock_fedora_startup_files(world):
    sb, first = world["sb"], world["install"]
    assert first.returncode == 0, I.describe(first)
    count, line = _local_bin_count(sb)
    assert count >= 1, f"a new login shell does not have ~/.local/bin on PATH at all: {line}"
    rerun = I.run_installer(sb)
    assert rerun.returncode == 0, "re-running install.sh failed:\n" + I.describe(rerun)
    rc_after_rerun = {n: (sb.home / n).read_text(encoding="utf-8") for n in FEDORA_SKEL}
    count_rerun, line_rerun = _local_bin_count(sb)
    assert rc_after_rerun == world["rc_after_install"], (
        "re-running the installer edited the startup files again: "
        f"{[n for n in FEDORA_SKEL if rc_after_rerun[n] != world['rc_after_install'][n]]}")
    with known_failure(r"~/\.local/bin appears [2-9]\d* times",
                       "gated on #123424: wire_shell_path's existing-setup regex misses bare PATH= assignments"):
        changed = [n for n, text in FEDORA_SKEL.items() if world["rc_after_install"][n] != text]
        assert count == 1 and count_rerun == 1, (
            f"~/.local/bin appears {count} times on a login shell's PATH after install ({count_rerun} after a re-run):\n"
            f"{line_rerun}\n(startup files the installer changed: {changed})")


def test_mcp_server_configured_with_the_users_node_runs_on_the_users_node(world, provider):
    sb, user = world["sb"], world["user"]
    assert world["install"].returncode == 0, I.describe(world["install"])
    X.configure(sb, provider)
    cfg = sb.hermes_home / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8")
                   + f"mcp_servers:\n  usernode:\n    command: node\n    args: [\"{user['server_js']}\"]\n",
                   encoding="utf-8")
    before = len(_calls(world))
    probe = sb.cli("mcp", "test", "usernode", timeout=180)
    ran_js = [c for c in _calls(world)[before:] if c.startswith("node ") and c.endswith("server.js")]
    managed = user["marker"].read_text(encoding="utf-8") if user["marker"].exists() else ""
    assert I.TRACEBACK not in probe.stdout + probe.stderr, I.describe(probe)
    assert ran_js or managed, "harness: neither the user's node nor any other node ran the MCP server\n" + I.describe(probe)
    with known_failure(r"MCP server configured with `command: node` ran on .*not the user's node",
                       "gated on #124264: the managed node is prepended to MCP stdio children's PATH"):
        assert not managed and ran_js, (
            f"MCP server configured with `command: node` ran on {managed!r}, not the user's node "
            f"({user['bin'] / 'node'}, {USER_NODE_VERSION}):\n" + I.describe(probe))
    assert probe.returncode == 0, "`hermes mcp test` failed although the user's node served the MCP server:\n" + I.describe(probe)
