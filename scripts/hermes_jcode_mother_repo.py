#!/usr/bin/env python3
"""Create a Rust-first Hermes/jcode supertool scaffold."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JCODE = ROOT / ".codex-research" / "jcode"
SCAFFOLD_VERSION = "hermes-jcode-mother-scaffold.v1"

BRIDGE_PLUGIN = ROOT / "plugins" / "jcode_bridge"
JCODE_TOOL_HERMES = ROOT / "bridges" / "jcode-tool-hermes"
JCODE_NATIVE_HERMES_TOOL = ROOT / "bridges" / "jcode-native-hermes-tool"
HERMES_MCP_SERVER = ROOT / "bridges" / "hermes-mcp-server"
CONTRACT_DIR = ROOT / "contracts" / "jcode_bridge" / "v1"
HERMES_SERVICE_CONTRACT_DIR = ROOT / "contracts" / "hermes_service" / "v1"
HERMES_MCP_CONTRACT_DIR = ROOT / "contracts" / "hermes_mcp" / "v1"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "jcode_bridge"
HERMES_SERVICE_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "hermes_service"
HERMES_MCP_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "hermes_mcp"
LATENCY_PROBE = ROOT / "scripts" / "jcode_bridge_latency_probe.py"
NATIVE_TOOL_CHECK = ROOT / "scripts" / "jcode_native_tool_check.py"
PLAN_DOCS = (
    ROOT / "docs" / "plans" / "2026-05-22-hermes-jcode-comparison.md",
    ROOT / "docs" / "plans" / "2026-05-22-hermes-jcode-bridge-implementation.md",
    ROOT / "docs" / "plans" / "2026-05-23-hermes-jcode-mother-repo-blueprint.md",
    ROOT / "docs" / "plans" / "2026-05-23-hermes-jcode-supertool-architecture.md",
    ROOT / "docs" / "plans" / "2026-05-23-hermes-jcode-upstream-sync-report.md",
)


CHECK_SCRIPT = """#!/usr/bin/env python3
\"\"\"Run the portable Hermes/jcode bridge contract check from this scaffold.\"\"\"

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / "bridges" / "hermes-plugin-jcode"
os.environ.setdefault("JCODE_BRIDGE_ROOT", str(ROOT))
MANIFEST = ROOT / "hermes-jcode.manifest.json"
if MANIFEST.exists():
    try:
        hermes_path = json.loads(MANIFEST.read_text(encoding="utf-8")).get(
            "upstreams", {}
        ).get("hermes", {}).get("path")
        if isinstance(hermes_path, str) and Path(hermes_path).is_dir():
            sys.path.insert(0, hermes_path)
    except Exception:
        pass
local_hermes = ROOT / "upstreams" / "hermes"
if local_hermes.is_dir():
    sys.path.insert(0, str(local_hermes))
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from plugins.jcode_bridge.hermes_service import service_contract_report  # noqa: E402
from plugins.jcode_bridge.tools import handle_jcode_contract_check  # noqa: E402


def _mcp_contract_report() -> dict:
    script = ROOT / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--check", "--live"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "error": "failed to parse Hermes MCP contract output",
            "stdout": completed.stdout,
        }
    payload["returncode"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr
    if completed.returncode != 0:
        payload["success"] = False
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jcode-bin", help="Optional path to a jcode executable.")
    parser.add_argument("--cwd", help="Optional working directory for live checks.")
    parser.add_argument("--live", action="store_true", help="Run jcode version check.")
    parser.add_argument(
        "--live-run",
        action="store_true",
        help="With --live, run a single jcode prompt and validate JSON output.",
    )
    parser.add_argument(
        "--live-run-message",
        default="Reply with exactly OK.",
        help="Prompt to use for --live-run.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for live jcode checks.",
    )
    ns = parser.parse_args(argv)
    jcode_report = json.loads(handle_jcode_contract_check({
        "jcode_bin": ns.jcode_bin,
        "cwd": ns.cwd,
        "live": ns.live,
        "live_run": ns.live_run,
        "live_run_message": ns.live_run_message,
        "timeout_seconds": ns.timeout_seconds,
    }))
    service_report = service_contract_report()
    mcp_report = _mcp_contract_report()
    report = {
        "success": (
            bool(jcode_report.get("success"))
            and bool(service_report.get("success"))
            and bool(mcp_report.get("success"))
        ),
        "jcode_bridge": jcode_report,
        "hermes_service": service_report,
        "hermes_mcp": mcp_report,
    }
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


SERVICE_SCRIPT = """#!/usr/bin/env python3
\"\"\"Run the jcode -> Hermes service bridge from this scaffold.\"\"\"

from __future__ import annotations

import os
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / "bridges" / "hermes-plugin-jcode"
os.environ.setdefault("JCODE_BRIDGE_ROOT", str(ROOT))
MANIFEST = ROOT / "hermes-jcode.manifest.json"
if MANIFEST.exists():
    try:
        hermes_path = json.loads(MANIFEST.read_text(encoding="utf-8")).get(
            "upstreams", {}
        ).get("hermes", {}).get("path")
        if isinstance(hermes_path, str) and Path(hermes_path).is_dir():
            sys.path.insert(0, hermes_path)
    except Exception:
        pass
local_hermes = ROOT / "upstreams" / "hermes"
if local_hermes.is_dir():
    sys.path.insert(0, str(local_hermes))
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))

from plugins.jcode_bridge.hermes_service import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
"""


README = """# Hermes/jcode Mother Repo

This scaffold is for a Rust-first Hermes/jcode supertool. jcode is the host
runtime: its TUI, server, tool loop, swarm/session model, browser path, and
low-latency Rust crates are the product shell. Hermes capabilities are imported
as native jcode tools and service modules.

The intended split:

- jcode owns the Rust hot path and primary UX.
- Hermes owns provider-rich research, external integrations, webhooks,
  plugins, cron, memory-provider integrations, and policy/approval gates.
- This repo owns the fusion layer: native jcode tools for Hermes capabilities,
  contracts, routing policy, update gates, and the small Python service host
  needed for Hermes' existing plugin ecosystem.

Start here:

```bash
python3 scripts/check_bridge_contract.py
```

Then wire `bridges/jcode-native-hermes-tool` into upstream jcode's native tool
registry. That is the supertool path: Hermes-backed capabilities appear inside
jcode's normal Rust agent loop rather than as a second agent.

`bridges/hermes-plugin-jcode`, `bridges/jcode-tool-hermes`, and
`bridges/hermes-mcp-server` are compatibility/bootstrap layers. They keep both
upstreams testable while the native jcode-hosted tool surface matures.

The generated contract check covers both directions:

- Hermes -> jcode through `jcode-bridge.v1`
- jcode -> Hermes through `hermes-service.v1`
- jcode -> Hermes MCP transport through `hermes-mcp.v1`

The Rust client scaffold lives at `bridges/jcode-tool-hermes/`. It can call the
generated `scripts/hermes_service_bridge.py stdio` wrapper from jcode-side
experiments or from a future native jcode tool.

The native jcode tool scaffold lives at `bridges/jcode-native-hermes-tool/`. It
implements jcode's `Tool` trait and is the preferred route for the final
supertool, because it keeps the UI, session, tool execution, and latency shape
inside jcode's Rust architecture.

The MCP server scaffold lives at `bridges/hermes-mcp-server/`. It exposes the
same reverse service contract to jcode's existing MCP manager, so no jcode
upstream patch is required for first integration. The generated
`configs/jcode-mcp.hermes.json` file is a ready-to-inspect jcode MCP config
pointing at this scaffold.

Use `scripts/jcode_bridge_latency_probe.py` to measure local bridge overhead
without model or network calls. That probe keeps the speed claim honest: jcode
can remain the Rust hot path while Hermes contributes higher-level autonomy.

Use `scripts/jcode_native_tool_check.py --jcode upstreams/jcode` after pinning
or symlinking jcode into `upstreams/jcode`. That check proves the native Hermes
tool crate still compiles against jcode's Rust tool architecture.
"""


UPSTREAMS_README = """# Upstreams

Keep upstream source trees pinned and replaceable.

Recommended approaches:

- submodules for exact commit pins and minimal merge noise
- subtrees if contributor ergonomics matter more than submodule purity
- external paths during local development, recorded in `hermes-jcode.manifest.json`

Do not put normal bridge edits inside upstream directories. If a bridge needs
an upstream change, prefer a public protocol/API extension first and a tiny
patch queue only when there is no viable public boundary.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _git(cwd: Path, *args: str) -> str | None:
    completed = _run(["git", *args], cwd=cwd)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _repo_state(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    exists = path.is_dir()
    state: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "commit": None,
        "branch": None,
        "dirty": None,
        "status_sample": [],
    }
    if not exists:
        return state
    status = _git(path, "status", "--short") or ""
    status_lines = [line for line in status.splitlines() if line.strip()]
    state.update({
        "commit": _git(path, "rev-parse", "HEAD"),
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status_lines),
        "status_sample": status_lines[:20],
    })
    return state


def build_manifest(hermes: Path, jcode: Path) -> dict[str, Any]:
    """Build the mother-repo manifest without touching the filesystem."""
    return {
        "manifest_version": SCAFFOLD_VERSION,
        "generated_at": _now_iso(),
        "upstreams": {
            "hermes": _repo_state(hermes),
            "jcode": _repo_state(jcode),
        },
        "bridge_contract": {
            "name": "jcode-bridge",
            "version": "jcode-bridge.v1",
            "schema_dir": "contracts/jcode_bridge/v1",
            "fixture_dir": "tests/fixtures/jcode_bridge",
            "hermes_plugin_dir": "bridges/hermes-plugin-jcode",
        },
        "reverse_service_contract": {
            "name": "hermes-service",
            "version": "hermes-service.v1",
            "schema_dir": "contracts/hermes_service/v1",
            "fixture_dir": "tests/fixtures/hermes_service",
            "stdio_command": "python3 scripts/hermes_service_bridge.py stdio",
            "jcode_client_dir": "bridges/jcode-tool-hermes",
            "jcode_native_tool_dir": "bridges/jcode-native-hermes-tool",
            "mcp_server_dir": "bridges/hermes-mcp-server",
            "jcode_mcp_config": "configs/jcode-mcp.hermes.json",
        },
        "supertool_architecture": {
            "primary_runtime": "jcode",
            "native_extension_point": "jcode_tool_core::Tool",
            "hermes_role": "capability host for integrations, providers, memory, webhooks, and policy",
            "bootstrap_layers": [
                "hermes-plugin-jcode",
                "jcode-tool-hermes",
                "hermes-mcp-server",
            ],
        },
        "mcp_bridge_contract": {
            "name": "hermes-mcp",
            "version": "hermes-mcp.v1",
            "schema_dir": "contracts/hermes_mcp/v1",
            "fixture_dir": "tests/fixtures/hermes_mcp",
            "server_dir": "bridges/hermes-mcp-server",
        },
        "routing_policy": {
            "jcode_first": [
                "low-latency local coding",
                "browser/profile workflows",
                "same-repo swarm work",
            ],
            "hermes_first": [
                "webhooks and external messaging",
                "deep research with provider breadth",
                "approval and safety-gated account actions",
            ],
        },
        "update_gate": [
            "update pinned upstream SHA",
            "run Graphify for both upstreams",
            "run scripts/check_bridge_contract.py",
            "run scripts/hermes_service_bridge.py check in the Hermes checkout",
            "run bridges/hermes-mcp-server/hermes_mcp_server.py --check --live",
            "run scripts/jcode_bridge_latency_probe.py --iterations 50",
            "run scripts/jcode_native_tool_check.py --jcode <jcode checkout>",
            "run Hermes-side jcode_bridge_smoke.py in the Hermes checkout",
            "generate and archive an upstream-sync report",
        ],
    }


def _copy_file(source: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite")
    shutil.copy2(source, destination)


def _copy_dir(source: Path, destination: Path, *, force: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing source directory: {source}")
    if destination.exists() and not force:
        raise FileExistsError(f"{destination} already exists; pass --force to overwrite")
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=force,
        ignore=shutil.ignore_patterns("target", "__pycache__", "*.pyc"),
    )


def _write_text(path: Path, text: str, *, force: bool, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _jcode_mcp_config(output: Path) -> dict[str, Any]:
    mcp_server = output / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"
    return {
        "servers": {
            "hermes": {
                "command": "python3",
                "args": [
                    str(mcp_server),
                    "--allow-tool",
                    "web_search",
                    "--allow-tool",
                    "web_extract",
                    "--allow-tool",
                    "session_search",
                    "--allow-tool",
                    "memory",
                ],
                "env": {
                    "JCODE_BRIDGE_ROOT": str(output),
                },
                "shared": True,
            }
        }
    }


def scaffold(output: Path, manifest: dict[str, Any], *, force: bool) -> dict[str, Any]:
    """Create a mother-repo scaffold and return copied artifact paths."""
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    _write_text(output / "README.md", README, force=force)
    copied.append("README.md")
    _write_text(output / "upstreams" / "README.md", UPSTREAMS_README, force=force)
    copied.append("upstreams/README.md")
    _write_text(
        output / "hermes-jcode.manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        force=force,
    )
    copied.append("hermes-jcode.manifest.json")

    _copy_dir(BRIDGE_PLUGIN, output / "bridges" / "hermes-plugin-jcode" / "plugins" / "jcode_bridge", force=force)
    copied.append("bridges/hermes-plugin-jcode/plugins/jcode_bridge")
    _copy_dir(JCODE_TOOL_HERMES, output / "bridges" / "jcode-tool-hermes", force=force)
    copied.append("bridges/jcode-tool-hermes")
    _copy_dir(JCODE_NATIVE_HERMES_TOOL, output / "bridges" / "jcode-native-hermes-tool", force=force)
    copied.append("bridges/jcode-native-hermes-tool")
    _copy_dir(HERMES_MCP_SERVER, output / "bridges" / "hermes-mcp-server", force=force)
    copied.append("bridges/hermes-mcp-server")
    _copy_dir(CONTRACT_DIR, output / "contracts" / "jcode_bridge" / "v1", force=force)
    copied.append("contracts/jcode_bridge/v1")
    _copy_dir(HERMES_SERVICE_CONTRACT_DIR, output / "contracts" / "hermes_service" / "v1", force=force)
    copied.append("contracts/hermes_service/v1")
    _copy_dir(HERMES_MCP_CONTRACT_DIR, output / "contracts" / "hermes_mcp" / "v1", force=force)
    copied.append("contracts/hermes_mcp/v1")
    _copy_dir(FIXTURE_DIR, output / "tests" / "fixtures" / "jcode_bridge", force=force)
    copied.append("tests/fixtures/jcode_bridge")
    _copy_dir(HERMES_SERVICE_FIXTURE_DIR, output / "tests" / "fixtures" / "hermes_service", force=force)
    copied.append("tests/fixtures/hermes_service")
    _copy_dir(HERMES_MCP_FIXTURE_DIR, output / "tests" / "fixtures" / "hermes_mcp", force=force)
    copied.append("tests/fixtures/hermes_mcp")

    docs_out = output / "docs" / "plans"
    for doc in PLAN_DOCS:
        if doc.exists():
            _copy_file(doc, docs_out / doc.name, force=force)
            copied.append(f"docs/plans/{doc.name}")

    _write_text(output / "scripts" / "check_bridge_contract.py", CHECK_SCRIPT, force=force, executable=True)
    copied.append("scripts/check_bridge_contract.py")
    _write_text(output / "scripts" / "hermes_service_bridge.py", SERVICE_SCRIPT, force=force, executable=True)
    copied.append("scripts/hermes_service_bridge.py")
    _copy_file(LATENCY_PROBE, output / "scripts" / "jcode_bridge_latency_probe.py", force=force)
    copied.append("scripts/jcode_bridge_latency_probe.py")
    _copy_file(NATIVE_TOOL_CHECK, output / "scripts" / "jcode_native_tool_check.py", force=force)
    copied.append("scripts/jcode_native_tool_check.py")
    _write_text(
        output / "configs" / "jcode-mcp.hermes.json",
        json.dumps(_jcode_mcp_config(output), indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        force=force,
    )
    copied.append("configs/jcode-mcp.hermes.json")

    return {
        "output": str(output),
        "copied": copied,
        "manifest": str(output / "hermes-jcode.manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Print the current mother-repo manifest.")
    manifest_parser.add_argument("--hermes", default=str(ROOT), help="Hermes repo path.")
    manifest_parser.add_argument("--jcode", default=str(DEFAULT_JCODE), help="jcode repo path.")

    scaffold_parser = subparsers.add_parser("scaffold", help="Create a local mother-repo scaffold.")
    scaffold_parser.add_argument("--output", required=True, help="Output directory for the scaffold.")
    scaffold_parser.add_argument("--hermes", default=str(ROOT), help="Hermes repo path.")
    scaffold_parser.add_argument("--jcode", default=str(DEFAULT_JCODE), help="jcode repo path.")
    scaffold_parser.add_argument("--force", action="store_true", help="Overwrite existing scaffold files.")

    ns = parser.parse_args(argv)
    hermes = Path(ns.hermes)
    jcode = Path(ns.jcode)
    manifest = build_manifest(hermes, jcode)

    if ns.command == "manifest":
        print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))
        return 0

    result = scaffold(Path(ns.output), manifest, force=bool(ns.force))
    print(json.dumps({
        "success": True,
        **result,
    }, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
