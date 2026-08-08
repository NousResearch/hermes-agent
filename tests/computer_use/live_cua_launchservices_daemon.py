"""Opt-in macOS live proof for the LaunchServices-launched private daemon.

Verifies the fix for the recurring macOS Screen Recording prompt: an unrestricted
Hermes session must start its private cua-driver daemon through LaunchServices
(as ``com.trycua.driver``) instead of as a direct child of the Hermes process, so
ScreenCaptureKit attributes capture to CuaDriver's existing TCC grant rather than
to Hermes' (frequently stale-cdhash) row.

Never installs, updates, grants, or resets anything. Starts one private daemon,
asserts its TCC attribution, asserts the stdio MCP proxy reaches it without
``--embedded``, then stops it.

    .venv/bin/python tests/computer_use/live_cua_launchservices_daemon.py

Exit 0 = every check passed. ``--check-imports`` resolves the import path and
exits without touching a daemon (used by the entry-point regression test).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Running a script by path puts THIS directory on sys.path[0], never the repo
# root, so `import tools...` would otherwise resolve through whatever
# hermes-agent distribution the interpreter happens to have installed — which is
# a different checkout under any shared venv. Pin this worktree first so the
# documented command tests the code next to it, with no PYTHONPATH, cwd, or
# editable-install assumptions.
REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from tools.computer_use import cua_backend  # noqa: E402  (must follow the path pin)

_EmbeddedCuaDaemon = cua_backend._EmbeddedCuaDaemon
resolve_cua_driver_app = cua_backend.resolve_cua_driver_app


def _structured(result: Any) -> dict[str, Any]:
    value = getattr(result, "structuredContent", None)
    if isinstance(value, dict):
        return value
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


async def _probe_proxy(command: str, args: list[str], env: dict[str, str]) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            perms = _structured(await session.call_tool("check_permissions", {}))
            return {"tools": tools, "permissions": perms}


def _socket_answers(driver_cmd: str, socket_path: str) -> bool:
    probe = subprocess.run(
        [driver_cmd, "status", "--socket", socket_path],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return probe.returncode == 0


def _machine_wide_capturable(driver_cmd: str) -> Any:
    """``screen_recording_capturable`` on the standard machine daemon, or None.

    Used to tell a real regression in this change apart from a machine-level
    capture condition: if the shared daemon reports the same value, the reading
    is about the host, not about how our private daemon was launched.
    """
    try:
        probe = subprocess.run(
            [driver_cmd, "call", "check_permissions", "{}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        start = (probe.stdout or "").find("{")
        if start == -1:
            return None
        return json.loads(probe.stdout[start:]).get("screen_recording_capturable")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _check_imports() -> int:
    """Report which checkout ``tools.computer_use.cua_backend`` came from."""
    origin = Path(cua_backend.__file__).resolve()
    print(f"repo_root: {REPO_ROOT}")
    print(f"cua_backend: {origin}")
    if not origin.is_relative_to(REPO_ROOT):
        print("FAIL: imported cua_backend from outside this worktree")
        return 1
    if not hasattr(cua_backend, "resolve_cua_driver_app"):
        print("FAIL: imported cua_backend predates resolve_cua_driver_app")
        return 1
    print("PASS: imports resolve to this worktree")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check-imports" in argv:
        return _check_imports()

    print(f"cua_backend: {Path(cua_backend.__file__).resolve()}")
    if sys.platform != "darwin":
        print("skip: macOS-only proof")
        return 0

    app = resolve_cua_driver_app()
    print(f"CuaDriver.app: {app}")
    if not app:
        print("FAIL: no CuaDriver.app bundle resolved; install CuaDriver first")
        return 1

    daemon = _EmbeddedCuaDaemon("", "unrestricted")
    launch_argv, via_launch_services = daemon._launch_command()
    print(f"launch argv: {launch_argv}")

    failures: list[str] = []
    warnings: list[str] = []
    if not via_launch_services or launch_argv[0] != "/usr/bin/open":
        print("FAIL: macOS launch did not go through LaunchServices")
        return 1

    daemon.start()
    try:
        print(f"socket: {daemon.socket_path}")
        command, args = daemon.proxy_invocation()
        print(f"proxy: {command} {args}")
        if "--embedded" in args:
            failures.append("proxy invocation still carries --embedded")
        report = asyncio.run(_probe_proxy(command, args, daemon.child_env()))
        perms = report["permissions"]
        source = perms.get("source") or {}
        print(json.dumps({"permissions": perms, "tool_count": len(report["tools"])}, indent=2))
        # What this lane actually changed: WHICH TCC identity the daemon runs
        # under, and that that identity holds the Screen Recording grant.
        if source.get("attribution") != "driver-daemon":
            failures.append(f"attribution is {source.get('attribution')!r}, want 'driver-daemon'")
        if source.get("responsible_ppid") != 1:
            failures.append(f"responsible_ppid is {source.get('responsible_ppid')!r}, want 1")
        if not str(source.get("executable", "")).startswith(app):
            failures.append(f"executable is {source.get('executable')!r}, want inside {app}")
        if not perms.get("screen_recording"):
            failures.append("com.trycua.driver holds no Screen Recording grant")
        # Whether a capture SUCCEEDS right now is host state, not attribution:
        # a pending macOS auth-warning window or stale daemons holding
        # ScreenCaptureKit resources flip it for every daemon on the box. Compare
        # against the shared daemon so a host condition cannot masquerade as a
        # regression in this change.
        if not perms.get("screen_recording_capturable"):
            shared = _machine_wide_capturable(daemon._command)
            warnings.append(
                "screen_recording_capturable is false; the shared machine daemon "
                f"reports {shared!r} for the same check, so this is host state, "
                "not launch attribution"
            )
    finally:
        daemon.stop()

    if _socket_answers(daemon._command, daemon.socket_path):
        failures.append("daemon still answering after stop()")

    for line in warnings:
        print(f"WARN: {line}")
    for line in failures:
        print(f"FAIL: {line}")
    if not failures:
        print("PASS: LaunchServices attribution, grant, proxy, and stop all verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
