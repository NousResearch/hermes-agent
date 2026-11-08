"""Opt-in macOS smoke test for the installed cua-driver live MCP contract.

This script never installs, updates, or grants an existing browser profile. Start
an isolated daemon separately, then point this script at its socket:

    cua-driver serve --embedded --socket /tmp/hermes-cua-0-9-live.sock \
        --no-permissions-gate --no-overlay
    CUA_DRIVER_LIVE_SOCKET=/tmp/hermes-cua-0-9-live.sock \
        .venv/bin/python tests/computer_use/live_cua_0_9_smoke.py

The output deliberately excludes process IDs, window IDs, socket paths, and
driver payloads. Each cell is classified as pass, structured_refusal,
environment_unavailable, or unproven.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def structured(result: Any) -> dict[str, Any]:
    value = getattr(result, "structuredContent", None)
    if isinstance(value, dict):
        return value
    dumped = result.model_dump(by_alias=True) if hasattr(result, "model_dump") else {}
    for key in ("structuredContent", "structured_content"):
        value = dumped.get(key)
        if isinstance(value, dict):
            return value
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def refusal_code(payload: dict[str, Any]) -> str | None:
    refusal = payload.get("refusal")
    return payload.get("code") or (
        refusal.get("code") if isinstance(refusal, dict) else None
    )


def textedit_process_contains(pid: int, marker: str) -> bool:
    """Read the exact throwaway process through the native AX script bridge."""
    script = """
on run argv
    set targetPid to item 1 of argv as integer
    set markerText to item 2 of argv
    tell application "System Events"
        tell first application process whose unix id is targetPid
            set documentText to value of text area 1 of scroll area 1 of window 1
        end tell
    end tell
    return (documentText contains markerText) as text
end run
"""
    try:
        result = subprocess.run(
            ["osascript", "-e", script, "--", str(pid), marker],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


async def run_smoke(socket_path: str) -> dict[str, dict[str, Any]]:
    session_id = f"hermes-cua-live-{uuid.uuid4().hex[:8]}"
    params = StdioServerParameters(
        command="cua-driver",
        args=["mcp", "--embedded", "--socket", socket_path],
    )
    report: dict[str, dict[str, Any]] = {
        "foreground": {"classification": "unproven"},
        "typed_browser": {"classification": "unproven"},
    }
    launched_pid: int | None = None
    isolated_browser_pid: int | None = None
    browser_pid: int | None = None
    prior_foreground_pids: set[int] = set()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="hermes-cua-live-", suffix=".txt"
    )
    os.close(file_descriptor)
    smoke_path = Path(temporary_name)

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                await client.call_tool("start_session", {"session": session_id})
                try:
                    before_windows = structured(
                        await client.call_tool(
                            "list_windows",
                            {"on_screen_only": True, "session": session_id},
                        )
                    )
                    prior_foreground_pids = {
                        pid
                        for row in before_windows.get("windows") or []
                        if "textedit" in str(row.get("app_name") or "").lower()
                        and isinstance((pid := row.get("pid")), int)
                    }
                    launched = structured(
                        await client.call_tool(
                            "launch_app",
                            {
                                "name": "TextEdit",
                                "urls": [smoke_path.as_uri()],
                                "creates_new_application_instance": True,
                                "session": session_id,
                            },
                        )
                    )
                    launched_pid = launched.get("pid")
                    windows = launched.get("windows") or []
                    if isinstance(launched_pid, int) and not windows:
                        await client.call_tool(
                            "wait", {"seconds": 1, "session": session_id}
                        )
                        refreshed = structured(
                            await client.call_tool(
                                "list_windows",
                                {"on_screen_only": True, "session": session_id},
                            )
                        )
                        windows = [
                            row
                            for row in refreshed.get("windows") or []
                            if row.get("pid") == launched_pid
                        ]
                    window_id = windows[0].get("window_id") if windows else None
                    if (
                        not isinstance(launched_pid, int)
                        or launched_pid in prior_foreground_pids
                        or not isinstance(window_id, int)
                    ):
                        report["foreground"] = {
                            "classification": "environment_unavailable",
                            "stage": "throwaway_target",
                        }
                    else:
                        focus = await client.call_tool(
                            "bring_to_front",
                            {"pid": launched_pid, "window_id": window_id},
                        )
                        before = structured(
                            await client.call_tool(
                                "get_window_state",
                                {
                                    "pid": launched_pid,
                                    "window_id": window_id,
                                    "session": session_id,
                                },
                            )
                        )
                        editor = next(
                            (
                                element
                                for element in before.get("elements") or []
                                if str(element.get("role") or "").lower()
                                in {"axtextarea", "axtextfield"}
                            ),
                            None,
                        )
                        if not isinstance(editor, dict):
                            report["foreground"] = {
                                "classification": "unproven",
                                "stage": "editor_discovery",
                            }
                        else:
                            marker = "hermes foreground smoke"
                            type_args = {
                                "pid": launched_pid,
                                "window_id": window_id,
                                "element_index": editor.get("index"),
                                "text": marker,
                                "delivery_mode": "foreground",
                                "session": session_id,
                            }
                            token = editor.get("element_token")
                            if isinstance(token, str) and token:
                                type_args["element_token"] = token
                            typed = structured(
                                await client.call_tool("type_text", type_args)
                            )
                            saved = structured(
                                await client.call_tool(
                                    "hotkey",
                                    {
                                        "pid": launched_pid,
                                        "window_id": window_id,
                                        "keys": ["cmd", "s"],
                                        "delivery_mode": "foreground",
                                        "session": session_id,
                                    },
                                )
                            )
                            await client.call_tool(
                                "wait", {"seconds": 0.5, "session": session_id}
                            )
                            after = structured(
                                await client.call_tool(
                                    "get_window_state",
                                    {
                                        "pid": launched_pid,
                                        "window_id": window_id,
                                        "session": session_id,
                                    },
                                )
                            )
                            fresh_contains_marker = marker in json.dumps(
                                after.get("elements") or []
                            )
                            native_document_confirmed = textedit_process_contains(
                                launched_pid, marker
                            )
                            file_contains_marker = marker in smoke_path.read_text(
                                encoding="utf-8"
                            )
                            report["foreground"] = {
                                "classification": (
                                    "pass"
                                    if not focus.isError
                                    and not refusal_code(typed)
                                    and not refusal_code(saved)
                                    and (
                                        typed.get("verified") is True
                                        or fresh_contains_marker
                                        or native_document_confirmed
                                        or file_contains_marker
                                    )
                                    else "unproven"
                                ),
                                "focus_transport_ok": not focus.isError,
                                "effect": typed.get("effect"),
                                "verified": typed.get("verified"),
                                "fresh_state": bool(after.get("elements")),
                                "fresh_state_confirmed": fresh_contains_marker,
                                "native_document_confirmed": (
                                    native_document_confirmed
                                ),
                                "saved_file_confirmed": file_contains_marker,
                                "action_schema_omitted_bring_to_front": (
                                    "bring_to_front" not in type_args
                                ),
                            }

                    # Use only a driver-owned isolated profile. Never request,
                    # mint, print, or persist an existing-profile grant token.
                    listed = structured(
                        await client.call_tool(
                            "list_windows",
                            {"on_screen_only": True, "session": session_id},
                        )
                    )
                    browser_row = next(
                        (
                            row
                            for row in listed.get("windows") or []
                            if "chrome" in str(row.get("app_name") or "").lower()
                        ),
                        None,
                    )
                    browser_pid = browser_row.get("pid") if browser_row else None
                    browser_window = (
                        browser_row.get("window_id") if browser_row else None
                    )
                    if not isinstance(browser_pid, int) or not isinstance(
                        browser_window, int
                    ):
                        report["typed_browser"] = {
                            "classification": "environment_unavailable",
                            "stage": "browser_target",
                        }
                    else:
                        prepared = structured(
                            await client.call_tool(
                                "browser_prepare",
                                {
                                    "pid": browser_pid,
                                    "window_id": browser_window,
                                    "allow_launch": True,
                                    "profile": {"mode": "isolated_new"},
                                    "session": session_id,
                                },
                            )
                        )
                        isolated_browser_pid = prepared.get("prepared_pid")
                        code = refusal_code(prepared)
                        if prepared.get("status") == "refused" or code:
                            report["typed_browser"] = {
                                "classification": "structured_refusal",
                                "code": code,
                            }
                        else:
                            prepared_pid = prepared.get("prepared_pid") or browser_pid
                            await client.call_tool(
                                "wait", {"seconds": 1, "session": session_id}
                            )
                            prepared_windows = structured(
                                await client.call_tool(
                                    "list_windows",
                                    {
                                        "on_screen_only": True,
                                        "session": session_id,
                                    },
                                )
                            )
                            prepared_row = next(
                                (
                                    row
                                    for row in prepared_windows.get("windows") or []
                                    if row.get("pid") == prepared_pid
                                ),
                                None,
                            )
                            prepared_window = (
                                prepared_row.get("window_id")
                                if prepared_row
                                else browser_window
                            )
                            bound = structured(
                                await client.call_tool(
                                    "get_browser_state",
                                    {
                                        "pid": prepared_pid,
                                        "window_id": prepared_window,
                                        "session": session_id,
                                    },
                                )
                            )
                            tabs = bound.get("tabs") or []
                            tab_id = tabs[0].get("tab_id") if tabs else None
                            target_id = bound.get("target_id")
                            if (
                                bound.get("status") == "ok"
                                and bound.get("binding_quality") == "exact"
                                and bound.get("mutation_allowed") is True
                                and isinstance(tab_id, str)
                                and isinstance(target_id, str)
                            ):
                                snapshot = structured(
                                    await client.call_tool(
                                        "get_browser_state",
                                        {
                                            "target_id": target_id,
                                            "tab_id": tab_id,
                                            "snapshot_format": "semantic_v2",
                                            "session": session_id,
                                        },
                                    )
                                )
                                navigated = structured(
                                    await client.call_tool(
                                        "browser_navigate",
                                        {
                                            "target_id": target_id,
                                            "tab_id": tab_id,
                                            "url": "about:blank",
                                            "session": session_id,
                                        },
                                    )
                                )
                                fresh = structured(
                                    await client.call_tool(
                                        "get_browser_state",
                                        {
                                            "target_id": target_id,
                                            "tab_id": tab_id,
                                            "snapshot_format": "semantic_v2",
                                            "session": session_id,
                                        },
                                    )
                                )
                                report["typed_browser"] = {
                                    "classification": "pass",
                                    "exact_binding": True,
                                    "mutation_allowed": True,
                                    "initial_snapshot": snapshot.get("status")
                                    in (None, "ok"),
                                    "mutation_transport": navigated.get("status")
                                    in (None, "ok"),
                                    "fresh_verification": fresh.get("status")
                                    in (None, "ok"),
                                }
                            else:
                                report["typed_browser"] = {
                                    "classification": "unproven",
                                    "stage": "exact_binding",
                                    "code": refusal_code(bound),
                                }
                finally:
                    if (
                        isinstance(launched_pid, int)
                        and launched_pid not in prior_foreground_pids
                    ):
                        await client.call_tool(
                            "kill_app", {"pid": launched_pid, "session": session_id}
                        )
                    if (
                        isinstance(isolated_browser_pid, int)
                        and isolated_browser_pid != browser_pid
                    ):
                        await client.call_tool(
                            "kill_app",
                            {"pid": isolated_browser_pid, "session": session_id},
                        )
                    await client.call_tool("end_session", {"session": session_id})
    finally:
        smoke_path.unlink(missing_ok=True)
    return report


def main() -> int:
    report: dict[str, dict[str, Any]] = {
        "foreground": {"classification": "environment_unavailable"},
        "typed_browser": {"classification": "environment_unavailable"},
    }
    if sys.platform != "darwin":
        for cell in report.values():
            cell["stage"] = "macos_host_required"
    else:
        socket_path = os.environ.get(
            "CUA_DRIVER_LIVE_SOCKET", "/tmp/hermes-cua-0-9-live.sock"
        )
        if not Path(socket_path).is_socket():
            for cell in report.values():
                cell["stage"] = "isolated_daemon_required"
        else:
            try:
                report = asyncio.run(run_smoke(socket_path))
            except Exception as exc:  # pragma: no cover - host/driver boundary
                report = {
                    "foreground": {
                        "classification": "environment_unavailable",
                        "stage": "driver_connection",
                        "error_type": type(exc).__name__,
                    },
                    "typed_browser": {
                        "classification": "environment_unavailable",
                        "stage": "driver_connection",
                        "error_type": type(exc).__name__,
                    },
                }
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(any(cell.get("classification") != "pass" for cell in report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
