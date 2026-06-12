"""CLI command for the QuestFrame Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="questframe_command")

    setup = subs.add_parser("setup", help="Save QuestFrame bridge paths")
    setup.add_argument("--launcher-exe", default="")
    setup.add_argument("--unity-python", default="")
    setup.add_argument("--vcc-project-root", action="append", default=[])

    subs.add_parser("status", help="Show QuestFrame bridge readiness")

    preflight = subs.add_parser("preflight", help="Run FH6VR preflight")
    preflight.add_argument("--launcher-exe", default="")
    preflight.add_argument("--report-path", default="")
    preflight.add_argument("--timeout-seconds", type=int, default=None)

    profiles = subs.add_parser("profiles", help="Print FH6VR RTX 3060 DIBR profiles")
    profiles.add_argument("--launcher-exe", default="")
    profiles.add_argument("--timeout-seconds", type=int, default=None)

    rtx3060 = subs.add_parser(
        "rtx3060-selftest", help="Validate FH6VR RTX 3060 DIBR profile budgets"
    )
    rtx3060.add_argument("--launcher-exe", default="")
    rtx3060.add_argument("--timeout-seconds", type=int, default=None)

    session = subs.add_parser(
        "session-readiness", help="Run FH6VR OpenXR session-readiness probe"
    )
    session.add_argument("--launcher-exe", default="")
    session.add_argument("--timeout-seconds", type=int, default=None)

    graphics_session = subs.add_parser(
        "graphics-session",
        help="Run FH6VR OpenXR graphics-bound session and swapchain-format probe",
    )
    graphics_session.add_argument("--launcher-exe", default="")
    graphics_session.add_argument("--timeout-seconds", type=int, default=None)

    frame_loop = subs.add_parser("frame-loop", help="Run FH6VR minimal OpenXR frame-loop probe")
    frame_loop.add_argument("--launcher-exe", default="")
    frame_loop.add_argument("--timeout-seconds", type=int, default=None)

    dibr_swapchain = subs.add_parser(
        "dibr-swapchain", help="Run FH6VR DIBR-to-OpenXR swapchain write probe"
    )
    dibr_swapchain.add_argument("--launcher-exe", default="")
    dibr_swapchain.add_argument("--timeout-seconds", type=int, default=None)

    capture_preflight = subs.add_parser(
        "capture-preflight", help="Run FH6VR non-invasive FH6/D3D12 capture preflight"
    )
    capture_preflight.add_argument("--launcher-exe", default="")
    capture_preflight.add_argument("--timeout-seconds", type=int, default=None)

    live_capture = subs.add_parser(
        "live-capture-selftest",
        help="Run FH6VR live D3D12/window color capture self-test (0.14 gate)",
    )
    live_capture.add_argument("--launcher-exe", default="")
    live_capture.add_argument("--attempt-window-capture", action="store_true")
    live_capture.add_argument("--timeout-seconds", type=int, default=None)

    depth_surface = subs.add_parser(
        "depth-surface-selftest",
        help="Run FH6VR approved D3D12 depth surface contract self-test (0.15 gate)",
    )
    depth_surface.add_argument("--launcher-exe", default="")
    depth_surface.add_argument("--timeout-seconds", type=int, default=None)

    support_report = subs.add_parser(
        "support-report", help="Create redacted JSON and HTML support reports"
    )
    support_report.add_argument("--launcher-exe", default="")
    support_report.add_argument("--json-path", default="")
    support_report.add_argument("--html-path", default="")
    support_report.add_argument("--include-live-openxr", action="store_true")
    support_report.add_argument("--no-openxr", action="store_true")
    support_report.add_argument("--include-sensitive-paths", action="store_true")
    support_report.add_argument("--timeout-seconds", type=int, default=None)

    unity_scan = subs.add_parser("unity-scan", help="Scan Unity/VCC projects")
    unity_scan.add_argument("--project-path", default="")
    unity_scan.add_argument("--max-projects", type=int, default=None)

    subparser.set_defaults(func=questframe_command)


def questframe_command(args: argparse.Namespace) -> int:
    command = getattr(args, "questframe_command", None)
    if not command:
        print(
            "usage: hermes questframe "
            "{setup,status,preflight,profiles,rtx3060-selftest,session-readiness,"
            "graphics-session,frame-loop,"
            "dibr-swapchain,capture-preflight,live-capture-selftest,"
            "depth-surface-selftest,support-report,unity-scan}"
        )
        return 2
    if command == "setup":
        return _print(
            core.save_setup_values(
                {
                    "launcher_exe": getattr(args, "launcher_exe", ""),
                    "unity_python": getattr(args, "unity_python", ""),
                    "vcc_project_roots": getattr(args, "vcc_project_root", []) or [],
                }
            )
        )
    if command == "status":
        return _print(core.status())
    if command == "preflight":
        return _print(
            core.run_launcher(
                "preflight",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=_preflight_args(getattr(args, "report_path", "")),
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "profiles":
        return _print(
            core.run_launcher(
                "profiles",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "rtx3060-selftest":
        return _print(
            core.run_launcher(
                "rtx3060-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "session-readiness":
        return _print(
            core.run_launcher(
                "session-readiness-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "graphics-session":
        return _print(
            core.run_launcher(
                "graphics-session-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "frame-loop":
        return _print(
            core.run_launcher(
                "frame-loop-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "dibr-swapchain":
        return _print(
            core.run_launcher(
                "dibr-swapchain-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "capture-preflight":
        return _print(
            core.run_launcher(
                "fh6-capture-preflight",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "live-capture-selftest":
        extra = ["--json"]
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        return _print(
            core.run_launcher(
                "fh6-live-capture-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "depth-surface-selftest":
        return _print(
            core.run_launcher(
                "fh6-depth-surface-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "support-report":
        return _print(
            core.support_report(
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                json_path=getattr(args, "json_path", "") or None,
                html_path=getattr(args, "html_path", "") or None,
                include_live_openxr=bool(getattr(args, "include_live_openxr", False)),
                no_openxr=bool(getattr(args, "no_openxr", False)),
                include_sensitive_paths=bool(
                    getattr(args, "include_sensitive_paths", False)
                ),
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "unity-scan":
        return _print(
            core.scan_unity_projects(
                project_path=getattr(args, "project_path", "") or None,
                max_projects=getattr(args, "max_projects", None),
            )
        )
    print(f"unknown command: {command}")
    return 2


def _preflight_args(report_path: str) -> list[str]:
    args = ["--json"]
    if report_path:
        args.extend(["--write-report", report_path])
    return args


def _print(data: dict) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if data.get("ok") else 1
