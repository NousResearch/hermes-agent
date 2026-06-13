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
    live_capture.add_argument("--require-foreground", action="store_true")
    live_capture.add_argument("--timeout-seconds", type=int, default=None)

    depth_surface = subs.add_parser(
        "depth-surface-selftest",
        help="Run FH6VR approved D3D12 depth surface contract self-test (0.15 gate)",
    )
    depth_surface.add_argument("--launcher-exe", default="")
    depth_surface.add_argument("--timeout-seconds", type=int, default=None)

    depth_reader = subs.add_parser(
        "depth-reader-selftest",
        help="Run FH6VR approved D3D12 depth reader self-test (0.16 gate)",
    )
    depth_reader.add_argument("--launcher-exe", default="")
    depth_reader.add_argument("--fixture", action="store_true")
    depth_reader.add_argument("--timeout-seconds", type=int, default=None)

    depth_producer = subs.add_parser(
        "depth-producer-selftest",
        help="Run FH6VR approved D3D12 depth producer metadata self-test (0.17 gate)",
    )
    depth_producer.add_argument("--launcher-exe", default="")
    depth_producer.add_argument("--fixture", action="store_true")
    depth_producer.add_argument("--metadata", default="")
    depth_producer.add_argument("--timeout-seconds", type=int, default=None)

    companion_producer = subs.add_parser(
        "companion-depth-producer-selftest",
        help="Run FH6VR companion depth producer handoff self-test (0.18 gate)",
    )
    companion_producer.add_argument("--launcher-exe", default="")
    companion_producer.add_argument("--approve", action="store_true")
    companion_producer.add_argument("--metadata", default="")
    companion_producer.add_argument("--output-dir", default="")
    companion_producer.add_argument("--frames", type=int, default=None)
    companion_producer.add_argument("--interval-ms", type=int, default=None)
    companion_producer.add_argument("--timeout-seconds", type=int, default=None)

    color_depth_pairing = subs.add_parser(
        "color-depth-pairing-selftest",
        help="Run FH6VR live color + companion depth pairing self-test (0.19 gate)",
    )
    color_depth_pairing.add_argument("--launcher-exe", default="")
    color_depth_pairing.add_argument("--approve", action="store_true")
    color_depth_pairing.add_argument("--attempt-window-capture", action="store_true")
    color_depth_pairing.add_argument("--require-foreground", action="store_true")
    color_depth_pairing.add_argument("--metadata", default="")
    color_depth_pairing.add_argument("--output-dir", default="")
    color_depth_pairing.add_argument("--timeout-seconds", type=int, default=None)

    openxr_presentation = subs.add_parser(
        "openxr-presentation-selftest",
        help="Run FH6VR OpenXR presentation self-test (0.19 gate)",
    )
    openxr_presentation.add_argument("--launcher-exe", default="")
    openxr_presentation.add_argument("--approve", action="store_true")
    openxr_presentation.add_argument("--attempt-window-capture", action="store_true")
    openxr_presentation.add_argument("--require-foreground", action="store_true")
    openxr_presentation.add_argument("--require-pairing", action="store_true")
    openxr_presentation.add_argument("--require-hmd", action="store_true")
    openxr_presentation.add_argument("--min-hmd-width", type=int, default=None)
    openxr_presentation.add_argument("--min-hmd-height", type=int, default=None)
    openxr_presentation.add_argument("--immersive-check", action="store_true")
    openxr_presentation.add_argument("--frames", type=int, default=None)
    openxr_presentation.add_argument("--timeout-seconds", type=int, default=None)

    kofi_parity = subs.add_parser(
        "kofi-parity-selftest",
        help="Run FH6 Ko-fi parity immersion metrics (0.22 gate)",
    )
    kofi_parity.add_argument("--launcher-exe", default="")
    kofi_parity.add_argument("--approve", action="store_true")
    kofi_parity.add_argument("--attempt-window-capture", action="store_true")
    kofi_parity.add_argument("--loop-frames", type=int, default=None)
    kofi_parity.add_argument("--cockpit-seconds", type=int, default=None)
    kofi_parity.add_argument("--target-hz", type=int, default=None)
    kofi_parity.add_argument("--timeout-seconds", type=int, default=None)

    pcvr_management = subs.add_parser(
        "pcvr-management-selftest",
        help="Run read-only QuestFrame PCVR management self-test (0.22 gate)",
    )
    pcvr_management.add_argument("--launcher-exe", default="")
    pcvr_management.add_argument("--allow-missing-runtime", action="store_true")
    pcvr_management.add_argument("--no-process-list", action="store_true")
    pcvr_management.add_argument("--timeout-seconds", type=int, default=None)

    hermes_bridge = subs.add_parser(
        "hermes-bridge-selftest",
        help="Validate QuestFrame C# backend contract for the Hermes Agent plugin",
    )
    hermes_bridge.add_argument("--launcher-exe", default="")
    hermes_bridge.add_argument("--timeout-seconds", type=int, default=None)

    hmd_controller_input = subs.add_parser(
        "hmd-controller-input-selftest",
        help="Validate HMD controller driving mapping and virtual gamepad readiness",
    )
    hmd_controller_input.add_argument("--launcher-exe", default="")
    hmd_controller_input.add_argument("--allow-missing-runtime", action="store_true")
    hmd_controller_input.add_argument("--require-virtual-gamepad", action="store_true")
    hmd_controller_input.add_argument("--no-process-list", action="store_true")
    hmd_controller_input.add_argument("--timeout-seconds", type=int, default=None)

    vcc_health = subs.add_parser(
        "vcc-health",
        help="Check VRChat Creator Companion and Unity Hub readiness",
    )
    vcc_health.add_argument("--project-path", default="")

    cockpit_presence = subs.add_parser(
        "cockpit-presence-selftest",
        help="Run FH6VR cockpit presence self-test with head pose parallax (0.21 gate)",
    )
    cockpit_presence.add_argument("--launcher-exe", default="")
    cockpit_presence.add_argument("--approve", action="store_true")
    cockpit_presence.add_argument("--attempt-window-capture", action="store_true")
    cockpit_presence.add_argument("--seconds", type=int, default=None)
    cockpit_presence.add_argument("--target-hz", type=int, default=None)
    cockpit_presence.add_argument("--timeout-seconds", type=int, default=None)

    immersive_presentation_loop = subs.add_parser(
        "immersive-presentation-loop-selftest",
        help="Run sustained FH6VR immersive OpenXR presentation loop (0.20 gate)",
    )
    immersive_presentation_loop.add_argument("--launcher-exe", default="")
    immersive_presentation_loop.add_argument("--approve", action="store_true")
    immersive_presentation_loop.add_argument("--attempt-window-capture", action="store_true")
    immersive_presentation_loop.add_argument("--seconds", type=int, default=None)
    immersive_presentation_loop.add_argument("--target-hz", type=int, default=None)
    immersive_presentation_loop.add_argument("--timeout-seconds", type=int, default=None)

    live_color_loop = subs.add_parser(
        "live-color-loop-selftest",
        help="Run FH6 live color + companion depth loop self-test (0.19.2 gate, DXGI capture)",
    )
    live_color_loop.add_argument("--launcher-exe", default="")
    live_color_loop.add_argument("--approve", action="store_true")
    live_color_loop.add_argument("--attempt-window-capture", action="store_true")
    live_color_loop.add_argument("--frames", type=int, default=None)
    live_color_loop.add_argument("--target-hz", type=int, default=None)
    live_color_loop.add_argument("--timeout-seconds", type=int, default=None)

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
            "depth-surface-selftest,depth-reader-selftest,depth-producer-selftest,"
            "companion-depth-producer-selftest,color-depth-pairing-selftest,"
            "openxr-presentation-selftest,live-color-loop-selftest,"
            "immersive-presentation-loop-selftest,cockpit-presence-selftest,"
            "kofi-parity-selftest,pcvr-management-selftest,vcc-health,"
            "hermes-bridge-selftest,hmd-controller-input-selftest,"
            "support-report,unity-scan}"
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
        if bool(getattr(args, "require_foreground", False)):
            extra.append("--require-foreground")
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
    if command == "depth-reader-selftest":
        extra = ["--json"]
        if bool(getattr(args, "fixture", False)):
            extra.append("--fixture")
        return _print(
            core.run_launcher(
                "fh6-depth-reader-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "depth-producer-selftest":
        extra = ["--json"]
        if bool(getattr(args, "fixture", False)):
            extra.append("--fixture")
        metadata_path = str(getattr(args, "metadata", "") or "").strip()
        if metadata_path:
            extra.extend(["--metadata", metadata_path])
        return _print(
            core.run_launcher(
                "fh6-depth-producer-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "companion-depth-producer-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        metadata_path = str(getattr(args, "metadata", "") or "").strip()
        if metadata_path:
            extra.extend(["--metadata", metadata_path])
        output_dir = str(getattr(args, "output_dir", "") or "").strip()
        if output_dir:
            extra.extend(["--output-dir", output_dir])
        frames = getattr(args, "frames", None)
        if frames:
            extra.extend(["--frames", str(frames)])
        interval_ms = getattr(args, "interval_ms", None)
        if interval_ms:
            extra.extend(["--interval-ms", str(interval_ms)])
        return _print(
            core.run_launcher(
                "fh6-companion-depth-producer-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "color-depth-pairing-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        if bool(getattr(args, "require_foreground", False)):
            extra.append("--require-foreground")
        metadata_path = str(getattr(args, "metadata", "") or "").strip()
        if metadata_path:
            extra.extend(["--metadata", metadata_path])
        output_dir = str(getattr(args, "output_dir", "") or "").strip()
        if output_dir:
            extra.extend(["--output-dir", output_dir])
        return _print(
            core.run_launcher(
                "fh6-color-depth-pairing-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "openxr-presentation-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        if bool(getattr(args, "require_foreground", False)):
            extra.append("--require-foreground")
        if bool(getattr(args, "require_pairing", False)):
            extra.append("--require-pairing")
        if bool(getattr(args, "require_hmd", False)):
            extra.append("--require-hmd")
        min_hmd_width = getattr(args, "min_hmd_width", None)
        if min_hmd_width:
            extra.extend(["--min-hmd-width", str(min_hmd_width)])
        min_hmd_height = getattr(args, "min_hmd_height", None)
        if min_hmd_height:
            extra.extend(["--min-hmd-height", str(min_hmd_height)])
        if bool(getattr(args, "immersive_check", False)):
            extra.append("--immersive-check")
        frames = getattr(args, "frames", None)
        if frames:
            extra.extend(["--frames", str(frames)])
        return _print(
            core.run_launcher(
                "openxr-presentation-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "immersive-presentation-loop-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        seconds = getattr(args, "seconds", None)
        if seconds:
            extra.extend(["--seconds", str(seconds)])
        target_hz = getattr(args, "target_hz", None)
        if target_hz:
            extra.extend(["--target-hz", str(target_hz)])
        return _print(
            core.run_launcher(
                "immersive-presentation-loop-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "kofi-parity-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        loop_frames = getattr(args, "loop_frames", None)
        if loop_frames:
            extra.extend(["--loop-frames", str(loop_frames)])
        cockpit_seconds = getattr(args, "cockpit_seconds", None)
        if cockpit_seconds:
            extra.extend(["--cockpit-seconds", str(cockpit_seconds)])
        target_hz = getattr(args, "target_hz", None)
        if target_hz:
            extra.extend(["--target-hz", str(target_hz)])
        return _print(
            core.run_launcher(
                "kofi-parity-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "pcvr-management-selftest":
        extra = ["--json"]
        if bool(getattr(args, "allow_missing_runtime", False)):
            extra.append("--allow-missing-runtime")
        if bool(getattr(args, "no_process_list", False)):
            extra.append("--no-process-list")
        return _print(
            core.run_launcher(
                "pcvr-management-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "hermes-bridge-selftest":
        return _print(
            core.run_launcher(
                "hermes-bridge-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=["--json"],
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "hmd-controller-input-selftest":
        extra = ["--json"]
        if bool(getattr(args, "allow_missing_runtime", False)):
            extra.append("--allow-missing-runtime")
        if bool(getattr(args, "require_virtual_gamepad", False)):
            extra.append("--require-virtual-gamepad")
        if bool(getattr(args, "no_process_list", False)):
            extra.append("--no-process-list")
        return _print(
            core.run_launcher(
                "hmd-controller-input-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "vcc-health":
        return _print(
            core.scan_vcc_health(
                project_path=getattr(args, "project_path", "") or None,
            )
        )
    if command == "cockpit-presence-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        seconds = getattr(args, "seconds", None)
        if seconds:
            extra.extend(["--seconds", str(seconds)])
        target_hz = getattr(args, "target_hz", None)
        if target_hz:
            extra.extend(["--target-hz", str(target_hz)])
        return _print(
            core.run_launcher(
                "cockpit-presence-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
                timeout_seconds=getattr(args, "timeout_seconds", None),
            )
        )
    if command == "live-color-loop-selftest":
        extra = ["--json"]
        if bool(getattr(args, "approve", False)):
            extra.append("--approve")
        if bool(getattr(args, "attempt_window_capture", False)):
            extra.append("--attempt-window-capture")
        frames = getattr(args, "frames", None)
        if frames:
            extra.extend(["--frames", str(frames)])
        target_hz = getattr(args, "target_hz", None)
        if target_hz:
            extra.extend(["--target-hz", str(target_hz)])
        return _print(
            core.run_launcher(
                "live-color-loop-selftest",
                launcher_exe=getattr(args, "launcher_exe", "") or None,
                extra_args=extra,
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
