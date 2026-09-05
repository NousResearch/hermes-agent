"""``hermes computer-use`` subcommand parser."""

from __future__ import annotations

import sys

from hermes_cli.subcommands._shared import add_json_flag


def _cu_install(args) -> int:
    from hermes_cli.tools_config import _cua_driver_contract_status, install_cua_driver
    if not install_cua_driver(upgrade=bool(getattr(args, "upgrade", False))):
        return 1
    return 0 if _cua_driver_contract_status().get("ready") else 1


def _cu_status(args) -> int:
    import os as _os
    import subprocess
    from hermes_cli.tools_config import _cua_driver_contract_status
    from tools.computer_use.cua_backend_driver import cua_driver_update_check, resolve_cua_driver_cmd
    # Must match the runtime resolver: Desktop/TUI processes can omit
    # ~/.local/bin even though the official installer put the driver there.
    path = resolve_cua_driver_cmd()
    override = _os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip()
    if not path:
        print("cua-driver: not installed")
        print("  Run: hermes computer-use install")
        return 1
    version = ""
    try:
        from hermes_cli.tools_config import _cua_driver_env
        version = subprocess.run(
            [path, "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            env=_cua_driver_env(),
        ).stdout.strip()
    except Exception:
        pass
    from hermes_cli.tools_config import _cua_version_summary
    version = _cua_version_summary(version)
    # Name the override here too. Without it the operator is told to repair an
    # install that `hermes computer-use install` will (correctly) refuse to touch,
    # with nothing pointing at the env var that actually selected the binary.
    origin = " [custom binary from HERMES_CUA_DRIVER_CMD]" if override else ""
    print(f"cua-driver: installed at {path}{origin}" + (f" ({version})" if version else ""))
    contract = _cua_driver_contract_status(path)
    if not contract.get("ready"):
        print("  ⚠ Repair required: " + (contract.get("reason") or "runtime contract is incomplete"))
        if override:
            print(
                "    Update the binary selected by HERMES_CUA_DRIVER_CMD, or unset "
                "the override and run: hermes computer-use install --upgrade")
        else:
            print("    Run: hermes computer-use install")
        return 1
    try:
        st = cua_driver_update_check()
        if st and st.get("update_available"):
            latest = st.get("latest_version") or "?"
            print(f"  ⬆ Update available: cua-driver {latest}.")
            print("    Run: hermes computer-use install --upgrade")
        elif st:
            print("  ✓ Up to date.")
        else:
            # Older driver (no check-update verb) or offline.
            print("  Refresh to latest: hermes computer-use install --upgrade")
    except Exception:
        print("  Refresh to latest: hermes computer-use install --upgrade")
    return 0


def _cu_doctor(args) -> None:
    from tools.computer_use.doctor import run_doctor
    sys.exit(run_doctor(
        include=list(getattr(args, "include", []) or []),
        skip=list(getattr(args, "skip", []) or []),
        json_output=bool(getattr(args, "json", False))))


def _cu_perms_status(args) -> None:
    import json as _json
    from tools.computer_use.permissions import computer_use_status
    st = computer_use_status()
    if bool(getattr(args, "json", False)):
        print(_json.dumps(st, indent=2, sort_keys=True))
        sys.exit(0 if st["ready"] else 1)
    if not st["platform_supported"]:
        print(f"Computer Use is not supported on {st['platform']}.")
        sys.exit(1)
    if not st["installed"]:
        print("cua-driver: not installed. Run: hermes computer-use install")
        sys.exit(1)
    glyph = lambda v: "✅" if v is True else ("❌" if v is False else "•")  # noqa: E731
    print(f"cua-driver: {st['version'] or 'installed'} ({st['platform']})")
    if st["can_grant"]:  # macOS TCC permissions
        print(f"  {glyph(st['accessibility'])} Accessibility")
        print(f"  {glyph(st['screen_recording'])} Screen Recording")
        if not st["ready"]:
            print("  Grant: hermes computer-use permissions grant")
    else:  # no TCC model — readiness is driver health
        print(f"  {glyph(st['ready'])} driver health (no permission toggles on {st['platform']})")
    for c in st["checks"]:
        if c["status"] != "ok":
            print(f"  ⚠ {c['label']}: {c['message']}")
    if st["error"]:
        print(f"  ⚠ {st['error']}")
    sys.exit(0 if st["ready"] else 1)


def _cu_perms_grant(args) -> None:
    from tools.computer_use.permissions import request_permissions_grant
    sys.exit(request_permissions_grant())


def _cu_host_bridge(args) -> int:
    from tools.computer_use.host_bridge_cli import run_host_bridge
    # Only the kwargs run_host_bridge defines today; new bridge options default host-side.
    run_host_bridge(allowed_hosts=args.allowed_hosts.split(","),
                    allowed_origins=(args.allowed_origins.split(",") if args.allowed_origins else []),
                    port=args.port, bind=args.bind)
    return 0


def build_computer_use_parser(subparsers) -> None:
    """Attach the ``computer-use`` subcommand to ``subparsers``."""
    computer_use_parser = subparsers.add_parser(
        "computer-use", help="Manage the Computer Use (cua-driver) backend (macOS/Windows/Linux)",
        description="Install or check the cua-driver binary used by the\n"
            "`computer_use` toolset. Supported on macOS, Windows, and\n"
            "Linux.\n\n"
            "Use `hermes computer-use install` to fetch and run the\n"
            "upstream cua-driver installer. This is equivalent to the\n"
            "post-setup hook that `hermes tools` runs when you first\n"
            "enable the Computer Use toolset, and is a stable target\n"
            "for re-running the install if it didn't fire (e.g. when\n"
            "toggling the toolset on a returning-user setup).\n\n"
            "Use `hermes computer-use doctor` to run cua-driver's\n"
            "`health_report` MCP tool and surface its check matrix\n"
            "(TCC, bundle identity, version, platform support, ...)\n"
            "in human-readable form.")
    computer_use_sub = computer_use_parser.add_subparsers(dest="computer_use_action")

    computer_use_install = computer_use_sub.add_parser(
        "install", help="Install or repair the cua-driver binary (macOS/Windows/Linux)")
    computer_use_install.add_argument(
        "--upgrade", action="store_true",
        help="Re-run the upstream installer even if cua-driver is already on "
            "PATH. The upstream install.sh always pulls the latest release, "
            "so this performs an in-place upgrade.")
    computer_use_sub.add_parser("status", help="Print whether cua-driver is installed and on PATH")
    computer_use_bridge = computer_use_sub.add_parser(
        "host-bridge", help="Run the CUA host bridge on this machine (drive this desktop remotely)",
        description="Expose this machine's cua-driver over an authenticated streamable-HTTP\n"
            "bridge so a remote Hermes agent (computer_use.provider=remote + remote.url +\n"
            "HERMES_CUA_REMOTE_TOKEN) can drive this desktop. --allowed-hosts is the mandatory\n"
            "Host allowlist — include the port (e.g. host:8765), the Host header carries it and\n"
            "comparison is exact; an empty allowlist would let any page drive the desktop, so\n"
            "the bridge refuses to start without it. Pass --bind 0.0.0.0 only when the remote\n"
            "agent reaches this host over a trusted network.")
    computer_use_bridge.add_argument("--port", type=int, default=8765,
        help="TCP port to listen on")
    computer_use_bridge.add_argument("--bind", default="127.0.0.1",
        help="Address to bind (default 127.0.0.1; use 0.0.0.0 for remote access)")
    computer_use_bridge.add_argument("--allowed-hosts", required=True,
        help="Comma-separated allowed Host headers (required; include the port, e.g. host:8765)")
    computer_use_bridge.add_argument("--allowed-origins", default="",
        help="Comma-separated extra allowed Origins (default: none beyond --allowed-hosts)")
    computer_use_bridge.add_argument("--session-idle-timeout", type=int, default=1800,
        help="Seconds an idle bridge session is kept before teardown")
    computer_use_doctor = computer_use_sub.add_parser(
        "doctor", help="Run cua-driver `health_report` and surface the check matrix",
        description="Drive cua-driver's stable `health_report` MCP tool and render\n"
            "its check matrix (TCC permissions, bundle identity, version,\n"
            "platform support, screenshot probe, …) as human-readable\n"
            "output. cua-driver owns the health model; this command stays\n"
            "thin so new checks added upstream surface here without code\n"
            "changes. Exits 0 when overall=ok, 1 when degraded/failed, 2\n"
            "when the binary is missing or unreachable.")
    computer_use_doctor.add_argument(
        "--include", action="append", default=[], metavar="CHECK",
        help="Run only the listed checks. Repeat for multiple "
            "(e.g. --include tcc_accessibility --include bundle_identity). "
            "Unknown names are reported by cua-driver.")
    computer_use_doctor.add_argument(
        "--skip", action="append", default=[], metavar="CHECK",
        help="Skip the listed checks. Repeat for multiple. Wins over --include.")
    add_json_flag(computer_use_doctor,
                  "Emit the raw structured payload as JSON (same shape as `tools/call`).")
    computer_use_perms = computer_use_sub.add_parser(
        "permissions", help="Check or grant macOS Accessibility + Screen Recording (macOS)",
        description="Computer Use drives the Mac through cua-driver, whose TCC grants\n"
            "attach to cua-driver's own identity (com.trycua.driver) — not the\n"
            "terminal or the Hermes app. `status` reports the driver's grant\n"
            "state; `grant` launches CuaDriver via LaunchServices so the macOS\n"
            "permission dialog is attributed to the process that does the work.")
    computer_use_perms_sub = computer_use_perms.add_subparsers(dest="computer_use_perms_action")
    computer_use_perms_status = computer_use_perms_sub.add_parser(
        "status", help="Report Accessibility + Screen Recording grant state (read-only)")
    add_json_flag(computer_use_perms_status, "Emit the normalized permission payload as JSON.")
    computer_use_perms_sub.add_parser(
        "grant", help="Request the grants (opens the dialog attributed to CuaDriver)")
    _perms_actions = {"grant": _cu_perms_grant, "status": _cu_perms_status}

    def _cu_permissions(args):
        handler = _perms_actions.get(getattr(args, "computer_use_perms_action", None))
        if handler is not None:
            return handler(args)
        computer_use_perms.print_help()

    _actions = {"install": _cu_install, "status": _cu_status, "doctor": _cu_doctor,
                "permissions": _cu_permissions, "host-bridge": _cu_host_bridge}

    def cmd_computer_use(args):
        handler = _actions.get(getattr(args, "computer_use_action", None))
        if handler is not None:
            return handler(args)
        computer_use_parser.print_help()  # no subcommand → help

    computer_use_parser.set_defaults(func=cmd_computer_use)
