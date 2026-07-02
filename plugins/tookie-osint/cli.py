from __future__ import annotations

from typing import Any

from . import core


def _print(payload: dict[str, Any]) -> None:
    print(core.to_json(payload))


def register_cli(subparser) -> None:
    actions = subparser.add_subparsers(dest="tookie_osint_action")

    actions.add_parser("status", help="Show Tookie-OSINT readiness.")

    setup_parser = actions.add_parser("setup", help="Save the Tookie-OSINT checkout path.")
    setup_parser.add_argument("--root", required=True, help="Path to Alfredredbird/tookie-osint checkout.")
    setup_parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements.txt into the active Hermes Python environment.",
    )

    scan_parser = actions.add_parser("scan", help="Scan a public username.")
    scan_parser.add_argument("username")
    scan_parser.add_argument("-t", "--threads", type=int, default=4)
    scan_parser.add_argument(
        "-o",
        "--output-format",
        choices=["json", "csv", "txt"],
        default="json",
    )
    scan_parser.add_argument("-a", "--all", action="store_true", dest="include_all")
    scan_parser.add_argument("--skip-headers", action="store_true")
    scan_parser.add_argument("--webscraper", action="store_true")
    scan_parser.add_argument("--harvest", action="store_true")
    scan_parser.add_argument("--delay", type=int)
    scan_parser.add_argument("--timeout-seconds", type=int, default=core.DEFAULT_TIMEOUT_SECONDS)

    subparser.set_defaults(func=tookie_osint_command)


def tookie_osint_command(args: Any) -> int:
    action = getattr(args, "tookie_osint_action", None) or "status"
    if action == "status":
        _print(core.status_payload({}))
        return 0
    if action == "setup":
        try:
            root = core.save_root(args.root)
        except Exception as exc:
            _print({"success": False, "error": str(exc)})
            return 1
        payload = {"success": True, "root": str(root), "status": core.status_payload({})}
        if args.install_deps:
            payload["install_dependencies"] = core.install_dependencies(root)
            payload["status"] = core.status_payload({})
        _print(payload)
        return 0 if payload.get("success") else 1
    if action == "scan":
        payload = core.scan_username(
            {
                "username": args.username,
                "threads": args.threads,
                "output_format": args.output_format,
                "include_all": args.include_all,
                "skip_headers": args.skip_headers,
                "webscraper": args.webscraper,
                "harvest": args.harvest,
                "delay": args.delay,
                "timeout_seconds": args.timeout_seconds,
            }
        )
        _print(payload)
        return 0 if payload.get("success") else 1
    print("usage: hermes tookie-osint {status,setup,scan}")
    return 2
