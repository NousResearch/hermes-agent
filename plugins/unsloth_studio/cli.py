from __future__ import annotations

from typing import Any

from . import core


def _print(payload: dict[str, Any]) -> None:
    print(core.to_json(payload))


def register_cli(subparser) -> None:
    actions = subparser.add_subparsers(dest="unsloth_studio_action")

    status_parser = actions.add_parser("status", help="Show Unsloth Studio status.")
    status_parser.add_argument("--host", default=core.DEFAULT_HOST)
    status_parser.add_argument("--port", type=int, default=core.DEFAULT_PORT)
    status_parser.add_argument("--no-probe-url", action="store_true")

    start_parser = actions.add_parser("start", help="Start Unsloth Studio.")
    start_parser.add_argument("--host", default=core.DEFAULT_HOST)
    start_parser.add_argument("--port", type=int, default=core.DEFAULT_PORT)
    start_parser.add_argument("--wait-seconds", type=float, default=core.DEFAULT_WAIT_SECONDS)
    start_parser.add_argument("--cwd")
    start_parser.add_argument("--confirm-public-host", action="store_true")
    start_parser.add_argument("extra_args", nargs="*")

    stop_parser = actions.add_parser("stop", help="Stop the recorded Unsloth Studio process.")
    stop_parser.add_argument("--pid", type=int)

    info_parser = actions.add_parser("install-info", help="Show official install commands.")
    info_parser.add_argument("--local-only", action="store_true")

    subparser.set_defaults(func=unsloth_studio_command)


def unsloth_studio_command(args: Any) -> int:
    action = getattr(args, "unsloth_studio_action", None)
    if action == "status":
        _print(
            core.status_payload(
                {
                    "host": args.host,
                    "port": args.port,
                    "probe_url": not args.no_probe_url,
                }
            )
        )
        return 0
    if action == "start":
        _print(
            core.start_studio(
                {
                    "host": args.host,
                    "port": args.port,
                    "wait_seconds": args.wait_seconds,
                    "cwd": args.cwd,
                    "extra_args": args.extra_args,
                    "confirm_public_host": args.confirm_public_host,
                }
            )
        )
        return 0
    if action == "stop":
        _print(core.stop_studio({"pid": args.pid}))
        return 0
    if action == "install-info":
        _print(core.install_info({"local_only": args.local_only}))
        return 0
    print("usage: hermes unsloth-studio {status,start,stop,install-info}")
    return 2
