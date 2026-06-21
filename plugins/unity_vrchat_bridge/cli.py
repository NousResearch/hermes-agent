from __future__ import annotations

from . import core


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "unity-vrchat-bridge",
        help="Unity/VRChat Editor bridge diagnostics",
    )
    parser.add_argument("args", nargs="*")


def unity_vrchat_bridge_command(args) -> int:
    argv = list(getattr(args, "args", []) or [])
    print(core.run_cli(argv))
    return 0
