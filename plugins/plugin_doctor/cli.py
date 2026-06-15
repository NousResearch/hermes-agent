from __future__ import annotations

from typing import Any

from . import core


def register_cli(subparser) -> None:
    subparser.add_argument("--plugins-dir", default=core.DEFAULT_PLUGINS_DIR)
    subparser.add_argument("--no-import-check", action="store_true")
    subparser.set_defaults(func=plugin_doctor_command)


def plugin_doctor_command(args: Any) -> int:
    payload = core.scan_plugins(
        {
            "plugins_dir": args.plugins_dir,
            "include_import_check": not args.no_import_check,
        }
    )
    print(core.to_json(payload))
    return 0 if payload.get("ok") else 1
