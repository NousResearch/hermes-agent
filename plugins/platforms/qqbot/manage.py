"""Explicit QQ menu/panel operations; mutations preview offline unless --apply."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

from .management import open_management_client, panel_path


@dataclass(frozen=True)
class RequestPlan:
    method: str
    path: str
    payload: dict | None = None


def setup_parser(parser):
    commands = parser.add_subparsers(dest="qqbot_operation", required=True)
    commands.add_parser("menu-get", help="Read the global C2C menu")
    listing = commands.add_parser("panel-list", help="Read one page of panels")
    listing.add_argument("--scope", required=True)
    listing.add_argument("--cursor")
    listing.add_argument("--limit", type=int)
    for name in ("menu-put", "panel-create", "panel-get", "panel-update", "panel-targets", "panel-delete"):
        command = commands.add_parser(name)
        if name.startswith("panel-") and name != "panel-create":
            command.add_argument("panel_id")
        if name != "panel-get":
            command.add_argument("--apply", action="store_true", help="Execute the remote write (default: offline preview)")
            if name != "panel-delete":
                command.add_argument("--file", type=Path, required=True, help="UTF-8 JSON object with complete QQ request body")


def build_plan(args):
    payload = None
    if getattr(args, "file", None) is not None:
        try:
            payload = json.loads(args.file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            raise ValueError("--file must be a readable UTF-8 JSON object") from None
        if not isinstance(payload, dict):
            raise ValueError("--file must contain a JSON object")
    params = {key: getattr(args, key) for key in ("scope", "cursor", "limit") if getattr(args, key, None) is not None}
    path = panel_path(args.panel_id) if hasattr(args, "panel_id") else ""
    specs = {
        "menu-get": ("GET", "/v2/menu"),
        "menu-put": ("PUT", "/v2/menu"),
        "panel-list": ("GET", "/v2/panels?" + urlencode(params)),
        "panel-create": ("POST", "/v2/panels"),
        "panel-get": ("GET", path),
        "panel-update": ("PUT", path),
        "panel-targets": ("PUT", path + "/target"),
        "panel-delete": ("DELETE", path),
    }
    return RequestPlan(*specs[args.qqbot_operation], payload)


async def execute_plan(plan, client=None):
    if client is not None:
        return await client.request(plan.method, plan.path, plan.payload)
    async with open_management_client() as owned:
        return await owned.request(plan.method, plan.path, plan.payload)


def run_args(args, *, client=None) -> int:
    try:
        plan = build_plan(args)
    except ValueError as exc:
        print(f"qqbot: {exc}", file=sys.stderr)
        return 2
    if plan.method != "GET" and not args.apply:
        print(json.dumps({"dry_run": True, "method": plan.method, "path": plan.path,
                          "payload": plan.payload}, ensure_ascii=False, indent=2))
        return 0
    import httpx

    try:
        result = asyncio.run(execute_plan(plan, client))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError):
        # Remote bodies and transport errors may embed credentials; do not echo them.
        print("qqbot: request failed; check configuration, permissions and connectivity", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv=None, *, client=None):
    parser = argparse.ArgumentParser(description=__doc__)
    setup_parser(parser)
    return run_args(parser.parse_args(argv), client=client)


if __name__ == "__main__":
    raise SystemExit(main())
