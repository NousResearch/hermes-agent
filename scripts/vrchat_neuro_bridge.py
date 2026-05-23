"""Run a Neuro API websocket bridge into Hermes VRChat safety gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.neuro_bridge import (  # noqa: E402
    DEFAULT_GAME_NAME,
    build_neuro_bridge_bootstrap,
    handle_neuro_action_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge VedalAI Neuro API websocket actions into Hermes VRChat safety gates."
    )
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8000",
        help="Neuro API websocket URL. Default: ws://127.0.0.1:8000",
    )
    parser.add_argument(
        "--game",
        default=DEFAULT_GAME_NAME,
        help=f"Neuro API game name. Default: {DEFAULT_GAME_NAME}",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Optional VRChat autonomy profile JSON path.",
    )
    parser.add_argument(
        "--context",
        default="Hermes VRChat bridge is connected. Actions are validated locally before any VRChat output.",
        help="Optional initial Neuro context message.",
    )
    parser.add_argument(
        "--visible-context",
        action="store_true",
        help="Send the startup context as visible rather than silent.",
    )
    parser.add_argument(
        "--retry-on-failure",
        action="store_true",
        help="Return action/result success=false on local rejection so Neuro may retry.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Handle one incoming action and exit.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        import websockets
    except ImportError:
        print(
            "websockets is required for scripts/vrchat_neuro_bridge.py. "
            "Install the project messaging extra or install websockets==15.0.1.",
            file=sys.stderr,
        )
        return 2

    profile_path = args.profile or None
    bootstrap = build_neuro_bridge_bootstrap(
        game=args.game,
        profile_path=profile_path,
        context=args.context,
        silent_context=not args.visible_context,
    )
    if not bootstrap["vendor"]["success"]:
        print(
            "Warning: vendor/neuro-sdk API files were not found; continuing with local protocol helpers.",
            file=sys.stderr,
        )

    async with websockets.connect(args.ws_url) as websocket:
        for message in bootstrap["messages"]:
            await websocket.send(json.dumps(message, ensure_ascii=False))
            print(json.dumps({"sent": message["command"], "game": message.get("game")}, ensure_ascii=False))

        while True:
            raw_message = await websocket.recv()
            if not isinstance(raw_message, str):
                print(json.dumps({"ignored": "binary_message"}, ensure_ascii=False))
                continue
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                print(json.dumps({"ignored": "invalid_json"}, ensure_ascii=False))
                continue
            if message.get("command") != "action":
                print(json.dumps({"ignored": message.get("command", "unknown")}, ensure_ascii=False))
                continue

            result = handle_neuro_action_message(
                message,
                profile_path=profile_path,
                game=args.game,
                retry_on_failure=args.retry_on_failure,
            )
            await websocket.send(json.dumps(result["action_result"], ensure_ascii=False))
            print(
                json.dumps(
                    {
                        "handled": result.get("action_name"),
                        "success": result.get("success"),
                        "dry_run": (result.get("turn") or {}).get("dry_run"),
                        "result_success": result["action_result"]["data"]["success"],
                    },
                    ensure_ascii=False,
                )
            )
            if args.once:
                return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
