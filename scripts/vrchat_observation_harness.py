"""Queue VRChat multimodal observations for Hermes autonomy loops."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.openclaw.vrchat_autonomy import load_autonomy_profile, vrchat_autonomy_profile_tick  # noqa: E402
from tools.openclaw.vrchat_observations import (  # noqa: E402
    build_observation_from_osc,
    ingest_observations,
    parse_jsonl_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue VRChat ChatBox, STT, vision, stream, and operator observations for Hermes."
    )
    parser.add_argument("--queue", default="", help="Optional observation queue JSONL path.")
    parser.add_argument("--profile", default="", help="Optional VRChat autonomy profile JSON path.")
    parser.add_argument("--stdin-jsonl", action="store_true", help="Read observation events from stdin as JSONL.")
    parser.add_argument("--listen-osc", action="store_true", help="Listen for incoming VRChat OSC events.")
    parser.add_argument("--osc-host", default="127.0.0.1", help="OSC listen host. Default: 127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9001, help="OSC listen port. Default: 9001")
    parser.add_argument(
        "--allow-avatar-parameters",
        action="store_true",
        help="Queue avatar parameter OSC changes as system observations.",
    )
    parser.add_argument(
        "--tick-profile",
        action="store_true",
        help="Run one profile tick after each accepted observation batch.",
    )
    parser.add_argument(
        "--allow-live-profile",
        action="store_true",
        help="Allow --tick-profile even when the profile has dry_run=false.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    if not args.stdin_jsonl and not args.listen_osc:
        print("Choose --stdin-jsonl, --listen-osc, or both.", file=sys.stderr)
        return 2

    tasks: list[asyncio.Task] = []
    if args.stdin_jsonl:
        tasks.append(asyncio.create_task(_read_stdin(args)))
    if args.listen_osc:
        tasks.append(asyncio.create_task(_listen_osc(args)))
    await asyncio.gather(*tasks)
    return 0


async def _read_stdin(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        parsed = parse_jsonl_observation(line)
        if not parsed["success"]:
            print(json.dumps({"accepted": False, "error": parsed["error"]}, ensure_ascii=False))
            continue
        result = await _queue_and_maybe_tick([parsed["observation"]], args)
        print(json.dumps(result, ensure_ascii=False))


async def _listen_osc(args: argparse.Namespace) -> None:
    try:
        from pythonosc import dispatcher, osc_server
    except ImportError:
        print("python-osc is required for --listen-osc. Install hermes-agent[vrchat].", file=sys.stderr)
        return

    loop = asyncio.get_running_loop()
    dispatch = dispatcher.Dispatcher()

    def handle(address: str, *values) -> None:
        converted = build_observation_from_osc(
            address,
            list(values),
            allow_avatar_parameters=args.allow_avatar_parameters,
        )
        if not converted["success"]:
            print(json.dumps({"accepted": False, "ignored": converted["ignored"]}, ensure_ascii=False))
            return
        asyncio.run_coroutine_threadsafe(_print_queued(converted["observation"], args), loop)

    dispatch.map("/chatbox/input", handle)
    dispatch.map("/avatar/parameters/*", handle)
    server = osc_server.AsyncIOOSCUDPServer((args.osc_host, args.osc_port), dispatch, loop)
    transport, _protocol = await server.create_serve_endpoint()
    print(json.dumps({"listening": True, "host": args.osc_host, "port": args.osc_port}, ensure_ascii=False))
    try:
        await asyncio.Event().wait()
    finally:
        transport.close()


async def _print_queued(observation: dict, args: argparse.Namespace) -> None:
    result = await _queue_and_maybe_tick([observation], args)
    print(json.dumps(result, ensure_ascii=False))


async def _queue_and_maybe_tick(observations: list[dict], args: argparse.Namespace) -> dict:
    result = ingest_observations(
        observations,
        queue_path=args.queue or None,
        persist=True,
    )
    tick = None
    if result["queued"] and args.tick_profile:
        loaded = load_autonomy_profile(args.profile or None)
        profile = loaded.get("profile", {})
        if not bool(profile.get("dry_run", True)) and not args.allow_live_profile:
            tick = {
                "success": False,
                "code": "LIVE_PROFILE_REQUIRES_ALLOW_FLAG",
                "message": "--allow-live-profile is required when profile dry_run=false.",
            }
        else:
            tick = vrchat_autonomy_profile_tick(profile_path=args.profile or None)
    return {"success": result["success"], "ingest": result, "tick": tick}


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
