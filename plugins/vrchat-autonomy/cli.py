"""CLI for the vrchat-autonomy Hermes plugin."""

from __future__ import annotations

import argparse

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="vrchat_autonomy_command")

    subs.add_parser("status", help="Profile, readiness, and worker state")
    subs.add_parser("doctor", help="Preflight + readiness bundle")

    setup = subs.add_parser("setup", help="Enable plugin and write operator profile")
    setup.add_argument("--arm-live", action="store_true", help="Arm live OSC actuation with ACK")
    setup.add_argument("--no-movement", action="store_true", help="Keep allow_movement=false")
    setup.add_argument("--no-chatbox", action="store_true", help="Keep allow_chatbox=false")
    setup.add_argument("--mode", default="private_test", help="Profile mode (default private_test)")

    arm = subs.add_parser(
        "arm-live",
        help="Set dry_run=false and write the exact live actuation ACK",
    )
    arm.add_argument("--no-movement", action="store_true")
    arm.add_argument("--no-chatbox", action="store_true")
    arm.add_argument("--mode", default="private_test")

    chatbox = subs.add_parser("chatbox", help="Send one ChatBox message (live profile required)")
    chatbox.add_argument("text", help="Message text")
    chatbox.add_argument("--keyboard", action="store_true", help="Route via keyboard UI (immediate=false)")

    move = subs.add_parser("move", help="Pulse movement input (live profile + allow_movement)")
    move.add_argument(
        "direction",
        choices=["forward", "back", "left", "right", "jump", "run", "stop"],
    )
    move.add_argument("--value", type=float, default=1.0)
    move.add_argument("--duration-ms", type=int, default=400)

    tick = subs.add_parser("tick", help="Run one autonomy loop tick")
    tick.add_argument("--emergency-stop", action="store_true")

    subs.add_parser("start", help="Start background autonomy worker")
    subs.add_parser("stop", help="Stop background worker and emergency-stop loop state")

    loop = subs.add_parser("loop", help="Run foreground tick loop (Ctrl+C to exit)")
    loop.add_argument("--interval", type=float, default=15.0)

    neuro = subs.add_parser("neuro", help="VedalAI Neuro API / neuro-sdk bridge helpers")
    neuro_subs = neuro.add_subparsers(dest="neuro_command")
    neuro_subs.add_parser("status", help="Vendor clone, profile, and action catalog")
    neuro_subs.add_parser("vendor", help="neuro-sdk submodule/vendor clone status and init hint")
    bootstrap = neuro_subs.add_parser("bootstrap", help="Startup/context/actions/register messages")
    bootstrap.add_argument("--context", default="", help="Optional initial context")
    bootstrap.add_argument("--visible-context", action="store_true", help="Send context as visible")
    build = neuro_subs.add_parser("build-messages", help="Bootstrap plus optional actions/force")
    build.add_argument("--context", default="")
    build.add_argument("--visible-context", action="store_true")
    build.add_argument("--force-query", default="")
    build.add_argument("--force-action", action="append", dest="force_actions", default=[])
    build.add_argument("--force-state", default="")
    build.add_argument(
        "--force-priority",
        default="low",
        choices=["low", "medium", "high", "critical"],
    )
    handle = neuro_subs.add_parser("handle", help="Handle one Neuro action JSON from --message")
    handle.add_argument("--message", required=True, help="JSON Neuro action message")
    handle.add_argument("--retry-on-failure", action="store_true")
    handle.add_argument("--force-dry-run", action="store_true")
    bridge = neuro_subs.add_parser(
        "bridge",
        help="Run websocket bridge (requires websockets; see scripts/vrchat_neuro_bridge.py)",
    )
    bridge.add_argument("--ws-url", default="")
    bridge.add_argument("--context", default="")
    bridge.add_argument("--visible-context", action="store_true")
    bridge.add_argument("--retry-on-failure", action="store_true")
    bridge.add_argument("--once", action="store_true")

    subparser.set_defaults(func=vrchat_autonomy_command)


def vrchat_autonomy_command(args: argparse.Namespace) -> int:
    command = getattr(args, "vrchat_autonomy_command", None)
    if not command:
        print("usage: hermes vrchat-autonomy {status,doctor,setup,chatbox,move,tick,start,stop,loop}")
        return 2

    if command == "status":
        payload = core.status()
    elif command == "doctor":
        payload = core.doctor()
    elif command == "setup":
        payload = core.setup(
            allow_chatbox=not args.no_chatbox,
            allow_movement=not args.no_movement,
            mode=args.mode,
            arm_live=bool(args.arm_live),
        )
    elif command == "arm-live":
        payload = core.arm_live_profile(
            allow_chatbox=not args.no_chatbox,
            allow_movement=not args.no_movement,
            mode=args.mode,
        )
    elif command == "chatbox":
        payload = core.send_chatbox(args.text, immediate=not args.keyboard)
    elif command == "move":
        payload = core.move(args.direction, value=args.value, duration_ms=args.duration_ms)
    elif command == "tick":
        payload = core.run_tick(emergency_stop=bool(args.emergency_stop))
    elif command == "start":
        payload = core.start_worker()
    elif command == "stop":
        payload = core.stop_worker()
    elif command == "loop":
        import time

        interval = max(5.0, min(float(args.interval), 300.0))
        print(core.to_json({"ok": True, "message": "foreground loop — Ctrl+C to stop", "interval_sec": interval}))
        try:
            while True:
                print(core.to_json(core.run_tick()))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
    elif command == "neuro":
        from . import neuro as neuro_sdk

        cfg = core.plugin_config()
        prof = core.profile_path(cfg)
        neuro_cmd = getattr(args, "neuro_command", None)
        if not neuro_cmd:
            print("usage: hermes vrchat-autonomy neuro {status,bootstrap,build-messages,handle,bridge}")
            return 2
        if neuro_cmd == "status":
            payload = neuro_sdk.neuro_status(profile=prof, config=cfg)
        elif neuro_cmd == "vendor":
            payload = neuro_sdk.neuro_vendor_status(config=cfg)
        elif neuro_cmd == "bootstrap":
            payload = neuro_sdk.neuro_bootstrap(
                profile=prof,
                config=cfg,
                context=args.context or "",
                silent_context=not args.visible_context,
            )
        elif neuro_cmd == "build-messages":
            payload = neuro_sdk.neuro_build_messages(
                profile=prof,
                config=cfg,
                context=args.context or "",
                silent_context=not args.visible_context,
                force_action_names=list(args.force_actions or []),
                force_query=args.force_query or "",
                force_state=args.force_state or "",
                force_priority=args.force_priority or "low",
            )
        elif neuro_cmd == "handle":
            import json

            try:
                message = json.loads(args.message)
            except json.JSONDecodeError as exc:
                print(core.to_json({"ok": False, "error": f"invalid_json: {exc}"}))
                return 1
            payload = neuro_sdk.neuro_handle_action(
                message,
                profile=prof,
                config=cfg,
                retry_on_failure=bool(args.retry_on_failure),
                force_dry_run=bool(args.force_dry_run),
            )
        elif neuro_cmd == "bridge":
            import subprocess
            import sys
            from pathlib import Path

            script = Path(__file__).resolve().parents[2] / "scripts" / "vrchat_neuro_bridge.py"
            cmd = [sys.executable, str(script), "--profile", str(prof)]
            ws_url = (args.ws_url or "").strip() or neuro_sdk.resolve_ws_url(cfg)
            cmd.extend(["--ws-url", ws_url, "--game", neuro_sdk.resolve_game_name(cfg)])
            if args.context:
                cmd.extend(["--context", args.context])
            if args.visible_context:
                cmd.append("--visible-context")
            if args.retry_on_failure:
                cmd.append("--retry-on-failure")
            if args.once:
                cmd.append("--once")
            return subprocess.call(cmd)
        else:
            print(f"unknown neuro subcommand: {neuro_cmd}")
            return 2
    else:
        print(f"unknown subcommand: {command}")
        return 2

    print(core.to_json(payload))
    return 0 if payload.get("ok", True) else 1
