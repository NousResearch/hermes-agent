"""CLI command for the AITuberKit Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import dev_server


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="aituber_kit_command")

    configure = subs.add_parser("configure", aliases=["setup"], help="Save AITuberKit bridge settings")
    configure.add_argument("--repo-root", default="")
    configure.add_argument("--port", type=int, default=None)
    configure.add_argument("--bridge-port", type=int, default=None)
    configure.add_argument("--client-id", default="")
    configure.add_argument("--base-url", default="")
    configure.add_argument("--external-linkage-url", default="")
    configure.add_argument("--message-receiver-enabled", action="store_true", default=None)
    configure.add_argument("--no-message-receiver", action="store_true")
    configure.add_argument("--external-linkage-enabled", action="store_true", default=None)
    configure.add_argument("--system-prompt", default="")
    configure.add_argument("--api-key-env", default="")

    subs.add_parser("status", help="Show AITuberKit readiness")

    install = subs.add_parser("install", help="Clone upstream and npm install")
    install.add_argument("--repo-root", default="")
    install.add_argument("--ref", default="main")
    install.add_argument("--force", action="store_true")
    install.add_argument("--skip-npm-install", action="store_true")

    prepare = subs.add_parser("prepare", help="Write .env.local and enable API flags")
    prepare.add_argument("--repo-root", default="")
    prepare.add_argument("--client-id", default="")
    prepare.add_argument("--message-receiver-enabled", action="store_true", default=None)
    prepare.add_argument("--no-message-receiver", action="store_true")
    prepare.add_argument("--external-linkage-enabled", action="store_true", default=None)
    prepare.add_argument("--external-linkage-url", default="")

    prepare.add_argument("--tailscale", action="store_true", help="Point External Linkage URL at Tailscale IP")

    start = subs.add_parser("start", help="Start Next.js dev server")
    start.add_argument("--repo-root", default="")
    start.add_argument("--port", type=int, default=None)
    start.add_argument("--host", default="", help="Client-facing URL host (Tailscale IP / MagicDNS)")
    start.add_argument("--bind", default="", help="Listen address (0.0.0.0 for tailnet)")
    start.add_argument("--tailscale", action="store_true", help="Bind 0.0.0.0 and report Tailscale URL")
    start.add_argument("--tailscale-serve", action="store_true", help="Register tailscale serve /aituber-kit")
    start.add_argument("--wait-seconds", type=float, default=60.0)

    stop = subs.add_parser("stop", help="Stop managed dev server")
    stop.add_argument("--force", action="store_true")
    stop.add_argument("--pid", type=int, default=None)

    speak = subs.add_parser("speak", help="POST /api/v1/speak")
    speak.add_argument("text", nargs="*")
    speak.add_argument("--client-id", default="")
    speak.add_argument("--emotion", default="")
    speak.add_argument("--priority", choices=["normal", "high"], default="")
    speak.add_argument("--interrupt", action="store_true")

    chat = subs.add_parser("chat", help="POST /api/v1/chat")
    chat.add_argument("text", nargs="*")
    chat.add_argument("--client-id", default="")
    chat.add_argument("--mode", choices=["user_input", "ai_generate"], default="user_input")
    chat.add_argument("--interrupt", action="store_true")

    stop_playback = subs.add_parser("stop-playback", aliases=["stop-speech"], help="POST /api/v1/stop")
    stop_playback.add_argument("--client-id", default="")
    stop_playback.add_argument("--mode", choices=["speech", "queue", "all"], default="all")

    bridge_start = subs.add_parser("bridge-start", aliases=["bridge"], help="Start External Linkage WS bridge")
    bridge_start.add_argument("--host", default="", help="Listen host (default 0.0.0.0 with --tailscale)")
    bridge_start.add_argument("--port", type=int, default=None)
    bridge_start.add_argument("--tailscale", action="store_true", help="Bind bridge on 0.0.0.0 for tailnet")
    bridge_start.add_argument("--confirm-public-host", action="store_true")
    bridge_start.add_argument("--system-prompt", default="")

    bridge_stop = subs.add_parser("bridge-stop", help="Stop External Linkage WS bridge")
    bridge_stop.add_argument("--force", action="store_true")

    subparser.set_defaults(func=aituber_kit_command)


def aituber_kit_command(args: argparse.Namespace) -> int:
    command = getattr(args, "aituber_kit_command", None)
    if not command:
        print(
            "usage: hermes aituber-kit "
            "{configure,status,install,prepare,start,stop,speak,chat,stop-playback,bridge-start,bridge-stop}"
        )
        return 2

    if command in {"configure", "setup"}:
        message_receiver = None
        if getattr(args, "message_receiver_enabled", None):
            message_receiver = True
        elif getattr(args, "no_message_receiver", False):
            message_receiver = False
        return _print(
            core.save_config({
                "repo_root": getattr(args, "repo_root", ""),
                "port": getattr(args, "port", None),
                "bridge_port": getattr(args, "bridge_port", None),
                "client_id": getattr(args, "client_id", ""),
                "base_url": getattr(args, "base_url", ""),
                "external_linkage_url": getattr(args, "external_linkage_url", ""),
                "message_receiver_enabled": message_receiver,
                "external_linkage_enabled": getattr(args, "external_linkage_enabled", None),
                "system_prompt": getattr(args, "system_prompt", ""),
                "api_key_env": getattr(args, "api_key_env", ""),
            })
        )
    if command == "status":
        return _print(core.status())
    if command == "install":
        return _print(
            core.install({
                "repo_root": getattr(args, "repo_root", ""),
                "ref": getattr(args, "ref", "main"),
                "force": getattr(args, "force", False),
                "skip_npm_install": getattr(args, "skip_npm_install", False),
            })
        )
    if command == "prepare":
        message_receiver = None
        if getattr(args, "message_receiver_enabled", None):
            message_receiver = True
        elif getattr(args, "no_message_receiver", False):
            message_receiver = False
        return _print(
            core.prepare({
                "repo_root": getattr(args, "repo_root", ""),
                "client_id": getattr(args, "client_id", ""),
                "message_receiver_enabled": message_receiver,
                "external_linkage_enabled": getattr(args, "external_linkage_enabled", None),
                "external_linkage_url": getattr(args, "external_linkage_url", ""),
                "use_tailscale": getattr(args, "tailscale", False),
            })
        )
    if command == "start":
        repo = core.resolve_repo_root(getattr(args, "repo_root", ""))
        return _print(
            dev_server.start_dev(
                repo=repo,
                port=core._plugin_port(getattr(args, "port", None)),
                host=getattr(args, "host", ""),
                bind=getattr(args, "bind", ""),
                tailscale=getattr(args, "tailscale", False),
                tailscale_serve=getattr(args, "tailscale_serve", False),
                wait_seconds=float(getattr(args, "wait_seconds", 60.0)),
            )
        )
    if command == "stop":
        return _print(
            dev_server.stop_dev(
                pid=getattr(args, "pid", None),
                force=getattr(args, "force", False),
            )
        )
    if command == "speak":
        return _print(
            core.speak({
                "text": " ".join(getattr(args, "text", [])).strip(),
                "client_id": getattr(args, "client_id", ""),
                "emotion": getattr(args, "emotion", ""),
                "priority": getattr(args, "priority", ""),
                "interrupt": getattr(args, "interrupt", False),
            })
        )
    if command == "chat":
        return _print(
            core.chat({
                "text": " ".join(getattr(args, "text", [])).strip(),
                "client_id": getattr(args, "client_id", ""),
                "mode": getattr(args, "mode", "user_input"),
                "interrupt": getattr(args, "interrupt", False),
            })
        )
    if command in {"stop-playback", "stop-speech"}:
        return _print(
            core.stop_playback({
                "client_id": getattr(args, "client_id", ""),
                "mode": getattr(args, "mode", "all"),
            })
        )
    if command in {"bridge-start", "bridge"}:
        return _print(
            core.start_bridge({
                "host": getattr(args, "host", ""),
                "port": getattr(args, "port", None),
                "tailscale": getattr(args, "tailscale", False),
                "confirm_public_host": getattr(args, "confirm_public_host", False),
                "system_prompt": getattr(args, "system_prompt", ""),
            })
        )
    if command == "bridge-stop":
        return _print(core.stop_bridge({"force": getattr(args, "force", False)}))

    print("unknown aituber-kit command")
    return 2


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    ok = payload.get("ok")
    if ok is None:
        ok = payload.get("success")
    return 0 if ok else 1
