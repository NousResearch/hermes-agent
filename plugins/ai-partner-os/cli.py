"""CLI for the AI Partner OS Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="ai_partner_os_command")

    configure = subs.add_parser("configure", aliases=["setup"], help="Save AI Partner OS settings")
    configure.add_argument("--exe-path", default="")
    configure.add_argument("--eel-port", type=int, default=None)
    configure.add_argument("--lan-host", default="")
    configure.add_argument("--lan-port", type=int, default=None)
    configure.add_argument("--bridge-port", type=int, default=None)
    configure.add_argument("--system-prompt", default="")
    configure.add_argument("--lan-pin", default="")
    configure.add_argument("--tts-provider", choices=["auto", "irodori", "voicevox", "none"], default="")
    configure.add_argument("--tts-voice", default="")
    configure.add_argument("--tts-speed", type=float, default=None)
    configure.add_argument("--gui-mode", action="store_true", default=None)

    subs.add_parser("status", help="Show AI Partner OS readiness")

    connect = subs.add_parser("connect-gui", aliases=["connect", "gui"], help="Bind as Hermes GUI")
    connect.add_argument("--no-start", action="store_true")
    connect.add_argument("--no-lan", action="store_true")
    connect.add_argument("--no-discover", action="store_true")
    connect.add_argument("--no-tts", action="store_true")
    connect.add_argument("--with-bridge", action="store_true")
    connect.add_argument("--tts-provider", choices=["auto", "irodori", "voicevox", "none"], default="")

    subs.add_parser("discover-lan", aliases=["discover"], help="Probe LAN HTTP routes on 8899")
    probe = subs.add_parser("probe-eel", help="Scan localhost for Eel WebSocket RPC")
    probe.add_argument("--eel-port", type=int, default=None)
    subs.add_parser("tts-status", aliases=["tts"], help="Hermes TTS readiness")
    start_tts = subs.add_parser("start-tts", help="Start Hermes irodori/VOICEVOX backend")
    start_tts.add_argument("--provider", choices=["auto", "irodori", "voicevox"], default="auto")

    start = subs.add_parser("start", help="Launch AI Partner OS")
    start.add_argument("--exe-path", default="")
    start.add_argument("--wait-seconds", type=float, default=8.0)

    stop = subs.add_parser("stop", help="Stop managed AI Partner OS process")
    stop.add_argument("--force", action="store_true")

    lan_on = subs.add_parser("lan-on", help="Enable LAN/mobile server via Eel")
    lan_off = subs.add_parser("lan-off", help="Disable LAN/mobile server via Eel")

    chat = subs.add_parser("chat", help="Chat via Hermes LLM + GUI present")
    chat.add_argument("text", nargs="*")
    chat.add_argument("--no-present", action="store_true")

    gui_say = subs.add_parser("gui-say", aliases=["say"], help="GUI mode chat + Hermes TTS")
    gui_say.add_argument("text", nargs="*")

    speak = subs.add_parser("speak", help="Speak via Hermes TTS on avatar")
    speak.add_argument("text", nargs="*")
    speak.add_argument("--tts-provider", choices=["auto", "irodori", "voicevox", "none"], default="")
    speak.add_argument("--lan-pin", default="")

    action = subs.add_parser("action", help="Queue an OS action")
    action.add_argument("name")
    action.add_argument("--params", default="{}", help="JSON object")
    action.add_argument("--lan-pin", default="")

    bridge_start = subs.add_parser("bridge-start", aliases=["bridge"], help="Start External Linkage WS bridge")
    bridge_start.add_argument("--host", default="")
    bridge_start.add_argument("--port", type=int, default=None)
    bridge_start.add_argument("--tailscale", action="store_true")
    bridge_start.add_argument("--confirm-public-host", action="store_true")
    bridge_start.add_argument("--system-prompt", default="")

    bridge_stop = subs.add_parser("bridge-stop", help="Stop External Linkage WS bridge")
    bridge_stop.add_argument("--force", action="store_true")


def ai_partner_os_command(args: argparse.Namespace) -> int:
    command = getattr(args, "ai_partner_os_command", None) or "status"

    if command in {"configure", "setup"}:
        payload: dict = {
            "exe_path": getattr(args, "exe_path", "") or None,
            "eel_port": getattr(args, "eel_port", None),
            "lan_host": getattr(args, "lan_host", "") or None,
            "lan_port": getattr(args, "lan_port", None),
            "bridge_port": getattr(args, "bridge_port", None),
            "system_prompt": getattr(args, "system_prompt", "") or None,
            "lan_pin": getattr(args, "lan_pin", "") or None,
            "tts_provider": getattr(args, "tts_provider", "") or None,
            "tts_voice": getattr(args, "tts_voice", "") or None,
            "tts_speed": getattr(args, "tts_speed", None),
        }
        if getattr(args, "gui_mode", None) is not None:
            payload["gui_mode"] = bool(args.gui_mode)
        return _print(core.configure(payload))

    if command == "status":
        return _print(core.status())

    if command in {"connect-gui", "connect", "gui"}:
        return _print(
            core.connect_gui(
                {
                    "start_app": not getattr(args, "no_start", False),
                    "enable_lan": not getattr(args, "no_lan", False),
                    "discover_lan": not getattr(args, "no_discover", False),
                    "start_tts": not getattr(args, "no_tts", False),
                    "start_bridge": getattr(args, "with_bridge", False),
                    "tts_provider": getattr(args, "tts_provider", "") or None,
                }
            )
        )

    if command in {"discover-lan", "discover"}:
        return _print(core.discover_lan({}))

    if command == "probe-eel":
        return _print(core.probe_eel({"eel_port": getattr(args, "eel_port", None)}))

    if command in {"tts-status", "tts"}:
        return _print(core.hermes_tts_status({}))

    if command == "start-tts":
        return _print(core.start_hermes_tts({"tts_provider": getattr(args, "provider", "auto")}))

    if command == "start":
        return _print(
            core.start_app(
                {
                    "exe_path": getattr(args, "exe_path", "") or None,
                    "wait_seconds": getattr(args, "wait_seconds", 8.0),
                }
            )
        )

    if command == "stop":
        return _print(core.stop_app({"force": getattr(args, "force", False)}))

    if command == "lan-on":
        return _print(core.enable_lan({"enabled": True}))

    if command == "lan-off":
        return _print(core.enable_lan({"enabled": False}))

    if command == "chat":
        text = " ".join(getattr(args, "text", []) or []).strip()
        if not text:
            print("usage: hermes ai-partner-os chat <message>")
            return 1
        return _print(core.chat({"text": text, "present": not getattr(args, "no_present", False), "gui_mode": True}))

    if command in {"gui-say", "say"}:
        text = " ".join(getattr(args, "text", []) or []).strip()
        if not text:
            print("usage: hermes ai-partner-os gui-say <message>")
            return 1
        return _print(core.gui_say({"text": text}))

    if command == "speak":
        text = " ".join(getattr(args, "text", []) or []).strip()
        if not text:
            print("usage: hermes ai-partner-os speak <text>")
            return 1
        return _print(
            core.speak(
                {
                    "text": text,
                    "lan_pin": getattr(args, "lan_pin", ""),
                    "tts_provider": getattr(args, "tts_provider", "") or None,
                }
            )
        )

    if command == "action":
        try:
            params = json.loads(getattr(args, "params", "{}"))
        except json.JSONDecodeError as exc:
            print(f"invalid --params JSON: {exc}")
            return 1
        if not isinstance(params, dict):
            print("--params must be a JSON object")
            return 1
        return _print(
            core.run_action(
                {
                    "action": getattr(args, "name", ""),
                    "params": params,
                    "lan_pin": getattr(args, "lan_pin", ""),
                }
            )
        )

    if command in {"bridge-start", "bridge"}:
        return _print(
            core.start_bridge(
                {
                    "host": getattr(args, "host", "") or None,
                    "port": getattr(args, "port", None),
                    "tailscale": getattr(args, "tailscale", False),
                    "confirm_public_host": getattr(args, "confirm_public_host", False),
                    "system_prompt": getattr(args, "system_prompt", "") or None,
                }
            )
        )

    if command == "bridge-stop":
        return _print(core.stop_bridge({"force": getattr(args, "force", False)}))

    print(
        "usage: hermes ai-partner-os "
        "{configure,status,connect-gui,discover-lan,probe-eel,tts-status,start-tts,start,stop,"
        "lan-on,lan-off,chat,gui-say,speak,action,bridge-start,bridge-stop}"
    )
    return 1
