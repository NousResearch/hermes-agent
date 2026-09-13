"""Standalone native Hermes registration. Nothing patches the host or prompt."""

import json
from pathlib import Path

from hermes_constants import get_hermes_home
import runpy

_load_runtime = runpy.run_path(str(Path(__file__).resolve().parent / "realms/_binding.py"))["load_runtime"]
_integration = _load_runtime("integration")
get_integration = _integration.get_integration
requirements_available = _integration.requirements_available


SCHEMA = {
    "name": "realm",
    "description": "Manage this conversation’s private desktop. Use on for a private realm (kind=omarchy-vm for a disposable Omarchy VM when the task needs a real Omarchy: shell plugins, Hyprland, system changes), off ONLY for explicit user-requested host access, status, size, stop, watch, or push/pull to copy files in and out of a VM realm. Use shot only when realm Cua app=screen capture fails; never fall back to host capture. Announce which kind is in use. Load skill hermes-realms:realms for natural-language desktop intent. Existing approvals always apply.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["on", "off", "status", "size", "stop", "watch", "shot",
                         "push", "pull"],
            },
            "size": {"type": "string", "description": "WIDTHxHEIGHT for action=size"},
            "kind": {
                "type": "string",
                "enum": ["realm", "omarchy-vm"],
                "description": "Desktop kind for action=on. realm (default) is a fast private labwc desktop; omarchy-vm is a disposable Omarchy guest in QEMU with its own kernel and disk.",
            },
            "path": {
                "type": "string",
                "description": "Host path for action=push, guest path for action=pull",
            },
            "destination": {
                "type": "string",
                "description": "Guest path for action=push, host path for action=pull",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _raw_command(args):
    """Render tool arguments as the equivalent slash-command text.

    One parser for both surfaces: whatever ``/realm ...`` accepts is exactly
    what the tool accepts, and neither can drift into having its own rules.
    """
    action = args.get("action", "status")
    arguments = {
        "size": [args.get("size", "")],
        "on": [{"omarchy-vm": "omarchy"}.get(args.get("kind", ""), args.get("kind", ""))],
        "push": [args.get("path", ""), args.get("destination", "")],
        "pull": [args.get("path", ""), args.get("destination", "")],
    }.get(action, [])
    return " ".join([action, *(value for value in arguments if value)])


def register(ctx):
    cli = _load_runtime("cli")
    configure_parser, run = cli.configure_parser, cli.run

    ctx.register_cli_command(
        "realms", "Manage private Linux desktops and explicitly install their driver",
        configure_parser, run,
    )
    service = get_integration(get_hermes_home())

    def command(raw, **identity):
        try:
            return json.dumps(service.command(raw, **identity))
        except Exception as exc:
            return json.dumps(
                {
                    "error": str(exc)
                    if isinstance(exc, (ValueError, PermissionError))
                    else "Realm operation failed; host fallback is disabled."
                }
            )

    def tool(args, **identity):
        return command(_raw_command(args), **identity)

    ctx.register_tool("realm", "realms", SCHEMA, tool, check_fn=requirements_available)
    ctx.register_command(
        "realm",
        command,
        description="Private desktop: on [omarchy], off, status, size, stop, watch, shot, push, pull",
    )
    ctx.register_hook("pre_tool_call", service.pre_tool)
    ctx.register_hook("on_session_identity", service.bind)
    ctx.register_hook("on_session_start", service.bind)
    ctx.register_hook("on_session_finalize", service.finalize)
    ctx.register_hook("on_session_reset", service.reset)
    ctx.on_unload(service.unload)
    ctx.register_skill(
        "realms",
        Path(__file__).resolve().parent / "skills/realms/SKILL.md",
        description="Choose private realm versus explicit host desktop and manage its lifetime.",
    )
