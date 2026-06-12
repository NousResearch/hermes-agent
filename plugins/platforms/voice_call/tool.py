"""The ``voice_call`` model tool and the ``/voicecall`` slash command.

Both run inside the gateway process and drive the live runtime directly.
The tool handler always returns a JSON string and never raises into the
agent loop; error paths return ``{"success": false, "error": ...}``.
"""

import json
import shlex
from typing import Any, Dict, Optional

VOICE_CALL_SCHEMA: Dict[str, Any] = {
    "name": "voice_call",
    "description": (
        "Make and manage real phone calls. Actions: initiate_call (dial a "
        "number and speak a message; mode 'notify' hangs up after the "
        "message, mode 'conversation' keeps the line open for a dialog), "
        "continue_call (say something and wait for the person's reply), "
        "speak_to_user (say something without waiting), send_dtmf (press "
        "keypad digits, e.g. for phone menus), end_call (hang up), and "
        "get_status (one call or all active calls). Phone numbers must be "
        "E.164 (+15555550123). Calls cost real money on real providers — "
        "only place calls the user asked for."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "initiate_call",
                    "continue_call",
                    "speak_to_user",
                    "send_dtmf",
                    "end_call",
                    "get_status",
                ],
                "description": "What to do.",
            },
            "to_number": {
                "type": "string",
                "description": (
                    "E.164 destination for initiate_call (e.g. +15555550123). "
                    "Omit to call the configured default number, if one is set."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "What to say (initiate_call greeting, continue_call / "
                    "speak_to_user content). Keep it short and speakable."
                ),
            },
            "call_id": {
                "type": "string",
                "description": (
                    "Call to operate on (from initiate_call/get_status). "
                    "Required for continue_call, speak_to_user, send_dtmf, "
                    "end_call."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["notify", "conversation"],
                "description": (
                    "initiate_call only. notify: speak and hang up. "
                    "conversation: stay on the line for replies. Defaults to "
                    "the configured outbound mode."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "initiate_call only, conversation mode: a brief for the "
                    "realtime voice to run THIS call — persona, the full "
                    "list of questions to ask, closing lines. The voice "
                    "conducts the whole call itself (no per-question "
                    "round-trips); read the call transcript afterwards for "
                    "the answers."
                ),
            },
            "digits": {
                "type": "string",
                "description": "DTMF digits for send_dtmf (0-9, *, #, w=0.5s pause).",
            },
        },
        "required": ["action"],
    },
}


def tool_check() -> bool:
    """The tool is available whenever the platform's dependencies are."""
    from .adapter import check_requirements

    return check_requirements()


def _ok(**payload) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False)


def _err(message: str, **payload) -> str:
    return json.dumps(
        {"success": False, "error": message, **payload}, ensure_ascii=False
    )


def _call_summary(record) -> Dict[str, Any]:
    return {
        "call_id": record.call_id,
        "state": record.state.value,
        "direction": record.direction,
        "mode": record.mode,
        "peer_number": record.peer_number,
        "started_at": record.started_at,
        "transcript_entries": len(record.transcript),
    }


async def _dispatch(action: str, args: Dict[str, Any]) -> str:
    from .manager import CallNotFoundError
    from .runtime import get_runtime

    runtime = get_runtime()
    if runtime is None or runtime.manager is None:
        # No in-process runtime (e.g. the agent runs in `hermes chat`, a
        # separate process from the gateway). Drive the running gateway
        # through its localhost admin endpoint instead — same path the
        # `hermes voicecall` CLI uses.
        return await _dispatch_via_admin(action, args)
    manager = runtime.manager

    call_id = str(args.get("call_id") or "")
    try:
        if action == "initiate_call":
            # to_number falls back to the configured default (to_number /
            # VOICE_CALL_TO_NUMBER) inside the manager, like OpenClaw.
            record = await manager.initiate_call(
                str(args.get("to_number") or "").strip() or None,
                message=args.get("message"),
                mode=args.get("mode"),
                instructions=args.get("instructions"),
            )
            return _ok(call=_call_summary(record))
        if action == "continue_call":
            message = str(args.get("message") or "").strip()
            if not call_id or not message:
                return _err("continue_call requires call_id and message")
            reply = await manager.continue_call(call_id, message)
            return _ok(call_id=call_id, reply=reply)
        if action == "speak_to_user":
            message = str(args.get("message") or "").strip()
            if not call_id or not message:
                return _err("speak_to_user requires call_id and message")
            await manager.speak(call_id, message)
            return _ok(call_id=call_id)
        if action == "send_dtmf":
            digits = str(args.get("digits") or "").strip()
            if not call_id or not digits:
                return _err("send_dtmf requires call_id and digits")
            await manager.send_dtmf(call_id, digits)
            return _ok(call_id=call_id, digits=digits)
        if action == "end_call":
            if not call_id:
                return _err("end_call requires call_id")
            await manager.end_call(call_id)
            return _ok(call_id=call_id, state="hangup-bot")
        if action == "get_status":
            if call_id:
                record = manager.get_call(call_id)
                if record is None:
                    return _err(f"no active call {call_id!r}", call_id=call_id)
                return _ok(call=_call_summary(record))
            return _ok(
                provider=runtime.config.provider,
                active_calls=[_call_summary(r) for r in manager.get_active_calls()],
            )
        return _err(f"unknown action {action!r}")
    except CallNotFoundError:
        return _err(f"no active call {call_id!r}", call_id=call_id)
    except TimeoutError:
        return _err(
            "timed out waiting for the caller's reply", call_id=call_id
        )
    except Exception as e:  # noqa: BLE001 — JSON contract, never raise
        return _err(str(e))


# Tool action → gateway admin command, with the payload fields each takes.
_ADMIN_COMMAND_MAP = {
    "initiate_call": ("call", {"to_number": "to", "message": "message",
                               "mode": "mode", "instructions": "instructions"}),
    "continue_call": ("continue", {"call_id": "call_id", "message": "message"}),
    "speak_to_user": ("speak", {"call_id": "call_id", "message": "message"}),
    "send_dtmf": ("dtmf", {"call_id": "call_id", "digits": "digits"}),
    "end_call": ("end", {"call_id": "call_id"}),
    "get_status": ("status", {}),
}


async def _dispatch_via_admin(action: str, args: Dict[str, Any]) -> str:
    from .cli import _admin_address, _admin_token, _load_extra

    mapping = _ADMIN_COMMAND_MAP.get(action)
    if mapping is None:
        return _err(f"unknown action {action!r}")
    command, fields = mapping
    token = _admin_token()
    if token is None:
        return _err(
            "voice_call runtime is not running — start the gateway with the "
            "voice_call platform enabled (hermes gateway run)"
        )
    payload: Dict[str, Any] = {"command": command}
    for tool_key, admin_key in fields.items():
        value = args.get(tool_key)
        if value is not None:
            payload[admin_key] = value

    import httpx

    url = _admin_address(_load_extra()) + "/voice/admin"
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                url, json=payload, headers={"x-voice-call-admin-token": token}
            )
            result = response.json()
    except Exception as e:  # noqa: BLE001 — connection refused etc.
        return _err(
            "could not reach the voice_call gateway endpoint — is the "
            f"gateway running? ({e})"
        )
    if not isinstance(result, dict):
        return _err("unexpected response from the voice_call gateway endpoint")
    if action == "get_status" and result.get("success"):
        calls = result.get("active_calls", [])
        call_id = str(args.get("call_id") or "")
        if call_id:
            match = [c for c in calls if c.get("call_id") == call_id]
            if not match:
                return _err(f"no active call {call_id!r}", call_id=call_id)
            return _ok(call=match[0])
        return _ok(provider=result.get("provider"), active_calls=calls)
    return json.dumps(result, ensure_ascii=False)


async def voice_call_handler(args: Dict[str, Any], **_kwargs) -> str:
    action = str(args.get("action") or "").strip()
    if not action:
        return _err("action is required")
    try:
        return await _dispatch(action, args)
    except Exception as e:  # noqa: BLE001 — belt and suspenders
        return _err(str(e))


# -- /voicecall slash command ---------------------------------------------------

_SLASH_USAGE = (
    "Usage: /voicecall status | call --to +1555... --message \"...\" "
    "[--mode notify|conversation] | speak --call-id ID --message \"...\" | "
    "dtmf --call-id ID --digits 123# | end --call-id ID"
)


def _parse_flags(tokens) -> Dict[str, str]:
    flags: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--") and i + 1 < len(tokens):
            flags[token[2:].replace("-", "_")] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return flags


async def slash_handler(raw_args: str = "", **_kwargs) -> Optional[str]:
    try:
        tokens = shlex.split(raw_args or "")
    except ValueError as e:
        return f"voice_call: {e}\n{_SLASH_USAGE}"
    if not tokens:
        return _SLASH_USAGE
    sub, flags = tokens[0].lower(), _parse_flags(tokens[1:])

    action_map = {
        "status": ("get_status", {"call_id": flags.get("call_id", "")}),
        "call": (
            "initiate_call",
            {
                "to_number": flags.get("to", ""),
                "message": flags.get("message"),
                # Operator surfaces default to conversation (like the CLI);
                # the model tool keeps the configured outbound default.
                "mode": flags.get("mode", "conversation"),
            },
        ),
        "speak": (
            "speak_to_user",
            {"call_id": flags.get("call_id", ""), "message": flags.get("message", "")},
        ),
        "dtmf": (
            "send_dtmf",
            {"call_id": flags.get("call_id", ""), "digits": flags.get("digits", "")},
        ),
        "end": ("end_call", {"call_id": flags.get("call_id", "")}),
    }
    if sub not in action_map:
        return f"voice_call: unknown subcommand {sub!r}\n{_SLASH_USAGE}"
    action, args = action_map[sub]
    result = json.loads(await _dispatch(action, args))
    if not result.get("success"):
        return f"voice_call error: {result.get('error')}"
    if action == "get_status":
        calls = result.get("active_calls")
        if calls is None:
            calls = [result["call"]]
        if not calls:
            return "voice_call: no active calls"
        lines = [
            f"• {c['call_id']} — {c['state']} {c['direction']} "
            f"{c['peer_number']} ({c['mode']})"
            for c in calls
        ]
        return "Active calls:\n" + "\n".join(lines)
    if action == "initiate_call":
        call = result["call"]
        return f"voice_call: dialing {call['peer_number']} — call_id {call['call_id']}"
    return f"voice_call: ok ({json.dumps({k: v for k, v in result.items() if k != 'success'})})"
