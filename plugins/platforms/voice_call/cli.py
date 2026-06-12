"""``hermes voicecall ...`` CLI.

The CLI runs in its own process; action commands drive the *running*
gateway through the webhook server's localhost admin endpoint,
authenticated with the pre-shared token at
``$HERMES_HOME/voice-calls/admin.token``. Read-only commands (``tail``,
parts of ``doctor``) work straight off the on-disk state.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="voicecall_command")

    status_p = subs.add_parser("status", help="Show active calls")
    status_p.add_argument("--json", action="store_true", dest="as_json")

    call_p = subs.add_parser("call", help="Place an outbound call")
    call_p.add_argument(
        "-t", "--to",
        help="E.164 destination (+1555...); falls back to config to_number",
    )
    call_p.add_argument("-m", "--message", help="What to say when answered")
    # CLI default is conversation (matches OpenClaw's voicecall CLI); the
    # model tool and cron delivery still follow outbound.default_mode.
    call_p.add_argument(
        "--mode", choices=("notify", "conversation"), default="conversation",
        help="notify: speak and hang up; conversation: stay open (default)",
    )

    speak_p = subs.add_parser("speak", help="Say something on a live call")
    speak_p.add_argument("--call-id", required=True)
    speak_p.add_argument("-m", "--message", required=True)

    cont_p = subs.add_parser("continue", help="Say something and wait for the reply")
    cont_p.add_argument("--call-id", required=True)
    cont_p.add_argument("-m", "--message", required=True)

    dtmf_p = subs.add_parser("dtmf", help="Send keypad digits")
    dtmf_p.add_argument("--call-id", required=True)
    dtmf_p.add_argument("--digits", required=True)

    end_p = subs.add_parser("end", help="Hang up a call")
    end_p.add_argument("--call-id", required=True)

    subs.add_parser("doctor", help="Diagnose voice_call configuration")

    tail_p = subs.add_parser("tail", help="Show recent call log entries")
    tail_p.add_argument("--lines", type=int, default=20)
    tail_p.add_argument(
        "-f", "--follow", action="store_true",
        help="Keep printing new entries as they are written",
    )
    tail_p.add_argument(
        "--poll", type=float, default=0.25,
        help="Follow-mode poll interval in seconds (default 0.25)",
    )


def dispatch(args: argparse.Namespace) -> int:
    command = getattr(args, "voicecall_command", None)
    if not command:
        print("usage: hermes voicecall "
              "{status,call,speak,continue,dtmf,end,doctor,tail} ...")
        return 1
    handler = {
        "status": _cmd_status,
        "call": _cmd_call,
        "speak": _cmd_speak,
        "continue": _cmd_continue,
        "dtmf": _cmd_dtmf,
        "end": _cmd_end,
        "doctor": _cmd_doctor,
        "tail": _cmd_tail,
    }[command]
    return handler(args)


# -- admin endpoint plumbing -----------------------------------------------------


def _voice_dir() -> Path:
    return get_hermes_home() / "voice-calls"


def _load_extra() -> Dict[str, Any]:
    """voice_call platform ``extra`` from config.yaml (best effort)."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        return (
            (config.get("gateway") or {}).get("platforms", {}).get("voice_call", {})
        ).get("extra") or {}
    except Exception:
        return {}


def _admin_address(extra: Dict[str, Any]) -> str:
    serve = extra.get("serve") or {}
    bind = str(serve.get("bind", "127.0.0.1"))
    if bind in ("0.0.0.0", "::"):
        bind = "127.0.0.1"
    port = int(serve.get("port", 3334))
    return f"http://{bind}:{port}"


def _admin_token() -> Optional[str]:
    try:
        token = (_voice_dir() / "admin.token").read_text(encoding="utf-8").strip()
        return token or None
    except OSError:
        return None


def _admin_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST a command to the running gateway's admin endpoint."""
    token = _admin_token()
    if token is None:
        return {
            "success": False,
            "error": (
                "no admin token found — is the gateway running with the "
                "voice_call platform enabled? (hermes gateway run)"
            ),
        }

    async def _post():
        import httpx

        url = _admin_address(_load_extra()) + "/voice/admin"
        async with httpx.AsyncClient(timeout=70.0) as client:
            resp = await client.post(
                url, json=payload, headers={"x-voice-call-admin-token": token}
            )
            return resp.json()

    try:
        return asyncio.run(_post())
    except Exception as e:  # noqa: BLE001 — connection refused etc.
        return {
            "success": False,
            "error": f"could not reach the voice_call gateway endpoint: {e}",
        }


def _print_result(result: Dict[str, Any], as_json: bool = False) -> int:
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1
    if not result.get("success"):
        print(f"error: {result.get('error')}", file=sys.stderr)
        return 1
    return 0


# -- commands -----------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    result = _admin_request({"command": "status"})
    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1
    if not result.get("success"):
        print(f"error: {result.get('error')}", file=sys.stderr)
        return 1
    calls = result.get("active_calls", [])
    print(f"provider: {result.get('provider')}")
    if result.get("public_url"):
        print(f"public url: {result['public_url']}")
    if not calls:
        print("no active calls")
        return 0
    for call in calls:
        peer = call.get("to_number") if call.get("direction") == "outbound" else call.get("from_number")
        print(
            f"  {call['call_id']}  {call['state']:<10} {call['direction']:<8} "
            f"{peer or '?'}  mode={call.get('mode')}"
        )
    return 0


def _cmd_call(args: argparse.Namespace) -> int:
    result = _admin_request(
        {"command": "call", "to": args.to, "message": args.message, "mode": args.mode}
    )
    if result.get("success"):
        print(
            f"dialing {args.to or 'configured to_number'} "
            f"({args.mode}) — call_id {result.get('call_id')}"
        )
    return _print_result(result)


def _cmd_speak(args: argparse.Namespace) -> int:
    result = _admin_request(
        {"command": "speak", "call_id": args.call_id, "message": args.message}
    )
    if result.get("success"):
        print("ok")
    return _print_result(result)


def _cmd_continue(args: argparse.Namespace) -> int:
    result = _admin_request(
        {"command": "continue", "call_id": args.call_id, "message": args.message}
    )
    if result.get("success"):
        print(f"reply: {result.get('reply')}")
    return _print_result(result)


def _cmd_dtmf(args: argparse.Namespace) -> int:
    result = _admin_request(
        {"command": "dtmf", "call_id": args.call_id, "digits": args.digits}
    )
    if result.get("success"):
        print("ok")
    return _print_result(result)


def _cmd_end(args: argparse.Namespace) -> int:
    result = _admin_request({"command": "end", "call_id": args.call_id})
    if result.get("success"):
        print("call ended")
    return _print_result(result)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report config validity and env presence — never secret values."""
    from .config import PROVIDER_REQUIRED_ENV, VoiceCallConfig

    extra = _load_extra()
    cfg = VoiceCallConfig.from_extra(extra)
    print(f"provider: {cfg.provider}")
    print(f"serve: {cfg.serve.bind}:{cfg.serve.port}{cfg.serve.path}")
    print(f"public exposure: "
          f"{cfg.public_url or cfg.tunnel.provider}")

    for env in PROVIDER_REQUIRED_ENV.get(cfg.provider, []):
        present = "present" if os.getenv(env, "").strip() else "MISSING"
        print(f"env {env}: {present}")
    if cfg.provider == "telnyx":
        present = "present" if os.getenv("TELNYX_PUBLIC_KEY", "").strip() else "MISSING"
        print(f"env TELNYX_PUBLIC_KEY: {present}")

    errors = cfg.validate()
    if errors:
        print("\nconfig errors:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\nconfig: ok")

    result = _admin_request({"command": "status"})
    if result.get("success"):
        print(f"gateway runtime: reachable ({len(result.get('active_calls', []))} "
              "active calls)")
    else:
        print(f"gateway runtime: not reachable — {result.get('error')}")
    return 1 if errors else 0


def _print_call_line(line: str) -> None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return
    print(
        f"{data.get('call_id')}  {data.get('state'):<12} "
        f"{data.get('direction'):<8} from={data.get('from_number')} "
        f"to={data.get('to_number')} reason={data.get('end_reason')}"
    )


def _cmd_tail(args: argparse.Namespace) -> int:
    import time

    calls_path = _voice_dir() / "calls.jsonl"
    if not calls_path.exists() and not args.follow:
        print("no call log yet")
        return 0
    position = 0
    if calls_path.exists():
        with open(calls_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines[-max(1, args.lines):]:
            _print_call_line(line)
        position = calls_path.stat().st_size
    if not args.follow:
        return 0
    try:
        while True:
            time.sleep(max(0.05, args.poll))
            if not calls_path.exists():
                continue
            size = calls_path.stat().st_size
            if size < position:
                position = 0  # log was compacted/rotated — start over
            if size == position:
                continue
            with open(calls_path, encoding="utf-8") as f:
                f.seek(position)
                chunk = f.read()
                position = f.tell()
            for line in chunk.splitlines():
                if line.strip():
                    _print_call_line(line.strip())
    except KeyboardInterrupt:
        return 0
