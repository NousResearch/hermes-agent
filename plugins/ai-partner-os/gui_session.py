"""Hermes GUI session — AI Partner OS as visual front-end for Hermes Agent."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

from . import eel_client, lan_client, process, tts_bridge

SESSION_STATE_NAME = "ai_partner_os_gui_session.json"
ACTION_PATTERN = re.compile(r"\[ACTION:\s*(\w+)\s*:\s*(\{.*?\})\s*\]", re.DOTALL)


def session_state_file() -> Path:
    return get_hermes_home() / SESSION_STATE_NAME


def read_session() -> dict[str, Any]:
    path = session_state_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_session(payload: dict[str, Any]) -> None:
    path = session_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def parse_actions(text: str) -> tuple[str, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    for match in ACTION_PATTERN.finditer(text):
        name = match.group(1)
        try:
            params = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(params, dict):
            params = {}
        actions.append({"action": name, "params": params})
    clean = ACTION_PATTERN.sub("", text).strip()
    return clean, actions


def disable_in_app_tts(eel_call: Callable[..., Any]) -> dict[str, Any]:
    """Turn off AI Partner OS built-in TTS so Hermes backends are used instead."""
    try:
        settings = eel_call("get_settings", timeout=15.0)
        if not isinstance(settings, dict):
            settings = {}
        ai = settings.setdefault("ai", {})
        ai["tts"] = "none"
        eel_call("save_settings", settings, timeout=20.0)
        return {"ok": True, "ai_tts": "none"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def present_reply(
    text: str,
    *,
    eel_call: Callable[..., Any],
    queue_action: Callable[..., dict[str, Any]],
    speak: bool = True,
    open_chat: bool = True,
    tts_provider: str | None = None,
    tts_voice: str | int | None = None,
    tts_speed: float | None = None,
    lan_host: str | None = None,
    lan_port: int | None = None,
    lan_pin: str = "",
) -> dict[str, Any]:
    display_text, actions = parse_actions(text)
    result: dict[str, Any] = {
        "ok": True,
        "text": display_text,
        "raw_text": text,
        "actions": [],
        "tts": None,
        "ui": {},
    }

    if open_chat:
        result["ui"]["open_chat"] = queue_action("openWindow", {"window": "chat"})

    if display_text:
        try:
            eel_call(
                "save_proactive_message",
                {
                    "type": "hermes-gui",
                    "message": display_text,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                timeout=20.0,
            )
            result["ui"]["proactive_saved"] = True
        except Exception as exc:
            result["ui"]["proactive_error"] = str(exc)
            try:
                lan_push = lan_client.send_proactive_message(
                    display_text,
                    host=lan_host or lan_client.DEFAULT_LAN_HOST,
                    hosts=lan_client.resolve_lan_hosts(lan_host or lan_client.DEFAULT_LAN_HOST),
                    port=int(lan_port or lan_client.DEFAULT_LAN_PORT),
                    pin=lan_pin,
                )
                if lan_push.get("ok") and isinstance(lan_push.get("body"), dict) and lan_push["body"].get("reply"):
                    result["ui"]["proactive_saved"] = True
                    result["ui"]["proactive_via"] = "lan"
                    result["ui"]["proactive_endpoint"] = lan_push.get("endpoint")
                    result["ui"]["proactive_note"] = "LAN /api/chat returned AI reply (uses in-app LLM, not Hermes display-only)."
                elif lan_push.get("ok"):
                    result["ui"]["proactive_saved"] = True
                    result["ui"]["proactive_via"] = "lan"
                    result["ui"]["proactive_endpoint"] = lan_push.get("endpoint")
                else:
                    result["ui"]["proactive_lan_error"] = lan_push.get("error")
            except Exception as lan_exc:
                result["ui"]["proactive_lan_error"] = str(lan_exc)

    if speak and display_text:
        try:
            tts = tts_bridge.synthesize_data_url(
                display_text,
                provider=tts_provider,
                voice=tts_voice,
                speed=tts_speed,
            )
            played_via = "eel"
            try:
                eel_call("play_tts_on_pc", tts["data_url"], timeout=30.0)
            except Exception as eel_exc:
                local = tts_bridge.play_audio_local(tts["file_path"], blocking=False)
                if not local.get("ok"):
                    raise RuntimeError(f"{eel_exc}; local playback also failed: {local.get('error')}") from eel_exc
                played_via = str(local.get("via") or "local")
                result["tts"] = {
                    "ok": True,
                    "provider": tts.get("provider"),
                    "file_path": tts.get("file_path"),
                    "played_via": played_via,
                    "eel_error": str(eel_exc),
                    "note": "Eel RPC unavailable — audio played on this PC instead of in-app avatar lip-sync.",
                }
            else:
                result["tts"] = {
                    "ok": True,
                    "provider": tts.get("provider"),
                    "file_path": tts.get("file_path"),
                    "played_via": played_via,
                }
        except Exception as exc:
            result["tts"] = {"ok": False, "error": str(exc)}

    for item in actions:
        queued = queue_action(item["action"], item.get("params") or {})
        result["actions"].append({**item, "queue": queued})

    return result
