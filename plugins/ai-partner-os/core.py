"""Core Hermes bridge for AI Partner OS (fluere / ver0.2.x)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from hermes_constants import display_hermes_home, get_hermes_home

from . import eel_client, gui_session, lan_client, process, tts_bridge

PLUGIN_ID = "ai-partner-os"
CONFIG_ALIASES = (PLUGIN_ID, "ai_partner_os", "aipartneros")
TOOLSET = "ai_partner_os"
DEFAULT_EXE = Path(r"C:\Users\downl\Downloads\AI_Partner_OS_ver0.2.3 (1)\AI_Partner_OS.exe")
DEFAULT_EEL_PORT = 8000
DEFAULT_LAN_PORT = lan_client.DEFAULT_LAN_PORT
DEFAULT_BRIDGE_PORT = 8010
BRIDGE_STATE_NAME = "ai_partner_os_bridge_state.json"

DEFAULT_SYSTEM_PROMPT = (
    "あなたは AI Partner OS 上の AI パートナーキャラクターです。"
    "自然で短い日本語で、ユーザーの生活リズムに寄り添って話してください。"
    "OS操作が必要なときは [ACTION:アクション名:{\"param\":\"value\"}] 形式を使えます。"
)

_llm_factory: Callable[[], Any] | None = None


def bind_llm_factory(factory: Callable[[], Any] | None) -> None:
    global _llm_factory
    _llm_factory = factory


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _path_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _is_loopback_host(host: str) -> bool:
    value = (host or "").strip().lower()
    return value in {"localhost", "::1"} or value.startswith("127.")


def _load_config_readonly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _plugin_config() -> dict[str, Any]:
    plugins = _load_config_readonly().get("plugins", {})
    if not isinstance(plugins, dict):
        return {}
    entries = plugins.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    for key in CONFIG_ALIASES:
        value = entries.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _plugin_exe_path(value: Any = None) -> Path:
    text = _path_text(value or _plugin_config().get("exe_path"))
    if text:
        return Path(text).expanduser()
    env = _path_text(os.environ.get("AI_PARTNER_OS_EXE"))
    if env:
        return Path(env).expanduser()
    if DEFAULT_EXE.is_file():
        return DEFAULT_EXE
    return DEFAULT_EXE


def _plugin_eel_port(value: Any = None) -> int:
    if value is not None:
        try:
            port = int(value)
            if 1024 <= port <= 65535:
                return port
        except (TypeError, ValueError):
            pass
    cfg = _plugin_config()
    try:
        port = int(cfg.get("eel_port") or 0)
        if 1024 <= port <= 65535:
            return port
    except (TypeError, ValueError):
        pass
    return DEFAULT_EEL_PORT


def _plugin_lan_host(value: Any = None) -> str:
    text = _path_text(value or _plugin_config().get("lan_host"))
    return text or lan_client.DEFAULT_LAN_HOST


def _plugin_lan_port(value: Any = None) -> int:
    if value is not None:
        try:
            port = int(value)
            if 1024 <= port <= 65535:
                return port
        except (TypeError, ValueError):
            pass
    cfg = _plugin_config()
    try:
        port = int(cfg.get("lan_port") or 0)
        if 1024 <= port <= 65535:
            return port
    except (TypeError, ValueError):
        pass
    return DEFAULT_LAN_PORT


def _plugin_bridge_port(value: Any = None) -> int:
    if value is not None:
        try:
            port = int(value)
            if 1024 <= port <= 65535:
                return port
        except (TypeError, ValueError):
            pass
    cfg = _plugin_config()
    try:
        port = int(cfg.get("bridge_port") or 0)
        if 1024 <= port <= 65535:
            return port
    except (TypeError, ValueError):
        pass
    return DEFAULT_BRIDGE_PORT


def _plugin_system_prompt(value: Any = None) -> str:
    text = _path_text(value or _plugin_config().get("system_prompt"))
    return text or DEFAULT_SYSTEM_PROMPT


def _plugin_lan_pin(value: Any = None) -> str:
    return _path_text(value or _plugin_config().get("lan_pin"))


def _resolve_lan_hosts(value: Any = None) -> list[str]:
    return lan_client.resolve_lan_hosts(_plugin_lan_host(value))


def _plugin_tts_provider(value: Any = None) -> str:
    text = _path_text(value or _plugin_config().get("tts_provider")) or "auto"
    return text if text in tts_bridge.SUPPORTED_PROVIDERS else "auto"


def _plugin_tts_voice(value: Any = None) -> str | int | None:
    raw = value if value is not None else _plugin_config().get("tts_voice")
    if raw is None or raw == "":
        return None
    if isinstance(raw, int):
        return raw
    text = _path_text(raw)
    if text.isdigit():
        return int(text)
    return text


def _plugin_tts_speed(value: Any = None) -> float | None:
    raw = value if value is not None else _plugin_config().get("tts_speed")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _plugin_gui_mode(value: Any = None) -> bool:
    if value is not None:
        return bool(value)
    return bool(_plugin_config().get("gui_mode"))


def _queue_action(action: str, params: dict[str, Any] | None = None, *, lan_pin: str = "") -> dict[str, Any]:
    params = params or {}
    lan_hosts = _resolve_lan_hosts()
    lan_port = _plugin_lan_port()
    pin = lan_pin or _plugin_lan_pin()

    if eel_client.eel_available(port=_resolve_eel_port()).get("rpc_ok"):
        for fn_name in ("queue_pc_action", "add_pc_action", "enqueue_pc_action"):
            try:
                result = _eel_call(fn_name, action, params, timeout=15.0)
                return {"ok": True, "via": "eel", "function": fn_name, "result": result}
            except Exception:
                continue

    if any(lan_client.lan_port_open(host=h, port=lan_port) for h in lan_hosts):
        lan_result = lan_client.queue_pc_action(
            action,
            params,
            host=_plugin_lan_host(),
            hosts=lan_hosts,
            port=lan_port,
            pin=pin,
        )
        if lan_result.get("ok"):
            body = lan_result.get("body") if isinstance(lan_result.get("body"), dict) else {}
            if body.get("success") is False:
                return {
                    "ok": False,
                    "via": "lan",
                    "host": lan_result.get("host"),
                    "error": body.get("message") or "LAN action rejected",
                    "result": lan_result,
                }
            return {"ok": True, "via": "lan", "host": lan_result.get("host"), "result": lan_result}

    return {"ok": False, "error": "Could not queue action", "action": action, "params": params}


def _bridge_state_file() -> Path:
    return get_hermes_home() / BRIDGE_STATE_NAME


def _read_bridge_state() -> dict[str, Any]:
    path = _bridge_state_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_bridge_state(payload: dict[str, Any]) -> None:
    path = _bridge_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_bridge_state() -> None:
    try:
        _bridge_state_file().unlink()
    except FileNotFoundError:
        pass


def check_available() -> bool:
    return True


def _resolve_eel_port(explicit: int | None = None) -> int:
    if explicit:
        return explicit
    configured = _plugin_eel_port()
    detected = eel_client.detect_eel_port(ports=(configured, *eel_client.DEFAULT_EEL_PORTS))
    return detected or configured


def _eel_call(name: str, *args: Any, port: int | None = None, timeout: float = 120.0) -> Any:
    return eel_client.eel_call(name, *args, port=_resolve_eel_port(port), timeout=timeout)


def _run_llm(user_text: str, *, system_prompt: str | None = None) -> str:
    if _llm_factory is None:
        raise RuntimeError("Hermes LLM is not available (plugin not registered).")
    llm = _llm_factory()
    if llm is None:
        raise RuntimeError("Hermes LLM factory returned None.")
    prompt = system_prompt or _plugin_system_prompt()
    if hasattr(llm, "complete"):
        result = llm.complete(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            purpose="ai_partner_os_chat",
        )
        text = getattr(result, "text", result)
        return str(text or "").strip()
    if callable(llm):
        return str(llm(f"{prompt}\n\nUser:\n{user_text}") or "").strip()
    raise RuntimeError("Unsupported Hermes LLM interface.")


def status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    exe = _plugin_exe_path(values.get("exe_path"))
    state = process.read_state()
    pid = int(state.get("pid") or 0)
    eel_port = _resolve_eel_port(_plugin_eel_port(values.get("eel_port")))
    lan_hosts = _resolve_lan_hosts(values.get("lan_host"))
    lan_port = _plugin_lan_port(values.get("lan_port"))
    bridge_port = _plugin_bridge_port(values.get("bridge_port"))
    bridge_state = _read_bridge_state()
    bridge_pid = int(bridge_state.get("pid") or 0)

    eel_info = eel_client.eel_available(port=eel_port)
    lan_open_hosts = [h for h in lan_hosts if lan_client.lan_port_open(host=h, port=lan_port)]
    lan_open = bool(lan_open_hosts)

    lan_status = eel_info.get("lan_status")
    if lan_status is None and not eel_info.get("rpc_ok"):
        lan_status = {"error": eel_info.get("error") or "Eel RPC unavailable"}

    lan_auth = None
    if lan_open and _plugin_lan_pin():
        lan_auth = lan_client.lan_authenticated_status(
            host=_plugin_lan_host(values.get("lan_host")),
            hosts=lan_hosts,
            port=lan_port,
            pin=_plugin_lan_pin(),
        )

    gui_sess = gui_session.read_session()
    tts_info = tts_bridge.tts_status(_plugin_tts_provider())

    eel_embedded_only = lan_open and not eel_info.get("rpc_ok")

    return {
        "plugin": PLUGIN_ID,
        "hermes_home": display_hermes_home(),
        "exe_path": str(exe),
        "exe_exists": exe.is_file(),
        "managed_pid": pid,
        "process_running": process.pid_alive(pid),
        "eel_port": eel_port,
        "eel_reachable": bool(eel_info.get("rpc_ok")),
        "eel_http_only": bool(eel_info.get("http_only")),
        "eel_embedded_only": eel_embedded_only,
        "eel_rpc_error": eel_info.get("error"),
        "eel_url": f"http://127.0.0.1:{eel_port}/",
        "lan_host": _plugin_lan_host(values.get("lan_host")),
        "lan_hosts": lan_hosts,
        "lan_port": lan_port,
        "lan_port_open": lan_open,
        "lan_open_hosts": lan_open_hosts,
        "lan_pin_configured": bool(_plugin_lan_pin()),
        "lan_auth": lan_auth,
        "lan_status": lan_status,
        "lan_routes_cache": str(lan_client.LAN_ROUTES_CACHE),
        "bridge_port": bridge_port,
        "bridge_url": f"ws://127.0.0.1:{bridge_port}/ws",
        "bridge_running": process.pid_alive(bridge_pid),
        "bridge_state": bridge_state,
        "gui_mode": _plugin_gui_mode(),
        "gui_session": gui_sess,
        "hermes_tts": tts_info,
        "role": "Hermes Agent visual GUI (avatar + OS shell) with Hermes LLM/TTS backend",
        "vrm_guidance": [
            "VRM avatar renders only in the desktop AI Partner OS window (chat tab → VRM mode → load .vrm).",
            "Default chatAvatarMode is 'none'; PNGtuber is exposed on LAN /api/pngtuber, not VRM.",
            "Hermes cannot drive VRM lip-sync without embedded Eel RPC (ver0.2.3: external Eel TCP unavailable).",
        ],
        "notes": [
            "Run `hermes ai-partner-os connect-gui` to bind AI Partner OS as Hermes GUI.",
            "Hermes TTS plays via play_tts_on_pc when Eel works; otherwise local PC speaker fallback.",
            "LAN PIN auth: POST /api/auth — but /api/action rejects OS_ACTIONS in ver0.2.3 (app bug/limit).",
            "Run `hermes ai-partner-os probe-eel` for full port/page scan.",
        ],
    }


def configure(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(values or {})
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    entry = dict(entries.get(PLUGIN_ID) or entries.get("ai_partner_os") or {})

    if values.get("exe_path") is not None:
        entry["exe_path"] = str(_plugin_exe_path(values.get("exe_path")))
    if values.get("eel_port") is not None:
        entry["eel_port"] = _plugin_eel_port(values.get("eel_port"))
    if values.get("lan_host") is not None:
        entry["lan_host"] = _plugin_lan_host(values.get("lan_host"))
    if values.get("lan_port") is not None:
        entry["lan_port"] = _plugin_lan_port(values.get("lan_port"))
    if values.get("bridge_port") is not None:
        entry["bridge_port"] = _plugin_bridge_port(values.get("bridge_port"))
    if values.get("system_prompt") is not None:
        entry["system_prompt"] = _plugin_system_prompt(values.get("system_prompt"))
    if values.get("lan_pin") is not None:
        entry["lan_pin"] = _plugin_lan_pin(values.get("lan_pin"))
    if values.get("tts_provider") is not None:
        entry["tts_provider"] = _plugin_tts_provider(values.get("tts_provider"))
    if values.get("tts_voice") is not None:
        entry["tts_voice"] = values.get("tts_voice")
    if values.get("tts_speed") is not None:
        entry["tts_speed"] = _plugin_tts_speed(values.get("tts_speed"))
    if values.get("gui_mode") is not None:
        entry["gui_mode"] = bool(values.get("gui_mode"))

    entries[PLUGIN_ID] = entry
    save_config(cfg)
    return {"ok": True, "entry": entry, "status": status()}


def start_app(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    exe = _plugin_exe_path(values.get("exe_path"))
    result = process.start_exe(exe)
    if not result.get("ok"):
        return result
    wait = float(values.get("wait_seconds") or 8.0)
    deadline = time.time() + wait
    eel_port = _plugin_eel_port(values.get("eel_port"))
    while time.time() < deadline:
        if eel_client.detect_eel_port(ports=(_resolve_eel_port(eel_port), *eel_client.DEFAULT_EEL_PORTS)):
            result["eel_ready"] = True
            result["eel_port"] = _resolve_eel_port(eel_port)
            break
        time.sleep(0.5)
    else:
        result["eel_ready"] = False
        result["hint"] = "Process started but Eel UI port not detected yet. Open the app window manually."
    return result


def stop_app(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    return process.stop_exe(force=bool(values.get("force")))


def enable_lan(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    enabled = values.get("enabled", True)
    try:
        result = _eel_call("set_lan_enabled", bool(enabled), timeout=20.0)
        return {"ok": True, "enabled": enabled, "lan_status": result}
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "hint": "Start AI Partner OS first, then enable LAN under Settings > Phone.",
        }


def chat(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(values or {})
    user_text = _path_text(values.get("text") or values.get("message"))
    if not user_text:
        return {"ok": False, "error": "text is required"}

    system_prompt = _plugin_system_prompt(values.get("system_prompt"))
    if _plugin_gui_mode(values.get("gui_mode")):
        system_prompt += (
            "\n\nYou control AI Partner OS. When the user asks to open apps or manage tasks, "
            "append [ACTION:name:{\"key\":\"value\"}] tags like the in-app OS action prompt."
        )
    raw_reply = _run_llm(user_text, system_prompt=system_prompt)

    if values.get("present", _plugin_gui_mode()):
        presented = gui_session.present_reply(
            raw_reply,
            eel_call=_eel_call,
            queue_action=lambda a, p: _queue_action(a, p, lan_pin=_plugin_lan_pin()),
            speak=bool(values.get("speak", True)),
            open_chat=bool(values.get("open_chat", True)),
            tts_provider=_plugin_tts_provider(values.get("tts_provider")),
            tts_voice=_plugin_tts_voice(values.get("tts_voice")),
            tts_speed=_plugin_tts_speed(values.get("tts_speed")),
            lan_host=_plugin_lan_host(),
            lan_port=_plugin_lan_port(),
            lan_pin=_plugin_lan_pin(),
        )
        presented["user_text"] = user_text
        return presented

    display_text, actions = gui_session.parse_actions(raw_reply)
    action_results = []
    if values.get("run_actions") and actions:
        for item in actions:
            action_results.append({**item, "queue": _queue_action(item["action"], item.get("params") or {})})

    return {
        "ok": True,
        "text": display_text,
        "raw_text": raw_reply,
        "actions": actions,
        "action_results": action_results,
    }


def speak(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(values or {})
    text = _path_text(values.get("text"))
    if not text:
        return {"ok": False, "error": "text is required"}

    return gui_session.present_reply(
        text,
        eel_call=_eel_call,
        queue_action=lambda a, p: _queue_action(a, p, lan_pin=_plugin_lan_pin(values.get("lan_pin"))),
        speak=True,
        open_chat=bool(values.get("open_chat", True)),
        tts_provider=_plugin_tts_provider(values.get("tts_provider")),
        tts_voice=_plugin_tts_voice(values.get("tts_voice")),
        tts_speed=_plugin_tts_speed(values.get("tts_speed")),
        lan_host=_plugin_lan_host(),
        lan_port=_plugin_lan_port(),
        lan_pin=_plugin_lan_pin(values.get("lan_pin")),
    )


def gui_say(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hermes LLM reply + Hermes TTS + AI Partner OS avatar/UI (GUI mode)."""
    values = dict(values or {})
    user_text = _path_text(values.get("text") or values.get("message"))
    if not user_text:
        return {"ok": False, "error": "text is required"}
    values.setdefault("present", True)
    values.setdefault("gui_mode", True)
    return chat(values)


def connect_gui(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind AI Partner OS as Hermes Agent GUI: app + LAN + Hermes TTS + disable in-app TTS."""
    values = dict(values or {})
    steps: dict[str, Any] = {}

    if values.get("start_app", True):
        steps["start"] = start_app(values)

    if values.get("enable_lan", True):
        try:
            steps["lan"] = enable_lan({"enabled": True})
        except Exception as exc:
            steps["lan"] = {"ok": False, "error": str(exc)}

    lan_hosts = _resolve_lan_hosts(values.get("lan_host"))
    lan_port = _plugin_lan_port(values.get("lan_port"))
    if values.get("discover_lan", True) and any(lan_client.lan_port_open(host=h, port=lan_port) for h in lan_hosts):
        steps["discover_lan"] = discover_lan(values)

    if values.get("start_tts", True):
        steps["tts"] = start_hermes_tts(values)

    if values.get("disable_in_app_tts", True):
        try:
            steps["disable_in_app_tts"] = gui_session.disable_in_app_tts(_eel_call)
        except Exception as exc:
            steps["disable_in_app_tts"] = {"ok": False, "error": str(exc)}

    if values.get("start_bridge", False):
        steps["bridge"] = start_bridge(values)

    configure({"gui_mode": True, **{k: values[k] for k in (
        "tts_provider", "tts_voice", "tts_speed", "exe_path", "eel_port",
        "lan_host", "lan_port", "bridge_port", "lan_pin",
    ) if k in values and values[k] is not None}})

    session = {
        "connected_at": time.time(),
        "gui_mode": True,
        "steps": steps,
    }
    gui_session.write_session(session)

    return {
        "ok": True,
        "gui_mode": True,
        "session": session,
        "status": status(),
        "hint": "Use ai_partner_os_chat / ai_partner_os_gui_say — voice uses Hermes irodori/voicevox.",
    }


def discover_lan(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    return lan_client.discover_lan_api(
        host=_plugin_lan_host(values.get("lan_host")),
        hosts=_resolve_lan_hosts(values.get("lan_host")),
        port=_plugin_lan_port(values.get("lan_port")),
        pin=_plugin_lan_pin(values.get("lan_pin")),
    )


def probe_eel(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    port = _plugin_eel_port(values.get("eel_port"))
    ports = tuple(dict.fromkeys((port, *eel_client.DEFAULT_EEL_PORTS)))
    return eel_client.probe_eel_endpoints(ports=ports)


def hermes_tts_status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    return tts_bridge.tts_status(_plugin_tts_provider(values.get("tts_provider")))


def start_hermes_tts(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    return tts_bridge.start_backend(
        _plugin_tts_provider(values.get("tts_provider")),
        timeout_seconds=int(values.get("timeout_seconds") or 120),
    )


def run_action(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(values or {})
    action = _path_text(values.get("action") or values.get("name"))
    if not action:
        return {"ok": False, "error": "action is required"}
    params = values.get("params") if isinstance(values.get("params"), dict) else {}
    result = _queue_action(action, params, lan_pin=_plugin_lan_pin(values.get("lan_pin")))
    if not result.get("ok"):
        return {
            **result,
            "hint": "Start app, enable LAN, run discover-lan, or keep the desktop UI open.",
        }
    return result


def start_bridge(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    host = _path_text(values.get("host")) or "127.0.0.1"
    if values.get("tailscale"):
        host = "0.0.0.0"
    if not _is_loopback_host(host) and not values.get("confirm_public_host"):
        return {
            "ok": False,
            "confirmation_required": True,
            "reason": "Binding the AI Partner OS bridge outside loopback exposes a noauth WebSocket that can invoke Hermes oneshot.",
            "host": host,
        }
    port = _plugin_bridge_port(values.get("port"))
    state = _read_bridge_state()
    old_pid = int(state.get("pid") or 0)
    if process.pid_alive(old_pid):
        return {"ok": True, "already_running": True, "pid": old_pid, "url": f"ws://127.0.0.1:{port}/ws"}

    worker = Path(__file__).resolve().parent / "bridge_worker.py"
    log_path = get_hermes_home() / "workspace" / "ai-partner-os" / "logs" / "bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    cmd = [
        sys.executable,
        str(worker),
        "--host",
        host,
        "--port",
        str(port),
        "--system-prompt",
        _plugin_system_prompt(values.get("system_prompt")),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=os.name != "nt",
    )
    payload = {"pid": proc.pid, "host": host, "port": port, "log": str(log_path), "cmd": cmd}
    _write_bridge_state(payload)
    return {"ok": True, "pid": proc.pid, "url": f"ws://127.0.0.1:{port}/ws", "log": str(log_path)}


def stop_bridge(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    state = _read_bridge_state()
    pid = int(state.get("pid") or 0)
    if not process.pid_alive(pid):
        _clear_bridge_state()
        return {"ok": True, "stopped": False}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        if values.get("force"):
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, stdin=subprocess.DEVNULL)
    _clear_bridge_state()
    return {"ok": True, "stopped": True, "pid": pid}


STATUS_SCHEMA = {
    "name": "ai_partner_os_status",
    "description": "Show AI Partner OS executable, Eel, LAN, and Hermes bridge readiness.",
    "parameters": {"type": "object", "properties": {}},
}

CONFIGURE_SCHEMA = {
    "name": "ai_partner_os_configure",
    "description": "Save AI Partner OS paths and ports to Hermes config.yaml.",
    "parameters": {
        "type": "object",
        "properties": {
            "exe_path": {"type": "string"},
            "eel_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "lan_host": {"type": "string", "description": "LAN server host (falls back to 127.0.0.1 on same PC)."},
            "lan_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "bridge_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "system_prompt": {"type": "string"},
            "lan_pin": {"type": "string", "description": "LAN PIN for POST /api/auth cookie session."},
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
            "tts_voice": {"type": "string", "description": "irodori voice id or VOICEVOX speaker id."},
            "tts_speed": {"type": "number"},
            "gui_mode": {"type": "boolean", "description": "Treat AI Partner OS as Hermes visual GUI."},
        },
    },
}

START_SCHEMA = {
    "name": "ai_partner_os_start",
    "description": "Launch AI Partner OS executable and wait for the Eel UI port.",
    "parameters": {
        "type": "object",
        "properties": {
            "exe_path": {"type": "string"},
            "wait_seconds": {"type": "number"},
        },
    },
}

STOP_SCHEMA = {
    "name": "ai_partner_os_stop",
    "description": "Stop the Hermes-managed AI Partner OS process.",
    "parameters": {
        "type": "object",
        "properties": {"force": {"type": "boolean"}},
    },
}

ENABLE_LAN_SCHEMA = {
    "name": "ai_partner_os_enable_lan",
    "description": "Toggle AI Partner OS LAN/mobile server (default port 8899).",
    "parameters": {
        "type": "object",
        "properties": {"enabled": {"type": "boolean"}},
    },
}

CHAT_SCHEMA = {
    "name": "ai_partner_os_chat",
    "description": "Chat via Hermes LLM; in GUI mode present reply + Hermes TTS on AI Partner OS avatar.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "present": {"type": "boolean", "description": "Show in app and play Hermes TTS."},
            "speak": {"type": "boolean"},
            "open_chat": {"type": "boolean"},
            "run_actions": {"type": "boolean"},
            "gui_mode": {"type": "boolean"},
            "system_prompt": {"type": "string"},
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
            "tts_voice": {"type": "string"},
            "tts_speed": {"type": "number"},
        },
        "required": ["text"],
    },
}

SPEAK_SCHEMA = {
    "name": "ai_partner_os_speak",
    "description": "Speak via Hermes VOICEVOX/irodoriTTS through AI Partner OS play_tts_on_pc.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "open_chat": {"type": "boolean"},
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
            "tts_voice": {"type": "string"},
            "tts_speed": {"type": "number"},
            "lan_pin": {"type": "string"},
        },
        "required": ["text"],
    },
}

GUI_SAY_SCHEMA = {
    "name": "ai_partner_os_gui_say",
    "description": "Hermes GUI mode: LLM reply + avatar UI + Hermes TTS in one step.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
        },
        "required": ["text"],
    },
}

CONNECT_GUI_SCHEMA = {
    "name": "ai_partner_os_connect_gui",
    "description": "Connect AI Partner OS as Hermes Agent GUI (app, LAN, Hermes TTS, disable in-app TTS).",
    "parameters": {
        "type": "object",
        "properties": {
            "start_app": {"type": "boolean"},
            "enable_lan": {"type": "boolean"},
            "discover_lan": {"type": "boolean"},
            "start_tts": {"type": "boolean"},
            "disable_in_app_tts": {"type": "boolean"},
            "start_bridge": {"type": "boolean"},
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
        },
    },
}

DISCOVER_LAN_SCHEMA = {
    "name": "ai_partner_os_discover_lan",
    "description": "Probe LAN HTTP endpoints on port 8899 and cache working action routes.",
    "parameters": {
        "type": "object",
        "properties": {
            "lan_host": {"type": "string"},
            "lan_port": {"type": "integer"},
            "lan_pin": {"type": "string"},
        },
    },
}

TTS_STATUS_SCHEMA = {
    "name": "ai_partner_os_tts_status",
    "description": "Show Hermes irodoriTTS / VOICEVOX readiness for AI Partner OS voice output.",
    "parameters": {
        "type": "object",
        "properties": {
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
        },
    },
}

START_TTS_SCHEMA = {
    "name": "ai_partner_os_start_tts",
    "description": "Start Hermes TTS backend (irodori script or VOICEVOX engine).",
    "parameters": {
        "type": "object",
        "properties": {
            "tts_provider": {"type": "string", "enum": ["auto", "irodori", "voicevox", "none"]},
            "timeout_seconds": {"type": "integer"},
        },
    },
}

ACTION_SCHEMA = {
    "name": "ai_partner_os_action",
    "description": "Queue an OS action (openWindow, addTask, controlMusic, etc.) in the running app.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "params": {"type": "object"},
            "lan_pin": {"type": "string"},
        },
        "required": ["action"],
    },
}

BRIDGE_START_SCHEMA = {
    "name": "ai_partner_os_bridge_start",
    "description": "Start Hermes External Linkage WebSocket bridge (AITuberKit v2 compatible).",
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "tailscale": {"type": "boolean"},
            "confirm_public_host": {
                "type": "boolean",
                "description": "Required when host/tailscale binds the bridge outside loopback.",
            },
            "system_prompt": {"type": "string"},
        },
    },
}

BRIDGE_STOP_SCHEMA = {
    "name": "ai_partner_os_bridge_stop",
    "description": "Stop the Hermes External Linkage WebSocket bridge.",
    "parameters": {
        "type": "object",
        "properties": {"force": {"type": "boolean"}},
    },
}


def handle_status(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(status(args or {}))


def handle_configure(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(configure(args or {}))


def handle_start(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(start_app(args or {}))


def handle_stop(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(stop_app(args or {}))


def handle_enable_lan(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(enable_lan(args or {}))


def handle_chat(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(chat(args or {}))


def handle_speak(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(speak(args or {}))


def handle_action(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(run_action(args or {}))


def handle_bridge_start(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(start_bridge(args or {}))


def handle_bridge_stop(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(stop_bridge(args or {}))


def handle_gui_say(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(gui_say(args or {}))


def handle_connect_gui(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(connect_gui(args or {}))


def handle_discover_lan(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(discover_lan(args or {}))


def handle_tts_status(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(hermes_tts_status(args or {}))


def handle_start_tts(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(start_hermes_tts(args or {}))


def conversation_prompt(event: dict[str, Any]) -> str:
    return (
        "You are responding through AI Partner OS, the Hermes Agent visual GUI. "
        "Keep replies concise and spoken-friendly for TTS. "
        "Use [ACTION:name:{json}] when the user wants OS operations."
    )


def matches_conversation_event(event: dict[str, Any]) -> bool:
    if not _plugin_gui_mode():
        return False
    channel = str(event.get("channel") or event.get("source") or "").lower()
    if channel in {"ai-partner-os", "ai_partner_os", "gui", "desktop"}:
        return True
    meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    return str(meta.get("surface") or "").lower() in {"ai-partner-os", "gui"}


def handle_slash(args: str = "", **kwargs: Any) -> str:
    parts = (args or "").strip().split()
    command = parts[0].lower() if parts else "status"
    if command in {"status", ""}:
        return handle_status({})
    if command == "configure":
        return handle_configure({})
    if command == "start":
        return handle_start({})
    if command == "stop":
        return handle_stop({})
    if command in {"connect-gui", "connect", "gui"}:
        return handle_connect_gui({})
    if command in {"discover-lan", "discover"}:
        return handle_discover_lan({})
    if command in {"tts-status", "tts"}:
        return handle_tts_status({})
    if command == "start-tts":
        return handle_start_tts({})
    if command in {"lan-on", "enable-lan"}:
        return handle_enable_lan({"enabled": True})
    if command in {"lan-off", "disable-lan"}:
        return handle_enable_lan({"enabled": False})
    if command == "chat":
        text = " ".join(parts[1:]).strip()
        return handle_chat({"text": text, "present": True}) if text else handle_chat({"text": "こんにちは", "present": True})
    if command in {"gui-say", "say"}:
        text = " ".join(parts[1:]).strip()
        return handle_gui_say({"text": text}) if text else _json({"ok": False, "error": "usage: /ai-partner-os gui-say <text>"})
    if command == "speak":
        text = " ".join(parts[1:]).strip()
        return handle_speak({"text": text}) if text else _json({"ok": False, "error": "usage: /ai-partner-os speak <text>"})
    if command == "bridge-start":
        return handle_bridge_start({})
    if command == "bridge-stop":
        return handle_bridge_stop({})
    return _json(
        {
            "commands": [
                "status",
                "configure",
                "connect-gui",
                "discover-lan",
                "start",
                "stop",
                "tts-status",
                "start-tts",
                "lan-on",
                "lan-off",
                "chat <text>",
                "gui-say <text>",
                "speak <text>",
                "bridge-start",
                "bridge-stop",
            ]
        }
    )
