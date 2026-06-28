"""VRChat autonomous chatbox, conversation loop, and movement plugin."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

from .movement import send_move
from . import neuro as neuro_sdk

PLUGIN_ID = "vrchat-autonomy"
DEFAULT_INTERVAL_SEC = 15.0
LIVE_ACK = "I understand this sends OSC and/or audio to VRChat."


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def plugin_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}
    plugins = cfg.get("plugins") if isinstance(cfg.get("plugins"), dict) else {}
    section = plugins.get(PLUGIN_ID) if isinstance(plugins.get(PLUGIN_ID), dict) else {}
    return dict(section)


def profile_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or plugin_config()
    raw = str(cfg.get("profile_path") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return get_hermes_home() / "config" / "vrchat-autonomy-profile.json"


def worker_pid_path() -> Path:
    return get_hermes_home() / "state" / "vrchat-autonomy-worker.pid"


def worker_log_path() -> Path:
    return get_hermes_home() / "logs" / "vrchat-autonomy-worker.log"


def check_available() -> bool:
    try:
        import pythonosc  # noqa: F401

        return True
    except ImportError:
        return False


def _require_osc() -> dict[str, Any] | None:
    if check_available():
        return None
    return {
        "ok": False,
        "error": "python-osc missing — install with: uv pip install 'hermes-agent[vrchat]'",
    }


def _profile_gate(*, need_chatbox: bool = False, need_movement: bool = False) -> dict[str, Any] | None:
    from tools.openclaw.vrchat_autonomy import load_autonomy_profile

    loaded = load_autonomy_profile(profile_path())
    if not loaded.get("success"):
        return {"ok": False, "error": "profile_invalid", "profile": loaded}
    profile = loaded["profile"]
    if profile.get("dry_run", True):
        return {"ok": False, "error": "dry_run_enabled", "hint": "Set dry_run=false and live_actuation_ack in profile"}
    if profile.get("mode") == "observe":
        return {"ok": False, "error": "observe_mode", "hint": "Set mode to private_test or trusted_instance"}
    if need_chatbox and not profile.get("allow_chatbox"):
        return {"ok": False, "error": "chatbox_not_allowed", "hint": "Set allow_chatbox=true in profile"}
    if need_movement and not profile.get("allow_movement"):
        return {"ok": False, "error": "movement_not_allowed", "hint": "Set allow_movement=true in profile"}
    ack = str(profile.get("live_actuation_ack") or "").strip()
    if ack != LIVE_ACK:
        return {
            "ok": False,
            "error": "live_ack_missing",
            "required_ack": LIVE_ACK,
            "hint": f'Set live_actuation_ack exactly to: {LIVE_ACK}',
        }
    return None


def status() -> dict[str, Any]:
    from tools.openclaw.vrchat_autonomy import load_autonomy_profile, vrchat_autonomy_readiness

    cfg = plugin_config()
    loaded = load_autonomy_profile(profile_path(cfg))
    profile = loaded.get("profile") or {}
    readiness = vrchat_autonomy_readiness(
        voicevox_url=str(profile.get("voicevox_url") or "http://127.0.0.1:50021"),
        harness_url=str(profile.get("harness_url") or "http://127.0.0.1:18794"),
        audio_output_device=profile.get("audio_output_device") or None,
        require_harness=bool(profile.get("require_harness", False)),
        tts_backend=str(profile.get("tts_backend") or "voicevox"),
        irodori_base_url=profile.get("irodori_base_url") or None,
        require_voice=bool(profile.get("allow_voice", False)),
    )
    worker = worker_status()
    prof = profile_path(cfg)
    neuro_summary = neuro_sdk.neuro_status(profile=prof, config=cfg)
    return {
        "ok": True,
        "plugin_id": PLUGIN_ID,
        "python_osc": check_available(),
        "profile_path": str(prof),
        "profile": loaded,
        "readiness": readiness,
        "neuro": neuro_summary,
        "neuro_readiness": neuro_sdk.neuro_readiness(cfg),
        "worker": worker,
        "config": cfg,
    }


def doctor() -> dict[str, Any]:
    blocked = _require_osc()
    if blocked:
        return blocked
    try:
        from tools.openclaw.vrchat_preflight import build_preflight_bundle

        preflight = build_preflight_bundle(profile_path=str(profile_path()))
    except Exception as exc:
        preflight = {"success": False, "error": str(exc)}
    base = status()
    base["preflight"] = preflight
    neuro_ready = bool((base.get("neuro_readiness") or {}).get("vendor_ok"))
    base["neuro_ready"] = neuro_ready
    core_ok = bool(base.get("python_osc")) and bool((base.get("readiness") or {}).get("ready"))
    # Neuro API vendor is optional for core VRChat autonomy; required only for Neuro bridge flows.
    base["core_ok"] = core_ok
    base["ok"] = core_ok
    base["neuro_bridge_ready"] = core_ok and neuro_ready
    if not neuro_ready:
        base.setdefault("hints", []).append(
            "Neuro bridge: git submodule update --init vendor/neuro-sdk "
            "(or hermes vrchat-autonomy neuro vendor)"
        )
    return base


def setup(
    *,
    enable_plugin: bool = True,
    allow_chatbox: bool = True,
    allow_movement: bool = True,
    allow_voice: bool = True,
    mode: str = "private_test",
    arm_live: bool = False,
) -> dict[str, Any]:
    from tools.openclaw.vrchat_profile import prepare_autonomy_profile

    home = get_hermes_home()
    config_path = home / "config.yaml"
    config = _load_yaml(config_path)
    if enable_plugin:
        plugins = config.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            plugins = {}
            config["plugins"] = plugins
        enabled = plugins.setdefault("enabled", [])
        if not isinstance(enabled, list):
            enabled = []
            plugins["enabled"] = enabled
        if PLUGIN_ID not in enabled:
            enabled.append(PLUGIN_ID)
        section = plugins.setdefault(PLUGIN_ID, {})
        if not isinstance(section, dict):
            section = {}
            plugins[PLUGIN_ID] = section
        section.setdefault("interval_sec", DEFAULT_INTERVAL_SEC)
        section.setdefault("neuro_game", neuro_sdk.resolve_game_name(section))
        section.setdefault("neuro_ws_url", neuro_sdk.resolve_ws_url(section))
        section["profile_path"] = str(profile_path(section))
        _save_yaml(config_path, config)

    profile_result = prepare_autonomy_profile(
        profile_path=profile_path(),
        enabled=True,
        mode=mode,
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        arm_live=arm_live,
        live_ack=LIVE_ACK if arm_live else "",
    )
    return {
        "ok": bool(profile_result.get("success")),
        "config_path": str(config_path),
        "profile": profile_result,
        "arm_live": arm_live,
        "live_ack_required": LIVE_ACK,
        "next": [
            "Enable OSC in VRChat Action Menu",
            "Start Irodori TTS (hermes irodori-tts start) when tts_backend=irodori",
            "Or start VOICEVOX Engine when tts_backend=voicevox",
            f"Run: hermes vrchat-autonomy doctor",
            "For live actuation: hermes vrchat-autonomy setup --arm-live",
            "Start loop: hermes vrchat-autonomy start",
            "Neuro API: hermes vrchat-autonomy neuro status",
            "Neuro bridge: py -3 scripts/vrchat_neuro_bridge.py --profile <profile.json>",
        ],
    }


def arm_live_profile(
    *,
    allow_chatbox: bool = True,
    allow_movement: bool = True,
    allow_voice: bool = True,
    mode: str = "private_test",
) -> dict[str, Any]:
    """Set dry_run=false and write the exact live actuation ACK into the profile."""
    from tools.openclaw.vrchat_profile import prepare_autonomy_profile

    result = prepare_autonomy_profile(
        profile_path=profile_path(),
        enabled=True,
        mode=mode,
        allow_voice=allow_voice,
        allow_chatbox=allow_chatbox,
        allow_movement=allow_movement,
        arm_live=True,
        live_ack=LIVE_ACK,
    )
    checks = {
        "live_armed": not bool((result.get("profile") or {}).get("dry_run", True)),
        "ack_ok": (result.get("profile") or {}).get("live_actuation_ack") == LIVE_ACK,
        "allow_movement": bool((result.get("profile") or {}).get("allow_movement")),
        "mode": (result.get("profile") or {}).get("mode"),
    }
    gate = _profile_gate(need_movement=True) if checks["live_armed"] else {"error": "not_live_armed"}
    return {
        "ok": bool(result.get("success")),
        "profile_result": result,
        "live_ack": LIVE_ACK,
        "checks": checks,
        "move_gate_preview": gate,
        "next": [
            "VRChat を起動し Action Menu で OSC を有効化",
            "hermes vrchat-autonomy doctor",
            "hermes vrchat-autonomy move forward",
        ],
    }


def send_chatbox(text: str, *, immediate: bool = True) -> dict[str, Any]:
    blocked = _require_osc() or _profile_gate(need_chatbox=True)
    if blocked:
        return blocked
    from tools.vrchat_osc_tool import vrchat_chatbox

    result = vrchat_chatbox(text, immediate=immediate)
    return {"ok": bool(result.get("success")), "result": result}


def move(direction: str, *, value: float = 1.0, duration_ms: int = 400) -> dict[str, Any]:
    blocked = _require_osc() or _profile_gate(need_movement=True)
    if blocked:
        return blocked
    if str((load_profile_summary().get("profile") or {}).get("mode")) == "public":
        return {"ok": False, "error": "movement_blocked_in_public_mode"}
    result = send_move(direction, value=value, duration_ms=duration_ms)
    return {"ok": bool(result.get("success")), "result": result}


def load_profile_summary() -> dict[str, Any]:
    from tools.openclaw.vrchat_autonomy import load_autonomy_profile

    return load_autonomy_profile(profile_path())


def run_tick(
    observations: list[dict[str, Any]] | None = None,
    *,
    emergency_stop: bool = False,
) -> dict[str, Any]:
    blocked = _require_osc()
    if blocked:
        return blocked
    from tools.openclaw.vrchat_autonomy import vrchat_autonomy_profile_tick

    tick = vrchat_autonomy_profile_tick(
        profile_path=profile_path(),
        observations=observations,
        emergency_stop=emergency_stop,
    )
    return {"ok": bool(tick.get("success")), "tick": tick}


def enqueue_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.openclaw.vrchat_observations import ingest_observations

    result = ingest_observations(observations)
    return {"ok": bool(result.get("success", True)), "result": result}


def worker_status() -> dict[str, Any]:
    pid_file = worker_pid_path()
    if not pid_file.is_file():
        return {"running": False, "pid": None}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return {"running": False, "pid": None, "error": "invalid_pid_file"}
    if not _pid_alive(pid):
        return {"running": False, "pid": pid, "stale": True}
    return {"running": True, "pid": pid, "log_path": str(worker_log_path())}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback after nt branch above
    except OSError:
        return False
    return True


def start_worker(*, interval_sec: float | None = None) -> dict[str, Any]:
    blocked = _require_osc()
    if blocked:
        return blocked
    current = worker_status()
    if current.get("running"):
        return {"ok": True, "already_running": True, **current}

    cfg = plugin_config()
    interval = float(interval_sec or cfg.get("interval_sec") or DEFAULT_INTERVAL_SEC)
    worker_script = Path(__file__).with_name("loop_worker.py")
    log_path = worker_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HERMES_VRCHAT_AUTONOMY_PROFILE"] = str(profile_path(cfg))
    env["HERMES_VRCHAT_AUTONOMY_INTERVAL"] = str(interval)

    log_handle = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(worker_script)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    worker_pid_path().parent.mkdir(parents=True, exist_ok=True)
    worker_pid_path().write_text(str(proc.pid), encoding="utf-8")
    return {
        "ok": True,
        "pid": proc.pid,
        "interval_sec": interval,
        "log_path": str(log_path),
        "profile_path": str(profile_path(cfg)),
    }


def stop_worker(*, emergency_stop: bool = True) -> dict[str, Any]:
    tick = run_tick(emergency_stop=emergency_stop) if emergency_stop else None
    current = worker_status()
    pid = current.get("pid")
    if not pid or not current.get("running"):
        return {"ok": True, "stopped": False, "worker": current, "emergency_tick": tick}

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.kill(int(pid), signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "pid": pid}

    try:
        worker_pid_path().unlink(missing_ok=True)
    except OSError:
        pass
    return {"ok": True, "stopped": True, "pid": pid, "emergency_tick": tick}


# --- Plugin tool schemas ---

STATUS_SCHEMA = {
    "name": "vrchat_autonomy_plugin_status",
    "description": "Readiness, profile, and background worker state for the VRChat autonomy plugin.",
    "parameters": {"type": "object", "properties": {}},
}

CHATBOX_SCHEMA = {
    "name": "vrchat_autonomy_plugin_chatbox",
    "description": (
        "Send VRChat ChatBox text via OSC when the operator profile allows live chatbox actuation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "ChatBox message (max 144 chars)."},
            "immediate": {"type": "boolean", "description": "Show immediately. Default true."},
        },
        "required": ["text"],
    },
}

MOVE_SCHEMA = {
    "name": "vrchat_autonomy_plugin_move",
    "description": (
        "Pulse VRChat movement input via OSC (/input/MoveForward etc.) when profile allows movement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "description": "forward, back, left, right, jump, run, or stop.",
            },
            "value": {"type": "number", "description": "Input magnitude 0-1. Default 1."},
            "duration_ms": {"type": "integer", "description": "Hold duration before reset. Default 400."},
        },
        "required": ["direction"],
    },
}

TICK_SCHEMA = {
    "name": "vrchat_autonomy_plugin_tick",
    "description": (
        "Run one VRChat autonomy loop tick: consume queued observations, call the auxiliary LLM, "
        "and actuate chatbox/voice/avatar actions per profile gates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Optional inline observations for this tick.",
            },
            "emergency_stop": {
                "type": "boolean",
                "description": "Emergency stop the loop state without actuation.",
            },
        },
    },
}

ENQUEUE_SCHEMA = {
    "name": "vrchat_autonomy_plugin_enqueue",
    "description": "Queue textBox/operator observations for the autonomous VRChat conversation loop.",
    "parameters": {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Observation objects with source and text fields.",
            },
        },
        "required": ["observations"],
    },
}

NEURO_STATUS_SCHEMA = {
    "name": "vrchat_autonomy_plugin_neuro_status",
    "description": (
        "Read-only Neuro API (VedalAI neuro-sdk) vendor status, game name, and action catalog "
        "for the VRChat autonomy profile. Does not open a websocket."
    ),
    "parameters": {"type": "object", "properties": {}},
}

NEURO_BOOTSTRAP_SCHEMA = {
    "name": "vrchat_autonomy_plugin_neuro_bootstrap",
    "description": (
        "Build Neuro API startup/context/actions/register websocket messages for Hermes VRChat. "
        "Does not connect to Neuro or VRChat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {"type": "string", "description": "Optional initial context after startup."},
            "silent_context": {
                "type": "boolean",
                "description": "Mark context as silent. Default true.",
            },
        },
    },
}

NEURO_HANDLE_SCHEMA = {
    "name": "vrchat_autonomy_plugin_neuro_handle_action",
    "description": (
        "Validate one incoming Neuro API action message and route through local VRChat safety gates. "
        "Live OSC/audio only when profile is enabled, armed, and not dry-run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "object",
                "description": "Neuro websocket message with command=action.",
            },
            "retry_on_failure": {
                "type": "boolean",
                "description": "Return action/result success=false on rejection for Neuro retry.",
            },
            "force_dry_run": {
                "type": "boolean",
                "description": "Force dry-run even when profile is live-armed.",
            },
        },
        "required": ["message"],
    },
}


def handle_status(_args: dict, **kwargs) -> str:
    return to_json(status())


def handle_chatbox(args: dict, **kwargs) -> str:
    return to_json(send_chatbox(args.get("text", ""), immediate=bool(args.get("immediate", True))))


def handle_move(args: dict, **kwargs) -> str:
    return to_json(
        move(
            args.get("direction", ""),
            value=float(args.get("value", 1.0)),
            duration_ms=int(args.get("duration_ms", 400)),
        )
    )


def handle_tick(args: dict, **kwargs) -> str:
    obs = list(args.get("observations") or [])
    return to_json(
        run_tick(observations=obs or None, emergency_stop=bool(args.get("emergency_stop", False)))
    )


def handle_enqueue(args: dict, **kwargs) -> str:
    return to_json(enqueue_observations(list(args.get("observations") or [])))


def handle_neuro_status(_args: dict, **kwargs) -> str:
    cfg = plugin_config()
    return to_json(neuro_sdk.neuro_status(profile=profile_path(cfg), config=cfg))


def handle_neuro_bootstrap(args: dict, **kwargs) -> str:
    cfg = plugin_config()
    return to_json(
        neuro_sdk.neuro_bootstrap(
            profile=profile_path(cfg),
            config=cfg,
            context=str(args.get("context") or ""),
            silent_context=bool(args.get("silent_context", True)),
        )
    )


def handle_neuro_action(args: dict, **kwargs) -> str:
    cfg = plugin_config()
    return to_json(
        neuro_sdk.neuro_handle_action(
            args.get("message") or {},
            profile=profile_path(cfg),
            config=cfg,
            retry_on_failure=bool(args.get("retry_on_failure", False)),
            force_dry_run=bool(args.get("force_dry_run", False)),
        )
    )


def handle_slash(args: str) -> str:
    parts = (args or "").strip().split()
    if not parts or parts[0] in {"status", "st"}:
        return to_json(status())
    if parts[0] == "doctor":
        return to_json(doctor())
    if parts[0] == "tick":
        return to_json(run_tick())
    if parts[0] == "start":
        return to_json(start_worker())
    if parts[0] == "stop":
        return to_json(stop_worker())
    if parts[0] == "neuro":
        sub = parts[1] if len(parts) > 1 else "status"
        cfg = plugin_config()
        prof = profile_path(cfg)
        if sub in {"status", "st"}:
            return to_json(neuro_sdk.neuro_status(profile=prof, config=cfg))
        if sub == "bootstrap":
            return to_json(neuro_sdk.neuro_bootstrap(profile=prof, config=cfg))
        return (
            "Usage: /vrchat-autonomy neuro [status|bootstrap]\n"
            "CLI: hermes vrchat-autonomy neuro status|bootstrap|build-messages|handle"
        )
    return (
        "Usage: /vrchat-autonomy [status|doctor|tick|start|stop|neuro]\n"
        f"Profile: {display_hermes_home()}/config/vrchat-autonomy-profile.json"
    )
