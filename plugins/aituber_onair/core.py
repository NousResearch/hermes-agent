"""Core implementation for the AITuber OnAir Hermes plugin."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - early import safety
    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


PLUGIN_ID = "aituber-onair"
PLUGIN_NAME = "aituber-onair"
CONFIG_ALIASES = (PLUGIN_ID, "aituber_onair", "aituber")
TOOLSET = "aituber-onair"
DEFAULT_FBX_PORT = 5174
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RESPONSE_LENGTH = "short"
DEFAULT_TTS_PROVIDER = "auto"
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"
DEFAULT_VOICEVOX_SPEAKER = 8
DEFAULT_TTS_FORMAT = "wav"
SUPPORTED_TTS_PROVIDERS = {"auto", "irodori", "voicevox", "none"}

DEFAULT_HAKUA_SYSTEM_PROMPT = (
    "あなたは「はくあ」、Codex Authで動くAIVTuberです。"
    "配信中の相手に、日本語で短く、落ち着いて、品よく返答してください。"
    "返答の先頭には必ず [happy] [sad] [angry] [surprised] [relaxed] [neutral] "
    "のどれか一つを置いてください。"
    "秘密情報、認証情報、ローカルファイルの中身は話題に出さず、"
    "見えていない映像や音声を操作できるとは言い切らないでください。"
)

STATUS_SCHEMA = {
    "name": "aituber_onair_status",
    "description": "Show AITuber OnAir bridge readiness without changing files.",
    "parameters": {"type": "object", "properties": {}},
}

CONFIGURE_HAKUA_SCHEMA = {
    "name": "aituber_onair_configure_hakua",
    "description": "Save non-secret Hakua AIVTuber settings to Hermes config.yaml.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Path to the aituber-onair checkout.",
            },
            "model": {
                "type": "string",
                "description": "Optional Codex SDK model id. Empty uses the local Codex CLI default.",
            },
            "fbx_port": {
                "type": "integer",
                "minimum": 1024,
                "maximum": 65535,
                "description": "Local Vite port for the FBX app.",
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional Hakua system prompt override.",
            },
            "tts_provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox", "none"],
                "description": "Local voice backend for Hakua. auto prefers irodoriTTS, then VOICEVOX.",
            },
            "voicevox_url": {
                "type": "string",
                "description": "VOICEVOX Engine URL. Default: http://127.0.0.1:50021",
            },
            "voicevox_speaker": {
                "type": "integer",
                "description": "VOICEVOX speaker/style id.",
            },
            "voicevox_engine_exe": {
                "type": "string",
                "description": "Optional path to vv-engine/run.exe.",
            },
            "tts_voice": {
                "type": "string",
                "description": "Default irodoriTTS voice id. Use hakua for the local Hakua reference voice.",
            },
            "tts_speed": {
                "type": "number",
                "description": "Default local TTS speed multiplier.",
            },
        },
    },
}

PREPARE_SCHEMA = {
    "name": "aituber_onair_prepare",
    "description": "Prepare the local AITuber OnAir checkout for Codex SDK character chat.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "install_codex_sdk": {
                "type": "boolean",
                "description": "Install @openai/codex-sdk locally without saving package metadata.",
            },
            "build_chat": {
                "type": "boolean",
                "description": "Build @aituber-onair/chat before running Hakua.",
            },
            "build_fbx_app": {
                "type": "boolean",
                "description": "Build the FBX React app after preparation.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 1800,
            },
        },
    },
}

START_SCHEMA = {
    "name": "aituber_onair_start",
    "description": "Start the local AITuber OnAir FBX React app.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "fbx_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "force": {
                "type": "boolean",
                "description": "Stop an existing plugin-managed FBX app before starting.",
            },
        },
    },
}

STOP_SCHEMA = {
    "name": "aituber_onair_stop",
    "description": "Stop the plugin-managed AITuber OnAir FBX app.",
    "parameters": {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Use platform force-kill semantics if a graceful stop does not finish.",
            },
        },
    },
}

SAY_SCHEMA = {
    "name": "aituber_onair_say",
    "description": "Ask Hakua to reply once through AITuber OnAir's Codex SDK character chat.",
    "parameters": {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "repo_root": {"type": "string"},
            "model": {"type": "string"},
            "response_length": {
                "type": "string",
                "enum": ["veryShort", "short", "medium", "long"],
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 900,
            },
            "speak": {
                "type": "boolean",
                "description": "Also synthesize Hakua's reply through the configured local TTS backend.",
            },
            "tts_provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
            },
            "output_path": {"type": "string"},
            "play": {"type": "boolean"},
            "tts_voice": {"type": "string"},
            "tts_speed": {"type": "number"},
        },
    },
}

SMOKE_SCHEMA = {
    "name": "aituber_onair_smoke",
    "description": "Run a short Hakua Codex-auth one-shot prompt as a readiness test.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 900,
            },
        },
    },
}

TTS_STATUS_SCHEMA = {
    "name": "aituber_onair_tts_status",
    "description": "Show local irodoriTTS and VOICEVOX readiness for Hakua voice output.",
    "parameters": {"type": "object", "properties": {}},
}

START_TTS_SCHEMA = {
    "name": "aituber_onair_start_tts",
    "description": "Start the configured local TTS backend for Hakua.",
    "parameters": {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
                "description": "TTS backend to start. auto prefers irodoriTTS, then VOICEVOX.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 180,
            },
            "voicevox_url": {"type": "string"},
            "voicevox_speaker": {"type": "integer"},
        },
    },
}

SPEAK_SCHEMA = {
    "name": "aituber_onair_speak",
    "description": "Synthesize Hakua speech through local irodoriTTS or VOICEVOX.",
    "parameters": {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
            },
            "output_path": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["wav", "mp3", "flac", "opus", "aac", "pcm"],
            },
            "voice": {"type": "string"},
            "model": {"type": "string"},
            "speed": {"type": "number"},
            "voicevox_speaker": {"type": "integer"},
            "play": {
                "type": "boolean",
                "description": "Play the synthesized wav locally after writing it.",
            },
        },
    },
}


def check_available() -> bool:
    return True


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _workspace_root() -> Path:
    return get_hermes_home() / "workspace" / "aituber-onair"


def _active_file() -> Path:
    return _workspace_root() / "active.json"


def _tts_active_file() -> Path:
    return _workspace_root() / "tts-active.json"


def _log_file(name: str) -> Path:
    path = _workspace_root() / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _audio_dir() -> Path:
    path = _workspace_root() / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


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


def _path_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _is_aituber_repo(path: Path) -> bool:
    package_json = path / "package.json"
    if not package_json.is_file():
        return False
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("name") == "aituber-onair" or (
        path / "packages" / "core" / "package.json"
    ).is_file()


def _default_repo_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "aituber-onair")
    candidates.extend(
        [
            Path.cwd() / "aituber-onair",
            Path.cwd().parent / "aituber-onair",
            Path.home() / "Documents" / "New project" / "aituber-onair",
        ]
    )
    return candidates


def resolve_repo_root(explicit: str | None = None) -> Path | None:
    cfg = _plugin_config()
    candidates = [
        explicit,
        cfg.get("repo_root"),
        os.environ.get("AITUBER_ONAIR_REPO"),
    ]
    for raw in candidates:
        text = _path_text(raw)
        if not text:
            continue
        path = Path(text).expanduser()
        if _is_aituber_repo(path):
            return path
    for path in _default_repo_candidates():
        if _is_aituber_repo(path):
            return path
    return None


def _resolve_required_repo(explicit: str | None = None) -> tuple[Path | None, dict[str, Any] | None]:
    repo = resolve_repo_root(explicit)
    if repo is None:
        return None, {
            "ok": False,
            "error": "aituber-onair checkout was not found.",
            "configure": "hermes aituber-onair configure --repo-root <path-to-aituber-onair>",
            "env": "AITUBER_ONAIR_REPO",
        }
    return repo, None


def _fbx_app_dir(repo_root: Path, explicit: str | None = None) -> Path:
    cfg = _plugin_config()
    raw = explicit or cfg.get("fbx_app_dir")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "packages" / "core" / "examples" / "react-fbx-app"


def _codex_chat_script(repo_root: Path) -> Path:
    cfg = _plugin_config()
    raw = cfg.get("codex_character_cli")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "packages" / "chat" / "examples" / "codex-character-chat" / "index.js"


def _chat_agent_dist(repo_root: Path) -> Path:
    return repo_root / "packages" / "chat" / "dist" / "cjs" / "agent.js"


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _is_windows() -> bool:
    return os.name == "nt"


def _node_exe() -> str | None:
    cfg = _plugin_config()
    configured = _path_text(cfg.get("node_exe"))
    if configured:
        return configured
    return _which("node")


def _npm_exe() -> str | None:
    cfg = _plugin_config()
    configured = _path_text(cfg.get("npm_exe"))
    if configured:
        return configured
    return _which("npm")


def _plugin_model(explicit: str | None = None) -> str:
    cfg = _plugin_config()
    return _path_text(explicit or cfg.get("model"))


def _plugin_response_length(explicit: str | None = None) -> str:
    cfg = _plugin_config()
    return _path_text(explicit or cfg.get("response_length")) or DEFAULT_RESPONSE_LENGTH


def _plugin_working_directory(repo_root: Path) -> str:
    cfg = _plugin_config()
    raw = _path_text(cfg.get("working_directory"))
    if raw:
        return raw
    return str(repo_root)


def _plugin_system_prompt() -> str:
    cfg = _plugin_config()
    return str(cfg.get("system_prompt") or DEFAULT_HAKUA_SYSTEM_PROMPT)


def _plugin_character_name() -> str:
    cfg = _plugin_config()
    return str(cfg.get("character_name") or "はくあ")


def _plugin_fbx_port(explicit: Any = None) -> int:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("fbx_port")
    if raw is None:
        return DEFAULT_FBX_PORT
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_FBX_PORT
    return port if 1024 <= port <= 65535 else DEFAULT_FBX_PORT


def _plugin_tts_provider(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("tts_provider")
    provider = _path_text(raw).lower() or DEFAULT_TTS_PROVIDER
    if provider in {"off", "disabled"}:
        return "none"
    return provider if provider in SUPPORTED_TTS_PROVIDERS else DEFAULT_TTS_PROVIDER


def _plugin_voicevox_url(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_url")
    return (_path_text(raw) or os.environ.get("VOICEVOX_URL") or DEFAULT_VOICEVOX_URL).rstrip("/")


def _plugin_voicevox_speaker(explicit: Any = None) -> int:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_speaker")
    if raw is None:
        raw = os.environ.get("VOICEVOX_SPEAKER")
    if raw is None:
        return DEFAULT_VOICEVOX_SPEAKER
    try:
        speaker = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_VOICEVOX_SPEAKER
    return speaker if speaker >= 0 else DEFAULT_VOICEVOX_SPEAKER


def _plugin_voicevox_engine_exe(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_engine_exe")
    return _path_text(raw) or os.environ.get("VOICEVOX_ENGINE_EXE", "")


def _plugin_tts_voice(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("tts_voice")
    return _path_text(raw)


def _plugin_tts_speed(explicit: Any = None) -> float | None:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("tts_speed")
    if raw in (None, ""):
        return None
    try:
        return max(0.25, min(4.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _coerce_tts_format(value: Any = None) -> str:
    fmt = _path_text(value).lower().lstrip(".")
    return fmt if fmt in {"wav", "mp3", "flac", "opus", "aac", "pcm"} else DEFAULT_TTS_FORMAT


def _bounded(text: str, limit: int = 16000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[TRUNCATED]"


def _process_output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"Command not found: {exc.filename}"}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"Command timed out after {timeout_seconds}s.",
            "stdout": _bounded(_process_output_text(exc.stdout)),
            "stderr": _bounded(_process_output_text(exc.stderr)),
        }
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": cmd,
        "cwd": str(cwd),
        "stdout": _bounded(completed.stdout or ""),
        "stderr": _bounded(completed.stderr or ""),
    }


def _codex_cli_auth_status() -> dict[str, Any]:
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    auth_path = codex_home / "auth.json"
    result: dict[str, Any] = {
        "path": str(auth_path),
        "exists": auth_path.is_file(),
        "parsed": False,
        "has_access_token": False,
        "has_refresh_token": False,
        "auth_mode": "",
    }
    if not auth_path.is_file():
        return result
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        return result
    tokens = data.get("tokens", {}) if isinstance(data, dict) else {}
    if not isinstance(tokens, dict):
        tokens = {}
    result.update(
        {
            "parsed": True,
            "auth_mode": str(data.get("auth_mode") or ""),
            "has_access_token": bool(tokens.get("access_token")),
            "has_refresh_token": bool(tokens.get("refresh_token")),
            "account_id_present": bool(tokens.get("account_id")),
        }
    )
    return result


def _hermes_codex_auth_status() -> dict[str, Any]:
    auth_path = get_hermes_home() / "auth.json"
    result: dict[str, Any] = {
        "path": str(auth_path),
        "exists": auth_path.is_file(),
        "parsed": False,
        "provider_entry": False,
        "credential_pool_entries": 0,
    }
    if not auth_path.is_file():
        return result
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        return result
    providers = data.get("providers", {}) if isinstance(data, dict) else {}
    pool = data.get("credential_pool", {}) if isinstance(data, dict) else {}
    provider_entry = isinstance(providers, dict) and "openai-codex" in providers
    pool_entries = 0
    if isinstance(pool, dict):
        value = pool.get("openai-codex")
        if isinstance(value, list):
            pool_entries = len(value)
        elif isinstance(value, dict):
            pool_entries = 1
    result.update(
        {
            "parsed": True,
            "provider_entry": provider_entry,
            "credential_pool_entries": pool_entries,
        }
    )
    return result


def _codex_sdk_installed(repo_root: Path) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "installed": False, "error": "npm was not found on PATH."}
    result = _run_command(
        [npm, "ls", "@openai/codex-sdk", "--workspaces=false", "--depth=0"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    stdout = result.get("stdout", "")
    installed = result.get("ok") is True and "@openai/codex-sdk" in stdout
    return {
        "ok": result.get("ok") is True,
        "installed": installed,
        "exit_code": result.get("exit_code"),
        "stdout": stdout.strip(),
        "stderr": str(result.get("stderr") or "").strip(),
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception:
        return False


def _active_status() -> dict[str, Any]:
    active = _read_json_file(_active_file())
    if not active:
        return {"ok": False, "reason": "no active AITuber OnAir process"}
    pid = int(active.get("pid") or 0)
    alive = _pid_alive(pid)
    return {**active, "ok": True, "alive": alive, "pid": pid}


def _url_ready(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, bytes]:
    headers: dict[str, str] = {}
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status), response.read()


def _voicevox_url_parts(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 50021)
    return host, port


def _voicevox_engine_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = _plugin_voicevox_engine_exe()
    if configured:
        candidates.append(Path(configured).expanduser())
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(
            [
                Path(local_appdata) / "Programs" / "VOICEVOX" / "vv-engine" / "run.exe",
                Path(local_appdata) / "voicevox-engine" / "voicevox-engine" / "run.exe",
            ]
        )
    candidates.extend(
        [
            Path("C:/Program Files/VOICEVOX/vv-engine/run.exe"),
            Path("C:/Program Files (x86)/VOICEVOX/vv-engine/run.exe"),
        ]
    )

    seen: set[str] = set()
    found: list[Path] = []
    for path in candidates:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            found.append(path)
    return found


def _voicevox_engine_status(url: str | None = None) -> dict[str, Any]:
    base = (url or _plugin_voicevox_url()).rstrip("/")
    engines = _voicevox_engine_candidates()
    try:
        status_code, raw = _http_request(f"{base}/version", timeout_seconds=3.0)
        version = raw.decode("utf-8", errors="replace").strip().strip('"')
        return {
            "ok": True,
            "provider": "voicevox",
            "reachable": 200 <= status_code < 300,
            "url": base,
            "version": version,
            "installed": bool(engines),
            "engine_candidates": [str(path) for path in engines],
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": "voicevox",
            "reachable": False,
            "url": base,
            "error": str(exc),
            "installed": bool(engines),
            "engine_candidates": [str(path) for path in engines],
        }


def _start_voicevox_tts(values: dict[str, Any]) -> dict[str, Any]:
    url = _plugin_voicevox_url(values.get("voicevox_url"))
    status_before = _voicevox_engine_status(url)
    if status_before.get("reachable"):
        return {"ok": True, "provider": "voicevox", "already_running": True, "status": status_before}

    engines = _voicevox_engine_candidates()
    if not engines:
        return {
            "ok": False,
            "provider": "voicevox",
            "error": "VOICEVOX engine executable was not found.",
            "expected": [
                "C:/Users/<user>/AppData/Local/Programs/VOICEVOX/vv-engine/run.exe",
                "VOICEVOX_ENGINE_EXE",
            ],
            "status": status_before,
        }

    engine = engines[0]
    host, port = _voicevox_url_parts(url)
    log_path = _log_file("voicevox-engine.log")
    cmd = [str(engine), "--host", host, "--port", str(port)]
    log_fh = open(log_path, "ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(engine.parent),
        "stdin": subprocess.DEVNULL,
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        log_fh.close()
        return {"ok": False, "provider": "voicevox", "error": str(exc), "command": cmd}
    finally:
        try:
            log_fh.close()
        except Exception:
            pass

    record = {
        "provider": "voicevox",
        "pid": proc.pid,
        "url": url,
        "engine": str(engine),
        "command": cmd,
        "started_at": time.time(),
        "log_path": str(log_path),
    }
    _write_json_file(_tts_active_file(), record)

    timeout = int(values.get("timeout_seconds") or 45)
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        current = _voicevox_engine_status(url)
        if current.get("reachable"):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    return {
        "ok": ready,
        "provider": "voicevox",
        "ready": ready,
        **record,
        "status": _voicevox_engine_status(url),
    }


def _irodori_status() -> dict[str, Any]:
    try:
        from plugins.irodori_tts import core as irodori_core
    except Exception as exc:
        return {
            "ok": False,
            "provider": "irodori",
            "available": False,
            "error": f"irodori_tts plugin is not importable: {exc}",
        }
    payload = irodori_core.status_payload()
    server = payload.get("server") if isinstance(payload, dict) else {}
    server_ok = isinstance(server, dict) and server.get("ok") is True
    return {
        **payload,
        "provider": "irodori",
        "usable": bool(payload.get("available") and server_ok),
    }


def _start_irodori_tts(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from plugins.irodori_tts import core as irodori_core
    except Exception as exc:
        return {"ok": False, "provider": "irodori", "error": str(exc)}

    before = _irodori_status()
    if before.get("usable"):
        return {"ok": True, "provider": "irodori", "already_running": True, "status": before}

    cfg = irodori_core.settings()
    ps = irodori_core.powershell_path()
    if not ps:
        return {"ok": False, "provider": "irodori", "error": "PowerShell was not found."}
    if not cfg.start_script.is_file():
        return {"ok": False, "provider": "irodori", "error": f"start script not found: {cfg.start_script}"}
    if not cfg.repo_dir.exists():
        return {"ok": False, "provider": "irodori", "error": f"repo not found: {cfg.repo_dir}"}

    timeout = int(values.get("timeout_seconds") or 120)
    result = _run_command(
        [
            ps,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(cfg.start_script),
            "-RepoDir",
            str(cfg.repo_dir),
        ],
        cwd=cfg.repo_dir,
        timeout_seconds=timeout,
    )
    return {
        "ok": result.get("ok") is True and _irodori_status().get("usable") is True,
        "provider": "irodori",
        "start": result,
        "status": _irodori_status(),
    }


def _select_tts_provider(explicit: Any = None) -> str:
    requested = _plugin_tts_provider(explicit)
    if requested != "auto":
        return requested
    irodori = _irodori_status()
    if irodori.get("available"):
        return "irodori"
    voicevox = _voicevox_engine_status()
    if voicevox.get("installed") or voicevox.get("reachable"):
        return "voicevox"
    return "none"


def tts_status() -> dict[str, Any]:
    requested = _plugin_tts_provider()
    irodori = _irodori_status()
    voicevox = _voicevox_engine_status()
    selected = _select_tts_provider(requested)
    ready = (
        (selected == "irodori" and bool(irodori.get("usable")))
        or (selected == "voicevox" and bool(voicevox.get("reachable")))
    )
    available = (
        (selected == "irodori" and bool(irodori.get("available")))
        or (selected == "voicevox" and bool(voicevox.get("installed") or voicevox.get("reachable")))
    )
    return {
        "ok": available,
        "requested_provider": requested,
        "selected_provider": selected,
        "ready": ready,
        "providers": {
            "irodori": irodori,
            "voicevox": voicevox,
        },
        "active": _read_json_file(_tts_active_file()),
    }


def start_tts(values: dict[str, Any]) -> dict[str, Any]:
    provider = _select_tts_provider(values.get("provider"))
    if provider == "irodori":
        return _start_irodori_tts(values)
    if provider == "voicevox":
        return _start_voicevox_tts(values)
    return {"ok": False, "provider": provider, "error": "No local TTS backend was found."}


def _tts_output_path(provider: str, output_path: Any = None, output_format: Any = None) -> Path:
    fmt = _coerce_tts_format(output_format)
    if provider == "voicevox":
        fmt = "wav"
    raw = _path_text(output_path)
    if raw:
        path = Path(raw).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = _audio_dir() / f"hakua-{provider}-{stamp}.{fmt}"
    if path.suffix.lower().lstrip(".") != fmt:
        path = path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _synthesize_voicevox(values: dict[str, Any]) -> dict[str, Any]:
    text = str(values.get("text") or "").strip()
    if not text:
        return {"ok": False, "provider": "voicevox", "error": "text is required."}
    url = _plugin_voicevox_url(values.get("voicevox_url"))
    if not _voicevox_engine_status(url).get("reachable"):
        started = _start_voicevox_tts(values)
        if not started.get("ok"):
            return {"ok": False, "provider": "voicevox", "start": started}

    speaker = _plugin_voicevox_speaker(values.get("voicevox_speaker"))
    output_path = _tts_output_path("voicevox", values.get("output_path"), "wav")
    query_url = (
        f"{url}/audio_query?"
        + urllib.parse.urlencode({"speaker": speaker, "text": text})
    )
    try:
        status_code, query_raw = _http_request(query_url, method="POST", timeout_seconds=15.0)
        if not 200 <= status_code < 300:
            return {"ok": False, "provider": "voicevox", "error": f"audio_query HTTP {status_code}"}
        query = json.loads(query_raw.decode("utf-8", errors="replace"))
        synthesis_url = f"{url}/synthesis?" + urllib.parse.urlencode({"speaker": speaker})
        status_code, wav_bytes = _http_request(
            synthesis_url,
            method="POST",
            payload=query,
            timeout_seconds=60.0,
        )
        if not 200 <= status_code < 300:
            return {"ok": False, "provider": "voicevox", "error": f"synthesis HTTP {status_code}"}
        output_path.write_bytes(wav_bytes)
    except Exception as exc:
        return {"ok": False, "provider": "voicevox", "error": str(exc)}

    result: dict[str, Any] = {
        "ok": True,
        "provider": "voicevox",
        "file_path": str(output_path),
        "format": "wav",
        "speaker": speaker,
        "size_bytes": output_path.stat().st_size,
        "media_tag": f"MEDIA:{output_path}",
    }
    if values.get("play"):
        result["playback"] = _play_wav_file(output_path)
    return result


def _synthesize_irodori(values: dict[str, Any]) -> dict[str, Any]:
    text = str(values.get("text") or "").strip()
    if not text:
        return {"ok": False, "provider": "irodori", "error": "text is required."}
    try:
        from plugins.irodori_tts import core as irodori_core

        fmt = _coerce_tts_format(values.get("format"))
        output_path = _tts_output_path("irodori", values.get("output_path"), fmt)
        result = irodori_core.synthesize_text(
            text=text,
            output_path=output_path,
            voice=_plugin_tts_voice(values.get("voice")) or None,
            model=_path_text(values.get("model")) or None,
            output_format=fmt,
            speed=_plugin_tts_speed(values.get("speed")),
        )
        if values.get("play"):
            result["playback"] = _play_wav_file(Path(result["file_path"]))
        return result
    except Exception as exc:
        return {"ok": False, "provider": "irodori", "error": str(exc)}


def _play_wav_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav":
        return {"ok": False, "error": "local playback currently supports wav output only."}
    try:
        if os.name == "nt":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return {"ok": True, "backend": "winsound"}
        player = shutil.which("afplay") or shutil.which("paplay") or shutil.which("aplay")
        if not player:
            return {"ok": False, "error": "No local wav player was found."}
        subprocess.run([player, str(path)], check=True, capture_output=True)
        return {"ok": True, "backend": player}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def synthesize_speech(values: dict[str, Any]) -> dict[str, Any]:
    provider = _select_tts_provider(values.get("provider"))
    if provider == "irodori":
        return _synthesize_irodori(values)
    if provider == "voicevox":
        return _synthesize_voicevox(values)
    return {"ok": False, "provider": provider, "error": "No local TTS backend was found."}


def status() -> dict[str, Any]:
    cfg = _plugin_config()
    repo = resolve_repo_root()
    fbx_app = _fbx_app_dir(repo) if repo else None
    chat_script = _codex_chat_script(repo) if repo else None
    fbx_port = _plugin_fbx_port()
    url = f"http://127.0.0.1:{fbx_port}/"
    active = _active_status()
    active["url_ready"] = _url_ready(str(active.get("url") or url)) if active.get("alive") else False
    tts = tts_status()
    readiness = {
        "repo_root": bool(repo),
        "node": bool(_node_exe()),
        "npm": bool(_npm_exe()),
        "codex_cli_auth": _codex_cli_auth_status().get("has_access_token") is True,
        "fbx_app": bool(fbx_app and (fbx_app / "package.json").is_file()),
        "codex_character_cli": bool(chat_script and chat_script.is_file()),
        "chat_dist": bool(repo and _chat_agent_dist(repo).is_file()),
        "tts_backend": bool(tts.get("ok")),
    }
    codex_sdk = _codex_sdk_installed(repo) if repo and _npm_exe() else {"installed": False}
    readiness["codex_sdk"] = bool(codex_sdk.get("installed"))
    ok = all(readiness.values())
    recommended: list[str] = []
    if not readiness["repo_root"]:
        recommended.append("hermes aituber-onair configure --repo-root <path-to-aituber-onair>")
    if not readiness["chat_dist"] or not readiness["codex_sdk"]:
        recommended.append("hermes aituber-onair prepare")
    if not readiness["codex_cli_auth"]:
        recommended.append("Authenticate Codex locally, then rerun status.")
    if not readiness["tts_backend"]:
        recommended.append("Install or configure irodoriTTS or VOICEVOX, then run hermes aituber-onair tts-status.")
    elif not tts.get("ready"):
        recommended.append("hermes aituber-onair start-tts")
    return {
        "ok": ok,
        "checked_at": _now_utc(),
        "plugin": PLUGIN_ID,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "config": {
            "character_name": cfg.get("character_name") or "はくあ",
            "model": cfg.get("model") or "Codex CLI default",
            "response_length": cfg.get("response_length") or DEFAULT_RESPONSE_LENGTH,
            "fbx_port": fbx_port,
            "url": url,
            "tts_provider": cfg.get("tts_provider") or DEFAULT_TTS_PROVIDER,
            "tts_voice": _plugin_tts_voice() or "provider default",
            "tts_speed": _plugin_tts_speed() or "provider default",
            "voicevox_url": _plugin_voicevox_url(),
            "voicevox_speaker": _plugin_voicevox_speaker(),
        },
        "paths": {
            "repo_root": str(repo) if repo else "",
            "fbx_app_dir": str(fbx_app) if fbx_app else "",
            "codex_character_cli": str(chat_script) if chat_script else "",
            "chat_agent_dist": str(_chat_agent_dist(repo)) if repo else "",
            "active_file": str(_active_file()),
        },
        "executables": {"node": _node_exe(), "npm": _npm_exe()},
        "auth": {
            "codex_cli": _codex_cli_auth_status(),
            "hermes_openai_codex": _hermes_codex_auth_status(),
        },
        "codex_sdk": codex_sdk,
        "tts": tts,
        "active": active,
        "readiness": readiness,
        "recommended_actions": recommended,
    }


def save_hakua_config(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        return {"ok": False, "error": f"Hermes config writer unavailable: {exc}"}

    repo = resolve_repo_root(values.get("repo_root"))
    if repo is None and values.get("repo_root"):
        repo = Path(_path_text(values.get("repo_root"))).expanduser()

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        cfg["plugins"] = plugins
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins["entries"] = entries
    entry = entries.setdefault(PLUGIN_ID, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[PLUGIN_ID] = entry

    if repo is not None:
        entry["repo_root"] = str(repo)
        entry["working_directory"] = str(repo)
    entry["character_name"] = "はくあ"
    entry["codex_provider"] = "codex-sdk"
    entry["codex_auth_source"] = "local-codex-cli"
    entry["response_length"] = str(values.get("response_length") or DEFAULT_RESPONSE_LENGTH)
    entry["skip_git_repo_check"] = True
    entry["fbx_app_dir"] = str(
        Path("packages") / "core" / "examples" / "react-fbx-app"
    )
    entry["fbx_port"] = _plugin_fbx_port(values.get("fbx_port"))
    entry["tts_provider"] = _plugin_tts_provider(values.get("tts_provider"))
    entry["voicevox_url"] = _plugin_voicevox_url(values.get("voicevox_url"))
    entry["voicevox_speaker"] = _plugin_voicevox_speaker(values.get("voicevox_speaker"))
    voicevox_engine = _plugin_voicevox_engine_exe(values.get("voicevox_engine_exe"))
    if voicevox_engine:
        entry["voicevox_engine_exe"] = voicevox_engine
    elif _voicevox_engine_candidates():
        entry["voicevox_engine_exe"] = str(_voicevox_engine_candidates()[0])
    tts_voice = _plugin_tts_voice(values.get("tts_voice"))
    if tts_voice:
        entry["tts_voice"] = tts_voice
    tts_speed = _plugin_tts_speed(values.get("tts_speed"))
    if tts_speed is not None:
        entry["tts_speed"] = tts_speed
    model = _path_text(values.get("model"))
    if model:
        entry["model"] = model
    prompt = str(values.get("system_prompt") or "").strip()
    entry["system_prompt"] = prompt or DEFAULT_HAKUA_SYSTEM_PROMPT

    save_config(cfg)
    return {
        "ok": True,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "repo_root": str(repo) if repo else "",
        "character_name": entry["character_name"],
        "model": entry.get("model") or "Codex CLI default",
        "fbx_url": f"http://127.0.0.1:{entry['fbx_port']}/",
        "tts_provider": entry["tts_provider"],
        "voicevox_url": entry["voicevox_url"],
        "voicevox_speaker": entry["voicevox_speaker"],
        "voicevox_engine_exe": entry.get("voicevox_engine_exe", ""),
        "tts_voice": entry.get("tts_voice", ""),
        "tts_speed": entry.get("tts_speed", ""),
    }


def handle_status(args: dict[str, Any] | None = None) -> str:
    return _json(status())


def handle_configure_hakua(args: dict[str, Any] | None = None) -> str:
    return _json(save_hakua_config(args or {}))


def prepare(values: dict[str, Any]) -> dict[str, Any]:
    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    timeout = int(values.get("timeout_seconds") or 300)
    install_codex_sdk = values.get("install_codex_sdk")
    build_chat = values.get("build_chat")
    build_fbx_app = bool(values.get("build_fbx_app"))
    if install_codex_sdk is None:
        install_codex_sdk = True
    if build_chat is None:
        build_chat = True

    steps: list[dict[str, Any]] = []
    if install_codex_sdk:
        existing = _codex_sdk_installed(repo)
        if existing.get("installed"):
            steps.append({"name": "install_codex_sdk", "ok": True, "skipped": True})
        else:
            steps.append(
                {
                    "name": "install_codex_sdk",
                    **_run_command(
                        [npm, "install", "--no-save", "--package-lock=false", "@openai/codex-sdk"],
                        cwd=repo,
                        timeout_seconds=timeout,
                    ),
                }
            )

    if build_chat:
        steps.append({"name": "build_chat", **_build_chat_for_codex(repo, npm, timeout)})

    if build_fbx_app:
        app_dir = _fbx_app_dir(repo)
        if not (app_dir / "package.json").is_file():
            steps.append(
                {
                    "name": "build_fbx_app",
                    "ok": False,
                    "error": f"FBX app package.json was not found: {app_dir}",
                }
            )
        else:
            steps.append(
                {
                    "name": "build_fbx_app",
                    **_run_command(
                        [npm, "run", "build"],
                        cwd=app_dir,
                        timeout_seconds=timeout,
                    ),
                }
            )

    return {
        "ok": all(step.get("ok") is True for step in steps),
        "repo_root": str(repo),
        "steps": steps,
        "chat_agent_dist": str(_chat_agent_dist(repo)),
    }


def _build_chat_for_codex(repo: Path, npm: str, timeout_seconds: int) -> dict[str, Any]:
    primary = _run_command(
        [npm, "-w", "@aituber-onair/chat", "run", "build"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    if primary.get("ok") is True:
        return primary

    combined = f"{primary.get('stdout') or ''}\n{primary.get('stderr') or ''}"
    windows_shell_gap = _is_windows() and (
        "'rm' is not recognized" in combined or "'mv' is not recognized" in combined
    )
    if not windows_shell_gap:
        return primary

    # The Codex character CLI only requires dist/cjs/agent.js. Upstream's full
    # build currently uses POSIX rm/mv in package scripts, so refresh the CJS
    # surface directly on Windows without editing AITuber OnAir package files.
    cjs_dir = repo / "packages" / "chat" / "dist" / "cjs"
    try:
        if cjs_dir.exists():
            shutil.rmtree(cjs_dir)
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Failed to remove stale CJS dist: {exc}",
            "primary": primary,
        }

    fallback = _run_command(
        [npm, "-w", "@aituber-onair/chat", "run", "build:cjs"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    return {
        **fallback,
        "fallback": "build:cjs",
        "primary": primary,
        "chat_agent_dist_exists": _chat_agent_dist(repo).is_file(),
        "ok": fallback.get("ok") is True and _chat_agent_dist(repo).is_file(),
    }


def handle_prepare(args: dict[str, Any] | None = None) -> str:
    return _json(prepare(args or {}))


def start_fbx_app(values: dict[str, Any]) -> dict[str, Any]:
    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    app_dir = _fbx_app_dir(repo)
    if not (app_dir / "package.json").is_file():
        return {"ok": False, "error": f"FBX app package.json was not found: {app_dir}"}
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    existing = _active_status()
    if existing.get("alive") and not values.get("force"):
        return {
            "ok": True,
            "already_running": True,
            "pid": existing.get("pid"),
            "url": existing.get("url"),
            "log_path": existing.get("log_path"),
        }
    if existing.get("alive") and values.get("force"):
        stop_fbx_app({"force": True})

    port = _plugin_fbx_port(values.get("fbx_port"))
    url = f"http://127.0.0.1:{port}/"
    log_path = _log_file("fbx-vite.log")
    cmd = [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port)]
    env = os.environ.copy()
    env["AITUBER_ONAIR_HERMES_PLUGIN"] = "1"
    env["AITUBER_ONAIR_CHARACTER_NAME"] = _plugin_character_name()
    env["AITUBER_ONAIR_CODEX_AUTH_SOURCE"] = "local-codex-cli"
    env["AITUBER_ONAIR_TTS_PROVIDER"] = _select_tts_provider()
    env["VOICEVOX_URL"] = _plugin_voicevox_url()
    env["VOICEVOX_SPEAKER"] = str(_plugin_voicevox_speaker())

    log_fh = open(log_path, "ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(app_dir),
        "stdin": subprocess.DEVNULL,
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "env": env,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        log_fh.close()
        return {"ok": False, "error": str(exc), "command": cmd, "cwd": str(app_dir)}
    finally:
        try:
            log_fh.close()
        except Exception:
            pass

    record = {
        "pid": proc.pid,
        "repo_root": str(repo),
        "app_dir": str(app_dir),
        "url": url,
        "port": port,
        "started_at": time.time(),
        "log_path": str(log_path),
        "command": cmd,
    }
    _write_json_file(_active_file(), record)
    ready = False
    for _ in range(20):
        if _url_ready(url, timeout_seconds=0.5):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    return {"ok": True, "ready": ready, **record}


def handle_start(args: dict[str, Any] | None = None) -> str:
    return _json(start_fbx_app(args or {}))


def stop_fbx_app(values: dict[str, Any]) -> dict[str, Any]:
    active = _active_status()
    if not active.get("ok"):
        return active
    pid = int(active.get("pid") or 0)
    if not active.get("alive"):
        _clear_active()
        return {"ok": True, "stopped": False, "reason": "record was stale", "pid": pid}

    try:
        from gateway.status import terminate_pid

        terminate_pid(pid, force=bool(values.get("force")))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "pid": pid}

    deadline = time.time() + 5
    while time.time() < deadline:
        if not _pid_alive(pid):
            _clear_active()
            return {"ok": True, "stopped": True, "pid": pid}
        time.sleep(0.2)

    if values.get("force"):
        return {"ok": False, "error": "process still appears alive after force stop", "pid": pid}
    return {
        "ok": False,
        "error": "process still appears alive; retry with force=true",
        "pid": pid,
    }


def handle_stop(args: dict[str, Any] | None = None) -> str:
    return _json(stop_fbx_app(args or {}))


def handle_tts_status(args: dict[str, Any] | None = None) -> str:
    return _json(tts_status())


def handle_start_tts(args: dict[str, Any] | None = None) -> str:
    return _json(start_tts(args or {}))


def handle_speak(args: dict[str, Any] | None = None) -> str:
    return _json(synthesize_speech(args or {}))


def _extract_character_reply(stdout: str, character_name: str) -> str:
    marker = f"{character_name}> "
    lines = stdout.splitlines()
    for line in reversed(lines):
        if line.startswith(marker):
            return line[len(marker) :].strip()
    useful = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith("===")
        and not line.startswith("character:")
        and not line.startswith("provider:")
        and not line.startswith("model:")
        and not line.startswith("workingDirectory:")
    ]
    return useful[-1] if useful else ""


def run_hakua_once(values: dict[str, Any]) -> dict[str, Any]:
    prompt = str(values.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required."}
    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    node = _node_exe()
    if not node:
        return {"ok": False, "error": "node was not found on PATH."}
    script = _codex_chat_script(repo)
    if not script.is_file():
        return {"ok": False, "error": f"Codex character chat script was not found: {script}"}
    if not _chat_agent_dist(repo).is_file():
        return {
            "ok": False,
            "error": "@aituber-onair/chat is not built.",
            "prepare": "hermes aituber-onair prepare",
        }
    sdk = _codex_sdk_installed(repo)
    if not sdk.get("installed"):
        return {
            "ok": False,
            "error": "@openai/codex-sdk is not installed in the local aituber-onair checkout.",
            "prepare": "hermes aituber-onair prepare",
        }
    auth = _codex_cli_auth_status()
    if not auth.get("has_access_token"):
        return {
            "ok": False,
            "error": "Codex CLI auth was not found.",
            "auth": auth,
        }

    character_name = _plugin_character_name()
    model = _plugin_model(values.get("model"))
    response_length = _plugin_response_length(values.get("response_length"))
    timeout = int(values.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    cmd = [
        node,
        str(script),
        f"--once={prompt}",
        f"--name={character_name}",
        f"--systemPrompt={_plugin_system_prompt()}",
        f"--responseLength={response_length}",
        f"--workingDirectory={_plugin_working_directory(repo)}",
        "--skipGitRepoCheck=true",
    ]
    if model:
        cmd.append(f"--model={model}")
    env = os.environ.copy()
    env["CODEX_CHARACTER_NAME"] = character_name
    env["CODEX_CHARACTER_SYSTEM_PROMPT"] = _plugin_system_prompt()
    env["CODEX_WORKING_DIRECTORY"] = _plugin_working_directory(repo)
    env["CODEX_SKIP_GIT_REPO_CHECK"] = "true"
    env["CODEX_RESPONSE_LENGTH"] = response_length
    if model:
        env["CODEX_SDK_MODEL"] = model

    result = _run_command(cmd, cwd=repo, env=env, timeout_seconds=timeout)
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    reply = _extract_character_reply(stdout, character_name)
    payload = {
        "ok": result.get("ok") is True,
        "character_name": character_name,
        "provider": "codex-sdk",
        "model": model or "Codex CLI default",
        "reply": reply,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.get("exit_code"),
        "command": result.get("command"),
        "cwd": result.get("cwd"),
    }
    if values.get("speak") and reply:
        payload["tts"] = synthesize_speech(
            {
                "text": reply,
                "provider": values.get("tts_provider"),
                "output_path": values.get("output_path"),
                "format": values.get("format"),
                "voice": values.get("tts_voice") or values.get("voice"),
                "speed": values.get("tts_speed") or values.get("speed"),
                "play": values.get("play"),
            }
        )
    return payload


def handle_say(args: dict[str, Any] | None = None) -> str:
    return _json(run_hakua_once(args or {}))


def handle_smoke(args: dict[str, Any] | None = None) -> str:
    values = dict(args or {})
    values.setdefault(
        "prompt",
        "はくあ、配信開始の短い挨拶を一文でお願いします。",
    )
    return _json(run_hakua_once(values))


HELP = """aituber commands:
  /aituber status
  /aituber configure
  /aituber prepare
  /aituber start [--force]
  /aituber stop [--force]
  /aituber tts-status
  /aituber start-tts
  /aituber speak <text>
  /aituber say <prompt>
  /aituber say --speak <prompt>
  /aituber smoke
"""


def handle_slash(raw_args: str) -> str:
    try:
        argv = shlex.split((raw_args or "").strip())
    except ValueError as exc:
        return _json({"ok": False, "error": str(exc)})
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP
    command = argv[0].lower()
    if command == "status":
        return handle_status({})
    if command in {"configure", "config", "setup"}:
        return handle_configure_hakua({})
    if command == "prepare":
        return handle_prepare({})
    if command == "start":
        return handle_start({"force": "--force" in argv})
    if command == "stop":
        return handle_stop({"force": "--force" in argv})
    if command in {"tts-status", "tts"}:
        return handle_tts_status({})
    if command == "start-tts":
        return handle_start_tts({})
    if command == "speak":
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_speak({"text": prompt, "play": "--play" in argv})
    if command == "smoke":
        return handle_smoke({})
    if command == "say":
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_say({"prompt": prompt, "speak": "--speak" in argv, "play": "--play" in argv})
    return HELP
