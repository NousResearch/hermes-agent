"""Core implementation for the AITuber OnAir Hermes plugin."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
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


def _log_file(name: str) -> Path:
    path = _workspace_root() / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
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


def status() -> dict[str, Any]:
    cfg = _plugin_config()
    repo = resolve_repo_root()
    fbx_app = _fbx_app_dir(repo) if repo else None
    chat_script = _codex_chat_script(repo) if repo else None
    fbx_port = _plugin_fbx_port()
    url = f"http://127.0.0.1:{fbx_port}/"
    active = _active_status()
    active["url_ready"] = _url_ready(str(active.get("url") or url)) if active.get("alive") else False
    readiness = {
        "repo_root": bool(repo),
        "node": bool(_node_exe()),
        "npm": bool(_npm_exe()),
        "codex_cli_auth": _codex_cli_auth_status().get("has_access_token") is True,
        "fbx_app": bool(fbx_app and (fbx_app / "package.json").is_file()),
        "codex_character_cli": bool(chat_script and chat_script.is_file()),
        "chat_dist": bool(repo and _chat_agent_dist(repo).is_file()),
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
    return {
        "ok": result.get("ok") is True,
        "character_name": character_name,
        "provider": "codex-sdk",
        "model": model or "Codex CLI default",
        "reply": _extract_character_reply(stdout, character_name),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.get("exit_code"),
        "command": result.get("command"),
        "cwd": result.get("cwd"),
    }


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
  /aituber say <prompt>
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
    if command == "smoke":
        return handle_smoke({})
    if command == "say":
        prompt = " ".join(arg for arg in argv[1:] if arg != "--")
        return handle_say({"prompt": prompt})
    return HELP
