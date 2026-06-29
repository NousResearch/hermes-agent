"""Core Hermes bridge for https://github.com/tegnike/aituber-kit."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hermes_constants import display_hermes_home, get_hermes_home

from . import dev_server

PLUGIN_ID = "aituber-kit"
CONFIG_ALIASES = (PLUGIN_ID, "aituber_kit", "aituberkkit")
TOOLSET = "aituber_kit"
DEFAULT_BRIDGE_PORT = 8000
BRIDGE_STATE_NAME = "aituber_kit_bridge_state.json"

DEFAULT_SYSTEM_PROMPT = (
    "あなたはAITuberKit上のAIキャラクターです。"
    "視聴者に向けて自然で短い日本語で話してください。"
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


def _plugin_port(value: Any = None) -> int:
    if value is not None:
        try:
            port = int(value)
            if 1024 <= port <= 65535:
                return port
        except (TypeError, ValueError):
            pass
    cfg = _plugin_config()
    try:
        port = int(cfg.get("port") or 0)
        if 1024 <= port <= 65535:
            return port
    except (TypeError, ValueError):
        pass
    return dev_server.DEFAULT_DEV_PORT


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


def _plugin_client_id(value: Any = None) -> str:
    text = _path_text(value or _plugin_config().get("client_id"))
    if text:
        return text
    return f"hermes-{uuid.uuid4().hex[:12]}"


def _plugin_base_url(value: Any = None) -> str:
    text = _path_text(value or _plugin_config().get("base_url"))
    if text:
        return text.rstrip("/")
    port = _plugin_port()
    return dev_server.dev_base_url(port)


def _plugin_api_key_env() -> str:
    return _path_text(_plugin_config().get("api_key_env")) or "AITUBERKIT_API_KEY"


def _read_secret(name: str) -> str:
    return _path_text(os.environ.get(name))


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


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if os.name == "nt":
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok - POSIX-only fallback when psutil is unavailable.
        return True
    except OSError:
        return False


def resolve_repo_root(value: Any = None) -> Path | None:
    text = _path_text(value)
    if text:
        path = Path(text).expanduser()
        if dev_server.is_aituber_kit_repo(path):
            return path
        if path.is_dir():
            return path
    configured = _path_text(_plugin_config().get("repo_root"))
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            return path
    default = dev_server.resolve_repo_path()
    return default if default.is_dir() else None


def check_available() -> bool:
    return True


STATUS_SCHEMA = {
    "name": "aituber_kit_status",
    "description": "Show AITuberKit checkout, dev server, and API bridge readiness.",
    "parameters": {"type": "object", "properties": {}},
}

CONFIGURE_SCHEMA = {
    "name": "aituber_kit_configure",
    "description": "Save non-secret AITuberKit settings to Hermes config.yaml.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string", "description": "Path to aituber-kit checkout."},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "client_id": {"type": "string", "description": "AITuberKit client id for /api/v1."},
            "base_url": {"type": "string", "description": "Override app base URL."},
            "bridge_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "external_linkage_url": {"type": "string"},
            "message_receiver_enabled": {"type": "boolean"},
            "external_linkage_enabled": {"type": "boolean"},
            "system_prompt": {"type": "string"},
            "api_key_env": {
                "type": "string",
                "description": "Env var name holding AITUBERKIT_API_KEY (secret stays in ~/.hermes/.env).",
            },
        },
    },
}

INSTALL_SCHEMA = {
    "name": "aituber_kit_install",
    "description": "Clone tegnike/aituber-kit and run npm install.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "ref": {"type": "string", "description": "Git branch (default main)."},
            "force": {"type": "boolean"},
            "skip_npm_install": {"type": "boolean"},
        },
    },
}

PREPARE_SCHEMA = {
    "name": "aituber_kit_prepare",
    "description": "Create .env.local, enable message receiver / external linkage flags.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "client_id": {"type": "string"},
            "message_receiver_enabled": {"type": "boolean"},
            "external_linkage_enabled": {"type": "boolean"},
            "external_linkage_url": {"type": "string"},
        },
    },
}

START_SCHEMA = {
    "name": "aituber_kit_start",
    "description": "Start the local AITuberKit Next.js dev server.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "host": {"type": "string", "description": "Client-facing URL host (Tailscale IP or MagicDNS)."},
            "bind": {"type": "string", "description": "Next.js listen address (default 127.0.0.1; use 0.0.0.0 for LAN/Tailscale)."},
            "tailscale": {
                "type": "boolean",
                "description": "Bind 0.0.0.0 and expose via Tailscale IPv4 (default false).",
            },
            "tailscale_serve": {
                "type": "boolean",
                "description": "Register tailscale serve path /aituber-kit → localhost (HTTPS on tailnet).",
            },
            "wait_seconds": {"type": "number"},
        },
    },
}

STOP_SCHEMA = {
    "name": "aituber_kit_stop",
    "description": "Stop the Hermes-managed AITuberKit dev server.",
    "parameters": {
        "type": "object",
        "properties": {
            "force": {"type": "boolean"},
            "pid": {"type": "integer"},
        },
    },
}

SPEAK_SCHEMA = {
    "name": "aituber_kit_speak",
    "description": "Make the AITuberKit character speak text via POST /api/v1/speak.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "client_id": {"type": "string"},
            "emotion": {"type": "string"},
            "priority": {"type": "string", "enum": ["normal", "high"]},
            "interrupt": {"type": "boolean"},
        },
        "required": ["text"],
    },
}

CHAT_SCHEMA = {
    "name": "aituber_kit_chat",
    "description": "Send chat input to AITuberKit via POST /api/v1/chat.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "client_id": {"type": "string"},
            "mode": {"type": "string", "enum": ["user_input", "ai_generate"]},
            "interrupt": {"type": "boolean"},
        },
        "required": ["text"],
    },
}

STOP_PLAYBACK_SCHEMA = {
    "name": "aituber_kit_stop_playback",
    "description": "Stop AITuberKit speech/queue via POST /api/v1/stop.",
    "parameters": {
        "type": "object",
        "properties": {
            "client_id": {"type": "string"},
            "mode": {"type": "string", "enum": ["speech", "queue", "all"]},
        },
    },
}

BRIDGE_START_SCHEMA = {
    "name": "aituber_kit_bridge_start",
    "description": "Start Hermes External Linkage WebSocket bridge for AITuberKit v2.",
    "parameters": {
        "type": "object",
        "properties": {
            "host": {"type": "string"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "tailscale": {
                "type": "boolean",
                "description": "Bind bridge on 0.0.0.0 for tailnet External Linkage clients.",
            },
            "confirm_public_host": {
                "type": "boolean",
                "description": "Required when host/tailscale binds the bridge outside loopback.",
            },
            "system_prompt": {"type": "string"},
        },
    },
}

BRIDGE_STOP_SCHEMA = {
    "name": "aituber_kit_bridge_stop",
    "description": "Stop the Hermes External Linkage WebSocket bridge.",
    "parameters": {
        "type": "object",
        "properties": {"force": {"type": "boolean"}},
    },
}


def status() -> dict[str, Any]:
    repo = resolve_repo_root()
    port = _plugin_port()
    bridge_port = _plugin_bridge_port()
    api_key_env = _plugin_api_key_env()
    api_key_present = bool(_read_secret(api_key_env))
    bridge_state = _read_bridge_state()
    bridge_pid = int(bridge_state.get("pid") or 0)
    dev = dev_server.dev_status(repo=repo, port=port)
    recommended: list[str] = []
    if not repo or not dev_server.is_aituber_kit_repo(repo):
        recommended.append(f"hermes aituber-kit install --repo-root {display_hermes_home()}/workspace/aituber-kit")
    elif not dev.get("repo", {}).get("has_node_modules"):
        recommended.append("hermes aituber-kit install")
    elif not dev.get("repo", {}).get("has_env_local"):
        recommended.append("hermes aituber-kit prepare")
    dev_probe = (dev.get("dev_server") or {}).get("probe") or {}
    if not dev_probe.get("running"):
        recommended.append("hermes aituber-kit start")
    if not api_key_present:
        recommended.append(f"Set {api_key_env} in {display_hermes_home()}/.env and restart AITuberKit")
    return {
        "ok": True,
        "plugin": PLUGIN_ID,
        "upstream": dev_server.UPSTREAM_REPO,
        "repo_root": str(repo) if repo else "",
        "base_url": _plugin_base_url(),
        "client_id": _plugin_client_id(),
        "api_key_env": api_key_env,
        "api_key_present": api_key_present,
        "bridge_port": bridge_port,
        "bridge_url": f"ws://127.0.0.1:{bridge_port}/ws",
        "bridge_running": _pid_alive(bridge_pid),
        "bridge_state": bridge_state,
        "dev": dev,
        "docs": {
            "message_receiver": "https://docs.aituberkit.com/en/guide/other/message-receiver",
            "external_linkage": "https://docs.aituberkit.com/en/guide/ai/external-linkage",
        },
        "recommended": recommended,
    }


def save_config(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config, save_config as persist_config
    except Exception as exc:
        return {"ok": False, "error": f"Hermes config writer unavailable: {exc}"}

    repo = resolve_repo_root(values.get("repo_root"))
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
    if values.get("port") is not None:
        entry["port"] = _plugin_port(values.get("port"))
    if values.get("bridge_port") is not None:
        entry["bridge_port"] = _plugin_bridge_port(values.get("bridge_port"))
    if values.get("client_id") is not None:
        entry["client_id"] = _plugin_client_id(values.get("client_id"))
    if values.get("base_url") is not None:
        entry["base_url"] = _path_text(values.get("base_url"))
    if values.get("external_linkage_url") is not None:
        entry["external_linkage_url"] = _path_text(values.get("external_linkage_url"))
    if values.get("message_receiver_enabled") is not None:
        entry["message_receiver_enabled"] = bool(values.get("message_receiver_enabled"))
    if values.get("external_linkage_enabled") is not None:
        entry["external_linkage_enabled"] = bool(values.get("external_linkage_enabled"))
    if values.get("system_prompt") is not None:
        entry["system_prompt"] = _path_text(values.get("system_prompt")) or DEFAULT_SYSTEM_PROMPT
    if values.get("api_key_env") is not None:
        entry["api_key_env"] = _path_text(values.get("api_key_env")) or "AITUBERKIT_API_KEY"

    persist_config(cfg)
    return {
        "ok": True,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "repo_root": entry.get("repo_root", ""),
        "port": _plugin_port(entry.get("port")),
        "bridge_port": _plugin_bridge_port(entry.get("bridge_port")),
        "client_id": _plugin_client_id(entry.get("client_id")),
        "base_url": _plugin_base_url(entry.get("base_url")),
    }


def install(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    target = resolve_repo_root(values.get("repo_root")) or dev_server.default_repo_path()
    clone = dev_server.clone_repo(
        target=target,
        ref=str(values.get("ref") or "main"),
        force=bool(values.get("force")),
    )
    if not clone.get("ok"):
        return {"ok": False, "clone": clone}
    if values.get("skip_npm_install"):
        return {"ok": True, "clone": clone, "npm_install": {"skipped": True}}
    npm = dev_server.install_deps(repo=target)
    return {"ok": bool(npm.get("ok")), "clone": clone, "npm_install": npm}


def _upsert_env_line(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    out = [line for line in lines if not line.startswith(prefix)]
    out.append(f'{prefix}"{value}"')
    return out


def prepare(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    repo = resolve_repo_root(values.get("repo_root"))
    if repo is None or not dev_server.is_aituber_kit_repo(repo):
        return {
            "ok": False,
            "error": "aituber-kit checkout not found.",
            "install": f"hermes aituber-kit install --repo-root {display_hermes_home()}/workspace/aituber-kit",
        }

    env_example = repo / ".env.example"
    env_local = repo / ".env.local"
    if not env_local.is_file() and env_example.is_file():
        shutil.copyfile(env_example, env_local)

    lines: list[str] = []
    if env_local.is_file():
        lines = env_local.read_text(encoding="utf-8", errors="replace").splitlines()

    client_id = _plugin_client_id(values.get("client_id"))
    message_enabled = values.get("message_receiver_enabled")
    if message_enabled is None:
        message_enabled = _plugin_config().get("message_receiver_enabled", True)
    external_enabled = values.get("external_linkage_enabled")
    if external_enabled is None:
        external_enabled = _plugin_config().get("external_linkage_enabled", False)
    bridge_port = _plugin_bridge_port()
    bridge_url = _path_text(values.get("external_linkage_url"))
    if not bridge_url and values.get("use_tailscale"):
        ts_ip = dev_server.tailscale_ipv4()
        if ts_ip:
            bridge_url = f"ws://{ts_ip}:{bridge_port}/ws"
    if not bridge_url:
        bridge_url = f"ws://127.0.0.1:{bridge_port}/ws"

    lines = _upsert_env_line(lines, "NEXT_PUBLIC_MESSAGE_RECEIVER_ENABLED", "true" if message_enabled else "false")
    lines = _upsert_env_line(lines, "NEXT_PUBLIC_CLIENT_ID", client_id)
    lines = _upsert_env_line(lines, "NEXT_PUBLIC_EXTERNAL_LINKAGE_MODE", "true" if external_enabled else "false")
    lines = _upsert_env_line(lines, "NEXT_PUBLIC_EXTERNAL_LINKAGE_URL", bridge_url)
    env_local.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    save_config({
        "repo_root": str(repo),
        "client_id": client_id,
        "message_receiver_enabled": bool(message_enabled),
        "external_linkage_enabled": bool(external_enabled),
        "external_linkage_url": bridge_url,
    })

    api_key_env = _plugin_api_key_env()
    suggested_key = secrets.token_urlsafe(24)
    return {
        "ok": True,
        "repo_root": str(repo),
        "env_local": str(env_local),
        "client_id": client_id,
        "message_receiver_enabled": bool(message_enabled),
        "external_linkage_enabled": bool(external_enabled),
        "external_linkage_url": bridge_url,
        "api_key_env": api_key_env,
        "api_key_present": bool(_read_secret(api_key_env)),
        "suggested_api_key": suggested_key,
        "next_steps": [
            f"Add {api_key_env}={suggested_key} to {display_hermes_home()}/.env",
            "Restart AITuberKit after changing .env.local",
            "Enable Message Receiver / External Linkage in the AITuberKit settings UI if needed",
            "hermes aituber-kit start",
        ],
    }


def _api_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    client_id: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    api_key_env = _plugin_api_key_env()
    api_key = _read_secret(api_key_env)
    if not api_key:
        return {
            "ok": False,
            "error": f"{api_key_env} is not set in the environment.",
            "hint": f"Add it to {display_hermes_home()}/.env",
        }

    cid = _plugin_client_id(client_id)
    base = (base_url or _plugin_base_url()).rstrip("/")
    query = urlencode({"clientId": cid})
    url = f"{base}{path}?{query}"
    data = json.dumps(body or {}).encode("utf-8")
    req = Request(
        url,
        data=data if method.upper() != "GET" else None,
        method=method.upper(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-aituber-kit/0.1",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "payload": payload, "url": url}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return {"ok": False, "status": exc.code, "payload": payload, "url": url, "error": str(exc)}
    except URLError as exc:
        return {"ok": False, "error": str(exc), "url": url}


def speak(values: dict[str, Any]) -> dict[str, Any]:
    text = _path_text(values.get("text"))
    if not text:
        return {"ok": False, "error": "text is required"}
    body: dict[str, Any] = {"text": text}
    if values.get("emotion"):
        body["emotion"] = _path_text(values.get("emotion"))
    if values.get("priority"):
        body["priority"] = _path_text(values.get("priority"))
    if values.get("interrupt") is not None:
        body["interrupt"] = bool(values.get("interrupt"))
    return _api_request("POST", "/api/v1/speak", body=body, client_id=values.get("client_id"))


def chat(values: dict[str, Any]) -> dict[str, Any]:
    text = _path_text(values.get("text"))
    if not text:
        return {"ok": False, "error": "text is required"}
    body: dict[str, Any] = {
        "text": text,
        "mode": _path_text(values.get("mode")) or "user_input",
    }
    if values.get("interrupt") is not None:
        body["interrupt"] = bool(values.get("interrupt"))
    return _api_request("POST", "/api/v1/chat", body=body, client_id=values.get("client_id"))


def stop_playback(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    body: dict[str, Any] = {"mode": _path_text(values.get("mode")) or "all"}
    return _api_request("POST", "/api/v1/stop", body=body, client_id=values.get("client_id"))


def api_status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    cid = _plugin_client_id(values.get("client_id"))
    query = urlencode({"clientId": cid})
    base = _plugin_base_url().rstrip("/")
    api_key_env = _plugin_api_key_env()
    api_key = _read_secret(api_key_env)
    if not api_key:
        return {"ok": False, "error": f"{api_key_env} is not set"}
    req = Request(
        f"{base}/api/v1/status/?{query}",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw else {}
            return {"ok": True, "status": resp.status, "payload": payload}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def start_bridge(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    tailscale = bool(values.get("tailscale"))
    ts = dev_server.tailscale_status()
    if tailscale and not ts.get("ipv4"):
        return {"ok": False, "error": "Tailscale is not running or has no IPv4 address.", "tailscale": ts}

    host = _path_text(values.get("host"))
    if tailscale and host in {"", "127.0.0.1", "localhost"}:
        host = "0.0.0.0"
    elif not host:
        host = "127.0.0.1"
    if not _is_loopback_host(host) and not values.get("confirm_public_host"):
        return {
            "ok": False,
            "confirmation_required": True,
            "reason": "Binding the AITuberKit bridge outside loopback exposes a noauth WebSocket that can invoke Hermes oneshot.",
            "host": host,
            "tailscale": ts,
        }
    port = _plugin_bridge_port(values.get("port"))
    system_prompt = _path_text(values.get("system_prompt")) or _plugin_config().get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    state = _read_bridge_state()
    pid = int(state.get("pid") or 0)
    if _pid_alive(pid):
        return {"ok": True, "already_running": True, "pid": pid, "url": state.get("url"), "tailscale": ts}

    worker = Path(__file__).resolve().parent / "bridge_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        "--host",
        host,
        "--port",
        str(port),
        "--system-prompt",
        system_prompt,
    ]
    log_path = get_hermes_home() / "workspace" / "aituber-kit" / "logs" / "bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
        )
    except OSError as exc:
        log_handle.close()
        return {"ok": False, "error": str(exc), "command": cmd}

    url = f"ws://{host}:{port}/ws"
    if tailscale and ts.get("ipv4"):
        url = f"ws://{ts['ipv4']}:{port}/ws"
    payload = {"pid": proc.pid, "host": host, "port": port, "url": url, "log": str(log_path), "tailscale": ts}
    _write_bridge_state(payload)
    return {"ok": True, "pid": proc.pid, "url": url, "log": str(log_path), "command": cmd, "tailscale": ts}


def stop_bridge(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    state = _read_bridge_state()
    pid = int(state.get("pid") or 0)
    if not pid:
        _clear_bridge_state()
        return {"ok": True, "stopped": False}
    if not _pid_alive(pid):
        _clear_bridge_state()
        return {"ok": True, "stopped": False, "note": "Process already exited."}
    force = bool(values.get("force"))
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdin=subprocess.DEVNULL,
            )
        else:
            import signal

            sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            os.kill(pid, sigkill if force else signal.SIGTERM)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "pid": pid}
    _clear_bridge_state()
    return {"ok": True, "stopped": True, "pid": pid}


def handle_status(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(status())


def handle_configure(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(save_config(args or {}))


def handle_install(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(install(args or {}))


def handle_prepare(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(prepare(args or {}))


def handle_start(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    values = args or {}
    repo = resolve_repo_root(values.get("repo_root"))
    result = dev_server.start_dev(
        repo=repo,
        port=_plugin_port(values.get("port")),
        host=_path_text(values.get("host")),
        bind=_path_text(values.get("bind")),
        tailscale=bool(values.get("tailscale")),
        tailscale_serve=bool(values.get("tailscale_serve")),
        wait_seconds=float(values.get("wait_seconds") or 60.0),
    )
    return _json(result)


def handle_stop(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    values = args or {}
    return _json(dev_server.stop_dev(pid=values.get("pid"), force=bool(values.get("force"))))


def handle_speak(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(speak(args or {}))


def handle_chat(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(chat(args or {}))


def handle_stop_playback(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(stop_playback(args or {}))


def handle_bridge_start(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(start_bridge(args or {}))


def handle_bridge_stop(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(stop_bridge(args or {}))


HELP = """aituber-kit commands:
  /aituber-kit status
  /aituber-kit configure
  /aituber-kit install
  /aituber-kit prepare
  /aituber-kit start
  /aituber-kit stop
  /aituber-kit speak <text>
  /aituber-kit chat <text>
  /aituber-kit bridge-start
  /aituber-kit bridge-stop
"""


def handle_slash(args: str) -> str:
    text = (args or "").strip()
    if not text or text in {"help", "-h", "--help"}:
        return HELP
    parts = text.split()
    command = parts[0].lower()
    rest = " ".join(parts[1:]).strip()

    if command in {"status", "st"}:
        return handle_status({})
    if command in {"configure", "config", "setup"}:
        return handle_configure({})
    if command == "install":
        return handle_install({})
    if command == "prepare":
        return handle_prepare({})
    if command == "start":
        return handle_start({})
    if command == "stop":
        return handle_stop({})
    if command == "speak" and rest:
        return handle_speak({"text": rest})
    if command == "chat" and rest:
        return handle_chat({"text": rest})
    if command in {"bridge-start", "bridge"}:
        return handle_bridge_start({})
    if command == "bridge-stop":
        return handle_bridge_stop({})
    return HELP
