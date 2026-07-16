from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


PLUGIN_ID = "sillytavern"
TOOLSET = "sillytavern"
REPO_URL = "https://github.com/SillyTavern/SillyTavern.git"
DEFAULT_PORT = 8000
MIN_PORT = 1024
MAX_PORT = 65535
DEFAULT_START_TIMEOUT_SECONDS = 90
DEFAULT_CHAT_TIMEOUT_SECONDS = 600
MAX_CHAT_TIMEOUT_SECONDS = 3600
MAX_MESSAGES = 64
MAX_CONTENT_CHARS = 100_000

CAPABILITIES_SCHEMA = {
    "description": (
        "Report the pinned SillyTavern revision, local bridge capabilities, "
        "and the explicit safety gates required for process control and generation."
    ),
    "type": "object",
    "properties": {},
}

STATUS_SCHEMA = {
    "description": "Report SillyTavern submodule, Node.js, process, and local endpoint readiness.",
    "type": "object",
    "properties": {},
}

START_SCHEMA = {
    "description": (
        "Start the pinned SillyTavern server in an isolated Hermes data directory. "
        "Requires plugin process control to be enabled and explicit acknowledgement."
    ),
    "type": "object",
    "properties": {
        "acknowledge_side_effects": {
            "type": "boolean",
            "description": "Must be true to start the local SillyTavern process.",
        },
    },
}

STOP_SCHEMA = {
    "description": (
        "Stop only a SillyTavern process previously recorded by this bridge. "
        "Unmanaged processes are never terminated."
    ),
    "type": "object",
    "properties": {
        "acknowledge_side_effects": {
            "type": "boolean",
            "description": "Must be true to stop the local SillyTavern process.",
        },
    },
}

GENERATE_SCHEMA = {
    "description": (
        "Send one non-streaming chat-completions request through the local "
        "SillyTavern server. Provider credentials remain in SillyTavern data."
    ),
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Convenience form for one user message.",
        },
        "messages": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_MESSAGES,
            "description": "Chat messages with role and content fields.",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant"],
                    },
                    "content": {"type": "string"},
                },
                "required": ["role", "content"],
            },
        },
        "model": {
            "type": "string",
            "description": "Provider model name. Falls back to plugins.entries.sillytavern.model.",
        },
        "chat_completion_source": {
            "type": "string",
            "description": "SillyTavern chat completion source, such as openai or openrouter.",
        },
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "max_tokens": {"type": "integer", "minimum": 1, "maximum": 32768},
        "top_p": {"type": "number", "minimum": 0, "maximum": 1},
        "timeout_seconds": {
            "type": "integer",
            "minimum": 10,
            "maximum": MAX_CHAT_TIMEOUT_SECONDS,
        },
        "acknowledge_side_effects": {
            "type": "boolean",
            "description": "Must be true because generation can call a paid or remote provider.",
        },
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_root() -> Path:
    return repo_root() / "vendor" / "SillyTavern"


def state_root() -> Path:
    return get_hermes_home() / PLUGIN_ID


def state_file() -> Path:
    return state_root() / "server.json"


def log_file() -> Path:
    return state_root() / "server.log"


def data_root() -> Path:
    return state_root() / "data"


def config_file() -> Path:
    return state_root() / "config.yaml"


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _load_entry() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        return {}

    plugins = config.get("plugins") if isinstance(config, dict) else {}
    entries = plugins.get("entries") if isinstance(plugins, dict) else {}
    entry = entries.get(PLUGIN_ID) if isinstance(entries, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _port(entry: dict[str, Any]) -> int:
    return _bounded_int(
        entry.get("port"),
        default=DEFAULT_PORT,
        minimum=MIN_PORT,
        maximum=MAX_PORT,
    )


def _base_url(entry: dict[str, Any] | None = None) -> str:
    return f"http://127.0.0.1:{_port(entry or _load_entry())}"


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dependency_status() -> dict[str, Any]:
    node = shutil.which("node")
    npm = shutil.which("npm")
    package_file = source_root() / "package.json"
    node_modules = source_root() / "node_modules"
    return {
        "node": node or "",
        "npm": npm or "",
        "node_version": _command_version(node, ["--version"]) if node else "",
        "package_file": str(package_file),
        "package_present": package_file.is_file(),
        "node_modules_present": node_modules.is_dir(),
        "runtime_ready": bool(node and package_file.is_file() and node_modules.is_dir()),
    }


def _command_version(command: str, args: list[str]) -> str:
    try:
        result = subprocess.run(
            [command, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def check_available() -> bool:
    dependencies = _dependency_status()
    return (
        dependencies["node"]
        and dependencies["package_present"]
        and (source_root() / "server.js").is_file()
    )


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(state_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(payload: dict[str, Any]) -> None:
    state_root().mkdir(parents=True, exist_ok=True)
    temporary = state_file().with_suffix(".tmp")
    temporary.write_text(_json(payload) + "\n", encoding="utf-8")
    temporary.replace(state_file())


def _clear_state() -> None:
    try:
        state_file().unlink()
    except FileNotFoundError:
        pass


def _pid_exists(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        return False


def _process_matches(pid: int | None, expected_data_root: Path) -> bool:
    if not _pid_exists(pid):
        return False
    try:
        import psutil

        command_line = " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return False

    def normalized(value: Path | str) -> str:
        return str(value).replace("\\", "/").rstrip("/").lower()

    command = command_line.replace("\\", "/").lower()
    return (
        normalized(source_root()) in command
        and "/server.js" in command
        and normalized(expected_data_root) in command
    )


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_000).decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {"raw": raw[-20_000:]}
            return {"ok": 200 <= response.status < 300, "status_code": response.status, "data": data}
    except urllib.error.HTTPError as error:
        raw = error.read(2_000_000).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {"raw": raw[-20_000:]}
        return {"ok": False, "status_code": error.code, "data": data}
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"ok": False, "status_code": None, "error": str(error)}


def _probe(base_url: str, timeout: int = 5) -> dict[str, Any]:
    result = _http_json("GET", f"{base_url}/csrf-token", timeout=timeout)
    return {
        "healthy": bool(result.get("ok")),
        "status_code": result.get("status_code"),
        "error": result.get("error", ""),
    }


def status_payload(_values: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = _load_entry()
    state = _read_state()
    dependencies = _dependency_status()
    root = source_root()
    base_url = _base_url(entry)
    pid = state.get("pid")
    pid = pid if isinstance(pid, int) else None
    expected_data = Path(state.get("data_root") or data_root())
    probe = _probe(base_url) if dependencies["runtime_ready"] else {
        "healthy": False,
        "status_code": None,
        "error": "Node dependencies are not installed.",
    }
    managed = _process_matches(pid, expected_data) if pid else False
    payload = {
        "success": True,
        "available": bool(check_available()),
        "repo_url": REPO_URL,
        "source_root": str(root),
        "source_present": root.is_dir(),
        "git_head": _git_head() if root.is_dir() else "unknown",
        "base_url": base_url,
        "port": _port(entry),
        "pid": pid,
        "pid_alive": _pid_exists(pid),
        "managed_process": managed,
        "healthy": probe["healthy"],
        "health_status_code": probe["status_code"],
        "health_error": probe["error"],
        "running": probe["healthy"],
        "runtime": dependencies,
        "data_root": str(data_root()),
        "config_file": str(config_file()),
        "log_file": str(log_file()),
        "process_control_enabled": bool(entry.get("allow_process_control", False)),
        "network_enabled": bool(entry.get("allow_network", False)),
        "configured_model": str(entry.get("model") or ""),
        "configured_source": str(entry.get("chat_completion_source") or "openai"),
    }
    if state:
        payload["state"] = {
            "started_at": state.get("started_at", ""),
            "command": state.get("command", []),
        }
    return payload


def capabilities() -> dict[str, Any]:
    return {
        "success": True,
        "plugin": PLUGIN_ID,
        "repo_url": REPO_URL,
        "toolset": TOOLSET,
        "features": [
            "pinned submodule status",
            "isolated local server lifecycle",
            "CSRF-protected non-streaming chat completion requests",
        ],
        "configuration": {
            "enable": "hermes plugins enable sillytavern",
            "process_control": "plugins.entries.sillytavern.allow_process_control: true",
            "network": "plugins.entries.sillytavern.allow_network: true",
            "credentials": "Configure provider secrets in the SillyTavern UI data directory; do not put them in Hermes config.yaml.",
        },
        "safety": [
            "The bridge binds to 127.0.0.1 and keeps SillyTavern data under the active Hermes home.",
            "Start, stop, and generate require explicit acknowledge_side_effects=true.",
            "Stop refuses processes that do not match the pinned source and bridge data root.",
        ],
    }


def _acknowledged(values: dict[str, Any]) -> bool:
    return bool(values.get("acknowledge_side_effects"))


def start_server(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    entry = _load_entry()
    current = status_payload({})
    if current["healthy"]:
        current["started"] = False
        current["message"] = "SillyTavern is already healthy at the configured local endpoint."
        return current
    if not bool(entry.get("allow_process_control", False)):
        return {
            "success": False,
            "status": "blocked",
            "error": "Enable plugins.entries.sillytavern.allow_process_control in config.yaml.",
        }
    if not _acknowledged(values):
        return {
            "success": False,
            "status": "blocked",
            "error": "Starting SillyTavern requires acknowledge_side_effects=true.",
        }
    if not check_available():
        return {
            "success": False,
            "status": "blocked",
            "error": "SillyTavern source or Node.js is unavailable.",
            "status_payload": current,
        }
    if not current["runtime"]["node_modules_present"]:
        return {
            "success": False,
            "status": "blocked",
            "error": "SillyTavern dependencies are not installed.",
            "install_hint": f"Run npm ci --omit=dev in {source_root()}",
        }

    state_root().mkdir(parents=True, exist_ok=True)
    data_root().mkdir(parents=True, exist_ok=True)
    log_handle = log_file().open("a", encoding="utf-8")
    command = [
        str(shutil.which("node")),
        str(source_root() / "server.js"),
        "--configPath",
        str(config_file()),
        "--dataRoot",
        str(data_root()),
        "--port",
        str(_port(entry)),
        "--listen=false",
        "--enableIPv4=true",
        "--enableIPv6=false",
        "--browserLaunchEnabled=false",
        "--heartbeatInterval",
        "1",
    ]
    environment = os.environ.copy()
    environment["NODE_ENV"] = "production"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if os.name == "nt":
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        process = subprocess.Popen(
            command,
            cwd=str(source_root()),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as error:
        log_handle.close()
        return {"success": False, "status": "failed", "error": str(error)}
    finally:
        if "process" not in locals():
            log_handle.close()

    state = {
        "pid": process.pid,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source_root()),
        "data_root": str(data_root()),
        "base_url": _base_url(entry),
        "git_head": _git_head(),
        "command": command,
    }
    _write_state(state)

    timeout = _bounded_int(
        entry.get("startup_timeout_seconds"),
        default=DEFAULT_START_TIMEOUT_SECONDS,
        minimum=5,
        maximum=300,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(process.pid):
            return {
                "success": False,
                "status": "failed",
                "error": "SillyTavern exited during startup.",
                "log_file": str(log_file()),
                "log_tail": _log_tail(),
            }
        probe = _probe(_base_url(entry), timeout=2)
        if probe["healthy"]:
            return {
                "success": True,
                "status": "running",
                "started": True,
                "pid": process.pid,
                "base_url": _base_url(entry),
                "git_head": state["git_head"],
                "log_file": str(log_file()),
            }
        time.sleep(0.25)

    return {
        "success": False,
        "status": "timed_out",
        "error": f"SillyTavern did not become healthy within {timeout} seconds.",
        "pid": process.pid,
        "base_url": _base_url(entry),
        "log_file": str(log_file()),
        "log_tail": _log_tail(),
    }


def _log_tail(limit: int = 12_000) -> str:
    try:
        return log_file().read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def stop_server(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    if not _acknowledged(values):
        return {
            "success": False,
            "status": "blocked",
            "error": "Stopping SillyTavern requires acknowledge_side_effects=true.",
        }
    state = _read_state()
    pid = state.get("pid")
    pid = pid if isinstance(pid, int) else None
    expected_data = Path(state.get("data_root") or data_root())
    if not pid or not _pid_exists(pid):
        _clear_state()
        return {"success": True, "status": "stopped", "stopped": False, "message": "No managed process was running."}
    if not _process_matches(pid, expected_data):
        return {
            "success": False,
            "status": "blocked",
            "error": "Refusing to stop an unmanaged process.",
            "pid": pid,
        }

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as error:
        return {"success": False, "status": "failed", "error": str(error), "pid": pid}

    deadline = time.monotonic() + 15
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if _pid_exists(pid):
        return {
            "success": False,
            "status": "timed_out",
            "error": "SillyTavern did not exit after SIGTERM.",
            "pid": pid,
        }
    _clear_state()
    return {"success": True, "status": "stopped", "stopped": True, "pid": pid}


def _messages(values: dict[str, Any]) -> list[dict[str, str]]:
    raw_messages = values.get("messages")
    if raw_messages is None:
        prompt = str(values.get("prompt") or "").strip()
        raw_messages = [{"role": "user", "content": prompt}] if prompt else []
    if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must contain 1-{MAX_MESSAGES} items")

    normalized: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("message role must be system, user, or assistant")
        if not content.strip():
            raise ValueError("message content must not be empty")
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"message content exceeds {MAX_CONTENT_CHARS} characters")
        normalized.append({"role": role, "content": content})
    return normalized


def _content_from_choice(choice: dict[str, Any]) -> str:
    raw_message = choice.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    content = message.get("content") or choice.get("text") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def generate(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    try:
        messages = _messages(values)
    except ValueError as error:
        return {"success": False, "status": "invalid_request", "error": str(error)}

    entry = _load_entry()
    if not bool(entry.get("allow_network", False)):
        return {
            "success": False,
            "status": "blocked",
            "error": "Enable plugins.entries.sillytavern.allow_network in config.yaml before generation.",
        }
    if not _acknowledged(values):
        return {
            "success": False,
            "status": "blocked",
            "error": "Generation requires acknowledge_side_effects=true.",
        }

    current = status_payload({})
    if not current["healthy"]:
        return {
            "success": False,
            "status": "blocked",
            "error": "SillyTavern is not healthy. Start it first or configure an existing local instance.",
            "status_payload": current,
        }

    model = str(values.get("model") or entry.get("model") or "").strip()
    if not model:
        return {
            "success": False,
            "status": "invalid_request",
            "error": "model is required or must be configured at plugins.entries.sillytavern.model.",
        }
    source = str(
        values.get("chat_completion_source")
        or entry.get("chat_completion_source")
        or "openai"
    ).strip()
    if not source or len(source) > 64 or any(char.isspace() for char in source):
        return {"success": False, "status": "invalid_request", "error": "chat_completion_source is invalid"}

    payload: dict[str, Any] = {
        "chat_completion_source": source,
        "model": model,
        "messages": messages,
        "stream": False,
    }
    for key in ("temperature", "max_tokens", "top_p"):
        if key in values and values[key] is not None:
            payload[key] = values[key]
    timeout = _bounded_int(
        values.get("timeout_seconds"),
        default=_bounded_int(
            entry.get("chat_timeout_seconds"),
            default=DEFAULT_CHAT_TIMEOUT_SECONDS,
            minimum=10,
            maximum=MAX_CHAT_TIMEOUT_SECONDS,
        ),
        minimum=10,
        maximum=MAX_CHAT_TIMEOUT_SECONDS,
    )

    token_result = _http_json("GET", f"{current['base_url']}/csrf-token", timeout=10)
    raw_token_data = token_result.get("data")
    token_data: dict[str, Any] = raw_token_data if isinstance(raw_token_data, dict) else {}
    token = str(token_data.get("token") or "")
    if not token_result.get("ok") or not token:
        return {
            "success": False,
            "status": "failed",
            "error": "SillyTavern did not return a CSRF token.",
            "http_status": token_result.get("status_code"),
        }

    result = _http_json(
        "POST",
        f"{current['base_url']}/api/backends/chat-completions/generate",
        payload=payload,
        headers={"X-CSRF-Token": token},
        timeout=timeout,
    )
    raw_data = result.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    if not result.get("ok"):
        return {
            "success": False,
            "status": "provider_error",
            "error": data.get("error") or data.get("message") or result.get("error") or "SillyTavern request failed",
            "http_status": result.get("status_code"),
            "provider_response": data,
        }

    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    reply = _content_from_choice(choice)
    return {
        "success": bool(reply),
        "status": "completed" if reply else "empty",
        "reply": reply,
        "model": data.get("model") or model,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage") if isinstance(data.get("usage"), dict) else {},
        "request_source": source,
    }


def handle_capabilities(values: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(capabilities())


def handle_status(values: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(status_payload(values))


def handle_start(values: dict[str, Any] | None = None, **kwargs: Any) -> str:
    merged = dict(values) if isinstance(values, dict) else {}
    merged.update(kwargs)
    return _json(start_server(merged))


def handle_stop(values: dict[str, Any] | None = None, **kwargs: Any) -> str:
    merged = dict(values) if isinstance(values, dict) else {}
    merged.update(kwargs)
    return _json(stop_server(merged))


def handle_generate(values: dict[str, Any] | None = None, **kwargs: Any) -> str:
    merged = dict(values) if isinstance(values, dict) else {}
    merged.update(kwargs)
    return _json(generate(merged))


def handle_slash(raw_args: str) -> str:
    parts = (raw_args or "").split(maxsplit=1)
    command = parts[0].lower() if parts else "status"
    if command == "status":
        return _json(status_payload({}))
    if command == "capabilities":
        return _json(capabilities())
    return _json(
        {
            "success": False,
            "status": "invalid_request",
            "error": "Use /sillytavern status or the hermes sillytavern CLI for side-effecting operations.",
        }
    )
