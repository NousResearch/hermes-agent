"""Local World Monitor Vite dev server lifecycle (npm run dev)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import api

DEFAULT_DEV_PORT = 3000
DEFAULT_REPO_URL = "https://github.com/koala73/worldmonitor.git"
ENV_REPO = "WORLDMONITOR_REPO"
ENV_DEV_PORT = "WORLDMONITOR_DEV_PORT"
ENV_DEV_HOST = "WORLDMONITOR_DEV_HOST"
ENV_DEV_BIND = "WORLDMONITOR_DEV_BIND"
STATE_FILE_NAME = "worldmonitor_dev_state.json"
TAILSCALE_SERVE_PATH = "/worldmonitor"
PROBE_PATH = "/api/news/v1/list-feed-digest?variant=full&lang=en"

DEV_STATUS_SCHEMA = {
    "name": "worldmonitor_dev_status",
    "description": "Show local World Monitor Vite dev server and repo status.",
    "parameters": {"type": "object", "properties": {}},
}

DEV_START_SCHEMA = {
    "name": "worldmonitor_dev_start",
    "description": "Start npm run dev for a local World Monitor checkout.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Checkout path override."},
            "port": {"type": "integer", "description": "Vite port (default 3000)."},
            "variant": {
                "type": "string",
                "description": "Optional variant: tech, finance, happy, commodity, energy.",
            },
            "wait_seconds": {"type": "number", "description": "Readiness wait (default 45)."},
            "bind": {
                "type": "string",
                "description": "Vite listen address (default localhost; use 0.0.0.0 for LAN/Tailscale).",
            },
            "host": {
                "type": "string",
                "description": "Client-facing URL host override (e.g. Tailscale IP or MagicDNS name).",
            },
            "tailscale": {
                "type": "boolean",
                "description": "Bind 0.0.0.0 and expose via Tailscale IP (default false).",
            },
            "tailscale_serve": {
                "type": "boolean",
                "description": "Also register tailscale serve proxy to localhost (default false).",
            },
        },
    },
}

DEV_STOP_SCHEMA = {
    "name": "worldmonitor_dev_stop",
    "description": "Stop the World Monitor dev server started by Hermes.",
    "parameters": {
        "type": "object",
        "properties": {
            "pid": {"type": "integer", "description": "Optional explicit process id."},
        },
    },
}


def _is_windows() -> bool:
    return os.name == "nt"


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _npm_exe() -> str | None:
    return os.environ.get("WORLDMONITOR_NPM") or _which("npm")


def _git_exe() -> str | None:
    return os.environ.get("WORLDMONITOR_GIT") or _which("git")


def state_file() -> Path:
    return get_hermes_home() / STATE_FILE_NAME


def _read_state() -> dict[str, Any]:
    path = state_file()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(payload: dict[str, Any]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_state() -> None:
    try:
        state_file().unlink()
    except FileNotFoundError:
        pass


def _dev_port() -> int:
    for candidate in (
        os.environ.get(ENV_DEV_PORT, "").strip(),
        str(_read_state().get("port") or ""),
        str(DEFAULT_DEV_PORT),
    ):
        if not candidate:
            continue
        try:
            return int(candidate)
        except ValueError:
            continue
    return DEFAULT_DEV_PORT


def dev_base_url(port: int | None = None, *, host: str = "localhost") -> str:
    return f"http://{host}:{port or _dev_port()}".rstrip("/")


def _tailscale_exe() -> str | None:
    return os.environ.get("WORLDMONITOR_TAILSCALE") or _which("tailscale")


def _run_tailscale(args: list[str], *, timeout: float = 15.0) -> dict[str, Any]:
    exe = _tailscale_exe()
    if not exe:
        return {"ok": False, "error": "tailscale was not found on PATH."}
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def tailscale_ipv4() -> str | None:
    result = _run_tailscale(["ip", "-4"])
    if not result.get("ok"):
        return None
    ip = (result.get("stdout") or "").splitlines()[0].strip()
    return ip or None


def tailscale_status() -> dict[str, Any]:
    ip = tailscale_ipv4()
    hostname = ""
    dns_name = ""
    result = _run_tailscale(["status", "--json"], timeout=20.0)
    if result.get("ok") and result.get("stdout"):
        try:
            payload = json.loads(result["stdout"])
            self = payload.get("Self") or {}
            hostname = str(self.get("HostName") or self.get("DNSName") or "").split(".")[0]
            dns_name = str(self.get("DNSName") or "").rstrip(".")
        except json.JSONDecodeError:
            pass
    return {
        "available": bool(ip),
        "ipv4": ip,
        "hostname": hostname or None,
        "dns_name": dns_name or None,
        "magic_dns_url": f"http://{dns_name}" if dns_name else None,
    }


def _resolve_bind_host(
    *,
    bind: str = "",
    host: str = "",
    tailscale: bool = False,
) -> tuple[str, str | None]:
    """Return (vite --host value, optional client-facing host for URLs)."""
    bind = (bind or os.environ.get(ENV_DEV_BIND, "")).strip()
    host = (host or os.environ.get(ENV_DEV_HOST, "")).strip()
    ts_ip = tailscale_ipv4() if tailscale else None

    if tailscale:
        bind = bind or "0.0.0.0"
        host = host or ts_ip or "0.0.0.0"
    elif bind in {"0.0.0.0", "::", "all"}:
        bind = "0.0.0.0"
        host = host or "127.0.0.1"
    else:
        bind = bind or "127.0.0.1"
        host = host or "localhost"

    return bind, ts_ip if tailscale else None


def _vite_npm_command(npm: str, script: str, *, port: int, bind: str) -> list[str]:
    command = [npm, "run", script, "--", "--host", bind, "--port", str(port)]
    return command


def _probe_hosts(port: int | None = None) -> tuple[str, ...]:
    hosts: list[str] = ["localhost", "127.0.0.1"]
    state = _read_state()
    for candidate in (
        str(state.get("access_host") or ""),
        str(state.get("tailscale_ip") or ""),
        tailscale_ipv4() or "",
    ):
        if candidate and candidate not in hosts:
            hosts.append(candidate)
    if port:
        for host in list(hosts):
            if host not in {"localhost", "127.0.0.1"} and not _socket_open(host, port, 1.5):
                hosts.remove(host)
    return tuple(hosts)


def _socket_open(host: str, port: int, timeout: float) -> bool:
    try:
        import socket

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(url: str, *, timeout: float, accept: str = "*/*") -> tuple[bool, int | None]:
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": accept, "User-Agent": "hermes-worldmonitor-osint/0.1"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500, resp.status
    except urllib.error.HTTPError as exc:
        # Vite + sebuf may return 401/403/404 while still serving the dev stack.
        return exc.code in {200, 401, 403, 404}, exc.code
    except Exception:
        return False, None


def resolve_repo_path() -> Path:
    """Return configured or discovered World Monitor checkout."""
    for candidate in (
        os.environ.get(ENV_REPO, "").strip(),
        str(_read_state().get("repo") or ""),
    ):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path

    sibling = Path(__file__).resolve().parents[3] / "worldmonitor"
    if sibling.is_dir() and (sibling / "package.json").is_file():
        return sibling

    return get_hermes_home() / "worldmonitor"


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if _is_windows():
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok - POSIX-only fallback when psutil is unavailable.
        return True
    except OSError:
        return False


def probe_dev_server(port: int | None = None, *, timeout: float = 20.0) -> dict[str, Any]:
    """Check whether the Vite dev server responds (localhost and optional Tailscale bind)."""
    port = port or _dev_port()
    probe_hosts = _probe_hosts(port)
    sock_ok = any(_socket_open(host, port, min(timeout, 5.0)) for host in probe_hosts)

    http_ok = False
    http_status: int | None = None
    active_host = "localhost"
    probe_paths = ("/@vite/client", "/", PROBE_PATH)
    for host in probe_hosts:
        base = dev_base_url(port, host=host)
        for path in probe_paths:
            ok, status = _http_probe(f"{base}{path}", timeout=min(timeout, 8.0))
            if ok and status is not None and status != 404:
                http_ok = True
                http_status = status
                active_host = host
                break
            if ok and path == "/@vite/client":
                http_ok = True
                http_status = status
                active_host = host
                break
        if http_ok:
            break

    base = dev_base_url(port, host=active_host)
    return {
        "port": port,
        "host": active_host,
        "base_url": base,
        "socket_open": sock_ok,
        "http_reachable": http_ok,
        "http_status": http_status,
        "running": sock_ok and http_ok,
        "mode": "vite_dev",
        "tailscale": tailscale_status(),
    }


def _configure_tailscale_serve(port: int, *, dry_run: bool = False) -> dict[str, Any]:
    """Register `tailscale serve` path → localhost dev port (tailnet-only HTTPS)."""
    target = f"http://127.0.0.1:{port}"
    command = ["serve", "--bg", "--set-path", TAILSCALE_SERVE_PATH, target]
    if dry_run:
        return {"ok": True, "dry_run": True, "command": command, "target": target}
    result = _run_tailscale(command)
    status = _run_tailscale(["serve", "status"])
    return {
        "ok": bool(result.get("ok")),
        "command": command,
        "target": target,
        "serve_path": TAILSCALE_SERVE_PATH,
        "tailscale_serve": result,
        "serve_status": status.get("stdout"),
    }


def _popen_kwargs(cwd: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _url_ready(url: str, wait_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() <= deadline:
        if probe_dev_server().get("running"):
            return True
        time.sleep(0.5)
    return False


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {timeout}s",
            "command": command,
            "stdout": (exc.stdout or "")[-2000:] if exc.stdout else "",
            "stderr": (exc.stderr or "")[-2000:] if exc.stderr else "",
        }
    except OSError as exc:
        return {"ok": False, "error": str(exc), "command": command}


def clone_repo(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    target: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    git = _git_exe()
    if not git:
        return {"success": False, "error": "git was not found on PATH."}

    dest = (target or resolve_repo_path()).expanduser()
    if dest.is_dir() and any(dest.iterdir()):
        return {
            "success": True,
            "already_exists": True,
            "repo": str(dest),
            "note": "Directory already present; skipping clone.",
        }
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "would_clone": repo_url,
            "repo": str(dest),
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command([git, "clone", repo_url, str(dest)], cwd=dest.parent, timeout=600)
    return {
        "success": bool(result.get("ok")),
        "repo": str(dest),
        "clone": result,
    }


def install_deps(
    *,
    repo: Path | None = None,
    dry_run: bool = False,
    timeout: int = 900,
) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"success": False, "error": "npm was not found on PATH."}

    root = (repo or resolve_repo_path()).expanduser()
    if not (root / "package.json").is_file():
        return {"success": False, "error": f"package.json not found under {root}"}
    if dry_run:
        return {"success": True, "dry_run": True, "repo": str(root), "command": [npm, "install"]}

    result = _run_command([npm, "install"], cwd=root, timeout=timeout)
    return {
        "success": bool(result.get("ok")),
        "repo": str(root),
        "install": result,
    }


def start_dev(
    *,
    repo: Path | None = None,
    port: int | None = None,
    variant: str = "",
    wait_seconds: float = 45.0,
    configure_api: bool = True,
    dry_run: bool = False,
    bind: str = "",
    host: str = "",
    tailscale: bool = False,
    tailscale_serve: bool = False,
) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"success": False, "error": "npm was not found on PATH."}

    root = (repo or resolve_repo_path()).expanduser()
    if not (root / "package.json").is_file():
        return {"success": False, "error": f"package.json not found under {root}"}

    port = port or _dev_port()
    bind_host, ts_ip = _resolve_bind_host(bind=bind, host=host, tailscale=tailscale)
    access_host = host.strip() or (ts_ip if tailscale and ts_ip else "localhost")
    access_url = dev_base_url(port, host=access_host)
    ts_meta = tailscale_status() if tailscale or tailscale_serve else {}

    existing = _read_state()
    existing_pid = int(existing.get("pid") or 0)
    if _pid_alive(existing_pid):
        probe = probe_dev_server(port)
        return {
            "success": True,
            "already_running": True,
            "pid": existing_pid,
            "url": existing.get("url") or access_url,
            "tailscale": ts_meta or None,
            "probe": probe,
        }

    live_probe = probe_dev_server(port)
    if live_probe.get("running"):
        if configure_api and not dry_run:
            _save_dev_base(port)
        return {
            "success": True,
            "already_running": True,
            "external_process": True,
            "url": existing.get("url") or access_url,
            "tailscale": ts_meta or None,
            "probe": live_probe,
        }

    if tailscale and not ts_ip:
        return {
            "success": False,
            "error": "Tailscale is not running or has no IPv4 address.",
            "tailscale": ts_meta,
        }

    script = "dev"
    if variant:
        script = f"dev:{variant}"
    command = _vite_npm_command(npm, script, port=port, bind=bind_host)
    if dry_run:
        serve_preview = (
            _configure_tailscale_serve(port, dry_run=True) if tailscale_serve else None
        )
        return {
            "success": True,
            "dry_run": True,
            "repo": str(root),
            "command": command,
            "port": port,
            "bind": bind_host,
            "url": access_url,
            "tailscale": ts_meta or None,
            "tailscale_serve": serve_preview,
        }

    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, **_popen_kwargs(root))
    except OSError as exc:
        return {"success": False, "error": str(exc), "command": command}

    serve_result: dict[str, Any] | None = None
    if tailscale_serve:
        serve_result = _configure_tailscale_serve(port)

    payload = {
        "pid": proc.pid,
        "port": port,
        "bind": bind_host,
        "access_host": access_host,
        "tailscale_ip": ts_ip,
        "tailscale": bool(tailscale),
        "tailscale_serve": bool(tailscale_serve),
        "url": access_url,
        "repo": str(root),
        "command": command,
        "started_at": time.time(),
        "variant": variant or "full",
    }
    if serve_result:
        payload["tailscale_serve_result"] = serve_result
    _write_state(payload)
    ready = _url_ready(access_url, wait_seconds)
    if configure_api and ready:
        _save_dev_base(port)

    return {
        "success": True,
        "pid": proc.pid,
        "url": access_url,
        "bind": bind_host,
        "ready": ready,
        "command": command,
        "repo": str(root),
        "tailscale": ts_meta or None,
        "tailscale_serve": serve_result,
        "probe": probe_dev_server(port),
        "state_file": str(state_file()),
    }


def _save_dev_base(port: int) -> None:
    from hermes_cli.config import save_env_value

    probe = probe_dev_server(port)
    base = str(probe.get("base_url") or dev_base_url(port))
    save_env_value(api.ENV_API_BASE, base)
    save_env_value(api.ENV_LOCAL_PORT, str(port))


def _terminate_pid(pid: int) -> dict[str, Any]:
    if not _pid_alive(pid):
        return {"ok": True, "already_stopped": True, "pid": pid}
    if _is_windows():
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return {
            "ok": result.returncode == 0,
            "pid": pid,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    os.kill(pid, signal.SIGTERM)
    return {"ok": True, "pid": pid, "signal": "SIGTERM"}


def stop_dev(*, pid: int | None = None) -> dict[str, Any]:
    state = _read_state()
    target = int(pid or state.get("pid") or 0)
    if not target:
        return {"success": False, "error": "No dev server pid recorded."}
    result = _terminate_pid(target)
    if result.get("ok") and int(state.get("pid") or 0) == target:
        _clear_state()
    return {"success": bool(result.get("ok")), **result}


def dev_status(*, probe: bool = True) -> dict[str, Any]:
    repo = resolve_repo_path()
    state = _read_state()
    pid = int(state.get("pid") or 0)
    port = int(state.get("port") or _dev_port())
    npm = _npm_exe()
    ts_meta = tailscale_status()
    access_host = str(state.get("access_host") or state.get("tailscale_ip") or "localhost")
    payload: dict[str, Any] = {
        "success": True,
        "repo": str(repo),
        "repo_exists": repo.is_dir() and (repo / "package.json").is_file(),
        "node_modules": (repo / "node_modules").is_dir() if repo.is_dir() else False,
        "npm": npm,
        "git": _git_exe(),
        "state_file": str(state_file()),
        "tailscale": ts_meta,
        "dev_server": {
            "pid": pid or None,
            "pid_alive": _pid_alive(pid),
            "port": port,
            "bind": state.get("bind"),
            "url": state.get("url") or dev_base_url(port, host=access_host),
            "localhost_url": dev_base_url(port, host="localhost"),
            "tailscale_url": (
                dev_base_url(port, host=ts_meta["ipv4"]) if ts_meta.get("ipv4") else None
            ),
            "command": state.get("command"),
            "variant": state.get("variant"),
            "tailscale_serve_path": TAILSCALE_SERVE_PATH if state.get("tailscale_serve") else None,
        },
    }
    if probe:
        payload["dev_server"]["probe"] = probe_dev_server(port)
        payload["dev_server"]["running"] = bool(payload["dev_server"]["probe"].get("running"))
    return payload


def setup_dev_stack(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    repo: Path | None = None,
    clone: bool = True,
    install: bool = True,
    start: bool = True,
    port: int | None = None,
    variant: str = "",
    dry_run: bool = False,
    bind: str = "",
    host: str = "",
    tailscale: bool = False,
    tailscale_serve: bool = False,
) -> dict[str, Any]:
    """Clone → npm install → npm run dev → point Hermes API at localhost."""
    target = (repo or resolve_repo_path()).expanduser()
    result: dict[str, Any] = {
        "success": False,
        "dry_run": dry_run,
        "repo": str(target),
        "steps": [],
    }

    if clone:
        step = clone_repo(repo_url=repo_url, target=target, dry_run=dry_run)
        result["steps"].append({"clone": step})
        if not step.get("success"):
            result["error"] = step.get("error") or "clone failed"
            return result

    if install:
        step = install_deps(repo=target, dry_run=dry_run)
        result["steps"].append({"install": step})
        if not step.get("success"):
            result["error"] = step.get("error") or step.get("install", {}).get("stderr") or "install failed"
            return result

    if start:
        step = start_dev(
            repo=target,
            port=port,
            variant=variant,
            configure_api=not dry_run,
            dry_run=dry_run,
            bind=bind,
            host=host,
            tailscale=tailscale,
            tailscale_serve=tailscale_serve,
        )
        result["steps"].append({"start": step})
        if not step.get("success"):
            result["error"] = step.get("error") or "start failed"
            return result

    ts = tailscale_status()
    next_urls = [f"Open dashboard: {dev_base_url(port or _dev_port())}"]
    if ts.get("ipv4"):
        next_urls.append(f"Tailscale: http://{ts['ipv4']}:{port or _dev_port()}")
    if ts.get("magic_dns_url") and tailscale_serve:
        next_urls.append(f"Tailscale Serve: {ts['magic_dns_url']}{TAILSCALE_SERVE_PATH}")
    result["success"] = True
    result["next_steps"] = next_urls + [
        "Verify API: `hermes worldmonitor-osint status`",
        "Snapshot: `hermes worldmonitor-osint snapshot --tier auto`",
        "Stop dev server: `hermes worldmonitor-osint dev stop`",
    ]
    return result
