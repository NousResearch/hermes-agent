"""Local AITuberKit Next.js dev server lifecycle."""

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

from hermes_constants import display_hermes_home, get_hermes_home

UPSTREAM_REPO = "https://github.com/tegnike/aituber-kit.git"
DEFAULT_DEV_PORT = 3000
STATE_FILE_NAME = "aituber_kit_dev_state.json"
TAILSCALE_SERVE_PATH = "/aituber-kit"
ENV_DEV_BIND = "AITUBER_KIT_DEV_BIND"
ENV_DEV_HOST = "AITUBER_KIT_DEV_HOST"


def _is_windows() -> bool:
    return os.name == "nt"


def _which(name: str) -> str | None:
    return shutil.which(name)


def _npm_exe() -> str | None:
    return os.environ.get("AITUBER_KIT_NPM") or _which("npm")


def _git_exe() -> str | None:
    return os.environ.get("AITUBER_KIT_GIT") or _which("git")


def _node_exe() -> str | None:
    return os.environ.get("AITUBER_KIT_NODE") or _which("node")


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


def default_repo_path() -> Path:
    return get_hermes_home() / "workspace" / "aituber-kit"


def resolve_repo_path(repo_root: str | Path | None = None) -> Path:
    for candidate in (
        str(repo_root or "").strip(),
        str(_read_state().get("repo") or "").strip(),
        os.environ.get("AITUBER_KIT_REPO", "").strip(),
    ):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path
    sibling = Path(__file__).resolve().parents[2] / "vendor" / "aituber-kit"
    if sibling.is_dir() and (sibling / "package.json").is_file():
        return sibling
    return default_repo_path()


def is_aituber_kit_repo(path: Path) -> bool:
    package_json = path / "package.json"
    if not package_json.is_file():
        return False
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("name") == "aituber-kit"


def dev_port(configured: int | None = None) -> int:
    if configured and configured > 0:
        return int(configured)
    for candidate in (
        str(_read_state().get("port") or "").strip(),
        os.environ.get("AITUBER_KIT_PORT", "").strip(),
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
    return f"http://{host}:{port or dev_port()}".rstrip("/")


def _tailscale_exe() -> str | None:
    return os.environ.get("AITUBER_KIT_TAILSCALE") or _which("tailscale")


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
        "app_url": f"http://{ip}:{dev_port()}" if ip else None,
    }


def _resolve_bind_host(
    *,
    bind: str = "",
    host: str = "",
    tailscale: bool = False,
) -> tuple[str, str | None]:
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


def _configure_tailscale_serve(port: int, *, dry_run: bool = False) -> dict[str, Any]:
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


def _socket_open(host: str, port: int, timeout: float) -> bool:
    try:
        import socket

        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_probe(url: str, *, timeout: float) -> tuple[bool, int | None]:
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "text/html,*/*", "User-Agent": "hermes-aituber-kit/0.1"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500, resp.status
    except urllib.error.HTTPError as exc:
        return exc.code in {200, 401, 403, 404, 500}, exc.code
    except Exception:
        return False, None


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


def probe_dev_server(port: int | None = None, *, timeout: float = 15.0) -> dict[str, Any]:
    port = port or dev_port()
    hosts = _probe_hosts(port)
    sock_ok = any(_socket_open(host, port, min(timeout, 5.0)) for host in hosts)
    http_ok = False
    http_status: int | None = None
    active_host = "localhost"
    for host in hosts:
        base = dev_base_url(port, host=host)
        for path in ("/", "/api/health"):
            ok, status = _http_probe(f"{base}{path}", timeout=min(timeout, 8.0))
            if ok:
                http_ok = True
                http_status = status
                active_host = host
                break
        if http_ok:
            break
    return {
        "port": port,
        "host": active_host,
        "base_url": dev_base_url(port, host=active_host),
        "socket_open": sock_ok,
        "http_reachable": http_ok,
        "http_status": http_status,
        "running": sock_ok and http_ok,
        "mode": "next_dev",
        "tailscale": tailscale_status(),
    }


def _popen_kwargs(cwd: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if _is_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 900,
    env: dict[str, str] | None = None,
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
            env=env,
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
    repo_url: str = UPSTREAM_REPO,
    target: Path | None = None,
    ref: str = "main",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    git = _git_exe()
    if not git:
        return {"ok": False, "error": "git was not found on PATH."}

    dest = (target or resolve_repo_path()).expanduser()
    if dest.is_dir() and any(dest.iterdir()):
        if force:
            shutil.rmtree(dest)
        elif is_aituber_kit_repo(dest):
            return {
                "ok": True,
                "skipped": True,
                "action": "already_cloned",
                "repo": str(dest),
            }
        else:
            return {
                "ok": False,
                "error": f"Target exists but is not an aituber-kit checkout: {dest}",
            }
    if dry_run:
        return {"ok": True, "dry_run": True, "would_clone": repo_url, "repo": str(dest)}

    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = (ref or "main").strip() or "main"
    attempts = [
        [git, "clone", "--depth", "1", "--branch", branch, repo_url, str(dest)],
        [git, "clone", "--depth", "1", repo_url, str(dest)],
    ]
    last: dict[str, Any] | None = None
    for cmd in attempts:
        last = _run_command(cmd, cwd=dest.parent, timeout=900)
        if last.get("ok") and is_aituber_kit_repo(dest):
            return {"ok": True, "action": "cloned", "repo": str(dest), "ref": branch, "clone": last}
    return {
        "ok": False,
        "action": "clone_failed",
        "repo": str(dest),
        "ref": branch,
        "clone": last,
    }


def install_deps(
    *,
    repo: Path | None = None,
    dry_run: bool = False,
    timeout: int = 1800,
) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    root = (repo or resolve_repo_path()).expanduser()
    if not is_aituber_kit_repo(root):
        return {"ok": False, "error": f"aituber-kit checkout not found at {root}"}
    if dry_run:
        return {"ok": True, "dry_run": True, "repo": str(root), "command": [npm, "install"]}

    result = _run_command([npm, "install"], cwd=root, timeout=timeout)
    return {"ok": bool(result.get("ok")), "repo": str(root), "install": result}


def _next_dev_command(npm: str, *, port: int, host: str) -> list[str]:
    return [npm, "run", "dev", "--", "-p", str(port), "-H", host]


def start_dev(
    *,
    repo: Path | None = None,
    port: int | None = None,
    host: str = "",
    bind: str = "",
    tailscale: bool = False,
    tailscale_serve: bool = False,
    wait_seconds: float = 60.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    root = (repo or resolve_repo_path()).expanduser()
    if not is_aituber_kit_repo(root):
        return {"ok": False, "error": f"aituber-kit checkout not found at {root}"}

    port = port or dev_port()
    bind_host, ts_ip = _resolve_bind_host(bind=bind, host=host, tailscale=tailscale)
    access_host = host.strip() or (ts_ip if tailscale and ts_ip else "localhost")
    access_url = dev_base_url(port, host=access_host)
    ts_meta = tailscale_status() if tailscale or tailscale_serve else {}

    existing = _read_state()
    existing_pid = int(existing.get("pid") or 0)
    if _pid_alive(existing_pid):
        probe = probe_dev_server(port)
        return {
            "ok": True,
            "already_running": True,
            "pid": existing_pid,
            "url": existing.get("url") or access_url,
            "tailscale": ts_meta or None,
            "probe": probe,
        }

    live_probe = probe_dev_server(port)
    if live_probe.get("running"):
        return {
            "ok": True,
            "already_running": True,
            "external_process": True,
            "url": existing.get("url") or access_url,
            "tailscale": ts_meta or None,
            "probe": live_probe,
        }

    if tailscale and not ts_ip:
        return {
            "ok": False,
            "error": "Tailscale is not running or has no IPv4 address.",
            "tailscale": ts_meta,
        }

    command = _next_dev_command(npm, port=port, host=bind_host)
    if dry_run:
        serve_preview = (
            _configure_tailscale_serve(port, dry_run=True) if tailscale_serve else None
        )
        return {
            "ok": True,
            "dry_run": True,
            "repo": str(root),
            "command": command,
            "port": port,
            "bind": bind_host,
            "url": access_url,
            "tailscale": ts_meta or None,
            "tailscale_serve": serve_preview,
        }

    env = os.environ.copy()
    env["PORT"] = str(port)
    try:
        proc = subprocess.Popen(command, env=env, **_popen_kwargs(root))
    except OSError as exc:
        return {"ok": False, "error": str(exc), "command": command}

    serve_result: dict[str, Any] | None = None
    if tailscale_serve:
        serve_result = _configure_tailscale_serve(port)

    payload = {
        "pid": proc.pid,
        "port": port,
        "bind": bind_host,
        "host": bind_host,
        "access_host": access_host,
        "tailscale_ip": ts_ip,
        "tailscale": bool(tailscale),
        "tailscale_serve": bool(tailscale_serve),
        "url": access_url,
        "repo": str(root),
        "command": command,
        "started_at": time.time(),
    }
    if serve_result:
        payload["tailscale_serve_result"] = serve_result
    _write_state(payload)

    deadline = time.monotonic() + max(0.0, wait_seconds)
    ready = False
    while time.monotonic() <= deadline:
        if probe_dev_server(port).get("running"):
            ready = True
            break
        if not _pid_alive(proc.pid):
            break
        time.sleep(0.75)

    probe = probe_dev_server(port)
    ts = tailscale_status()
    return {
        "ok": ready or probe.get("running", False),
        "pid": proc.pid,
        "url": access_url,
        "repo": str(root),
        "ready": ready,
        "bind": bind_host,
        "tailscale": ts,
        "tailscale_url": f"http://{ts['ipv4']}:{port}" if ts.get("ipv4") else None,
        "tailscale_serve_path": TAILSCALE_SERVE_PATH if tailscale_serve else None,
        "tailscale_serve": serve_result,
        "probe": probe,
        "display_repo": str(root),
        "display_url": access_url,
    }


def stop_dev(*, pid: int | None = None, force: bool = False) -> dict[str, Any]:
    state = _read_state()
    target_pid = int(pid or state.get("pid") or 0)
    if not target_pid:
        _clear_state()
        return {"ok": True, "stopped": False, "note": "No managed dev server pid recorded."}

    if not _pid_alive(target_pid):
        _clear_state()
        return {"ok": True, "stopped": False, "note": f"Pid {target_pid} is not running."}

    try:
        if _is_windows():
            kill_cmd = ["taskkill", "/PID", str(target_pid), "/T"]
            if force:
                kill_cmd.append("/F")
            subprocess.run(kill_cmd, capture_output=True, text=True, check=False)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(target_pid, sig)
    except OSError as exc:
        return {"ok": False, "error": str(exc), "pid": target_pid}

    _clear_state()
    return {"ok": True, "stopped": True, "pid": target_pid, "force": force}


def dev_status(*, repo: Path | None = None, port: int | None = None) -> dict[str, Any]:
    root = repo or resolve_repo_path()
    port = port or dev_port()
    state = _read_state()
    probe = probe_dev_server(port)
    node = _node_exe()
    npm = _npm_exe()
    git = _git_exe()
    package_json = root / "package.json"
    node_modules = root / "node_modules"
    env_local = root / ".env.local"
    return {
        "ok": True,
        "upstream_repo": UPSTREAM_REPO,
        "repo": {
            "path": str(root),
            "present": root.is_dir(),
            "valid": is_aituber_kit_repo(root),
            "has_node_modules": node_modules.is_dir(),
            "has_env_local": env_local.is_file(),
            "has_env_example": (root / ".env.example").is_file(),
        },
        "toolchain": {
            "node": node,
            "npm": npm,
            "git": git,
        },
        "dev_server": {
            "port": port,
            "url": dev_base_url(port),
            "state": state,
            "probe": probe,
            "tailscale_url": (
                f"http://{state.get('tailscale_ip')}:{port}"
                if state.get("tailscale_ip")
                else (tailscale_status().get("app_url") if tailscale_status().get("ipv4") else None)
            ),
            "tailscale_serve_path": TAILSCALE_SERVE_PATH if state.get("tailscale_serve") else None,
        },
        "tailscale": tailscale_status(),
        "workspace_default": str(default_repo_path()),
        "display_workspace": f"{display_hermes_home()}/workspace/aituber-kit",
    }
