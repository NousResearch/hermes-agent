"""Managed Cloudflare Quick Tunnel helpers for the Photon plugin."""
from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import signal
import ssl
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_WEBHOOK_PORT = 8788
DEFAULT_WEBHOOK_PATH = "/photon/webhook"
DEFAULT_START_TIMEOUT_SECONDS = 30.0
CLOUDFLARED_RELEASE_API = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
_TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


@dataclass
class TunnelStartResult:
    success: bool
    public_url: str = ""
    webhook_url: str = ""
    reused: bool = False
    pid: Optional[int] = None
    error: str = ""
    log_path: Optional[Path] = None
    command: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CloudflaredAsset:
    name: str
    download_url: str
    sha256: str
    size: int
    version: str = ""


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return Path(get_hermes_home())
    except Exception:
        return Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser()


def state_dir() -> Path:
    return hermes_home() / "photon"


def state_path() -> Path:
    return state_dir() / "tunnel.json"


def log_path() -> Path:
    return state_dir() / "cloudflared.log"


def managed_bin_dir() -> Path:
    return hermes_home() / "bin"


def managed_cloudflared_path() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return managed_bin_dir() / f"cloudflared{suffix}"


def managed_cloudflared_manifest_path() -> Path:
    path = managed_cloudflared_path()
    return path.with_name(f"{path.name}.manifest.json")


def _get_env_value(key: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value  # type: ignore
        return get_env_value(key)
    except Exception:
        return os.getenv(key)


def webhook_port() -> int:
    raw = _get_env_value("PHOTON_WEBHOOK_PORT")
    if not raw:
        return DEFAULT_WEBHOOK_PORT
    try:
        port = int(str(raw).strip())
    except ValueError:
        return DEFAULT_WEBHOOK_PORT
    if 1 <= port <= 65535:
        return port
    return DEFAULT_WEBHOOK_PORT


def webhook_path() -> str:
    raw = (_get_env_value("PHOTON_WEBHOOK_PATH") or DEFAULT_WEBHOOK_PATH).strip()
    if not raw:
        return DEFAULT_WEBHOOK_PATH
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw


def local_url() -> str:
    return f"http://127.0.0.1:{webhook_port()}"


def webhook_url_for_base(public_url: str) -> str:
    return public_url.rstrip("/") + webhook_path()


def health_url_for_webhook_url(webhook_url: str) -> str:
    parsed = urlparse(webhook_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/healthz"


def check_public_health(webhook_url: str, timeout_seconds: float = 5.0) -> tuple[bool, str]:
    health_url = health_url_for_webhook_url(webhook_url)
    if not health_url:
        return False, "missing public webhook URL"
    try:
        with urllib.request.urlopen(  # noqa: S310
            health_url,
            timeout=timeout_seconds,
            context=_ssl_context(),
        ) as response:
            body = response.read(64).decode("utf-8", errors="replace").strip()
            if 200 <= int(getattr(response, "status", 0) or 0) < 300 and body == "ok":
                return True, health_url
            return False, f"{health_url} returned {getattr(response, 'status', '?')}"
    except Exception as e:
        return False, f"{health_url} failed: {e}"


def parse_quick_tunnel_url(text: str) -> str:
    match = _TRYCLOUDFLARE_RE.search(text or "")
    return match.group(0) if match else ""


def is_trycloudflare_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    return host == "trycloudflare.com" or host.endswith(".trycloudflare.com")


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(data: dict[str, Any]) -> None:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = state_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def pid_is_running(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _pid_looks_like_cloudflared(pid: Any) -> bool:
    if os.name != "posix":
        return True
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            ["ps", "-p", str(parsed), "-o", "comm="],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    name = Path((proc.stdout or "").strip()).name
    return name == "cloudflared"


def status() -> dict[str, Any]:
    state = load_state()
    pid = state.get("pid")
    running = pid_is_running(pid) and _pid_looks_like_cloudflared(pid)
    return {
        "running": running,
        "pid": pid if running else None,
        "public_url": str(state.get("public_url") or ""),
        "webhook_url": str(state.get("webhook_url") or ""),
        "managed": bool(state.get("managed")),
        "started_at": state.get("started_at"),
        "state_path": str(state_path()),
        "log_path": str(log_path()),
    }


def _cloudflared_command(binary: str) -> list[str]:
    return [
        binary,
        "tunnel",
        "--config",
        os.devnull,
        "--url",
        local_url(),
        "--no-autoupdate",
    ]


def _platform_asset_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"arm64", "aarch64"}:
        arch = "arm64"
    elif machine.startswith("arm"):
        arch = "arm"
    else:
        raise RuntimeError(f"unsupported CPU architecture for cloudflared: {machine}")

    if system == "darwin":
        if arch == "arm":
            raise RuntimeError("cloudflared does not publish a Darwin armv7 binary")
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "windows":
        if arch != "amd64":
            raise RuntimeError("cloudflared does not publish a Windows arm64 binary")
        return "cloudflared-windows-amd64.exe"
    raise RuntimeError(f"unsupported operating system for cloudflared: {system}")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _urlopen_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "hermes-agent-photon",
        },
    )
    with urllib.request.urlopen(  # noqa: S310
        request,
        timeout=30,
        context=_ssl_context(),
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _cloudflared_release_asset(asset_name: str) -> CloudflaredAsset:
    data = _urlopen_json(CLOUDFLARED_RELEASE_API)
    if not isinstance(data, dict):
        raise RuntimeError("Cloudflare release metadata was not an object")
    for item in data.get("assets") or []:
        if not isinstance(item, dict) or item.get("name") != asset_name:
            continue
        digest = str(item.get("digest") or "")
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Cloudflare release asset {asset_name} has no SHA-256 digest")
        download_url = str(item.get("browser_download_url") or "")
        if not download_url:
            raise RuntimeError(f"Cloudflare release asset {asset_name} has no download URL")
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return CloudflaredAsset(
            name=asset_name,
            download_url=download_url,
            sha256=digest.removeprefix("sha256:"),
            size=size,
            version=str(data.get("tag_name") or ""),
        )
    raise RuntimeError(f"Cloudflare release asset not found: {asset_name}")


def _download_url(url: str, destination: Path) -> None:
    with urllib.request.urlopen(  # noqa: S310
        url,
        timeout=60,
        context=_ssl_context(),
    ) as response:
        with destination.open("wb") as fh:
            shutil.copyfileobj(response, fh)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_download(path: Path, asset: CloudflaredAsset) -> None:
    if asset.size and path.stat().st_size != asset.size:
        raise RuntimeError(
            f"cloudflared download size mismatch for {asset.name}: "
            f"expected {asset.size}, got {path.stat().st_size}"
        )
    actual = _sha256_file(path)
    if actual.lower() != asset.sha256.lower():
        raise RuntimeError(
            f"cloudflared download checksum mismatch for {asset.name}"
        )


def _extract_cloudflared_tgz(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if Path(member.name).name != "cloudflared" or not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                break
            with destination.open("wb") as out:
                shutil.copyfileobj(src, out)
            return
    raise RuntimeError("cloudflared archive did not contain a cloudflared binary")


def _load_managed_cloudflared_manifest() -> dict[str, Any]:
    path = managed_cloudflared_manifest_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_managed_cloudflared_manifest(
    target: Path,
    asset: CloudflaredAsset,
    *,
    binary_sha256: Optional[str] = None,
) -> None:
    try:
        binary_sha256 = binary_sha256 or _sha256_file(target)
        data = {
            "asset": asset.name,
            "asset_sha256": asset.sha256,
            "binary_sha256": binary_sha256,
            "version": asset.version,
        }
        path = managed_cloudflared_manifest_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        pass


def _ensure_executable(path: Path) -> None:
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


def _managed_cloudflared_matches_asset(target: Path, asset: CloudflaredAsset) -> bool:
    try:
        current_binary_sha256 = _sha256_file(target).lower()
    except OSError:
        return False

    manifest = _load_managed_cloudflared_manifest()
    if (
        str(manifest.get("asset") or "") == asset.name
        and str(manifest.get("asset_sha256") or "").lower() == asset.sha256.lower()
        and str(manifest.get("binary_sha256") or "").lower() == current_binary_sha256
    ):
        return True

    if not asset.name.endswith(".tgz") and current_binary_sha256 == asset.sha256.lower():
        _save_managed_cloudflared_manifest(
            target,
            asset,
            binary_sha256=current_binary_sha256,
        )
        return True

    return False


def install_managed_cloudflared(*, emit: Optional[Any] = None) -> str:
    """Install cloudflared into the active Hermes profile and return its path."""
    target = managed_cloudflared_path()
    asset_name = _platform_asset_name()
    try:
        asset = _cloudflared_release_asset(asset_name)
    except Exception:
        if target.exists():
            _ensure_executable(target)
            return str(target)
        raise

    if target.exists() and _managed_cloudflared_matches_asset(target, asset):
        _ensure_executable(target)
        return str(target)

    managed_bin_dir().mkdir(parents=True, exist_ok=True)
    if emit:
        version = f" {asset.version}" if asset.version else ""
        if target.exists():
            emit(f"  updating managed cloudflared copy{version} ({asset_name})")
        else:
            emit(f"  cloudflared not found — installing managed copy{version} ({asset_name})")

    had_target = target.exists()
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-cloudflared-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            download_path = tmp_path / asset_name
            _download_url(asset.download_url, download_path)
            _verify_download(download_path, asset)
            staged = tmp_path / target.name
            if asset_name.endswith(".tgz"):
                _extract_cloudflared_tgz(download_path, staged)
            else:
                shutil.copyfile(download_path, staged)
            os.chmod(staged, 0o755)
            staged.replace(target)
    except Exception:
        if had_target:
            _ensure_executable(target)
            return str(target)
        raise

    _ensure_executable(target)
    _save_managed_cloudflared_manifest(target, asset)
    return str(target)


def resolve_cloudflared_binary(
    *,
    auto_install: bool = True,
    emit: Optional[Any] = None,
) -> Optional[str]:
    binary = shutil.which("cloudflared")
    if binary:
        return binary
    managed = managed_cloudflared_path()
    if managed.exists() and not auto_install:
        return str(managed)
    if not auto_install:
        return None
    return install_managed_cloudflared(emit=emit)


def start(
    timeout_seconds: float = DEFAULT_START_TIMEOUT_SECONDS,
    *,
    auto_install: bool = True,
    on_install: Optional[Any] = None,
) -> TunnelStartResult:
    current = status()
    if current.get("running") and is_trycloudflare_url(str(current.get("public_url") or "")):
        public_url = str(current.get("public_url") or "")
        webhook_url = str(current.get("webhook_url") or "") or webhook_url_for_base(public_url)
        return TunnelStartResult(
            success=True,
            public_url=public_url,
            webhook_url=webhook_url,
            reused=True,
            pid=int(current["pid"]) if current.get("pid") else None,
            log_path=log_path(),
        )

    try:
        binary = resolve_cloudflared_binary(
            auto_install=auto_install,
            emit=on_install,
        )
    except Exception as e:
        return TunnelStartResult(
            success=False,
            error=f"cloudflared install failed: {e}",
            log_path=log_path(),
        )
    if not binary:
        return TunnelStartResult(
            success=False,
            error="cloudflared is not installed or not on PATH",
            log_path=log_path(),
        )

    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    command = _cloudflared_command(binary)
    log_file = log_path()
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting: {' '.join(command)}\n")
        fh.flush()
        proc = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    deadline = time.monotonic() + timeout_seconds
    public_url = ""
    while time.monotonic() < deadline:
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        public_url = parse_quick_tunnel_url(text)
        if public_url:
            break
        if proc.poll() is not None:
            return TunnelStartResult(
                success=False,
                error=f"cloudflared exited before publishing a tunnel URL (exit {proc.returncode})",
                pid=proc.pid,
                log_path=log_file,
                command=command,
            )
        time.sleep(0.2)

    if not public_url:
        return TunnelStartResult(
            success=False,
            error=f"timed out waiting for a trycloudflare.com URL after {timeout_seconds:.0f}s",
            pid=proc.pid,
            log_path=log_file,
            command=command,
        )

    webhook_url = webhook_url_for_base(public_url)
    save_state({
        "managed": True,
        "pid": proc.pid,
        "public_url": public_url,
        "webhook_url": webhook_url,
        "local_url": local_url(),
        "command": command,
        "started_at": int(time.time()),
        "log_path": str(log_file),
    })
    return TunnelStartResult(
        success=True,
        public_url=public_url,
        webhook_url=webhook_url,
        pid=proc.pid,
        log_path=log_file,
        command=command,
    )


def stop(timeout_seconds: float = 5.0) -> dict[str, Any]:
    current = status()
    pid = current.get("pid")
    if not pid:
        return {"stopped": False, "message": "managed tunnel is not running"}

    parsed = int(pid)
    try:
        if hasattr(os, "killpg"):
            os.killpg(parsed, signal.SIGTERM)
        else:
            os.kill(parsed, signal.SIGTERM)
    except ProcessLookupError:
        return {"stopped": False, "message": "managed tunnel was already stopped"}
    except OSError as e:
        return {"stopped": False, "message": f"could not stop tunnel: {e}"}

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not pid_is_running(parsed):
            return {"stopped": True, "message": f"stopped managed tunnel pid {parsed}"}
        time.sleep(0.1)

    try:
        if hasattr(os, "killpg"):
            os.killpg(parsed, signal.SIGKILL)
        else:
            os.kill(parsed, signal.SIGKILL)
    except OSError:
        pass
    return {"stopped": True, "message": f"stopped managed tunnel pid {parsed}"}


def tail_logs(line_count: int = 80) -> str:
    path = log_path()
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])
