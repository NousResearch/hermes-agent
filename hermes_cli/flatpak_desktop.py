"""Linux Flatpak lifecycle for the official Hermes Desktop client.

This module owns only the Flathub Electron client. Hermes CLI, gateway,
configuration, credentials, plugins, skills, cron, and workspace data remain in
the user's normal native Hermes installation.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

APP_ID = "com.nousresearch.Hermes"
FLATHUB_REMOTE = "flathub"
FLATHUB_REPO = "https://dl.flathub.org/repo/flathub.flatpakrepo"
_READY_TIMEOUT_SECONDS = 90


def _flatpak() -> Optional[str]:
    return shutil.which("flatpak")


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, **kwargs)


def _available() -> bool:
    if _flatpak():
        return True
    print("Hermes Desktop requires Flatpak on Linux. Install Flatpak, then run `hermes desktop` again.")
    return False


def ensure_installed() -> bool:
    """Ensure the official user-scoped Hermes Desktop Flatpak is installed."""
    flatpak = _flatpak()
    if not flatpak:
        _available()
        return False

    remote = _run([flatpak, "remote-add", "--if-not-exists", "--user", FLATHUB_REMOTE, FLATHUB_REPO])
    if remote.returncode:
        print("Could not add the Flathub remote for Hermes Desktop.")
        return False

    present = _run([flatpak, "info", "--user", APP_ID], capture_output=True, text=True)
    if present.returncode == 0:
        return True

    print("Installing Hermes Desktop from Flathub...")
    installed = _run([flatpak, "install", "--user", "--noninteractive", FLATHUB_REMOTE, APP_ID])
    if installed.returncode == 0:
        return True
    print("Hermes Desktop installation from Flathub failed.")
    return False


def update_if_installed() -> bool:
    """Update an already-installed official user Flatpak after `hermes update`."""
    flatpak = _flatpak()
    if not flatpak:
        return False
    present = _run([flatpak, "info", "--user", APP_ID], capture_output=True, text=True)
    if present.returncode:
        return False
    print("Updating Hermes Desktop Flatpak...")
    result = _run([flatpak, "update", "--user", "--noninteractive", APP_ID])
    if result.returncode:
        print("Hermes Desktop Flatpak update failed; native Hermes was still updated.")
        return False
    return True


def _start_native_backend(command: list[str], *, cwd: str) -> tuple[subprocess.Popen, str, str]:
    """Start native `hermes serve` and return its loopback URL and random token."""
    token = secrets.token_urlsafe(32)
    ready_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="hermes-desktop-ready-", delete=False)
    ready_path = Path(ready_handle.name)
    ready_handle.close()
    ready_path.unlink(missing_ok=True)

    env = dict(os.environ)
    env["HERMES_DASHBOARD_SESSION_TOKEN"] = token
    env["HERMES_DESKTOP_READY_FILE"] = str(ready_path)
    env["HERMES_DESKTOP"] = "1"
    # Ensure the native backend has access to system tools
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin:" + env.get("PATH", "")

    process = subprocess.Popen(
        [*command, "serve", "--isolated", "--host", "127.0.0.1", "--port", "0"],
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("native Hermes backend exited before it became ready")
            try:
                payload = json.loads(ready_path.read_text(encoding="utf-8"))
                port = int(payload["port"])
                if port > 0:
                    return process, f"http://127.0.0.1:{port}", token
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                pass
            time.sleep(0.1)
    except Exception:
        _stop_native_backend(process)
        raise
    finally:
        ready_path.unlink(missing_ok=True)

    _stop_native_backend(process)
    raise RuntimeError("timed out waiting for the native Hermes backend to become ready")


def _stop_native_backend(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def launch_with_native_backend(command: list[str], *, cwd: str) -> int:
    """Launch the Flatpak client against a temporary native local backend."""
    if not ensure_installed():
        return 1
    flatpak = _flatpak()
    assert flatpak is not None

    try:
        backend, base_url, token = _start_native_backend(command, cwd=cwd)
    except RuntimeError as exc:
        print(f"Could not start the native Hermes backend: {exc}")
        return 1

    try:
        result = _run(
            [
                flatpak,
                "run",
                f"--env=HERMES_DESKTOP_REMOTE_URL={base_url}",
                f"--env=HERMES_DESKTOP_REMOTE_TOKEN={token}",
                APP_ID,
            ],
            cwd=cwd,
        )
        return result.returncode
    finally:
        _stop_native_backend(backend)
