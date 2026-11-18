"""Fixed-argv browser launcher for the graphical Context Cockpit.

Safety contract:
  * Never shell=True
  * Never interpolate arbitrary user text into a shell
  * Only launch known entrypoints (hermes-context-visor / python + script)
  * Bind cockpit server to localhost only
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from .web import build_cockpit_url, open_browser


@dataclass
class LaunchResult:
    ok: bool
    message: str
    argv: List[str] = field(default_factory=list)
    method: str = ""
    already_running: bool = False
    url: str = ""


def default_profile() -> str:
    return (
        os.environ.get("HERMES_VISOR_PROFILE")
        or os.environ.get("HERMES_PROFILE")
        or "default"
    )


def resolve_visor_script(profile: str) -> Path:
    override = os.environ.get("HERMES_VISOR_SCRIPT")
    if override:
        return Path(override)
    # Prefer the copy shipped inside this plugin.
    bundled = Path(__file__).resolve().parents[1] / "context_visor.py"
    if bundled.exists():
        return bundled
    return Path.home() / ".hermes" / "profiles" / profile / "scripts" / "context_visor.py"


def resolve_visor_bin() -> Optional[Path]:
    override = os.environ.get("HERMES_CONTEXT_VISOR_BIN")
    if override and Path(override).exists():
        return Path(override)
    which = shutil.which("hermes-context-visor")
    if which:
        return Path(which)
    local = Path.home() / ".local" / "bin" / "hermes-context-visor"
    if local.exists():
        return local
    try:
        bundled = Path(__file__).resolve().parents[1] / "hermes-context-visor"
        if bundled.exists():
            return bundled
    except Exception:
        pass
    return None


def resolve_python() -> Path:
    override = os.environ.get("HERMES_PYTHON")
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3"


def default_server_port(profile: Optional[str] = None) -> int:
    profile = profile or default_profile()
    return 8421 + (sum(ord(ch) for ch in profile) % 200)


def build_visor_url(profile: Optional[str] = None, host: str = "127.0.0.1", port: Optional[int] = None) -> str:
    profile = profile or default_profile()
    return build_cockpit_url(host, port or default_server_port(profile))


def build_visor_argv(profile: Optional[str] = None, extra: Sequence[str] = ()) -> List[str]:
    profile = profile or default_profile()
    if not profile.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"invalid profile name: {profile!r}")
    for arg in extra:
        if not isinstance(arg, str) or any(ch in arg for ch in "\n\r;|&`$"):
            raise ValueError("refusing unsafe extra argv")

    bin_path = resolve_visor_bin()
    if bin_path is not None and bin_path.exists():
        return [str(bin_path), *list(extra)]

    script = resolve_visor_script(profile)
    py = resolve_python()
    return [str(py), str(script), "--profile", profile, *list(extra)]


def _spawn_env(profile: str) -> dict:
    env = os.environ.copy()
    env["HERMES_VISOR_PROFILE"] = profile
    return env


def _server_ready(url: str, profile: str) -> bool:
    health = f"{url.rstrip('/')}/healthz?profile={profile}"
    try:
        with urlopen(health, timeout=1.5) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            return bool(body.get("ok")) and body.get("profile") == profile
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def _fetch_status(url: str, profile: str) -> Optional[Dict[str, Any]]:
    endpoint = f"{url.rstrip('/')}/api/status"
    try:
        with urlopen(endpoint, timeout=1.5) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    metrics = body.get("metrics") or {}
    if metrics.get("profile") != profile:
        return None
    return body


def _runtime_ready(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    metrics = payload.get("metrics") or {}
    live = metrics.get("liveness") or {}
    freshness = metrics.get("freshness")
    return bool(live.get("running")) or freshness in {"fresh", "quiet", "idle", "stale"}


def _warm_runtime_status(url: str, profile: str, *, deadline: float) -> Optional[Dict[str, Any]]:
    last = None
    while time.time() < deadline:
        last = _fetch_status(url, profile)
        if _runtime_ready(last):
            return last
        time.sleep(0.25)
    return last


def platform_fallback_instructions(profile: Optional[str] = None) -> str:
    profile = profile or default_profile()
    port = default_server_port(profile)
    url = build_visor_url(profile, port=port)
    return (
        "Could not open the graphical cockpit automatically.\n"
        "Start the local browser cockpit with:\n"
        f"  hermes-context-visor --profile {profile} --serve --port {port}\n"
        "Then open this URL in a browser:\n"
        f"  {url}"
    )


def _spawn_server(argv: List[str], *, profile: str) -> None:
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": _spawn_env(profile),
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(argv, **kwargs)


def launch_context_visor(
    profile: Optional[str] = None,
    *,
    force: bool = False,
) -> LaunchResult:
    profile = profile or default_profile()
    port = default_server_port(profile)
    url = build_visor_url(profile, port=port)
    if not force and _server_ready(url, profile):
        opened = open_browser(url)
        if opened:
            return LaunchResult(
                True,
                "Graphical Context Cockpit already running; opened it in the browser.",
                [],
                "browser_existing",
                already_running=True,
                url=url,
            )
        return LaunchResult(
            True,
            f"Graphical Context Cockpit already running at {url}.",
            [],
            "browser_existing",
            already_running=True,
            url=url,
        )

    extra = ["--serve", "--host", "127.0.0.1", "--port", str(port), "--no-browser"]
    try:
        argv = build_visor_argv(profile, extra=extra)
    except ValueError as exc:
        return LaunchResult(False, str(exc), [], "rejected", url=url)

    if not Path(argv[0]).exists() and shutil.which(argv[0]) is None:
        return LaunchResult(
            False,
            f"Visor entrypoint not found: {argv[0]}\n{platform_fallback_instructions(profile)}",
            argv,
            "missing",
            url=url,
        )
    if len(argv) >= 2 and argv[0].endswith(("python", "python3")) and not Path(argv[1]).exists():
        return LaunchResult(
            False,
            f"Visor script not found: {argv[1]}\n{platform_fallback_instructions(profile)}",
            argv,
            "missing",
            url=url,
        )

    try:
        _spawn_server(argv, profile=profile)
    except Exception as exc:
        return LaunchResult(
            False,
            f"Launch failed: {exc}\n{platform_fallback_instructions(profile)}",
            argv,
            "failed",
            url=url,
        )

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if _server_ready(url, profile):
            payload = _warm_runtime_status(
                url,
                profile,
                deadline=min(deadline, time.time() + 3.0),
            )
            opened = open_browser(url)
            warming = payload is not None and not _runtime_ready(payload)
            if opened:
                message = f"Started graphical Context Cockpit at {url} and opened it in the browser."
            else:
                message = f"Started graphical Context Cockpit at {url}. Open that URL in a browser if it did not appear automatically."
            if warming:
                message += (
                    " Hermes is still warming up; if the banner briefly shows HERMES OFFLINE, "
                    "wait a few seconds for auto-refresh or confirm Desktop is open on personal-ops."
                )
            return LaunchResult(True, message, argv, "browser_started", already_running=False, url=url)
        time.sleep(0.2)

    return LaunchResult(
        False,
        f"Cockpit server did not become ready at {url}.\n{platform_fallback_instructions(profile)}",
        argv,
        "timeout",
        url=url,
    )
