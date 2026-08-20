"""Termux browser-hosted Hermes Desktop runtime helpers.

Electron's Linux binaries target glibc and cannot run natively in Termux's
Android/Bionic userspace.  The supported Termux desktop transport therefore
keeps the *renderer and Hermes backend* unchanged, serves the renderer on the
loopback FastAPI surface, and hosts that URL in the Termux:X11 Chromium build.
The same loopback URL remains usable from the phone's normal Android browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping, MutableMapping, Sequence

TERMUX_X11_PACKAGE = "termux-x11-nightly"
TERMUX_CHROMIUM_PACKAGE = "chromium"
TERMUX_X11_ANDROID_PACKAGE = "com.termux.x11"
TERMUX_X11_DISPLAY = ":1"
TERMUX_X11_APK_URL = (
    "https://github.com/termux/termux-x11/releases/download/nightly/"
    "termux-x11-universal-debug.apk"
)
TERMUX_X11_RELEASE_API = "https://api.github.com/repos/termux/termux-x11/releases/tags/nightly"
TERMUX_X11_APK_NAME = "termux-x11-universal-debug.apk"


@dataclass(frozen=True)
class TermuxX11ApkAsset:
    url: str
    sha256: str



@dataclass(frozen=True)
class TermuxDesktopRuntime:
    browser: str
    display: str
    x11: str


def _which_first(names: Sequence[str], which: Callable[[str], str | None] = shutil.which) -> str | None:
    for name in names:
        resolved = which(name)
        if resolved:
            return resolved
    return None


def _run_ok(
    argv: Sequence[str],
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    env: Mapping[str, str] | None = None,
    quiet: bool = False,
) -> bool:
    kwargs = {
        "check": False,
        "env": dict(env) if env is not None else None,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return run(list(argv), **kwargs).returncode == 0


def termux_x11_android_app_installed(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> bool:
    """Return whether the Termux:X11 Android companion application is installed."""
    pm = "/system/bin/pm" if Path("/system/bin/pm").exists() else which("pm")
    if not pm:
        return False
    return _run_ok([pm, "path", TERMUX_X11_ANDROID_PACKAGE], run=run, quiet=True)


def ensure_termux_x11_packages(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
) -> TermuxDesktopRuntime:
    """Install missing Termux:X11/Chromium packages and resolve their executables."""
    pkg = which("pkg")
    if not pkg:
        raise RuntimeError("Termux package manager `pkg` was not found on PATH")

    x11 = which("termux-x11")
    browser = _which_first(("chromium-browser", "chromium"), which)
    if not x11 or not browser:
        # x11-repo is the official repository carrying both the X11 companion
        # and GUI browser packages. Installing it is idempotent.
        if not _run_ok([pkg, "install", "-y", "x11-repo"], run=run, env=env):
            raise RuntimeError("Could not enable the official Termux x11-repo")

        missing_packages: list[str] = []
        if not x11:
            missing_packages.append(TERMUX_X11_PACKAGE)
        if not browser:
            missing_packages.append(TERMUX_CHROMIUM_PACKAGE)
        if missing_packages and not _run_ok(
            [pkg, "install", "-y", *missing_packages], run=run, env=env
        ):
            raise RuntimeError(
                "Could not install Termux desktop packages: "
                + ", ".join(missing_packages)
            )

        x11 = which("termux-x11")
        browser = _which_first(("chromium-browser", "chromium"), which)

    if not x11:
        raise RuntimeError("`termux-x11` is still unavailable after package installation")
    if not browser:
        raise RuntimeError("Termux Chromium is still unavailable after package installation")

    return TermuxDesktopRuntime(browser=browser, display=TERMUX_X11_DISPLAY, x11=x11)


def resolve_official_termux_x11_apk(
    *,
    urlopen=urllib.request.urlopen,
    timeout: float = 20.0,
) -> TermuxX11ApkAsset | None:
    """Resolve the current official nightly APK and its GitHub-published digest."""
    request = urllib.request.Request(
        TERMUX_X11_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "hermes-agent-termux-desktop",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None

    for asset in payload.get("assets") or []:
        if asset.get("name") != TERMUX_X11_APK_NAME:
            continue
        url = str(asset.get("browser_download_url") or "").strip()
        digest = str(asset.get("digest") or "").strip().lower()
        if url != TERMUX_X11_APK_URL or not digest.startswith("sha256:"):
            return None
        sha256 = digest.removeprefix("sha256:")
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            return None
        return TermuxX11ApkAsset(url=url, sha256=sha256)
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_termux_x11_android_app(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    env: MutableMapping[str, str] | None = None,
    wait_seconds: float = 90.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Download the official Termux:X11 APK and open Android's installer.

    Android intentionally requires user confirmation for installing another APK;
    Hermes does not attempt to bypass that platform security boundary.  We wait
    briefly for the one-time confirmation so the same `hermes desktop` command
    can continue when the user accepts it.
    """
    if termux_x11_android_app_installed(run=run, which=which):
        return True

    curl = which("curl")
    opener = which("termux-open")
    if not curl or not opener:
        return False

    active_env = env if env is not None else os.environ
    asset = resolve_official_termux_x11_apk()
    if asset is None:
        return False

    cache_root = Path(active_env.get("TMPDIR") or (Path.home() / ".cache" / "hermes"))
    cache_root.mkdir(parents=True, exist_ok=True)
    apk = cache_root / TERMUX_X11_APK_NAME
    pending_apk = apk.with_suffix(".apk.part")
    pending_apk.unlink(missing_ok=True)

    if not _run_ok(
        [
            curl,
            "--fail",
            "--location",
            "--retry",
            "3",
            "--output",
            str(pending_apk),
            asset.url,
        ],
        run=run,
        env=active_env,
    ):
        pending_apk.unlink(missing_ok=True)
        return False

    if _sha256_file(pending_apk) != asset.sha256:
        pending_apk.unlink(missing_ok=True)
        return False
    pending_apk.replace(apk)

    # termux-open delegates the verified APK to Android's package installer. This is the
    # supported non-root path and preserves Android's explicit install prompt.
    if not _run_ok([opener, str(apk)], run=run, env=active_env):
        return False

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        if termux_x11_android_app_installed(run=run, which=which):
            return True
        sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return termux_x11_android_app_installed(run=run, which=which)


def termux_x11_display_ready(
    *, env: Mapping[str, str] | None = None, display: str = TERMUX_X11_DISPLAY
) -> bool:
    """Return whether the Termux X socket for *display* already exists."""
    active_env = env if env is not None else os.environ
    number = display.lstrip(":").split(".", 1)[0]
    if not number.isdigit():
        return False
    tmpdir = active_env.get("TMPDIR")
    if not tmpdir:
        return False
    return (Path(tmpdir) / ".X11-unix" / f"X{number}").exists()


def _foreground_termux_x11_activity(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    env: Mapping[str, str] | None = None,
) -> None:
    am = which("am")
    if not am:
        return
    _run_ok(
        [
            am,
            "start",
            "--user",
            "0",
            "-n",
            "com.termux.x11/com.termux.x11.MainActivity",
        ],
        run=run,
        env=env,
        quiet=True,
    )


def launch_termux_x11(
    runtime: TermuxDesktopRuntime,
    *,
    env: MutableMapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    sleep: Callable[[float], None] = time.sleep,
    ready_timeout: float = 8.0,
) -> subprocess.Popen | None:
    """Start the X server on :1 and bring the Android activity to foreground.

    Reuses an already-live X socket so repeated `hermes desktop` launches do
    not create competing servers for the same display.
    """
    active_env = env if env is not None else os.environ
    active_env["DISPLAY"] = runtime.display

    if termux_x11_display_ready(env=active_env, display=runtime.display):
        _foreground_termux_x11_activity(run=run, which=which, env=active_env)
        return None

    log_root = Path(active_env.get("TMPDIR") or (Path.home() / ".cache" / "hermes"))
    log_root.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_root / "termux-x11.log", "ab", buffering=0)
    try:
        proc = popen(
            [runtime.x11, runtime.display],
            env=dict(active_env),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # Popen duplicated the descriptor for its child; the parent must not keep
        # a second log handle around for every Desktop launch.
        log_handle.close()

    _foreground_termux_x11_activity(run=run, which=which, env=active_env)

    deadline = time.monotonic() + max(0.0, ready_timeout)
    while time.monotonic() < deadline:
        if termux_x11_display_ready(env=active_env, display=runtime.display):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                "Termux:X11 exited before DISPLAY became ready; see $TMPDIR/termux-x11.log"
            )
        sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    if not termux_x11_display_ready(env=active_env, display=runtime.display):
        raise RuntimeError(
            f"Termux:X11 did not make DISPLAY={runtime.display} ready within "
            f"{max(0.0, ready_timeout):g}s; see $TMPDIR/termux-x11.log"
        )
    return proc


def chromium_browser_spec(runtime: TermuxDesktopRuntime) -> str:
    """Return a Python-webbrowser BROWSER command for a standalone app window."""
    return (
        f"{shlex.quote(runtime.browser)} "
        "--app=%s --no-first-run --no-default-browser-check"
    )
