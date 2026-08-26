"""Shared helpers for attaching Hermes to a local Chromium-family CDP port."""

from __future__ import annotations

import logging
import ntpath
import os
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_BROWSER_CDP_PORT = 9222
DEFAULT_BROWSER_CDP_URL = f"http://127.0.0.1:{DEFAULT_BROWSER_CDP_PORT}"

_DARWIN_APPS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)

_WINDOWS_BROWSER_GROUPS = (
    (("chrome.exe", "chrome"), (("Google", "Chrome", "Application", "chrome.exe"),)),
    (
        ("chromium.exe", "chromium"),
        (("Chromium", "Application", "chrome.exe"), ("Chromium", "Application", "chromium.exe")),
    ),
    (("brave.exe", "brave"), (("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),)),
    (("msedge.exe", "msedge"), (("Microsoft", "Edge", "Application", "msedge.exe"),)),
)

_WINDOWS_BIN_NAMES = tuple(name for names, _ in _WINDOWS_BROWSER_GROUPS for name in names)
_WINDOWS_INSTALL_PARTS = tuple(parts for _, group in _WINDOWS_BROWSER_GROUPS for parts in group)

_LINUX_BROWSER_GROUPS = (
    (
        ("google-chrome", "google-chrome-stable"),
        ("/opt/google/chrome/chrome", "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"),
    ),
    (
        ("chromium-browser", "chromium"),
        ("/usr/bin/chromium-browser", "/usr/bin/chromium"),
    ),
    (
        ("brave-browser", "brave-browser-stable", "brave"),
        (
            "/usr/bin/brave-browser",
            "/usr/bin/brave-browser-stable",
            "/usr/bin/brave",
            "/snap/bin/brave",
            "/opt/brave.com/brave/brave-browser",
            "/opt/brave.com/brave/brave",
            "/opt/brave-bin/brave",
        ),
    ),
    (
        ("microsoft-edge", "microsoft-edge-stable", "msedge"),
        (
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
            "/opt/microsoft/msedge/microsoft-edge",
            "/opt/microsoft/msedge/msedge",
        ),
    ),
)

_LINUX_BIN_NAMES = tuple(name for names, _ in _LINUX_BROWSER_GROUPS for name in names)
_LINUX_INSTALL_PATHS = tuple(path for _, paths in _LINUX_BROWSER_GROUPS for path in paths)


# ---------------------------------------------------------------------------
# Real-profile (default Chromium) resolution
#
# Used by the browser tool's ``browser.use_real_profile`` consent path: when a
# local Chromium is launched, point agent-browser at the user's REAL default
# browser profile (``--profile <user-data-dir>`` + ``--executable-path``) so
# their live logins/cookies are available. Only Chromium-family browsers are
# supported; a non-Chromium default (Firefox, Safari) resolves to None and the
# caller fails closed with a clear message.
# ---------------------------------------------------------------------------

# Canonical Chromium browser keys we support for real-profile driving.
_CHROMIUM_BROWSERS = ("chrome", "edge", "brave", "chromium")

# Windows UserChoice ProgId prefixes → canonical browser key. Matched
# case-insensitively by prefix so channel/version suffixes (e.g.
# ``ChromeHTML.X``, ``MSEdgeHTM``) still resolve.
_WINDOWS_PROGID_MAP = (
    ("chromehtml", "chrome"),
    ("msedgehtm", "edge"),
    ("bravehtml", "brave"),
    ("chromiumhtm", "chromium"),
)

# Linux xdg default-web-browser .desktop name fragments → canonical key.
_LINUX_DESKTOP_MAP = (
    ("google-chrome", "chrome"),
    ("chromium", "chromium"),
    ("brave", "brave"),
    ("microsoft-edge", "edge"),
    ("msedge", "edge"),
)

# macOS LaunchServices bundle-id fragments → canonical key.
_DARWIN_BUNDLE_MAP = (
    ("com.google.chrome", "chrome"),
    ("com.microsoft.edgemac", "edge"),
    ("com.brave.browser", "brave"),
    ("org.chromium.chromium", "chromium"),
)


def _real_profile_relparts(browser: str) -> tuple:
    """(mac_support_subdir, windows_localappdata_parts, linux_config_name)."""
    return {
        "chrome": (
            ("Google", "Chrome"),
            ("Google", "Chrome", "User Data"),
            "google-chrome",
        ),
        "edge": (
            ("Microsoft Edge",),
            ("Microsoft", "Edge", "User Data"),
            "microsoft-edge",
        ),
        "brave": (
            ("BraveSoftware", "Brave-Browser"),
            ("BraveSoftware", "Brave-Browser", "User Data"),
            "BraveSoftware/Brave-Browser",
        ),
        "chromium": (
            ("Chromium",),
            ("Chromium", "User Data"),
            "chromium",
        ),
    }[browser]


def real_profile_data_dir(browser: str, system: str | None = None) -> str | None:
    """Return the default user-data-dir for a Chromium ``browser`` on ``system``.

    Returns None for unknown browsers. Does not check existence — callers that
    need that should stat the result. Paths are built with the TARGET system's
    separator (posix for Darwin/Linux, backslash for Windows) so an explicit
    ``system`` argument resolves correctly regardless of the host OS.
    """
    if browser not in _CHROMIUM_BROWSERS:
        return None
    system = system or platform.system()
    mac_parts, win_parts, linux_name = _real_profile_relparts(browser)
    home = os.path.expanduser("~")
    if system == "Darwin":
        return posixpath.join(home, "Library", "Application Support", *mac_parts)
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA") or ntpath.join(home, "AppData", "Local")
        return ntpath.join(local, *win_parts)
    # Linux / other POSIX
    config = os.environ.get("XDG_CONFIG_HOME") or posixpath.join(home, ".config")
    return posixpath.join(config, *linux_name.split("/"))


def chromium_executable(browser: str, system: str | None = None) -> str | None:
    """Return the first present executable for a Chromium ``browser``."""
    if browser not in _CHROMIUM_BROWSERS:
        return None
    system = system or platform.system()

    def first_present(paths: tuple) -> str | None:
        for p in paths:
            if p and os.path.isfile(p):
                return p
        return None

    if system == "Darwin":
        app = {
            "chrome": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "chromium": "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "brave": "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "edge": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        }[browser]
        return app if os.path.isfile(app) else None
    if system == "Windows":
        groups = {
            "chrome": (("Google", "Chrome", "Application", "chrome.exe"),),
            "chromium": (("Chromium", "Application", "chrome.exe"), ("Chromium", "Application", "chromium.exe")),
            "brave": (("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),),
            "edge": (("Microsoft", "Edge", "Application", "msedge.exe"),),
        }[browser]
        bases = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")),
        ]
        cands = tuple(os.path.join(base, *parts) for base in bases for parts in groups)
        return first_present(cands)
    # Linux
    linux = {
        "chrome": ("google-chrome", "google-chrome-stable"),
        "chromium": ("chromium-browser", "chromium"),
        "brave": ("brave-browser", "brave-browser-stable", "brave"),
        "edge": ("microsoft-edge", "microsoft-edge-stable"),
    }[browser]
    for name in linux:
        found = shutil.which(name)
        if found:
            return found
    # fall back to the known absolute paths from the launch tables
    for names, paths in _LINUX_BROWSER_GROUPS:
        if any(n in linux for n in names):
            hit = first_present(tuple(paths))
            if hit:
                return hit
    return None


def _detect_default_windows() -> str | None:
    try:
        import winreg  # type: ignore
    except Exception:
        return None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        )
        prog_id, _ = winreg.QueryValueEx(key, "ProgId")
        winreg.CloseKey(key)
    except Exception:
        return None
    low = str(prog_id or "").lower()
    for prefix, browser in _WINDOWS_PROGID_MAP:
        if low.startswith(prefix):
            return browser
    return None


_LS_HANDLERS_READER = (
    "defaults",
    "read",
    "com.apple.LaunchServices/com.apple.launchservices.secure",
    "LSHandlers",
)


def _launchservices_https_handler(dump: str) -> str | None:
    """Return the bundle id registered for the ``https`` URL scheme.

    ``dump`` is the ``defaults read … LSHandlers`` output: an array of
    ``{ … }`` dictionaries, one per handler. Only the entry whose
    ``LSHandlerURLScheme`` is ``https`` counts — a browser registered for
    another scheme or a file type must not be mistaken for the default.
    Returns None when no https handler is recorded, which is what macOS
    stores while Safari (the implicit default) has never been replaced.
    """
    entries: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in dump:
        if ch == "{":
            depth += 1
            if depth == 1:
                buf = []
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                entries.append("".join(buf))
                continue
        if depth >= 1:
            buf.append(ch)
    for entry in entries:
        low = entry.lower()
        if not re.search(r'lshandlerurlscheme\s*=\s*"?https"?\s*;', low):
            continue
        # The nested LSHandlerPreferredVersions block carries "-" placeholders
        # under the same key names; the real bundle id is the first non-"-".
        for role in re.findall(r'lshandlerrole(?:all|viewer)\s*=\s*"?([a-z0-9.\-]+)"?\s*;', low):
            if role != "-":
                return role
        return None
    return None


def _detect_default_darwin() -> str | None:
    try:
        out = subprocess.run(
            list(_LS_HANDLERS_READER),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        ).stdout
    except Exception:
        return None
    bundle = _launchservices_https_handler(out)
    if not bundle:
        return None
    for frag, browser in _DARWIN_BUNDLE_MAP:
        if bundle.startswith(frag):
            return browser
    # A non-Chromium https handler (Safari, Firefox, Arc, …): fail closed.
    # No "first installed Chromium wins" fallback — that would drive a
    # browser the user never made their default.
    return None


def _detect_default_linux() -> str | None:
    try:
        out = subprocess.run(
            ["xdg-settings", "get", "default-web-browser"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        ).stdout.strip().lower()
    except Exception:
        out = ""
    for frag, browser in _LINUX_DESKTOP_MAP:
        if frag in out:
            return browser
    return None


def detect_default_chromium(system: str | None = None) -> str | None:
    """Return the canonical key of the default Chromium browser, or None.

    None means the default browser is non-Chromium (Firefox, Safari) or could
    not be determined — the caller fails closed rather than guessing.
    """
    system = system or platform.system()
    if system == "Windows":
        return _detect_default_windows()
    if system == "Darwin":
        return _detect_default_darwin()
    return _detect_default_linux()


def get_chrome_debug_candidates(system: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen or not os.path.isfile(path):
            return
        candidates.append(path)
        seen.add(normalized)

    def add_windows_install_paths(
        bases: tuple[str | None, ...],
        install_groups: tuple[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]], ...],
    ) -> None:
        for _, group in install_groups:
            for base in filter(None, bases):
                for parts in group:
                    # Only called with WSL ``/mnt/c/...`` bases — those are
                    # POSIX paths regardless of the host OS, so join with
                    # posixpath (os.path.join would emit backslashes on nt).
                    add(posixpath.join(base, *parts))

    if system == "Darwin":
        for app in _DARWIN_APPS:
            add(app)
        return candidates

    if system == "Windows":
        install_bases = (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        )
        for names, install_parts in _WINDOWS_BROWSER_GROUPS:
            for name in names:
                add(shutil.which(name))
            for base in filter(None, install_bases):
                for parts in install_parts:
                    add(os.path.join(base, *parts))
        return candidates

    for names, paths in _LINUX_BROWSER_GROUPS:
        for name in names:
            add(shutil.which(name))
        for path in paths:
            add(path)
    add_windows_install_paths(("/mnt/c/Program Files", "/mnt/c/Program Files (x86)"), _WINDOWS_BROWSER_GROUPS)
    return candidates


def chrome_debug_data_dir() -> str:
    return str(get_hermes_home() / "chrome-debug")


def _chrome_debug_args(port: int) -> list[str]:
    return [
        f"--remote-debugging-port={port}",
        f"--user-data-dir={chrome_debug_data_dir()}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def is_browser_debug_ready(url: str, timeout: float = 1.0) -> bool:
    """Return True when ``url`` exposes a reachable Chrome DevTools endpoint."""
    import socket
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return False

    if parsed.scheme in {"ws", "wss"} and parsed.path.startswith("/devtools/browser/"):
        if not parsed.hostname:
            return False
        try:
            with socket.create_connection((parsed.hostname, port), timeout=timeout):
                return True
        except OSError:
            return False

    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if scheme not in {"http", "https"} or not parsed.netloc:
        return False

    root = f"{scheme}://{parsed.netloc}".rstrip("/")
    for probe in (f"{root}/json/version", f"{root}/json"):
        try:
            with urllib.request.urlopen(probe, timeout=timeout) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    return True
        except Exception:
            continue
    return False


# Both loopback literals: Windows (and some Linux setups) can hand the IPv4
# loopback to one process and the IPv6 loopback to another. Chrome asked to
# bind :9222 while e.g. VS Code's js-debug holds 127.0.0.1:9222 will come up
# on [::1]:9222 only — reachable, but invisible to an IPv4-only probe.
_LOOPBACK_PROBE_HOSTS = ("127.0.0.1", "[::1]")
_LOOPBACK_SOCKET_HOSTS = ("127.0.0.1", "::1")


def discover_local_cdp_url(port: int, timeout: float = 1.0) -> str | None:
    """Return the first loopback URL (IPv4 first, then IPv6) speaking CDP.

    Dual-stack discovery: when another application squats the IPv4
    loopback on ``port``, a debug browser launched with
    ``--remote-debugging-port`` may bind only ``[::1]``. Probing both
    literals finds it either way. Returns ``None`` when neither
    loopback exposes a CDP discovery endpoint.
    """
    for host in _LOOPBACK_PROBE_HOSTS:
        url = f"http://{host}:{port}"
        if is_browser_debug_ready(url, timeout=timeout):
            return url
    return None


def local_port_in_use(port: int, timeout: float = 0.5) -> bool:
    """Return True when either loopback accepts TCP on ``port``.

    Callers use this AFTER a failed CDP probe to distinguish "port is
    free, we can launch a browser on it" from "another application
    (IDE debugger, dev server) is squatting the port and a launch
    would fight it".
    """
    import socket

    for host in _LOOPBACK_SOCKET_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def find_free_debug_port(preferred: int = DEFAULT_BROWSER_CDP_PORT, attempts: int = 10) -> int:
    """Return the first port after ``preferred`` bindable on both loopbacks.

    Used when ``preferred`` is occupied by a non-CDP application: rather
    than launching a browser into a bind conflict, pick a nearby free
    port. Falls back to ``preferred + 1`` if nothing binds (the launch
    will then fail with a clear browser-side error instead of silently
    doing nothing).
    """
    import socket

    for port in range(preferred + 1, preferred + 1 + attempts):
        bindable = True
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((host, port))
            except OSError:
                bindable = False
                break
        if bindable:
            return port
    return preferred + 1


def manual_chrome_debug_command(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> str | None:
    system = system or platform.system()
    candidates = get_chrome_debug_candidates(system)

    if candidates:
        argv = [candidates[0], *_chrome_debug_args(port)]
        return subprocess.list2cmdline(argv) if system == "Windows" else shlex.join(argv)

    if system == "Darwin":
        data_dir = chrome_debug_data_dir()
        return (
            f'open -a "Google Chrome" --args --remote-debugging-port={port} '
            f'--user-data-dir="{data_dir}" --no-first-run --no-default-browser-check'
        )

    return None


def _detach_kwargs(system: str) -> dict:
    if system != "Windows":
        return {"start_new_session": True}
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    return {"creationflags": flags} if flags else {}


def _wait_for_browser_debug_ready_or_exit(
    proc: subprocess.Popen,
    port: int,
    timeout: float = 2.0,
    interval: float = 0.1,
) -> str:
    """Classify a launched browser as ready, exited, or still starting.

    We only need to wait long enough to catch the common failure mode where a
    candidate binary exists but exits immediately before exposing the CDP port.
    Slower browsers can still finish starting after this grace window.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Dual-stack: a squatter on the IPv4 loopback can push the browser
        # to bind [::1] only — check both so a successful launch is seen.
        if discover_local_cdp_url(port, timeout=min(interval, 0.2)):
            return "ready"
        if proc.poll() is not None:
            return "exited"
        time.sleep(interval)

    return "starting"


_LAUNCH_STDERR_LOG = "launch-stderr.log"
_STDERR_TAIL_LIMIT = 2000


@dataclass
class LaunchAttempt:
    """Outcome of one candidate-binary launch attempt."""

    binary: str
    state: str  # "ready" | "starting" | "exited" | "spawn-failed"
    returncode: int | None = None
    stderr_tail: str = ""


@dataclass
class ChromeDebugLaunch:
    """Structured result of ``launch_chrome_debug``.

    ``launched`` mirrors the legacy boolean contract: a launch command was
    executed and the browser is ready or still starting (it does NOT
    guarantee the CDP port ever opens). ``attempts`` carries per-candidate
    diagnostics so callers can explain *why* nothing came up.
    """

    launched: bool = False
    attempts: list[LaunchAttempt] = field(default_factory=list)

    @property
    def hint(self) -> str | None:
        """Best user-facing explanation for a failed/soft launch, if any."""
        for attempt in self.attempts:
            if attempt.state == "exited" and attempt.returncode == 0:
                name = os.path.basename(attempt.binary)
                return (
                    f"{name} exited immediately without opening the debug port — an already-running "
                    f"{name} instance likely absorbed the launch (Chromium's single-instance "
                    "behavior). Close ALL of its processes (including background/tray instances) "
                    "and retry /browser connect."
                )
        for attempt in self.attempts:
            if attempt.state == "exited" and attempt.stderr_tail:
                return (
                    f"{os.path.basename(attempt.binary)} exited before the debug port opened: "
                    f"{attempt.stderr_tail.splitlines()[-1].strip()}"
                )
        return None


def _read_stderr_tail(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return data[-_STDERR_TAIL_LIMIT:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def launch_chrome_debug(
    port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None
) -> ChromeDebugLaunch:
    """Launch a Chromium-family browser with remote debugging, with diagnostics.

    Tries each detected candidate binary in turn. A candidate that exits
    before the CDP port opens (crash, singleton forward to an existing
    instance, bad profile dir) is logged — with exit code and a stderr tail —
    and the next candidate is tried.
    """
    system = system or platform.system()
    result = ChromeDebugLaunch()
    candidates = get_chrome_debug_candidates(system)
    if not candidates:
        logger.info("browser debug launch: no Chromium-family binary found (system=%s)", system)
        return result

    data_dir = chrome_debug_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    stderr_path = os.path.join(data_dir, _LAUNCH_STDERR_LOG)

    for candidate in candidates:
        try:
            with open(stderr_path, "wb") as stderr_file:
                proc = subprocess.Popen(
                    [candidate, *_chrome_debug_args(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    **_detach_kwargs(system),
                )
        except Exception as exc:
            result.attempts.append(LaunchAttempt(binary=candidate, state="spawn-failed"))
            logger.info("browser debug launch: failed to spawn %s: %s", candidate, exc)
            continue

        logger.info(
            "browser debug launch: spawned %s (pid=%s) with --remote-debugging-port=%d",
            candidate,
            getattr(proc, "pid", None),
            port,
        )
        state = _wait_for_browser_debug_ready_or_exit(proc, port)
        attempt = LaunchAttempt(binary=candidate, state=state)
        result.attempts.append(attempt)

        if state != "exited":
            result.launched = True
            return result

        attempt.returncode = getattr(proc, "returncode", None)
        attempt.stderr_tail = _read_stderr_tail(stderr_path)
        logger.warning(
            "browser debug launch: %s exited (code=%s) before port %d opened%s",
            candidate,
            attempt.returncode,
            port,
            f"; stderr tail: {attempt.stderr_tail}" if attempt.stderr_tail else "",
        )

    return result


def try_launch_chrome_debug(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> bool:
    return launch_chrome_debug(port, system).launched
