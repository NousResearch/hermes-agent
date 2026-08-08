"""Shared helpers for attaching Hermes to a local Chromium-family CDP port."""

from __future__ import annotations

import logging
import os
import platform
import posixpath
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field

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


def _managed_install() -> bool:
    """True when a package manager (NixOS) owns this install's modes.

    Read once per creation so the *creation* mode honours the same carve-out
    ``hermes_cli.config._secure_dir`` applies to reconciliation. Import is
    local and failure means "not managed": an unimportable config module is
    the single-user source-install case, where 0700 is the right default.
    """
    try:
        from hermes_cli.config import is_managed

        return bool(is_managed())
    except Exception:  # pragma: no cover - defensive
        return False


def _ensure_chrome_debug_data_dir(data_dir: str) -> None:
    """Create the Chromium user-data-dir owner-only (0700), except managed.

    ``chrome-debug`` is a real Chromium profile: Cookies, Login Data, and
    Local Storage live here. Created with a bare ``os.makedirs`` it inherited
    the umask and landed 0755, so every other local account could list and
    read those stores. ``HERMES_HOME`` is 0700 by default, which contains the
    damage — but the documented ``HERMES_HOME_MODE=0701`` hatch (so nginx can
    traverse to a served subdirectory) makes a 0755 child genuinely
    world-readable. Defence in depth, at the one directory where the payoff
    for reading it is a logged-in session.

    The mode is passed to ``os.makedirs`` so it is set *at creation* — no
    window where the profile sits world-readable before a follow-up chmod.
    Policy is then reconciled by ``hermes_cli.config._secure_dir``, the house
    helper, rather than a hand-rolled chmod: it skips managed/NixOS installs,
    honours ``HERMES_HOME_MODE``, and applies ``HERMES_UID``/``HERMES_GID``
    ownership so a root-created dir does not lock out uid-mapped Docker
    workers (#34107).

    **Managed installs are skipped at creation too, not just at
    reconciliation.** ``_secure_dir`` returns early under ``is_managed()``
    because the NixOS module deliberately shares state group-wise, and
    ``chrome-debug`` is *not* one of the directories its ``systemd.tmpfiles``
    rules pre-create — it is made lazily at runtime, so a hardcoded 0700 here
    would be the only thing setting its mode and would silently override that
    design. The module pins ``stateDir/.hermes`` to ``2770`` (setgid,
    group-rwx), runs the gateway with ``UMask = "0007"`` precisely so "files
    created by the gateway should be group-writable so interactive users in
    the hermes group can read/write them", and avoids ``chown -R`` to keep
    the setgid bit alive. On such a host the gateway and a hostUsers CLI share
    one ``$HERMES_HOME`` through the hermes group, so a 0700 profile created
    by whichever ran first locks the other out of the browser entirely.
    Omitting the explicit mode there lets the inherited setgid + umask land
    2770, matching ``ensure_hermes_home``'s managed branch and the
    ``logs/curator`` lazy-mkdir precedent in ``hermes_cli.config``.

    Reconciling unconditionally also heals a profile an older Hermes left at
    0755, which is the whole point — the exposure is on disk already. It is
    safe against a *running* browser: only group/other bits are dropped, the
    owner keeps ``rwx``, and POSIX checks the mode at ``open()`` rather than
    on already-open descriptors, so an attached Chromium keeps reading and
    writing its profile. On Windows POSIX mode bits are advisory (``chmod``
    only toggles the read-only flag), so this is best-effort there and the
    directory keeps its inherited ACLs.
    """
    if _managed_install():
        # Managed mode: let the NixOS-configured umask/setgid decide, and
        # skip _secure_dir (which no-ops here anyway).
        os.makedirs(data_dir, exist_ok=True)
        return
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    try:
        from hermes_cli.config import _secure_dir

        _secure_dir(data_dir)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("browser debug launch: profile dir chmod skipped: %s", exc)


def _open_launch_stderr_log(path: str):
    """Open the launch stderr log owner-only (0600), truncating as before.

    Opened with a plain ``open(path, "wb")`` this landed 0644 under a default
    umask — a fixed, guessable name inside the profile dir hardened just
    above. Under ``HERMES_HOME_MODE=0701`` the directory is traversable but
    unlistable, so a predictable filename is exactly the case that stays
    reachable; the uuid-named files elsewhere in the profile do not.

    Two halves, covering different installs:

    * **At creation** the mode is passed to ``os.open``, so the file is never
      briefly group/other-readable. ``O_TRUNC`` keeps the existing
      per-candidate overwrite semantics.
    * **On an already-existing log** ``O_CREAT`` applies ``mode`` only to a
      file it actually creates, so the inode keeps whatever it had. An older
      Hermes left this file at 0644 on disk, so creation-time hardening alone
      would leave every *upgrading* install exposed at the one path in this
      change with a guessable name — the exposure this is meant to close is
      already on disk. ``hermes_cli.config._secure_file`` reconciles it, for
      the same reason ``_ensure_chrome_debug_data_dir`` delegates to
      ``_secure_dir``: that helper is the single owner of the owner-only file
      policy. It skips managed/NixOS installs and containers, where broader
      modes are deliberate, and it is where Windows ACL enforcement lands
      (#77527) — so delegating inherits that instead of needing a second
      implementation here. A hand-rolled ``os.chmod(path, 0o600)`` would
      fight all three.

    Ordering matters: the reconcile runs *after* the truncating open and
    *before* any bytes are written, so the tighten lands while the file is
    empty and no fresh Chromium stderr ever sits in a widely-readable file.
    It is safe against a *running* browser for the same reason the directory
    tighten is — only group/other bits drop, the owner keeps ``rw``, and POSIX
    checks the mode at ``open()`` rather than on already-open descriptors.

    **The managed/NixOS carve-out applies at creation here too**, not only at
    reconciliation — the same defect the profile directory had. This log is
    created lazily at runtime and is not covered by the module's
    ``systemd.tmpfiles`` rules, so on a managed host a hardcoded 0600 would be
    the only thing setting its mode. There the gateway and an interactive
    ``hostUsers`` CLI share one ``$HERMES_HOME`` at two uids through the hermes
    group; a 0600 log created by whichever ran first makes the other's
    truncating open fail with ``EACCES`` — and because every candidate binary
    reuses this one path, that fails *the whole launch*, not merely the
    diagnostic. Omitting the explicit mode lets the service's
    ``UMask = "0007"`` land 0660, which is what the merge base produced.
    """
    managed = _managed_install()
    # 0o666 rather than 0o600 on managed installs so the configured umask
    # decides, matching the merge base's plain open() and the carve-out
    # _ensure_chrome_debug_data_dir applies to the directory.
    create_mode = 0o666 if managed else 0o600
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    try:
        fd = os.open(path, flags, create_mode)
    except OSError:
        # Same fallback the pre-fix code had: let a genuinely unopenable path
        # surface to launch_chrome_debug's per-candidate handler.
        handle = open(path, "wb")
    else:
        handle = os.fdopen(fd, "wb")
    if not managed:
        try:
            from hermes_cli.config import _secure_file

            _secure_file(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("browser debug launch: stderr log chmod skipped: %s", exc)
    return handle


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
    _ensure_chrome_debug_data_dir(data_dir)
    stderr_path = os.path.join(data_dir, _LAUNCH_STDERR_LOG)

    for candidate in candidates:
        try:
            with _open_launch_stderr_log(stderr_path) as stderr_file:
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
