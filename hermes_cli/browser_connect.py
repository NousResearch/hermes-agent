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
# case-insensitively by prefix so version suffixes (e.g. ``ChromeHTML.X``)
# still resolve to STABLE. Pre-release channels have their own ProgIds and
# MUST be matched first (see _WINDOWS_CHANNEL_PROGIDS) so they are never
# swallowed into the stable family — driving the wrong profile is a
# wrong-principal bug (#95549 invariant).
_WINDOWS_PROGID_MAP = (
    ("chromehtml", "chrome"),
    ("msedgehtm", "edge"),
    ("bravehtml", "brave"),
    ("chromiumhtm", "chromium"),
)

# Pre-release ProgId prefixes we recognize but do NOT support (their profiles
# live in channel-specific dirs the resolver tables don't carry). Matched
# BEFORE the stable map; a hit fails closed rather than resolving to stable.
# ``ChromeBHTML`` = Beta, ``ChromeDHTML`` = Dev, ``ChromeSSHTML`` = Canary
# (SxS); ``MSEdgeBHTML`` / ``MSEdgeDHTML`` / ``MSEdgeCHTML`` = Edge channels.
_WINDOWS_CHANNEL_PROGIDS = (
    "chromebhtml", "chromedhtml", "chromesshtml", "chromecanaryhtml",
    "msedgebhtml", "msedgedhtml", "msedgechtml",
    "bravebetahtml", "bravenightlyhtml",
)

# Linux xdg default-web-browser .desktop name fragments → canonical STABLE key.
# Includes the Flatpak application ids (``com.google.Chrome.desktop`` etc.),
# which share none of the native package name fragments. Anchored so a channel
# .desktop (``google-chrome-beta``, ``com.google.chrome.beta``) does NOT match
# the stable fragment — channels are caught by _LINUX_CHANNEL_FRAGMENTS first.
_LINUX_DESKTOP_MAP = (
    ("google-chrome", "chrome"),
    ("com.google.chrome", "chrome"),
    ("chromium", "chromium"),
    ("brave", "brave"),
    ("microsoft-edge", "edge"),
    ("com.microsoft.edge", "edge"),
    ("msedge", "edge"),
)

# Non-stable Linux channel .desktop fragments — recognized, unsupported.
# Checked before the stable map; a hit fails closed.
_LINUX_CHANNEL_FRAGMENTS = (
    "google-chrome-beta", "google-chrome-unstable", "google-chrome-canary",
    "com.google.chrome.beta", "com.google.chrome.dev", "com.google.chrome.canary",
    "microsoft-edge-beta", "microsoft-edge-dev", "microsoft-edge-canary",
    "brave-browser-beta", "brave-browser-nightly", "brave-browser-dev",
)

# Where sandboxed Linux packages keep the profile instead of $XDG_CONFIG_HOME.
_LINUX_FLATPAK_IDS = {
    "chrome": "com.google.Chrome",
    "chromium": "org.chromium.Chromium",
    "brave": "com.brave.Browser",
    "edge": "com.microsoft.Edge",
}
_LINUX_SNAP_PROFILE_PARTS = {
    "chromium": ("snap", "chromium", "common", "chromium"),
    "brave": ("snap", "brave", "current", ".config", "BraveSoftware", "Brave-Browser"),
}

# macOS LaunchServices bundle-id → canonical STABLE key. EXACT match (not
# prefix): ``com.google.chrome.beta`` must not be read as ``com.google.chrome``.
_DARWIN_BUNDLE_MAP = (
    ("com.google.chrome", "chrome"),
    ("com.microsoft.edgemac", "edge"),
    ("com.brave.browser", "brave"),
    ("org.chromium.chromium", "chromium"),
)

# Non-stable macOS channel bundle ids — recognized, unsupported. Checked first.
_DARWIN_CHANNEL_BUNDLES = (
    "com.google.chrome.beta", "com.google.chrome.dev", "com.google.chrome.canary",
    "com.microsoft.edgemac.beta", "com.microsoft.edgemac.dev", "com.microsoft.edgemac.canary",
    "com.brave.browser.beta", "com.brave.browser.nightly",
)

# Sentinel returned when the OS default is a recognized-but-unsupported
# Chromium CHANNEL (Beta/Dev/Canary). Distinct from None (non-Chromium) so the
# caller fails closed with a channel-specific message instead of driving the
# stable profile of a different account.
UNSUPPORTED_CHANNEL = "__unsupported_channel__"


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

    Returns None for unknown browsers. On Linux the native ($XDG_CONFIG_HOME),
    snap and Flatpak locations are tried and the first existing one wins; the
    native path is returned when none exists so the caller's error names it.
    Darwin/Windows paths are not stat'ed. Paths are built with the TARGET
    system's separator (posix for Darwin/Linux, backslash for Windows) so an
    explicit ``system`` argument resolves correctly regardless of the host OS.
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
    candidates = [posixpath.join(config, *linux_name.split("/"))]
    snap_parts = _LINUX_SNAP_PROFILE_PARTS.get(browser)
    if snap_parts:
        candidates.append(posixpath.join(home, *snap_parts))
    flatpak_id = _LINUX_FLATPAK_IDS.get(browser)
    if flatpak_id:
        candidates.append(
            posixpath.join(home, ".var", "app", flatpak_id, "config", *linux_name.split("/"))
        )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


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
    # Channels first: a recognized Beta/Dev/Canary ProgId must fail closed, not
    # fall through to a stable prefix match and drive the stable profile.
    for chan in _WINDOWS_CHANNEL_PROGIDS:
        if low.startswith(chan):
            return UNSUPPORTED_CHANNEL
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
        # Strip the nested LSHandlerPreferredVersions block first: on macOS 26
        # it carries a VERSION NUMBER (e.g. LSHandlerRoleAll = "7559.97";), not
        # the "-" placeholder older releases used. Left in, the role regex below
        # would match that version before the real bundle id sitting at the
        # entry's own level and return "7559.97" — which maps to no browser, so
        # detection fails on a machine whose default IS Chrome (PR #95620 review).
        low = re.sub(r"lshandlerpreferredversions\s*=\s*\{[^}]*\}\s*;", "", low)
        # The real bundle id is the first non-"-" role value at this level.
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
    b = bundle.lower()
    # Channels first (exact): a Beta/Dev/Canary bundle must fail closed.
    if b in _DARWIN_CHANNEL_BUNDLES:
        return UNSUPPORTED_CHANNEL
    for frag, browser in _DARWIN_BUNDLE_MAP:
        if b == frag:
            return browser
    # A non-Chromium https handler (Safari, Firefox, Arc, …) or an unknown
    # channel bundle: fail closed. No "first installed Chromium wins" fallback
    # — that would drive a browser the user never made their default.
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
    # Channels first: ``google-chrome-beta.desktop`` contains the stable
    # ``google-chrome`` fragment, so a substring match would drive stable.
    # Catch recognized channels and fail closed instead.
    for frag in _LINUX_CHANNEL_FRAGMENTS:
        if frag in out:
            return UNSUPPORTED_CHANNEL
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


# ---------------------------------------------------------------------------
# Real-profile SNAPSHOT launch
#
# The consent path (``browser.use_real_profile``) never drives the live
# default user-data-dir. Chromium ≥136 (Google-branded builds) refuses
# remote debugging on the default dir no matter who launches it, and the
# live dir is usually held by the user's running browser (SingletonLock).
# Instead we snapshot the real profile into ``~/.hermes/browser-profile/``
# — a non-default dir Chrome will happily debug, that never contends with
# the user's browser — launch the user's real binary on the copy with a
# devtools port, and hand the CDP URL to whichever browser lane is active
# (Browser Use CLI or the built-in tools). We launch the browser ourselves
# precisely so NO mock-keychain/basic-store switches are added: cookies
# encrypted with the OS keyring (gnome-keyring / kwallet / macOS Keychain)
# decrypt exactly like they do in the user's own browser.
# ---------------------------------------------------------------------------

# Directory names excluded from the profile snapshot: caches/telemetry AND the
# heavy, replay-prone state that hangs a fresh Chromium's renderer (extensions
# and their service workers spin up on launch and wedge JS eval; IndexedDB /
# GPUCache add hundreds of MB for nothing). We keep ONLY auth/login state
# (cookies, Login Data, Web Data, Preferences, Local State) — the point of the
# feature — which turns a multi-hundred-MB profile into a few MB.
_SNAPSHOT_IGNORES = (
    "*Cache*",          # Cache, Code Cache, GPUCache, GrShaderCache, ShaderCache, GraphiteDawnCache, component_crx_cache, ...
    "Extensions",       # wallets/etc.: 100s of MB, and hang the renderer headless
    "Extension*",       # Extension State, Extension Rules, Extension Scripts
    "Local Extension Settings",
    "Service Worker",   # replays on launch → wedges the renderer
    "IndexedDB",
    "Crash Reports",
    "Crashpad",
    "BrowserMetrics*",
    "Snapshots",
    "OptimizationGuide*",
    "optimization_guide_model_store",
    "Safe Browsing",
    "SafetyTips",
    "OnDeviceHeadSuggestModel",
    "segmentation_platform",
    "Sync Data",
    "Shared Dictionary",
    "History*",         # large; not needed for auth
    "Favicons*",
    "Singleton*",       # live-instance symlinks; never valid in a copy
    "RunningChromeVersion",
    "SingletonSocket",
    "*.tmp",
    "*-journal",         # SQLite rollback journals — sidecars of the auth DBs,
    "*-wal",             # which are copied via online-backup; a stale sidecar
    "*-shm",             # next to a backed-up DB corrupts it.
    "BrowserMetrics-spare.pma",
)

# Small, auth-bearing files re-synced from the live profile on EVERY consented
# launch (the full tree is only copied when the snapshot doesn't exist yet).
# Paths here are RELATIVE TO A PROFILE DIR (Default, "Profile 6", …) — the
# caller resolves which source profile is active and mirrors these into the
# copy's ``Default`` so the launched Chromium (which opens ``Default``) lands
# on the user's real signed-in session. No ``-journal``/``-wal`` sidecars: the
# SQLite DBs are copied via the online-backup API (see _copy_auth_file), which
# produces a self-contained DB with committed state folded in — copying a
# stale raw journal on top of that would corrupt it.
_AUTH_REFRESH_PROFILE_FILES = (
    "Cookies",
    "Network/Cookies",
    "Login Data",
    "Login Data For Account",
    "Web Data",
    "Preferences",
)

def real_profile_copy_dir(browser: str) -> str:
    """Return the hermes-owned snapshot dir for ``browser``'s real profile."""
    return str(get_hermes_home() / "browser-profile" / browser)


def _last_used_profile(src: str) -> str:
    """Return the profile dir Chrome last used (``Local State`` → profile.last_used).

    Chromium opens ``Default`` inside a user-data-dir unless told otherwise, but
    the user's signed-in session usually lives in whichever profile they
    actually browse (``Profile 6`` etc.). We read that here and mirror its auth
    into the copy's ``Default`` so the launched browser is signed in. Falls back
    to ``Default`` when Local State is missing/unreadable or names a profile
    dir that doesn't exist.
    """
    import json

    try:
        with open(os.path.join(src, "Local State"), encoding="utf-8", errors="replace") as fh:
            state = json.load(fh)
        last = ((state.get("profile") or {}).get("last_used")) or "Default"
    except (OSError, ValueError, AttributeError):
        last = "Default"
    if not isinstance(last, str) or not os.path.isdir(os.path.join(src, last)):
        return "Default"
    return last


def _secure_snapshot_root(path: str) -> None:
    """Lock down a snapshot dir through Hermes' canonical secret-store policy.

    The snapshot holds copies of the user's Cookies / Login Data, so it is a
    credential store and must get the same owner-only permissions (and
    managed-mode / NixOS group-share carve-out, HERMES_UID/GID ownership) as
    every other Hermes secret dir — via ``hermes_cli.config._secure_dir``,
    not a bespoke chmod. Deferred import avoids a config↔browser import cycle.
    """
    try:
        from hermes_cli.config import _secure_dir

        _secure_dir(path)
    except Exception as e:  # never block a launch on a permissions best-effort
        logger.debug("could not secure real-profile snapshot dir %s: %s", path, e)


# Auth files that are SQLite databases: on Windows a running Chrome holds these
# with an exclusive lock, so a raw file copy raises WinError 32 ("being used by
# another process") and a naive best-effort skip leaves the copy signed-out.
# These are copied via SQLite's online-backup API instead, which reads a
# consistent committed snapshot while the lock is held. Matched by basename.
_SQLITE_AUTH_DBS = frozenset({
    "Cookies", "Login Data", "Login Data For Account", "Web Data",
})


def _copy_auth_file(src_file: str, dst_file: str) -> bool:
    """Copy one auth file, lock-aware. Returns True on success.

    For SQLite DBs (Cookies/Login Data/…), use the online-backup API so the
    copy works even while the browser holds the file's write lock (Windows).
    Everything else is a plain copy. A DB whose backup fails falls through to a
    raw copy attempt; only if BOTH fail do we report failure to the caller.
    """
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    if os.path.basename(src_file) in _SQLITE_AUTH_DBS:
        try:
            import sqlite3

            # Read-only URI + immutable-free: we want a consistent committed
            # snapshot, not to fight the writer. Short busy timeout so a truly
            # wedged DB fails fast rather than hanging the launch.
            source = sqlite3.connect(f"file:{src_file}?mode=ro", uri=True, timeout=5)
            try:
                out = sqlite3.connect(dst_file)
                try:
                    with out:
                        source.backup(out)
                finally:
                    out.close()
            finally:
                source.close()
            return True
        except Exception as e:
            logger.debug("real-profile: sqlite-backup of %s failed (%s); trying raw copy",
                         src_file, e)
    # Non-DB file, or DB whose backup failed: raw copy.
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except OSError as e:
        logger.debug("real-profile: could not copy %s: %s", src_file, e)
        return False


def _mirror_profile_auth(src: str, dst: str, source_profile: str) -> int:
    """Copy ``source_profile``'s auth files into the copy's ``Default`` slot.

    agent-browser launches ``Default`` in the copied user-data-dir; mirroring
    the active source profile's cookies/logins/prefs there is what makes the
    session actually signed in (the LinkedIn/Gmail "logged out" bug when the
    real session lives in a non-Default profile). Lock-aware (Windows), so a
    running Chrome doesn't block the cookie DBs.

    Returns the number of DB auth files that could NOT be copied (0 = clean).
    """
    dst_default = os.path.join(dst, "Default")
    failed_dbs = 0
    for rel in _AUTH_REFRESH_PROFILE_FILES:
        s = os.path.join(src, source_profile, rel)
        if not os.path.isfile(s):
            continue
        ok = _copy_auth_file(s, os.path.join(dst_default, rel))
        if not ok and os.path.basename(rel) in _SQLITE_AUTH_DBS:
            failed_dbs += 1
    return failed_dbs


_SNAPSHOT_DONE_MARKER = ".hermes-snapshot-complete"


def snapshot_real_profile(browser: str, src: str | None = None) -> tuple[str | None, str | None]:
    """Snapshot ``browser``'s real ACTIVE profile into the hermes copy dir.

    Copies only what the launched browser needs: the user-data-dir's
    ``Local State`` plus the auth-bearing files of the profile the user
    actually browses (``Local State → profile.last_used``, e.g. ``Profile 6``),
    mirrored into the copy's ``Default`` — which is what agent-browser opens.
    We deliberately do NOT copy every profile dir: non-active profiles are
    unused here and would just be stale credential copies sitting on disk.

    A ``.hermes-snapshot-complete`` marker is written only after a copy fully
    succeeds; a torn/interrupted first copy (disk full, Ctrl+C) therefore never
    looks "already populated" on the next run — it is redone from scratch.

    Auth files are re-synced on every call so fresh logins from the user's own
    browsing show up. Locked-file copy errors are tolerated best-effort.

    Returns ``(copy_dir, None)`` on success, ``(None, error)`` on failure.
    """
    src = src or real_profile_data_dir(browser)
    if not src or not os.path.isdir(src):
        return None, (
            f"profile directory for '{browser}' was not found ({src!r}). "
            "Launch that browser at least once, or turn browser.use_real_profile off."
        )
    dst = real_profile_copy_dir(browser)
    source_profile = _last_used_profile(src)
    marker = os.path.join(dst, _SNAPSHOT_DONE_MARKER)
    # Only a copy that previously COMPLETED counts as populated. A half-written
    # tree (no marker) is treated as absent and rebuilt — otherwise a torn first
    # copy poisons freshness forever and only ever gets auth overlays.
    populated = os.path.isfile(marker)
    try:
        os.makedirs(dst, exist_ok=True)
        # Secure the snapshot dir AND its browser-profile parent on EVERY
        # launch: a failed first attempt or an older-build dir must still
        # converge to owner-only perms; the parent enumerates every browser we
        # hold cookies for.
        parent = os.path.dirname(dst)
        if parent:
            _secure_snapshot_root(parent)
        _secure_snapshot_root(dst)

        # Base user-data-dir file the browser reads at startup. Cheap; always
        # re-synced so last_used etc. stay current.
        ls_src = os.path.join(src, "Local State")
        if os.path.isfile(ls_src):
            try:
                shutil.copy2(ls_src, os.path.join(dst, "Local State"))
            except OSError as e:
                logger.debug("real-profile snapshot: skipped Local State: %s", e)

        if not populated:
            # Fresh (or torn-and-rebuilding): drop any partial Default and copy
            # the ACTIVE profile's full dir (minus caches AND the locked auth
            # DBs) into the copy's Default. The SQLite auth DBs are excluded
            # here because a raw copytree of a file a running Chrome holds open
            # raises on Windows; they are copied lock-aware by
            # _mirror_profile_auth below (sqlite online-backup).
            dst_default = os.path.join(dst, "Default")
            try:
                shutil.rmtree(dst_default, ignore_errors=True)
                shutil.copytree(
                    os.path.join(src, source_profile),
                    dst_default,
                    dirs_exist_ok=True,
                    symlinks=False,
                    ignore=shutil.ignore_patterns(*_SNAPSHOT_IGNORES, *_SQLITE_AUTH_DBS),
                    ignore_dangling_symlinks=True,
                )
            except shutil.Error as multi:
                # Per-file failures (browser mid-write) are non-fatal.
                logger.info(
                    "real-profile snapshot: %d file(s) skipped copying %s/%s",
                    len(multi.args[0]) if multi.args else 0, src, source_profile,
                )

        # Both paths: copy the active profile's auth DBs into Default,
        # lock-aware (sqlite online-backup) so a running Chrome on Windows
        # doesn't block them. This is also the per-launch fresh-login re-sync.
        failed_dbs = _mirror_profile_auth(src, dst, source_profile)
        if failed_dbs:
            # We could not read the user's cookie/login DBs at all — even the
            # online-backup fallback failed. Rather than launch a silently
            # signed-out session, fail closed with an actionable message.
            return None, (
                f"could not read the '{browser}' profile's login data "
                f"({failed_dbs} database(s) locked). Close {browser} and retry, "
                "or turn browser.use_real_profile off."
            )

        # Never carry live-instance leftovers into the copy.
        for leftover in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            try:
                os.unlink(os.path.join(dst, leftover))
            except OSError:
                pass
        # Mark complete only after everything above succeeded.
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(source_profile)
        except OSError as e:
            logger.debug("real-profile snapshot: could not write done marker: %s", e)
    except OSError as e:
        return None, f"could not snapshot the '{browser}' profile into {dst}: {e}"
    return dst, None


def cleanup_real_profile_snapshots() -> None:
    """Delete the whole real-profile snapshot store (all copied credentials).

    Called when consent is OFF: the copied Cookies / Login Data must not
    outlive the toggle. Best-effort and idempotent — missing dir is fine.
    """
    root = str(get_hermes_home() / "browser-profile")
    try:
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
            logger.info("real-profile: removed snapshot store %s (consent off)", root)
    except OSError as e:
        logger.debug("real-profile cleanup failed for %s: %s", root, e)


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
