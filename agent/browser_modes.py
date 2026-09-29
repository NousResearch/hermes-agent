"""Which browser the agent works in, and whether it is signed in there.

Three user-facing modes sit on top of mechanisms the engine already has:

* ``managed``  - the default headless browser, no logins.
* ``own``      - a dedicated visible Chromium profile (``chrome-debug``) the agent attaches to over
  CDP. The user signs in to Google once in that window; the session persists on disk.
* ``copy``     - ``browser.use_real_profile``: a Hermes-owned snapshot of the default browser's
  profile, so the agent starts already signed in (cookie import). Chrome >= 136 refuses remote
  debugging on the default profile, which is why this is a copy rather than the live profile.

Login checks read cookie *names and hosts only*, from a temp copy of the cookie database. Values
never leave SQLite and are never returned, logged or stored.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from typing import Any, Optional

MODES = ("managed", "own", "copy")

# Cookies Google sets only for a signed-in account (any one is enough).
_GOOGLE_AUTH_COOKIES = ("SID", "__Secure-1PSID", "__Secure-3PSID", "SAPISID", "__Secure-1PAPISID")
_SIGN_IN_URL = "https://accounts.google.com/"


def _hb():
    from hermes_cli import browser_connect
    return browser_connect


def _browser_cfg() -> dict[str, Any]:
    from hermes_cli.config import read_raw_config
    cfg = read_raw_config().get("browser")
    return cfg if isinstance(cfg, dict) else {}


def _update_browser_cfg(**values: Any) -> None:
    from hermes_cli.config import read_raw_config, save_config
    raw = read_raw_config()
    browser = raw.get("browser") if isinstance(raw.get("browser"), dict) else {}
    browser.update(values)
    raw["browser"] = browser
    save_config(raw)


def _own_cdp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def current_mode() -> str:
    cfg = _browser_cfg()
    if cfg.get("use_real_profile"):
        return "copy"
    if str(cfg.get("cdp_url") or "").strip():
        return "own"
    return "managed"


def _cookie_paths(profile_dir: str) -> list[str]:
    return [p for p in (os.path.join(profile_dir, "Network", "Cookies"), os.path.join(profile_dir, "Cookies"))
            if os.path.isfile(p)]


def google_signed_in(profile_dir: str) -> Optional[bool]:
    """True/False whether a Google auth cookie exists under ``profile_dir`` (a Chromium profile
    folder such as ``.../Default``); None when there is no cookie database to look at."""
    paths = _cookie_paths(profile_dir)
    if not paths:
        return None
    with tempfile.TemporaryDirectory(prefix="czesiek-cookies-") as tmp:
        dst = os.path.join(tmp, "Cookies")
        try:
            shutil.copyfile(paths[0], dst)
            con = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        except (OSError, sqlite3.Error):
            return None
        try:
            marks = ",".join("?" for _ in _GOOGLE_AUTH_COOKIES)
            row = con.execute(
                f"SELECT 1 FROM cookies WHERE name IN ({marks}) AND host_key LIKE '%google.com' LIMIT 1",
                _GOOGLE_AUTH_COOKIES).fetchone()
            return row is not None
        except sqlite3.Error:
            return None
        finally:
            con.close()


def _own_profile_dir() -> str:
    return os.path.join(_hb().chrome_debug_data_dir(), "Default")


def _default_browser() -> Optional[str]:
    try:
        return _hb().detect_default_chromium()
    except Exception:
        return None


def _copy_status(browser: Optional[str]) -> dict[str, Any]:
    if not browser:
        return {"available": False}
    hb = _hb()
    src = hb.real_profile_data_dir(browser)
    copy_dir = hb.real_profile_copy_dir(browser)
    marker = os.path.join(copy_dir, hb._SNAPSHOT_DONE_MARKER)
    has_copy = os.path.isfile(marker)
    return {
        "available": bool(src and os.path.isdir(src)),
        "browser": browser,
        "has_copy": has_copy,
        "copied_at": int(os.path.getmtime(marker)) if has_copy else None,
        "google_signed_in": google_signed_in(os.path.join(copy_dir, "Default")) if has_copy else None,
        "pinned_profile": _browser_cfg().get("real_profile_pin") or None,
    }


def status() -> dict[str, Any]:
    hb = _hb()
    cfg = _browser_cfg()
    port = hb.DEFAULT_BROWSER_CDP_PORT
    own_dir = _own_profile_dir()
    return {
        "mode": current_mode(),
        "own": {
            "has_profile": os.path.isdir(own_dir),
            "window_open": hb.is_browser_debug_ready(_own_cdp_url(port)),
            "google_signed_in": google_signed_in(own_dir),
            "cdp_url": str(cfg.get("cdp_url") or "") or None,
        },
        "copy": _copy_status(_default_browser()),
        # Chrome >= 136 refuses --remote-debugging-port on the default profile, so "the live
        # browser as-is" is not offered; the copy mode is the supported way to reuse logins.
        "live_profile_supported": False,
    }


def set_mode(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if mode == "own":
        _update_browser_cfg(use_real_profile=False, cdp_url=_own_cdp_url(_hb().DEFAULT_BROWSER_CDP_PORT))
    elif mode == "copy":
        _update_browser_cfg(use_real_profile=True, cdp_url="")
    else:
        _update_browser_cfg(use_real_profile=False, cdp_url="")
    return status()


def open_sign_in() -> dict[str, Any]:
    """Open the agent's own visible browser at the Google sign-in page and switch to ``own`` mode."""
    hb = _hb()
    port = hb.DEFAULT_BROWSER_CDP_PORT
    url = _own_cdp_url(port)
    if not hb.is_browser_debug_ready(url):
        launch = hb.launch_chrome_debug(port)
        if not launch.launched:
            raise RuntimeError(launch.hint or "no Chromium-family browser (Chrome, Edge, Brave) was found")
    set_mode("own")
    _open_tab(port, _SIGN_IN_URL)
    return status()


def _open_tab(port: int, url: str) -> None:
    import urllib.request
    from urllib.parse import quote
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{quote(url, safe=':/')}", method="PUT")
        urllib.request.urlopen(req, timeout=3).close()
    except OSError:
        pass  # the window is open; the user can navigate by hand


def import_now() -> dict[str, Any]:
    """Consent-gated cookie import: (re)copy the default browser's logins into the Hermes copy."""
    hb = _hb()
    browser = _default_browser()
    if not browser:
        raise RuntimeError("no default Chromium-family browser was detected")
    _update_browser_cfg(use_real_profile=True, cdp_url="")
    _, err = hb.snapshot_real_profile(browser)
    if err:
        raise RuntimeError(err)
    return status()


def clear_copy() -> dict[str, Any]:
    """Delete the copied logins and fall back to the managed browser if that mode was active."""
    _hb().cleanup_real_profile_snapshots()
    if current_mode() == "copy":
        _update_browser_cfg(use_real_profile=False)
    return status()


def clear_own_profile() -> dict[str, Any]:
    """Sign the agent's own browser out everywhere by deleting its profile directory."""
    hb = _hb()
    if hb.is_browser_debug_ready(_own_cdp_url(hb.DEFAULT_BROWSER_CDP_PORT)):
        raise RuntimeError("close the agent's browser window first")
    shutil.rmtree(hb.chrome_debug_data_dir(), ignore_errors=True)
    return status()
