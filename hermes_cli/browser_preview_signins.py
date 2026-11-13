"""Export the real-profile browser's cookies for the Desktop in-app browser (``hermes browser export-signins``).

Desktop's Electron main runs this as a LOCAL child of the owning profile and sets the cookies on the in-app
browser's partition itself. No cookie decryption happens here: the user's REAL browser binary, running on the
Hermes-owned snapshot (``real_profile_cdp``), decrypts its own store with the OS keychain, and the cookies are
read from it over CDP (``Storage.getCookies``). The list goes to stdout only (Electron's private pipe), never
to a file or a log. Consent (``browser.use_real_profile``) is re-checked here, for this profile.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

_CDP_TIMEOUT_S = 20.0
# Fields Electron's ``cookies.set`` can carry; the rest of CDP's cookie record (size, priority, source port)
# has no partition-side equivalent.
_COOKIE_FIELDS = ("name", "value", "domain", "path", "expires", "httpOnly", "secure", "session", "sameSite")


def _read_cookies(cdp_url: str) -> list[dict[str, Any]]:
    from websockets.sync.client import connect

    from hermes_agent.agent.proxy_bypass import loopback_connect_kwargs
    with connect(cdp_url, open_timeout=_CDP_TIMEOUT_S, close_timeout=2, max_size=None,
                 **loopback_connect_kwargs(cdp_url)) as ws:
        ws.send(json.dumps({"id": 1, "method": "Storage.getCookies", "params": {}}))
        while True:  # browser-level events may arrive before the reply
            msg = json.loads(ws.recv(timeout=_CDP_TIMEOUT_S))
            if msg.get("id") != 1:
                continue
            if "error" in msg:
                raise RuntimeError(f"Storage.getCookies failed: {msg['error'].get('message', msg['error'])}")
            return list(msg.get("result", {}).get("cookies") or [])


def _portable(cookie: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The fields the partition takes, or None for a cookie it cannot hold faithfully: a partitioned (CHIPS)
    cookie set unpartitioned would be sent in every top-level context, a wider scope than the site gave it."""
    if cookie.get("partitionKey") or not cookie.get("name") or not cookie.get("domain"):
        return None
    return {k: cookie[k] for k in _COOKIE_FIELDS if k in cookie}


def export_signins() -> dict[str, Any]:
    """``{"status": "disabled"}`` when consent is off (the caller clears what it imported), ``{"status":
    "error", "error": msg}`` when the real-profile browser can't serve, else ``{"status": "ok", "cookies":
    [...], "skipped": n}``."""
    from hermes_agent.tools.browser_session_real_profile import real_profile_cdp, terminate_real_profile_browser
    cdp_url, err = real_profile_cdp()
    if cdp_url is None:
        return {"status": "error", "error": err} if err else {"status": "disabled"}
    try:
        raw = _read_cookies(cdp_url)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"status": "error", "error": f"could not read cookies from the real-profile browser: {exc}"}
    finally:
        # Drops only this process's hold: a browser a running agent session also uses keeps running.
        terminate_real_profile_browser()
    cookies = [c for c in map(_portable, raw) if c is not None]
    return {"status": "ok", "cookies": cookies, "skipped": len(raw) - len(cookies)}


def run_export_signins(args) -> int:
    """CLI entry. Refuses an interactive stdout: the output is cookie values, meant for Desktop's pipe."""
    if not getattr(args, "json", False) or sys.stdout.isatty():
        print("✗ export-signins is used by Hermes Desktop; it prints cookie values and needs --json "
              "with stdout redirected to a pipe.", file=sys.stderr)
        return 2
    result = export_signins()
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0 if result["status"] != "error" else 1
