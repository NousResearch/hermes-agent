"""Guided Google connection for the desktop wizard.

The ``google-workspace`` skill already owns the OAuth flow (``setup.py``) and the API calls
(``google_api.py``). This module is the thin, testable layer the desktop needs on top: a status
that costs no network, storing the client file the person uploads, the two-step consent, and a
real "does it work" check that reads something back (the next events, unread mail) so a green light
means the connection actually answers — not merely that a token file exists.

Every script runs as a subprocess of the backend's own interpreter with ``HERMES_HOME`` pinned to the
active profile, so tokens land where the skill expects them and a hung call cannot wedge the backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_bundled_skills_dir, get_hermes_home

TOKEN_FILE = "google_token.json"
CLIENT_FILE = "google_client_secret.json"

Runner = Callable[[str, list[str], int], tuple[int, str, str]]


def scripts_dir() -> Path:
    return get_bundled_skills_dir(Path(__file__).parent.parent / "skills") / "productivity" / "google-workspace" / "scripts"


def _run(script: str, args: list[str], timeout: int) -> tuple[int, str, str]:
    env = {**os.environ, "HERMES_HOME": str(get_hermes_home()), "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, str(scripts_dir() / script), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"{script} timed out after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


def status() -> dict[str, Any]:
    """What is in place, from files only (no network): the client file, the token, and its scopes."""
    home = get_hermes_home()
    token = home / TOKEN_FILE
    scopes: list[str] = []
    if token.is_file():
        try:
            raw = json.loads(token.read_text(encoding="utf-8")).get("scopes") or []
            scopes = [str(s) for s in raw]
        except (OSError, ValueError):
            pass
    short = sorted({s.rsplit("/", 1)[-1] for s in scopes})
    return {
        "client_secret": (home / CLIENT_FILE).is_file(),
        "token": token.is_file(),
        "connected": token.is_file(),
        "services": short,
        "can_send_mail": any(s.endswith("gmail.send") for s in scopes),
    }


def save_client_secret(content: str) -> dict[str, Any]:
    """Validate an uploaded OAuth client JSON and store it for the skill. Never echoes its contents."""
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ValueError("This is not a JSON file.") from exc
    if not isinstance(data, dict) or not ({"installed", "web"} & data.keys()):
        raise ValueError("This is not a Google OAuth client file (it has no 'installed' section). "
                         "Create an OAuth client of type 'Desktop app' and download its JSON.")
    body = data.get("installed") or data.get("web") or {}
    if not body.get("client_id") or not body.get("client_secret"):
        raise ValueError("The client file is missing client_id or client_secret.")
    path = get_hermes_home() / CLIENT_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return {"ok": True}


def _with_deps(args: list[str], timeout: int, run: Runner) -> tuple[int, str, str]:
    """Run a setup command; if the Google libraries are not installed yet, install them once and retry."""
    code, out, err = run("setup.py", args, timeout)
    if code != 0 and ("ModuleNotFoundError" in err or "No module named" in err or "MISSING_DEPS" in out):
        run("setup.py", ["--install-deps"], 240)
        code, out, err = run("setup.py", args, timeout)
    return code, out, err


def auth_url(run: Runner = _run) -> str:
    code, out, err = _with_deps(["--auth-url"], 120, run)
    for line in out.splitlines():
        if line.strip().startswith("https://"):
            return line.strip()
    raise RuntimeError((out + err).strip()[-400:] or "Could not start Google sign-in.")


def exchange_code(code_or_url: str, run: Runner = _run) -> dict[str, Any]:
    code, out, err = _with_deps(["--auth-code", code_or_url.strip()], 120, run)
    if code != 0 or "OK: Authenticated" not in out:
        raise RuntimeError((out + err).strip()[-400:] or "Google did not accept that code.")
    return {"ok": True, "warning": "Some permissions were not granted." if "WARNING" in out else None}


def _json_list(out: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(out)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("items") or data.get("messages") or data.get("events") or []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def verify(run: Runner = _run, now: Optional[datetime] = None) -> dict[str, Any]:
    """A live round trip: authenticate, then read the next events and the newest unread mail."""
    code, out, err = run("setup.py", ["--check-live"], 60)
    if code != 0 or "LIVE_CHECK_OK" not in out:
        return {"ok": False, "reason": (out + err).strip()[-300:] or "Google did not answer."}
    now = now or datetime.now(timezone.utc)
    ecode, eout, _ = run("google_api.py", ["calendar", "list", "--start", now.isoformat(),
                                           "--end", (now + timedelta(days=2)).isoformat(), "--max", "3"], 60)
    mcode, mout, _ = run("google_api.py", ["gmail", "search", "is:unread", "--max", "3"], 60)
    events = [{"summary": e.get("summary", ""), "start": e.get("start", "")} for e in _json_list(eout)] if ecode == 0 else []
    mail = [{"from": m.get("from", ""), "subject": m.get("subject", "")} for m in _json_list(mout)] if mcode == 0 else []
    return {"ok": True, "events": events, "unread": mail,
            "calendar_ok": ecode == 0, "gmail_ok": mcode == 0}


def revoke(run: Runner = _run) -> dict[str, Any]:
    code, out, err = run("setup.py", ["--revoke"], 60)
    return {"ok": code == 0, "detail": (out + err).strip()[-200:]}


def today_snapshot(run: Runner = _run, now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """Today's calendar and the newest unread mail for the daily briefing; None when Google is not
    connected, and each half degrades to empty on its own — a briefing never fails over Google."""
    if not (get_hermes_home() / TOKEN_FILE).is_file():
        return None
    now = now or datetime.now(timezone.utc)
    ecode, eout, _ = run("google_api.py", ["calendar", "list", "--start", now.isoformat(),
                                           "--end", (now + timedelta(hours=36)).isoformat(), "--max", "8"], 25)
    mcode, mout, _ = run("google_api.py", ["gmail", "search", "is:unread newer_than:2d", "--max", "5"], 25)
    events = [{"summary": e.get("summary", ""), "start": e.get("start", "")} for e in _json_list(eout)] if ecode == 0 else []
    unread = [{"from": m.get("from", ""), "subject": m.get("subject", "")} for m in _json_list(mout)] if mcode == 0 else []
    if ecode != 0 and mcode != 0:
        return None
    return {"events": events, "unread": unread}
