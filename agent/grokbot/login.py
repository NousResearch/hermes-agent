#!/usr/bin/env python3
"""Grok Bot (Cursor/sand) PKCE login — browser OAuth, no app modification.

Reverse-engineered from /Applications/Grok Bot.app (dist/electron-main/main.cjs).

Flow (matches the app's own LoginManager):
  1. Generate PKCE verifier (32 random bytes, base64url) + challenge (S256).
  2. Build https://cursor.com/loginDeepControl?challenge=..&uuid=..&mode=login
     &redirectTarget=sand&supportsSelectedTeamLogin=true
  3. User signs in in the browser.
  4. Poll {apiBaseUrl}/auth/poll?uuid=..&verifier=..  (404 = keep waiting)
  5. On success -> {"accessToken", "refreshToken"}  (plaintext, usable directly).

Writes (mode 0600, never printed):
  ~/.grokbot/session.json   {accessToken, refreshToken, authId, email, expiresAt}

Library used by hermes_cli.grokbot_oauth. Tokens stay in ~/.grokbot/session.json mode 0600.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api2.cursor.sh"
WEBSITE = "https://cursor.com"
CLIENT_TYPE = "sand"
CLIENT_VERSION = "0.24.0"
HOME = Path(os.environ.get("GROKBOT_HOME", Path.home() / ".grokbot"))


# ---------------------------------------------------------------- storage ----

def _save(obj: dict, name: str = "session.json") -> Path:
    HOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(HOME, 0o700)
    except OSError:
        pass
    p = HOME / name
    payload = json.dumps(obj, indent=2)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(p), flags, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


def _load() -> dict | None:
    p = HOME / "session.json"
    if not p.is_file():
        return None
    try:
        mode = p.stat().st_mode & 0o777
    except OSError:
        return None
    if mode & 0o077:
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# ------------------------------------------------------------- checksum -----

def _b2s(b: bytes) -> bytes:
    """Rolling XOR obfuscation lifted verbatim from main.cjs (function b2s)."""
    e = 165
    out = bytearray()
    for i, v in enumerate(b):
        v = (v ^ e) + i % 256
        e = v
        out.append(v & 0xFF)
    return bytes(out)


def checksum(machine_id: str) -> str:
    """x-cursor-checksum = obfuscated 6-byte coarse timestamp + machine id."""
    n = int(time.time() * 1000) // 1_000_000
    ts = bytes([n >> 40 & 255, n >> 32 & 255, n >> 24 & 255,
                n >> 16 & 255, n >> 8 & 255, n & 255])
    return base64.urlsafe_b64encode(_b2s(ts)).decode().rstrip("=") + machine_id


def machine_id() -> str:
    """Reuse the app's own machine id when present; else random (server tolerates it)."""
    store = Path.home() / "Library/Application Support/Grok Bot/sand-secrets.json"
    try:
        d = json.loads(store.read_text())
        mid = d.get("cursor-machine-id")
        if isinstance(mid, str) and mid:
            return mid
    except Exception:
        pass
    return "djEw" + secrets.token_urlsafe(16)


def headers(token: str | None = None) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "x-cursor-checksum": checksum(machine_id()),
        "x-cursor-client-type": CLIENT_TYPE,
        "x-cursor-client-version": CLIENT_VERSION,
        "x-ghost-mode": "true",
        "x-request-id": secrets.token_urlsafe(8),
        "connect-protocol-version": "1",
    }
    if token:
        h["authorization"] = f"Bearer {token}"
    return h


# ----------------------------------------------------------------- http -----

def _req(url: str, body: bytes | None = None, hdrs: dict | None = None,
         method: str = "POST", timeout: int = 30):
    r = urllib.request.Request(url, data=body, headers=hdrs or {}, method=method)
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# --------------------------------------------------------------- pkce -------

def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def start_login() -> dict:
    verifier = _b64u(secrets.token_bytes(32))
    challenge = _b64u(hashlib.sha256(verifier.encode()).digest())
    uuid = secrets.token_urlsafe(16)
    u = f"{WEBSITE}/loginDeepControl"
    qs = (f"?challenge={challenge}&uuid={uuid}&mode=login"
          f"&redirectTarget=sand&supportsSelectedTeamLogin=true")
    return {"uuid": uuid, "verifier": verifier,
            "challenge": challenge, "loginUrl": u + qs}


def poll(uuid: str, verifier: str, seconds: float = 300.0) -> dict | None:
    url = f"{API_BASE}/auth/poll?uuid={uuid}&verifier={verifier}"
    deadline = time.time() + seconds
    delay = 1.0
    while time.time() < deadline:
        code, body = _req(url, None, headers(), method="GET")
        # 404 == not yet completed -> keep waiting (per main.cjs c5i)
        if code == 404:
            time.sleep(min(delay, 10.0))
            delay *= 1.2
            continue
        if code == 200:
            try:
                j = json.loads(body)
            except Exception:
                return None
            if isinstance(j, dict) and "accessToken" in j:
                return j
        time.sleep(min(delay, 10.0))
        delay *= 1.2
    return None


def login_session(open_browser: bool = True, timeout: float = 300.0) -> dict:
    """PKCE login. Raises RuntimeError on timeout. Never prints tokens."""
    s = start_login()
    print("Open this URL and sign in:\n")
    print(f"  {s['loginUrl']}\n")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(s["loginUrl"], new=2)
        except Exception:
            pass
    print("Waiting for authorization", end="", flush=True)
    res = poll(s["uuid"], s["verifier"], seconds=timeout)
    print()
    if not res:
        raise RuntimeError("Grok Bot PKCE timed out or failed. Re-run and finish the browser sign-in.")
    out = {"accessToken": res["accessToken"],
           "refreshToken": res.get("refreshToken", ""),
           "authId": res.get("authId"), "email": res.get("email"),
           "expiresAt": res.get("expiresAt"), "obtainedAt": int(time.time())}
    p = _save(out)
    print(f"Saved -> {p} (mode 0600)")
    return out


def do_login(open_browser: bool = True) -> dict:
    try:
        return login_session(open_browser=open_browser)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------- refresh ------

def refresh_session() -> dict:
    """Rotate the access token. Raises RuntimeError; never prints secrets."""
    sess = _load()
    if not sess or not sess.get("refreshToken"):
        raise RuntimeError("No Grok Bot session. Run: python3 grokbot_login.py login")
    body = json.dumps({"client_id": "KbZUR41cY7W6zRSdpSUJ7I7mLYBKOCmB",
                       "grant_type": "refresh_token",
                       "refresh_token": sess["refreshToken"]}).encode()
    code, raw = _req(f"{API_BASE}/oauth/token", body, headers())
    if code != 200:
        raise RuntimeError(f"refresh HTTP {code}")
    j = json.loads(raw)
    if not j.get("access_token"):
        raise RuntimeError("refresh returned empty access_token")
    sess["accessToken"] = j["access_token"]
    if j.get("refresh_token"):
        sess["refreshToken"] = j["refresh_token"]
    sess["obtainedAt"] = int(time.time())
    _save(sess)
    return sess


def do_refresh() -> dict:
    try:
        sess = refresh_session()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print("Access token refreshed.")
    return sess


# --------------------------------------------------------------- calls ------

def _tok() -> str:
    s = _load()
    if not s:
        print("No session. Run: login", file=sys.stderr)
        sys.exit(1)
    return s["accessToken"]


def rpc(service_method: str, payload_obj: dict) -> tuple[int, bytes]:
    url = f"{API_BASE}/aiserver.v1.{service_method}"
    body = json.dumps(payload_obj).encode()
    return _req(url, body, headers(_tok()))


def do_models() -> None:
    code, raw = rpc("AiService/AvailableModels", {})
    print(f"HTTP {code}  ({len(raw)} bytes)")
    if code != 200:
        print(raw[:400])
        return
    try:
        j = json.loads(raw)
    except Exception:
        print(raw[:400])
        return
    models = j.get("models") or []
    if not models:
        print(json.dumps(j, indent=2)[:1200])
        return
    for m in models:
        name = m.get("name") or m.get("modelId") or m.get("id") or "?"
        extra = []
        if m.get("vendor"):
            extra.append(str(m["vendor"]))
        if m.get("maxRequestTokens"):
            extra.append(f"{m['maxRequestTokens']}tok")
        print(f"  {name}" + (f"  ({', '.join(extra)})" if extra else ""))


def do_chat(text: str, model: str = "grok-4.6") -> None:
    """Minimal StreamChat probe. Proto field numbers are best-effort."""
    body = {"model": model,
            "messages": [{"role": "user", "content": text}]}
    code, raw = rpc("AiService/StreamChat", body)
    print(f"HTTP {code} ({len(raw)} bytes)")
    print(raw[:800])


# ----------------------------------------------------------------- cli ------

def cmd_status() -> None:
    s = _load()
    if not s:
        print("No session at", HOME / "session.json")
        return
    import hashlib as _h
    red = lambda v: (_h.sha256(v.encode()).hexdigest()[:12] if v else "(none)")
    print(f"session file : {HOME / 'session.json'}")
    print(f"mode         : {oct((HOME / 'session.json').stat().st_mode & 0o777)}")
    print(f"email        : {s.get('email') or '(unknown)'}")
    print(f"authId       : {str(s.get('authId'))[:8]}..." if s.get("authId") else "authId       : (none)")
    print(f"accessToken  : len={len(s.get('accessToken') or '')} sha256:{red(s.get('accessToken',''))}")
    print(f"refreshToken : len={len(s.get('refreshToken') or '')} sha256:{red(s.get('refreshToken',''))}")
    print(f"obtainedAt   : {s.get('obtainedAt')}")
    code, raw = _req(f"{API_BASE}/aiserver.v1.AiService/GetUserInfo", b"{}",
                     headers(s.get("accessToken")))
    print(f"\nGetUserInfo  : HTTP {code}")
    if code == 200:
        print(raw[:500].decode("utf-8", "replace"))
    else:
        print(raw[:240].decode("utf-8", "replace"))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "login"
    if cmd == "login":
        do_login()
    elif cmd == "status":
        cmd_status()
    elif cmd == "refresh":
        do_refresh()
    elif cmd == "models":
        do_models()
    elif cmd == "chat":
        if len(sys.argv) < 3:
            print('usage: chat "message" [model]', file=sys.stderr)
            sys.exit(1)
        do_chat(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "grok-4.6")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
