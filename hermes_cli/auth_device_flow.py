"""Shared device-code / loopback-PKCE / browser / TLS helpers for interactive OAuth logins.

Split out of ``hermes_cli/auth.py`` and re-exported there; origin helpers are imported lazily
inside each function so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations

import base64
import hashlib
import html
import logging
import os
import ssl
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Optional
from urllib.parse import parse_qs, urlparse
from hermes_cli.auth_constants import (
    AuthError, DEFAULT_NOUS_PORTAL_URL, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    DEVICE_CODE_GRANT_TYPE, OAUTH_OVER_SSH_DOCS_URL, httpx)
from hermes_cli.auth_error_copy import DeviceCodeExpired
from utils import is_truthy_value

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")

# Console/text-mode browsers that ``webbrowser`` will launch INSIDE the terminal, hijacking the
# user's TTY with an unusable text browser. When the resolved browser is one of these we refuse
# to auto-open and fall back to the print-the-URL path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset({
    "w3m", "lynx", "links", "links2", "elinks", "www-browser",
    "browsh",  # TUI browser — still hijacks the terminal
})

# Browser-only remote IDEs / cloud shells (they don't set SSH_CLIENT / SSH_TTY). Keep this list
# narrow — well-known env vars set by the host platform — so a local shell never trips it.
_REMOTE_IDE_ENV_VARS = (
    "CLOUD_SHELL",  # GCP Cloud Shell
    "CODESPACES", "CODESPACE_NAME",  # GitHub Codespaces
    "GITPOD_WORKSPACE_ID",  # Gitpod
    "REPL_ID",  # Replit
    "STACKBLITZ",  # StackBlitz
)


def _is_remote_session() -> bool:
    """Detect environments where loopback OAuth can't reach the local browser.

    Historically only SSH was checked, but #26923 surfaced that **browser-only remote consoles** (GCP Cloud
    Shell, GitHub Codespaces, AWS EC2 Instance Connect, Gitpod, Replit, etc.) hit the exact same problem —
    the user has a browser on their laptop but the loopback listener is bound on the remote VM that the
    laptop's browser can't reach. These environments typically don't set ``SSH_CLIENT`` / ``SSH_TTY``, so
    the SSH-only check left them with no guidance and no fallback.
    """
    return bool(
        os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY")
        or any(os.getenv(var) for var in _REMOTE_IDE_ENV_VARS))


def _names_console_browser(value: str) -> bool:
    token = value.strip().split()[0] if value.strip() else ""
    return os.path.basename(token).lower() in _CONSOLE_BROWSER_NAMES


def _can_open_graphical_browser() -> bool:
    """Return True only when a *graphical* browser is likely to open.

    On a headless Linux box ``webbrowser.open()`` often resolves to a text-mode browser that takes
    over the terminal. Heuristics: a ``$BROWSER`` naming a console browser refuses; on Linux a
    display server (``$DISPLAY`` / ``$WAYLAND_DISPLAY``) is required unless ``$BROWSER`` is set
    (a console one already returned False, so a set ``$BROWSER`` here is graphical).
    """
    browser_env = os.environ.get("BROWSER", "")
    if browser_env and _names_console_browser(browser_env):
        return False
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display and not browser_env:
            return False
    try:
        controller = webbrowser.get()
    except Exception:
        return False  # No browser resolvable at all → definitely don't auto-open.
    candidate = getattr(controller, "name", "") or getattr(controller, "basename", "") or ""
    return not (candidate and _names_console_browser(candidate))


def _ssh_user_at_host() -> str:
    """Best-effort 'user@hostname' for the SSH tunnel hint; placeholders keep it valid syntax."""
    try:
        import socket as _socket
        hostname = _socket.gethostname() or "<this-host>"
    except OSError:
        hostname = "<this-host>"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "<user>"
    return f"{user}@{hostname}"


def _pkce_code_verifier(length: int = 64) -> str:
    return base64.urlsafe_b64encode(os.urandom(length)).decode("ascii").rstrip("=")[:128]


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


_FAVICON_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAARGVYSWZNTQAqAAAACAABh2kABAAAAAEAAAAaAAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAABAoAMABAAAAAEAAABAAAAAAEZRQrAAAAHLaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U+CiAgICAgICAgIDxleGlmOlBpeGVsWERpbWVuc2lvbj40MDA8L2V4aWY6UGl4ZWxYRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFlEaW1lbnNpb24+NDAwPC9leGlmOlBpeGVsWURpbWVuc2lvbj4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+ClLygeQAABA9SURBVGgFtdoFjF1FFwBgSou7OxR3d4eiwd3dPYTgEEKQ4u7u7u7u7g7F3d3p/23P5uzsfe/e3RT+m/R17syZ4zZzt89555232WabDR48eJj/9OnTp08F3/+DBOb7VcgM9WuFY+z+9NNPv/32G4QjjDDCqKOOOuywwyby/1CYfytA8v3jjz9+9tlnb7311tNPP/3ss8++++6733777e+//47pkUceefzxx5922mnnnnvuBRdccLbZZjNj/j8RY+gFCNax+OWXX+IeN//8888EE0ywyCKLTD311M8999w999zz/PPPh9bfe++9J5988pJLLunbt+8MM8ywwgorrLPOOvPMM8+/F6PPUMRAsP72228/9thjr776Kj/p16/f2GOPPeWUU9IxTQfTf/zxx913333YYYc9/PDDMVP+8iti7LHHHmwydGJgA/PD+BfKo78eH5Cexx9/fIMNNsBxyVCMRxpppGWWWeamm24KSL+//vrrgQceSMJWYDMjjjjirrvu+t1334HskXoFwJZzzz23twIEQ2+88cYmm2xCeW25KSc32mijTz/9NHb5paaGXYzw2muvAauw2PwKvrcCAKXIo48+etxxxy25bB7PPPPMojnY8nvmmWcKgLotU0wxxSOPPBLAzXznKuBeCQCOejhGHe3W+cyYE0888VNPPQVDPHvvvXcrMO8K48wyyywvvfQSyGSxeQCyBwGC6o033jjppJOWhOsUGXxbHX744ccbbzyD0UYbrX///oMGDQpUzLjYYouVqGwBBr/EJX4OPvjgcLxm1mM1BOgqLiVe40g1p5122qabbvrVV195jRklabjhhotxucWM+Wmmmebvv/8ec8wx8bTVVlsdeeSRSy+99JZbbvnnn3+iJ2SPOeaYUUYZxca0kmQA//TTTz/ddNONMcYY0lpUjxJ5w7jvaqutNsccc7RCYOjkk08+4ogjUEKe0GyNwOSTT05JtBVbsJV7Aay00kq4f/PNN8866yzpRaa38frrryfbvPPOC3iSSSb55JNP+BW+xxprrB9++AF+ziP/7rzzzgZSM9KoJNq6AbCOOtM2jaJ0zTXXKEkLLbQQpU444YRYH3300SlVTR1nnHG8zjrrrIGaXg14AgACiEVgMHjoEt9bbLEFO+A1Jl9//XWQpFUEwjlXWWWVWCItyI033thGM+Eqdb8AxECb9EwyhfO+++5D+/7770dDbE000UTcFOEvvvhCqfr888/XXntt6U/7oGDdddddApHfy57zzTffzz//rBJ7tfGvv/4yPv/889955x2mRpWrDBw4kOIZX0jIEKuuuiqr6i9oVFFXZzbffPMll1yyTvflfBsBSCz9MSjPWXbZZQ866CCEP/744+22246aPdx00UUXXXjhheecc05u8Msvv2y//fYrrrgim8w444ywzzTTTAsssAArffPNN9gSxDvssAN1zD777FYpCAAriWnF+6STTsK3CqM4GrCMLvCyyy4jG1oELtltM664kA1wcQOqNRaRDz744FFHHSUXUUyUTAo2uOKK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"
_PORTAL_ART_CACHE: Optional[bytes] = None
_FAVICON_CACHE: Optional[bytes] = None


def _load_portal_art_svg() -> Optional[bytes]:
    """Load the Nous Portal illustration SVG from local assets, caching in-memory."""
    global _PORTAL_ART_CACHE
    if _PORTAL_ART_CACHE is not None:
        return _PORTAL_ART_CACHE

    candidates = [
        Path(__file__).resolve().parent.parent / "assets" / "portal-art.svg",
        Path(__file__).resolve().parent / "assets" / "portal-art.svg",
        Path.home() / ".hermes" / "hermes-agent" / "assets" / "portal-art.svg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                _PORTAL_ART_CACHE = candidate.read_bytes()
                return _PORTAL_ART_CACHE
            except OSError:
                pass
    return None


def _load_favicon_png() -> Optional[bytes]:
    """Load the Nous Girl favicon PNG from local assets, caching in-memory."""
    global _FAVICON_CACHE
    if _FAVICON_CACHE is not None:
        return _FAVICON_CACHE

    candidates = [
        Path(__file__).resolve().parent.parent / "assets" / "favicon.png",
        Path(__file__).resolve().parent / "assets" / "favicon.png",
        Path.home() / ".hermes" / "hermes-agent" / "assets" / "favicon.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                _FAVICON_CACHE = candidate.read_bytes()
                return _FAVICON_CACHE
            except OSError:
                pass
    return None


def _render_loopback_callback_html(
    display_name: str,
    outcome: str,
    *,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
) -> bytes:
    is_success = outcome != "failed" and not error
    safe_name = html.escape(display_name)
    title = f"{safe_name} authorization {outcome}."
    status_text = f"{safe_name} authorization {outcome}."

    if is_success:
        status_label = "STATUS: 200 OK"
        dot_color = "var(--color-hermes-accent)"
        tag_text = "// AUTHENTICATION COMPLETE"
        headline_lead = (
            f"Successfully authenticated with <strong>{safe_name}</strong>. "
            "Your session access tokens and refresh grants have been securely registered to your local credential pool."
        )
        terminal_flow_html = f"""
        <div class="cli-terminal-block">
          <div class="cli-terminal-header">
            <div class="terminal-mac-dots" title="macOS Window Controls">
              <span class="mac-dot mac-dot-red"><svg viewBox="0 0 10 10"><path d="M2.5 2.5l5 5m0-5l-5 5" fill="none" stroke="rgba(0,0,0,0.65)" stroke-width="1.2" stroke-linecap="round"/></svg></span>
              <span class="mac-dot mac-dot-yellow"><svg viewBox="0 0 10 10"><path d="M2 5h6" fill="none" stroke="rgba(0,0,0,0.65)" stroke-width="1.2" stroke-linecap="round"/></svg></span>
              <span class="mac-dot mac-dot-green"><svg viewBox="0 0 10 10"><path d="M2.2 2.2H6.2L2.2 6.2ZM7.8 7.8H3.8L7.8 3.8Z" fill="rgba(0,0,0,0.65)"/></svg></span>
            </div>
            <span>hermes-cli &bull; auth receiver</span>
            <span>PORT :1455</span>
          </div>

          <div class="cli-terminal-body">
            <div class="cli-line-cmd">
              <span class="cli-prompt">%</span>
              <span>hermes auth add {safe_name.lower().replace(' ', '-')} --browser</span>
            </div>
            <div class="cli-line-log">
              Waiting for callback on http://127.0.0.1:1455/auth/callback...
            </div>
            <div class="cli-line-log">
              Exchanging authorization code for {safe_name} tokens...
            </div>
            <div class="cli-line-success">
              <svg class="pixel-check-icon" viewBox="0 0 24 24">
                <path d="M10 18H8v-2h2v2Zm-2-2H6v-2h2v2Zm4-2v2h-2v-2h2Zm-6 0H4v-2h2v2Zm8 0h-2v-2h2v2Zm2-2h-2v-2h2v2Zm2-2h-2V8h2v2Zm2-2h-2V6h2v2Z"/>
              </svg>
              <span>Added credential to ~/.hermes/auth.json (credential_pool)</span>
            </div>
            <div class="cli-line-footer">
              You can close this tab and return to your terminal to continue your session. You can close this tab and return to your terminal.
            </div>
          </div>
        </div>"""
        btn_text = "Close This Tab"
    else:
        status_label = "STATUS: FAILED"
        dot_color = "#f87171"
        tag_text = "// AUTHENTICATION FAILED"
        err_msg = html.escape(error_description or error or "Authorization was denied or expired.")
        headline_lead = (
            f"Failed to authenticate with <strong>{safe_name}</strong>.<br>"
            f'<div style="margin-top:14px;padding:12px 16px;background:rgba(239,68,68,0.15);'
            f'border:1px solid rgba(239,68,68,0.3);border-radius:4px;font-size:13px;color:#fca5a5;'
            f'word-break:break-word;font-family:var(--font-mono);">'
            f'ERROR: {err_msg}</div>'
        )
        terminal_flow_html = f"""
        <div class="cli-terminal-block" style="border-color:rgba(239,68,68,0.3);background:rgba(80,0,0,0.3);">
          <div class="cli-terminal-header">
            <div class="terminal-mac-dots" title="macOS Window Controls">
              <span class="mac-dot mac-dot-red"><svg viewBox="0 0 10 10"><path d="M2.5 2.5l5 5m0-5l-5 5" fill="none" stroke="rgba(0,0,0,0.65)" stroke-width="1.2" stroke-linecap="round"/></svg></span>
              <span class="mac-dot mac-dot-yellow"><svg viewBox="0 0 10 10"><path d="M2 5h6" fill="none" stroke="rgba(0,0,0,0.65)" stroke-width="1.2" stroke-linecap="round"/></svg></span>
              <span class="mac-dot mac-dot-green"><svg viewBox="0 0 10 10"><path d="M2.2 2.2H6.2L2.2 6.2ZM7.8 7.8H3.8L7.8 3.8Z" fill="rgba(0,0,0,0.65)"/></svg></span>
            </div>
            <span>hermes-cli &bull; auth receiver</span>
            <span style="color:#f87171;">FAILED</span>
          </div>

          <div class="cli-terminal-body">
            <div class="cli-line-cmd">
              <span class="cli-prompt">%</span>
              <span>hermes auth add {safe_name.lower().replace(' ', '-')} --browser</span>
            </div>
            <div class="cli-line-log">
              Waiting for callback on http://127.0.0.1:1455/auth/callback...
            </div>
            <div class="cli-line-log" style="color:#fca5a5;">
              [&#x2717;] Error: {err_msg}
            </div>
            <div class="cli-line-footer" style="color:#fca5a5;">
              Please return to your terminal and try signing in again.
            </div>
          </div>
        </div>"""
        btn_text = "Close This Tab"

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Hermes Agent &mdash; {title}</title>
  <link rel="icon" type="image/png" href="{_FAVICON_DATA_URI}">
  <link rel="shortcut icon" href="/favicon.png">
  <style>
    /* ==========================================================================
       NOUS RESEARCH // HERMES AGENT OFFICIAL SYSTEM
       Authentic Editorial & Architectural Identity
       ========================================================================== */
    :root {{
      --color-hermes: #0000f2;
      --color-hermes-dark: #0000b8;
      --color-hermes-terminal: rgba(0, 0, 70, 0.55);
      --color-hermes-fg: #f5f5f5;
      --color-hermes-muted: rgba(245, 245, 245, 0.75);
      --color-hermes-dim: rgba(245, 245, 245, 0.45);
      --color-hermes-paper: #f5f5f5;
      --color-hermes-ink: #000000;
      --color-hermes-accent: #edff45;
      --color-hermes-green: #34d399;
      --font-display: "Times New Roman", "Sigurd", Georgia, serif;
      --font-mono: ui-monospace, "SF Mono", "Aeonik Fono", "JetBrains Mono", Menlo, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      background-color: var(--color-hermes);
      color: var(--color-hermes-fg);
      min-height: 100vh;
      font-family: var(--font-sans);
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
    }}

    body {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
    }}

    .hermes-shell {{
      width: 100%;
      max-width: 1280px;
      margin: 0 auto;
      padding: 0 36px;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }}

    @media (max-width: 768px) {{
      .hermes-shell {{
        padding: 0 20px;
      }}
    }}

    .hermes-nav {{
      display: grid;
      grid-template-columns: 1fr 1fr auto 1fr 1fr;
      align-items: center;
      padding-top: 36px;
      padding-bottom: 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
      font-family: var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .nav-item-left {{
      text-align: left;
    }}

    .nav-item-mid-left {{
      text-align: center;
    }}

    .nav-item-mid-right {{
      text-align: center;
    }}

    .nav-item-right {{
      text-align: right;
    }}

    .nav-item-left a,
    .nav-item-mid-left a,
    .nav-item-mid-right a,
    .nav-item-right a {{
      color: var(--color-hermes-fg);
      text-decoration: none;
      font-size: 16px;
      font-weight: 500;
      opacity: 0.85;
      transition: opacity 0.15s ease;
    }}

    .nav-item-left a:hover,
    .nav-item-mid-left a:hover,
    .nav-item-mid-right a:hover,
    .nav-item-right a:hover {{
      opacity: 1;
      text-decoration: underline;
    }}

    .nav-brand-group {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      padding: 0 24px;
    }}

    .nav-brand {{
      display: flex;
      flex-direction: column;
      align-items: center;
      text-decoration: none;
      color: var(--color-hermes-fg);
      line-height: 0.88;
      text-align: center;
      gap: 4px;
    }}

    .nav-brand-word-1, .nav-brand-word-2 {{
      font-family: var(--font-sans);
      font-weight: 800;
      font-size: 38px;
      letter-spacing: -0.02em;
      text-transform: uppercase;
      display: block;
    }}

    .nav-brand-socials {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .social-icon {{
      width: 18px;
      height: 18px;
      color: var(--color-hermes-fg);
      opacity: 0.85;
      transition: opacity 0.15s ease, transform 0.15s ease;
      display: block;
    }}

    .nav-brand-socials a:hover .social-icon {{
      opacity: 1;
      transform: translateY(-1px);
    }}

    .hero-stage {{
      flex: 1;
      display: grid;
      grid-template-columns: 1.18fr 0.82fr;
      gap: 56px;
      align-items: center;
      padding: 48px 0;
    }}

    @media (max-width: 960px) {{
      .hero-stage {{
        grid-template-columns: 1fr;
        gap: 36px;
        padding: 32px 0;
      }}
    }}

    .hero-content {{
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      max-width: 660px;
    }}

    .hero-meta-label {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--color-hermes-accent);
      margin-bottom: 18px;
    }}

    .hero-title {{
      font-family: var(--font-display);
      font-size: 52px;
      font-weight: 400;
      line-height: 1.05;
      letter-spacing: -0.02em;
      color: #ffffff;
      margin-bottom: 22px;
    }}

    @media (max-width: 640px) {{
      .hero-title {{
        font-size: 36px;
      }}
    }}

    .hero-description {{
      font-size: 16.5px;
      line-height: 1.65;
      color: var(--color-hermes-muted);
      margin-bottom: 32px;
    }}

    .hero-description strong {{
      color: #ffffff;
      font-weight: 600;
    }}

    .cli-terminal-block {{
      width: 100%;
      background: var(--color-hermes-terminal);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      padding: 16px 20px;
      font-family: var(--font-mono);
      font-size: 13px;
      line-height: 1.7;
      margin-bottom: 36px;
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
    }}

    .cli-terminal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 10px;
      margin-bottom: 14px;
      font-size: 11px;
      color: var(--color-hermes-dim);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}

    .terminal-mac-dots {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .mac-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.25);
    }}

    .cli-terminal-body {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .cli-line-cmd {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      color: #ffffff;
      font-weight: 600;
    }}

    .cli-prompt {{
      color: var(--color-hermes-accent);
      font-weight: 700;
      user-select: none;
    }}

    .cli-line-log {{
      color: var(--color-hermes-dim);
      font-size: 12px;
      padding-left: 18px;
    }}

    .cli-line-success {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      color: var(--color-hermes-green);
      font-weight: 500;
      padding-left: 18px;
    }}

    .pixel-check-icon {{
      width: 13px;
      height: 13px;
      fill: var(--color-hermes-green);
      flex-shrink: 0;
      position: relative;
      top: 2px;
    }}

    .cli-line-footer {{
      margin-top: 8px;
      padding-top: 10px;
      border-top: 1px dashed rgba(255, 255, 255, 0.1);
      color: var(--color-hermes-muted);
      font-size: 12px;
    }}

    .actions-box {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .hermes-btn-primary {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      height: 48px;
      padding: 0 32px;
      background: var(--color-hermes-paper);
      color: var(--color-hermes);
      font-family: var(--font-mono);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      border: none;
      border-radius: 2px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
      transition: all 0.15s ease-out;
      text-decoration: none;
    }}

    .hermes-btn-primary:hover {{
      background: #ffffff;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }}

    .hermes-btn-primary:active {{
      transform: translateY(0);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}

    .kbd-shortcut {{
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--color-hermes-dim);
      padding: 6px 10px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 2px;
    }}

    .hero-art-col {{
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }}

    .nous-girl-display {{
      max-height: 580px;
      width: 100%;
      height: auto;
      object-fit: contain;
      filter: drop-shadow(0 20px 40px rgba(0, 0, 0, 0.3));
    }}

    .hermes-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 24px 0 36px;
      border-top: 1px solid rgba(255, 255, 255, 0.12);
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--color-hermes-dim);
    }}

    .footer-nav-links {{
      display: flex;
      align-items: center;
      gap: 24px;
    }}

    .footer-nav-links a {{
      color: var(--color-hermes-muted);
      text-decoration: none;
      transition: color 0.15s;
    }}

    .footer-nav-links a:hover {{
      color: #ffffff;
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="hermes-shell">
    <header class="hermes-nav">
      <div class="nav-item-left">
        <a href="https://nousresearch.com" target="_blank" rel="noopener noreferrer">Nous</a>
      </div>

      <div class="nav-item-mid-left">
        <a href="https://hermes-agent.nousresearch.com/" target="_blank">Hermes</a>
      </div>

      <div class="nav-brand-group">
        <a href="https://hermes-agent.nousresearch.com/" target="_blank" class="nav-brand">
          <span class="nav-brand-word-1">Hermes</span>
          <span class="nav-brand-word-2">Agent</span>
        </a>
        <div class="nav-brand-socials">
          <a href="https://discord.gg/nousresearch" target="_blank" rel="noopener noreferrer" aria-label="Discord">
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" role="img" viewBox="-3.36 -3.36 30.72 30.72" class="social-icon">
              <path d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z"></path>
            </svg>
          </a>
          <a href="https://github.com/NousResearch/hermes-agent" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" role="img" viewBox="-3.36 -3.36 30.72 30.72" class="social-icon">
              <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"></path>
            </svg>
          </a>
          <a href="https://x.com/NousResearch" target="_blank" rel="noopener noreferrer" aria-label="X">
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" role="img" viewBox="-3.36 -3.36 30.72 30.72" class="social-icon">
              <path d="M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 11.09h-3.182z"></path>
            </svg>
          </a>
        </div>
      </div>

      <div class="nav-item-mid-right">
        <a href="https://hermes-agent.nousresearch.com/docs" target="_blank">Docs</a>
      </div>

      <div class="nav-item-right">
        <a href="https://portal.nousresearch.com/" target="_blank">Portal</a>
      </div>
    </header>

    <main class="hero-stage">
      <section class="hero-content">
        <div class="hero-meta-label">{tag_text}</div>
        
        <h1 class="hero-title">{status_text}</h1>
        
        <p class="hero-description">
          {headline_lead}
        </p>

        {terminal_flow_html}

        <div class="actions-box">
          <button class="hermes-btn-primary" onclick="window.close()">
            {btn_text}
          </button>
          <span class="kbd-shortcut">&#x2318;W / Ctrl+W</span>
        </div>
      </section>

      <section class="hero-art-col">
        <img class="nous-girl-display" src="/portal-art.svg" onerror="this.onerror=null;this.src='https://web-assets.nousresearch.com/nousnet-web/img/landing/portal-art.4fc19cfb0cdaa444.svg'" alt="Nous Girl Illustration">
      </section>
    </main>

    <footer class="hermes-footer">
      <div>&copy; 2026, Nous Research, Inc. &bull; Open Source under MIT License</div>
      <div class="footer-nav-links">
        <a href="https://hermes-agent.nousresearch.com/docs" target="_blank">Docs</a>
        <a href="https://portal.nousresearch.com/" target="_blank">Nous Portal</a>
        <a href="https://github.com/NousResearch/hermes-agent" target="_blank">GitHub</a>
      </div>
    </footer>
  </div>

  <script>
    window.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') {{
        window.close();
      }}
    }});
  </script>
</body>
</html>"""
    return page.encode("utf-8")


def _make_loopback_callback_handler(
    expected_path: str, *, display_name: str,
) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    """Handler class for an RFC 8252 loopback redirect plus the dict it fills in.

    Only a GET on *expected_path* is accepted (anything else is a 404 and leaves the result
    untouched), so a nonce embedded in the path acts as the CSRF ``state`` for authorization
    servers that do not echo an explicit ``state`` parameter.
    """
    result: dict[str, Any] = {"code": None, "state": None, "error": None, "error_description": None}

    class _LoopbackCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/favicon.png", "/favicon.ico"):
                fav_data = _load_favicon_png()
                if fav_data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(fav_data)
                    return
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Asset not found.")
                return

            if parsed.path in ("/portal-art.svg", "/assets/portal-art.svg"):
                svg_data = _load_portal_art_svg()
                if svg_data:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(svg_data)
                    return
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Asset not found.")
                return

            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return

            params = parse_qs(parsed.query)
            for key in result:
                result[key] = params.get(key, [None])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            outcome = "failed" if result["error"] else "received"
            self.wfile.write(
                _render_loopback_callback_html(
                    display_name,
                    outcome,
                    error=result.get("error"),
                    error_description=result.get("error_description"),
                )
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _LoopbackCallbackHandler, result


def _bind_loopback_callback_server(
    host: str, port: int, handler_cls: type[BaseHTTPRequestHandler], *, err: Callable[..., AuthError],
    bind_failed_code: str,
) -> HTTPServer:
    """Bind the loopback listener up front (``port=0`` = OS-assigned) so the redirect URI sent to
    the authorization server names a port we already own — no probe-close-rebind race."""

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        return _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise err(f"Could not bind callback server on {host}:{port}: {exc}", bind_failed_code) from exc


def _serve_loopback_callback(
    server: HTTPServer, result: dict[str, Any], *, timeout_seconds: float, err: Callable[..., AuthError],
    timeout_code: str,
) -> dict[str, Any]:
    """Serve *server* until the redirect lands in *result* or the deadline passes; always closes."""
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise err("Authorization timed out waiting for the local callback.", timeout_code)


def _print_loopback_ssh_hint(redirect_uri: str, *, docs_url: str | None = None) -> None:
    """Print an SSH tunnel hint when a loopback-redirect OAuth flow runs on a remote host.

    The auth server redirects the browser to ``127.0.0.1:<port>/callback``; when the browser is
    on another machine (the SSH case) the redirect needs a local port forward to reach us.
    """
    from hermes_cli.auth import _is_remote_session
    if not _is_remote_session():
        return
    try:
        parsed = urlparse(redirect_uri)
    except Exception:
        return
    host, port = parsed.hostname or "", parsed.port
    if host not in {"127.0.0.1", "::1", "localhost"} or not port:
        return
    divider = "-" * 60
    print(
        f"\n{divider}\nRemote session detected — SSH tunnel required\n{divider}\n"
        f"Hermes is waiting for the OAuth callback on {redirect_uri}\n"
        "but your browser is on a different machine. Run this command\n"
        "in a NEW terminal on your local machine BEFORE opening the URL:\n\n"
        f"  ssh -N -L {port}:127.0.0.1:{port} {_ssh_user_at_host()}\n\n"
        "Then open the authorize URL above in your local browser.")
    if docs_url:
        print(f"Provider docs:      {docs_url}")
    print(f"SSH/jump-box guide: {OAUTH_OVER_SSH_DOCS_URL}\n{divider}\n")


def _default_verify() -> bool | ssl.SSLContext:
    """Platform-aware default SSL verify for httpx clients.

    On macOS with Homebrew Python the system OpenSSL cannot find the system trust store, so pin
    certifi's bundle when importable; elsewhere defer to httpx's built-in default.
    """
    if sys.platform == "darwin":
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
    return True


def _resolve_verify(
    *, insecure: Optional[bool] = None, ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None) -> bool | ssl.SSLContext:
    from hermes_cli.auth import _default_verify
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}
    effective_insecure = (
        is_truthy_value(insecure, default=False) if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False))
    effective_ca = (
        ca_bundle or tls_state.get("ca_bundle") or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE"))
    if effective_insecure:
        return False
    if effective_ca:
        ca_path = str(effective_ca)
        if not os.path.isfile(ca_path):
            logger.warning(
                "CA bundle path does not exist: %s — falling back to default certificates",
                ca_path)
            return _default_verify()
        return ssl.create_default_context(cafile=ca_path)
    return _default_verify()


def _request_device_code(
    client: httpx.Client, portal_base_url: str, client_id: str, scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={"client_id": client_id, **({"scope": scope} if scope else {})})
    response.raise_for_status()
    data = response.json()
    required_fields = [
        "device_code", "user_code", "verification_uri", "verification_uri_complete", "expires_in",
        "interval"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _nous_device_auth_timeout_message(portal_base_url: str) -> str:
    """Actionable timeout text: the usual cause is Portal sign-in failing in the browser tab.

    A bare "Timed out waiting for device authorization" gives the user nothing to act on. The most common
    cause is Portal sign-in failing in the opened browser tab (including the server-side CAPTCHA loop from
    20605), so point at the Portal login page and the retry command. See #20605.
    """
    portal = (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    return (
        "Timed out waiting for device authorization.\n"
        "  Portal sign-in is required before the device code can be approved.\n"
        "  If the browser showed a CAPTCHA / 'You did not pass CAPTCHA' error,\n"
        "  finish signing in at the Portal in a normal browser tab, then retry:\n"
        "    hermes portal\n"
        f"  Portal login: {portal}/login")


def _print_device_code_instructions(
    verification_url: str, user_code: str, *, open_browser: bool, failure_dash: str = "--",
    swallow_open_errors: bool = False) -> None:
    """Print the shared "To continue" device-code block and optionally open the browser.

    Callers decide *whether* to open (remote-session / graphical-browser gating differs per
    provider); *failure_dash* keeps each provider's historical hint wording.
    """
    print()
    print("To continue:")
    print(f"  1. Open: {verification_url}")
    print(f"  2. If prompted, enter code: {user_code}")
    if not open_browser:
        return
    try:
        opened = webbrowser.open(verification_url)
    except Exception:
        if not swallow_open_errors:
            raise
        opened = False
    if opened:
        print("  (Opened browser for verification)")
    else:
        print(f"  Could not open browser automatically {failure_dash} use the URL above.")


def _poll_device_token_generic(
    post: Callable[[], "httpx.Response"], *, expires_in: int, poll_interval: int,
    validate_success: Callable[[Dict[str, Any]], None],
    on_non_json_error: Callable[["httpx.Response"], Exception],
    on_error: Callable[["httpx.Response", Dict[str, Any]], Exception],
    on_timeout: Callable[[], Exception]) -> Dict[str, Any]:
    """RFC 8628 device-code polling loop shared by the Nous and xAI flows.

    ``authorization_pending`` sleeps and retries; ``slow_down`` grows the interval by 1s (cap 30s).
    A non-JSON 408/429/5xx, or a 403 carrying ``x-vercel-mitigated`` (edge/WAF mitigation, never a
    real OAuth error), backs off — honoring ``Retry-After``, capped at 60s and at the device-code
    deadline — instead of aborting a login the user may still be approving. Every other error, a
    non-JSON error body, and the deadline become provider-specific exceptions via the supplied
    factories so each caller keeps its exact error contract.
    """
    deadline = time.monotonic() + max(1, expires_in)
    current_interval = poll_interval
    edge_backoff = 0.0  # kept apart from current_interval so slow_down/pending pacing is untouched
    unavailable = 0  # HTTP status of the latest edge/service failure; 0 once the endpoint answers again
    while time.monotonic() < deadline:
        response = post()
        if response.status_code == 200:
            payload = response.json()
            validate_success(payload)
            return payload
        try:
            error_payload = response.json()
        except Exception:
            status = response.status_code
            # Edge/WAF mitigation: back off and keep polling until the device code expires.
            if status in {408, 429} or status >= 500 or (
                    status == 403 and response.headers.get("x-vercel-mitigated")):
                from agent.retry_utils import parse_retry_after_seconds
                unavailable = status
                retry_after = parse_retry_after_seconds(response.headers)
                if retry_after is not None:
                    edge_backoff = min(max(current_interval, retry_after), 60)
                else:
                    edge_backoff = min(max(edge_backoff * 2, current_interval * 2, 5), 60)
                time.sleep(max(0.0, min(edge_backoff, deadline - time.monotonic())))
                continue
            response.raise_for_status()
            raise on_non_json_error(response)
        edge_backoff, unavailable = 0.0, 0
        error_code = str(error_payload.get("error") or "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue
        raise on_error(response, error_payload)
    # Still failing when the code ran out: a service outage (``__cause__``), not a sign-in left unapproved.
    raise on_timeout() from (ConnectionError(f"token endpoint answered HTTP {unavailable}") if unavailable else None)


def _poll_for_token(
    client: httpx.Client, portal_base_url: str, client_id: str, device_code: str,
    expires_in: int, poll_interval: int) -> Dict[str, Any]:
    """Poll the Nous token endpoint until the user approves or the code expires."""
    def _validate(payload: Dict[str, Any]) -> None:
        if "access_token" not in payload:
            raise ValueError("Token response did not include access_token")

    def _error(_response, error_payload) -> Exception:
        # Plain copy per OAuth error code; the raw ``code: description`` stays on a Details line.
        from hermes_cli.auth_error_copy import device_flow_error
        return device_flow_error(
            str(error_payload.get("error", "") or ""),
            str(error_payload.get("error_description") or "Unknown authentication error"))

    return _poll_device_token_generic(
        lambda: client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE, "client_id": client_id,
                "device_code": device_code}),
        expires_in=expires_in,
        poll_interval=max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS)),
        validate_success=_validate, on_error=_error,
        on_non_json_error=lambda _r: RuntimeError(
            "Token endpoint returned a non-JSON error response"),
        # Enriched at the SOURCE so the CLI login and the dashboard/desktop poller
        # (web_server_oauth._nous_promotion_poller surfaces it to the UI) both inherit the guidance.
        on_timeout=lambda: DeviceCodeExpired(_nous_device_auth_timeout_message(portal_base_url)))


def _prompt_yes_no(prompt: str, *, default: str) -> bool:
    """``input()`` a [Y/n]-style question; EOF/Ctrl-C count as *default*."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = default
    return answer in {"", "y", "yes"} if default == "y" else answer in {"y", "yes"}


def _print_login_success(
    provider_id: str, config_path: Path, *, show_auth_state: bool = False) -> None:
    print()
    print("Login successful!")
    if show_auth_state:
        from hermes_constants import display_hermes_home as _dhh
        print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider={provider_id})")


def _offer_existing_oauth_credentials(
    provider_id: str, *, resolve: Callable[[], Dict[str, Any]],
    is_expiring: Callable[[str, int], bool], display_name: str, default_base_url: str,
    expired_notice: Optional[str] = None) -> bool:
    """Offer to reuse still-valid stored OAuth credentials. Returns True when the user accepted.

    *resolve* attempts a refresh, so a resolved token should be valid — but double-check the
    expiry before telling the user "Login successful!".
    """
    from hermes_cli.auth import _update_config_for_provider
    try:
        existing = resolve()
        api_key = existing.get("api_key", "")
        if isinstance(api_key, str) and api_key and not is_expiring(api_key, 60):
            print(f"Existing {display_name} credentials found in Hermes auth store.")
            if _prompt_yes_no("Use existing credentials? [Y/n]: ", default="y"):
                config_path = _update_config_for_provider(
                    provider_id, existing.get("base_url", default_base_url))
                _print_login_success(provider_id, config_path)
                return True
        elif expired_notice:
            print(expired_notice)
    except AuthError:
        pass
    return False
