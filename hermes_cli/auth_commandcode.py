"""Command Code OAuth (command-code CLI token file and browser OAuth) credentials and status.

Follows the OpenCodex / Command Code CLI specification:
- Local credentials path: ~/.commandcode/auth.json
- Browser OAuth flow: https://commandcode.ai/studio/auth/cli?callback=...&state=...
- Loopback callback server on port 5959 (POST /callback)
- Identity verification: https://api.commandcode.ai/alpha/whoami
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from hermes_cli.auth_constants import (
    AuthError, DEFAULT_COMMANDCODE_BASE_URL, COMMANDCODE_STUDIO_URL,
    COMMANDCODE_WHOAMI_URL, COMMANDCODE_CALLBACK_PORT, _commandcode_err, httpx,
)

logger = logging.getLogger("hermes_cli.auth")

_RERUN = "Re-run 'hermes auth add commandcode --type oauth'."


def _commandcode_cli_auth_path() -> Path:
    return Path.home() / ".commandcode" / "auth.json"


def _read_commandcode_cli_tokens() -> Dict[str, Any]:
    auth_path = _commandcode_cli_auth_path()
    if not auth_path.exists():
        raise _commandcode_err(
            "Command Code CLI credentials not found. Run 'hermes auth add commandcode --type oauth' to sign in.",
            "commandcode_auth_missing",
        )
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _commandcode_err(
            f"Failed to read Command Code credentials from {auth_path}: {exc}",
            "commandcode_auth_read_failed",
        ) from exc
    if not isinstance(data, dict):
        raise _commandcode_err(
            f"Invalid Command Code credentials format in {auth_path}.",
            "commandcode_auth_invalid",
        )
    api_key = str(data.get("apiKey", "") or "").strip()
    if not api_key:
        raise _commandcode_err(
            f"Command Code credentials in {auth_path} missing apiKey.",
            "commandcode_auth_missing_key",
        )
    return data


def _save_commandcode_cli_tokens(tokens: Dict[str, Any]) -> Path:
    from hermes_cli.auth import _write_private_file_atomic
    auth_path = _commandcode_cli_auth_path()
    _write_private_file_atomic(auth_path, json.dumps(tokens, indent=2, sort_keys=True) + "\n")
    return auth_path


def validate_commandcode_api_key(
    api_key: str, *, timeout_seconds: float = 10.0
) -> Optional[Dict[str, Any]]:
    """Validate token with https://api.commandcode.ai/alpha/whoami."""
    key = (api_key or "").strip()
    if not key:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(COMMANDCODE_WHOAMI_URL)
        req.add_header("Authorization", f"Bearer {key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "hermes-cli")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        user = data.get("user") if isinstance(data, dict) else None
        if isinstance(user, dict):
            return {
                "userId": str(user.get("id") or ""),
                "userName": str(user.get("userName") or ""),
            }
        return {"userId": "", "userName": ""}
    except Exception as exc:
        logger.debug("validate_commandcode_api_key failed: %s", exc)
        return None


def _make_commandcode_callback_handler(expected_state: str) -> tuple[type[BaseHTTPRequestHandler], dict[str, Any]]:
    result: dict[str, Any] = {"payload": None, "error": None}

    class _CommandCodeCallbackHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self, origin: str = COMMANDCODE_STUDIO_URL) -> None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            origin = self.headers.get("Origin", COMMANDCODE_STUDIO_URL)
            self._send_cors_headers(origin)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success": false, "error": "Not found"}')
                return

            origin = self.headers.get("Origin", COMMANDCODE_STUDIO_URL)
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
            except Exception as exc:
                self.send_response(400)
                self._send_cors_headers(origin)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(exc)}).encode("utf-8"))
                return

            if not isinstance(payload, dict) or payload.get("state") != expected_state:
                self.send_response(400)
                self._send_cors_headers(origin)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success": false, "error": "State mismatch"}')
                result["error"] = "OAuth state mismatch"
                return

            api_key = str(payload.get("apiKey", "") or "").strip()
            if not api_key:
                self.send_response(400)
                self._send_cors_headers(origin)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"success": false, "error": "Missing apiKey"}')
                result["error"] = "Missing apiKey"
                return

            result["payload"] = payload
            self.send_response(200)
            self._send_cors_headers(origin)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return _CommandCodeCallbackHandler, result


def _commandcode_wait_for_callback(
    state: str, *, port: int = COMMANDCODE_CALLBACK_PORT, timeout_seconds: float = 120.0
) -> Dict[str, Any]:
    handler_cls, result = _make_commandcode_callback_handler(state)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    server: Optional[HTTPServer] = None
    used_port = port
    try:
        server = _ReuseHTTPServer(("127.0.0.1", used_port), handler_cls)
    except OSError:
        # Fallback to ephemeral port if 5959 is in use
        try:
            server = _ReuseHTTPServer(("127.0.0.1", 0), handler_cls)
            used_port = server.server_port
        except OSError as exc:
            raise _commandcode_err(
                f"Could not bind Command Code callback server: {exc}",
                "commandcode_callback_bind_failed",
            ) from exc

    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["payload"] or result["error"]:
                if result["error"]:
                    raise _commandcode_err(f"Command Code OAuth error: {result['error']}", "commandcode_oauth_error")
                return result["payload"]
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise _commandcode_err("Command Code authorization timed out waiting for the browser callback.", "commandcode_callback_timeout")


def _commandcode_oauth_login(args=None, *, force_browser: bool = False) -> Dict[str, Any]:
    """Execute Command Code OAuth login: auto-import ~/.commandcode/auth.json or run browser flow."""
    auth_path = _commandcode_cli_auth_path()

    # 1. Try local CLI credentials first unless forced
    if not force_browser and auth_path.exists():
        try:
            tokens = _read_commandcode_cli_tokens()
            api_key = tokens.get("apiKey", "").strip()
            if api_key:
                identity = validate_commandcode_api_key(api_key)
                user_id = identity.get("userId") if identity else tokens.get("userId", "")
                user_name = identity.get("userName") if identity else tokens.get("userName", "")
                print(f"Found existing Command Code CLI credentials for '{user_name or user_id or 'user'}' at {auth_path}")
                return {
                    "api_key": api_key,
                    "userId": user_id,
                    "userName": user_name,
                    "keyName": tokens.get("keyName", "local-cli"),
                    "source": "commandcode-cli",
                    "base_url": DEFAULT_COMMANDCODE_BASE_URL,
                }
        except Exception as exc:
            logger.debug("Local Command Code credential auto-import failed: %s", exc)

    # 2. Interactive browser OAuth flow
    state = secrets.token_urlsafe(32)
    port = COMMANDCODE_CALLBACK_PORT
    callback_url = f"http://127.0.0.1:{port}/callback"
    auth_url = f"{COMMANDCODE_STUDIO_URL}/studio/auth/cli?callback={callback_url}&state={state}"

    print("\nSign in with Command Code in your browser:")
    print(f"  {auth_url}\n")
    print("Waiting for authentication callback...")

    from hermes_cli.auth import _can_open_graphical_browser
    if _can_open_graphical_browser() and not getattr(args, "no_browser", False):
        try:
            import webbrowser
            webbrowser.open(auth_url)
        except Exception:
            pass

    timeout = getattr(args, "timeout", None) or 120.0
    payload = _commandcode_wait_for_callback(state, port=port, timeout_seconds=timeout)
    api_key = payload["apiKey"]
    user_id = payload.get("userId", "")
    user_name = payload.get("userName", "")
    key_name = payload.get("keyName", "hermes-oauth")

    # Save to ~/.commandcode/auth.json for cross-tool interoperability
    saved_tokens = {
        "apiKey": api_key,
        "userId": user_id,
        "userName": user_name,
        "keyName": key_name,
        "authenticatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    }
    try:
        _save_commandcode_cli_tokens(saved_tokens)
    except Exception as exc:
        logger.debug("Could not write back to %s: %s", auth_path, exc)

    return {
        "api_key": api_key,
        "userId": user_id,
        "userName": user_name,
        "keyName": key_name,
        "source": "commandcode-oauth",
        "base_url": DEFAULT_COMMANDCODE_BASE_URL,
    }


def resolve_commandcode_runtime_credentials() -> Dict[str, Any]:
    """Resolve runtime credentials from ~/.commandcode/auth.json or environment."""
    tokens = _read_commandcode_cli_tokens()
    api_key = tokens.get("apiKey", "").strip()
    if not api_key:
        raise _commandcode_err(f"Command Code access token missing. {_RERUN}", "commandcode_access_token_missing")

    return {
        "provider": "commandcode",
        "base_url": os.getenv("COMMANDCODE_BASE_URL", "").strip().rstrip("/") or DEFAULT_COMMANDCODE_BASE_URL,
        "api_key": api_key,
        "source": "commandcode-cli",
        "user_id": tokens.get("userId", ""),
        "user_name": tokens.get("userName", ""),
        "auth_file": str(_commandcode_cli_auth_path()),
    }


def get_commandcode_auth_status() -> Dict[str, Any]:
    auth_path = _commandcode_cli_auth_path()
    try:
        creds = resolve_commandcode_runtime_credentials()
        return {
            "logged_in": True,
            "auth_file": str(auth_path),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
            "user_name": creds.get("user_name"),
        }
    except AuthError as exc:
        return {"logged_in": False, "auth_file": str(auth_path), "error": str(exc)}
