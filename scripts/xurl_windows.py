#!/usr/bin/env python3
"""Windows-safe subset of xurl for Hermes Agent.

This script exists because the official Go xurl binary can hang under some
native Windows agent shells. It intentionally implements the commands Hermes
needs for safe setup checks and posting, while never printing token values from
the local ~/.xurl store.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import http.server
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only on incomplete installs.
    yaml = None


DEFAULT_REDIRECT_URI = "http://localhost:8080/callback"
DEFAULT_AUTH_URL = "https://x.com/i/oauth2/authorize"
DEFAULT_TOKEN_URL = "https://api.x.com/2/oauth2/token"
DEFAULT_API_BASE_URL = "https://api.x.com"

OAUTH2_SCOPES = [
    "tweet.read",
    "tweet.write",
    "users.read",
    "offline.access",
]


class XurlError(RuntimeError):
    pass


def _reject_placeholder(value: str, label: str) -> None:
    normalized = value.strip().upper()
    placeholders = {
        "YOUR_CLIENT_ID",
        "YOUR_CLIENT_SECRET",
        "YOUR_X_HANDLE",
        "YOUR_USERNAME",
        "YOUR_HANDLE",
        "REPLACE_ME",
    }
    if normalized in placeholders or normalized.startswith("YOUR_"):
        raise XurlError(f"{label} still contains a placeholder: {value}")


def _home_dir() -> Path:
    raw = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    return Path(raw).expanduser() if raw else Path.home()


def _xurl_path() -> Path:
    return _home_dir() / ".xurl"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"apps": {}, "default_app": ""}
    if yaml is None:
        raise XurlError("PyYAML is required to read the xurl token store.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise XurlError(f"Invalid xurl token store: {path}")
    data.setdefault("apps", {})
    data.setdefault("default_app", "")
    if not isinstance(data["apps"], dict):
        raise XurlError(f"Invalid xurl apps map: {path}")
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise XurlError("PyYAML is required to write the xurl token store.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


class TokenStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _xurl_path()
        self.data = _load_yaml(self.path)

    @property
    def apps(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("apps", {})

    @property
    def default_app(self) -> str:
        return str(self.data.get("default_app") or "")

    @default_app.setter
    def default_app(self, value: str) -> None:
        self.data["default_app"] = value

    def save(self) -> None:
        _save_yaml(self.path, self.data)

    def list_apps(self) -> list[str]:
        return sorted(self.apps)

    def get_app(self, name: str) -> dict[str, Any] | None:
        return self.apps.get(name)

    def resolve_app_name(self, explicit: str | None = None) -> str:
        if explicit:
            if explicit not in self.apps:
                raise XurlError(f'app "{explicit}" not found')
            return explicit
        if self.default_app:
            if self.default_app not in self.apps:
                raise XurlError(f'default app "{self.default_app}" not found')
            return self.default_app
        if len(self.apps) == 1:
            return next(iter(self.apps))
        if self.apps:
            raise XurlError("No default app set. Run: xurl auth default APP_NAME")
        raise XurlError("No apps registered. Run: xurl auth apps add APP_NAME ...")

    def resolve_app(self, explicit: str | None = None) -> tuple[str, dict[str, Any]]:
        name = self.resolve_app_name(explicit)
        app = self.apps.get(name)
        if app is None:
            raise XurlError(f'app "{name}" not found')
        app.setdefault("oauth2_tokens", {})
        return name, app


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any = None,
    form: dict[str, str] | None = None,
    timeout: int = 30,
) -> Any:
    headers = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {"error": f"HTTP {exc.code}", "body": raw.decode("utf-8", "replace")}
        raise XurlError(json.dumps(payload, ensure_ascii=False))
    except urllib.error.URLError as exc:
        raise XurlError(str(exc.reason)) from exc

    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {"body": raw.decode("utf-8", "replace")}


def _api_url(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    base = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    return f"{base}/{endpoint.lstrip('/')}"


def _info_url() -> str:
    return os.environ.get("INFO_URL") or _api_url("/2/users/me")


def _token_url() -> str:
    return os.environ.get("TOKEN_URL", DEFAULT_TOKEN_URL)


def _auth_url() -> str:
    return os.environ.get("AUTH_URL", DEFAULT_AUTH_URL)


def _redirect_uri(app: dict[str, Any]) -> str:
    return os.environ.get("REDIRECT_URI") or app.get("redirect_uri") or DEFAULT_REDIRECT_URI


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    encoded_id = urllib.parse.quote(client_id, safe="")
    encoded_secret = urllib.parse.quote(client_secret, safe="")
    raw = f"{encoded_id}:{encoded_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _exchange_token(app: dict[str, Any], form: dict[str, str]) -> dict[str, Any]:
    client_id = str(app.get("client_id") or "")
    client_secret = str(app.get("client_secret") or "")
    headers = {"User-Agent": "hermes-xurl-windows/1"}
    if client_secret:
        headers["Authorization"] = _basic_auth_header(client_id, client_secret)
    form.setdefault("client_id", client_id)
    return _http_json("POST", _token_url(), headers=headers, form=form)


def _select_oauth2_token(
    app: dict[str, Any],
    username: str | None,
) -> tuple[str, dict[str, Any]]:
    tokens = app.setdefault("oauth2_tokens", {})
    if username:
        token = tokens.get(username)
        if token:
            return username, token
        raise XurlError(f'oauth2 user "{username}" not found in app')

    default_user = str(app.get("default_user") or "")
    if default_user and default_user in tokens:
        return default_user, tokens[default_user]

    for key in sorted(k for k in tokens if k):
        return key, tokens[key]
    if "" in tokens:
        return "", tokens[""]
    raise XurlError("No OAuth2 token found. Run: xurl auth oauth2 --app APP_NAME")


def _save_oauth2_token(
    store: TokenStore,
    app_name: str,
    app: dict[str, Any],
    username: str,
    token_payload: dict[str, Any],
    previous_refresh_token: str = "",
) -> None:
    access_token = token_payload.get("access_token")
    if not access_token:
        raise XurlError("Token response did not include access_token.")
    refresh_token = token_payload.get("refresh_token") or previous_refresh_token
    expires_in = int(token_payload.get("expires_in") or 7200)
    app.setdefault("oauth2_tokens", {})[username] = {
        "type": "oauth2",
        "oauth2": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiration_time": int(time.time()) + expires_in,
        },
    }
    store.apps[app_name] = app
    store.save()


def _fetch_username(access_token: str) -> str:
    payload = _http_json(
        "GET",
        _info_url(),
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "hermes-xurl-windows/1",
        },
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict) and data.get("username"):
        return str(data["username"])
    raise XurlError("UsernameNotFound")


def _oauth2_access_token(
    store: TokenStore,
    *,
    app_name: str | None = None,
    username: str | None = None,
) -> str:
    resolved_name, app = store.resolve_app(app_name)
    selected_user, record = _select_oauth2_token(app, username)
    oauth2 = record.get("oauth2") if isinstance(record, dict) else None
    if not isinstance(oauth2, dict):
        raise XurlError("Invalid OAuth2 token record.")

    access_token = str(oauth2.get("access_token") or "")
    if int(oauth2.get("expiration_time") or 0) > int(time.time()) + 60:
        return access_token

    refresh_token = str(oauth2.get("refresh_token") or "")
    if not refresh_token:
        raise XurlError("OAuth2 token expired and has no refresh token.")

    token_payload = _exchange_token(
        app,
        {"grant_type": "refresh_token", "refresh_token": refresh_token},
    )
    _save_oauth2_token(
        store,
        resolved_name,
        app,
        selected_user,
        token_payload,
        previous_refresh_token=refresh_token,
    )
    return str(token_payload["access_token"])


def _post_json(endpoint: str, body: dict[str, Any], access_token: str) -> Any:
    return _http_json(
        "POST",
        _api_url(endpoint),
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "hermes-xurl-windows/1",
        },
        body=body,
    )


def cmd_auth_status(_: list[str]) -> int:
    store = TokenStore()
    apps = store.list_apps()
    if not apps:
        print("No apps registered. Use 'xurl auth apps add' to register one.")
        return 0

    for index, name in enumerate(apps):
        app = store.get_app(name) or {}
        marker = ">" if name == store.default_app else " "
        client_id = str(app.get("client_id") or "")
        client_hint = f"client_id: {client_id[:8]}..." if client_id else "no credentials"
        print(f"{marker} {name}  [{client_hint}]")
        source = "REDIRECT_URI environment variable" if os.environ.get("REDIRECT_URI") else (
            "app config" if app.get("redirect_uri") else "built-in default"
        )
        print(f"      redirect_uri: {_redirect_uri(app)}  [{source}]")

        tokens = app.get("oauth2_tokens") or {}
        if tokens:
            default_user = str(app.get("default_user") or "")
            for username in sorted(tokens):
                user_marker = ">" if username and username == default_user else " "
                label = username or "(unnamed)"
                print(f"    {user_marker} oauth2: {label}")
        else:
            print("      oauth2: (none)")
        print(f"      oauth1: {'yes' if app.get('oauth1_token') else 'no'}")
        print(f"      bearer: {'yes' if app.get('bearer_token') else 'no'}")
        if index < len(apps) - 1:
            print()
    return 0


def cmd_auth_apps(args: list[str]) -> int:
    if not args or args[0] == "list":
        store = TokenStore()
        apps = store.list_apps()
        if not apps:
            print("No apps registered.")
            return 0
        for name in apps:
            marker = ">" if name == store.default_app else " "
            print(f"{marker} {name}")
        return 0

    if args[0] == "remove":
        if len(args) != 2:
            raise XurlError("Usage: xurl auth apps remove APP")
        store = TokenStore()
        app_name = args[1]
        if app_name not in store.apps:
            raise XurlError(f'app "{app_name}" not found')
        del store.apps[app_name]
        if store.default_app == app_name:
            store.default_app = next(iter(store.apps), "")
        store.save()
        print(f'App "{app_name}" removed.')
        return 0

    if args[0] != "add":
        raise XurlError("Usage: xurl auth apps [list|add|remove]")

    parser = argparse.ArgumentParser(prog="xurl auth apps add")
    parser.add_argument("name")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", default="")
    parser.add_argument("--prompt-client-secret", action="store_true")
    parser.add_argument("--redirect-uri", default="")
    parsed = parser.parse_args(args[1:])
    _reject_placeholder(parsed.client_id, "client id")
    client_secret = parsed.client_secret
    if parsed.prompt_client_secret and not client_secret:
        client_secret = getpass.getpass("Client secret: ")
    if client_secret:
        _reject_placeholder(client_secret, "client secret")

    store = TokenStore()
    if parsed.name in store.apps:
        raise XurlError(f'app "{parsed.name}" already exists')
    store.apps[parsed.name] = {
        "client_id": parsed.client_id,
        "client_secret": client_secret,
        "oauth2_tokens": {},
    }
    if parsed.redirect_uri:
        store.apps[parsed.name]["redirect_uri"] = parsed.redirect_uri
    if not store.default_app:
        store.default_app = parsed.name
    store.save()
    print(f'App "{parsed.name}" registered successfully.')
    return 0


def cmd_auth_default(args: list[str]) -> int:
    store = TokenStore()
    if not args:
        print(store.default_app or "(none)")
        return 0
    app_name = args[0]
    if app_name not in store.apps:
        raise XurlError(f'app "{app_name}" not found')
    store.default_app = app_name
    if len(args) > 1:
        username = args[1]
        _reject_placeholder(username, "username")
        tokens = store.apps[app_name].get("oauth2_tokens") or {}
        if username not in tokens:
            raise XurlError(f'user "{username}" not found in app "{app_name}"')
        store.apps[app_name]["default_user"] = username
    store.save()
    print(f'Default app set to "{app_name}".')
    return 0


class _IPv6HTTPServer(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _make_callback_handler(
    callback_path: str,
    expected_state: str,
    code_queue: "queue.Queue[str]",
    error_queue: "queue.Queue[str]",
):
    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return
            if params.get("state", [""])[0] != expected_state:
                error_queue.put("Invalid state parameter.")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid state parameter.")
                return
            code = params.get("code", [""])[0]
            if not code:
                error_queue.put("Missing authorization code.")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code.")
                return
            code_queue.put(code)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authentication complete. You can close this tab.")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return CallbackHandler


def _start_callback_servers(redirect_uri: str, state: str, code_queue, error_queue):
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 8080
    callback_path = parsed.path or "/callback"
    host = parsed.hostname or "localhost"
    handler = _make_callback_handler(callback_path, state, code_queue, error_queue)

    targets: list[tuple[type[http.server.ThreadingHTTPServer], str]] = []
    if host.lower() == "localhost":
        targets = [(http.server.ThreadingHTTPServer, "127.0.0.1"), (_IPv6HTTPServer, "::1")]
    else:
        targets = [(http.server.ThreadingHTTPServer, host)]

    servers = []
    for server_cls, bind_host in targets:
        try:
            server = server_cls((bind_host, port), handler)
        except OSError:
            continue
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
    if not servers:
        raise XurlError(f"Could not listen for OAuth callback on port {port}.")
    return servers


def cmd_auth_oauth2(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="xurl auth oauth2")
    parser.add_argument("username", nargs="?")
    parser.add_argument("--app", dest="app_name", default="")
    parser.add_argument("--no-browser", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.username:
        _reject_placeholder(parsed.username, "username")

    store = TokenStore()
    app_name, app = store.resolve_app(parsed.app_name)
    client_id = str(app.get("client_id") or "")
    if not client_id:
        raise XurlError(f'app "{app_name}" has no client id')

    redirect_uri = _redirect_uri(app)
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(OAUTH2_SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    authorize_url = f"{_auth_url()}?{urllib.parse.urlencode(params)}"

    code_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    error_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    servers = _start_callback_servers(redirect_uri, state, code_queue, error_queue)
    print("Open this URL to authorize X OAuth2:")
    print(authorize_url)
    if not parsed.no_browser:
        print("Opening browser for X OAuth2...")
        webbrowser.open(authorize_url)

    deadline = time.time() + 300
    try:
        while time.time() < deadline:
            try:
                err = error_queue.get_nowait()
                raise XurlError(err)
            except queue.Empty:
                pass
            try:
                code = code_queue.get(timeout=0.5)
                break
            except queue.Empty:
                continue
        else:
            raise XurlError("Timed out waiting for OAuth callback.")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()

    token_payload = _exchange_token(
        app,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    username = parsed.username or ""
    if not username:
        try:
            username = _fetch_username(str(token_payload["access_token"]))
        except XurlError:
            username = ""
            print(
                "Warning: authenticated, but username lookup failed. "
                "Re-run with: xurl auth oauth2 --app APP_NAME YOUR_USERNAME"
            )
    _save_oauth2_token(store, app_name, app, username, token_payload)
    print("OAuth2 authentication successful.")
    return 0


def cmd_auth(args: list[str]) -> int:
    if not args:
        raise XurlError("Usage: xurl auth [status|apps|default|oauth2]")
    subcmd, rest = args[0], args[1:]
    if subcmd == "status":
        return cmd_auth_status(rest)
    if subcmd == "apps":
        return cmd_auth_apps(rest)
    if subcmd == "default":
        return cmd_auth_default(rest)
    if subcmd == "oauth2":
        return cmd_auth_oauth2(rest)
    raise XurlError(f"Unsupported auth subcommand: {subcmd}")


def cmd_whoami(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="xurl whoami")
    parser.add_argument("--app", dest="app_name", default="")
    parser.add_argument("-u", "--username", default="")
    parsed = parser.parse_args(args)
    store = TokenStore()
    access_token = _oauth2_access_token(
        store,
        app_name=parsed.app_name or None,
        username=parsed.username or None,
    )
    payload = _http_json(
        "GET",
        _info_url(),
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "hermes-xurl-windows/1",
        },
    )
    _print_json(payload)
    return 0


def cmd_post(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="xurl post")
    parser.add_argument("text")
    parser.add_argument("--media-id", action="append", default=[])
    parser.add_argument("--app", dest="app_name", default="")
    parser.add_argument("-u", "--username", default="")
    parsed = parser.parse_args(args)

    body: dict[str, Any] = {"text": parsed.text}
    if parsed.media_id:
        body["media"] = {"media_ids": parsed.media_id}

    store = TokenStore()
    access_token = _oauth2_access_token(
        store,
        app_name=parsed.app_name or None,
        username=parsed.username or None,
    )
    _print_json(_post_json("/2/tweets", body, access_token))
    return 0


def cmd_reply(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="xurl reply")
    parser.add_argument("post_id")
    parser.add_argument("text")
    parser.add_argument("--media-id", action="append", default=[])
    parser.add_argument("--app", dest="app_name", default="")
    parser.add_argument("-u", "--username", default="")
    parsed = parser.parse_args(args)
    post_id = parsed.post_id.rstrip("/").split("/")[-1]
    body: dict[str, Any] = {
        "text": parsed.text,
        "reply": {"in_reply_to_tweet_id": post_id},
    }
    if parsed.media_id:
        body["media"] = {"media_ids": parsed.media_id}
    store = TokenStore()
    access_token = _oauth2_access_token(
        store,
        app_name=parsed.app_name or None,
        username=parsed.username or None,
    )
    _print_json(_post_json("/2/tweets", body, access_token))
    return 0


def cmd_quote(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="xurl quote")
    parser.add_argument("post_id")
    parser.add_argument("text")
    parser.add_argument("--app", dest="app_name", default="")
    parser.add_argument("-u", "--username", default="")
    parsed = parser.parse_args(args)
    post_id = parsed.post_id.rstrip("/").split("/")[-1]
    store = TokenStore()
    access_token = _oauth2_access_token(
        store,
        app_name=parsed.app_name or None,
        username=parsed.username or None,
    )
    _print_json(
        _post_json(
            "/2/tweets",
            {"text": parsed.text, "quote_tweet_id": post_id},
            access_token,
        )
    )
    return 0


def _help() -> str:
    return """xurl Windows shim for Hermes Agent

Supported commands:
  xurl auth status
  xurl auth apps list
  xurl auth apps add APP --client-id ID [--client-secret SECRET] [--prompt-client-secret] [--redirect-uri URI]
  xurl auth apps remove APP
  xurl auth default [APP [USER]]
  xurl auth oauth2 --app APP [USER] [--no-browser]
  xurl whoami [--app APP] [-u USER]
  xurl post "text" [--media-id ID] [--app APP] [-u USER]
  xurl reply POST_ID "text" [--media-id ID] [--app APP] [-u USER]
  xurl quote POST_ID "text" [--app APP] [-u USER]

This shim never prints token values from ~/.xurl.
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_help())
        return 0

    command, args = argv[0], argv[1:]
    try:
        if command == "auth":
            return cmd_auth(args)
        if command == "whoami":
            return cmd_whoami(args)
        if command == "post":
            return cmd_post(args)
        if command == "reply":
            return cmd_reply(args)
        if command == "quote":
            return cmd_quote(args)
        raise XurlError(f"Unsupported command on Windows shim: {command}")
    except XurlError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
