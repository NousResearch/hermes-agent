"""Off-turn MCP browser callbacks. The SDK owns OAuth; the gateway owns identity.

Pending secrets live only in memory. Restart requires a new attempt. Tokens are
staged until token exchange succeeds and the attempt is still live.
"""
from __future__ import annotations

import asyncio
import copy
import re
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from tools.mcp_dashboard_oauth import DashboardOAuthFlow


def principal(source, home):
    return (str(Path(home).resolve()), source.profile, source.platform.value,
            source.scope_id, source.chat_id, source.thread_id, source.user_id)


def callback_candidate(text):
    # Deliberately broader than acceptance: malformed/orphan callbacks are also
    # secrets, and must never become a model turn. No URL fetching occurs here.
    from tools.mcp_oauth_redact import contains_oauth_parameters
    return isinstance(text, str) and (contains_oauth_parameters(text) or bool(re.search(
        r"(?i)https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?/[^\s]*callback", text)))


def callback_url(text):
    """Remove only lossless chat autolink envelopes; never extract a URL from prose."""
    if text.startswith("<") and text.endswith(">"):
        inner = text[1:-1]
        target, separator, label = inner.partition("|")
        if not separator or label == target:
            return target.replace("&amp;", "&")
    return text


def _query(url):
    if (not isinstance(url, str) or len(url) > 16384 or
            re.search(r"[\s\x00-\x1f\x7f]|%(?![0-9a-fA-F]{2})", url)):
        raise ValueError("Invalid OAuth URL")
    parsed = urlsplit(url)
    if parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Invalid OAuth URL")
    values = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True,
                      encoding="utf-8", errors="strict", max_num_fields=32)
    if any(len(v) != 1 for v in values.values()):
        raise ValueError("Invalid OAuth URL")
    for key in ("code", "state"):
        if key in values and (not values[key][0].isascii() or
                              re.search(r"[\x00-\x20\x7f]", values[key][0])):
            raise ValueError("Invalid OAuth URL")
    return parsed, {k: v[0] for k, v in values.items()}


class MessagingOAuthFlow(DashboardOAuthFlow):
    def __init__(self, source, home, server, redirect_uri):
        super().__init__(secrets.token_hex(16), server, source.profile, home, redirect_uri)
        self.source = copy.copy(source)
        self.owner = principal(source, home)
        self.deadline = time.monotonic() + 300
        self._lock = threading.RLock()
        self.credentials_committed = False

    def __repr__(self):
        return f"MessagingOAuthFlow(status={self.status!r})"

    def check_live(self):
        if self.status in {"error", "approved"}:
            raise RuntimeError("OAuth attempt ended")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("OAuth attempt expired")

    async def publish_authorization_url(self, url):
        parsed, values = _query(url)
        if (parsed.scheme != "https" or not parsed.hostname or
                values.get("redirect_uri") != self.redirect_uri or not values.get("state")):
            raise ValueError("Invalid authorization URL")
        def publish():
            with self._lock:
                self.check_live()
                if self.expected_state is not None:
                    raise RuntimeError("OAuth attempt already published its authorization URL")
                asyncio.run(super(MessagingOAuthFlow, self).publish_authorization_url(url))
        await asyncio.to_thread(publish)

    def deliver_url(self, source, home, url):
        try:
            parsed, values = _query(url)
            expected = urlsplit(self.redirect_uri)
            if (principal(source, home) != self.owner or
                    (parsed.scheme, parsed.netloc, parsed.path) !=
                    (expected.scheme, expected.netloc, expected.path) or
                    bool(values.get("code")) == bool(values.get("error")) or
                    ("code" in values and "error" in values)):
                return False
            with self._lock:
                self.check_live()
                # Never propagate provider-controlled error_description/error text.
                self.deliver_callback(code=values.get("code"), state=values.get("state"),
                                      error="Authorization declined" if values.get("error") else None)
            return True
        except (ValueError, RuntimeError, TimeoutError, UnicodeError):
            return False

    async def wait_for_callback(self, timeout=300):
        def check():
            with self._lock:
                self.check_live()
        await asyncio.to_thread(check)
        result = await super().wait_for_callback(min(timeout, max(0, self.deadline - time.monotonic())))
        await asyncio.to_thread(check)
        return result

    def cancel(self):
        with self._lock:
            if not self.credentials_committed:
                self.mark_error("OAuth attempt cancelled")

    def mark_error(self, error):
        # Exception strings may contain callback URLs, codes or provider bodies.
        super().mark_error("OAuth connection failed or cancelled. Start a new login.")


async def intercept_callback(runner, event):
    from tools.mcp_oauth_redact import redact_oauth_log
    # Later messages can quote platform history containing an earlier callback.
    # Scrub those context fields even when this message is ordinary conversation.
    for field in ("channel_context", "reply_to_text"):
        value = getattr(event, field, None)
        if isinstance(value, str):
            setattr(event, field, redact_oauth_log(value))
    if not callback_candidate(event.text):
        return False
    raw = callback_url(event.text)
    event.text = "[OAuth callback redacted]"
    event.raw_message = None
    event.metadata = {}
    if event.internal or not event.allow_gateway_control or event.source.is_bot:
        return True
    relay = getattr(runner, "_mcp_oauth_relay", None) if runner is not None else None
    if isinstance(relay, MessagingOAuthRelay):
        await relay.deliver(event.source, raw)
    return True


class MessagingOAuthRelay:
    def __init__(self, runner):
        self.runner = runner
        self.loop = asyncio.get_running_loop()
        self.attempts = {}
        self.tasks = set()
        self.closed = False

    def request(self, params):
        """Local control socket entry. Resolve origin from the gateway, never a
        caller-supplied chat destination. The OS-user-owned socket is trusted like
        the existing terminal/config surface; it is not a remote auth endpoint.
        """
        from gateway.run import _profile_runtime_scope
        from gateway.slash_access import policy_for_source
        from hermes_cli.mcp_config import _get_mcp_servers
        from hermes_cli.mcp_security import validate_mcp_server_entry
        required = ("session_id", "user_id", "home", "server")
        if any(not isinstance(params.get(k), str) or not params[k] for k in required):
            raise ValueError("Missing OAuth origin")
        entry = self.runner.session_store.lookup_by_session_id(params["session_id"])
        source = entry.origin if entry else None
        if (source is None or source.user_id != params["user_id"] or
                not self.runner._is_user_authorized_for_source(source)):
            raise ValueError("Invalid OAuth origin")
        policy = policy_for_source(self.runner.config, source)
        if policy.enabled and not policy.is_admin(source.user_id):
            raise ValueError("MCP configuration is restricted")
        home = self.runner._resolve_profile_home_for_source(source)
        if Path(home).resolve() != Path(params["home"]).resolve():
            raise ValueError("Invalid OAuth profile")
        server = params["server"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", server):
            raise ValueError("Invalid MCP server name")
        action = params.get("action", "start")
        if action not in {"start", "cancel"}:
            raise ValueError("Invalid OAuth operation")
        with _profile_runtime_scope(home):
            cfg = _get_mcp_servers().get(server)
        if params.get("url"):
            if cfg is not None:
                raise ValueError("MCP server already configured")
            cfg = {"url": params["url"], "auth": "oauth"}
        if action == "start" and (not isinstance(cfg, dict) or cfg.get("auth") != "oauth"
                                  or not cfg.get("url") or validate_mcp_server_entry(server, cfg)):
            raise ValueError("Configure an OAuth MCP server first")

        async def execute():
            try:
                if action == "cancel":
                    attempt = self.attempts.get((str(Path(home).resolve()), server))
                    if attempt is not None and attempt.owner == principal(source, home):
                        await asyncio.to_thread(attempt.cancel)
                else:
                    self.start(source, home, server, cfg)
            except Exception:
                try:
                    await self.notify(source, "OAuth login could not start. Check the server configuration or cancel its active login.")
                except Exception:
                    pass
        def schedule():
            task = asyncio.create_task(execute())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        self.loop.call_soon_threadsafe(schedule)
        return {"status": "queued"}

    def start(self, source, home, server, cfg):
        from tools.mcp_oauth import HermesTokenStorage, _cached_redirect
        from gateway.slash_commands_login import _LOGIN_BLOCKED_PLATFORMS
        if self.closed:
            raise RuntimeError("Gateway OAuth relay is stopping")
        if (not source.user_id or not source.chat_id or source.is_bot or
                source.platform.value in _LOGIN_BLOCKED_PLATFORMS):
            raise ValueError("OAuth requires an identified sender")
        home = str(Path(home).resolve())
        # Credentials are profile/server scoped, not per sender. Serialize even
        # different principals sharing that credential slot to prevent overwrites.
        key = (home, server)
        if key in self.attempts:
            raise RuntimeError("An OAuth login is already active for this server")
        oauth = cfg.get("oauth") or {}
        cached, _ = _cached_redirect(HermesTokenStorage(server, hermes_home=home))
        port = int(oauth.get("redirect_port") or 8765)
        if not 1 <= port <= 65535:
            raise ValueError("Invalid OAuth redirect port")
        redirect = oauth.get("redirect_uri") or cached or f"http://127.0.0.1:{port}/callback"
        parsed = urlsplit(redirect)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("Messaging OAuth requires a loopback redirect")
        attempt = MessagingOAuthFlow(source, home, server, redirect)
        self.attempts[key] = attempt
        task = asyncio.create_task(self._run(attempt, cfg, key))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return attempt

    async def _run(self, attempt, cfg, key):
        from gateway.mcp_oauth_worker import probe_and_commit
        worker = asyncio.create_task(asyncio.to_thread(probe_and_commit, attempt, cfg))
        try:
            url = await attempt.wait_for_authorization_url(30)
            await self.notify(attempt.source, "Open this authorization URL, then paste the exact loopback callback URL here, even if the browser cannot load it. Expires in five minutes.\n" + url)
            await asyncio.shield(worker)
            if attempt.status != "approved":
                raise RuntimeError("OAuth failed")
            await self.notify(attempt.source, "MCP connected. Start a new session to use the discovered tools.")
        except asyncio.CancelledError:
            await asyncio.to_thread(attempt.cancel)
            raise
        except Exception:
            await asyncio.to_thread(attempt.cancel)
            if attempt.status == "approved":
                return  # A delivery failure does not turn a completed login into failure.
            try:
                message = ("OAuth credentials saved, but MCP discovery failed. Try /reload-mcp." if attempt.credentials_committed else
                           "OAuth connection failed, expired, or was cancelled. Start a new login.")
                await self.notify(attempt.source, message)
            except Exception:
                pass
        finally:
            # The worker sees the cancelled flow and cannot commit staged tokens.
            # Keep ownership until it unwinds, including cancellation during I/O.
            try:
                await asyncio.shield(worker)
            except (Exception, asyncio.CancelledError):
                pass
            if self.attempts.get(key) is attempt:
                self.attempts.pop(key, None)

    async def notify(self, source, text):
        adapter = self.runner._adapter_for_source(source)
        if adapter is None:
            raise RuntimeError("Messaging transport unavailable")
        metadata = dict(self.runner._thread_metadata_for_source(source) or {})
        metadata["_interim_send"] = True
        result = await adapter.send(source.chat_id, text, metadata=metadata)
        if not result or not result.success:
            raise RuntimeError("Messaging delivery failed")

    async def deliver(self, source, url):
        accepted = False
        try:
            if not self.runner._is_user_authorized_for_source(source):
                return
            home = self.runner._resolve_profile_home_for_source(source)
            for attempt in tuple(self.attempts.values()):
                if await asyncio.to_thread(attempt.deliver_url, source, home, url):
                    accepted = True
                    break
            await self.notify(source, "OAuth callback accepted; connecting." if accepted else
                              "OAuth callback rejected. Use the exact URL in the originating chat during an active login.")
        except Exception:
            # Transport exceptions may embed the original payload. Never log them.
            return

    async def close(self):
        self.closed = True
        await asyncio.gather(*(asyncio.to_thread(attempt.cancel)
                               for attempt in tuple(self.attempts.values())))
        for task in self.tasks:
            task.cancel()
