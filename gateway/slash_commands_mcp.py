"""The paired-DM-only ``/mcp`` command: connect an OAuth MCP server from chat.

A gateway host has no browser, so the browser flow's loopback callback cannot work there (#78169).
The device flow (RFC 8628, ``tools.mcp_oauth_device``) needs none: Hermes sends the sign-in link
and code to the chat that asked, the person approves on any device, and the server is saved and
reloaded only once a token was granted, so a failed or abandoned attempt writes nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from gateway.slash_access import policy_for_runner_source
from gateway.slash_commands_login import _LOGIN_BLOCKED_PLATFORMS, _publish_local_notice

logger = logging.getLogger("gateway.run")

MCP_USAGE = ("Connect an MCP server from chat:\n"
             "• `/mcp install <name>` — a server from the MCP catalog\n"
             "• `/mcp add <name> <https-url>` — any remote server with OAuth sign-in\n"
             "• `/mcp login <name>` — sign in again to a configured server\n"
             "• `/mcp remove <name>` — disconnect a server and forget its sign-in")
MCP_DM_ONLY = "Connect MCP servers from a direct message with Hermes, so the sign-in link reaches only you."
MCP_NOT_ALLOWED = "Only an admin can connect MCP servers."
MCP_BUSY_ELSEWHERE = "Another MCP sign-in is in progress. Try again when it has finished."
MCP_BAD_NAME = "A server name is letters, digits, `-` and `_` only (up to 64)."
MCP_BAD_URL = "The server address must be a full `https://` URL."
MCP_STARTING = "Starting sign-in for {name}…"
MCP_CODE = ("To connect {name}, open this link, sign in and approve:\n{link}\n\n"
            "Code: {code} (works for {minutes} min). I'll tell you here when it's connected.")
MCP_CONNECTED = "Connected {name}."
MCP_FAILED = "{name} was not connected: {error}"
MCP_REJECTED = "{name} was not saved: its configuration looks unsafe."
MCP_REMOVED = "Removed {name} and forgot its sign-in."


def _unknown(name: str) -> str:
    return f"No MCP catalog entry named `{name}`. Use `/mcp add {name} <https-url>` for any other server."


def _needs_terminal(name: str) -> str:
    return (f"`{name}` needs setup a chat cannot do (credentials or a local install). "
            f"Run `hermes mcp install {name}` on the Hermes host, or use the dashboard.")


def _exists(name: str) -> str:
    return f"`{name}` is already configured. Use `/mcp login {name}` to sign in again."


def _not_oauth(name: str) -> str:
    return f"`{name}` is not a remote server with OAuth sign-in, so there is nothing to sign in to."


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MCP_SINGLE = "mcp"


@dataclasses.dataclass
class _McpAttempt:
    key: tuple
    source: Any
    name: str
    url: str
    cfg: dict
    save: bool  # install/add save the server once signed in; login keeps the existing block
    catalog: bool = False
    fresh: bool = False


def _configured_servers() -> dict:
    from hermes_cli.config import load_config
    servers = (load_config() or {}).get("mcp_servers")
    return servers if isinstance(servers, dict) else {}


def _device_cfg(cfg: dict) -> dict:
    oauth = cfg.get("oauth") if isinstance(cfg.get("oauth"), dict) else {}
    return {**cfg, "auth": "oauth", "oauth": {**oauth, "flow": "device"}}


class GatewayMcpCommandsMixin:

    def _mcp_profile_scope(self, source):
        """The requesting profile's runtime scope under multiplex, as ``_execute_mcp_reload``."""
        from gateway.run import _profile_runtime_scope
        from hermes_constants import get_hermes_home_override
        if bool(getattr(self.config, "multiplex_profiles", False)) and not get_hermes_home_override():
            return _profile_runtime_scope(Path(self._resolve_profile_home_for_source(source)))
        return contextlib.nullcontext()

    async def _handle_mcp_command(self, event) -> str:
        src = event.source
        platform = str(getattr(src.platform, "value", src.platform)).lower()
        if not (getattr(src, "chat_type", None) in {"dm", "private"} and getattr(src, "chat_id", None)
                and platform not in _LOGIN_BLOCKED_PLATFORMS):
            return MCP_DM_ONLY
        policy = policy_for_runner_source(self, src)
        if policy.enabled and not policy.is_admin(getattr(src, "user_id", None)):
            return MCP_NOT_ALLOWED

        get_args = getattr(event, "get_command_args", None)
        parts = (get_args() if callable(get_args) else "").split()
        if not parts or parts[0].lower() not in {"install", "add", "login", "remove"}:
            return MCP_USAGE
        sub, rest = parts[0].lower(), parts[1:]
        if len(rest) != (2 if sub == "add" else 1):
            return MCP_USAGE
        name = rest[0]
        if not _NAME_RE.match(name):
            return MCP_BAD_NAME
        if sub == "remove":
            return await self._mcp_remove(src, name, event)

        with self._mcp_profile_scope(src):
            built = self._mcp_attempt_config(sub, name, rest)
        if isinstance(built, str):
            return built
        url, cfg, save, catalog, fresh = built

        key = (platform, str(src.chat_id), str(getattr(src, "user_id", "") or ""))
        attempts = getattr(self, "_mcp_attempts", None)
        if not isinstance(attempts, dict):
            attempts = self._mcp_attempts = {}
        live = attempts.get(_MCP_SINGLE)
        if live is not None and not live[1].done():
            if live[0].key != key:
                return MCP_BUSY_ELSEWHERE
            live[1].cancel()  # the same person asking again replaces their own attempt
        attempt = _McpAttempt(key=key, source=src, name=name, url=url, cfg=cfg, save=save,
                              catalog=catalog, fresh=fresh)
        task = asyncio.create_task(self._run_mcp_connect(attempt, event))
        attempts[_MCP_SINGLE] = (attempt, task)
        self._retain_background_task(task)
        return MCP_STARTING.format(name=name)

    async def _mcp_remove(self, src, name: str, event) -> str:
        """Drop the server block and its stored tokens, as ``hermes mcp remove``, then reload."""
        live = (getattr(self, "_mcp_attempts", None) or {}).get(_MCP_SINGLE)
        if live is not None and not live[1].done() and live[0].name == name:
            live[1].cancel()  # a sign-in still waiting for this server would save it back
        with self._mcp_profile_scope(src):
            if name not in _configured_servers():
                return f"`{name}` is not configured, so there is nothing to remove."
            from hermes_cli.mcp_config import _remove_mcp_server
            _remove_mcp_server(name)
            with contextlib.suppress(Exception):
                from tools.mcp_oauth_manager import get_manager
                get_manager().remove(name)
        summary = await self._execute_mcp_reload(event)
        return MCP_REMOVED.format(name=name) + "\n\n" + summary

    def _mcp_attempt_config(self, sub: str, name: str, rest: list):
        """(url, cfg, save, catalog, fresh) for the attempt, or the reply that refuses it."""
        servers = _configured_servers()
        if sub == "login":
            cfg = servers.get(name)
            if not isinstance(cfg, dict):
                return f"`{name}` is not configured. Use `/mcp install {name}` or `/mcp add {name} <https-url>`."
            if cfg.get("auth") != "oauth" or not cfg.get("url"):
                return _not_oauth(name)
            return str(cfg["url"]), _device_cfg(cfg), False, False, False
        if name in servers:
            return _exists(name)
        if sub == "add":
            url = rest[1]
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname:
                return MCP_BAD_URL
            return url, {"url": url, "auth": "oauth", "oauth": {"flow": "device"}, "enabled": True}, True, False, True
        from hermes_cli.mcp_catalog import card_install_config, get_entry
        entry = get_entry(name)
        if entry is None:
            return _unknown(name)
        if entry.transport.type != "http" or entry.auth.type != "oauth" or entry.auth.provider:
            return _not_oauth(name) if entry.auth.type != "oauth" else _needs_terminal(name)
        if entry.auth.env or entry.install is not None:
            return _needs_terminal(name)
        cfg = card_install_config(entry)
        return str(cfg.get("url") or entry.transport.url), _device_cfg(cfg), True, True, True

    async def _run_mcp_connect(self, attempt: _McpAttempt, event) -> None:
        from tools.mcp_oauth_device import login_device

        async def on_code(link: str, code: str, minutes: int) -> None:
            await self._push_mcp(attempt, MCP_CODE.format(name=attempt.name, link=link, code=code, minutes=minutes))

        try:
            with self._mcp_profile_scope(attempt.source):
                await login_device(attempt.name, attempt.url, attempt.cfg.get("oauth") or {}, on_code=on_code)
                if attempt.save:
                    from hermes_cli.mcp_config import _save_mcp_server
                    if not _save_mcp_server(attempt.name, attempt.cfg):
                        self._record_mcp_install(attempt, "failed", failure_class="config_rejected")
                        await self._push_mcp(attempt, MCP_REJECTED.format(name=attempt.name))
                        return
                self._record_mcp_install(attempt, "success")
            summary = await self._execute_mcp_reload(event)
            await self._push_mcp(attempt, MCP_CONNECTED.format(name=attempt.name) + "\n\n" + summary)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("/mcp %s: sign-in failed", attempt.name, exc_info=True)
            self._record_mcp_install(attempt, "failed", error=exc)
            # login_device raises RuntimeError with fixed, credential-free wording.
            error = str(exc) if isinstance(exc, RuntimeError) else "the sign-in did not complete"
            await self._push_mcp(attempt, MCP_FAILED.format(name=attempt.name, error=error))

    def _record_mcp_install(self, attempt: _McpAttempt, outcome: str, **kwargs) -> None:
        """A first catalog install, recorded as the dashboard card records one."""
        if not (attempt.fresh and attempt.catalog):
            return
        with contextlib.suppress(Exception):
            from hermes_cli.mcp_catalog import record_mcp_install
            record_mcp_install("catalog", attempt.name, outcome, **kwargs)

    async def _push_mcp(self, attempt: _McpAttempt, text: str) -> None:
        try:
            if _publish_local_notice(self, attempt.source, text):
                return
            await self._deliver_platform_notice(attempt.source, text)
        except Exception:
            logger.warning("/mcp %s: push failed", attempt.name, exc_info=True)
