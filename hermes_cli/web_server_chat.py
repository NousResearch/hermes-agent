"""Chat/terminal WebSocket plumbing: PTY bridge selection and registry, WS
client/origin/auth gates, chat argv resolution, gateway/sidecar URL building.
"""

import logging
import asyncio
import atexit
import concurrent.futures
import contextlib
import hmac
import os
import re
import sys
import tempfile
import threading
import urllib.request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from typing import Optional
from hermes_cli.pty_session import PtySessionRegistry

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# /api/pty spawns ``hermes --tui`` behind a pseudo-terminal and forwards bytes +
# resize escapes to xterm.js.  POSIX uses pty_bridge (fcntl/termios); native
# Windows uses win_pty_bridge (pywinpty/ConPTY); same surface, no handler guards.
try:
    if sys.platform.startswith("win"):
        from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, PtyUnavailableError
    else:
        from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
    _PTY_BRIDGE_AVAILABLE = True
except ImportError:  # pragma: no cover - pywinpty / ptyprocess missing
    PtyBridge = None  # type: ignore[assignment]
    _PTY_BRIDGE_AVAILABLE = False

    class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
        """Stub when the platform PTY bridge cannot be imported."""
_RESIZE_RE = re.compile(rb"\x1b\[RESIZE:(\d+);(\d+)\]")
_PTY_READ_CHUNK_TIMEOUT = 0.2

# Back-off between idle PTY reads so a quiet terminal does not spin the event
# loop (keeps dashboard idle CPU low).
# A positive sleep lets other coroutines run and keeps dashboard idle CPU low (#42627).
_PTY_IDLE_BACKOFF = 0.05
PTY_REGISTRY = PtySessionRegistry(
    ttl=30 * 60, max_sessions=16, buffer_cap=1 * 1024 * 1024, read_timeout=_PTY_READ_CHUNK_TIMEOUT)


async def _close_stalled_pty_input(ws: "WebSocket", *, path: str) -> None:
    """Close only the terminal socket when its child stops accepting input."""
    _log.warning("pty input stalled path=%s; recycling terminal session", path)
    try:
        await ws.close(code=1013, reason="PTY input stalled")
    except Exception:
        pass


async def _legacy_pump(ws: "WebSocket", bridge) -> None:
    """Original 1:1 socket<->PTY pump: stream until disconnect, then close the
    bridge. Used when no ``?attach=`` token is supplied (keep-alive opt-in).

    Behavior is identical to the pre-keep-alive ``pty_ws`` body, including the 54028 half-open-socket
    protection (reader EOF → close the WS so the writer's ``ws.receive()`` unparks) and the #53227
    ``to_thread`` offloads for the blocking ``bridge.close()``.
    """
    loop = asyncio.get_running_loop()

    async def pump_pty_to_ws() -> None:
        try:
            while True:
                chunk = await loop.run_in_executor(None, bridge.read, _PTY_READ_CHUNK_TIMEOUT)
                if chunk is None:  # EOF
                    return
                if not chunk:  # no data this tick; yield control and retry
                    await asyncio.sleep(_PTY_IDLE_BACKOFF)
                    continue
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    return
        finally:
            # Close the WS so the writer's ``ws.receive()`` returns instead of
            # blocking forever on a half-open browser socket (fds would leak and
            # auto-reconnect stacks a fresh PTY on each orphan).  Reap the bridge
            # here too (idempotent): cancelling the handler the instant the WS
            # closes can skip the writer's ``finally``.
            with contextlib.suppress(Exception):
                # The child has exited (EOF) or the send side broke. Closing from the EOF path makes the
                # reap independent of that cancellation race (#54028).
                await asyncio.to_thread(bridge.close)
            with contextlib.suppress(Exception):
                await ws.close()

    reader_task = asyncio.create_task(pump_pty_to_ws())

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # ws.receive() after the socket is already disconnected
                # (e.g. closed by the reader task above).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue
            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue
            if not await bridge.write(raw):
                await _close_stalled_pty_input(ws, path="legacy")
                break
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader_task
        await asyncio.to_thread(bridge.close)


# Starlette's TestClient reports the peer as "testclient"; treat it as
# loopback so tests don't need to rewrite request scope.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _ws_client_reason(ws: "WebSocket") -> Optional[str]:
    """Return a rejection reason token for the peer IP, or None when allowed.

    Loopback bind: only loopback peers (the legacy ``?token=`` is the only auth,
    LAN hosts must not get to guess it); an empty peer fails closed.  Explicit
    non-loopback bind (``--insecure``) or gated mode: any peer — DNS-rebinding is
    blocked by :func:`_ws_host_origin_reason`, and in gated mode
    ``ws.client.host`` is the X-Forwarded-For value anyway.
    """
    from hermes_cli.web_server import app
    if getattr(app.state, "auth_required", False):
        return None
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return None
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        return f"missing_or_empty_peer bound={bound_host or '?'}"
    if client_host in _LOOPBACK_HOSTS:
        return None
    return f"peer_not_loopback peer={client_host} bound={bound_host or '?'}"


def _ws_client_is_allowed(ws: "WebSocket") -> bool:
    """True when the peer IP passes :func:`_ws_client_reason`."""
    return _ws_client_reason(ws) is None


def _ws_host_origin_reason(ws: "WebSocket") -> Optional[str]:
    """Return ``host_mismatch …`` / ``origin_mismatch …``, or None when allowed.

    HTTP middleware does not run for WebSocket routes, so the DNS-rebinding
    Host check is repeated here; an Origin header, when present, must target the
    bound host.  Non-web origins (packaged Electron: file://, null, app://) are
    trusted — the credential check is the real auth boundary there.
    """
    from hermes_cli.web_server import _is_accepted_host, app
    bound_host = getattr(app.state, "bound_host", None)
    if not bound_host:
        return None
    trusted_public_hosts = getattr(app.state, "trusted_public_hosts", frozenset())
    host_header = ws.headers.get("host", "")
    if not _is_accepted_host(host_header, bound_host, trusted_public_hosts):
        return f"host_mismatch host={host_header or '?'} bound={bound_host}"
    origin = ws.headers.get("origin", "")
    if not origin:
        return None
    parsed = urllib.parse.urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc or not _is_accepted_host(parsed.netloc, bound_host, trusted_public_hosts):
        return f"origin_mismatch origin={origin} bound={bound_host}"
    return None


def _ws_host_origin_is_allowed(ws: "WebSocket") -> bool:
    """True when the upgrade passes the dashboard Host/Origin guard."""
    return _ws_host_origin_reason(ws) is None


def _ws_request_is_allowed(ws: "WebSocket") -> bool:
    """Return True when the WebSocket upgrade matches dashboard boundaries."""
    return _ws_host_origin_is_allowed(ws) and _ws_client_is_allowed(ws)


_GATEWAY_WS_PROTOCOL = "hermes-gateway-v1"
_GATEWAY_WS_TICKET_PROTOCOL_PREFIX = "hermes-gateway-ticket."


def _gateway_ws_ticket_from_subprotocol(ws: "WebSocket") -> tuple[str, str]:
    """Return ``(ticket, reason)`` from an unambiguous gateway protocol set."""
    raw = str(ws.headers.get("sec-websocket-protocol", "") or "")
    protocols = [value.strip() for value in raw.split(",") if value.strip()]
    ticket_protocols = [
        value for value in protocols if value.startswith(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX)]
    if not ticket_protocols:
        return "", "none"
    if _GATEWAY_WS_PROTOCOL not in protocols or len(ticket_protocols) != 1:
        return "", "invalid"
    ticket = ticket_protocols[0][len(_GATEWAY_WS_TICKET_PROTOCOL_PREFIX):]
    return (ticket, "ok") if ticket else ("", "invalid")


def _ws_auth_reason(ws: "WebSocket") -> tuple[Optional[str], str]:
    """Validate WS-upgrade auth; return ``(reason, credential)``.

    ``reason`` is None when accepted, else a short token (``no_credential``,
    ``token_mismatch``, ``ticket_invalid``, ``internal_invalid``);
    ``credential`` names what was presented so the accept path can log *how*.

    Loopback / ``--insecure``: legacy ``?token=`` (constant-time compared).
    Gated: ``?ticket=`` (browser-minted, single-use, 30s TTL) or ``?internal=``
    (process-lifetime, multi-use, only for server-spawned WS clients so the PTY
    child can reconnect; never injected into the SPA).  The legacy token is
    rejected in gated mode: a leaked ``_SESSION_TOKEN`` must not grant access.
    """
    from hermes_cli.web_server import _SESSION_TOKEN, app
    auth_required = bool(getattr(app.state, "auth_required", False))
    if auth_required:
        # Lazy import — keeps this function importable in test harnesses
        # that don't bring in the dashboard_auth layer.
        from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
        from hermes_cli.dashboard_auth.ws_tickets import (
            TicketInvalid, consume_internal_credential, consume_ticket)

        def _reject(reason: str) -> None:
            audit_log(
                AuditEvent.WS_TICKET_REJECTED, reason=reason,
                ip=(ws.client.host if ws.client else ""), path=ws.url.path)

        def _stamp_identity(info) -> None:
            # Server-minted {user_id, provider} stamped onto the WS object is the
            # sole identity authority downstream (gateway transport / controller
            # registration); a client can never supply it through RPC params.
            # Only the two identity fields are carried — bookkeeping such as
            # ``minted_at`` is not part of the identity contract.
            ws._hermes_auth_identity = {
                "user_id": info.get("user_id"), "provider": info.get("provider")}

        internal = ws.query_params.get("internal", "")
        if internal:
            try:
                _stamp_identity(consume_internal_credential(internal))
                return None, "internal"
            except TicketInvalid as exc:
                _reject(f"internal: {exc}")
                return "internal_invalid", "internal"

        protocol_ticket, protocol_reason = _gateway_ws_ticket_from_subprotocol(ws)
        if protocol_reason == "invalid":
            return "ticket_invalid", "ticket-subprotocol"
        ticket = protocol_ticket or ws.query_params.get("ticket", "")
        if not ticket:
            return "no_credential", "none"

        try:
            info = consume_ticket(ticket)
            if info.get("provider") == "bot-desktop":
                # A display ticket admits one RFB bridge on /api/display/ws (a watch-only
                # capability handed to a screen viewer); it must not double as a login here.
                raise TicketInvalid("display ticket presented as a gateway login")
            _stamp_identity(info)
            if protocol_ticket:
                # Select only the stable public protocol during accept. The
                # ticket-bearing protocol is a credential and must never be
                # reflected back to the browser or retained after admission.
                ws._hermes_ws_subprotocol = _GATEWAY_WS_PROTOCOL
                return None, "ticket-subprotocol"
            return None, "ticket"
        except TicketInvalid as exc:
            _reject(str(exc))
            return "ticket_invalid", "ticket"

    token = ws.query_params.get("token", "")
    if not token:
        return "no_credential", "none"
    if hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        return None, "token"
    return "token_mismatch", "token"


def _ws_auth_ok(ws: "WebSocket") -> bool:
    """True when the WS-upgrade credential is accepted. See _ws_auth_reason."""
    return _ws_auth_reason(ws)[0] is None


def _resolve_chat_argv(
    resume: Optional[str] = None, sidecar_url: Optional[str] = None, profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
    workspace_cwd: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve the argv + cwd + env for the chat PTY (what ``hermes --tui`` runs).

    Tests monkeypatch this with a tiny fake command.  Env contract: resume goes
    through ``HERMES_TUI_RESUME`` (``ui-tui`` does not parse argv), resolved to
    the newest descendant; ``HERMES_TUI_GATEWAY_URL`` attaches to this process's
    in-memory gateway but is SKIPPED for profile-scoped chats (that gateway runs
    under the dashboard's own profile, so a scoped chat spawns its own);
    ``profile`` scopes the ENTIRE chat by pointing ``HERMES_HOME`` at the profile
    dir, the same propagation ``hermes -p <name>`` performs. ``workspace_cwd``
    (an already-validated host directory, ``chat_workspaces.resolve_chat_cwd``)
    is the workspace the user picked for a FRESH chat: it becomes ``HERMES_CWD``
    (where a self-spawned gateway starts) and ``HERMES_TUI_CWD`` (what the TUI
    passes as the explicit ``cwd`` of ``session.create`` when attached to the
    in-memory gateway, whose own cwd is the dashboard's launch dir).
    """
    from hermes_cli.web_server_profiles import (
        _config_profile_scope,
        _hermes_home_scope,
        _resolve_profile_dir,
    )
    from hermes_cli.web_server_sessions import (
        _open_session_db_at_path,
        _open_session_db_for_profile,
        _session_latest_descendant,
    )
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.main_tui_launch import _apply_tui_python_env, _make_tui_argv

    from tui_gateway.launch_profile_policy import (
        capture_launch_authority,
        launch_authority,
    )

    profile_dir: Optional[Path] = None
    requested = (profile or "").strip()
    if requested and requested.lower() != "current":
        # Freeze routing identity before resolving a user-selected profile name.
        # Later process-global HERMES_HOME mutations must not redirect this name
        # to another profiles tree.
        authority = capture_launch_authority()
        profile_dir = _resolve_profile_dir(requested, launch_home=authority.home)
    else:
        authority = launch_authority()

    argv, cwd = _make_tui_argv(PROJECT_ROOT / "ui-tui", tui_dev=False)
    # One launch-authority snapshot feeds both the current-profile environment and
    # the named-profile child base. Once multiplexing starts this is frozen, never
    # a live os.environ that secondary profile work may have mutated.
    from tools.environments.local import (
        build_subprocess_env,
        hermes_subprocess_env,
        served_profile_child_env,
    )
    if profile_dir is None:
        env = None
    else:
        from hermes_cli.config import read_raw_config
        from tools.environments.local_env_policy import declared_credential_env_names

        with _hermes_home_scope(authority.home):
            with _config_profile_scope("current", launch_home=authority.home):
                launch_credential_env_names = declared_credential_env_names(
                    read_raw_config()
                )
        # Entering the first routed scope freezes both launch values and launch-home
        # identity before any secondary profile work can mutate process state.
        with _config_profile_scope(
            requested,
            launch_home=authority.home,
            resolved_profile_dir=profile_dir,
        ):
            selected_credential_env_names = declared_credential_env_names(read_raw_config())
            credential_env_names = (
                launch_credential_env_names | selected_credential_env_names
            )
            launch_base = dict(authority.env)
            env = served_profile_child_env(
                base=hermes_subprocess_env(
                    base_env=launch_base, inherit_credentials=True
                ),
                target_home=profile_dir,
                inherit_credentials=True,
                launch_home=authority.home,
                credential_env_names=credential_env_names,
            )
    launch_base = dict(authority.env)
    launch_env = build_subprocess_env(
        base=launch_base, scrub_secrets=False, inherit_profile_home=True
    )
    if env is None:
        env = launch_env
    try:
        from hermes_cli.config import (
            TERMINAL_CONFIG_ENV_MAP, apply_terminal_config_to_env, read_raw_config,
            terminal_config_owned_env_vars)

        if profile_dir is not None:
            # served_profile_child_env drops every bridged TERMINAL_* value. Restore
            # only operator exports not owned by the launch profile, then let the
            # selected profile's explicit config override them as before. Values
            # loaded from the launch profile's .env or external secret sources are
            # profile residue, not exports.
            from agent.secret_scope import load_env_file
            from hermes_cli.env_loader import get_secret_source_owned_names

            with _hermes_home_scope(authority.home):
                raw_launch_terminal = read_raw_config().get("terminal")
            launch_owned = terminal_config_owned_env_vars(raw_launch_terminal)
            if isinstance(raw_launch_terminal, dict) and "home_mode" in raw_launch_terminal:
                launch_owned.add("TERMINAL_HOME_MODE")
            launch_dotenv_owned = set(load_env_file(authority.home / ".env"))
            launch_external_owned = set(get_secret_source_owned_names(authority.home))
            restorable_terminal = set(TERMINAL_CONFIG_ENV_MAP.values()) | {
                "TERMINAL_HOME_MODE"
            }
            launch_profile_owned = {
                name.upper()
                for name in (
                    launch_owned | launch_dotenv_owned | launch_external_owned
                )
            }
            for env_var in restorable_terminal:
                if (
                    env_var.upper() not in launch_profile_owned
                    and env_var in launch_env
                    and env_var not in env
                ):
                    env[env_var] = launch_env[env_var]
            with _config_profile_scope(
                requested,
                launch_home=authority.home,
                resolved_profile_dir=profile_dir,
            ):
                raw_selected_terminal = read_raw_config().get("terminal")
                if isinstance(raw_selected_terminal, dict) and "home_mode" in raw_selected_terminal:
                    selected_home_mode = raw_selected_terminal["home_mode"]
                    if selected_home_mode is None:
                        env.pop("TERMINAL_HOME_MODE", None)
                    else:
                        env["TERMINAL_HOME_MODE"] = str(selected_home_mode)
                apply_terminal_config_to_env(env=env)
        else:
            # Current-profile config must follow the immutable launch-home owner,
            # and resolve ${VAR} through its frozen launch secret scope, not later
            # process-global mutations from routed work.
            with _hermes_home_scope(authority.home):
                with _config_profile_scope("current", launch_home=authority.home):
                    apply_terminal_config_to_env(env=env)
    except Exception:
        _log.warning("Failed to apply terminal config bridge for dashboard chat", exc_info=True)
    if profile_dir is not None:
        # Profile .env files hold credentials, not routing authority. Re-pin the
        # selected home after overlaying credentials, then apply the subprocess
        # HOME policy using the launch process's real-home baseline.
        from hermes_constants import apply_subprocess_home_env

        env["HERMES_HOME"] = str(profile_dir)
        env["HERMES_PROFILE"] = requested
        real_home = launch_env.get("HERMES_REAL_HOME") or launch_env.get("HOME")
        if real_home:
            env["HERMES_REAL_HOME"] = real_home
            env["HOME"] = real_home
        else:
            env.pop("HERMES_REAL_HOME", None)
            env.pop("HOME", None)
        apply_subprocess_home_env(env, allow_process_fallback=False)
    if workspace_cwd:
        env["HERMES_CWD"] = workspace_cwd
        env["HERMES_TUI_CWD"] = workspace_cwd
    _apply_tui_python_env(env)
    env.setdefault("NODE_ENV", "production")
    # Mouse tracking would swallow wheel events the browser needs for
    # transcript scrolling; disable it for the dashboard PTY only.
    env.setdefault("HERMES_TUI_DISABLE_MOUSE", "1")
    env.setdefault("HERMES_TUI_INLINE", "1")
    # chalk in the child picks its color depth from the SERVER env; hosted
    # deploys have no COLORTERM, so hex colors would snap to the 256 palette.
    env.setdefault("COLORTERM", "truecolor")
    env["HERMES_TUI_DASHBOARD"] = "1"

    if resume:
        if profile_dir is not None:
            _resume_db = _open_session_db_at_path(
                profile_dir / "state.db", read_only=True
            )
        else:
            _resume_db = _open_session_db_for_profile(None, read_only=True)
        try:
            latest_resume, _latest_path = _session_latest_descendant(resume, _resume_db)
        finally:
            _resume_db.close()
        if latest_resume:
            resume = latest_resume
        env["HERMES_TUI_RESUME"] = resume

    if sidecar_url:
        env["HERMES_TUI_SIDECAR_URL"] = sidecar_url

    if active_session_file:
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file

    # Without the attach URL, gatewayClient spawns its own `tui_gateway.entry`,
    # which inherits the profile HERMES_HOME set above. A profile .env cannot
    # override this routing boundary.
    if profile_dir is not None:
        env.pop("HERMES_TUI_GATEWAY_URL", None)
    elif gateway_ws_url := _build_gateway_ws_url():
        env["HERMES_TUI_GATEWAY_URL"] = gateway_ws_url

    return list(argv), str(cwd) if cwd else None, env


# Wildcard bind hosts an in-container client must NOT dial: behind a forward
# proxy (HTTPS_PROXY without 0.0.0.0 in NO_PROXY) the handshake gets MITM'd.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def _resolve_client_ws_host() -> Optional[str]:
    """Host the in-container WS client should dial: ``HERMES_DASHBOARD_WS_HOST``
    wins always; a wildcard bind becomes ``127.0.0.1``; others verbatim."""
    from hermes_cli.web_server import app
    explicit = os.environ.get("HERMES_DASHBOARD_WS_HOST", "").strip()
    if explicit:
        return explicit
    host = getattr(app.state, "bound_host", None)
    if not host:
        return None
    return "127.0.0.1" if host in _WILDCARD_HOSTS else host


def _server_internal_ws_url(path: str, **extra_qs) -> Optional[str]:
    """``ws://<host>:<port><path>?<auth>&<extra>`` for server-spawned WS clients,
    or None when unbound.

    Gated mode uses the process-lifetime internal credential, NOT a single-use
    browser ticket: the child reads the URL once and reuses it on every
    reconnect, and a 30s-TTL ticket can expire before a slow cold boot dials.
    """
    from hermes_cli.web_server import _SESSION_TOKEN, app
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)
    if not host or not port:
        return None
    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    if getattr(app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        auth = {"internal": internal_ws_credential()}
    else:
        auth = {"token": _SESSION_TOKEN}
    return f"ws://{netloc}{path}?{urllib.parse.urlencode({**auth, **extra_qs})}"


def _build_gateway_ws_url() -> Optional[str]:
    """ws:// URL the PTY child attaches to for JSON-RPC gateway traffic."""
    return _server_internal_ws_url("/api/ws")


def _build_sidecar_url(channel: str) -> Optional[str]:
    """ws:// URL the PTY child publishes events to, or None when unbound."""
    return _server_internal_ws_url("/api/pub", channel=channel)


async def _resolve_chat_argv_async(
    resume: Optional[str] = None, sidecar_url: Optional[str] = None, profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
    workspace_cwd: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve chat argv off the event loop (it may run ``npm run build``); the
    async lock keeps one-build-at-a-time without parking worker threads."""
    from hermes_cli.web_server import _get_chat_argv_lock, app
    kwargs = {"resume": resume, "sidecar_url": sidecar_url, "profile": profile}
    if active_session_file is not None:
        kwargs["active_session_file"] = active_session_file
    if workspace_cwd is not None:
        kwargs["workspace_cwd"] = workspace_cwd

    async with _get_chat_argv_lock(app):
        return await asyncio.to_thread(_resolve_chat_argv, **kwargs)


def _active_session_file_for_channel(app: "FastAPI", channel: str) -> Path:
    """Return the per-channel file where a dashboard TUI writes its active sid."""
    from hermes_cli.web_server import _get_pty_active_session_files
    files = _get_pty_active_session_files(app)
    if files.get(channel) is None:
        fd, raw_path = tempfile.mkstemp(prefix="hermes-pty-active-", suffix=".json")
        os.close(fd)
        files[channel] = Path(raw_path)
    return files[channel]


# On timeout asyncio cancels the awaitable but the console thread keeps running;
# a small dedicated pool caps the leak instead of exhausting the default pool.
_CONSOLE_EXECUTOR_MAX_WORKERS = 4
_console_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_console_executor_lock = threading.Lock()


def _get_console_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the bounded console worker pool (once per process)."""
    global _console_executor
    if _console_executor is None:
        with _console_executor_lock:
            if _console_executor is None:
                _console_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CONSOLE_EXECUTOR_MAX_WORKERS, thread_name_prefix="hermes-console")
                # Tear down on interpreter exit without waiting on in-flight
                # workers: a stuck 60s console command must not block shutdown.
                atexit.register(
                    lambda: _console_executor
                    and _console_executor.shutdown(wait=False, cancel_futures=True))
    return _console_executor
