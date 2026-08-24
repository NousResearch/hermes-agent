"""Dashboard OAuth/login-status helpers: provider catalog, per-provider device pollers,
Anthropic/Copilot/Claude-Code status probes.
"""

import logging
import functools
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


_LOGGED_OUT: Dict[str, Any] = {"logged_in": False, "source": None}


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """``…XXXXXX`` (last N chars) for UI display. JWTs show only the tail of the signature
    segment. A callable (Azure Foundry Entra-ID bearer provider) is NEVER invoked — it yields
    the ``<entra-id-bearer>`` placeholder."""
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        return "<entra-id-bearer>"
    s = str(value)
    if s.count(".") >= 2:
        s = s.rsplit(".", 1)[-1]
    return s if len(s) <= visible else f"…{s[-visible:]}"


def _token_status(source: str, source_label: str, creds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "logged_in": True, "source": source, "source_label": source_label,
        "token_preview": _truncate_token(creds.get("accessToken")),
        "expires_at": creds.get("expiresAt"), "has_refresh_token": bool(creds.get("refreshToken")),
    }


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Status for the "Anthropic API Key" card: Hermes-managed PKCE file first, then the
    registry-ordered env vars (process env — where Bitwarden-sourced secrets land — then .env).

    Claude Code's ``~/.claude/.credentials.json`` is deliberately NOT read here; it has its own
    ``claude-code`` entry, and counting it here would shadow a real ANTHROPIC_API_KEY.
    """
    try:
        from agent.anthropic_credentials import read_hermes_oauth_credentials, _get_hermes_oauth_file
        hermes_creds = read_hermes_oauth_credentials()
    except Exception:
        hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return _token_status("hermes_pkce", f"Hermes PKCE ({_get_hermes_oauth_file()})", hermes_creds)

    env_var_order: tuple = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        env_var_order = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    except (ImportError, KeyError):
        pass
    from hermes_cli.config import get_env_value
    from hermes_cli.env_loader import format_secret_source_suffix
    for var in env_var_order:
        value = get_env_value(var) or os.getenv(var)
        if value:
            return {
                "logged_in": True, "source": "env_var", "source_label": f"{var}{format_secret_source_suffix(var)}",
                "token_preview": _truncate_token(value), "expires_at": None, "has_refresh_token": False,
            }
    return dict(_LOGGED_OUT)


def _claude_code_only_status() -> Dict[str, Any]:
    """Claude Code CLI credentials as their own entry, independent of the Anthropic card."""
    try:
        from agent.anthropic_credentials import read_claude_code_credentials
        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return _token_status("claude_code_cli", "~/.claude/.credentials.json", creds)
    return dict(_LOGGED_OUT)


def _copilot_acp_status() -> Dict[str, Any]:
    """Status for copilot-acp. ``logged_in`` only on positive evidence (env token or known on-disk
    store); the CLI may hold its session in an OS keychain Hermes can't read, so the unverified
    state reads "managed by the Copilot CLI" — never signed out."""
    try:
        from hermes_cli.auth import get_external_process_provider_status
        status = get_external_process_provider_status("copilot-acp") or {}
    except Exception:
        status = {}
    verified = bool(status.get("auth_verified"))
    configured = bool(status.get("configured"))
    if verified:
        source_label = status.get("auth_source") or "Copilot credentials detected"
    elif configured:
        found = status.get("resolved_command") or status.get("command") or "copilot"
        source_label = f"Managed by the GitHub Copilot CLI ({found})"
    else:
        source_label = "GitHub Copilot CLI not found on PATH"
    return {
        "logged_in": verified, "source": "copilot_cli", "source_label": source_label, "token_preview": None,
        "expires_at": None, "has_refresh_token": False, "configured": configured,
    }


def _external_process_cli_command(provider_id: str, default: str) -> str:
    """Render an external-process provider's sign-in command with the CLI actually configured
    (``HERMES_COPILOT_ACP_COMMAND`` / ``COPILOT_CLI_PATH``); others get ``default`` untouched."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, get_external_process_provider_status
        pconfig = PROVIDER_REGISTRY.get(provider_id)
        if not pconfig or pconfig.auth_type != "external_process":
            return default
        status = get_external_process_provider_status(provider_id) or {}
        command = str(status.get("command") or "").strip()
        if command:
            parts = default.split(" ", 1)
            tail = f" {parts[1]}" if len(parts) > 1 else ""
            return f"{command}{tail}"
    except Exception:
        pass
    return default


# Hand-tuned OAuth/account cards: the bits not derivable from the unified provider catalog
# (``flow``, ``status_fn``, ``cli_command``, display order). OVERRIDE BASE for
# ``_build_oauth_catalog()``, which unions them with every accounts-tab provider so new
# providers appear automatically. Also carries two non-catalog rows the Accounts tab needs:
# the Anthropic credential-status card and the synthetic ``claude-code`` row.
# ``flow``: ``device_code`` = show code + URL + poll; ``external`` = delegated to a terminal/CLI.
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    # status_fn None → dispatched via auth.get_<provider>_auth_status.
    {"id": "nous", "name": "Nous Portal", "flow": "device_code", "cli_command": "hermes auth add nous",
     "docs_url": "https://portal.nousresearch.com", "status_fn": None},
    {"id": "openai-codex", "name": "ChatGPT or Codex Subscription", "flow": "device_code",
     "cli_command": "hermes auth add openai-codex", "docs_url": "https://platform.openai.com/docs",
     "status_fn": None},
    {"id": "qwen-oauth", "name": "Qwen (via Qwen CLI)", "flow": "external",
     "cli_command": "hermes auth add qwen-oauth", "docs_url": "https://github.com/QwenLM/qwen-code",
     "status_fn": None},
    # Structurally device-code (verification URI + user code + token polling) with a PKCE
    # code-binding extension that doesn't change the operator UX.
    {"id": "minimax-oauth", "name": "MiniMax (OAuth)", "flow": "device_code",
     "cli_command": "hermes auth add minimax-oauth", "docs_url": "https://www.minimax.io", "status_fn": None},
    # Device code works in remote shells/containers without a reachable 127.0.0.1 callback.
    {"id": "xai-oauth", "name": "xAI Grok OAuth (SuperGrok / Premium+)", "flow": "device_code",
     "cli_command": "hermes auth add xai-oauth",
     "docs_url": "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth", "status_fn": None},
    # `copilot login` is the non-interactive subcommand; `copilot /login` is not valid
    # (slash-commands only exist inside an interactive session).
    {"id": "copilot-acp", "name": "GitHub Copilot (ACP)", "flow": "external", "cli_command": "copilot login",
     "docs_url": "https://docs.github.com/en/copilot", "status_fn": _copilot_acp_status},
    # Anthropic / Claude entries sit at the bottom. Deliberately flow == "external": an
    # in-dashboard Connect button would let a scriptable HTTP endpoint mint Claude Pro/Max
    # subscription tokens outside Anthropic's own client, against its OAuth usage policies.
    # Login works via the terminal (`hermes auth add anthropic`) or a plain API key.
    {"id": "anthropic", "name": "Anthropic API Key", "flow": "external", "cli_command": "hermes auth add anthropic",
     "docs_url": "https://docs.claude.com/en/api/getting-started", "status_fn": _anthropic_oauth_status},
    {"id": "claude-code", "name": "Anthropic OAuth: Required Extra Usage Credits to Use Subscription",
     "flow": "external", "cli_command": "claude setup-token",
     "docs_url": "https://docs.claude.com/en/docs/claude-code", "status_fn": _claude_code_only_status},
)
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _oauth_session_profile(session_id: str, fallback: Optional[str] = None) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _oauth_poller(label: str):
    """Wrap a device-code poller body ``fn(session_id, sess)``: vanished session is a no-op,
    success marks ``approved``, any exception records ``error`` + ``error_message`` on the
    session instead of raising (the thread has no caller; the dashboard reads the status)."""
    def deco(fn):
        @functools.wraps(fn)
        def poller(session_id: str) -> None:
            with _oauth_sessions_lock:
                sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            try:
                fn(session_id, sess)
                with _oauth_sessions_lock:
                    sess["status"] = "approved"
                _log.info("oauth/device: %s login completed (session=%s)", label, session_id)
            except Exception as e:
                _log.warning("%s device-code poll failed (session=%s): %s", label, session_id, e)
                with _oauth_sessions_lock:
                    sess["status"] = "error"
                    sess["error_message"] = str(e)
        return poller
    return deco


@_oauth_poller("nous")
def _nous_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.web_server_profiles import _profile_scope
    from hermes_cli.auth import _poll_for_token, persist_nous_credentials, refresh_nous_oauth_from_state
    import httpx
    portal_base_url, client_id = sess["portal_base_url"], sess["client_id"]
    with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
        token_data = _poll_for_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            device_code=sess["device_code"], expires_in=max(60, int(sess["expires_at"] - time.time())),
            poll_interval=sess["interval"],
        )
    # Same post-processing as _nous_device_code_login (validate/refresh JWT)
    now = datetime.now(timezone.utc)
    token_ttl = int(token_data.get("expires_in") or 0)
    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": token_data.get("inference_base_url"),
        "client_id": client_id,
        "scope": token_data.get("scope") or sess.get("scope"),
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": (
            datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
            if token_ttl else None
        ),
        "expires_in": token_ttl,
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        full_state = refresh_nous_oauth_from_state(auth_state, timeout_seconds=15.0, force_refresh=False)
        persist_nous_credentials(full_state)


@_oauth_poller("minimax")
def _minimax_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """MiniMax poller: PKCE-style ``code_verifier`` + ``user_code`` instead of Nous's
    ``device_code``. Builds the same auth_state as the CLI's ``_minimax_oauth_login`` and persists
    via ``_minimax_save_auth_state`` so the system ends up as after ``hermes auth add minimax-oauth``.
    Region is fixed to "global" here; cn-region operators use the CLI's ``--region cn``."""
    from hermes_cli.web_server_profiles import _profile_scope
    from hermes_cli.auth import (
        _minimax_poll_token, _minimax_resolve_token_expiry_unix, _minimax_save_auth_state,
        MINIMAX_OAUTH_GLOBAL_INFERENCE, MINIMAX_OAUTH_SCOPE,
    )
    import httpx
    portal_base_url, client_id = sess["portal_base_url"], sess["client_id"]
    with httpx.Client(
        timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}, follow_redirects=True
    ) as client:
        token_data = _minimax_poll_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            user_code=sess["user_code"], code_verifier=sess["code_verifier"],
            expired_in=sess["expired_in_raw"], interval_ms=sess.get("interval_ms"),
        )
    now = datetime.now(timezone.utc)
    expires_at_ts = _minimax_resolve_token_expiry_unix(int(token_data["expired_in"]), now=now)
    auth_state = {
        "provider": "minimax-oauth",
        "region": sess.get("region", "global"),
        "portal_base_url": portal_base_url,
        "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
        "client_id": client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at_ts, tz=timezone.utc).isoformat(),
        "expires_in": max(0, int(expires_at_ts - now.timestamp())),
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        _minimax_save_auth_state(auth_state)


@_oauth_poller("xai")
def _xai_device_poller(session_id: str, sess: Dict[str, Any]) -> None:
    """Background poller for xAI's OAuth device-code flow."""
    from hermes_cli.web_server_profiles import _profile_scope
    import httpx
    from hermes_cli.auth import (
        _save_xai_oauth_tokens, _xai_oauth_discovery, _xai_oauth_poll_device_token,
        mark_provider_active_if_unset, unsuppress_credential_source,
    )

    discovery = _xai_oauth_discovery(20.0)
    with httpx.Client(timeout=httpx.Timeout(20.0), headers={"Accept": "application/json"}) as client:
        token_data = _xai_oauth_poll_device_token(
            client, token_endpoint=discovery["token_endpoint"], device_code=sess["device_code"],
            expires_in=max(60, int(sess["expires_at"] - time.time())), poll_interval=int(sess["interval"]),
        )
    tokens = {
        "access_token": str(token_data.get("access_token", "") or "").strip(),
        "refresh_token": str(token_data.get("refresh_token", "") or "").strip(),
        "id_token": str(token_data.get("id_token", "") or "").strip(),
        "expires_in": token_data.get("expires_in"),
        "token_type": str(token_data.get("token_type") or "Bearer").strip() or "Bearer",
    }
    with _profile_scope(_oauth_session_profile(session_id)):
        # set_active=False: persist without hijacking an existing active chat provider.
        _save_xai_oauth_tokens(
            tokens, discovery=discovery, auth_mode="oauth_device_code", set_active=False,
            last_refresh=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        # Mirror `hermes auth add xai-oauth`: first credential may become active; never overwrite.
        mark_provider_active_if_unset("xai-oauth")
        # The singleton write is the source of truth (the pool load seeds it as the canonical
        # ``device_code`` entry). Do NOT add a parallel ``manual:dashboard_*`` pool entry — it
        # duplicates the single-use refresh token and triggers ``refresh_token_reused`` churn.
        # An interactive login is an explicit re-enable, so clear any prior suppression.
        unsuppress_credential_source("xai-oauth", "device_code")
