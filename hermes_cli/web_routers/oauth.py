"""OAuth provider dashboard routes: catalog/status, disconnect, and in-browser device-code login flows.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are late-bound (cycle-safe).
"""

import asyncio
import logging
import os
import secrets
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_oauth import (
    _external_process_cli_command, _minimax_poller, _nous_poller, _oauth_profile_name, _oauth_sessions, _oauth_sessions_lock, _truncate_token, _xai_device_poller,
)
from hermes_cli.web_models import OAuthSubmitBody
from hermes_cli.web_routers._common import scoped_to_thread

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
_require_token = late("_require_token")
_resolve_profile_dir = late("_resolve_profile_dir", "hermes_cli.web_server_profiles")
_OAUTH_PROVIDER_CATALOG = LateState("_OAUTH_PROVIDER_CATALOG", "hermes_cli.web_server_oauth")

_CODEX_ISSUER = "https://auth.openai.com"
_JSON_HEADERS = {"Content-Type": "application/json"}
_CODEX_EXPIRES_IN = 15 * 60  # OpenAI's effective limit
_CANCELLED: Any = object()
_OAUTH_SESSION_TTL_SECONDS = 15 * 60


def _http_response_error_detail(resp: Any) -> str:
    """Best-effort extraction of a short provider error detail."""
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts = [s for s in (str(error.get(k, "")).strip() for k in ("message", "error_description", "code", "type")) if s]
            if parts:
                return ": ".join(parts)
        for value in (error, *(payload.get(k) for k in ("detail", "message", "error_description"))):
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(getattr(resp, "text", "") or "").strip()[:500]


def _codex_device_code_start_error(resp: Any) -> str:
    """Dashboard-facing OpenAI Codex device-code start failure."""
    status = getattr(resp, "status_code", "unknown")
    detail = _http_response_error_detail(resp)
    lower = detail.lower()
    if "device" in lower and ("authori" in lower or "enable" in lower):
        message = (
            "OpenAI rejected the device-code login request. Your OpenAI "
            "account may need device-code authorization enabled before Hermes "
            "can start this dashboard login. Enable device-code authorization "
            "in OpenAI, then return here and click Login again."
        )
    else:
        message = (
            "OpenAI rejected the device-code login request. Please try Login "
            "again from the dashboard after checking your OpenAI account settings."
        )
    return f"{message} (HTTP {status}: {detail})" if detail else f"{message} (HTTP {status})"


def _new_oauth_session(provider_id: str, flow: str, profile: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    sess = {
        "session_id": sid, "provider": provider_id, "flow": flow, "profile": _oauth_profile_name(profile),
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _start_poller(target, sid: str, prefix: str = "oauth-poll") -> None:
    threading.Thread(target=target, args=(sid,), daemon=True, name=f"{prefix}-{sid[:6]}").start()


def _device_session_started(
    provider_id: str, profile: Optional[str], poller, fields: Dict[str, Any],
    user_code, verification_url, expires_in: int, poll_interval: int,
) -> Dict[str, Any]:
    """Register a device-code session carrying ``fields``, start its poller, return the /start body."""
    sid, sess = _new_oauth_session(provider_id, "device_code", profile=profile)
    sess.update(fields)
    _start_poller(poller, sid)
    return {
        "session_id": sid, "flow": "device_code", "user_code": user_code,
        "verification_url": verification_url, "expires_in": expires_in, "poll_interval": poll_interval,
    }


async def _httpx_call(fn: Callable[[Any], Any], timeout: float = 15.0, **client_kwargs) -> Any:
    """Run ``fn(client)`` off-loop with a short-lived JSON-accepting ``httpx.Client``."""
    import httpx

    def _call():
        with httpx.Client(
            timeout=httpx.Timeout(timeout), headers={"Accept": "application/json"}, **client_kwargs
        ) as client:
            return fn(client)

    return await asyncio.get_running_loop().run_in_executor(None, _call)


# OpenAI Codex device-code worker. Codex's own deviceauth/usercode (returns
# device_auth_id) + deviceauth/token (polled until 200) endpoints yield
# authorization_code + code_verifier exchanged at CODEX_OAUTH_TOKEN_URL. Replicated
# here rather than calling ``_codex_device_code_login``, which prints/blocks/polls in
# one function — the dashboard needs the user_code before polling completes.


def _codex_cancelled(sess: Dict[str, Any], session_id: str, stage: str = "") -> bool:
    if not sess.get("cancelled"):
        return False
    _log.info("oauth/device: openai-codex login cancelled%s (session=%s)", stage, session_id)
    return True


def _codex_request_user_code(httpx) -> Dict[str, Any]:
    """Step 1: request device code; returns device_data with ``interval`` clamped (>= 3s)."""
    from hermes_cli.auth import CODEX_OAUTH_CLIENT_ID

    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        resp = client.post(
            f"{_CODEX_ISSUER}/api/accounts/deviceauth/usercode", json={"client_id": CODEX_OAUTH_CLIENT_ID},
            headers=_JSON_HEADERS,
        )
    if resp.status_code != 200:
        raise RuntimeError(_codex_device_code_start_error(resp))
    device_data = resp.json()
    device_data["interval"] = max(3, int(device_data.get("interval", "5")))
    if not device_data.get("user_code") or not device_data.get("device_auth_id"):
        raise RuntimeError("device-code response missing user_code or device_auth_id")
    return device_data


def _codex_poll_authorization(httpx, sess: Dict[str, Any], session_id: str) -> Any:
    """Step 2: poll until authorized. ``None`` = expired; ``_CANCELLED`` = user cancelled."""
    deadline = time.monotonic() + sess["expires_in"]
    payload = {"device_auth_id": sess["device_auth_id"], "user_code": sess["user_code"]}
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        while time.monotonic() < deadline:
            if _codex_cancelled(sess, session_id):
                return _CANCELLED
            time.sleep(sess["interval"])
            if _codex_cancelled(sess, session_id):
                return _CANCELLED
            poll = client.post(f"{_CODEX_ISSUER}/api/accounts/deviceauth/token", json=payload, headers=_JSON_HEADERS)
            if poll.status_code == 200:
                return poll.json()
            if poll.status_code in {403, 404}:
                continue  # user hasn't authorized yet
            raise RuntimeError(f"deviceauth/token poll returned {poll.status_code}")
    return None


def _codex_exchange_tokens(httpx, code_resp: Dict[str, Any]) -> Dict[str, str]:
    """Step 3: exchange authorization_code for tokens."""
    from hermes_cli.auth import CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL

    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    if not authorization_code or not code_verifier:
        raise RuntimeError("device-auth response missing authorization_code/code_verifier")
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        token_resp = client.post(
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code", "code": authorization_code,
                "redirect_uri": f"{_CODEX_ISSUER}/deviceauth/callback",
                "client_id": CODEX_OAUTH_CLIENT_ID, "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if token_resp.status_code != 200:
        raise RuntimeError(f"token exchange returned {token_resp.status_code}")
    tokens = token_resp.json()
    if not tokens.get("access_token"):
        raise RuntimeError("token exchange did not return access_token")
    return {"access_token": tokens.get("access_token", ""), "refresh_token": tokens.get("refresh_token", "")}


def _codex_full_login_worker(session_id: str) -> None:
    """Run the complete OpenAI Codex device-code flow (see comment above)."""
    try:
        import httpx

        device_data = _codex_request_user_code(httpx)
        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            sess.update(
                user_code=device_data["user_code"], verification_url=f"{_CODEX_ISSUER}/codex/device",
                device_auth_id=device_data["device_auth_id"], interval=device_data["interval"],
                expires_in=_CODEX_EXPIRES_IN, expires_at=time.time() + _CODEX_EXPIRES_IN,
            )
            # Captured now (not re-derived after cancel pops the session) so a
            # cancelled session can never fall back to the caller's current
            # profile scope at save time.
            session_profile = sess.get("profile")

        code_resp = _codex_poll_authorization(httpx, sess, session_id)
        if code_resp is None:
            with _oauth_sessions_lock:
                sess["status"] = "expired"
                sess["error_message"] = "Device code expired before approval"
            return
        if code_resp is _CANCELLED or _codex_cancelled(sess, session_id, " before token exchange"):
            return

        tokens = _codex_exchange_tokens(httpx, code_resp)
        from hermes_cli.auth import _save_codex_tokens

        # The cancellation check and the save are one atomic critical section
        # under the lock cancel_oauth_session() uses; otherwise DELETE could
        # flip "cancelled" between the check and the save and tokens would be
        # persisted after the user believed the login was aborted.
        with _oauth_sessions_lock:
            if _codex_cancelled(sess, session_id, " before token save"):
                return
            with _profile_scope(session_profile):
                _save_codex_tokens(tokens)
            sess["status"] = "approved"
        _log.info("oauth/device: openai-codex login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("codex device-code worker failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s:
                s["status"] = "error"
                s["error_message"] = str(e)


_OMIT: Any = object()


def _status_card(
    raw: dict, source, source_label, token_preview, expires_at, has_refresh_token, last_refresh=_OMIT
) -> Dict[str, Any]:
    card = {
        "logged_in": bool(raw.get("logged_in")), "source": source, "source_label": source_label,
        "token_preview": token_preview, "expires_at": expires_at, "has_refresh_token": has_refresh_token,
    }
    if last_refresh is not _OMIT:
        card["last_refresh"] = last_refresh
    return card


# Hand-written status cards per provider id: (hauth getter name, raw -> card).
# Providers absent here fall through to the slug-driven ``get_auth_status``.
# nous: refresh-free local snapshot so listing providers never performs an OAuth
# refresh. xai: source_label is a human-readable origin (auth-store path /
# credential source), not the internal auth_mode string ("oauth_pkce").
_PROVIDER_STATUS: Dict[str, tuple[str, Callable[[dict], dict]]] = {
    "nous": ("get_nous_auth_status_local", lambda r: _status_card(
        r, "nous_portal", r.get("portal_base_url") or "Nous Portal",
        _truncate_token(r.get("access_token")), r.get("access_expires_at"), bool(r.get("has_refresh_token")),
    )),
    "openai-codex": ("get_codex_auth_status", lambda r: _status_card(
        r, r.get("source") or "openai_codex", r.get("auth_mode") or "OpenAI Codex",
        _truncate_token(r.get("api_key")), None, False, r.get("last_refresh"),
    )),
    "qwen-oauth": ("get_qwen_auth_status", lambda r: _status_card(
        r, "qwen_cli", r.get("auth_store_path") or "Qwen CLI",
        _truncate_token(r.get("access_token")), r.get("expires_at"), bool(r.get("has_refresh_token")),
    )),
    "minimax-oauth": ("get_minimax_oauth_auth_status", lambda r: _status_card(
        r, "minimax_oauth", f"MiniMax ({r.get('region', 'global')})", None, r.get("expires_at"), True,
    )),
    "xai-oauth": ("get_xai_oauth_auth_status", lambda r: _status_card(
        r, r.get("source") or "xai_oauth", r.get("auth_store") or r.get("source") or "xAI Grok OAuth",
        _truncate_token(r.get("api_key")), None, True, r.get("last_refresh"),
    )),
}


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    try:
        if status_fn is not None:
            return status_fn()
        from hermes_cli import auth as hauth
        entry = _PROVIDER_STATUS.get(provider_id)
        if entry is not None:
            getter, shape = entry
            return shape(getattr(hauth, getter)())
        # Catalog-derived providers (status_fn=None, no hand-written card) still
        # reflect real login state via the canonical slug-driven dispatcher, so
        # a new OAuth/account provider plugin never renders permanently logged-out.
        raw = hauth.get_auth_status(provider_id)
        if isinstance(raw, dict) and "logged_in" in raw:
            return _status_card(
                raw,
                raw.get("source") or raw.get("provider") or provider_id,
                raw.get("source_label") or raw.get("auth_store") or raw.get("auth_store_path")
                or raw.get("base_url") or raw.get("name") or "",
                _truncate_token(raw.get("access_token") or raw.get("api_key")),
                raw.get("expires_at") or raw.get("access_expires_at"),
                bool(raw.get("has_refresh_token")),
            )
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


async def _start_nous_device_code(profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli.auth import PROVIDER_REGISTRY, _request_device_code
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL") or pconfig.portal_base_url
    ).rstrip("/")
    device_data = await _httpx_call(lambda client: _request_device_code(
        client=client, portal_base_url=portal_base_url, client_id=pconfig.client_id, scope=pconfig.scope,
    ))
    return _device_session_started(
        "nous", profile, _nous_poller,
        dict(
            device_code=str(device_data["device_code"]), interval=int(device_data["interval"]),
            expires_at=time.time() + int(device_data["expires_in"]), portal_base_url=portal_base_url,
            client_id=pconfig.client_id, scope=pconfig.scope,
        ),
        str(device_data["user_code"]), str(device_data["verification_uri_complete"]),
        int(device_data["expires_in"]), int(device_data["interval"]),
    )


async def _start_codex_device_code(profile: Optional[str]) -> Dict[str, Any]:
    # The full Codex helper polls inline, so it runs in a worker thread and
    # proxies user_code + verification_url back via the session dict; block
    # briefly until the worker has populated the user_code, OR errored.
    sid, _ = _new_oauth_session("openai-codex", "device_code", profile=profile)
    _start_poller(_codex_full_login_worker, sid, prefix="oauth-codex")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid)
        if s and (s.get("user_code") or s["status"] != "pending"):
            break
        await asyncio.sleep(0.1)
    with _oauth_sessions_lock:
        s = _oauth_sessions.get(sid, {})
    if s.get("status") == "error":
        raise HTTPException(status_code=500, detail=s.get("error_message") or "device-auth failed")
    if not s.get("user_code"):
        raise HTTPException(status_code=504, detail="device-auth timed out before returning a user code")
    return {
        "session_id": sid, "flow": "device_code", "user_code": s["user_code"], "verification_url": s["verification_url"],
        "expires_in": int(s.get("expires_in") or 900), "poll_interval": int(s.get("interval") or 5),
    }


async def _start_minimax_device_code(profile: Optional[str]) -> Dict[str, Any]:
    # Device-code flow with a PKCE extension: verifier + challenge from
    # _minimax_pkce_pair bind the token exchange to the original session.
    from hermes_cli.auth import (
        MINIMAX_OAUTH_CLIENT_ID, MINIMAX_OAUTH_GLOBAL_BASE, _minimax_pkce_pair, _minimax_request_user_code,
    )
    verifier, challenge, state = _minimax_pkce_pair()
    portal_base_url = (os.getenv("MINIMAX_PORTAL_BASE_URL") or MINIMAX_OAUTH_GLOBAL_BASE).rstrip("/")
    device_data = await _httpx_call(lambda client: _minimax_request_user_code(
        client=client, portal_base_url=portal_base_url, client_id=MINIMAX_OAUTH_CLIENT_ID,
        code_challenge=challenge, state=state,
    ), follow_redirects=True)
    # MiniMax's `interval` is in milliseconds (defensive default 2000ms in _minimax_poll_token).
    interval_raw = device_data.get("interval")
    interval_ms = int(interval_raw) if interval_raw is not None else None
    # `expired_in` is overloaded — a unix-ms timestamp OR seconds-from-now.
    # Mirror the heuristic in _minimax_poll_token; keep the raw value for the
    # poller and derive expires_at + UI-friendly expires_in seconds.
    expired_in_raw = int(device_data["expired_in"])
    if expired_in_raw > 1_000_000_000_000:  # likely unix-ms
        expires_at_ts = expired_in_raw / 1000.0
        expires_in_seconds = max(0, int(expires_at_ts - time.time()))
    else:
        expires_at_ts = time.time() + expired_in_raw
        expires_in_seconds = expired_in_raw
    return _device_session_started(
        "minimax-oauth", profile, _minimax_poller,
        dict(
            interval_ms=interval_ms, user_code=str(device_data["user_code"]), code_verifier=verifier, state=state,
            portal_base_url=portal_base_url, client_id=MINIMAX_OAUTH_CLIENT_ID, region="global",
            expired_in_raw=expired_in_raw, expires_at=expires_at_ts,
        ),
        str(device_data["user_code"]), str(device_data["verification_uri"]), expires_in_seconds,
        max(2, (interval_ms or 2000) // 1000),
    )


async def _start_xai_device_code(profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli.auth import _xai_oauth_request_device_code
    device_data = await _httpx_call(_xai_oauth_request_device_code, timeout=20.0)
    return _device_session_started(
        "xai-oauth", profile, _xai_device_poller,
        dict(
            device_code=str(device_data["device_code"]), interval=int(device_data["interval"]),
            expires_at=time.time() + int(device_data["expires_in"]),
        ),
        str(device_data["user_code"]),
        str(device_data.get("verification_uri_complete") or device_data["verification_uri"]),
        int(device_data["expires_in"]), int(device_data["interval"]),
    )


_DEVICE_CODE_STARTERS = {
    "nous": _start_nous_device_code, "openai-codex": _start_codex_device_code,
    "minimax-oauth": _start_minimax_device_code, "xai-oauth": _start_xai_device_code,
}


async def _start_device_code_flow(provider_id: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Hit the provider's device-auth endpoint, spawn its poller, return the display fields."""
    starter = _DEVICE_CODE_STARTERS.get(provider_id)
    if starter is None:
        raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")
    return await starter(profile)


def _oauth_provider_disconnect_command(provider: Dict[str, Any]) -> Optional[str]:
    """Shell command that clears an external provider's credentials, or None.

    The disconnect API never silently deletes files another CLI owns; the GUI runs
    this in its embedded terminal so the user sees exactly what executes. Claude Code
    has no scriptable logout, so remove what logout would: the macOS Keychain entry
    and/or ``~/.claude/.credentials.json`` (the two ``read_claude_code_credentials()`` sources).
    """
    if provider.get("flow") != "external" or provider.get("id") != "claude-code":
        return None
    rm_file = "rm -f ~/.claude/.credentials.json"
    if sys.platform == "darwin":
        return f'security delete-generic-password -s "Claude Code-credentials" 2>/dev/null; {rm_file}'
    return rm_file


def _oauth_provider_disconnect_hint(provider: Dict[str, Any], status: Dict[str, Any]) -> Optional[str]:
    """Return the manual disconnect path when the API cannot clear this provider."""
    # "anthropic" is flow == "external" (no in-dashboard login) but Hermes still
    # OWNS its credential (the PKCE file ~/.hermes/.anthropic_oauth.json and its
    # credential-pool entry, written by `hermes auth add anthropic`), so it is
    # excluded from the "external providers can't be auto-disconnected" rule.
    if provider.get("flow") == "external" and provider.get("id") != "anthropic":
        if _oauth_provider_disconnect_command(provider):
            # Fallback wording for surfaces without the one-click "run in terminal" path.
            return "Managed outside Hermes — run the disconnect command to remove it."
        return "Managed by that provider's CLI; remove it there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    """Accounts-tab provider list: ``_OAUTH_PROVIDER_CATALOG`` cards first (curated
    order, win on metadata), then every other accounts-tab ``provider_catalog()`` entry
    in ``hermes model`` order, so plugin-added OAuth/external providers appear automatically."""
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in _OAUTH_PROVIDER_CATALOG:
        if entry["id"] not in seen:
            seen.add(entry["id"])
            rows.append(dict(entry))
    try:
        from hermes_cli.provider_catalog import provider_catalog
        for d in provider_catalog():
            if d.tab != "accounts" or d.slug in seen:
                continue
            seen.add(d.slug)
            rows.append({
                "id": d.slug, "name": d.label, "flow": "external",
                "cli_command": f"hermes auth add {d.slug}", "docs_url": d.signup_url or "", "status_fn": None,
            })
    except Exception:
        pass
    return rows


@router.get("/api/providers/oauth")
async def list_oauth_providers(profile: Optional[str] = None):
    """Every OAuth-capable provider with current status (token_preview is the last
    N chars, never the full token; disconnect_command only for external providers)."""
    def _run():
        providers = []
        for p in _build_oauth_catalog():
            status = _resolve_provider_status(p["id"], p.get("status_fn"))
            disconnect_hint = _oauth_provider_disconnect_hint(p, status)
            providers.append({
                "id": p["id"], "name": p["name"], "flow": p["flow"],
                "cli_command": _external_process_cli_command(p["id"], p["cli_command"]),
                "docs_url": p["docs_url"], "disconnect_hint": disconnect_hint,
                "disconnect_command": _oauth_provider_disconnect_command(p),
                "disconnectable": disconnect_hint is None, "status": status,
            })
        return {"providers": providers}

    return await scoped_to_thread(profile, _run)


def _reject_if_not_disconnectable(provider: Dict[str, Any], status: Dict[str, Any]) -> None:
    disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
    if disconnect_hint:
        raise HTTPException(400, f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}")


def _clear_anthropic_auth() -> bool:
    """Clear only the Hermes-managed PKCE file and auth-store entry (never ~/.claude/*)."""
    cleared = False
    try:
        from agent.anthropic_credentials import _get_hermes_oauth_file
        oauth_file = _get_hermes_oauth_file()
        if oauth_file.exists():
            oauth_file.unlink()
            cleared = True
    except Exception:
        pass
    try:
        from hermes_cli.auth import clear_provider_auth
        cleared = clear_provider_auth("anthropic") or cleared
    except Exception:
        pass
    return cleared


@router.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(provider_id: str, request: Request, profile: Optional[str] = None):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    def _run():
        catalog_by_id = {p["id"]: p for p in _build_oauth_catalog()}
        provider = catalog_by_id.get(provider_id)
        if provider is None:
            raise HTTPException(400, f"Unknown provider: {provider_id}. Available: {', '.join(sorted(catalog_by_id))}")
        # Flow-only rejection first so external providers never reach status resolution.
        _reject_if_not_disconnectable(provider, {})
        _reject_if_not_disconnectable(provider, _resolve_provider_status(provider_id, provider.get("status_fn")))

        if provider_id == "anthropic":
            cleared = _clear_anthropic_auth()
            _log.info("oauth/disconnect: %s", provider_id)
            return {"ok": bool(cleared), "provider": provider_id}
        try:
            from hermes_cli.auth import clear_provider_auth, invalidate_nous_auth_status_cache
            cleared = clear_provider_auth(provider_id)
            if provider_id == "nous":
                invalidate_nous_auth_status_cache()
            _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
            return {"ok": bool(cleared), "provider": provider_id}
        except Exception as e:
            _log.exception("disconnect %s failed", provider_id)
            raise HTTPException(status_code=500, detail=str(e))

    return await scoped_to_thread(profile, _run)


# In-browser device-code sessions: /start spawns a poller thread and returns the
# display fields; the UI polls .../poll/{session_id} until status != "pending" (on
# "approved" the poller has already saved creds). Anthropic has NO dashboard PKCE
# flow — an unattended endpoint minting Claude subscription tokens outside
# Anthropic's own client violates its OAuth usage policy; that card is "external".
# Sessions are in-memory (single-process), expire after 15 min, GC'd on /start.


def _gc_oauth_sessions() -> None:
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        for sid in [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]:
            _oauth_sessions.pop(sid, None)


def _validate_oauth_profile(profile: Optional[str]) -> str:
    """Validate the requested profile (404 via ``_resolve_profile_dir``) and return its name."""
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)
    return profile_name


@router.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(provider_id: str, request: Request, profile: Optional[str] = None):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
    catalog_entry = next((p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id), None)
    if catalog_entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    if catalog_entry["flow"] == "external":
        raise HTTPException(400, f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually")
    try:
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id, profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("oauth/start %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported flow")


@router.post("/api/providers/oauth/{provider_id}/submit")
async def submit_oauth_code(
    provider_id: str, body: OAuthSubmitBody, request: Request, profile: Optional[str] = None,
):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@router.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(provider_id: str, session_id: str, profile: Optional[str] = None):
    """Poll a session's status (no auth — read-only state). One endpoint serves
    every device-code flow: all report progress via the worker-updated ``status``."""
    requested_profile = _validate_oauth_profile(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    if sess.get("profile") != requested_profile:
        raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
    return {
        "session_id": session_id, "status": sess["status"],
        "error_message": sess.get("error_message"), "expires_at": sess.get("expires_at"),
    }


@router.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(session_id: str, request: Request, profile: Optional[str] = None):
    """Cancel a pending OAuth session. Token-protected.

    Marks the session dict ``cancelled`` before popping it so a background
    worker still holding that dict (e.g. the Codex poller) stops
    polling/exchanging/saving instead of completing the login after the user
    believed it was aborted.
    """
    _require_token(request)
    requested_profile = _validate_oauth_profile(profile)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        if sess is not None:
            if sess.get("profile") != requested_profile:
                raise HTTPException(status_code=400, detail="OAuth session profile mismatch")
            sess["cancelled"] = True
            _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    return {"ok": True, "session_id": session_id}
