"""Official ChatGPT plan OAuth, separate from the Codex CLI's public-client grant.

Each pool row is one verified account/client registration. The profile auth store
owns the stable host ID and selected row; tokens never have a second authority.
"""
from __future__ import annotations

import hmac
import secrets
import sys
import time
import uuid
import webbrowser
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode, urlparse

from hermes_cli.auth_constants import AuthError, httpx

PROVIDER = "openai-chatgpt"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = ("openid", "profile", "email", "offline_access", "resource.invoke", DIRECT_SCOPE)
CALLBACK_PATH = "/auth/callback"
_TERMINAL_REFRESH_ERRORS = frozenset({"invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"})


def _error(message: str, code: str, *, relogin: bool = False) -> AuthError:
    return AuthError(message, provider=PROVIDER, code=code, relogin_required=relogin)


def _local_state() -> tuple[dict, list[dict]]:
    from hermes_cli.auth import _load_auth_store
    store = _load_auth_store()
    state = store.get("providers", {}).get(PROVIDER, {})
    rows = list(store.get("credential_pool", {}).get(PROVIDER, []))
    # The pool may prune a long-dead token row. Account/client registrations
    # survive that cleanup without keeping a second copy of any credentials.
    known_ids = {row.get("id") for row in rows}
    rows.extend(row for row in state.get("registrations", []) if row.get("id") not in known_ids)
    return state, rows


def _host_id() -> str:
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    with _auth_store_lock():
        store = _load_auth_store()
        state = store.setdefault("providers", {}).setdefault(PROVIDER, {})
        if not state.get("ext_agent_host_id"):
            state["ext_agent_host_id"] = "urn:uuid:" + str(uuid.uuid4())
            _save_auth_store(store)
        return state["ext_agent_host_id"]


def _discovery() -> dict:
    from hermes_cli.auth import _default_verify
    try:
        response = httpx.get(ISSUER + "/.well-known/openid-configuration", timeout=15,
                             verify=_default_verify())
        response.raise_for_status()
        metadata = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise _error("Could not load OpenAI's sign-in configuration.", "chatgpt_discovery_failed") from exc
    if not isinstance(metadata, dict) or metadata.get("issuer") != ISSUER or not metadata.get("jwks_uri"):
        raise _error("OpenAI sign-in configuration has an unexpected issuer.", "chatgpt_issuer_mismatch")
    origin = urlparse(ISSUER)
    for key in ("jwks_uri", "revocation_endpoint"):
        if metadata.get(key):
            endpoint = urlparse(metadata[key])
            if (endpoint.scheme, endpoint.netloc) != (origin.scheme, origin.netloc):
                raise _error("OpenAI sign-in configuration has an unexpected endpoint.", "chatgpt_endpoint_mismatch")
    return metadata


@lru_cache(maxsize=4)
def _jwks_client(uri: str):
    import jwt
    return jwt.PyJWKClient(uri, cache_jwk_set=True, lifespan=300, timeout=15)


def _validate_identity(id_token: str, client_id: str, *, nonce: str | None,
                       subject: str | None = None) -> dict:
    import jwt
    metadata = _discovery()
    try:
        signing_key = _jwks_client(metadata["jwks_uri"]).get_signing_key_from_jwt(id_token)
        identity = jwt.decode(id_token, signing_key.key, algorithms=["RS256"], issuer=ISSUER,
                              audience=client_id, leeway=5,
                              options={"require": ["iss", "aud", "sub", "exp", "iat"]})
    except jwt.PyJWKClientConnectionError as exc:
        raise _error("Could not load OpenAI's identity verification keys.", "chatgpt_discovery_failed") from exc
    except (jwt.PyJWTError, ValueError) as exc:
        raise _error("OpenAI's ID token could not be verified.", "chatgpt_invalid_identity") from exc
    if nonce is not None and not hmac.compare_digest(str(identity.get("nonce") or ""), nonce):
        raise _error("OpenAI's ID token did not match this sign-in attempt.", "chatgpt_nonce_mismatch")
    if not isinstance(identity.get("sub"), str) or not identity["sub"]:
        raise _error("OpenAI's ID token did not include an account identity.", "chatgpt_missing_subject")
    if subject is not None and identity["sub"] != subject:
        raise _error("The signed-in ChatGPT account differs from the selected registration.",
                     "chatgpt_account_mismatch")
    return identity


def _post_token(form: dict[str, str]) -> dict:
    from hermes_cli.auth import _default_verify
    try:
        response = httpx.post(TOKEN_URL, data=form, headers={"Accept": "application/json"},
                              timeout=20, verify=_default_verify())
    except httpx.HTTPError as exc:
        raise _error("ChatGPT token exchange could not reach OpenAI. Try again later.",
                     "chatgpt_token_transport") from exc
    if response.status_code != 200:
        try:
            payload = response.json()
            code = payload.get("error") if isinstance(payload, dict) else None
        except ValueError:
            code = None
        code = code if isinstance(code, str) else "chatgpt_token_exchange_failed"
        raise _error(f"ChatGPT token exchange failed (HTTP {response.status_code}, {code}).", code,
                     relogin=form["grant_type"] == "refresh_token" and code in _TERMINAL_REFRESH_ERRORS)
    try:
        payload = response.json()
    except ValueError as exc:
        raise _error("OpenAI returned an invalid token response.", "chatgpt_token_response_invalid") from exc
    if not isinstance(payload, dict):
        raise _error("OpenAI returned an invalid token response.", "chatgpt_token_response_invalid")
    return payload


def _token_fields(payload: dict, metadata: dict, *, nonce: str | None = None,
                  initial: bool = False, received_at_ms: int | None = None) -> dict:
    from hermes_cli.auth import _utc_now_z
    if (not isinstance(payload.get("access_token"), str) or not payload["access_token"]
            or str(payload.get("token_type", "")).lower() != "bearer"):
        raise _error("OpenAI did not return a usable bearer token.", "chatgpt_token_response_invalid")
    try:
        ttl = int(payload["expires_in"])
    except (KeyError, ValueError, TypeError) as exc:
        raise _error("OpenAI did not return token expiry information.", "chatgpt_token_response_invalid") from exc
    scope = payload.get("scope")
    if not initial and scope is None:
        # OAuth refresh may omit an unchanged scope; never infer requested grants.
        scope = " ".join(metadata.get("scopes", []))
    if ttl <= 0 or not isinstance(scope, str):
        raise _error("OpenAI did not return token expiry and granted scopes.", "chatgpt_token_response_invalid")
    updated = {**metadata, "scopes": scope.split(), "token_type": "Bearer",
               "earliest_refresh_at": payload.get("earliest_refresh_at")}
    updated.pop("pending_refresh", None)
    if "offline_access" in updated["scopes"] and not payload.get("refresh_token"):
        raise _error("OpenAI did not return the renewable session token.", "chatgpt_token_response_invalid")
    id_token = payload.get("id_token")
    if initial or id_token:
        if not isinstance(id_token, str) or not id_token:
            raise _error("OpenAI did not return an ID token.", "chatgpt_missing_identity")
        identity = _validate_identity(id_token, metadata["client_id"], nonce=nonce,
                                      subject=metadata.get("subject"))
        updated.update(issuer=identity["iss"], subject=identity["sub"], email=identity.get("email", ""),
                       id_token=id_token)
    received_at_ms = int(time.time() * 1000) if received_at_ms is None else received_at_ms
    return {"access_token": payload["access_token"], "refresh_token": payload.get("refresh_token"),
            "expires_at_ms": received_at_ms + ttl * 1000, "last_refresh": _utc_now_z(),
            "chatgpt": updated}


def login(args: Any, registration: dict | None = None, *, _retry: bool = False) -> dict:
    """Validate a pending registration before returning any credentials to the caller."""
    from hermes_cli.auth_device_flow import (
        _bind_loopback_callback_server, _can_open_graphical_browser, _make_loopback_callback_handler,
        _pkce_code_challenge, _pkce_code_verifier, _print_loopback_ssh_hint, _serve_loopback_callback)
    saved = dict((registration or {}).get("chatgpt") or {})
    host_id = _host_id()
    client_id = saved.get("client_id") or "dynamic_agent_client"
    handler, result = _make_loopback_callback_handler(CALLBACK_PATH, display_name="ChatGPT",
                                                       extra_fields=("client_id",))
    err = lambda message, code: _error(message, code)
    server = _bind_loopback_callback_server("127.0.0.1", 0, handler, err=err,
                                            bind_failed_code="chatgpt_callback_bind_failed")
    redirect = f"http://127.0.0.1:{server.server_address[1]}{CALLBACK_PATH}"
    verifier, state, nonce = _pkce_code_verifier(), secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    params = {"client_id": client_id, "ext_agent_host_id": host_id, "redirect_uri": redirect,
              "scope": " ".join(SCOPES), "resource": RESOURCE, "response_type": "code",
              "state": state, "nonce": nonce, "code_challenge": _pkce_code_challenge(verifier),
              "code_challenge_method": "S256"}
    if client_id == "dynamic_agent_client":
        params["agent_name_hint"] = "Hermes Agent"
    else:
        if saved.get("email"):
            params["login_hint"] = saved["email"]
        if saved.get("id_token"):
            params["id_token_hint"] = saved["id_token"]
        if DIRECT_SCOPE not in saved.get("scopes", []):
            params["prompt"] = "consent"
    browser_url = AUTHORIZE_URL + "?" + urlencode(params)
    # The copyable fallback intentionally omits the retained ID token, including --no-browser.
    printable = AUTHORIZE_URL + "?" + urlencode({k: v for k, v in params.items() if k != "id_token_hint"})
    print("\nContinue with ChatGPT — authorize Hermes Agent to use your ChatGPT plan.")
    print(f"Open this URL to continue:\n  {printable}\n")
    _print_loopback_ssh_hint(redirect)
    if not getattr(args, "no_browser", False) and _can_open_graphical_browser():
        try:
            webbrowser.open(browser_url)
        except (OSError, webbrowser.Error):
            print("Could not open the browser; use the URL above.")
    try:
        callback = _serve_loopback_callback(server, result, timeout_seconds=float(getattr(args, "timeout", None) or 300),
                                             err=err, timeout_code="chatgpt_callback_timeout")
    except KeyboardInterrupt:
        raise _error("ChatGPT sign-in was cancelled.", "chatgpt_sign_in_cancelled") from None
    if not hmac.compare_digest(str(callback.get("state") or ""), state):
        raise _error("The callback did not match this ChatGPT sign-in.", "chatgpt_state_mismatch")
    if callback.get("error"):
        raise _error("ChatGPT authorization was declined or could not complete.", "chatgpt_authorization_denied")
    issued = callback.get("client_id") or (client_id if saved else "")
    if not issued or issued == "dynamic_agent_client":
        raise _error("ChatGPT registration did not return an issued client ID.", "chatgpt_registration_incomplete")
    if saved and issued != client_id:
        raise _error("The callback changed the selected ChatGPT client registration.", "chatgpt_client_mismatch")
    if not callback.get("code"):
        raise _error("ChatGPT did not return an authorization code.", "chatgpt_code_missing")
    metadata = {**saved, "client_id": issued, "ext_agent_host_id": host_id}
    try:
        payload = _post_token({"grant_type": "authorization_code", "client_id": issued,
                               "code": callback["code"], "code_verifier": verifier,
                               "redirect_uri": redirect, "resource": RESOURCE})
    except AuthError as exc:
        if exc.code == "invalid_grant" and not _retry:
            # The registration exists even when its first code expired. Reuse it
            # with fresh PKCE/state/nonce, but do not save an unverified identity.
            print("The authorization code expired. Restarting sign-in for this registration.")
            return login(args, {"chatgpt": metadata}, _retry=True)
        raise
    return _token_fields(payload, metadata, nonce=nonce, initial=True)


def refresh_credential(entry: Any) -> dict:
    """The credential pool owns the cross-process read/refresh/write transaction."""
    metadata = dict(entry.extra.get("chatgpt") or {})
    client_id = metadata.get("client_id")
    if not client_id or client_id == "dynamic_agent_client" or not metadata.get("subject"):
        raise _error("ChatGPT registration is incomplete; sign in again.", "chatgpt_registration_incomplete", relogin=True)
    pending = metadata.get("pending_refresh")
    if pending is None:
        payload = _post_token({"grant_type": "refresh_token", "client_id": client_id,
                               "refresh_token": entry.refresh_token, "resource": RESOURCE})
        pending = {"response": payload, "received_at_ms": int(time.time() * 1000)}
    try:
        return _token_fields(pending["response"], metadata, received_at_ms=pending["received_at_ms"])
    except AuthError as exc:
        if exc.code != "chatgpt_discovery_failed":
            # The old refresh token is already spent; a permanently invalid replacement
            # cannot safely fund inference or recover through another refresh POST.
            raise _error(str(exc), exc.code, relogin=True) from exc
        # Persist through the pool's existing locked refresh transaction. Keep the old
        # verified identity, but mark it expired until this same response is verified.
        return {"access_token": entry.access_token,
                "refresh_token": pending["response"].get("refresh_token"), "expires_at_ms": 0,
                "chatgpt": {**metadata, "pending_refresh": pending}}


def clear_credential(entry: Any) -> dict:
    """Drop a terminal session without losing its account/client registration."""
    metadata = dict(entry.extra.get("chatgpt") or {})
    metadata.pop("id_token", None)
    metadata.pop("earliest_refresh_at", None)
    metadata.pop("pending_refresh", None)
    return {"access_token": "", "refresh_token": None, "expires_at_ms": None, "chatgpt": metadata}


def credential_is_eligible(entry: Any) -> bool:
    """Only the explicitly selected, authorized account may fund inference."""
    state, rows = _local_state()
    info = entry.extra.get("chatgpt") or {}
    current = next((row for row in rows if row.get("id") == entry.id), {})
    return (entry.id == state.get("active_credential_id")
            and DIRECT_SCOPE in info.get("scopes", [])
            and DIRECT_SCOPE in current.get("chatgpt", {}).get("scopes", [])
            and bool(entry.access_token) and bool(current.get("access_token")))


def assert_active_access_token(access_token: str) -> None:
    """Reject cached clients after logout, account selection or token replacement."""
    state, rows = _local_state()
    current = next((row for row in rows if row.get("id") == state.get("active_credential_id")), {})
    stored = current.get("access_token")
    if (not isinstance(access_token, str) or not access_token or not isinstance(stored, str)
            or not hmac.compare_digest(access_token.encode(), stored.encode())
            or current.get("chatgpt", {}).get("pending_refresh") is not None
            or DIRECT_SCOPE not in current.get("chatgpt", {}).get("scopes", [])):
        raise _error("The selected ChatGPT account or session changed. Reinitialize this session before sending another request.",
                     "chatgpt_session_changed")


def _select_registration(args: Any, *, adding: bool) -> dict | None:
    state, rows = _local_state()
    label = str(getattr(args, "label", None) or "").strip()
    if label:
        matches = [row for row in rows if row.get("label") == label]
        if len(matches) > 1:
            raise _error("The ChatGPT account label is ambiguous.", "chatgpt_account_ambiguous")
        return matches[0] if matches else None
    if not rows:
        return None
    if len(rows) == 1 and not adding:
        return rows[0]
    if not sys.stdin.isatty():
        if adding:
            raise _error("Choose an existing account with --label, or supply a new label to add one.",
                         "chatgpt_account_selection_required")
        return next((row for row in rows if row["id"] == state.get("active_credential_id")), None)
    from hermes_cli.curses_ui import curses_radiolist
    choices = [f"{row['label']} ({row.get('chatgpt', {}).get('email', '')})" for row in rows]
    if adding:
        choices.append("Add another ChatGPT account or workspace")
    selected = curses_radiolist("Choose a ChatGPT account", choices, cancel_returns=-1)
    if selected < 0:
        raise _error("ChatGPT account selection was cancelled.", "chatgpt_sign_in_cancelled")
    return rows[selected] if selected < len(rows) else None


def _save_login(fields: dict, registration: dict | None, label: str) -> None:
    from agent.credential_pool import PooledCredential
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    with _auth_store_lock():
        store = _load_auth_store()
        state = store.setdefault("providers", {}).setdefault(PROVIDER, {})
        rows = store.setdefault("credential_pool", {}).setdefault(PROVIDER, [])
        existing = next((row for row in rows if registration and row["id"] == registration["id"]), None)
        identity = fields.pop("chatgpt")
        entry_id = (existing or registration or {}).get("id") or uuid.uuid4().hex[:8]
        entry = PooledCredential(provider=PROVIDER, id=entry_id,
            label=label or (existing or registration or {}).get("label") or f"{identity.get('email') or 'ChatGPT'} / {identity['client_id']}",
            auth_type="oauth", priority=(existing or {}).get("priority", len(rows)), source="manual:chatgpt",
            base_url=RESOURCE, extra={"chatgpt": identity}, **fields)
        rows[:] = [entry.to_dict() if row is existing else row for row in rows]
        if existing is None:
            rows.append(entry.to_dict())
        registrations = state.setdefault("registrations", [])
        registrations[:] = [row for row in registrations if row.get("id") != entry.id]
        registrations.append({"id": entry.id, "label": entry.label, "chatgpt": {
            key: identity[key] for key in ("client_id", "subject", "issuer", "email", "ext_agent_host_id")
            if key in identity}})
        state["active_credential_id"] = entry.id
        _save_auth_store(store)


def _revoke(registration: dict) -> bool:
    from hermes_cli.auth import _default_verify
    refresh = registration.get("refresh_token")
    if not refresh:
        return True
    try:
        endpoint = _discovery().get("revocation_endpoint")
        if not endpoint:
            return False
    except AuthError:
        return False
    for attempt in range(2):
        try:
            response = httpx.post(endpoint, data={"token": refresh, "token_type_hint": "refresh_token",
                "client_id": registration["chatgpt"]["client_id"]}, timeout=15, verify=_default_verify())
            if response.status_code == 200:
                return True
            if response.status_code < 500:
                return False
        except httpx.HTTPError:
            pass
        if not attempt:
            time.sleep(0.2)
    return False


def _logout(registration: dict) -> None:
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    # Keep refresh and revocation serialized with a running pool's token rotation.
    with _auth_store_lock(timeout_seconds=40):
        store = _load_auth_store()
        rows = store.get("credential_pool", {}).get(PROVIDER, [])
        current = next((row for row in rows if row["id"] == registration["id"]), None)
        if current is None:
            return
        state = store.get("providers", {}).get(PROVIDER, {})
        if state.get("active_credential_id") == current["id"]:
            state.pop("active_credential_id", None)
            # Stop new requests in other processes while revocation is in flight.
            _save_auth_store(store)
        confirmed = _revoke(current)
        current.update(access_token="", refresh_token=None, expires_at_ms=None)
        current["chatgpt"].pop("id_token", None)
        current["chatgpt"].pop("earliest_refresh_at", None)
        current["chatgpt"].pop("pending_refresh", None)
        _save_auth_store(store)
    print(f"Signed out of ChatGPT account {registration['label']}.")
    if not confirmed:
        print("Remote session revocation was not confirmed. Disconnect Hermes Agent in ChatGPT Settings.")


def auth_handler(action: str, args: Any) -> bool:
    if action == "add":
        selected = _select_registration(args, adding=True)
        fields = login(args, selected)
        can_infer = DIRECT_SCOPE in fields["chatgpt"]["scopes"]
        _save_login(fields, selected, str(getattr(args, "label", None) or "").strip())
        print("Signed in to ChatGPT." if can_infer else
              "Signed in to ChatGPT; plan usage is disabled. Reconnect this account to enable it, or choose an API-key provider.")
        return True
    if action == "status":
        state, rows = _local_state()
        if not rows:
            print("ChatGPT: no saved accounts")
        for row in rows:
            info = row["chatgpt"]
            status = "signed out" if not row.get("access_token") else (
                "plan usage enabled" if DIRECT_SCOPE in info.get("scopes", []) else "plan usage disabled")
            active = " (active)" if row["id"] == state.get("active_credential_id") else ""
            print(f"{row['label']}{active}: {info.get('email', '')} — {status}")
        return True
    if action == "logout":
        selected = _select_registration(args, adding=False)
        if selected:
            _logout(selected)
        else:
            print("ChatGPT: no active account to sign out")
        return True
    return False
