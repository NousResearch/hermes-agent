"""Nous Portal OAuth: device-code login, refresh, shared-store mirroring, JWT selection, status.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily inside each function
so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

from auth.providers.nous import _NOUS_EMPTY_AGENT_KEY_FIELDS, _iso_after, _nous_http_client, _portal_entitlement_message, logger
from auth.providers.nous_store import _nous_shared_store_path, _try_import_shared_nous_state


import logging
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional
from urllib.parse import urlparse
from auth.providers.codex import _pool_entries
from auth.token_validation import _decode_jwt_claims
from auth.errors import AuthError
from auth.constants import DEFAULT_NOUS_CLIENT_ID, DEFAULT_NOUS_INFERENCE_URL, DEFAULT_NOUS_SCOPE, DEFAULT_NOUS_WELCOME_URL, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS, NOUS_AUTH_PATH_INVOKE_JWT, NOUS_BILLING_MANAGE_SCOPE, NOUS_DEVICE_CODE_SOURCE, NOUS_INFERENCE_INVOKE_SCOPE, NOUS_INVOKE_JWT_MIN_TTL_SECONDS, _nous_err, httpx
from auth.store import AUTH_LOCK_TIMEOUT_SECONDS
from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").


# Nous agent-key slots; a fresh login persists them as None, quarantine strips them.


# Allowlist of hosts the Nous Portal proxy will forward inference JWTs to — a bearer sent anywhere
# else would leak. Consulted only for URLs from the NETWORK side (Portal refresh responses);
# the NOUS_INFERENCE_BASE_URL env override bypasses it (documented dev/staging escape hatch, the
# user set it themselves).


# Derived from expires_at/JWT exp and tick down between reads; persisting only these changes makes
# auth.json noisy and defeats the mtime-keyed auth-status cache.


# OAuth fields mirrored between a profile's Nous state and the shared cross-profile store.


def _model_priority(mid: str) -> tuple:
    """Sort key: opus > pro > haiku/flash > sonnet (sonnet is cheap/fast; best model first)."""
    low = mid.lower()
    rank = (0 if "opus" in low else 1 if "pro" in low and "sonnet" not in low
            else 3 if "sonnet" in low else 2)
    return (rank, mid)


def fetch_nous_models(
    *, inference_base_url: str, api_key: str, timeout_seconds: float = 15.0,
    verify: bool | str = True) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    from auth.token_validation import _nonempty_str
    with _nous_http_client(timeout_seconds, verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise _nous_err(description, "models_fetch_failed")
    data = response.json().get("data")
    if not isinstance(data, list):
        return []
    model_ids: List[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        # Hermes models aren't reliable for agentic tool-calling
        if _nonempty_str(model_id) and "hermes" not in model_id.lower():
            model_ids.append(model_id.strip())
    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


# Enum values reported on the dashboard /api/status as ``nous_session_valid``. NAS's health sweep
# re-mints the bootstrap session ONLY on "terminal"; "valid" and "unknown" are no-ops. Keep this
# set small and stable — NAS parses it permissively, so new members are non-breaking but rare.


def _nous_device_code_login(
    *, portal_base_url: Optional[str] = None, inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None, scope: Optional[str] = None, open_browser: bool = True,
    timeout_seconds: float = 15.0, insecure: bool = False, ca_bundle: Optional[str] = None,
    on_verification: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_device_flow import _is_remote_session, _print_device_code_instructions
    from hermes_cli.auth_error_copy import format_auth_error
    from auth.oauth import _coerce_ttl_seconds, _optional_base_url, _poll_for_token, _request_device_code, _tls_state_from_verify
    from auth.providers.nous import refresh_nous_oauth_from_state
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url or os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url).rstrip("/")
    requested_inference_url = (
        inference_base_url or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)
    if _is_remote_session():
        open_browser = False
    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")
    with _nous_http_client(timeout_seconds, verify) as client:
        device_data = _request_device_code(
            client=client, portal_base_url=portal_base_url, client_id=client_id, scope=scope)
        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])
        _print_device_code_instructions(
            verification_url, user_code, open_browser=open_browser, failure_dash="—")
        # Out-of-band consumer (e.g. the TUI gateway, whose stdout is a JSON-RPC pipe): fired AFTER
        # the print/browser block and BEFORE polling so it can render the link while we wait.
        if on_verification is not None:
            with suppress(Exception):
                on_verification(verification_url, user_code)
        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")
        token_data = _poll_for_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            device_code=str(device_data["device_code"]), expires_in=expires_in,
            poll_interval=interval)
    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url")) or requested_inference_url)
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")
    auth_state = {
        "portal_base_url": portal_base_url, "inference_base_url": resolved_inference_url,
        "client_id": client_id, "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(), "expires_at": _iso_after(now, token_expires_in),
        "expires_in": token_expires_in, "tls": _tls_state_from_verify(verify),
        **_NOUS_EMPTY_AGENT_KEY_FIELDS}
    try:
        return refresh_nous_oauth_from_state(
            auth_state, timeout_seconds=timeout_seconds, force_refresh=False)
    except AuthError as exc:
        if exc.code == "subscription_required":
            portal_url = auth_state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print()
            print(format_auth_error(exc))
            print(f"  Subscribe here: {portal_url}/billing")
            print()
            print("After subscribing, run `hermes model` again to finish setup.")
            raise SystemExit(1)
        raise


def _mirror_nous_state_best_effort(auth_state: Dict[str, Any]) -> None:
    """Mirror to the shared store + reseed the pool, swallowing all errors (same as _login_nous)."""
    from auth.providers.nous import _sync_nous_pool_from_auth_store
    from auth.providers.nous_store import _write_shared_nous_state
    with suppress(Exception):
        _write_shared_nous_state(auth_state)
    with suppress(Exception):
        _sync_nous_pool_from_auth_store(environment=_phase6_auth_environment())


def step_up_nous_billing_scope(
    *, open_browser: bool = True, timeout_seconds: float = 15.0,
    on_verification: Optional[Callable[[str, str], None]] = None) -> bool:
    """Re-run the device flow requesting ``billing:manage`` (step-up on 403 insufficient_scope).

    The user must be ADMIN/OWNER and select "Allow Remote Spending" in the portal, else the
    server silently downscopes and this returns False. Persists like ``_login_nous`` minus the
    model picker.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_nous import _nous_device_code_login
    from auth.provider_state import _save_active_provider_state, get_provider_auth_state
    prior = get_provider_auth_state("nous") or {}
    pconfig = PROVIDER_REGISTRY["nous"]
    # Step-up scope: existing scopes (if any) + billing:manage, deduped, order-stable. Falls back
    # to the standard inference+billing set.
    _raw_scope = prior.get("scope")
    prior_scope = _raw_scope.split() if isinstance(_raw_scope, str) else []
    requested = list(dict.fromkeys([
        *(prior_scope or [NOUS_INFERENCE_INVOKE_SCOPE]), NOUS_BILLING_MANAGE_SCOPE]))
    auth_state = _nous_device_code_login(
        portal_base_url=prior.get("portal_base_url") or None,
        inference_base_url=prior.get("inference_base_url") or None,
        client_id=prior.get("client_id") or pconfig.client_id, scope=" ".join(requested),
        open_browser=open_browser, timeout_seconds=timeout_seconds, on_verification=on_verification)
    _save_active_provider_state("nous", auth_state)
    _mirror_nous_state_best_effort(auth_state)
    granted = auth_state.get("scope")
    return isinstance(granted, str) and NOUS_BILLING_MANAGE_SCOPE in granted.split()


def _pick_nous_model_after_login(
    auth_state: Dict[str, Any], inference_base_url: str) -> Optional[str]:
    """Fetch the curated Nous model list (tier/policy-filtered) and run the interactive picker.

    Returns the selected model id, or None when the user skipped / nothing was selectable.
    Raises on any fetch failure so the caller can print the "Login succeeded, but..." notice.
    """
    from hermes_cli.auth_model_picker import _prompt_model_selection
    runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
    if not isinstance(runtime_key, str) or not runtime_key:
        raise _nous_err("No runtime API key available to fetch models", "invalid_token")
    from hermes_cli.models import (
        get_curated_nous_model_ids,
        check_nous_free_tier,
        partition_nous_models_by_tier,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )
    from hermes_cli.models_pricing import (
        get_pricing_for_provider,
        nous_policy_allowed_ids,
        restrict_to_nous_policy,
    )
    model_ids = get_curated_nous_model_ids()
    _portal = auth_state.get("portal_base_url", "")
    print()
    unavailable_models: list = []
    unavailable_message = ""
    _policy_narrowed = False
    if model_ids:
        pricing = get_pricing_for_provider("nous")
        # Force fresh account data so recent credit purchases are reflected immediately.
        free_tier = check_nous_free_tier(force_fresh=True)
        # Narrow before the tier split, so a rescued id still has to pass the free/paid predicate.
        _policy_allowed = nous_policy_allowed_ids()
        if free_tier:
            with suppress(Exception):
                unavailable_message = _portal_entitlement_message("paid Nous models", environment=_phase6_auth_environment())
        # The Portal's free/paidRecommendedModels endpoint is the source of truth for what's
        # available *right now*: newly-launched models show without a CLI release.
        union = (
            union_with_portal_free_recommendations if free_tier
            else union_with_portal_paid_recommendations)
        model_ids, pricing = union(model_ids, pricing, _portal)
        _before_policy = model_ids
        model_ids = restrict_to_nous_policy(model_ids, _policy_allowed, rescue_empty=True)
        _policy_narrowed = model_ids != _before_policy
        if free_tier:
            model_ids, unavailable_models = partition_nous_models_by_tier(
                model_ids, pricing, free_tier=True)
    if model_ids:
        from hermes_cli.nous_account import nous_policy_notice
        _policy_notice = nous_policy_notice(removed=_policy_narrowed)
        if _policy_notice:
            print(_policy_notice)
        print(
            f"Showing {len(model_ids)} curated models — "
            "use \"Enter custom model name\" for others.")
        return _prompt_model_selection(
            model_ids, pricing=pricing, unavailable_models=unavailable_models, portal_url=_portal,
            unavailable_message=unavailable_message, confirm_provider="nous",
            confirm_base_url=inference_base_url, confirm_api_key=runtime_key)
    if unavailable_models:
        _url = (_portal or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        print("No free models currently available.")
        print(unavailable_message or f"Upgrade at {_url} to access paid models.")
    else:
        print("No curated models available for Nous Portal.")
    return None


def _offer_shared_nous_import(timeout_seconds: float) -> Optional[Dict[str, Any]]:
    """Codex-style auto-import: offer to rehydrate a Nous credential from another profile.

    Checks the shared store before launching a fresh device-code flow. Returns the refreshed
    auth state when the user accepted and the import succeeded, else None.
    """
    from hermes_cli.auth_device_flow import _prompt_yes_no
    from auth.providers.nous_store import _read_shared_nous_state
    from auth.providers.nous_guest import is_guest_state
    shared = _read_shared_nous_state()
    if not shared or is_guest_state(shared):
        # A free-tier identity is not an OAuth credential to import; a real sign-in replaces it
        # (persist_nous_credentials overwrites the singleton and the shared store).
        return None
    try:
        shared_path = _nous_shared_store_path()
    except RuntimeError:
        shared_path = None
    print()
    print(f"Found existing Nous OAuth credentials at {shared_path}" if shared_path
          else "Found existing shared Nous OAuth credentials")
    if not _prompt_yes_no("Import these credentials? [Y/n]: ", default="y"):
        return None
    print("Rehydrating Nous session from shared credentials...")
    auth_state = _try_import_shared_nous_state(timeout_seconds=timeout_seconds)
    if auth_state is None:
        print("Could not refresh shared credentials — falling back to device-code login.")
    return auth_state




def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    from hermes_cli.auth_nous import _nous_device_code_login
    from hermes_cli.auth_model_picker import _save_model_choice
    from hermes_cli.auth import _update_config_for_provider
    from hermes_cli.auth_error_copy import format_auth_error
    from auth.providers.nous import _sync_nous_pool_from_auth_store
    from auth.providers.nous_store import _write_shared_nous_state
    from auth.provider_state import _save_active_provider_state, get_active_provider, restore_active_provider
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    ca_bundle = (
        getattr(args, "ca_bundle", None) or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE"))
    try:
        auth_state = _offer_shared_nous_import(timeout_seconds)
        if auth_state is None:
            auth_state = _nous_device_code_login(
                portal_base_url=getattr(args, "portal_url", None),
                inference_base_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None) or pconfig.client_id,
                scope=getattr(args, "scope", None),
                open_browser=not getattr(args, "no_browser", False),
                timeout_seconds=timeout_seconds, insecure=bool(getattr(args, "insecure", False)),
                ca_bundle=ca_bundle)
        inference_base_url = auth_state["inference_base_url"]
        # Snapshot BEFORE _save_provider_state overwrites active_provider to "nous", so a
        # model-picker "Skip (keep current)" can restore the user's previous provider.
        prior_active_provider = get_active_provider()
        saved_to = _save_active_provider_state("nous", auth_state)
        # Mirror to the shared store so other profiles can one-tap import (best-effort inside).
        _write_shared_nous_state(auth_state)
        _sync_nous_pool_from_auth_store(environment=_phase6_auth_environment())
        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")
        # Pick the model BEFORE writing the provider to config.yaml so config is never half-updated.
        selected_model = None
        try:
            selected_model = _pick_nous_model_after_login(auth_state, inference_base_url)
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")
        # No model (Skip, fetch failed, nothing curated): keep the previous provider rather than
        # switch to Nous with a mismatched model; the Nous tokens stay saved for future use.
        if not selected_model:
            restore_active_provider(prior_active_provider)
            print()
            print("No provider change. Nous credentials saved for future use.")
            print("  Run `hermes model` again to switch to Nous Portal.")
            return
        config_path = _update_config_for_provider(
            "nous", inference_base_url, default_model=selected_model)
        _save_model_choice(selected_model)
        print(f"Default model set to: {selected_model}")
        print(f"  Config updated: {config_path} (model.provider=nous)")
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        from hermes_cli.auth_error_copy import sign_in_failure_lines
        logger.debug("nous login failed: %r", exc)
        print()
        for line in sign_in_failure_lines(exc, service_host=_portal_host(getattr(args, "portal_url", None))):
            print(line)
        raise SystemExit(1)


def _portal_host(portal_url: Optional[str]) -> str:
    return urlparse(portal_url or DEFAULT_NOUS_PORTAL_URL).hostname or "portal.nousresearch.com"

from auth.token_validation import _scope_values, _nous_invoke_jwt_status, _nous_invoke_jwt_is_usable
