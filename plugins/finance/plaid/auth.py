"""Plaid credential resolution, secure token storage, and Link onboarding.

Per-item Plaid ``access_token``s are the crown jewels: they grant ongoing
read access to a user's bank data. They are stored in a 0600 JSON file under
``<HERMES_HOME>/finance/plaid_tokens.json`` — never in ``config.yaml``, never
in ``.env``, never in the queryable SQLite DB, and never returned to the model.

Onboarding uses Plaid's Hosted Link flow: create a link token with
``hosted_link`` enabled, hand the user a URL to open, then poll
``/link/token/get`` until a ``public_token`` appears and exchange it for an
access token. This needs no local web server or redirect handling.

References:
* https://plaid.com/docs/link/hosted-link/
* https://plaid.com/docs/api/link/
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home, secure_parent_dir

from plugins.finance.errors import FinanceConfigError, FinanceError
from plugins.finance.plaid.client import PlaidClient

_TOKEN_STORE_VERSION = 1

# Plaid sandbox test institution ("First Platypus Bank").
SANDBOX_INSTITUTION_ID = "ins_109508"


def token_store_path() -> Path:
    return get_hermes_home() / "finance" / "plaid_tokens.json"


def get_plaid_credentials() -> Tuple[str, str]:
    """Return ``(client_id, secret)`` from the environment.

    Raises :class:`FinanceConfigError` when either is missing so the CLI/tools
    can print a setup hint instead of failing deep in an HTTP call.
    """
    client_id = (os.getenv("PLAID_CLIENT_ID") or "").strip()
    secret = (os.getenv("PLAID_SECRET") or "").strip()
    if not client_id or not secret:
        raise FinanceConfigError(
            "Plaid is not configured. Add PLAID_CLIENT_ID and PLAID_SECRET to "
            "your .env (get them from https://dashboard.plaid.com/developers/keys)."
        )
    return client_id, secret


def credentials_present() -> bool:
    """Cheap, non-raising check used to gate tool availability."""
    return bool((os.getenv("PLAID_CLIENT_ID") or "").strip() and (os.getenv("PLAID_SECRET") or "").strip())


def build_client(environment: str) -> PlaidClient:
    client_id, secret = get_plaid_credentials()
    return PlaidClient(client_id, secret, environment=environment)


# ---------------------------------------------------------------------------
# Secure token storage
# ---------------------------------------------------------------------------

def _write_secure_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    os.replace(str(tmp), str(path))
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def load_token_store() -> Dict[str, Any]:
    path = token_store_path()
    if not path.exists():
        return {"version": _TOKEN_STORE_VERSION, "items": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": _TOKEN_STORE_VERSION, "items": {}}
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        return {"version": _TOKEN_STORE_VERSION, "items": {}}
    return data


def save_item_token(
    item_id: str,
    access_token: str,
    *,
    institution_id: Optional[str] = None,
    institution_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> None:
    store = load_token_store()
    store.setdefault("items", {})[item_id] = {
        "access_token": access_token,
        "institution_id": institution_id,
        "institution_name": institution_name,
        "environment": environment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_secure_json(token_store_path(), store)


def get_access_token(item_id: str) -> Optional[str]:
    item = load_token_store().get("items", {}).get(item_id)
    return item.get("access_token") if isinstance(item, dict) else None


def list_token_items() -> List[Dict[str, Any]]:
    """Return per-item metadata WITHOUT access tokens (safe to display/log)."""
    items = load_token_store().get("items", {})
    safe = []
    for item_id, meta in items.items():
        if not isinstance(meta, dict):
            continue
        safe.append({
            "item_id": item_id,
            "institution_id": meta.get("institution_id"),
            "institution_name": meta.get("institution_name"),
            "environment": meta.get("environment"),
            "created_at": meta.get("created_at"),
        })
    return safe


def remove_item_token(item_id: str) -> bool:
    store = load_token_store()
    items = store.get("items", {})
    if item_id in items:
        del items[item_id]
        _write_secure_json(token_store_path(), store)
        return True
    return False


# ---------------------------------------------------------------------------
# Hosted Link onboarding
# ---------------------------------------------------------------------------

def start_hosted_link(
    client: PlaidClient,
    *,
    products: Optional[List[str]] = None,
    webhook: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a hosted Link token and return the URL the user should open."""
    response = client.link_token_create(
        user_id=user_id or f"hermes-{uuid.uuid4().hex[:12]}",
        products=products or ["transactions"],
        hosted_link={},
        webhook=webhook or None,
    )
    hosted_url = response.get("hosted_link_url")
    if not hosted_url:
        raise FinanceError(
            "Plaid did not return a hosted Link URL. Ensure Hosted Link is "
            "enabled for your Plaid client (dashboard → Link → Hosted Link)."
        )
    return {
        "link_token": response.get("link_token"),
        "hosted_link_url": hosted_url,
        "expiration": response.get("expiration"),
    }


def _extract_public_token(link_get_response: Dict[str, Any]) -> Optional[str]:
    """Pull the first completed ``public_token`` out of a /link/token/get body."""
    for session in link_get_response.get("link_sessions") or []:
        results = (session or {}).get("results") or {}
        for add_result in results.get("item_add_results") or []:
            public_token = (add_result or {}).get("public_token")
            if public_token:
                return public_token
    return None


def poll_for_public_token(
    client: PlaidClient,
    link_token: str,
    *,
    timeout: float = 300.0,
    interval: float = 3.0,
    sleep=time.sleep,
) -> Optional[str]:
    """Poll ``/link/token/get`` until the user completes Link or *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.link_token_get(link_token)
        public_token = _extract_public_token(response)
        if public_token:
            return public_token
        sleep(interval)
    return None


def exchange_public_token(client: PlaidClient, public_token: str) -> Dict[str, Any]:
    response = client.item_public_token_exchange(public_token)
    return {"access_token": response.get("access_token"), "item_id": response.get("item_id")}


def resolve_institution_name(client: PlaidClient, institution_id: Optional[str]) -> Optional[str]:
    if not institution_id:
        return None
    try:
        response = client.institutions_get_by_id(institution_id)
        return ((response.get("institution") or {}).get("name"))
    except FinanceError:
        return None


def complete_link(
    client: PlaidClient,
    public_token: str,
    *,
    environment: str,
    institution_id: Optional[str] = None,
    institution_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange a public token, persist the access token, return item metadata."""
    exchanged = exchange_public_token(client, public_token)
    access_token = exchanged.get("access_token")
    item_id = exchanged.get("item_id")
    if not access_token or not item_id:
        raise FinanceError("Plaid public-token exchange did not return an item.")
    if not institution_name:
        institution_name = resolve_institution_name(client, institution_id)
    save_item_token(
        item_id, access_token,
        institution_id=institution_id, institution_name=institution_name,
        environment=environment,
    )
    return {
        "item_id": item_id,
        "institution_id": institution_id,
        "institution_name": institution_name,
        "environment": environment,
    }


def sandbox_connect(
    client: PlaidClient,
    *,
    institution_id: str = SANDBOX_INSTITUTION_ID,
    products: Optional[List[str]] = None,
    environment: str = "sandbox",
) -> Dict[str, Any]:
    """Create and link a sandbox item end-to-end (for local testing)."""
    created = client.sandbox_public_token_create(
        institution_id=institution_id, initial_products=products or ["transactions"]
    )
    public_token = created.get("public_token")
    if not public_token:
        raise FinanceError("Plaid sandbox did not return a public token.")
    return complete_link(
        client, public_token, environment=environment,
        institution_id=institution_id,
        institution_name="Plaid Sandbox Bank",
    )
