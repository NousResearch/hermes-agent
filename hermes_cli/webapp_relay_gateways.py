"""The hub side of the gateway relay: the gateways this one's Webapp was paired with.

Pairing (``start_pairing`` → the spoke's consent page → ``finish_pairing``) is an authorization-code
exchange with PKCE, started by a signed-in browser here and approved by the same person signed in to
the spoke. The spoke issues a grant to this gateway's key (``webapp_relay.hub_key``); this side keeps
``webapp/relay-gateways.json`` (user-only): each gateway's id, label, base URL, grant id and the
identity the grant acts as. It holds no secret of the spoke's: the private key is the only credential.

Only a browser signed in here as that same identity sees a gateway or relays to it
(``gateways_for``), so another account on a shared hub cannot use your grants.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx

from hermes_cli import webapp_relay as relay

_log = logging.getLogger(__name__)

GATEWAYS_PATH = Path("webapp") / "relay-gateways.json"
PAIRING_TTL_SECONDS = 600
_MAX_PENDING = 32
_MAX_GATEWAYS = 32
#: Connection ids the Webapp itself names (its own serving gateway) and never gives a relay.
_RESERVED_IDS = frozenset({"local", "primary"})

_lock = threading.Lock()
_pending: dict[str, tuple[float, dict[str, Any]]] = {}


class PairingError(ValueError):
    """A pairing that cannot start or finish; ``str()`` is shown to the user."""


class RevokeUnconfirmed(Exception):
    """The spoke did not confirm it revoked a grant; ``str()`` is shown to the user."""


def _path(home) -> Path:
    return Path(home) / GATEWAYS_PATH


def _read(home) -> list[dict[str, Any]]:
    """Every gateway on file. Raises relay.StoreUnreadable: what writes the file must see all of it."""
    rows = relay.read_store(_path(home), "gateways", list)
    return [row for row in rows if isinstance(row, dict) and row.get("id") and row.get("url")]


def _lookup(home) -> list[dict[str, Any]]:
    """The gateways a read that changes nothing sees: none while the file cannot be read."""
    try:
        return _read(home)
    except relay.StoreUnreadable as exc:
        _log.warning("Gateway relay registry could not be read: %s", exc)
        return []


def _write(home, rows: list[dict[str, Any]]) -> None:
    from gateway.webapp_listener import _close_to_other_users
    from utils import atomic_write_text

    path = _path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _close_to_other_users(path.parent)
    atomic_write_text(path, json.dumps({"gateways": rows}, indent=1, sort_keys=True) + "\n", mode=0o600)


def _owned_by(row: dict[str, Any], identity: dict[str, str]) -> bool:
    return (row.get("subject") == identity.get("user_id") and row.get("provider") == identity.get("provider")
            and (row.get("issuer") or "") == (identity.get("issuer") or ""))


def gateways_for(home, identity: dict[str, str]) -> list[dict[str, Any]]:
    """The gateways *identity* paired this hub with, in the order they were added."""
    with _lock:
        rows = _lookup(home)
    return [row for row in rows if _owned_by(row, identity)]


def gateway_for(home, identity: dict[str, str], gateway_id: str) -> Optional[dict[str, Any]]:
    return next((row for row in gateways_for(home, identity) if row["id"] == gateway_id), None)


def remove(home, identity: dict[str, str], gateway_id: str) -> Optional[dict[str, Any]]:
    """Forget gateway *gateway_id* for *identity*; the removed row, or None."""
    with _lock:
        rows = _read(home)
        removed = next((row for row in rows if row["id"] == gateway_id and _owned_by(row, identity)), None)
        if removed is not None:
            _write(home, [row for row in rows if row is not removed])
    return removed


def normalized_url(raw: str) -> str:
    """*raw* as a spoke base URL (``http(s)://host[:port][/path]``, no trailing slash); PairingError
    when it is not one."""
    text = str(raw or "").strip().rstrip("/")
    try:
        parts = urllib.parse.urlsplit(text)
        parts.port  # raises ValueError on a malformed port
    except ValueError:
        raise PairingError("Enter the gateway's Webapp address, like https://mini.example.ts.net") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise PairingError("Enter the gateway's Webapp address, like https://mini.example.ts.net")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))


def _slug(label: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:32] or "gateway"
    candidate, n = base, 2
    while candidate in taken or candidate in _RESERVED_IDS:
        candidate, n = f"{base}-{n}", n + 1
    return candidate


def start_pairing(home, *, identity: dict[str, str], url: str, label: str, hub_name: str,
                  callback_url: str) -> str:
    """Begin pairing with the spoke at *url* for *identity*; returns the spoke's consent URL for the
    browser to open. *callback_url* is this hub's ``/webapp/relay/callback``."""
    base = normalized_url(url)
    label = " ".join(str(label or "").split())[:64] or urllib.parse.urlsplit(base).hostname or "Gateway"
    if not relay.relay_configured(home):
        raise PairingError("Turn on webapp.relay.enabled in this gateway's config.yaml first.")
    with _lock:
        rows = _read(home)
        if any(row["url"] == base and _owned_by(row, identity) for row in rows):
            raise PairingError(f"{base} is already connected.")
        if len(rows) >= _MAX_GATEWAYS:
            raise PairingError("This gateway already lists the most gateways it can.")
    key = relay.hub_key(home)
    verifier = relay.b64url(secrets.token_bytes(48))
    state = relay.b64url(secrets.token_bytes(24))
    now = time.monotonic()
    with _lock:
        for stale in [s for s, (until, _p) in _pending.items() if until <= now]:
            del _pending[stale]
        if len(_pending) >= _MAX_PENDING:
            raise PairingError("Too many pairings are waiting; try again in a few minutes.")
        _pending[state] = (now + PAIRING_TTL_SECONDS, {
            "url": base, "label": label, "verifier": verifier, "identity": dict(identity),
            "callback_url": callback_url})
    query = urllib.parse.urlencode({
        "hub_name": hub_name, "redirect_uri": callback_url, "state": state,
        "code_challenge": relay.pkce_challenge(verifier), "code_challenge_method": "S256",
        "public_key": relay.public_key_text(key), "scope": " ".join(sorted(_scopes()))})
    return f"{base}/webapp/relay/authorize?{query}"


def cancel_pairing(state: str) -> None:
    """Drop the pending pairing *state* names (the spoke denied it or failed)."""
    with _lock:
        _pending.pop(state, None)


def _scopes() -> set[str]:
    from hermes_cli.webapp_relay_grants import SCOPES
    return set(SCOPES)


async def finish_pairing(home, *, identity: dict[str, str], state: str, code: str, client) -> dict[str, Any]:
    """Exchange the code the spoke returned with the browser for a grant, check it acts as
    *identity*, and list the gateway. *client* is an ``httpx.AsyncClient``. Raises PairingError."""
    with _lock:
        entry = _pending.pop(state, None)
    if entry is None or entry[0] <= time.monotonic():
        raise PairingError("This pairing expired or was already used. Start it again from Settings.")
    pairing = entry[1]
    if pairing["identity"] != identity:
        raise PairingError("Finish pairing signed in as the account that started it.")
    key = relay.hub_key(home)
    try:
        response = await client.post(f"{pairing['url']}/api/webapp/relay/token", json={
            "code": code, "code_verifier": pairing["verifier"], "redirect_uri": pairing["callback_url"],
            "audience": pairing["url"], "proof": relay.sign_token_request(key, code, pairing["url"])},
            timeout=15.0)
    except httpx.HTTPError as exc:  # transport failure: the spoke is unreachable from this host
        raise PairingError(f"Could not reach {pairing['url']} from this gateway ({type(exc).__name__}).") from exc
    if response.status_code != 200:
        raise PairingError(f"{pairing['url']} refused the pairing ({response.status_code}).")
    grant = response.json()
    granted = {"user_id": grant.get("subject"), "provider": grant.get("provider"), "issuer": grant.get("issuer") or ""}
    if granted != identity:
        # Approved there as someone else: this hub will not act as that account. The spoke keeps a
        # grant nobody here uses; revoke it so it does not linger, and say so when that fails.
        failure = await _revoke_remote(home, key, pairing["url"], grant, client)
        message = f"{pairing['url']} approved the pairing as a different account."
        if failure is not None:
            message += (f" The grant it made could not be revoked: {failure}. Revoke it there in "
                        "Settings → Gateways.")
        raise PairingError(f"{message} Sign in to both gateways with the same account and pair again.")
    try:
        row = _record(home, pairing, grant, identity)
    except OSError as exc:  # unreadable or unwritable: nobody here could use or revoke the grant
        failure = await _revoke_remote(home, key, pairing["url"], grant, client)
        message = f"This gateway could not save the pairing: {exc}."
        if failure is not None:
            message += (f" The grant {pairing['url']} made could not be revoked: {failure}. Revoke it there "
                        "in Settings → Gateways.")
        raise PairingError(message) from None
    _log.info("Gateway relay: paired with %s as %s", pairing["url"], row["id"])
    return row


def _record(home, pairing: dict[str, Any], grant: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    with _lock:
        rows = _read(home)
        row = {"id": _slug(pairing["label"], {r["id"] for r in rows}), "label": pairing["label"],
               "url": pairing["url"], "grant_id": grant["grant_id"], "subject": identity["user_id"],
               "provider": identity["provider"], "issuer": identity["issuer"],
               "scopes": sorted(s for s in grant.get("scopes", []) if isinstance(s, str)),
               "added_at": int(time.time())}
        rows.append(row)
        _write(home, rows)
    return row


async def _revoke_remote(home, key, url: str, grant: dict[str, Any], client) -> Optional[str]:
    """Revoke *grant* on the spoke at *url*, signed as the grant itself; None once the spoke confirms
    it, otherwise why it is not confirmed (shown to the user)."""
    grant_id, subject = grant.get("grant_id"), grant.get("subject")
    if not grant_id or not subject:
        return "it named no grant"
    target = f"/api/webapp/relay/grants/{grant_id}"
    headers = relay.sign_request(key, method="DELETE", target=target, audience=url, grant_id=grant_id,
                                 subject=subject)
    try:
        response = await client.delete(f"{url}{target}", headers=headers, timeout=10.0)
    except httpx.HTTPError as exc:  # transport failure: the spoke is unreachable from this host
        _log.warning("Gateway relay: could not revoke grant at %s: %s", url, exc)
        return f"it is not reachable from this gateway ({type(exc).__name__})"
    if response.is_success:
        return None
    _log.warning("Gateway relay: %s answered %s to revoking its grant", url, response.status_code)
    if response.status_code == 401:
        return "it no longer accepts this gateway's key, so the grant may already be revoked there"
    return f"it answered {response.status_code}"


async def unpair(home, identity: dict[str, str], gateway_id: str, client, *, forget: bool = False) -> bool:
    """Revoke *gateway_id*'s grant on the spoke, then forget the gateway here; False when *identity*
    has no such gateway.

    Raises RevokeUnconfirmed, and keeps the gateway so Remove can run again, when the spoke does not
    confirm the revocation. *forget* forgets it here without asking the spoke: its grant stays until
    the user revokes it there."""
    row = gateway_for(home, identity, gateway_id)
    if row is None:
        return False
    if not forget:
        failure = await _revoke_remote(home, relay.hub_key(home), row["url"],
                                       {"grant_id": row["grant_id"], "subject": row["subject"]}, client)
        if failure is not None:
            raise RevokeUnconfirmed(f"{row['label']} did not confirm it revoked this gateway's access: {failure}.")
    return remove(home, identity, gateway_id) is not None


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """What the browser sees of a gateway: no grant id."""
    return {key: row[key] for key in ("id", "label", "url", "scopes", "added_at") if key in row}


def _reset_for_tests() -> None:
    with _lock:
        _pending.clear()
