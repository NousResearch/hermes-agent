"""Gateway relay: one gateway's Webapp reaching other gateways' through a scoped grant.

``webapp.relay.enabled`` in a gateway's config turns on both roles, so any gateway can serve as a hub
and two gateways can each list the other:

- **Hub.** Its Webapp lists the gateways it was paired with (``webapp_relay_gateways``) and relays a
  signed-in browser's REST calls and sockets to them (``web_routers.webapp_relay_proxy``). The tab
  shows them in the gateway switcher, the way native Desktop shows its registered gateways.
- **Spoke.** It issues grants to hubs (``webapp_relay_grants``) and accepts their signed requests as
  the user who approved the grant.

Identity. A user signed in on the spoke (its own sign-in gate: Nous Portal on a Portal-signed-in
Webapp) approves the grant on the spoke's consent page, which binds that login's provider, issuer
and user id. The hub keeps the grant only when the user who started the pairing is signed in to the
hub as the same identity, and relays for that identity alone.

Proof. The hub holds an Ed25519 key (``webapp/relay-key.pem`` in its home, user-only); the spoke keeps
only the public half. Every relayed request carries a signature over the method, the target (raw path
and query, relative to the spoke's base URL), the spoke's base URL as the hub knows it (the audience
the spoke recorded at pairing), the time, a nonce, the grant id and the subject. The spoke checks the
time window and refuses a nonce it has already seen. The audience keeps a request the hub signed for
one spoke from being replayed at another, even by a spoke that named another's grant as its own.

One hop. A spoke never relays a relayed request onward, and a relayed request can neither manage
grants nor reach sign-in or Web Access routes (``webapp_relay_grants.refusal``).

Both sides require the sign-in gate: without one there is no identity to bind a grant to.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

_log = logging.getLogger(__name__)

KEY_PATH = Path("webapp") / "relay-key.pem"
#: The signature header set every relayed HTTP request carries (lower case, as Starlette reads them).
GRANT_HEADER = "x-hermes-relay-grant"
SUBJECT_HEADER = "x-hermes-relay-subject"
TIME_HEADER = "x-hermes-relay-time"
NONCE_HEADER = "x-hermes-relay-nonce"
SIGNATURE_HEADER = "x-hermes-relay-signature"
RELAY_HEADERS = (GRANT_HEADER, SUBJECT_HEADER, TIME_HEADER, NONCE_HEADER, SIGNATURE_HEADER)
#: How far a signed request's time may be from the spoke's clock, either way.
CLOCK_WINDOW_SECONDS = 60
_SIGNED_PREFIX = b"hermes-relay-v1"
_TOKEN_PREFIX = b"hermes-relay-token-v1"
_MAX_NONCES = 65536
_CONFIG_TTL_SECONDS = 5.0


class RelayAuthError(Exception):
    """A relayed request's signature, time or nonce did not check out; ``str()`` says which."""


class StoreUnreadable(OSError):
    """A relay store file is there but could not be read. Each holds every account's records, so a
    change that needs it fails and leaves the file as it is rather than write it back without them."""


def read_store(path: Path, key: str, kind: type) -> Any:
    """The *key* collection of the relay store at *path*, an empty one when there is no file. Raises
    StoreUnreadable when the file cannot be read or does not hold one."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return kind()
    except (OSError, ValueError) as exc:
        raise StoreUnreadable(f"{path.name} could not be read ({exc})") from exc
    held = stored.get(key) if isinstance(stored, dict) else None
    if not isinstance(held, kind):
        raise StoreUnreadable(f"{path.name} does not hold its {key}")
    return held


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    if not isinstance(text, str) or not text or len(text) > 512:
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not base64url") from exc


# ---- Configuration --------------------------------------------------------------------------

_config_lock = threading.Lock()
_config_cache: dict[str, tuple[float, bool]] = {}


def relay_configured(home) -> bool:
    """``webapp.relay.enabled`` in *home*'s config (re-read at most every few seconds). Anything
    but an explicit ``true`` reads as off."""
    key = str(home)
    now = time.monotonic()
    with _config_lock:
        cached = _config_cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    from ruamel.yaml.error import YAMLError

    from gateway.webapp_listener import _config_document
    try:
        document = _config_document(Path(home))
    except (OSError, ValueError, YAMLError) as exc:  # unreadable: the relay stays off, as the listener reads it
        _log.warning("Gateway relay stays off: %s/config.yaml could not be read (%s)", home, exc)
        document = {}
    block = document.get("webapp") if isinstance(document, dict) else None
    relay = block.get("relay") if isinstance(block, dict) else None
    enabled = isinstance(relay, dict) and relay.get("enabled") is True
    with _config_lock:
        _config_cache[key] = (now + _CONFIG_TTL_SECONDS, enabled)
    return enabled


def relay_enabled(state) -> bool:
    """Whether the listener with app *state* serves the relay: the Webapp surface, behind the
    sign-in gate, with ``webapp.relay.enabled`` on in the gateway home."""
    from hermes_cli.web_server_surface import policy
    home = getattr(state, "webapp_home", None)
    return (home is not None and policy(state).relay and bool(getattr(state, "auth_required", False))
            and relay_configured(home))


def _reset_for_tests() -> None:
    with _config_lock:
        _config_cache.clear()
    with _nonce_lock:
        _nonces.clear()


# ---- The hub's key --------------------------------------------------------------------------

_key_lock = threading.Lock()


def hub_key(home):
    """The hub's Ed25519 private key, created user-only in *home* on first use. A stored key that is
    unreadable, malformed or readable by other users is replaced, which ends every grant made to it
    (the user pairs again)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from gateway.webapp_listener import _close_to_other_users, _readable_by_other_users
    from utils import atomic_write_text

    path = Path(home) / KEY_PATH
    with _key_lock:
        _close_to_other_users(path.parent)
        try:
            if not _readable_by_other_users(path):
                key = serialization.load_pem_private_key(path.read_bytes(), password=None)
                if isinstance(key, Ed25519PrivateKey):
                    return key
        except (OSError, ValueError, TypeError):
            pass
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode("ascii")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write_text(path, pem, mode=0o600)
        return key


def public_key_text(key) -> str:
    """*key*'s public half as the base64url of its 32 raw bytes."""
    from cryptography.hazmat.primitives import serialization
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return b64url(raw)


def load_public_key(text: str):
    """The Ed25519 public key *text* (:func:`public_key_text`) names; ValueError when it names none."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    raw = b64url_decode(text)
    if len(raw) != 32:
        raise ValueError("not an Ed25519 public key")
    return Ed25519PublicKey.from_public_bytes(raw)


# ---- Signing --------------------------------------------------------------------------------

def _canonical(method: str, target: str, audience: str, when: str, nonce: str, grant_id: str,
               subject: str) -> bytes:
    return b"\n".join(part.encode("utf-8") if isinstance(part, str) else part for part in (
        _SIGNED_PREFIX, method.upper(), target, audience, when, nonce, grant_id, subject))


def sign_request(key, *, method: str, target: str, audience: str, grant_id: str, subject: str,
                 now: Optional[float] = None) -> dict[str, str]:
    """The relay headers for one request: *target* is the raw path and query (``/api/x?y=1``) as the
    spoke will see them, relative to its base URL *audience*."""
    when = str(int(time.time() if now is None else now))
    nonce = secrets.token_urlsafe(18)
    signature = key.sign(_canonical(method, target, audience, when, nonce, grant_id, subject))
    return {GRANT_HEADER: grant_id, SUBJECT_HEADER: subject, TIME_HEADER: when, NONCE_HEADER: nonce,
            SIGNATURE_HEADER: b64url(signature)}


def presented(headers: Mapping[str, str]) -> bool:
    """Whether *headers* carry a relay signature (any of its headers)."""
    return any(headers.get(name) for name in RELAY_HEADERS)


def signed_grant(headers: Mapping[str, str]) -> tuple[str, str]:
    """``(grant id, subject)`` a relayed request names, unverified; RelayAuthError when absent."""
    grant_id, subject = headers.get(GRANT_HEADER, ""), headers.get(SUBJECT_HEADER, "")
    if not grant_id or not subject or len(grant_id) > 64 or len(subject) > 256:
        raise RelayAuthError("relay headers incomplete")
    return grant_id, subject


def verify_request(public_key: str, headers: Mapping[str, str], *, method: str, target: str, audience: str,
                   now: Optional[float] = None) -> None:
    """Check a relayed request's signature with the grant's *public_key* and *audience*, its time and
    its nonce (spent here, so a replay is refused). Raises RelayAuthError."""
    grant_id, subject = signed_grant(headers)
    when, nonce = headers.get(TIME_HEADER, ""), headers.get(NONCE_HEADER, "")
    signature = headers.get(SIGNATURE_HEADER, "")
    if not when.isdigit() or not nonce or len(nonce) > 64 or not signature:
        raise RelayAuthError("relay headers incomplete")
    current = time.time() if now is None else now
    if abs(current - int(when)) > CLOCK_WINDOW_SECONDS:
        raise RelayAuthError("relay request outside the clock window")
    from cryptography.exceptions import InvalidSignature
    try:
        load_public_key(public_key).verify(b64url_decode(signature),
                                           _canonical(method, target, audience, when, nonce, grant_id, subject))
    except (InvalidSignature, ValueError, UnicodeEncodeError) as exc:  # forged, or a malformed key or signature
        raise RelayAuthError("relay signature does not verify") from exc
    _spend_nonce(f"{grant_id}:{nonce}", current)


def _token_canonical(code: str, audience: str) -> bytes:
    return b"\n".join((_TOKEN_PREFIX, code.encode("utf-8"), audience.encode("utf-8")))


def sign_token_request(key, code: str, audience: str) -> str:
    """The hub's proof, on the code exchange, that it holds the key the code was issued for and
    names the spoke at *audience*."""
    return b64url(key.sign(_token_canonical(code, audience)))


def verify_token_request(public_key: str, code: str, audience: str, signature: str) -> bool:
    from cryptography.exceptions import InvalidSignature
    try:
        load_public_key(public_key).verify(b64url_decode(signature), _token_canonical(code, audience))
    except (InvalidSignature, ValueError, UnicodeEncodeError):
        return False
    return True


def pkce_challenge(verifier: str) -> str:
    import hashlib
    return b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def pkce_matches(verifier: str, challenge: str) -> bool:
    if not isinstance(verifier, str) or not 43 <= len(verifier) <= 128:
        return False
    try:
        return hmac.compare_digest(pkce_challenge(verifier), challenge)
    except (UnicodeEncodeError, TypeError):
        return False


# ---- Replay protection ----------------------------------------------------------------------

_nonce_lock = threading.Lock()
_nonces: dict[str, float] = {}


def _spend_nonce(key: str, now: float) -> None:
    expires = now + 2 * CLOCK_WINDOW_SECONDS
    with _nonce_lock:
        if key in _nonces and _nonces[key] > now:
            raise RelayAuthError("relay nonce already used")
        if len(_nonces) >= _MAX_NONCES:
            for stale in [k for k, until in _nonces.items() if until <= now]:
                del _nonces[stale]
            if len(_nonces) >= _MAX_NONCES:
                raise RelayAuthError("too many relayed requests in flight; try again shortly")
        _nonces[key] = expires
