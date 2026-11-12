"""Agent Card signature verification — A2A v1.0 §8.4.

A card's signatures are a JWS (RFC 7515) over the JCS-canonicalised card (RFC 8785). The spec's
security guidance is that an *unverified* card is attacker-controlled data: its endpoints and
skills must not be trusted until a signature is checked against a key the client already trusts.
Without this check, a peer that serves a card ``signatures`` entry forged with an attacker's key
under a legitimate signer's ``kid`` is accepted silently (hermes-agent#150793).

This module is inert until ``A2A_TRUSTED_SIGNERS`` is configured: with no trust anchors a card
cannot be verified either way, so the behaviour is unchanged except for a one-line warning. Once
trust anchors are configured, a card that carries a signature which does not verify against one of
them is refused — a present-but-invalid signature is an attack signal, never silently accepted.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Algorithms a card signature may use. "none" and symmetric HMAC (HS*) are never accepted for a
# card: the key is public, so an HS* "verification" would accept anyone who knows the public JWK.
_ALLOWED_ALGS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
                           "ES256", "ES256K", "ES384", "ES512", "EdDSA"})


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _reject_floats(obj: Any) -> None:
    """RFC 8785 number canonicalisation for non-integers (ECMAScript ``Number::toString``) is out of
    scope: the Agent Card schema carries no floating-point members, so a float is treated as a
    malformed card rather than silently serialised in a way a signer would not reproduce."""
    if isinstance(obj, float):
        raise ValueError("card contains a floating-point value")
    if isinstance(obj, dict):
        for value in obj.values():
            _reject_floats(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _reject_floats(value)


def canonicalize(card: dict) -> bytes:
    """RFC 8785 JCS bytes of the card **without** its ``signatures`` member (the detached payload the
    signatures are computed over). Objects, arrays, strings, booleans, null and integers only.

    ``json.dumps`` with sorted keys, compact separators and ``ensure_ascii=False`` is exactly RFC
    8785 for this value space: keys are ASCII so code-point order equals the spec's UTF-16 order,
    control characters escape as lowercase ``\\uXXXX`` / the short forms, and non-ASCII stays literal
    UTF-8. ``allow_nan=False`` rejects Infinity/NaN, which JSON cannot represent.
    """
    payload = {k: v for k, v in card.items() if k != "signatures"}
    _reject_floats(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _verify_one(entry: dict, payload_b64: str, trusted: dict) -> bool:
    """One JWS ``signatures`` entry against the trusted JWKs. True only on a real cryptographic match."""
    protected_b64 = entry.get("protected")
    signature_b64 = entry.get("signature")
    if not isinstance(protected_b64, str) or not isinstance(signature_b64, str):
        return False
    try:
        header = json.loads(_b64url_decode(protected_b64))
        signature = _b64url_decode(signature_b64)
    except Exception:
        return False
    if not isinstance(header, dict):
        return False
    alg = header.get("alg")
    if not isinstance(alg, str) or alg not in _ALLOWED_ALGS:
        return False
    import jwt.algorithms as algorithms

    registry = algorithms.get_default_algorithms()
    if alg not in registry:
        return False
    algo = registry[alg]
    signing_input = (protected_b64 + "." + payload_b64).encode("ascii")
    kid = header.get("kid")
    # The card author chooses the kid in its own header, so it is only a hint: try the kid-matched
    # trusted key first, then every other trusted key. A match is still a real signature check
    # against a key the operator configured, so a forged kid cannot help the attacker.
    candidates = []
    if isinstance(kid, str) and kid in trusted:
        candidates.append(trusted[kid])
    candidates.extend(jwk for k, jwk in trusted.items() if k != kid)
    for jwk in candidates:
        if not isinstance(jwk, dict):
            continue
        if isinstance(jwk.get("alg"), str) and jwk["alg"] != alg:
            continue  # a JWK that pins its own alg must match
        try:
            key = algo.from_jwk(json.dumps(jwk))
            if algo.verify(signing_input, key, signature):
                return True
        except Exception:
            continue
    return False


def verify_card_signatures(card: dict, trusted_signers: dict, *, require: bool) -> tuple[bool, str]:
    """``(trusted, reason)``. ``trusted`` is True when the card's endpoints and skills may be used.

    - **No trusted signers configured** — inert. Nothing can be verified, so the card is accepted
      (unchanged behaviour); a card that carries a signature logs a warning. ``require`` overrides
      this and refuses, since verification was asked for but is impossible.
    - **Card carries signatures** — at least one must verify against a trusted key, otherwise the
      card is refused. This holds even when ``require`` is off: a present-but-invalid signature
      (the forged-card attack) is never silently accepted.
    - **Unsigned card** — refused only when ``require`` is set.
    """
    if not isinstance(card, dict):
        return (not require), "no card"
    signatures = card.get("signatures")
    has_signatures = isinstance(signatures, list) and len(signatures) > 0

    if not trusted_signers:
        if require:
            return False, "A2A_REQUIRE_CARD_SIGNATURE is set but no A2A_TRUSTED_SIGNERS are configured"
        if has_signatures:
            logger.warning("A2A: peer Agent Card carries a signature but A2A_TRUSTED_SIGNERS is not "
                           "configured; the signature was not verified")
            return True, "unverified: no trusted signers configured"
        return True, "no signature to verify"

    if not has_signatures:
        if require:
            return False, "card has no signatures and A2A_TRUSTED_SIGNERS is configured"
        return True, "unsigned card accepted (A2A_REQUIRE_CARD_SIGNATURE is off)"

    try:
        payload_b64 = _b64url_encode(canonicalize(card))
    except Exception as exc:
        return False, f"card could not be canonicalised: {exc}"
    for entry in signatures:
        if isinstance(entry, dict) and _verify_one(entry, payload_b64, trusted_signers):
            return True, "signature verified against a trusted signer"
    return False, "no signature verified against a trusted signer (forged card or unknown key)"


def load_trusted_signers(raw: str) -> dict:
    """Parse ``A2A_TRUSTED_SIGNERS`` into ``{kid: jwk}``. The value is an inline JWKS JSON object
    (``{"keys": [ ... ]}``) or a path to a file holding one. A key with no ``kid`` is kept under a
    synthetic handle and still tried (see ``_verify_one``). Returns ``{}`` on anything unparseable,
    so a misconfiguration fails inert rather than blocking all peers."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    text = raw
    if not raw.startswith("{"):
        try:
            with open(raw, encoding="utf-8-sig") as handle:
                text = handle.read()
        except OSError as exc:
            logger.warning("A2A: A2A_TRUSTED_SIGNERS path %r could not be read: %s", raw, exc)
            return {}
    try:
        data = json.loads(text)
    except Exception as exc:
        logger.warning("A2A: A2A_TRUSTED_SIGNERS is not valid JSON: %s", exc)
        return {}
    keys = data.get("keys") if isinstance(data, dict) else None
    if not isinstance(keys, list):
        logger.warning("A2A: A2A_TRUSTED_SIGNERS must be a JWKS object with a 'keys' array")
        return {}
    out: dict = {}
    for index, jwk in enumerate(keys):
        if isinstance(jwk, dict):
            out[str(jwk.get("kid") or f"__nokid_{index}")] = jwk
    return out
