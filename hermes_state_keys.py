"""Pure identity derivations the canonical store and the gateway both need.

stdlib-only on purpose: ``hermes_state_*`` modules must not import ``gateway/`` to compute a
digest or parse a key (the gateway imports the store, so the reverse edge is a latent cycle).
The gateway modules that historically defined these re-export them from here.
"""
import hashlib
import json


def admission_fingerprint(*, canonical_target: str, payload: dict) -> str:
    """Canonical payload identity after authorization and immutable media capture."""
    encoded = json.dumps(
        {'target': canonical_target, 'payload': payload}, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8', errors='surrogatepass')
    return hashlib.sha256(encoded).hexdigest()


def local_identity(profile_id, principal_id, request_id):
    """The deterministic session id of a LOCAL creation receipt."""
    identity = json.dumps([profile_id, principal_id, request_id], separators=(',', ':'))
    return 'local-' + hashlib.sha256(identity.encode()).hexdigest()


def profile_from_session_key_namespace(namespace: str) -> str:
    """Inverse of ``gateway.session._session_key_namespace`` for the ``<ns>`` slot of a key:
    ``"default"`` for ``main``, ``"main"`` for the marked ``main~``, else the slot is the profile id."""
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def profile_from_session_key(session_key):
    """Profile namespace encoded in an ``agent:<ns>:...`` gateway session key, else None."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    return profile_from_session_key_namespace(parts[1] or "main")
