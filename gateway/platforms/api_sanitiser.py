"""
Response sanitiser for the Remote Management API (`/api/*`).

The endpoints introduced in #23742 ship per-user identity content
verbatim:

* ``/api/memory`` returns ``user.content`` (the full USER.md
  profile).
* ``/api/sessions`` returns each row's ``system_prompt`` (the
  persona + skills index + the SAME USER.md content embedded
  inside, several KB per row).
* ``/api/sessions/{id}/messages`` repeats the ``role: "system"``
  content on every system message.

Bearer-token auth is the only access gate. A leaked or stolen
token exposes the user's identity profile in one request.

This module is the hard redaction layer applied as a
post-serialisation step on every response from the four affected
handlers. It is **not** opt-out and not opt-in: the Remote
Management surface is structural-only by policy.

If a legitimate identity-read flow ever needs to be added (e.g. a
desktop Persona-tab that wants to *render* USER.md content), it
should arrive via a separate auth class (loopback token,
SSH-tunnel-gated header, etc.) — not via re-loosening this filter.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

__all__ = [
    "sanitize_value",
    "sanitize_response",
    "REDACTED",
]

# Sentinel string that replaces any leaked identity / secret value.
# Tests assert against this literal — keep stable.
REDACTED = "<redacted>"

# Layer 1: field-name heuristic. Case-insensitive substring match
# against the *key name*, not the value. When any of these
# substrings appears in a key, the entire value is replaced with
# the sentinel.
_SECRET_FIELD_PATTERNS: Tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "bearer",
    # Note: deliberately NOT including "auth" — the API server
    # already ships `auth: {type: bearer, required: true}` in
    # /v1/capabilities, and stripping that map would lie to clients
    # about how auth works. The bearer-token value itself is in a
    # field named "key" / "token" which the patterns above catch.
)

# Layer 2a: URL with embedded basic-auth credentials.
# https://user:pass@host/path → https://<redacted>@host/path
_URL_CRED_RE = re.compile(
    r"(https?://)[^/\s@]+:[^/\s@]+@",
    re.IGNORECASE,
)

# Layer 2b: Token-in-querystring patterns.
# ?api_key=xxx&q=foo → ?api_key=<redacted>&q=foo
_TOKEN_QS_RE = re.compile(
    r"([?&](?:api_key|apikey|token|access_token|secret)=)[^&\s]+",
    re.IGNORECASE,
)

# Layer 3: hard identity-field redaction. Applied at the
# response-payload level after the structural pass.
#
# - Top-level fields whose VALUE is the identity content
#   (replace the value with REDACTED, preserve siblings).
_IDENTITY_FIELDS: Tuple[str, ...] = (
    "system_prompt",
)

# - Nested object paths: dict[key1][key2] = REDACTED.
#   Used for /api/memory which nests user content one level deep.
_IDENTITY_OBJECT_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("user", "content"),
)


# ---------------------------------------------------------------------------
# Field-name heuristic
# ---------------------------------------------------------------------------


def _looks_like_secret_field(name: str) -> bool:
    """True when a key name suggests the value is a secret."""
    n = name.lower()
    return any(pattern in n for pattern in _SECRET_FIELD_PATTERNS)


# ---------------------------------------------------------------------------
# Structural pass — recursive walk
# ---------------------------------------------------------------------------


def sanitize_value(value: Any, _field: str = "") -> Any:
    """Apply the structural filters to one value.

    * Field-name secret detection → wholesale replacement.
    * String content → URL-credential + querystring-token regex.
    * Dicts/lists → recursive walk; field name carries down.

    Does NOT touch identity-block fields here — that pass runs on
    the full payload after the structural walk completes (see
    :func:`sanitize_response`).
    """
    if _looks_like_secret_field(_field):
        return REDACTED
    if isinstance(value, str):
        v = _URL_CRED_RE.sub(r"\1" + REDACTED + "@", value)
        v = _TOKEN_QS_RE.sub(r"\1" + REDACTED, v)
        return v
    if isinstance(value, dict):
        return {k: sanitize_value(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Identity-block pass
# ---------------------------------------------------------------------------


def _redact_identity_fields_in_dict(d: Dict[str, Any]) -> None:
    """In-place: replace any top-level identity field with REDACTED."""
    for field in _IDENTITY_FIELDS:
        if field in d:
            d[field] = REDACTED


def _redact_identity_object_paths_in_dict(d: Dict[str, Any]) -> None:
    """In-place: walk known identity-object paths and redact the leaf."""
    for path in _IDENTITY_OBJECT_PATHS:
        node: Any = d
        for segment in path[:-1]:
            if not isinstance(node, dict) or segment not in node:
                node = None
                break
            node = node[segment]
        leaf_key = path[-1]
        if isinstance(node, dict) and leaf_key in node:
            node[leaf_key] = REDACTED


def _walk_and_redact_identity(value: Any) -> Any:
    """Recursively apply identity redaction to any dict found in the
    response.  Sessions list responses nest the dict at
    ``data["sessions"][n]``; we don't want to hardcode the nesting,
    so we walk."""
    if isinstance(value, dict):
        # Also redact `role: system` message bodies — system-role
        # messages carry the same identity content as
        # ``system_prompt`` and travel in
        # ``/api/sessions/{id}/messages``.
        if value.get("role") == "system" and "content" in value:
            value["content"] = REDACTED
        _redact_identity_fields_in_dict(value)
        _redact_identity_object_paths_in_dict(value)
        for v in value.values():
            _walk_and_redact_identity(v)
    elif isinstance(value, list):
        for item in value:
            _walk_and_redact_identity(item)
    return value


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sanitize_response(payload: Any) -> Any:
    """Top-level sanitiser for a Remote Management API response.

    Two passes:

    1. Structural pass (:func:`sanitize_value`) — recursive walk;
       replaces secrets by field name + scrubs URL credentials +
       querystring tokens.
    2. Identity pass — recursively walks the payload again and
       redacts known identity fields (``system_prompt``,
       ``user.content``, ``role: "system"`` message ``content``).

    The two passes don't conflict: pass 1 doesn't touch the
    identity fields (their names don't match the secret heuristic),
    pass 2 doesn't touch structural data.

    Returns a new sanitised value; the original payload is not
    mutated. Safe to call on whatever shape Codex's handlers
    return — dict, list, primitive.
    """
    structural = sanitize_value(payload)
    # Identity pass mutates in-place on the new value returned by
    # the structural pass (sanitize_value already produces fresh
    # containers via dict/list comprehensions, so no shared state
    # with the caller's input).
    _walk_and_redact_identity(structural)
    return structural


# Explicit endpoint scope for /v1/capabilities advertising.
# Kept as module-level tuples so the declared policy and the
# handlers stay in lockstep — any future expansion of the
# sanitised surface must update these tuples AND wire the new
# handler through sanitize_response().
_SANITISED_ENDPOINTS: Tuple[str, ...] = (
    "/api/memory",
    "/api/sessions",
    "/api/sessions/{id}/messages",
)

# Endpoints deliberately NOT in scope of this PR. SOUL.md is the
# owner's direct edit target and stays pass-through so the
# PUT / RESET round-trip works; structural sanitisation of SOUL
# (URL credentials + querystring tokens, no identity-block
# redaction) is a planned follow-up.
_OUT_OF_SCOPE_ENDPOINTS: Tuple[str, ...] = (
    "/api/profiles/{name}/soul",
)


def declared_policy() -> Dict[str, Any]:
    """Policy summary for /v1/capabilities advertising.

    Stays in this module so a future policy change only needs to
    update one source of truth. Clients consume
    ``sanitised_endpoints`` to decide whether to render
    "click to reveal" affordances per endpoint (always false for
    the listed three); ``out_of_scope`` makes it explicit that
    a missing endpoint from ``sanitised_endpoints`` is not an
    accidental omission but a deliberate scope decision.
    """
    return {
        "enabled": True,
        "identity_blocks_redacted": True,
        "opt_in_supported": False,
        "sanitised_endpoints": list(_SANITISED_ENDPOINTS),
        "out_of_scope": list(_OUT_OF_SCOPE_ENDPOINTS),
    }
