"""Muse CLI login reuse for the Meta Model API provider family.

Background
----------
Hermes' ``meta-ai`` provider (``plugins/model-providers/meta-ai``) only knows
API-key auth (``MODEL_API_KEY`` / ``META_API_KEY`` / ``META_MODEL_API_KEY``),
i.e. pay-as-you-go billing.  The official ``muse`` CLI, however, supports a
second billing path: ``muse login`` (Meta-account OAuth device-code flow)
provisions a subscription-bound credential and ``META_API_KEY`` takes
priority over it (``muse login --help``).  Users on the Power Usage
subscription therefore pay per-token in Hermes while the same model
(``muse-spark-1.3-contributor``) bills to their subscription in ``muse``.

What ``muse login`` stores
--------------------------
On macOS the login session lives in the login keychain as a generic-password
item, service ``ai.meta.dev.credentials``, account ``meta``, holding JSON::

    {"secret_schema_version": 1, "api_key": "LLM|...", "access_token": "dca..."}

``api_key`` (``LLM|...``, login-provisioned — note the ``|``, distinct from
dev-console pay-as-you-go ``LLM_...`` keys) authenticates to
``https://api.meta.ai/v1`` as a Bearer token exactly like ``MODEL_API_KEY``
(verified: ``GET /v1/models`` returns the same 7-model catalog with both).
``access_token`` (``dca...``) is NOT an inference credential (401 on
``/v1/models``) and is ignored here.  The provisioned key is stable — the
payload carries no expiry/refresh material — so a read-through fallback with
a short TTL cache is sufficient; no refresh flow is needed.

Resolution order (enforced by the caller in ``hermes_cli.auth``)
---------------------------------------------------------------
Explicit configuration always beats implicit cross-app reuse, mirroring
``muse``' own priority rule:

1. env (``MODEL_API_KEY`` → ``META_API_KEY`` → ``META_MODEL_API_KEY``),
2. Hermes credential pool (``hermes auth add`` entries),
3. this module (local ``muse login`` session) — last resort only.

Only the last-resort branch calls into this module, so machines without a
Muse login never spawn a subprocess (cf. #60800), and setting any explicit
Meta key disables the keychain read entirely.

Platform notes
--------------
macOS reads via the ``security`` CLI.  The non-macOS ``muse`` credential
fallback-file location is undocumented, so other platforms currently resolve
to "no login found"; that gap is a deliberate follow-up, not silent
misbehavior.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# Keychain identity used by `muse login` (macOS login keychain).
MUSE_KEYCHAIN_SERVICE = "ai.meta.dev.credentials"
MUSE_KEYCHAIN_ACCOUNT = "meta"

# Provider ids served by the bundled meta-ai plugin (name + aliases).
MUSE_LOGIN_PROVIDER_IDS = frozenset({
    "meta-ai",
    "meta",
    "muse",
    "muse-spark",
    "model-api",
    "msl",
})

# Source label reported alongside the resolved key (mirrors the
# "env:VAR" / "credential_pool:<id>" labels used by _resolve_api_key_provider_secret).
MUSE_LOGIN_SOURCE = "muse-login:keychain"

# TTL for the in-process login lookup cache (seconds).  Caches both hits and
# misses: a miss must not re-spawn `security` on every resolution, and a hit
# re-reads periodically so a `muse login` rotation is picked up.
_LOGIN_CACHE_TTL_SECONDS = 300.0

_cache_checked_at = 0.0
_cache_key = ""
_cache_source = ""


def _reset_login_cache() -> None:
    """Clear the in-process lookup cache (tests only)."""
    global _cache_checked_at, _cache_key, _cache_source
    _cache_checked_at = 0.0
    _cache_key = ""
    _cache_source = ""


def _read_keychain_payload(timeout_seconds: float = 10.0) -> dict | None:
    """Return the parsed `muse login` keychain payload, or None.

    macOS only.  Never logs secret material.
    """
    if sys.platform != "darwin":
        return None
    if shutil.which("security") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                MUSE_KEYCHAIN_SERVICE,
                "-a",
                MUSE_KEYCHAIN_ACCOUNT,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except Exception as exc:
        logger.debug("muse login keychain read failed: %s", type(exc).__name__)
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads((proc.stdout or "").strip())
    except Exception:
        logger.debug("muse login keychain payload is not JSON")
        return None
    return payload if isinstance(payload, dict) else None


def read_muse_login_key() -> tuple[str, str]:
    """Return ``(api_key, source)`` from the local `muse login` session.

    Returns ``("", "")`` when there is no usable login (wrong platform, no
    `security` helper, no keychain item, unparseable payload, or empty key).
    Results are cached in-process for ``_LOGIN_CACHE_TTL_SECONDS``.
    """
    global _cache_checked_at, _cache_key, _cache_source
    now = time.monotonic()
    if now - _cache_checked_at < _LOGIN_CACHE_TTL_SECONDS:
        return _cache_key, _cache_source

    key, source = "", ""
    try:
        from hermes_cli.auth import has_usable_secret

        payload = _read_keychain_payload()
        candidate = (payload or {}).get("api_key")
        if has_usable_secret(candidate):
            key, source = str(candidate).strip(), MUSE_LOGIN_SOURCE
    except Exception as exc:
        logger.debug("muse login resolution failed: %s", type(exc).__name__)
        key, source = "", ""

    _cache_checked_at = now
    _cache_key, _cache_source = key, source
    return key, source
