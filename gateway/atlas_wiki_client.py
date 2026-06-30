"""WIKI-SLACK P6-B — Atlas wiki HTTP client + deterministic name→IRI resolver.

Fail-soft by construction (mirrors ``clarify_relay``'s env + aiohttp pattern):
when ``ATLAS_BASE_URL`` is unset, or Atlas times out / 5xx's, ``fetch_entity_page``
returns a ``{"degraded": True, ...}`` marker and NEVER raises — the /wiki
command shows "Atlas unreachable" rather than guessing. A 404 returns
``{"not_found": True, "iri": ...}``.

CROSS-REPO SLUG CONTRACT: ``resolve_entity_iri`` vendors the army-of-one
``core/ids.slugify`` rules (NFKD ASCII-fold, lowercase, hyphenate, collapse).
A change to army-of-one ``backend/src/atlas/core/ids.py`` slugify (or the
``ATLAS_ORG`` namespace) MUST be mirrored here — the golden pairs in
``tests/gateway/test_atlas_wiki_client.py`` pin the contract so drift fails CI.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

# Mirrored from army-of-one backend/src/atlas/core/namespaces.py
ATLAS_ORG = "https://atlas.blakeaber.dev/org/"

# Mirrored from army-of-one backend/src/atlas/core/ids.py
_SLUG_STRIP = re.compile(r"[^a-z0-9-]+")
_SLUG_COLLAPSE = re.compile(r"-+")

_BASE_URL_ENV = "ATLAS_BASE_URL"
_TOKEN_ENV = "ATLAS_BEARER_TOKEN"
_TIMEOUT_S = 8.0


def slugify(text: str, *, max_len: int = 64) -> str:
    """Deterministic slug — byte-for-byte the army-of-one core/ids.slugify."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_only = ascii_only.replace(" ", "-")
    stripped = _SLUG_STRIP.sub("-", ascii_only)
    collapsed = _SLUG_COLLAPSE.sub("-", stripped).strip("-")
    return collapsed[:max_len] or ""


def resolve_entity_iri(name: str) -> str:
    """Map a raw entity name to its deterministic org IRI (slug-based).

    Mirrors army-of-one ``org_iri(canonical_name)`` = ``ATLAS_ORG + slug``.
    Org is the default entity class for the pre-call /wiki use case.
    """
    return f"{ATLAS_ORG}{slugify(name)}"


async def fetch_entity_page(iri: str, *, viewer: str = "blake") -> dict[str, Any]:
    """GET {ATLAS_BASE_URL}/v1/wiki/{iri}?viewer=<viewer>, fail-soft.

    Returns the page dict on 200; ``{"not_found": True, "iri": iri}`` on 404;
    ``{"degraded": True, "reason": ...}`` when Atlas is unconfigured/unreachable.
    Never raises.
    """
    import urllib.parse  # noqa: PLC0415

    base = os.environ.get(_BASE_URL_ENV, "").strip().rstrip("/")
    if not base:
        return {"degraded": True, "reason": f"{_BASE_URL_ENV} unset"}

    url = f"{base}/v1/wiki/{iri}?viewer={urllib.parse.quote(viewer, safe='')}"
    headers = {"accept": "application/json"}
    token = os.environ.get(_TOKEN_ENV, "").strip()
    if token:
        headers["authorization"] = f"Bearer {token}"

    try:
        import aiohttp  # noqa: PLC0415

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 404:
                    return {"not_found": True, "iri": iri}
                return {
                    "degraded": True,
                    "reason": f"atlas returned {resp.status}",
                    "iri": iri,
                }
    except Exception as exc:  # noqa: BLE001 — fail-soft: never raise to the caller
        return {"degraded": True, "reason": str(exc), "iri": iri}


__all__ = ["ATLAS_ORG", "fetch_entity_page", "resolve_entity_iri", "slugify"]
