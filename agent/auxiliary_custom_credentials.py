"""Credential selection for an explicit custom endpoint.

The route that names a URL owns its credential, in this order: the main key when the URL shares the
main endpoint's origin (so it also wins over an entry owning another path on that host), then the key
of the ``providers.<id>`` entry owning the exact URL, then the host-gated ambient ``OPENAI_API_KEY``
(#91308), which therefore never reaches a third-party endpoint that declares its own key (#145452).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _normalize_api_key(raw: Any) -> str | Callable[[], str]:
    """A key_cmd/Entra callable passes through uncalled; strings are stripped; anything else is ''."""
    if callable(raw) and not isinstance(raw, str):
        return raw
    return raw.strip() if isinstance(raw, str) else ""


def declared_custom_key_for_base_url(base_url: str, provider: str) -> Any:
    """The credential declared by the ``providers:`` entry owning *base_url*, else ``""``.

    A bare ``custom`` route can name a URL that a config-declared provider owns; that entry's own
    key_env/api_key/key_cmd must beat the ambient ``OPENAI_API_KEY`` fallback. A local server
    entry that declares no credential stays keyless (its placeholder is never a borrowed key).
    """
    try:
        from agent.auxiliary_client import _named_custom_api_key
        from agent.custom_entry_pool import custom_entry_for_endpoint, entry_declares_credential
    except ImportError:
        logger.debug("Declared custom credential imports unavailable", exc_info=True)
        return ""
    try:
        found = custom_entry_for_endpoint(base_url, provider)
        if not found:
            return ""
        name, entry = found
        if not entry_declares_credential(entry):
            return ""
        key = _named_custom_api_key(entry, provider or name, base_url)
        return "" if key == "no-key-required" else key
    except Exception:
        logger.debug("Declared custom credential lookup failed for %s", base_url, exc_info=True)
        return ""


def declared_custom_credential_identity(base_url: str, provider: str) -> str:
    """Client-cache identity of the credential the entry owning *base_url* declares, else ``""``.

    Built from the entry's pointers (inline api_key, the key_env value, ``cmd:<key_cmd>``) without
    minting a key_cmd token or selecting from its credential pool, so a cache lookup never runs a
    helper or advances pool rotation; a rotated key_env value still changes it.
    """
    from agent.auxiliary_client import _scoped_key_env
    from agent.custom_entry_pool import custom_entry_for_endpoint, entry_declares_credential

    found = custom_entry_for_endpoint(base_url, provider)
    if not found or not entry_declares_credential(found[1]):
        return ""
    entry = found[1]
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    return "\0".join((str(entry.get("api_key") or "").strip(), _scoped_key_env(key_env) if key_env else "",
                      f"cmd:{str(entry.get('key_cmd') or '').strip()}"))


def fast_catalog_provider(runtime: dict[str, Any], main_provider: str) -> str:
    """Provider whose catalog supplies the fast model: a ``custom`` runtime resolved from a
    ``providers.<id>`` entry that owns its URL reads that entry's catalog with its own key (#145452)."""
    requested = str(runtime.get("requested_provider") or "").strip()
    if main_provider != "custom" or not requested or requested == "custom":
        return main_provider
    from agent.custom_entry_pool import custom_entry_for_endpoint, named_custom_entry
    from hermes_cli.auth import is_known_auth_provider
    found = named_custom_entry(requested)
    owns = found is not None and custom_entry_for_endpoint(str(runtime.get("base_url") or ""), requested) == found
    # The entry's ``providers:`` key (``custom:<name>`` spellings included), never a built-in's: a
    # registry name would read that provider's catalog and send its fast model to this entry's URL.
    key = str(found[1].get("provider_key") or "") if owns else ""
    return key if key and not is_known_auth_provider(key) else main_provider
