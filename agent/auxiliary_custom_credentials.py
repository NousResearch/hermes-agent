"""Credential selection for an explicit custom endpoint.

The route that names a URL owns its credential: the same-origin main key or the
``providers.<id>`` entry's own key. An ambient ``OPENAI_API_KEY`` is never a route credential,
because sending a live OpenAI key to a third-party endpoint is credential misdirection
(#145452).
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
