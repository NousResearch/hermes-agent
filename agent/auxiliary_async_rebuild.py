"""Carry a sync OpenAI client's credential and configured headers onto its async rebuild.

``_to_async_client`` recreates ``AsyncOpenAI`` from a resolved sync client. Two things do not
live on ``.api_key`` / the header set it recomputes: a per-request token provider (``key_cmd``,
Entra ID — the SDK parks the callable in ``_api_key_provider`` and leaves ``.api_key`` empty, so
an ``auth_headers`` built from the snapshot is ``{}`` and the request carries NO Authorization at
all) and the ``default_headers`` the client was constructed with (a named provider's
``extra_headers``). See #109595.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Dict

from openai import OpenAI


def async_api_key(sync_client: Any) -> Any:
    """Credential for the async twin: the sync token provider wrapped for ``await``, else the static key.

    The sync provider is ``Callable[[], str]``; ``AsyncOpenAI`` awaits its provider before every request,
    so the callable is run off-loop (it may shell out to a ``key_cmd``).
    """
    provider = getattr(sync_client, "_api_key_provider", None) if isinstance(sync_client, OpenAI) else None
    if not callable(provider):
        return sync_client.api_key

    async def _provide() -> str:
        return str(await asyncio.to_thread(provider))

    return _provide


def configured_default_headers(sync_client: Any) -> Dict[str, str]:
    """The ``default_headers`` mapping the sync client was constructed with (SDK ``_custom_headers``).

    SECURITY: values may carry credentials — never log them.
    """
    configured = getattr(sync_client, "_custom_headers", None) if isinstance(sync_client, OpenAI) else None
    return dict(configured) if isinstance(configured, Mapping) else {}
