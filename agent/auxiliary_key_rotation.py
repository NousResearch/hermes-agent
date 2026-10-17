"""Rotate the auxiliary main-runtime key after a credential refresh or pool rotation.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _pool_runtime_api_key(entry: Any) -> str:
    # runtime_api_key handles provider-specific fallback (e.g. agent_key for nous); None entry → "".
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def rotate_runtime_main_api_key(old: Any, new: Any) -> None:
    """Swap a revoked main key for its replacement in the published runtime, IN PLACE.

    A refresh can run inside a request worker's copied Context, where rebinding the ContextVar
    would stay invisible to the turn thread; the published dict is shared, so mutating it is not.
    """
    from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT, _normalize_api_key

    # The legacy mirrors are left alone: _compat_runtime_main() ignores them while they equal the compat
    # snapshot, and republishing from here cannot tell which runtime published them.
    runtime = _RUNTIME_MAIN_CONTEXT.get()
    if isinstance(runtime, dict) and runtime.get("api_key") == old:
        runtime["api_key"] = _normalize_api_key(new)


def pool_retry_key(provider: str, failed_key: str = "") -> str:
    """Key the pool hands the next request for ``provider`` ("" when unknown), without mutating selection.

    Warns when the pool has no distinct key to offer: the retry then re-sends ``failed_key``.
    """
    from agent.auxiliary_client import _normalize_aux_provider, _peek_pool_entry

    normalized = _normalize_aux_provider(provider)
    key = _pool_runtime_api_key(_peek_pool_entry(normalized))
    if failed_key and (not key or key == failed_key):
        logger.warning("Auxiliary: %s pool recovered but offers no other key; the retry re-sends the failed one",
                       normalized)
    return key


def swap_retry_key(retry_kwargs: dict[str, Any], failed_key: str, new_key: str) -> dict[str, Any]:
    """Point a post-rotation retry at ``new_key`` wherever it still carries ``failed_key``.

    The main-provider route pins aux calls to the main session's key. Without the swap the retry
    re-sends the key the pool just benched, and its error is then charged to the healthy entry
    the pool rotated to.
    """
    if not failed_key or not new_key or failed_key == new_key:
        return retry_kwargs
    from agent.auxiliary_client import _normalize_main_runtime

    swapped = dict(retry_kwargs)
    runtime = _normalize_main_runtime(retry_kwargs.get("main_runtime"))
    if runtime.get("api_key") == failed_key:
        swapped["main_runtime"] = dict(runtime, api_key=new_key)
    if retry_kwargs.get("resolved_api_key") == failed_key:
        swapped["resolved_api_key"] = new_key
    return swapped
