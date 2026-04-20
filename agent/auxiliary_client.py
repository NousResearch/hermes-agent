"""Legacy re-export shim for the auxiliary client (F-C3 step 1).

The implementation moved to ``agent/auxiliary/`` as a package. This
module stays in place so every existing caller that does
``from agent.auxiliary_client import <name>`` keeps resolving
without an edit. New code should import from ``agent.auxiliary``
directly.

``from agent.auxiliary.base import *`` skips underscore-prefixed
names, but several call sites in ``agent/`` and tests import
private helpers by name (``_OR_HEADERS``, ``_get_task_timeout``,
``_read_codex_access_token``, …). We re-export those explicitly
below so the shim is a drop-in replacement for the legacy module.
"""

from __future__ import annotations

from agent.auxiliary import *  # noqa: F401,F403
from agent.auxiliary.base import *  # noqa: F401,F403

# Explicit re-exports for private symbols that callers import by name.
# Keep this list synchronized with any new underscore-prefixed helper
# that becomes part of the external API surface.
from agent.auxiliary.base import (  # noqa: F401,E402
    _DEFAULT_AUX_TIMEOUT,
    _OR_HEADERS,
    _build_call_kwargs,
    _convert_content_for_responses,
    _current_custom_base_url,
    _force_close_async_httpx,
    _get_auxiliary_env_override,
    _get_auxiliary_provider,
    _get_cached_client,
    _get_task_timeout,
    _nous_api_key,
    _nous_base_url,
    _pool_runtime_api_key,
    _pool_runtime_base_url,
    _read_codex_access_token,
    _read_main_model,
    _read_nous_auth,
    _resolve_api_key_provider,
    _resolve_auto,
    _resolve_custom_runtime,
    _resolve_forced_provider,
    _resolve_task_provider_model,
    _select_pool_entry,
    _to_async_client,
    _try_anthropic,
    _try_codex,
    _try_custom_endpoint,
    _try_nous,
    _try_openrouter,
)
from agent.auxiliary.vision import (  # noqa: F401,E402
    _VISION_AUTO_PROVIDER_ORDER,
    _normalize_vision_provider,
    _preferred_main_vision_provider,
    _resolve_strict_vision_backend,
    _strict_vision_backend_available,
)
