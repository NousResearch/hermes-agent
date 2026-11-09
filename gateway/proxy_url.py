"""The ``gateway.proxy_url`` a turn forwards through, read from a config that may not be a mapping.

A profile config carries its own status line (``gateway: running``) under the same top-level key the
nested proxy settings live in, and the policy hands that config back verbatim — so every reader of a
nested ``gateway.*`` setting has to treat a non-mapping ``gateway:`` as "not configured" instead of
calling ``.get`` on a string. Lives outside ``gateway/run_turn.py`` because that facade is over its
``FILE_LINES`` cap and may only shrink; the resolution stays in one place for its callers and tests.
"""

from __future__ import annotations

from typing import Any, Optional


def configured_proxy_url(config: Optional[dict] = None) -> Optional[str]:
    """``gateway.proxy_url`` stripped, or None: absent, empty, or a status scalar where the mapping goes.

    *config* defaults to the gateway's loaded config (imported lazily: ``gateway.run`` imports this
    module's callers, so the config loader cannot be a module-level import).
    """
    if config is None:
        from gateway.run import _load_gateway_config
        config = _load_gateway_config()
    section: Any = (config or {}).get("gateway")
    value = (section.get("proxy_url") if isinstance(section, dict) else None) or ""
    return str(value).strip() or None
