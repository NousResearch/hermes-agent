"""Detect an actively-configured Honcho install for memory-provider auto-migration.

When ``memory.provider`` is not set in the user's config but the Honcho
plugin's own ``honcho.json`` shows the integration is enabled with at
least one credential, the agent auto-activates Honcho as the memory
provider.  This module exposes the read-only detection step; persistence
to the user's main config is done by the caller AFTER the provider
passes ``is_available()`` so a broken Honcho setup never writes a stale
``memory.provider: honcho`` entry.
"""

from __future__ import annotations


def detect_honcho_auto_migrate() -> str:
    """Return ``"honcho"`` when the Honcho client is actively configured.

    Conditions:
      - ``HonchoClientConfig.from_global_config()`` resolves successfully
      - ``enabled`` is True
      - At least one credential is present (``api_key`` or ``base_url``)

    Returns ``""`` in every other case (plugin not installed, disabled,
    or no credentials) so the caller can use the truthy check
    ``if not _mem_provider_name: _mem_provider_name = detect_honcho_auto_migrate()``.
    """
    try:
        from plugins.memory.honcho.client import HonchoClientConfig as _HCC
        _hcfg = _HCC.from_global_config()
        if _hcfg.enabled and (_hcfg.api_key or _hcfg.base_url):
            return "honcho"
    except Exception:
        pass
    return ""
