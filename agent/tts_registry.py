"""TTS Provider Registry.

Registered plugin TTS providers, populated at import-time via
:meth:`PluginContext.register_tts_provider` and consulted by :mod:`tools.tts_tool`
only when ``tts.provider`` is neither a built-in nor a command-type provider.
Built-ins always win: a colliding plugin name is rejected here with a warning
(re-checked at dispatch). Command-providers-win-over-plugins is enforced by the
dispatcher (a name in the user's config.yaml is more specific than a plugin).
"""

from __future__ import annotations

import logging

from agent.provider_registry import ProviderRegistry, lower_key
from agent.tts_provider import TTSProvider
from speech.command_provider import BUILTIN_TTS_PROVIDERS as _BUILTIN_NAMES

logger = logging.getLogger(__name__)


def _warn_builtin_collision(key: str) -> None:
    logger.warning(
        "TTS provider '%s' shadows a built-in name; registration ignored. "
        "Built-in TTS providers (%s) always win — pick a different name.",
        key, ", ".join(sorted(_BUILTIN_NAMES)),
    )


# Case-insensitive, whitespace-tolerant keys mirror how
# ``tools.tts_tool._get_provider`` normalizes the configured ``tts.provider``.
_registry: ProviderRegistry[TTSProvider] = ProviderRegistry(
    label="TTS", provider_cls=TTSProvider, logger=logger, normalize=lower_key,
    builtin_names=_BUILTIN_NAMES, on_builtin_collision=_warn_builtin_collision,
)
_registry.export(globals())
