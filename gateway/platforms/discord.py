"""Compatibility wrapper for the Discord platform adapter.

The runtime adapter lives under the bundled Discord platform plugin, but legacy
gateway tests and imports still use ``gateway.platforms.discord``.
"""

from plugins.platforms.discord.adapter import *  # noqa: F401,F403
