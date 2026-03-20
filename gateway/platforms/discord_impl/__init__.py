# Internal implementation package for Discord v2.
# The public import surface remains gateway.platforms.discord (discord.py).
# External consumers should NEVER import from this package directly.
# Imports flow: discord.py -> discord_impl.*, not the reverse.
"""Internal implementation package for Discord v2.

This package exists only to house Discord-specific implementation modules
behind the stable public adapter surface in ``gateway.platforms.discord``.
It is not a public API.
"""
