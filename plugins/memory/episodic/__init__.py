"""Episodic Memory Provider plugin — bridges memory/episodic_provider.py
into the plugins/memory/ discovery system.

This file exists solely so the plugin loader can discover and instantiate
EpisodicMemoryProvider. The actual implementation lives in
hermes-agent/memory/episodic_provider.py.
"""

from memory.episodic_provider import EpisodicMemoryProvider

__all__ = ["EpisodicMemoryProvider"]
