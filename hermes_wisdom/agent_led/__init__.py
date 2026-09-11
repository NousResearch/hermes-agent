"""Agent-led Collective Wisdom sharing.

This package layers an agent-authored review on top of the deterministic
qualification signals in :mod:`hermes_wisdom.qualification`. Nothing in it
publishes, installs, or mutes on its own: every outbound action passes
through the existing owner-consent gates in :mod:`hermes_wisdom.service`.
"""

from __future__ import annotations

from .policy import AgentLedPolicy, load_policy

__all__ = ["AgentLedPolicy", "load_policy"]
