"""Host service facades on ``PluginContext``: subagent lifecycle and the in-process agent pool.

Split out of ``hermes_cli/plugins.py`` so the facade keeps shrinking rather than growing: each
property is a lazily built, plugin-safe service object, never a live agent or private registry.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any


class PluginServiceCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``manifest``, ``plugin_id``)."""

    #: The owning context supplies this (``PluginContext.plugin_id``).
    plugin_id: str

    @cached_property
    def subagent_lifecycle(self) -> Any:
        """Plugin-safe subagent lifecycle service: serializable handles and immutable snapshots,
        never a live agent or private registry."""
        from agent.subagent_lifecycle import SubagentLifecycleService, get_active_subagent_parent
        return SubagentLifecycleService(get_active_subagent_parent)

    @cached_property
    def agent_loop(self) -> Any:
        """The host's in-process agent pool (``agent.agent_loop_service.PluginAgentLoop``).

        ``ctx.agent_loop.run(message, **params)`` blocks and returns an ``AgentRunResult``;
        ``submit(message, **params)`` returns a job whose ``future`` settles with it. Each call
        runs on its own pool thread (bounds: ``agent_loop.max_workers`` / ``max_queued``), with
        the caller's profile scope, and the keyword parameters are
        :func:`agent.agent_run.run`'s. ``capability_version`` names the surface so a plugin can
        feature-detect."""
        from agent.agent_loop_service import PluginAgentLoop
        return PluginAgentLoop(self.plugin_id)
