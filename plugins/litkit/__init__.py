"""LitKit toolset plugin: registers the ``litkit`` toolset from :mod:`litco.litkit.tools`.

A bundled ``kind: backend`` plugin, so it auto-loads with no ``plugins.enabled`` entry. The
tools stay hidden (``check_fn``) until ``LITCO_INSTANCE_URL`` and ``LITCO_AGENT_TOKEN`` are set,
so a Hermes install that is not a matter host never sees them.

It also registers :func:`litco.held.held_tool_result`, which acts only during a LitCo turn: a
command held for approval reaches the model as "held, not run, not a failure".
"""

from __future__ import annotations

__all__ = ["register"]


def register(ctx) -> None:
    from litco.held import held_tool_result
    from litco.litkit.tools import register as register_litkit

    register_litkit(ctx)
    ctx.register_hook("transform_tool_result", held_tool_result)
