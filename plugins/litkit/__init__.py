"""LitKit toolset plugin: registers the ``litkit`` toolset from :mod:`litco.litkit.tools`.

A bundled ``kind: backend`` plugin, so it auto-loads with no ``plugins.enabled`` entry. The
tools stay hidden (``check_fn``) until ``LITCO_INSTANCE_URL`` and ``LITCO_AGENT_TOKEN`` are set,
so a Hermes install that is not a matter host never sees them.
"""

from __future__ import annotations

__all__ = ["register"]


def register(ctx) -> None:
    from litco.litkit.tools import register as register_litkit

    register_litkit(ctx)
