"""User-layer normalization that must run BEFORE the DEFAULT_CONFIG merge.

``hermes_cli/config`` applies these to the merged user layer (config.yaml + config.d drop-ins)
before the schema defaults overlay it. They live here rather than in the facade because the
schema default for a key can inject a present-but-None value that defeats a presence check —
promoting a legacy key AFTER the defaults merge would then find the key "already set" and
silently drop the user's value.
"""
from __future__ import annotations

from typing import Any


def promote_legacy_max_turns(user_layer: dict[str, Any]) -> dict[str, Any]:
    """Move a legacy root-level ``max_turns`` under ``agent`` when the nested key is unset.

    Distinct from ``config._normalize_max_turns_config``, which runs later on the fully merged
    config and only promotes when ``agent.max_turns`` is ABSENT: by then ``DEFAULT_CONFIG`` has
    injected ``agent.max_turns: None``, so a drop-in's root ``max_turns`` would be dropped.
    Here the nested key counts as unset when it is None, which is what the pre-defaults layer
    can actually contain.
    """
    if "max_turns" not in user_layer:
        return user_layer
    agent_user_config = dict(user_layer.get("agent") or {})
    if agent_user_config.get("max_turns") is None:
        agent_user_config["max_turns"] = user_layer["max_turns"]
    user_layer["agent"] = agent_user_config
    user_layer.pop("max_turns", None)
    return user_layer
