"""Pure nested configuration lookup."""
from typing import Any, Optional

def cfg_get(cfg: Optional[dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Traverse nested dict keys safely, returning ``default`` on any miss.
    Explicit ``None`` values are returned as-is (``dict.get`` semantics: ``default`` only when the
    key is absent). Named ``cfg_get`` to avoid shadowing the ubiquitous ``cfg_path`` local."""
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
