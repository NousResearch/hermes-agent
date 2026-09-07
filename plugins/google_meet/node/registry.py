"""Local JSON registry of approved remote meet nodes.

``$HERMES_HOME/workspace/meetings/nodes.json``::

    {"nodes": {"<name>": {"url": "ws://host:port", "token": "...", "added_at": <epoch>}}}
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

from plugins.google_meet._jsonfile import read_json, write_json_atomic


def _default_path() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings" / "nodes.json"


class NodeRegistry:
    """File-backed registry; single writer assumed (the gateway CLI)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else _default_path()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """The ``nodes`` map (name → entry); empty when the file is missing or malformed."""
        data = read_json(self.path)
        nodes = data.get("nodes") if isinstance(data, dict) else None
        return nodes if isinstance(nodes, dict) else {}

    def _save(self, nodes: Dict[str, Dict[str, Any]]) -> None:
        write_json_atomic(self.path, {"nodes": nodes})

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        entry = self._load().get(name)
        return None if entry is None else {"name": name, **entry}

    def add(self, name: str, url: str, token: str) -> None:
        for label, value in (("node name", name), ("url", url), ("token", token)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        nodes = self._load()
        nodes[name] = {"url": url, "token": token, "added_at": time.time()}
        self._save(nodes)

    def remove(self, name: str) -> bool:
        nodes = self._load()
        if name not in nodes:
            return False
        del nodes[name]
        self._save(nodes)
        return True

    def list_all(self) -> List[Dict[str, Any]]:
        return [{"name": name, **entry} for name, entry in sorted(self._load().items())]

    def resolve(self, chrome_node: Optional[str]) -> Optional[Dict[str, Any]]:
        """Named node's entry, or (``chrome_node`` falsy) the sole registered node; None if unknown
        or when zero / several nodes are registered (ambiguous)."""
        if chrome_node:
            return self.get(chrome_node)
        nodes = self.list_all()
        return nodes[0] if len(nodes) == 1 else None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
