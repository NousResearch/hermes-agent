"""Local JSON registry of approved remote meet nodes.

Lives at ``$HERMES_HOME/workspace/meetings/nodes.json``. The gateway
consults it to resolve a ``chrome_node`` name to a ``(url, token)`` pair
before opening a WebSocket to the remote bot host.

Schema
------
    {
      "nodes": {
        "<name>": {
          "url":   "ws://host:port",
          "token": "...",
          "added_at": <epoch_float>
        }
      }
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


def _default_path() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings" / "nodes.json"


class NodeRegistry:
    """Simple file-backed registry. Not concurrent-safe across processes
    — single writer assumed (the gateway CLI)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else _default_path()

    # ----- storage ------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"nodes": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"nodes": {}}
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), dict):
            return {"nodes": {}}
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ----- public API ---------------------------------------------------

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        entry = data["nodes"].get(name)
        if entry is None:
            return None
        return {"name": name, **entry}

    def add(self, name: str, url: str, token: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("node name must be a non-empty string")
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        data = self._load()
        data["nodes"][name] = {
            "url": url,
            "token": token,
            "added_at": time.time(),
        }
        self._save(data)

    def remove(self, name: str) -> bool:
        data = self._load()
        if name in data["nodes"]:
            del data["nodes"][name]
            self._save(data)
            return True
        return False

    def list_all(self) -> List[Dict[str, Any]]:
        data = self._load()
        out: List[Dict[str, Any]] = []
        for name, entry in sorted(data["nodes"].items()):
            out.append({"name": name, **entry})
        return out

    def resolve(self, chrome_node: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve a node name to its entry.

        If ``chrome_node`` is provided, return that named node (or None).
        If ``chrome_node`` is None, return the sole registered node when
        exactly one is registered; otherwise return None (ambiguous or
        empty).
        """
        if chrome_node:
            return self.get(chrome_node)
        nodes = self.list_all()
        if len(nodes) == 1:
            return nodes[0]
        return None
