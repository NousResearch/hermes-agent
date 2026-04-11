"""CLI-only client for Nowledge Mem.

Shells out to ``nmem --json <args>``. The CLI handles server URL, API key,
and remote access configuration, so this client has zero config surface.

If ``nmem`` is not installed, ``is_available`` returns False and the
provider gracefully disables tools. On machines running the Nowledge Mem
desktop app, ``nmem`` is already bundled. Otherwise: ``pip install nmem-cli``.

No external dependencies: stdlib only (subprocess, json).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NowledgeMemClient:
    """Thin wrapper around the ``nmem`` CLI.

    Every domain method builds CLI args and calls ``nmem --json <args>``.
    The CLI owns auth, server URL, and remote config.
    """

    def __init__(self, timeout: int = 30, *, space: Optional[str] = None) -> None:
        self._timeout = timeout
        self._has_explicit_space = isinstance(space, str)
        self._space = space.strip() if isinstance(space, str) else None

    @staticmethod
    def is_available() -> bool:
        """Return True if ``nmem`` CLI is on PATH and responds."""
        try:
            result = subprocess.run(
                ["nmem", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def health(self) -> bool:
        """Check that Nowledge Mem is reachable."""
        try:
            result = subprocess.run(
                ["nmem", "--json", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def working_memory(self) -> Dict[str, Any]:
        """Fetch the user's Working Memory briefing."""
        return self._cli(["wm", "read"])

    def search(
        self,
        query: str = "",
        *,
        limit: int = 10,
        filter_labels: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> Any:
        """Search memories. CLI requires a query string."""
        if not query:
            return {"memories": []}
        cmd = ["m", "search", query]
        if limit != 10:
            cmd.extend(["-n", str(limit)])
        if filter_labels:
            for label in filter_labels:
                cmd.extend(["-l", label])
        if mode == "deep":
            cmd.extend(["--mode", "deep"])
        return self._cli(cmd)

    def save(
        self,
        content: str,
        *,
        memory_id: Optional[str] = None,
        title: Optional[str] = None,
        importance: Optional[float] = None,
        labels: Optional[List[str]] = None,
        unit_type: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Any:
        """Save a new memory, or upsert by memory_id."""
        cmd = ["m", "add", content]
        cmd.extend(["-s", "hermes"])
        if memory_id:
            cmd.extend(["--id", memory_id])
        if title:
            cmd.extend(["-t", title])
        if importance is not None:
            cmd.extend(["-i", str(importance)])
        if labels:
            for label in labels:
                cmd.extend(["-l", label])
        if unit_type:
            cmd.extend(["--unit-type", unit_type])
        if event_date:
            cmd.extend(["--event-start", event_date])
        return self._cli(cmd)

    def update(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        title: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> Any:
        """Update an existing memory (content, title, importance)."""
        cmd = ["m", "update", memory_id]
        if content:
            cmd.extend(["-c", content])
        if title:
            cmd.extend(["-t", title])
        if importance is not None:
            cmd.extend(["-i", str(importance)])
        return self._cli(cmd)

    def delete(self, memory_id: str) -> Any:
        """Delete a single memory."""
        return self._cli(["m", "delete", memory_id, "-f"])

    def delete_many(self, memory_ids: List[str]) -> Any:
        """Delete multiple memories."""
        if not memory_ids:
            raise ValueError("memory_ids must not be empty")
        return self._cli(["m", "delete", *memory_ids, "-f"])

    def thread_search(
        self,
        query: str = "",
        *,
        limit: int = 10,
        source: Optional[str] = None,
    ) -> Any:
        """Search past conversations. CLI requires a query string."""
        if not query:
            return {"threads": []}
        cmd = ["t", "search", query]
        if limit != 10:
            cmd.extend(["-n", str(limit)])
        if source:
            cmd.extend(["--source", source])
        return self._cli(cmd)

    def thread_messages(
        self,
        thread_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Any:
        """Fetch messages from a thread."""
        cmd = ["t", "show", thread_id]
        cmd.extend(["-n", str(limit)])
        if offset:
            cmd.extend(["--offset", str(offset)])
        return self._cli(cmd)

    def _cli(self, args: List[str]) -> Any:
        """Run ``nmem --json <args>`` and return parsed JSON."""
        cmd = ["nmem", "--json"] + args
        env = os.environ.copy()
        if self._has_explicit_space:
            env.pop("NMEM_SPACE", None)
            env.pop("NMEM_SPACE_ID", None)
        if self._has_explicit_space and self._space:
            env["NMEM_SPACE"] = self._space
            env["NMEM_SPACE_ID"] = self._space
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "nmem CLI not found. Install: pip install nmem-cli, "
                "or enable CLI in Nowledge Mem: Settings > Developer Tools"
            ) from None
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(stderr or f"nmem exited with code {result.returncode}")
        output = result.stdout.strip()
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"nmem returned non-JSON output: {output[:200]}"
            ) from error
