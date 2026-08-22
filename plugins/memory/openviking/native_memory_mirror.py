"""Stable mirroring for Hermes native MEMORY.md / USER.md entries.

Hermes' built-in memory tool identifies entries by unique text substrings rather
than durable IDs. OpenViking, by contrast, needs an exact ``viking://`` file URI
to update or delete a memory safely. This module keeps the missing identity map
in a small profile-scoped registry and serializes mirror operations through one
FIFO worker.

The registry is intentionally narrow: it tracks only memories created through
the built-in-memory mirror. Session-extracted memories and explicit
``viking_remember`` writes are outside its ownership and are never guessed at or
deleted by similarity.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils import atomic_json_write

logger = logging.getLogger("plugins.memory.openviking")

_REGISTRY_VERSION = 1
_REGISTRY_RELATIVE_PATH = Path("openviking") / "memory_mirror_registry.json"
_SUPPORTED_ACTIONS = frozenset({"add", "replace", "remove"})
_POLL_SECONDS = 0.05


class _MappingError(RuntimeError):
    """A destructive mirror operation could not resolve one exact URI."""


class NativeMemoryMirror:
    """FIFO, profile-scoped mirror of Hermes native memory into OpenViking."""

    def __init__(self, provider: Any):
        self._provider = provider
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._state_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._shutting_down = False

    def enqueue(
        self,
        action: str,
        target: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        subdir: str,
    ) -> None:
        """Queue one committed built-in-memory mutation in call order."""
        if action not in _SUPPORTED_ACTIONS:
            return
        if action in {"add", "replace"} and not content:
            return

        event = {
            "action": action,
            "target": str(target or "memory"),
            "content": str(content or ""),
            "metadata": dict(metadata or {}),
            "subdir": str(subdir),
        }

        with self._state_lock:
            if self._shutting_down:
                logger.warning(
                    "OpenViking memory mirror skipped %s during provider shutdown",
                    action,
                )
                return
            self._queue.put(event)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name="openviking-memory-mirror",
                )
                self._worker.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain queued mirror operations, then stop the FIFO worker."""
        with self._state_lock:
            self._shutting_down = True
            worker = self._worker

        if worker is None:
            return

        deadline = time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

        remaining = max(0.0, deadline - time.monotonic())
        if worker.is_alive() and remaining:
            worker.join(timeout=remaining)
        if worker.is_alive():
            logger.warning("OpenViking memory mirror worker did not drain before shutdown")

    def _run(self) -> None:
        while True:
            with self._state_lock:
                stopping = self._shutting_down
            if stopping and self._queue.empty():
                return

            try:
                event = self._queue.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue

            try:
                self._apply(event)
            except _MappingError as exc:
                logger.warning("OpenViking memory mirror skipped: %s", exc)
            except Exception as exc:
                logger.warning("OpenViking memory mirror failed: %s", exc)
            finally:
                self._queue.task_done()

    def _registry_path(self) -> Path:
        root = str(getattr(self._provider, "_hermes_home", "") or "").strip()
        if not root:
            from hermes_constants import get_hermes_home

            root = str(get_hermes_home())
        return Path(root) / _REGISTRY_RELATIVE_PATH

    def _load_registry(self) -> Dict[str, Any]:
        path = self._registry_path()
        if not path.exists():
            return {"version": _REGISTRY_VERSION, "entries": []}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot read mirror registry {path}: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("version") != _REGISTRY_VERSION:
            raise RuntimeError(f"unsupported or invalid mirror registry: {path}")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise RuntimeError(f"invalid mirror registry entries: {path}")

        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(f"invalid mirror registry entry: {path}")
            if not all(isinstance(entry.get(key), str) for key in ("target", "uri", "content")):
                raise RuntimeError(f"invalid mirror registry entry fields: {path}")
        return payload

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        path = self._registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, registry, mode=0o600)

    @staticmethod
    def _resolve_mapping(
        registry: Dict[str, Any],
        *,
        target: str,
        old_text: str,
        action: str,
    ) -> tuple[int, Dict[str, str]]:
        old_text = str(old_text or "").strip()
        if not old_text:
            raise _MappingError(f"{action} requires old_text for stable URI resolution")

        matches = [
            (index, entry)
            for index, entry in enumerate(registry["entries"])
            if entry["target"] == target and old_text in entry["content"]
        ]
        if not matches:
            raise _MappingError(
                f"{action} has no stable OpenViking URI mapping for target={target!r}; "
                "leaving OpenViking unchanged"
            )
        if len(matches) != 1:
            raise _MappingError(
                f"{action} matched {len(matches)} OpenViking URI mappings for "
                f"target={target!r}; leaving OpenViking unchanged"
            )
        return matches[0]

    def _apply(self, event: Dict[str, Any]) -> None:
        action = event["action"]
        target = event["target"]
        content = event["content"]
        metadata = event["metadata"]
        registry = self._load_registry()
        client = self._provider._new_client()

        if action == "add":
            # The native store treats exact duplicate adds as idempotent. Mirror
            # the same way if a duplicate notification ever reaches us.
            if any(
                entry["target"] == target and entry["content"] == content
                for entry in registry["entries"]
            ):
                return

            requested_uri = self._provider._build_memory_uri(event["subdir"])
            response = client.post(
                "/api/v1/content/write",
                {"uri": requested_uri, "content": content, "mode": "create"},
            )
            result = response.get("result", {}) if isinstance(response, dict) else {}
            canonical_uri = (
                str(result.get("uri") or "").strip()
                if isinstance(result, dict)
                else ""
            ) or requested_uri
            registry["entries"].append(
                {"target": target, "uri": canonical_uri, "content": content}
            )
            self._save_registry(registry)
            return

        old_text = metadata.get("old_text")
        index, mapping = self._resolve_mapping(
            registry,
            target=target,
            old_text=str(old_text or ""),
            action=action,
        )
        uri = mapping["uri"]

        if action == "replace":
            client.post(
                "/api/v1/content/write",
                {"uri": uri, "content": content, "mode": "replace", "wait": True},
            )
            registry["entries"][index] = {
                "target": target,
                "uri": uri,
                "content": content,
            }
            self._save_registry(registry)
            return

        client.delete(
            "/api/v1/fs",
            params={"uri": uri, "recursive": False, "wait": True},
        )
        registry["entries"].pop(index)
        self._save_registry(registry)


_MIRROR_ATTR = "_native_memory_mirror"


def enqueue_native_memory_write(
    provider: Any,
    action: str,
    target: str,
    content: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    subdir: str,
) -> None:
    """Lazily create the provider's FIFO mirror and enqueue one operation."""
    mirror = getattr(provider, _MIRROR_ATTR, None)
    if mirror is None:
        mirror = NativeMemoryMirror(provider)
        setattr(provider, _MIRROR_ATTR, mirror)
    mirror.enqueue(
        action,
        target,
        content,
        metadata=metadata,
        subdir=subdir,
    )


def shutdown_native_memory_mirror(provider: Any, timeout: float = 5.0) -> None:
    """Drain the provider's native-memory mirror if it was ever used."""
    mirror = getattr(provider, _MIRROR_ATTR, None)
    if mirror is not None:
        mirror.shutdown(timeout=timeout)
