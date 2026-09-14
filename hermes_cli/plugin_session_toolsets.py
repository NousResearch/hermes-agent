"""Session-owned tool registrations for out-of-tree plugins."""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any, Callable


_LOCK = threading.RLock()
_SESSION_TOOLSETS: dict[tuple[str, str], set["PluginSessionToolset"]] = {}


def session_toolset_names(session_key: str, *, scope: str) -> list[str]:
    """Freeze and return the plugin toolsets owned by one gateway session."""
    with _LOCK:
        handles = tuple(_SESSION_TOOLSETS.get((scope, session_key), ()))
        for handle in handles:
            handle._frozen = True
        return sorted(handle.name for handle in handles if handle.active)


def active_session_toolset_names(*, scope: str) -> set[str]:
    """Return all live session-owned names so generic plugin defaults can exclude them."""
    with _LOCK:
        return {
            handle.name
            for (owner_scope, _session_key), handles in _SESSION_TOOLSETS.items()
            if owner_scope == scope
            for handle in handles
            if handle.active
        }


class PluginSessionToolset:
    """A disposable, cache-stable set of tools owned by one gateway session."""

    def __init__(
        self, context: Any, session_key: str, name: str, description: str, *, direct: bool,
    ) -> None:
        if not isinstance(session_key, str) or not session_key:
            raise ValueError("session_key must be a non-empty string")
        clean_name = re.sub(r"[^a-z0-9_-]+", "-", str(name).strip().lower()).strip("-")
        if not clean_name:
            raise ValueError("session toolset name must contain a letter or digit")
        self._context = context
        self.session_key = session_key
        self.description = str(description)
        self.scope = context._manager.scope_key
        digest = hashlib.sha256(
            f"{context.plugin_id}\0{self.scope}\0{session_key}\0{clean_name}".encode()
        ).hexdigest()[:16]
        self.name = f"plugin-{clean_name}-{digest}"
        self._direct = bool(direct)
        self._registrations: list[Any] = []
        self._tool_names: set[str] = set()
        self._frozen = False
        self._active = True
        with _LOCK:
            owners = _SESSION_TOOLSETS.setdefault((self.scope, self.session_key), set())
            if any(handle.name == self.name and handle.active for handle in owners):
                raise ValueError(f"session toolset already exists: {clean_name}")
            owners.add(self)
        from tools.registry import registry
        registry.register_session_toolset(
            self.name, self.description, direct=self._direct, scope=self.scope,
        )

    @property
    def active(self) -> bool:
        return self._active

    def register_tool(
        self, name: str, schema: dict, handler: Callable, *, is_async: bool = False,
        description: str = "", emoji: str = "",
    ) -> str:
        """Register one tool and return its session-unique model-facing name."""
        with _LOCK:
            if not self._active:
                raise RuntimeError("session toolset is closed")
            if self._frozen:
                raise RuntimeError("session toolset is frozen for prompt-cache stability")
            logical_name = str(name).strip()
            if not logical_name or logical_name in self._tool_names:
                raise ValueError(f"duplicate or empty session tool name: {logical_name!r}")
            digest = hashlib.sha256(f"{self.name}\0{logical_name}".encode()).hexdigest()[:16]
            readable = re.sub(r"[^A-Za-z0-9_-]+", "_", logical_name)[:40] or "tool"
            registry_name = f"pst_{digest}_{readable}"[:64]

            registration = self._context.register_tool(
                name=registry_name, toolset=self.name, schema=dict(schema), handler=handler,
                is_async=is_async, description=description, emoji=emoji,
                _owner_session_key=self.session_key,
            )
            if registration is None:
                raise RuntimeError(f"failed to register session tool: {logical_name}")
            self._registrations.append(registration)
            self._tool_names.add(logical_name)
            return registry_name

    def dispose(self) -> None:
        """Remove all registrations without affecting sibling sessions."""
        with _LOCK:
            if not self._active:
                return
            self._active = False
            handles = self._registrations[::-1]
            self._registrations.clear()
            owners = _SESSION_TOOLSETS.get((self.scope, self.session_key))
            if owners is not None:
                owners.discard(self)
                if not owners:
                    _SESSION_TOOLSETS.pop((self.scope, self.session_key), None)
        for handle in handles:
            handle.dispose()
        from tools.registry import registry
        registry.deregister_session_toolset(self.name, scope=self.scope)

    close = dispose

    def __enter__(self) -> "PluginSessionToolset":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.dispose()
