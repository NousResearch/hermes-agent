"""Session-owned plugin toolsets (#110515): the public registration surface for tools whose
definition, visibility and lifetime belong to one gateway session (realtime clients that
hand the server a per-session function catalog). Backs ``PluginContext.session_toolset()``;
the registry (``tools/registry.py``) owns visibility gating and teardown semantics."""

import logging
import re
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SESSION_TOOLSET_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


class SessionToolsetRegistrar:
    """Collects tool registrations into one session's catalog.

    ``register_tool`` mirrors ``PluginContext.register_tool``'s argument shape minus the
    global-registry concerns (override/shadow/scope) that do not apply: a session tool can
    never shadow anything outside its own session slot. Late registration after the
    session's first model request raises RuntimeError (prompt-cache stability)."""

    def __init__(self, registry, session_key: str, toolset: str, scope: str,
                 description: str = "", default_direct: bool = True):
        self._registry = registry
        self._session_key = session_key
        self._toolset = toolset
        self._scope = scope
        self.description = description
        self._default_direct = bool(default_direct)
        self._count = 0

    @property
    def toolset(self) -> str:
        return self._toolset

    @property
    def registered_count(self) -> int:
        return self._count

    def register_tool(
            self, name: str, schema: dict, handler: Callable, *, check_fn: Optional[Callable] = None,
            requires_env: Optional[list] = None, is_async: bool = False, description: str = "",
            emoji: str = "", direct: Optional[bool] = None) -> None:
        self._registry.register_session_tool(
            self._session_key, name, self._toolset, schema, handler, check_fn=check_fn,
            requires_env=requires_env, is_async=is_async, description=description,
            emoji=emoji, direct=self._default_direct if direct is None else direct)
        self._count += 1


@contextmanager
def session_toolset(ctx, session_key: str, *, name: str, description: str = "", direct: bool = True):
    """Open a session-owned toolset for *session_key* (see ``PluginContext.session_toolset``).

    Teardown removes every registration made through the yielded registrar — and only
    those, so sibling sessions' catalogs are untouched. Teardown runs on body exception
    too: a crashed session must not leak tools into the process."""
    from tools.registry import registry

    toolset_name = str(name or "").strip().lower()
    if not _SESSION_TOOLSET_NAME_RE.fullmatch(toolset_name):
        raise ValueError(
            f"session_toolset name must match [a-z0-9][a-z0-9_-]{{0,63}}, got {name!r}")
    if not str(session_key or "").strip():
        raise ValueError("session_toolset requires a non-empty session_key")

    # Capture the registry's authoritative profile key at enter: teardown must target the
    # same slot even if the calling thread's context has moved to another profile by then.
    scope = registry.current_scope_key()
    registrar = SessionToolsetRegistrar(
        registry, str(session_key).strip(), toolset_name, scope, description=str(description or ""),
        default_direct=bool(direct))
    logger.debug("Plugin %s opened session toolset %r for session %r",
                 ctx.plugin_id, toolset_name, session_key)
    try:
        yield registrar
    finally:
        removed = registry.remove_session_tools(str(session_key).strip(), scope=scope)
        if removed:
            logger.debug("Plugin %s closed session toolset %r for session %r (%d tools removed)",
                         ctx.plugin_id, toolset_name, session_key, removed)
