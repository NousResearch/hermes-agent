"""Public, immutable per-dispatch context for native plugin slash commands.

The object is constructed by a host surface (CLI, messaging gateway, TUI/Desktop
backend) at dispatch time and never resolves a session from process-global state.
Plugins receive provenance and may request a tool through the host-provided
dispatcher; no live agent, gateway, or adapter object leaks through this API.

``dispatch_tool`` is deliberately the only execution capability. Surfaces bind it
to that session's canonical agent tool path (request/execution middleware, plugin
pre/post tool hooks, edit/approval guards) — never a bare registry dispatch — and
an absent context fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, FrozenSet

#: Host-supplied executor: ``fn(name, args) -> tool result`` (JSON string or dict).
ToolDispatcher = Callable[[str, dict[str, Any]], str | dict[str, Any]]


@dataclass(frozen=True)
class PluginInvocation:
    """Immutable provenance and authority for one plugin slash-command call.

    Fields are populated by the invoking surface:

    - ``session_id`` / ``session_key``: the host session identity (empty when the
      surface has no active session).
    - ``surface``: ``"cli"``, ``"gateway"``, ``"tui"``, ``"desktop"``, ...
    - ``platform``: the delivery platform (``"cli"``, ``"telegram"``, ...).
    - ``cwd`` / ``workspace``: the session's resolved working directory and
      workspace root, when the surface knows them (else ``None``).
    - ``tool_names``: the effective tool set of the session's agent; a dispatch
      to anything else is refused.
    - ``authorized``: whether this context may execute tools at all.

    ``dispatch_tool`` runs only when *authorized*, the tool is in ``tool_names``,
    and the surface supplied a session-bound dispatcher; any missing piece raises
    rather than silently degrading.
    """

    session_id: str = ""
    session_key: str = ""
    surface: str = ""
    platform: str = ""
    cwd: Path | None = None
    workspace: Path | None = None
    tool_names: FrozenSet[str] = frozenset()
    authorized: bool = False
    _dispatch: ToolDispatcher | None = field(default=None, repr=False, compare=False)

    def dispatch_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        """Run one tool through the active session's normal tool path.

        Raises ``PermissionError`` when this invocation is not authorized or the
        tool is outside the session's effective tool set, ``RuntimeError`` when the
        surface provided no session-bound dispatcher, and ``TypeError`` for
        non-dict arguments. A dispatcher may raise whatever the underlying tool
        path raises.
        """
        if not self.authorized:
            raise PermissionError("Plugin command tool dispatch is not authorized in this session")
        if self._dispatch is None:
            raise RuntimeError("No session-bound tool dispatcher is available")
        if name not in self.tool_names:
            raise PermissionError(f"Tool {name!r} is not available in this session")
        if not isinstance(args, dict):
            raise TypeError("Tool arguments must be a dictionary")
        return self._dispatch(name, args)
