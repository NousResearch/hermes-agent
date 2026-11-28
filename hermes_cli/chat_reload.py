"""``/reload``: one refresh of the current chat (#159205).

Order: ``.env`` → plugins (forced rediscovery: changed plugin code is re-imported) → MCP servers →
skills → ONE rebuild of the chat's tool snapshot with ``preserve_prefix=False``, so plugin Python
tools and MCP tools are callable in this chat. A rebuilt tools[] breaks the provider prompt cache,
so surfaces run this only after the ``approvals.mcp_reload_confirm`` consent (the explicit
``/reload-mcp`` contract). Plugin system-prompt sections still apply from the next session: the
system prompt is never rebuilt mid-conversation. ``/reload env`` stays the ``.env``-only reload.

:func:`run_steps` + :func:`publish` serve the classic CLI's in-process agent and the TUI sidecar's
``reload.all``; the shared gateway owner composes its own verbs (``gateway/session_reload.py``)
into the same :class:`ReloadReport`. :func:`summary_lines` renders it for the user,
:func:`model_note` for the model (English, like the other reload notes).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)


@dataclass
class ReloadReport:
    """What one ``/reload`` did. ``plugins`` is ``plugins_activation.rediscover_plugins()``, ``mcp``
    ``{added, removed, reconnected, failed}``, ``skills`` ``agent.skill_commands.reload_skills()``;
    ``errors`` maps a step that raised to its error (the other steps still ran)."""

    env_updated: int = 0
    plugins: dict = field(default_factory=dict)
    mcp: dict = field(default_factory=dict)
    skills: dict = field(default_factory=dict)
    tools_added: list = field(default_factory=list)
    tools_removed: list = field(default_factory=list)
    tool_count: Optional[int] = None
    errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ReloadReport:
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


def reload_mcp_servers() -> dict:
    """``/reload-mcp``'s in-process pass: shut every server down, re-probe tool availability and
    rediscover from config (plugin servers included: discovery merges the plugin manager's)."""
    from tools.mcp_tool import _lock, _servers
    from tools.mcp_tool_agent import reprobe_tool_availability
    from tools.mcp_tool_discovery import discover_mcp_tools, mcp_servers_not_connected
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import _key_name

    def names():
        with _lock:
            return {_key_name(key) for key in _servers}
    old = names()
    shutdown_mcp_servers()
    reprobe_tool_availability()
    discover_mcp_tools()
    new = names()
    return {"added": sorted(new - old), "removed": sorted(old - new), "reconnected": sorted(new & old),
            "failed": mcp_servers_not_connected()}


def _step(report: ReloadReport, name: str, run: Callable[[], None]) -> None:
    try:
        run()
    except Exception as exc:  # health: allow BLE001 -- one failed step is reported; the rest still reload
        logger.warning("/reload: %s step failed", name, exc_info=True)
        report.errors[name] = str(exc) or type(exc).__name__


def run_steps(report: ReloadReport, *, mcp: Optional[Callable[[], dict]] = None,
              scope: Callable[[], Any] = contextlib.nullcontext) -> None:
    """``.env`` → plugins → MCP (*mcp*: the surface's own pass, default :func:`reload_mcp_servers`)
    → skills, in that order. *scope*
    binds the chat's profile around the plugin and skill rescans; ``.env`` keeps ``/reload env``'s
    process scope."""
    from hermes_cli.config import reload_env

    def env():
        report.env_updated = int(reload_env())

    def plugins():
        from hermes_cli.plugins_activation import rediscover_plugins
        with scope():
            report.plugins = rediscover_plugins()

    def skills():
        from agent.skill_commands import reload_skills
        with scope():
            report.skills = reload_skills()
    for name, run in (("env", env), ("plugins", plugins),
                      ("mcp", lambda: report.mcp.update((mcp or reload_mcp_servers)())),
                      ("skills", skills)):
        _step(report, name, run)


def tool_names(agent) -> set:
    return {(tool.get("function") or {}).get("name") for tool in getattr(agent, "tools", None) or ()} - {None}


def record_tools(report: ReloadReport, agent, before: set) -> None:
    after = tool_names(agent)
    report.tools_added, report.tools_removed = sorted(after - before), sorted(before - after)
    report.tool_count = len(after)


def publish(report: ReloadReport, agent, *, enabled_toolsets=None) -> Optional[list]:
    """Rebuild *agent*'s tool snapshot from the live registry with ``preserve_prefix=False`` (the
    consented cache break) and record the tools it gained or lost. A restricted *enabled_toolsets*
    takes the MCP servers this reload connected, as ``/reload-mcp`` does; returns that selection
    (None: unchanged). A failed rebuild is reported as the ``tools`` step."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    before = tool_names(agent)
    override = None
    if enabled_toolsets and "all" not in enabled_toolsets and "*" not in enabled_toolsets:
        servers = [*report.mcp.get("added", ()), *report.mcp.get("reconnected", ())]
        override = [*enabled_toolsets, *sorted(set(servers) - set(enabled_toolsets))]
    _step(report, "tools", lambda: refresh_agent_mcp_tools(
        agent, enabled_override=override, quiet_mode=True, preserve_prefix=False))
    record_tools(report, agent, before)
    return None if "tools" in report.errors else override


def _names(items) -> str:
    return ", ".join(str(i.get("name", "")) if isinstance(i, dict) else str(i) for i in items)


def _line(header: str, parts: list) -> str:
    return f"  {header}" + (f" — {'; '.join(parts)}" if parts else "")


def summary_lines(report: ReloadReport) -> list[str]:
    """The user's summary, one line per step (localized)."""
    lines = [_line(t("cli.reload_all.env", count=report.env_updated), [])] if "env" not in report.errors else []
    plugins = report.plugins
    if plugins:
        parts = [t(f"cli.reload_all.{k}", names=_names(plugins[k]))
                 for k in ("added", "removed", "changed") if plugins.get(k)]
        if plugins.get("failed"):
            parts.append(t("cli.reload_all.failed", names=", ".join(f"{n} ({e})" for n, e in plugins["failed"].items())))
        if plugins.get("restart"):
            parts.append(t("cli.reload_all.restart", names=", ".join(
                f"{n} ({', '.join(d)})" for n, d in plugins["restart"].items())))
        if plugins.get("prompt"):
            parts.append(t("cli.reload_all.prompt_next_session", names=_names(plugins["prompt"])))
        lines.append(_line(t("cli.reload_all.plugins", count=len(plugins.get("loaded") or ())), parts))
    mcp = report.mcp
    if mcp:
        connected = len(mcp.get("added") or ()) + len(mcp.get("reconnected") or ())
        if "connected" in mcp:  # the gateway owner reports its live set; it reconciles, never reconnects
            connected = len(mcp["connected"])
        parts = [t(f"cli.reload_all.{k}", names=_names(mcp[k]))
                 for k in ("added", "removed", "reconnected", "failed") if mcp.get(k)]
        lines.append(_line(t("cli.reload_all.mcp", count=connected), parts))
    skills = report.skills
    if skills:
        parts = [t(f"cli.reload_all.{k}", names=_names(skills[k])) for k in ("added", "removed") if skills.get(k)]
        lines.append(_line(t("cli.reload_all.skills", count=int(skills.get("total") or 0)), parts))
    if report.tool_count is not None:
        parts = [t(f"cli.reload_all.{k}", names=_names(getattr(report, f"tools_{k}")))
                 for k in ("added", "removed") if getattr(report, f"tools_{k}")]
        lines.append(_line(t("cli.reload_all.tools", count=report.tool_count), parts))
    lines.extend(f"  {t('cli.reload_all.step_failed', step=step, error=error)}" for step, error in report.errors.items())
    return lines


def model_note(report: ReloadReport) -> Optional[str]:
    """One note for the model's next turn, or None when nothing it can use changed."""
    parts = []
    for label, names in (("Tools added", report.tools_added), ("Tools removed", report.tools_removed),
                         ("MCP servers added", report.mcp.get("added")),
                         ("MCP servers removed", report.mcp.get("removed")),
                         ("Plugins changed", [*report.plugins.get("added", ()), *report.plugins.get("changed", ())]),
                         ("Plugins removed", report.plugins.get("removed")),
                         ("Skills added", report.skills.get("added")),
                         ("Skills removed", report.skills.get("removed"))):
        if names:
            parts.append(f"{label}: {_names(names)}")
    if not parts:
        return None
    return ("[USER INITIATED RELOAD: .env, plugins, MCP servers and skills were reloaded and this "
            "conversation's tool list was updated. " + ". ".join(parts)
            + ". Use skills_list to see the updated skill catalog.]")
