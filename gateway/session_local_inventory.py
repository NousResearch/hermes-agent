"""Main's banner / status-bar inventory for a canonical local session.

``tui_gateway/server.py::_session_info`` on main put the session's tools (by toolset), skills (by
category), MCP servers, version / release date, update notice, git branch, project and provider on
every ``session.info`` / create / resume ``info``; the Ink banner (``components/branding.tsx``) and
the classic CLI banner (``hermes_cli/banner.py::build_welcome_banner``) render them. A canonical
session's agent is built on its first turn, so the inventory is computed here from the session's
frozen policy (the toolsets that turn will run with), with main's own builders, in the owner
process. Every builder does I/O (skills tree walk, tool ``check_fn`` probes, git), so callers run
:func:`session_inventory` off the owner loop (:func:`snapshot_info`).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)

_update_prefetch_started = False


def _start_update_prefetch():
    """Main's TUI gateway started the passive update check at import; the owner starts it once,
    on the first snapshot, so ``update_behind`` reports like main's banner instead of staying null."""
    global _update_prefetch_started
    if _update_prefetch_started:
        return
    _update_prefetch_started = True
    from hermes_cli.banner import prefetch_update_check
    prefetch_update_check()


def _grouped(tools):
    """``{toolset: [tool names]}`` exactly as main's ``_session_info`` grouped ``agent.tools``."""
    from model_tools import get_toolset_for_tool
    grouped: dict[str, list[str]] = {}
    for tool in tools or []:
        name = tool["function"]["name"]
        grouped.setdefault(get_toolset_for_tool(name) or "other", []).append(name)
    return grouped


def _policy_tools(policy, config):
    """The tool list the session's next agent build loads (``agent/agent_init.py`` tool snapshot):
    the policy's toolsets minus the configured and platform-gated disabled ones, one-shot pruned."""
    import model_tools
    from agent.oneshot_footprint import ONESHOT_HIDDEN_TOOLS
    from agent.skill_utils import parse_config_string_list
    from toolsets import session_disabled_toolsets, session_platform_tool_drops
    enabled = list(policy.current_toolsets(config))
    disabled = session_disabled_toolsets(
        parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets")) or None, policy.platform)
    tools = model_tools.get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True)
    drops = set(session_platform_tool_drops(policy.platform))
    if policy.source == "oneshot":
        drops |= ONESHOT_HIDDEN_TOOLS
    return enabled, [tool for tool in tools or [] if tool["function"]["name"] not in drops]


def _project(cwd):
    """``tui_gateway/server.py::_project_info_for_cwd``: the named project owning *cwd*, or None."""
    if not str(cwd or "").strip():
        return None
    from hermes_cli import projects_db as pdb
    try:
        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, cwd)
    except Exception:  # health: allow BLE001 -- main's resolver: an unreadable projects.db names no project
        logger.debug("failed to resolve project for cwd", exc_info=True)
        return None
    return None if project is None else {
        "id": project.id, "slug": project.slug, "name": project.name, "primary_path": project.primary_path}


def _cli_startup_notices(enabled):
    """The lines main's HermesCLI printed under its banner, computed where the tools run: the toolset
    availability warning (``cli.py::_show_tool_availability_warnings``) and the Browser Use downgrade
    notice (``_show_browser_backend_notice``). The classic CLI renders them with its own console."""
    notices: dict = {"tool_availability_warnings": [], "browser_backend_notice": None}
    with contextlib.suppress(Exception):
        from hermes_cli.tool_availability_notices import (
            current_terminal_backend, filter_to_enabled_toolsets, tool_availability_warning_lines)
        from model_tools import check_tool_availability
        from tools.terminal_tool import terminal_backend_unavailable_reason
        from toolsets import resolve_toolset
        _, unavailable = check_tool_availability()
        unavailable = filter_to_enabled_toolsets(unavailable, enabled or [], resolve_toolset)
        notices["tool_availability_warnings"] = tool_availability_warning_lines(
            unavailable, terminal_reason=terminal_backend_unavailable_reason(),
            terminal_backend=current_terminal_backend())
    with contextlib.suppress(Exception):
        from tools.browser_use_cli import default_downgrade_notice
        notices["browser_backend_notice"] = default_downgrade_notice()
    return notices


def _unknown_toolsets(enabled, config):
    """Main's "Unknown toolsets" startup warning (``HermesCLI._init_toolsets``), judged where the plugins
    load: the session's toolset names the profile's registry, MCP servers, plugin toolsets and platform
    bundles cannot place (an uninstalled plugin's stale ``platform_toolsets`` entry)."""
    if "all" in enabled or "*" in enabled:
        return []
    with contextlib.suppress(Exception):
        from hermes_cli.toolset_validation import saved_toolset_resolver
        known = saved_toolset_resolver(config)
        return [name for name in enabled if not known(name)]
    return []


def _cli_skills():
    """``hermes_cli.banner.get_available_skills`` as main's classic CLI computed it: ``cmd_chat`` pins
    ``HERMES_KANBAN_BOARD`` before HermesCLI starts, so ``environments: [kanban]`` skills are offered.
    The owner never sets that env; the kanban relevance is scoped to this scan (a ContextVar) and the
    catalog cache entry it would poison is restored, so other surfaces keep their own view."""
    from tools import skills_tool
    from tools.kanban_toolset_context import scoped_kanban_toolset_selection
    key = ("filtered", False)
    saved = skills_tool._SKILLS_CACHE.pop(key, None)
    try:
        with scoped_kanban_toolset_selection(["kanban"]):
            found = skills_tool._find_all_skills()
    finally:
        skills_tool._SKILLS_CACHE.pop(key, None)
        if saved is not None:
            skills_tool._SKILLS_CACHE[key] = saved
    grouped: dict[str, list[str]] = {}
    for skill in found:
        grouped.setdefault(skill.get("category") or "general", []).append(skill["name"])
    return grouped


def session_inventory(authority, policy, agent, cwd):
    """Main's banner/status fields for one session. Each section degrades to main's empty value on
    its own, so one failing builder never blanks the others (main wrapped each in ``suppress``)."""
    from gateway.session_authorities import owner_scope
    from tui_gateway import git_probe
    info: dict = {"tools": {}, "skills": {}, "mcp_servers": [], "version": "", "release_date": "",
                  "update_behind": None, "update_command": "",
                  "branch": git_probe.branch(cwd) if cwd else "", "project": _project(cwd)}
    with owner_scope(authority):
        config = policy.config(authority, secrets=False) if policy is not None else {}
        if agent is not None:
            from agent.voice_turn_route import session_runtime_view
            view = session_runtime_view(agent)
            info["provider"] = getattr(view, "provider", "") or ""
            info["tools"] = _grouped(getattr(agent, "tools", None))
            info["system_prompt"] = getattr(agent, "_cached_system_prompt", "") or ""
            compressor = getattr(agent, "context_compressor", None)
            if compressor is not None and getattr(compressor, "context_length", None):
                info["context_length"] = compressor.context_length
            enabled = list(getattr(agent, "enabled_toolsets", None) or [])
        else:
            info["provider"] = str((config.get("model") or {}).get("provider") or "")
            enabled = []
            if policy is not None:
                try:
                    enabled, tools = _policy_tools(policy, config)
                    info["tools"] = _grouped(tools)
                except Exception:  # health: allow BLE001 -- main's suppress: the banner shows no tools, not an error
                    logger.warning("session inventory: tool snapshot failed", exc_info=True)
        info["enabled_toolsets"] = sorted(enabled)
        with contextlib.suppress(Exception):
            # The classic CLI banner colours lazy (yellow) / disabled (red) tools from this.
            from hermes_cli.banner import compute_toolset_availability
            info["tool_availability"] = compute_toolset_availability(enabled)
        if policy is not None and policy.source == "cli":
            info.update(_cli_startup_notices(enabled), unknown_toolsets=_unknown_toolsets(enabled, config))
        with contextlib.suppress(Exception):
            if policy is not None and policy.source == "cli":
                info["skills"] = _cli_skills()
            else:
                from hermes_cli.banner import get_available_skills
                info["skills"] = get_available_skills()
        with contextlib.suppress(Exception):
            from tools.mcp_tool_discovery import get_mcp_status
            info["mcp_servers"] = get_mcp_status()
    with contextlib.suppress(Exception):
        from hermes_cli import __release_date__
        from hermes_cli.version_info import get_version_info
        info.update(version=get_version_info().base_version, release_date=__release_date__)
    with contextlib.suppress(Exception):
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command
        _start_update_prefetch()
        # Two assignments, as on main: a failing command lookup still reports update_behind.
        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    return info


async def snapshot_info(authority, ref):
    """``local_session_info`` plus main's inventory, the builders run off the owner loop."""
    from gateway.session_local import local_session_info
    from gateway.session_policy import policy_for_source
    info = local_session_info(authority, ref)
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    agent = authority.agent(ref)
    inventory = await asyncio.to_thread(session_inventory, authority, policy, agent, info.get("cwd"))
    if info.get('tool_count') is None:
        # A lazy session: the tools its first turn loads (the banner's own count).
        info['tool_count'] = sum(len(names) for names in (inventory.get('tools') or {}).values())
    return {**inventory, **info}


def agent_inventory(agent) -> dict:
    """The banner fields only a built agent knows (main's ``_session_info`` read them off it): the
    tools it loaded, its cached system prompt (the ``▸ System Prompt`` row), provider and context
    length. Cheap attribute reads, safe on the agent thread at turn end."""
    info: dict = {}
    with contextlib.suppress(Exception):
        from agent.voice_turn_route import session_runtime_view
        info["provider"] = getattr(session_runtime_view(agent), "provider", "") or ""
    with contextlib.suppress(Exception):
        info["tools"] = _grouped(getattr(agent, "tools", None))
    info["system_prompt"] = getattr(agent, "_cached_system_prompt", "") or ""
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None and getattr(compressor, "context_length", None):
        info["context_length"] = compressor.context_length
    return info


def publish_settled_info(turn, agent) -> None:
    """Main's end-of-turn ``session.info`` (``session_workdir._emit_settled_session_info``) for a local
    viewer: the first turn builds the agent, so this is when the banner's ``▸ System Prompt`` row, the
    agent's tool list and provider first become known. Never breaks the turn.

    A ``model_override`` admission (``/model --once``, ``/moa``) built its agent on the one-turn
    route. That agent must not become the route Desktop, the TUI, and resume read. Drop it from
    the cache and publish the policy model and provider."""
    if agent is None:
        return
    with contextlib.suppress(Exception):
        from gateway.session_cli_turn import cli_turn
        from gateway.session_contract import SessionRef
        from gateway.session_local import local_session_info
        from gateway.session_policy import policy_for_source
        authority, session_id, _generation = turn._approval_owner
        message_id = getattr(getattr(turn, "_ctx", None), "inbound_message_id", None)
        one_turn = bool(cli_turn(message_id).get("model_override"))
        live = None
        if one_turn:
            sessions = getattr(authority, "sessions", None)
            live = sessions.get(session_id) if hasattr(sessions, "get") else None
            evict = getattr(getattr(turn, "_runner", None), "_evict_cached_agent", None)
            if live is not None and callable(evict):
                try:
                    evict(live.route)
                except Exception:
                    logger.debug("one-turn agent cache drop failed", exc_info=True)
        ref = SessionRef(authority.profile_id, session_id)
        info = {**agent_inventory(agent), **local_session_info(authority, ref)}
        if one_turn and live is not None:
            try:
                runner = getattr(authority, "runner", None) or turn._runner
                policy = policy_for_source(runner, live.source)
            except Exception:
                logger.debug("one-turn policy route unavailable", exc_info=True)
                policy = None
            if policy is not None:
                info["model"] = policy.current_model(authority)
                if policy.provider:
                    info["provider"] = policy.provider
        turn._publish_execution("session.info", info)
