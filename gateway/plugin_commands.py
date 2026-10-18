"""Authenticated opt-in plugin commands, before ordinary busy/agent routing."""
from __future__ import annotations

import asyncio
import inspect
import logging

from hermes_cli.plugin_commands import (
    PluginAgentDispatch, PluginCommandContext, _command_context,
)
from hermes_constants import (
    get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
)

logger = logging.getLogger(__name__)


async def dispatch_contextual_command(runner, event, session_key):
    """Return (handled, reply). A dispatch changes only the current event text.

    Called AFTER human authorization and emergency-stop checks. No event is
    emitted, no agent is constructed, and no conversation/system prompt changes.
    Legacy plugin commands remain at their existing cold-path dispatch site.
    """
    command = event.get_command()
    if not command or event.internal or not event.allow_gateway_control:
        return False, None
    from hermes_cli.commands import resolve_command
    if resolve_command(command) is not None:
        return False, None
    config = runner.config
    quick = config.get("quick_commands", {}) if isinstance(config, dict) else getattr(config, "quick_commands", {})
    if command in (quick or {}):
        return False, None
    source = event.source
    home = (runner._resolve_profile_home_for_source(source)
            if getattr(config, "multiplex_profiles", False) else get_hermes_home())
    home_token = set_hermes_home_override(home)
    try:
        from hermes_cli.plugins import get_plugin_commands, fire_pre_command_hook
        name = command.replace("_", "-")
        entry = get_plugin_commands().get(name)
        if not entry or not entry.get("contextual"):
            return False, None
        denied = runner._check_slash_access(source, name)
        if denied is not None:
            return True, denied
        # Preserve target data verbatim after the command separator. The legacy
        # extractor normalizes dashes for CLI flags, which can corrupt paths.
        parts = (event.text or "").lstrip().split(maxsplit=1)
        raw_args = parts[1] if len(parts) > 1 else ""
        fire_pre_command_hook(surface="gateway", command=name, alias_used=command,
                              args_raw=raw_args, session_key=session_key,
                              platform=source.platform.value)
        results = await runner.hooks.emit_collect("command:" + name, {
            "platform": source.platform.value, "user_id": source.user_id,
            "command": name, "raw_command": command, "args": raw_args, "raw_args": raw_args,
        })
        for result in results or ():
            if not isinstance(result, dict):
                continue
            decision = result.get("decision")
            if decision in {"deny", "handled"}:
                return True, result.get("message") or (f"Command `/{name}` was blocked by a hook." if decision == "deny" else None)
            if decision == "rewrite":
                # A rewritten command must re-enter normal access/command dispatch;
                # it must never retain the old command's authority or context.
                target = str(result.get("command_name", "")).strip().lstrip("/")
                if target:
                    event.text = f"/{target} {result.get('raw_args', '')}".strip()
                    return False, None
        session_id = runner.session_store.peek_session_id(session_key) or ""
        context = PluginCommandContext(
            platform=source.platform.value,
            profile=source.profile or runner._active_profile_name(),
            profile_home=home, session_key=session_key, session_id=session_id,
            user_id=str(source.user_id or ""), chat_id=str(source.chat_id or ""),
            thread_id=str(source.thread_id or ""), message_id=str(event.message_id or ""),
            interaction_id=str(event.interaction_id or ""),
        )

        async def invoke():
            # A separate asyncio context prevents command bindings from leaking into
            # the caller, including exception/cancellation paths. Not an agent task.
            from gateway.session_context import set_session_vars, clear_session_vars
            adapter = runner._adapter_for_source(source)
            tokens = set_session_vars(
                platform=context.platform, source="gateway", profile=context.profile,
                session_key=session_key, session_id=session_id, user_id=context.user_id,
                user_id_alt=str(source.user_id_alt or ""), user_name=source.user_name or "",
                chat_id=context.chat_id, chat_type=source.chat_type or "",
                chat_name=source.chat_name or "", thread_id=context.thread_id,
                scope_id=str(source.scope_id or ""), message_id=context.message_id,
                cron_session="", async_delivery=getattr(adapter, "supports_async_delivery", True),
            )
            token = _command_context.set(context)
            try:
                value = entry["handler"](raw_args, context=context)
                return await value if inspect.isawaitable(value) else value
            finally:
                _command_context.reset(token)
                clear_session_vars(tokens)

        value = await asyncio.create_task(invoke())
        if isinstance(value, PluginAgentDispatch):
            event.text = value.user_message(name, raw_args)
            return False, None
        return True, str(value) if value else None
    except Exception:
        # Never fall through to an LLM after a failed privileged command, and
        # never disclose exception strings containing paths/credentials.
        logger.exception("Contextual plugin command failed")
        return True, "Plugin command failed; no completion claimed. Inspect local gateway diagnostics."
    finally:
        reset_hermes_home_override(home_token)
