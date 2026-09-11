"""run_turn.py — facade: composes the turn mixins (split from one 3903-line class)."""

from __future__ import annotations
from gateway.run_turn_hmwa import GatewayTurnHmwaMixin
from gateway.run_turn_exec import GatewayTurnExecMixin

import logging
import asyncio
import dataclasses
import json
import os
import time
from agent.i18n import t
from contextlib import nullcontext, suppress
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, _session_key_namespace
from hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


_CONTEXT_OVERFLOW_ERROR_PHRASES = (
    "context length", "context size", "context window",
    "maximum context", "token limit", "too many tokens",
    "reduce the length", "exceeds the limit",
    "request entity too large", "prompt is too long",
    "payload too large", "input is too long",
)


def is_context_overflow_failure_result(agent_result: dict, history_len: int) -> bool:
    """One verdict for "this failed turn is a context overflow", shared by transcript persistence
    (#1630 skip) and the user-facing reply so the two can never disagree.

    Multi-word phrases (not bare "exceed"/"token") avoid matching "rate limit exceeded" or
    "invalid authentication token"; a bare 400 only counts on a long session."""
    if not agent_result.get("failed"):
        return False
    if agent_result.get("compression_exhausted"):
        return True
    err = str(agent_result.get("error") or "").lower()
    return any(p in err for p in _CONTEXT_OVERFLOW_ERROR_PHRASES) or ("400" in err and history_len > 50)

class GatewayTurnMixin(GatewayTurnHmwaMixin, GatewayTurnExecMixin):
    """Agent-turn execution for GatewayRunner (see module docstring)."""

    def _resolve_session_agent_runtime(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session.

        Priority (highest first): session ``/model`` → ``channel_overrides`` → global config/env
        (``_resolve_gateway_model(user_config)`` and default provider resolution)."""
        from gateway.run import (
            _credential_pool_for_provider, _get_channel_override, _resolve_gateway_model,
            _resolve_runtime_agent_kwargs, _resolve_runtime_agent_kwargs_for_provider,
        )
        skey = self._resolve_session_key_or_none(source, session_key)

        model = _resolve_gateway_model(user_config)
        if skey:
            self._rehydrate_session_model_override(skey)
        _override_state = self._peek_session_state(skey) if skey else None
        override = _override_state.conversation.model_override if _override_state else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                k: override.get(k) for k in (
                    "provider", "requested_provider", "api_key", "base_url", "api_mode",
                    "max_tokens", "credential_pool", "request_overrides", "capabilities",
                )
            }
            override_runtime["capabilities"] = dict(override_runtime["capabilities"] or {})
            if override_runtime.get("api_key"):
                if override_runtime.get("credential_pool") is None:
                    override_runtime["credential_pool"] = _credential_pool_for_provider(override.get("provider"))
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    skey or "", model, override_model, override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # No api_key on the override: env-based resolution below, override model/provider on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                skey or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                skey or "", model,
                [
                    _key for _key, _st in list(self._sessions_map().items())
                    if _st.conversation.model_override is not None
                ][:5] or "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info("Runtime provider supplied explicit model override: %s -> %s", model, runtime_model)
            model = runtime_model

        cfg = getattr(self, "config", None)  # getattr: bare object.__new__ test runners
        if cfg and source is not None:
            ch = _get_channel_override(
                cfg, source.platform, str(source.chat_id) if source.chat_id else "",
                thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
                parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
            )
            if ch:
                if ch.model:
                    model = ch.model
                if ch.provider:
                    runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(ch.provider)
                    ch_runtime_model = runtime_kwargs.pop("model", None)
                    # Adopt the provider's bundled model only when the override named none.
                    if ch_runtime_model and not ch.model:
                        model = ch_runtime_model

        if override and skey:
            model, runtime_kwargs = self._apply_session_model_override(skey, model, runtime_kwargs)

        # Provider resolved but no model.default (`hermes auth add` without `hermes model`): use the
        # provider's first catalog model.
        if not model and runtime_kwargs.get("provider"):
            with suppress(Exception):
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s", model, runtime_kwargs["provider"],
                    )

        # Final safety net: an empty model (transient config-cache miss) makes every API call 400 and
        # the session goes silent — reuse the last model resolved for this session, else process-wide.
        if not model:
            _lr_state = self._peek_session_state(skey) if skey else None
            _lr_star = self._peek_session_state("*")
            _recovered = (
                (_lr_state.conversation.last_resolved_model if _lr_state else "")
                or (_lr_star.conversation.last_resolved_model if _lr_star else "")
            )
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)", skey or "", _recovered,
                )
                model = _recovered
        else:
            # Cache the good resolution for future recovery turns.
            if skey:
                self._session_state(skey).conversation.last_resolved_model = model
            self._session_state("*").conversation.last_resolved_model = model

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Effective model/runtime config for one turn. With `/fast` priority on, fast-mode
        ``request_overrides`` are deep-merged OVER the per-provider ones so both reach the model."""
        from gateway.run import _deep_merge_request_overrides
        from hermes_cli.models import resolve_fast_mode_overrides
        # Tests bind this method onto bare namespaces, so no class-level tables here.
        runtime = {
            k: runtime_kwargs.get(k) for k in (
                "api_key", "base_url", "provider", "requested_provider", "api_mode", "command", "args",
                "credential_pool", "max_tokens", "capabilities",
            )
        }
        runtime["args"] = list(runtime["args"] or [])
        runtime["capabilities"] = dict(runtime["capabilities"] or {})
        base_request_overrides = dict(runtime_kwargs.get("request_overrides") or {})
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model, runtime["provider"], runtime["requested_provider"], runtime["base_url"],
                runtime["api_mode"], runtime["command"], tuple(runtime["args"]),
            ),
        }
        if getattr(self, "_service_tier", None) != "priority":
            # None / auto / cold: the bounded window is applied per request by agent.fast_mode.
            route["request_overrides"] = base_request_overrides
            return route
        try:
            overrides = resolve_fast_mode_overrides(
                route["model"], provider=runtime["provider"], base_url=runtime["base_url"],
            )
        except Exception:
            overrides = None
        # Fast-mode keys (service_tier / speed) are top-level and don't collide with extra_body.
        route["request_overrides"] = _deep_merge_request_overrides(base_request_overrides, overrides or {})
        return route

    def _sync_session_model_from_agent(self, session_id: str, agent: Any) -> None:
        """Persist the runtime model/provider a gateway turn actually used (provider fallback can
        switch them after the row was created). Runs in the ``run_sync`` executor thread, so it
        uses the sync ``SessionDB`` (``_db``), not the AsyncSessionDB forwarder."""
        if not session_id or agent is None or self._session_db is None:
            return
        model = getattr(agent, "model", None)
        if not model:
            return
        runtime = {k: getattr(agent, k, None) for k in ("provider", "base_url", "api_mode")}
        runtime["fallback_active"] = bool(getattr(agent, "_fallback_activated", False))
        runtime = {k: v for k, v in runtime.items() if v not in (None, "")}
        try:
            db = self._session_db._db
            row = db.get_session(session_id)
            if not row:
                return
            # Legacy backfill: canonical Bot Chats created BEFORE the follow_profile_config contract existed
            # carry no marker, yet they are still the plugin-owned forever-DM. The plugin's own identity
            # rule is "the profile's session titled exactly 'Bot Chat'" (UNIQUE(title) makes that an exact
            # registry, and pre-policy rows may be visible OR hidden), so mirror that rule here. Without
            # this, every Bot Chat that already exists in the field stays pinned to its stale stored
            # provider until the user deletes it — the exact live-report shape (#89497 / #94818).
            raw_config = row.get("model_config")
            config = {}
            with suppress(Exception):
                config = json.loads(raw_config) if raw_config else {}
            if not isinstance(config, dict):
                config = {}
            gateway_runtime = dict(config.get("gateway_runtime") or {})
            if row.get("model") == model and all(gateway_runtime.get(k) == v for k, v in runtime.items()):
                return
            config["gateway_runtime"] = runtime
            db.update_session_meta(session_id, json.dumps(config), model=model)
        except Exception:
            logger.debug("Failed to sync gateway session model metadata", exc_info=True)

    def _event_thread_metadata(self, event, source):
        """Thread metadata for a send that replies to ``event`` on ``source``."""
        return self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

    @staticmethod
    def _pop_post_delivery_callback(adapter, key, generation):
        """Pop the adapter's deferred post-delivery callback for ``key`` (legacy dict fallback)."""
        if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
            return adapter.pop_post_delivery_callback(key, generation=generation)
        if adapter and hasattr(adapter, "_post_delivery_callbacks"):
            return adapter._post_delivery_callbacks.pop(key, None)
        return None

    @staticmethod
    def _is_intentional_silence(agent_result, response) -> bool:
        try:
            from gateway.response_filters import is_intentional_silence_agent_result
            return is_intentional_silence_agent_result(agent_result, response)
        except Exception:
            return False

    @dataclasses.dataclass
    class _HygienePlan:
        """Hygiene pre-check outcome for one turn."""

        needs_compress: bool
        approx_tokens: int
        msg_count: int
        warn_token_threshold: int

    # reasoning_style → (header line, per-line quote prefix for blank / non-blank lines)
    _REASONING_QUOTE_STYLES = {
        "subtext": ("-# 💭 Reasoning", "-# ", "-#"), "blockquote": ("> 💭 **Reasoning:**", "> ", ">")
    }

    _STATUS_HINTS = {
        401: " Check your API key or run `claude /login` to refresh OAuth credentials.",
        402: " Your API balance or quota is exhausted. Check your provider dashboard.",
        529: " The API is temporarily overloaded. Please try again shortly.",
    }

    @dataclasses.dataclass
    class _PreparedTurn:
        """Inputs to the agent run assembled by ``_hmwa_prepare_turn``."""

        history: Any
        context_prompt: str
        message_text: Any
        persist_user_message: Any
        persist_user_timestamp: Any
        persist_user_display_kind: Optional[str]
        persistence_session_id: Optional[str] = None
        persistence_owner: Optional[str] = None

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", (event.text or "")[:80].replace("\n", " "),
            getattr(event, "reply_to_message_id", None),
            (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " "),
        )

        resolved = await self._hmwa_resolve_session(event, source)
        if resolved is None:
            return
        source, session_entry, session_key = resolved
        prepared, _session_env_tokens = await self._hmwa_prepare_turn(
            event, source, session_entry, session_key, _quick_key, run_generation,
        )
        if not isinstance(prepared, self._PreparedTurn):
            return prepared
        history, message_text = prepared.history, prepared.message_text

        try:
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(source.thread_id) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Capture the launch session id so post-run compression publication is identity-guarded
            # (a /new may move session_entry.session_id while the old run is still unwinding).
            from gateway.run_heartbeat_acceptance import heartbeat_owner_is_current
            if not heartbeat_owner_is_current(self, event, session_key):
                return
            _run_start_session_id = session_entry.session_id
            _turn_started_monotonic = time.monotonic()
            # Admission/typing is not execution. All routing, authorization and
            # turn preparation gates have passed when the agent runner is entered.
            event._heartbeat_execution_started = True
            agent_result = await self._run_agent(
                message=message_text, context_prompt=prepared.context_prompt, history=history, source=source,
                session_id=_run_start_session_id, session_key=session_key,
                run_generation=run_generation, event_message_id=self._reply_anchor_for_event(event),
                inbound_message_id=str(event.message_id) if event.message_id else None,
                channel_prompt=event.channel_prompt, moa_config=getattr(event, "_moa_config", None),
                persist_user_message=prepared.persist_user_message,
                persist_user_timestamp=prepared.persist_user_timestamp,
                persist_user_display_kind=prepared.persist_user_display_kind,
                persist_user_display_metadata={"gateway_input_owner": prepared.persistence_owner},
                message_type=event.message_type,
            )
            _turn_seconds = time.monotonic() - _turn_started_monotonic

            # A queued (/queue) chain answered the LAST message of the chain, so the outer final
            # send (bracketed by the adapter against this event) must be ledgered under that
            # message's id or it collides with an earlier turn's row carrying the same text. Reply
            # routing is untouched: the anchor still comes from this event.
            if isinstance(agent_result, dict):
                _terminal_inbound = agent_result.get("queued_terminal_inbound_id")
                if _terminal_inbound:
                    event.ledger_message_id = str(_terminal_inbound)

            await self._hmwa_stop_typing_for_turn(event, source)

            if not self._is_session_run_current(_quick_key, run_generation):
                self._hmwa_discard_stale_result(source, _quick_key, run_generation)
                return None

            response, _intentional_silence, agent_messages = await self._hmwa_shape_agent_response(
                agent_result, source, history, session_entry, session_key,
                _quick_key, run_generation, _run_start_session_id, _platform_name, _msg_start_time,
            )
            response = self._hmwa_prepend_reasoning(agent_result, response, source, _intentional_silence)
            _footer_line = self._hmwa_runtime_footer_line(agent_result, source, _turn_seconds)
            # Streaming already delivered the body: the footer goes out as a trailing send instead.
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"
            await self._hmwa_post_turn_hooks(hook_ctx, agent_result, response)

            agent_failed_early, hidden_reasoning_incomplete, is_context_overflow_failure = (
                self._hmwa_classify_turn_failure(agent_result, history, session_entry)
            )
            response, session_entry = await self._hmwa_compression_exhaustion_reset(
                agent_result, response, session_entry, session_key, source,
            )
            await self._hmwa_persist_turn_transcript(
                event=event, source=source, session_entry=session_entry, session_key=session_key,
                agent_result=agent_result, agent_messages=agent_messages, prepared=prepared,
                response=response, agent_failed_early=agent_failed_early,
                hidden_reasoning_incomplete=hidden_reasoning_incomplete,
                is_context_overflow_failure=is_context_overflow_failure,
            )
            return await self._hmwa_deliver_turn_response(
                event, source, session_entry, session_key, run_generation,
                agent_result, agent_messages, response, _footer_line, _intentional_silence,
            )

        except Exception as e:
            return await self._hmwa_agent_error_reply(e, event, source, session_entry, session_key, prepared)
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)

    def _profile_scope_for_source(self, source: SessionSource):
        """``_profile_runtime_scope`` for ``source``'s profile when multiplexing, else a no-op context.

        Under multiplexing config/skills/memory resolve to the source profile's home AND credentials
        come from its secret scope (never process-global ``os.environ``)."""
        from gateway.run import _profile_runtime_scope
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return _profile_runtime_scope(self._resolve_profile_home_for_source(source))
        return nullcontext()

    def _reset_notice_session_info(self, source: SessionSource) -> str:
        """Session-info block for the auto-reset notice, resolved inside the profile serving ``source``.

        Call via ``asyncio.to_thread``: resolution can block (credential refresh, context-length
        probes), and the scope is entered here so contextvars behave in the worker thread."""
        with self._profile_scope_for_source(source):
            return self._format_session_info()

    def _format_session_info(self) -> str:
        """Model / provider / context-length / endpoint block so users can spot bad context detection."""
        from gateway.run import _resolve_gateway_model_context
        resolved = _resolve_gateway_model_context()
        context_length = resolved.context_length
        ctx_source = {
            "config": "config",
            "default": "default — set model.context_length in config to override",
        }.get(resolved.context_source, "detected")
        ctx_display = (
            f"{context_length / 1_000_000:.1f}M" if context_length >= 1_000_000
            else f"{context_length // 1_000}K" if context_length >= 1_000 else str(context_length)
        )
        lines = [
            f"◆ Model: `{resolved.model}`",
            f"◆ Provider: {resolved.provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]
        base_url = resolved.base_url
        if base_url and base_url_hostname(base_url) in ("localhost", "127.0.0.1", "0.0.0.0"):
            lines.append(f"◆ Endpoint: {base_url}")
        return "\n".join(lines)

    def _resolve_enabled_toolsets_for_source(
        self, user_config: dict, source: "SessionSource", platform_key: str,
    ) -> list:
        """Enabled toolsets for an agent run, honoring an adapter ``toolsets_for_source()`` override
        validated through the SAME ``_get_platform_tools`` path (unknown / platform-restricted
        toolsets dropped, not trusted)."""
        from hermes_cli.tools_config import _get_platform_tools
        try:
            adapter = self._adapter_for_source(source)
            override = adapter.toolsets_for_source(source) if adapter is not None else None
        except Exception:
            override = None
        if override and isinstance(override, list):
            pts = dict(user_config.get("platform_toolsets") or {})
            pts[platform_key] = [str(x) for x in override]
            user_config = {**user_config, "platform_toolsets": pts}
        return sorted(_get_platform_tools(user_config, platform_key))

    def _resolve_turn_toolsets(self, user_config: dict, source: "SessionSource", platform_key: str):
        """``(enabled_toolsets, disabled_toolsets)`` for an agent run on ``source``."""
        from agent.skill_utils import parse_config_string_list
        enabled = self._resolve_enabled_toolsets_for_source(user_config, source, platform_key)
        disabled = parse_config_string_list((user_config.get("agent") or {}).get("disabled_toolsets")) or None
        return enabled, disabled

    def _mcp_reload_refresh_cached_agents(self, multiplex: bool, profile) -> None:
        """Refresh cached agents so existing sessions see new MCP tools on their next turn without
        a history-destroying ``/new``. Each agent keeps its build-time toolset selection EXACTLY: a
        session built with restricted enabled_toolsets (e.g. ["safe"]) must NOT silently gain tools."""
        try:
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            _cache = getattr(self, "_agent_cache", None)
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            if _cache_lock is None or not _cache:
                return
            # Multiplex: only this profile's sessions (another profile's agent would get this registry).
            _ns_prefix = _session_key_namespace(profile) + ":" if multiplex else None
            with _cache_lock:
                for _sess_key, _entry in list(_cache.items()):
                    if _ns_prefix and not str(_sess_key).startswith(_ns_prefix):
                        continue
                    _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                    if _agent is not None:
                        refresh_agent_mcp_tools(_agent, quiet_mode=True)
        except Exception as _exc:
            logger.debug("Failed to update cached agent tools after MCP reload: %s", _exc)

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Disconnect, reconnect, and notify MCP tool changes (shared by button / text / no-confirm paths).

        Under multiplex the reload runs inside the requesting profile's runtime scope (entered here
        when the caller did not) and only that profile's servers are torn down and rediscovered.

        See #95518.
        """
        from gateway.run import _profile_runtime_scope
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        if multiplex and not get_hermes_home_override():
            profile_home = self._resolve_profile_home_for_source(event.source)
            with _profile_runtime_scope(Path(profile_home)):
                return await self._execute_mcp_reload(event)
        try:
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers
            from tools.mcp_tool_discovery import discover_mcp_tools
            from tools.mcp_tool import _servers, _lock, _server_visible_in_scope
            from tools.mcp_tool_agent import reprobe_tool_availability
            from tools.registry import registry

            reload_scope = registry.current_scope_key() if multiplex else None

            def _scoped_server_names() -> set:
                with _lock:
                    return {
                        name for name in _servers
                        if _server_visible_in_scope(name, reload_scope)
                    }

            old_servers = _scoped_server_names()
            await self._run_in_executor_with_context(lambda: shutdown_mcp_servers(scope=reload_scope))
            # Explicit reload also re-probes tool availability (check_fn).
            reprobe_tool_availability()
            # Reconnect by discovering tools (reads config.yaml fresh).
            new_tools = await self._run_in_executor_with_context(discover_mcp_tools)

            connected_servers = _scoped_server_names()
            if reload_scope is not None:
                from tools.mcp_tool import _mcp_tool_server_names
                with _lock:
                    new_tools = [n for n in new_tools if _mcp_tool_server_names.get(n) in connected_servers]
            # (label, i18n key, names); i18n lines list reconnected first, the injected note added first.
            changes = (
                ("Reconnected", "gateway.reload_mcp.reconnected", connected_servers & old_servers),
                ("Added", "gateway.reload_mcp.added", connected_servers - old_servers),
                ("Removed", "gateway.reload_mcp.removed", old_servers - connected_servers),
            )
            lines = [t("gateway.reload_mcp.header")] + [
                t(key, names=", ".join(sorted(names))) for _label, key, names in changes if names
            ]
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            self._mcp_reload_refresh_cached_agents(multiplex, event.source.profile)

            # Append a note at the END of the history (preserves the prompt-cache prefix).
            change_parts = [
                f"{label} servers: {', '.join(sorted(names))}"
                for label, _key, names in (changes[1], changes[2], changes[0]) if names
            ]
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            with suppress(Exception):  # Best-effort; don't fail the reload over a transcript write
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(session_entry.session_id, reload_msg)

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)

    def _get_proxy_url(self) -> Optional[str]:
        """Proxy URL if proxy mode is configured (GATEWAY_PROXY_URL env wins over ``gateway.proxy_url``)."""
        from gateway.run import _load_gateway_config
        url = os.getenv("GATEWAY_PROXY_URL", "").strip()
        if not url:
            url = ((_load_gateway_config().get("gateway") or {}).get("proxy_url") or "").strip()
        return url.rstrip("/") if url else None

    def _build_stream_consumer_config(
        self, source: "SessionSource", scfg: Any, adapter: Any, *, on_missing_cursor: str,
    ) -> "tuple[Any, Optional[Callable[[], None]]]":
        """Build the shared ``StreamConsumerConfig`` and optional Telegram pause-typing closure.
        For non-editing adapters ``on_missing_cursor="fallback"`` streams with an empty cursor;
        ``"raise"`` raises ``RuntimeError`` so the caller skips streaming entirely."""
        from gateway.stream_consumer import StreamConsumerConfig
        _pause_typing_before_finalize = None
        if source.platform == Platform.TELEGRAM and hasattr(adapter, "pause_typing_for_chat"):
            def _pause_typing_before_finalize(_adapter=adapter, _chat_id=source.chat_id) -> None:
                _adapter.pause_typing_for_chat(_chat_id)
        # Non-editing platforms (QQ, WeChat) skip streaming — the partial first message could never
        # be updated — unless they have a native-streaming transport (WeCom msgtype "stream").
        _adapter_supports_edit = getattr(adapter, "SUPPORTS_MESSAGE_EDITING", True)
        _adapter_supports_native_stream = bool(getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False))
        if not _adapter_supports_edit and not _adapter_supports_native_stream and on_missing_cursor == "raise":
            raise RuntimeError("skip streaming for non-editable platform")
        _effective_cursor = scfg.cursor if _adapter_supports_edit else ""
        # Some Matrix clients render the cursor as tofu: stream text, no cursor.
        _buffer_only = source.platform == Platform.MATRIX
        if _buffer_only:
            _effective_cursor = ""
        # Fresh-final applies to Telegram only (others edit in place cheaply).
        # Fresh-final applies to Telegram only — other platforms either edit in place cheaply (Discord,
        # Slack) or don't have the timestamp-on-edit / edit-timestamp-stays-stale problem. (Ported from
        # openclaw/openclaw#72038.)
        _fresh_final_secs = (
            float(getattr(scfg, "fresh_final_after_seconds", 0.0) or 0.0)
            if source.platform == Platform.TELEGRAM else 0.0
        )
        _consumer_cfg = StreamConsumerConfig(
            edit_interval=scfg.edit_interval, buffer_threshold=scfg.buffer_threshold,
            cursor=_effective_cursor, buffer_only=_buffer_only,
            fresh_final_after_seconds=_fresh_final_secs, transport=scfg.transport or "edit",
            chat_type=getattr(source, "chat_type", "") or "",
        )
        return _consumer_cfg, _pause_typing_before_finalize

    @staticmethod
    def _proxy_error_result(text: str) -> Dict[str, Any]:
        return {"final_response": text, "messages": [], "api_calls": 0, "tools": []}

    def _proxy_stream_consumer(self, source: "SessionSource", event_message_id, _thread_metadata, _run_still_current):
        """Platform stream consumer for the proxy path when streaming is enabled, else ``None``."""
        from gateway.run import _load_gateway_config, _platform_config_key
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        # #60671 — streaming TTS consumer is created on the outer event-loop thread before run_sync
        # launches.  run_sync only reads it via ``streaming_tts_consumer_holder[0]`` for delta callback
        # wiring.
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(_load_gateway_config(), _platform_config_key(source.platform), "streaming")
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off" if _plat_streaming is None else bool(_plat_streaming)
        )
        if not _streaming_enabled:
            return None
        try:
            from gateway.stream_consumer import GatewayStreamConsumer
            _adapter = self._adapter_for_source(source)
            if not _adapter:
                return None
            _consumer_cfg, _pause_typing_before_finalize = self._build_stream_consumer_config(
                source, _scfg, _adapter, on_missing_cursor="fallback",
            )
            return GatewayStreamConsumer(
                adapter=_adapter, chat_id=source.chat_id, config=_consumer_cfg,
                metadata=_thread_metadata, on_before_finalize=_pause_typing_before_finalize,
                initial_reply_to_id=event_message_id, run_still_current=_run_still_current,
            )
        except Exception as _sc_err:
            logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)
            return None

    # _RunAgentDisplay fields copied verbatim onto the TurnContext.
    _DISPLAY_TO_TURN_CTX = (
        "_live_status_adapter", "_live_status_mode", "_thinking_enabled", "progress_mode",
        "progress_grouping", "tool_progress_enabled", "log_queue", "resolve_display_setting",
        "user_config", "enabled_toolsets", "disabled_toolsets", "log_mode_enabled",
        "interim_assistant_messages_enabled", "needs_progress_queue", "_native_slack_task_cards",
    )

    def _thread_metadata_for_progress(
        self, source: SessionSource, event_message_id: Optional[str], _progress_thread_id: Any,
        _relay_prospective_thread_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Thread metadata for a progress-lane send; relay Discord auto-thread lane falls back to the reply anchor.

        The connector will auto-thread on the reply anchor (thread is born on its FIRST send), so
        carrying it routes progress / status bubbles into the same thread as the final reply."""
        if not _progress_thread_id:
            metadata = None
        elif _progress_thread_id == source.thread_id:
            metadata = self._thread_metadata_for_source(source, event_message_id)
        else:
            metadata = self._thread_metadata_for_target(
                source.platform, source.chat_id, _progress_thread_id,
                chat_type=getattr(source, "chat_type", None), reply_to_message_id=event_message_id,
            )
        if metadata is None and _relay_prospective_thread_id:
            metadata = {"reply_to_message_id": event_message_id}
        return metadata

    @staticmethod
    async def _await_stream_task(stream_task) -> None:
        """Give the stream consumer task 5s to flush, then cancel it."""
        try:
            await asyncio.wait_for(stream_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task

    @staticmethod
    def _reaper_kwargs(worker: "GatewayRunner._RunAgentWorker") -> dict:
        """Shared kwargs of the watchdog + timeout-reaper threads."""
        return {
            **{k: getattr(worker, k) for k in ("task_id", "process_baseline", "worker_done", "timeout_fired", "cleanup_lock")},
            "is_still_current": worker.is_current,
        }

    @staticmethod
    def _agent_activity_summary(agent: Any) -> dict:
        """``agent.get_activity_summary()`` or ``{}`` when unavailable / failing."""
        if agent and hasattr(agent, "get_activity_summary"):
            with suppress(Exception):
                return agent.get_activity_summary()
        return {}
