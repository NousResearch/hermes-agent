"""Extracted gateway session_execution responsibility; consumed by TurnRunner."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import queue
import re
import threading
import time
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.interrupt_compat import _accepts_keyword
from agent.replay_cleanup import strip_stale_dangerous_confirmations
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.turn_context import TurnContext
from hermes_cli.config import cfg_get
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewaySessionAgentMixin:
    # ── agent resolution (cache reuse vs fresh build) ───────────────────────────────────────

    @dataclasses.dataclass
    class _CachedAgentLookup:
        agent: Any = None
        reused: bool = False
        evicted: Any = None  # agent evicted under the lock; released off-lock on a daemon thread

    def _skip_context_files(self, platform_key) -> bool:
        """gateway.platforms.<plat>.skip_context_files: messaging platforms may opt out of
        filesystem-heavy context-file discovery (SOUL.md, AGENTS.md, .cursorrules)."""
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(self._runner, self._ctx.source)
        if policy:
            return policy.ignore_rules
        platforms_cfg = (self._ctx.user_config.get("gateway") or {}).get("platforms") or {}
        # ``hermes gateway setup`` writes ``gateway.platforms`` as a LIST of enabled platform names,
        # not a dict; treat any non-dict shape as "no per-platform overrides" rather than crashing.
        if not isinstance(platforms_cfg, dict):
            return False
        return bool((platforms_cfg.get(platform_key) or {}).get("skip_context_files"))

    def _cached_sid_is_dead(self, cache_lock, cache) -> tuple:
        """(peeked cached session_id, is_dead) — checked OUTSIDE the cache lock. "cached sid != current
        sid" normally means an intentional switch (reuse), but the routing-key self-heal yields the same
        shape with an agent bound to a DEAD session; reusing it re-binds the dead sid and loops."""
        ctx = self._ctx
        peek_sid = None
        if cache_lock and cache is not None:
            with cache_lock:
                entry = cache.get(ctx.session_key)
            if entry and len(entry) > 3:
                peek_sid = entry[3]
        dead = False
        if peek_sid is not None and ctx.session_id is not None and peek_sid != ctx.session_id:
            with suppress(Exception):
                dead = self._runner.session_store._is_session_ended_in_db(peek_sid)
        return peek_sid, dead

    def _current_message_count(self):
        """Cross-process write guard input: the session's current DB message_count (or None)."""
        ctx = self._ctx
        if self._runner._session_db is None or not ctx.session_id:
            return None
        count = None
        with suppress(Exception):
            # run_sync is off-loop (executor); sync DB is fine.
            row = self._runner._session_db._db.get_session(ctx.session_id)
            if row:
                count = row.get("message_count", 0)
        return count

    def _pop_cached_agent_for_eviction(self):
        """Evict under the lock but DEFER release (release_clients can block on memory-provider /
        socket teardown while the idle sweeper waits on this lock). The turn rebuilds a fresh agent, so
        the caller does a SOFT release that keeps sandbox / browser / bg processes."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        evicted = self._runner._agent_cache.pop(self._ctx.session_key, None)
        agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
        return agent if agent and agent is not _AGENT_PENDING_SENTINEL else None

    def _lookup_cached_agent(self, sig, cache_lock, cache, max_iterations, peek_sid, dead, msg_count):
        ctx = self._ctx
        out = self._CachedAgentLookup()
        if not (cache_lock and cache is not None):
            return out
        with cache_lock:
            cached = cache.get(ctx.session_key)
            if not (cached and cached[1] == sig):
                return out
            # cached[2] = message_count at cache time (stale when a second process appended rows);
            # cached[3] = the session_id the snapshot was taken for.
            cached_mc = cached[2] if len(cached) > 2 else None
            cached_sid = cached[3] if len(cached) > 3 else None
            # Same session_key, other conversation: the counts track DIFFERENT DB rows, so the
            # comparison is meaningless — REUSE rather than bust the prompt cache on every switch.
            sid_mismatch = cached_sid is not None and ctx.session_id is not None and cached_sid != ctx.session_id
            # Re-validate the outside-lock dead-session peek against the tuple read under THIS lock:
            # a stale "dead" verdict must never be applied to a different (possibly live) agent.
            if sid_mismatch and dead and cached_sid == peek_sid:
                logger.info(
                    "Agent cache invalidated for session %s: "
                    "cached agent's session_id %s is ended in "
                    "state.db (stale self-heal artifact, "
                    "#54878 x #54947) — discarding instead of "
                    "reusing across the routing recovery", ctx.session_key, cached_sid,
                )
            elif not sid_mismatch and cached_mc is not None and msg_count is not None and msg_count != cached_mc:
                logger.info(
                    "Agent cache invalidated for session %s: "
                    "message_count changed (%s -> %s), "
                    "possible cross-process write", ctx.session_key, cached_mc, msg_count,
                )
            else:
                out.agent = cached[0]
                # Refresh LRU order so cap enforcement evicts truly-oldest entries.
                if hasattr(cache, "move_to_end"):
                    with suppress(KeyError):
                        cache.move_to_end(ctx.session_key)
                self._runner._init_cached_agent_for_turn(out.agent, ctx._interrupt_depth)
                # Cached agent may have been created with old config.
                out.agent.max_iterations = max_iterations
                logger.debug("Reusing cached agent for session %s", ctx.session_key)
                out.reused = True
                return out
            out.evicted = self._pop_cached_agent_for_eviction()
        return out

    def _release_evicted_agent(self, agent) -> None:
        """Off-lock soft release on a daemon thread so teardown never blocks the gateway loop."""
        self._runner._spawn_release_thread(
            self._runner._release_evicted_agent_soft, (agent,), f"agent-xproc-evict-{str(self._ctx.session_key)[:24]}",
            inline_fallback=True,
        )

    def _build_fresh_agent(self, turn_route, platform_key, combined_ephemeral, max_iterations,
                           reasoning_config, pr, skip_context_files):
        from gateway.run import _checkpoint_agent_kwargs
        ctx = self._ctx
        runner = self._runner
        src = ctx.source
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(runner, src)
        return ctx.AIAgent(
            model=turn_route["model"], **turn_route["runtime"], **_checkpoint_agent_kwargs(ctx.user_config),
            max_iterations=max_iterations, quiet_mode=True, verbose_logging=False,
            enabled_toolsets=ctx.enabled_toolsets, disabled_toolsets=ctx.disabled_toolsets,
            ephemeral_system_prompt=combined_ephemeral or None,
            prefill_messages=runner._prefill_messages or None,
            reasoning_config=reasoning_config, service_tier=runner._service_tier,
            request_overrides=turn_route.get("request_overrides"),
            providers_allowed=pr.get("only"), providers_ignored=pr.get("ignore"), providers_order=pr.get("order"),
            provider_sort=pr.get("sort"), provider_require_parameters=pr.get("require_parameters", False),
            provider_data_collection=pr.get("data_collection"),
            session_id=ctx.session_id, platform=platform_key,
            user_id=src.user_id, user_id_alt=src.user_id_alt, user_name=src.user_name,
            chat_id=src.chat_id, chat_name=src.chat_name, chat_type=src.chat_type, thread_id=src.thread_id,
            gateway_session_key=ctx.session_key,
            session_db=getattr(runner._session_db, "_db", runner._session_db),
            # Reload from disk — do not reuse the startup snapshot.
            # See #60955.
            fallback_model=self._runner._refresh_fallback_model(),
            skip_context_files=skip_context_files,
            # Keep the persona even with minimal context: soul identity is one small file.
            load_soul_identity=not bool(policy and policy.ignore_rules),
            skip_memory=bool(policy and policy.ignore_rules),
        )

    def _resolve_turn_agent(self, turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr):
        """Reuse this session's cached AIAgent (frozen system prompt + tool schemas → prompt cache
        hits) or build a fresh one. Returns (agent, reused_cached_agent)."""
        ctx = self._ctx
        runner = self._runner
        skip_context_files = self._skip_context_files(platform_key)
        sig = runner._agent_config_signature(
            turn_route["model"], turn_route["runtime"], ctx.enabled_toolsets, combined_ephemeral,
            cache_keys=runner._extract_cache_busting_config(ctx.user_config),
            user_id=getattr(ctx.source, "user_id", None),
            user_id_alt=getattr(ctx.source, "user_id_alt", None),
            skip_context_files=skip_context_files,
        )
        cache_lock = getattr(runner, "_agent_cache_lock", None)
        cache = getattr(runner, "_agent_cache", None)
        peek_sid, dead = self._cached_sid_is_dead(cache_lock, cache)
        msg_count = self._current_message_count()
        found = self._lookup_cached_agent(sig, cache_lock, cache, max_iterations, peek_sid, dead, msg_count)
        agent = found.agent
        # Lock released — refresh the reused agent's fallback chain from disk OUTSIDE the cache lock
        # (disk I/O under the lock stalls the idle-sweep watcher and Discord heartbeats). A chain
        # configured after caching must reach the next turn; per-session serialization keeps it safe.
        if found.reused and agent is not None:
            self._runner._apply_fallback_chain_to_agent(agent, runner._refresh_fallback_model())
        if found.evicted is not None:
            self._release_evicted_agent(found.evicted)
        if agent is None:
            agent = self._build_fresh_agent(
                turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr, skip_context_files,
            )
            if cache_lock and cache is not None:
                with cache_lock:
                    # Record the snapshot's session_id with message_count so the cross-process guard
                    # can skip the meaningless count comparison if the active session_id switches.
                    cache[ctx.session_key] = (agent, sig, msg_count, ctx.session_id)
                    runner._enforce_agent_cache_cap()
            logger.debug("Created new agent for session %s (sig=%s)", ctx.session_key, sig)
        return agent, found.reused
