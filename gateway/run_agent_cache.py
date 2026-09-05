"""Agent cache, session model overrides, turn leases, run generations and conversation-scope reset
for GatewayRunner (MRO mixin). ``gateway.run`` internals are imported lazily inside method bodies
(import cycle), so ``patch("gateway.run.X")`` keeps intercepting them at call time."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from contextlib import nullcontext, suppress
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.interrupt_compat import _accepts_keyword
from gateway.config import Platform
from gateway.session import SessionSource, build_session_context_prompt
from hermes_cli.config import cfg_get

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Override fields layered onto runtime kwargs when non-None (partial overrides don't clobber defaults).
_OVERRIDE_APPLY_KEYS = (
    "provider", "requested_provider", "api_key", "base_url", "api_mode", "credential_pool", "capabilities", "max_tokens",
)


def _first_agent(entry: Any) -> Any:
    """Unwrap a cache entry (``(agent, sig, ...)`` tuple or bare agent) to its agent."""
    return entry[0] if isinstance(entry, tuple) and entry else entry


def _tuple_agent(entry: Any) -> Any:
    """Agent of a ``(agent, sig, ...)`` cache tuple; None for any other entry shape."""
    return entry[0] if isinstance(entry, tuple) and entry else None


class GatewayAgentCacheMixin:
    """Agent cache, session model overrides, turn leases, run generations and conversation-scope reset for GatewayRunner."""

    @classmethod
    def _extract_honcho_cache_busting_config(cls) -> dict[str, Any]:
        """Extract Honcho identity keys, memoized by honcho.json mtime; all-None when unavailable."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
            path = resolve_config_path()
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            memo_key = (str(path), mtime_ns)
            cached = cls._HONCHO_CACHE_BUSTING_MEMO.get(memo_key)
            if cached is not None:
                return dict(cached)
            hcfg = HonchoClientConfig.from_global_config(config_path=path)
            aliases = hcfg.user_peer_aliases or {}
            values = {
                "honcho.peer_name": hcfg.peer_name,
                "honcho.ai_peer": hcfg.ai_peer,
                "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
                "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
                "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
                "honcho.session_peer_prefix": bool(hcfg.session_peer_prefix),
                "honcho.session_ai_peer_prefix": bool(hcfg.session_ai_peer_prefix),
            }
            cls._HONCHO_CACHE_BUSTING_MEMO = {memo_key: values}
            return dict(values)
        except Exception:
            return dict.fromkeys(cls._HONCHO_CACHE_BUSTING_KEYS)

    @classmethod
    def _extract_cache_busting_config(cls, user_config: dict | None) -> dict:
        """Values that must bust the cached agent, as a flat dict keyed by 'section.key'. Missing keys /
        non-dict sections yield None (still enters the signature). Includes the live tool registry
        generation: MCP reloads mutate the registry without touching config.yaml."""
        out: Dict[str, Any] = {}
        cfg = user_config if isinstance(user_config, dict) else {}
        for section, key in cls._CACHE_BUSTING_CONFIG_KEYS:
            section_val = cfg.get(section)
            if section == "checkpoints" and isinstance(section_val, bool):
                # Legacy ``checkpoints: true``: a live toggle must still rebuild the cached agent.
                out[f"{section}.{key}"] = section_val if key == "enabled" else None
            else:
                out[f"{section}.{key}"] = section_val.get(key) if isinstance(section_val, dict) else None
        try:
            from tools.registry import registry
            out["tools.registry_generation"] = getattr(registry, "_generation", None)
        except Exception:
            out["tools.registry_generation"] = None
        # Honcho identity-mapping keys live in honcho.json, not user_config; only read that file
        # when Honcho is the active memory provider.
        provider = cfg_get(cfg, "memory", "provider")
        honcho = isinstance(provider, str) and provider.lower() == "honcho"
        out.update(cls._extract_honcho_cache_busting_config() if honcho else dict.fromkeys(cls._HONCHO_CACHE_BUSTING_KEYS))
        return out

    @staticmethod
    def _agent_config_signature(
        model: str, runtime: dict, enabled_toolsets: list, ephemeral_prompt: str,
        cache_keys: dict | None = None, user_id: str | None = None, user_id_alt: str | None = None,
        skip_context_files: bool = False,
    ) -> str:
        """Stable key from agent config: change → cached AIAgent rebuilt; unchanged → reused (frozen
        prompt + schemas for cache hits). ``user_id`` / ``user_id_alt`` participate because Honcho
        freezes them at init; omitting them in shared-thread keys would cross-attribute messages.

        ``user_id`` and ``user_id_alt`` are the runtime user identities carried by the current message's
        gateway source. They participate in the cache key because the Honcho memory provider freezes them
        into ``HonchoSessionManager`` at first-message init (see
        ``plugins/memory/honcho/__init__.py::_do_session_init``). Without them in the signature, a
        shared-thread session_key (one in which ``build_session_key`` intentionally omits the participant
        ID, e.g. ``thread_sessions_per_user=False``) would reuse the cached AIAgent across distinct users,
        causing the second user's messages to be attributed to the first user's resolved Honcho peer. This
        broke #27371's per-user-peer contract in multi-user gateways. Per-user agent rebuilds in shared
        threads trade prompt-cache warmth for correct memory attribution.
        """
        import hashlib, json as _j
        # Fingerprint the FULL credential, not a short prefix: OAuth/JWT-style tokens often share a
        # common prefix (e.g. "eyJhbGci"), so a prefix would give false cache hits across auth switches.
        _api_key = str(runtime.get("api_key", "") or "")
        blob = _j.dumps(
            [
                model,
                hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else "",
                runtime.get("base_url", ""), runtime.get("provider", ""),
                runtime.get("requested_provider", ""), runtime.get("api_mode", ""),
                sorted((runtime.get("capabilities") or {}).items()),
                sorted(enabled_toolsets) if enabled_toolsets else [],
                # reasoning_config excluded — set per-message on the cached agent; no prompt/tool effect.
                ephemeral_prompt or "",
                sorted((cache_keys or {}).items()),
                str(user_id or ""), str(user_id_alt or ""),
                # skip_context_files changes the agent's frozen system prompt (context files in vs out):
                # a toggled edit must rebuild the cached agent, not silently reuse it.
                bool(skip_context_files),
            ],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _session_model_override(self, session_key: str) -> Optional[dict]:
        """Current in-memory /model override for ``session_key`` (None when absent)."""
        state = self._peek_session_state(session_key)
        return state.conversation.model_override if state else None

    def _rehydrate_session_model_override(self, session_key: str) -> None:
        """Lazily restore a persisted /model override after a gateway restart: non-secret parts
        (model/provider/base_url) are written through on /model and read back on first use; api_key
        is never persisted and is re-resolved. No-op when an in-memory override or nothing exists."""
        from gateway.run import _resolve_runtime_agent_kwargs_for_provider
        store = getattr(self, "session_store", None)
        if self._session_model_override(session_key) is not None or store is None:
            return
        try:
            persisted = store.get_model_override(session_key)
        except Exception:
            logger.debug("Failed to read persisted session model override", exc_info=True)
            return
        if not persisted:
            return
        override: Dict[str, Any] = {k: persisted.get(k) for k in ("model", "provider", "base_url")}
        provider = persisted.get("provider")
        if provider:
            # Re-resolve credentials for the persisted provider. On failure (e.g. credentials removed
            # since the switch) keep the credential-less override — _resolve_session_agent_runtime
            # falls back to env resolution and layers model/provider.
            try:
                runtime = _resolve_runtime_agent_kwargs_for_provider(provider)
                for k in ("api_key", "api_mode", "credential_pool", "requested_provider", "max_tokens"):
                    override[k] = runtime.get(k)
                override["request_overrides"] = dict(runtime.get("request_overrides") or {})
                override["capabilities"] = dict(runtime.get("capabilities") or {})
                if not override.get("base_url"):
                    override["base_url"] = runtime.get("base_url")
            except Exception:
                logger.debug(
                    "Credential re-resolution failed for persisted override "
                    "(provider=%s); using credential-less override", provider, exc_info=True,
                )
        self._session_state(session_key).conversation.model_override = override
        logger.info(
            "Rehydrated persisted /model override for session=%s: model=%s provider=%s",
            session_key, override.get("model"), provider or "",
        )

    def _apply_session_model_override(self, session_key: str, model: str, runtime_kwargs: dict) -> tuple:
        """Apply /model session overrides (precedence over config.yaml defaults; ``None`` fields skipped
        so partial overrides don't clobber defaults), returning (model, runtime_kwargs)."""
        from gateway.run import _credential_pool_for_provider
        override = self._session_model_override(session_key)
        if not override:
            return model, runtime_kwargs
        model = override.get("model", model)
        for key in _OVERRIDE_APPLY_KEYS:
            val = override.get(key)
            if val is not None:
                runtime_kwargs[key] = val
        # request_overrides reflects the switched-to provider; apply whenever the override recorded
        # it (even as None) so switching to a provider without configured overrides clears a stale
        # value left by the default provider's runtime resolution.
        if "request_overrides" in override:
            ro = override.get("request_overrides")
            runtime_kwargs["request_overrides"] = dict(ro) if isinstance(ro, dict) and ro else ro
        if (
            runtime_kwargs.get("api_key")
            and runtime_kwargs.get("credential_pool") is None
            and override.get("provider")
        ):
            runtime_kwargs["credential_pool"] = _credential_pool_for_provider(override.get("provider"))
        return model, runtime_kwargs

    def _snapshot_session_model_override(self, session_key: str) -> dict:
        """Capture a gateway session override before a one-turn switch."""
        override = self._session_model_override(session_key)
        return {"had_override": override is not None, "override": dict(override) if override is not None else None}

    def _restore_session_model_override(self, session_key: str, snapshot: dict) -> None:
        """Restore the session override captured before a one-turn switch."""
        if not session_key:
            return
        if snapshot.get("had_override"):
            self._session_state(session_key).conversation.model_override = dict(snapshot.get("override") or {})
        elif (state := self._peek_session_state(session_key)) is not None:
            state.conversation.model_override = None
        self._evict_cached_agent(session_key)

    def _is_intentional_model_switch(self, session_key: str, agent_model: str) -> bool:
        """Return True if *agent_model* matches an active /model session override."""
        override = self._session_model_override(session_key)
        return override is not None and override.get("model") == agent_model

    def _release_running_agent_state(
        self, session_key: str, *, run_generation: Optional[int] = None
    ) -> bool:
        """Pop ALL per-running-agent state for ``session_key`` (call at every site that ends a running
        turn); True when cleared. Persistent state (model overrides, voice mode, approvals) is NOT
        touched. With ``run_generation``, only clear if still current — a stale async unwind bumped
        by /stop or /new must not clobber a newer run (returns False)."""
        if not session_key or (
            run_generation is not None and not self._is_session_run_current(session_key, run_generation)
        ):
            return False
        state = self._peek_session_state(session_key)
        if state is not None:
            if state.turn.lease is not None:
                try:
                    state.turn.lease.release()
                except Exception:
                    logger.debug("Failed to release active session slot", exc_info=True)
            # One structured reset instead of a drifting pop-list. Turn-lease tokens are deliberately NOT
            # cleared here — _release_turn_lease owns them.
            state.turn.clear()
        # Turn boundary: a running-agent slot was just released; persist the new (lower) in-flight count
        # so the dashboard readout stays current. Preserves gateway_state (see _persist_active_agents).
        self._persist_active_agents()
        return True

    def _held_turn_lease(self, session_key: str, run_generation: int):
        """Return ``(registry, turn)`` when ``session_key`` holds a lease token for ``run_generation``, else None."""
        registry = getattr(self, "_turn_leases", None)
        state = self._peek_session_state(session_key) if session_key and registry is not None else None
        if state is None or state.turn.lease_token is None or state.turn.lease_generation != run_generation:
            return None
        return registry, state.turn

    def _release_turn_lease(self, session_key: str, run_generation: int) -> bool:
        """Release the turn lease acquired by (``session_key``, ``run_generation``). Keyed by (routing
        key, run generation) so a stale unwind pops only ITS token; the registry's identity check
        refuses it if a newer turn holds the lease. Idempotent."""
        held = self._held_turn_lease(session_key, run_generation)
        if held is None:
            return False
        registry, turn = held
        token, turn.lease_token, turn.lease_generation = turn.lease_token, None, None
        try:
            return registry.release(token)
        except Exception:
            logger.debug("Failed to release turn lease", exc_info=True)
            return False

    def _rebind_turn_lease(self, session_key: str, run_generation: int, new_session_id: str) -> bool:
        """Follow a mid-turn session_id rotation (compression) with the held turn lease, or an alias
        key resolving the new id could start a concurrent turn the lease never sees. Call at every
        mid-turn reassignment; no-op if no token."""
        held = self._held_turn_lease(session_key, run_generation) if new_session_id else None
        if held is None:
            return False
        registry, turn = held
        try:
            return registry.rebind(turn.lease_token, new_session_id)
        except Exception:
            logger.debug("Failed to rebind turn lease", exc_info=True)
            return False

    def _clear_conversation_scope(self, session_key: str, *, reason: str) -> None:
        """THE single conversation-boundary funnel (/new, /resume, auto-reset, expiry finalization,
        compression-exhausted reset). New conversation-scoped dicts go in _CONVERSATION_SCOPED_STATE
        so every boundary picks them up. Turn-scoped state (_running_agents/_ts, slot leases, turn-
        lease tokens) is owned by _release_running_agent_state and NOT cleared. Idle agent-cache
        eviction is NOT a boundary (a resumed turn rebuilds from these). getattr-guarded.

        Why a funnel: these boundaries used to each carry a hand-copied pop-list of the per-session dicts,
        and the lists drifted every time a new dict was added (#48031, #58403, #10702, #35809 were all
        "boundary X forgot dict Y" bugs — e.g. /new cleared the /model override but not the /model --once
        restore snapshot). Adding a new conversation-scoped dict now means adding its attribute name to
        _CONVERSATION_SCOPED_STATE below; every boundary picks it up automatically.
        """
        from gateway.run import _CONVERSATION_SCOPED_STATE
        if not session_key:
            return
        state = self._peek_session_state(session_key)
        if state is not None:
            state.conversation.clear()
        # Legacy plain-dict stores still in _CONVERSATION_SCOPED_STATE (not yet folded into
        # SessionState), e.g. _pending_model_notes. SessionState-backed names resolve to MutableMapping
        # views (not dict), so the isinstance(dict) guard skips them — already handled above.
        for attr in _CONVERSATION_SCOPED_STATE:
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(session_key, None)
        self._clear_session_boundary_security_state(session_key)
        logger.debug("Cleared conversation scope for %s (%s)", session_key, reason)

    def _clear_session_boundary_security_state(self, session_key: str) -> None:
        """Clear per-session control state that must not survive a boundary switch."""
        if not session_key:
            return
        pending_skills_reload_notes = getattr(self, "_pending_skills_reload_notes", None)
        if isinstance(pending_skills_reload_notes, dict):
            pending_skills_reload_notes.pop(session_key, None)
        state = self._peek_session_state(session_key)
        if state is not None:
            state.persistent.approvals = None
            state.persistent.update_prompt_pending = False
        for mod, attr, what in (
            ("tools.slash_confirm", "clear", "slash-confirm"), ("tools.approval", "clear_session", "approval"),
        ):
            try:
                clear = getattr(importlib.import_module(mod), attr)
            except Exception:
                continue
            try:
                clear(session_key)
            except Exception as e:
                logger.debug("Failed to clear %s state for session boundary %s: %s", what, session_key, e)

    def _begin_session_run_generation(self, session_key: str) -> int:
        """Claim a fresh, monotonically increasing run generation token (NEVER reset): a late result
        from a worker /stop or /new invalidated is recognized and dropped."""
        if not session_key:
            return 0
        persistent = self._session_state(session_key).persistent
        # Monotonic by design (#28686): incremented here, NEVER reset.
        persistent.run_generation = int(persistent.run_generation) + 1
        return persistent.run_generation

    def _invalidate_session_run_generation(self, session_key: str, *, reason: str = "") -> int:
        """Invalidate any in-flight run token for ``session_key``."""
        generation = self._begin_session_run_generation(session_key)
        if reason:
            logger.info("Invalidated run generation for %s → %d (%s)", session_key, generation, reason)
        return generation

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        """Return True when ``generation`` is still current for ``session_key``."""
        if not session_key:
            return True
        state = self._peek_session_state(session_key)
        current = state.persistent.run_generation if state is not None else 0
        return int(current) == int(generation)

    def _bind_adapter_run_generation(self, adapter: Any, session_key: str, generation: int | None) -> None:
        """Bind a gateway run generation to the adapter's active-session event."""
        if not adapter or not session_key or generation is None:
            return
        with suppress(Exception):
            interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
            if interrupt_event is not None:
                interrupt_event._hermes_run_generation = int(generation)

    async def _interrupt_and_clear_session(
        self, session_key: str, source: SessionSource, *, interrupt_reason: str,
        invalidation_reason: str, release_running_state: bool = True,
    ) -> None:
        """Interrupt the current run and clear queued session state consistently."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _reap_gateway_turn_processes, request_hard_interrupt
        if not session_key:
            return
        state = self._peek_session_state(session_key)
        running_agent = state.turn.agent if state else None
        _process_task_id, _process_baseline = "", None
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            request_hard_interrupt(running_agent, interrupt_reason)
            _process_task_id = getattr(running_agent, "_gateway_turn_process_task_id", "")
            _process_baseline = getattr(running_agent, "_gateway_turn_process_baseline", None)
        # Bump the generation BEFORE scheduling the reap thread and capture the post-bump value:
        # task_id is session-scoped, so a replacement turn spawning before the reap runs bumps it
        # again and the closure sees a stale generation and skips — the replacement's own baseline
        # covers its cleanup, so nothing stays unreaped.
        _generation_at_interrupt = self._invalidate_session_run_generation(session_key, reason=invalidation_reason)
        if _process_task_id and _process_baseline is not None:
            threading.Thread(
                target=_reap_gateway_turn_processes,
                args=(_process_task_id, _process_baseline),
                kwargs={
                    "source": "gateway_turn_interrupt",
                    "is_still_current": lambda: self._is_session_run_current(session_key, _generation_at_interrupt),
                },
                name=f"gateway-turn-reaper-{_process_task_id[:12]}",
                daemon=True,
            ).start()
        adapter = self._adapter_for_source(source)
        interrupt_session_activity = getattr(type(adapter), "interrupt_session_activity", None)
        if adapter and callable(interrupt_session_activity):
            metadata = self._thread_metadata_for_source(source)
            if _accepts_keyword(interrupt_session_activity, "metadata"):
                await adapter.interrupt_session_activity(session_key, source.chat_id, metadata=metadata)
            else:
                await adapter.interrupt_session_activity(session_key, source.chat_id)
        if adapter and hasattr(adapter, "get_pending_message"):
            adapter.get_pending_message(session_key)  # consume and discard
        if state is not None:
            state.persistent.pending_command_text = None
        if release_running_state:
            self._release_running_agent_state(session_key)
            # Evict the cached agent: ``_interrupt_requested`` is only cleared by the turn finalizer,
            # so on a hung/still-draining run the flag survives and silently kills the session's NEXT
            # message (interrupted=True, api_calls=0, empty response). Like /new and /model, the next
            # message rebuilds from history; the old agent keeps its flag so a hung drain still dies.
            # See #44212.
            self._evict_cached_agent(session_key)

    async def _refresh_agent_cache_message_count(self, session_key: str, session_id: Optional[str]) -> None:
        """Re-baseline a cached agent's stored message_count after THIS turn — the coherence guard
        rebuilds on mismatch, so without this every turn would rebuild and destroy prompt caching.
        Only the count is refreshed, only if the same agent is still cached. DB errors leave the
        snapshot as-is (one spare rebuild).

        But the snapshot is taken at agent-BUILD time — before this turn writes its own user + assistant (+
        tool) rows — and the cache entry is never rewritten on a reuse. See #45966.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if self._session_db is None or not session_id or not _cache_lock or _cache is None:
            return
        try:
            _sess_row = await self._session_db.get_session(session_id)
            _live = _sess_row.get("message_count", 0) if _sess_row else None
        except Exception:
            return
        if _live is None:
            return
        with _cache_lock:
            cached = _cache.get(session_key)
            # Only re-baseline a live 3-tuple entry; skip pending sentinels, legacy 2-tuples (they opt
            # out of the guard), and entries evicted/rebuilt mid-turn. A snapshot taken for a different
            # session_id (same session_key, different conversation) is a different DB row — leave it.
            if not (isinstance(cached, tuple) and len(cached) > 2 and cached[0] is not _AGENT_PENDING_SENTINEL):
                return
            _snapshot_sid = cached[3] if len(cached) > 3 else None
            if (_snapshot_sid is not None and _snapshot_sid != session_id) or cached[2] == _live:
                return
            # Legacy 3-tuple keeps its 3-element shape for callers indexing ``cached[2]``.
            _cache[session_key] = (cached[0], cached[1], _live) + (() if _snapshot_sid is None else (_snapshot_sid,))

    def _set_pending_turn_sidecar_notes(self, session_key: str, notes: List[str]) -> None:
        """Stage per-turn must-deliver notes for the next agent run (one-shot)."""
        if not session_key or not notes:
            return
        self._session_state(session_key).conversation.sidecar_notes = list(notes)

    def _consume_pending_turn_sidecar_notes(self, session_key: str) -> List[str]:
        state = self._peek_session_state(session_key) if session_key else None
        if state is None:
            return []
        staged, state.conversation.sidecar_notes = state.conversation.sidecar_notes, []
        return list(staged) if isinstance(staged, list) else []

    def _voice_channel_sidecar_note(self, event, source: SessionSource, session_key: str) -> Optional[str]:
        """``[Voice channel now: ...]`` note when VC state changed; ``None`` when unchanged so per-turn
        member/speaking churn can't touch the prompt."""
        if source.platform != Platform.DISCORD:
            return None
        adapter = self.adapters.get(Platform.DISCORD)
        guild_id = self._get_guild_id(event)
        if not (guild_id and adapter and hasattr(adapter, "get_voice_channel_context")):
            return None
        try:
            vc_now = adapter.get_voice_channel_context(guild_id) or ""
        except Exception:
            logger.debug("voice-channel context read failed", exc_info=True)
            return None
        vc_prev = None
        if session_key:
            _vc_state = self._session_state(session_key)
            vc_prev, _vc_state.conversation.vc_last = _vc_state.conversation.vc_last, vc_now
        if vc_now == (vc_prev if vc_prev is not None else ""):
            return None
        return f"[Voice channel now: {vc_now or 'not connected to a voice channel'}]"

    def _pinned_session_context_prompt(self, context, redact_pii: bool, session_key: Optional[str]) -> str:
        """Session-context prompt pinned per session: key hit → pinned bytes reused VERBATIM (immune
        to renderer nondeterminism); key miss → re-render and re-pin (rename, topic edit, /sethome)."""
        _eph_key = self._ephemeral_change_key(context, redact_pii)
        _pin_state = self._peek_session_state(session_key) if session_key else None
        _eph_pin = _pin_state.conversation.ephemeral_pin if _pin_state else None
        if _eph_pin is not None and _eph_pin[0] == _eph_key:
            return _eph_pin[1]
        text = build_session_context_prompt(context, redact_pii=redact_pii)
        if session_key:
            self._session_state(session_key).conversation.ephemeral_pin = (_eph_key, text)
        return text

    @staticmethod
    def _ephemeral_change_key(context, redact_pii: bool) -> str:
        """Hash the exact inputs ``build_session_context_prompt`` renders. Invariant
        (test_prompt_tail_freeze.py): any input whose change alters the rendered bytes MUST appear
        here — omission means a stale pinned prompt; extras only re-render."""
        import hashlib
        src = context.source

        def _s(v) -> str:
            return str(v or "")

        discord_ids: tuple = ()
        discord_tools = ""
        if src.platform == Platform.DISCORD:
            from gateway.session import _discord_tools_loaded
            discord_tools = "1" if _discord_tools_loaded() else "0"
            # message_id: only PRESENCE is rendered (the id itself arrives per-turn in the user
            # message) — keying on the value would re-render every message for zero byte change.
            discord_ids = (
                _s(src.guild_id), _s(src.parent_chat_id), _s(src.thread_id), _s(src.chat_id),
                "1" if src.message_id else "0",
            )
        # Slack's capability-aware platform note is gated on _slack_tools_loaded() — the gate state must
        # be in the key (same parity contract as the Discord gate above) so a config / MCP-registration
        # flip re-renders once instead of serving a stale pinned note for the rest of the session.
        slack_tools = ""
        if src.platform == Platform.SLACK:
            from gateway.session import _slack_tools_loaded
            slack_tools = "1" if _slack_tools_loaded() else "0"
        try:
            from hermes_constants import display_hermes_home
            home_display = str(display_hermes_home())
        except Exception:
            home_display = ""
        key_tuple = (
            src.platform.value if src.platform else "",
            _s(src.chat_id), _s(src.thread_id), _s(src.chat_type), _s(src.chat_name), _s(src.chat_topic),
            _s(src.user_name), _s(src.user_id), _s(getattr(src, "profile", None)),
            bool(context.shared_multi_user_session), discord_ids, discord_tools, slack_tools,
            tuple(p.value for p in context.connected_platforms),
            tuple(
                (p.value, _s(getattr(hc, "name", "")), _s(getattr(hc, "chat_id", "")))
                for p, hc in context.home_channels.items()
            ),
            bool(redact_pii), home_display,
        )
        return hashlib.sha256(repr(key_tuple).encode("utf-8")).hexdigest()

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent (/new, /model, ...) and soft-release its LLM client pool (AIAgent
        holds reference cycles; without it RSS grows across /new). Soft = frees clients and child
        subagents but PRESERVES terminal sandbox / browser / bg processes since the session may
        resume; true boundaries call ``_cleanup_agent_resources`` first. Cleanup runs on a daemon
        thread so ``_agent_cache_lock`` never spans slow socket teardown.

        Pops the entry AND soft-releases the evicted agent's LLM client pool so the httpx connection
        (sockets + held buffers) is freed promptly rather than waiting on CPython GC — AIAgent holds
        reference cycles (callbacks, tool state) that delay refcount collection, so a manual release is
        required to keep gateway RSS flat across many /new, /model, undo and reset operations (#29298, same
        leak class as #25315).
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        # Prompt-stability state rides the agent-cache lifecycle: a fresh agent must re-render its
        # session-context bytes (the pin) and re-see the current voice-channel state once.
        state = self._peek_session_state(session_key)
        if state is not None:
            state.conversation.ephemeral_pin = None
            state.conversation.vc_last = None
        # Tests build runners with ``_agent_cache_lock = None``; evict lock-free then. With the lock
        # present ``_agent_cache`` is read directly (an initialized runner always has it).
        _lock = getattr(self, "_agent_cache_lock", None)
        evicted = None
        if _lock:
            with _lock:
                evicted = self._agent_cache.pop(session_key, None)
        else:
            _cache = getattr(self, "_agent_cache", None)
            if _cache is not None:
                evicted = _cache.pop(session_key, None)
        agent = _first_agent(evicted)
        # Never tear down an agent that's mid-turn — its client, sandbox and child subagents are in use.
        if agent is None or agent is _AGENT_PENDING_SENTINEL or id(agent) in self._running_agent_ids():
            return
        self._spawn_release_thread(
            self._release_evicted_agent_soft, (agent,), f"agent-evict-{str(session_key)[:24]}", inline_fallback=True,
        )

    def _spawn_release_thread(self, target, args: tuple, name: str, *, inline_fallback: bool) -> None:
        """Run a release on a daemon thread. ``inline_fallback`` runs it inline (best-effort) when no
        thread can start (interpreter shutdown); otherwise a spawn failure propagates, as on main."""
        try:
            threading.Thread(target=target, args=args, daemon=True, name=name).start()
        except Exception:
            if not inline_fallback:
                raise
            with suppress(Exception):
                target(*args)

    def _finalizable_unexpired_session_entry(self, key: str):
        """Session-store entry for ``key`` when the expiry watcher will still finalize it; None when
        missing, not finalizable (``mode == "none"``) or already expired (the watcher handles those)."""
        _store = getattr(self, "session_store", None)
        if _store is None:
            return None
        try:
            _store._ensure_loaded()
            entry = _store._entries.get(key)
        except Exception:
            return None
        ok = entry is not None and _store.is_session_finalizable(entry) and not _store._is_session_expired(entry)
        return entry if ok else None

    def _commit_memory_before_soft_evict(self, agent: Any, key: str) -> None:
        """Fire on_session_end extraction before soft-evicting a live agent: the expiry watcher only
        finalizes what it finds in ``_agent_cache``, so an LRU soft-evict first would hide the
        transcript from memory providers. Commit via ``commit_memory_session`` (no teardown), only for
        finalizable, not-yet-expired sessions. Best-effort."""
        # No external memory provider (``_memory_manager`` None) — nothing to commit.
        if agent is None or not hasattr(agent, "commit_memory_session") or getattr(agent, "_memory_manager", None) is None:
            return
        try:
            if self._finalizable_unexpired_session_entry(key) is None:
                return
            messages = getattr(agent, "_session_messages", None)
            agent.commit_memory_session(messages if isinstance(messages, list) else None)
            logger.debug(
                "Committed on_session_end extraction before soft-evicting "
                "finalizable session=%s (cache pressure, pre-expiry)", key,
            )
        except Exception as _e:
            logger.debug("Pre-evict memory commit failed for %s: %s", key, _e)

    def _commit_then_release_soft(self, agent: Any, key: str) -> None:
        """Commit end-of-session memory (if warranted), then soft-release — on the daemon eviction
        thread. Order matters: commit needs the live memory manager before ``release_clients``."""
        self._commit_memory_before_soft_evict(agent, key)
        self._release_evicted_agent_soft(agent)

    def _release_evicted_agent_soft(self, agent: Any) -> None:
        """Soft cleanup for cache-evicted agents: unlike _cleanup_agent_resources, the session may
        resume, so terminal sandbox, browser daemon and bg processes outlive the AIAgent instance."""
        if agent is None:
            return
        with suppress(Exception):
            if hasattr(agent, "release_clients"):
                agent.release_clients()
            else:
                # Older agent instance (shouldn't happen in practice) — legacy full-close path.
                self._cleanup_agent_resources(agent)
        # Free conversation history — tens of MB of tool output on heavy 100+-tool-call sessions.
        # release_clients() preserves session tool state for resume, but the message list is rebuilt from
        # persisted session JSON on the next turn, so dropping it here is safe.
        if hasattr(agent, "_session_messages"):
            agent._session_messages = []
        # _db_flush_scan_prefix (run_agent.py, stamped on every successful flush) is a shallow copy
        # sharing every message dict of the flushed transcript, so leaving it pins the multi-MB strings
        # this eviction frees. Pressure-evictable agents have flushed by definition, so it's populated.
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = None

    def _agent_cache_bounds(self):
        """Operator-configured agent-cache bounds, resolved once per process (lazily, not in
        ``__init__``, so ``__new__``-constructed test / slash-command runners work too)."""
        from gateway.run import _load_gateway_config
        bounds = getattr(self, "_agent_cache_bounds_cache", None)
        if bounds is None:
            from gateway.agent_cache_pressure import resolve_agent_cache_bounds
            try:
                bounds = resolve_agent_cache_bounds(_load_gateway_config())
            except Exception as _e:
                logger.debug("Agent cache bounds config read failed: %s", _e)
                # Resolve from an empty config rather than bare AgentCacheBounds(): the dataclass default
                # has memory_high_mb=None (pressure pass OFF) but an *absent* section means "auto" — a
                # transient config read failure must not permanently disable the OOM valve.
                bounds = resolve_agent_cache_bounds({})
            self._agent_cache_bounds_cache = bounds
        return bounds

    def _agent_cache_cap(self) -> int:
        """Effective LRU cap — the configured override, else the default."""
        from gateway.run import _AGENT_CACHE_MAX_SIZE
        return self._agent_cache_bounds().max_size or _AGENT_CACHE_MAX_SIZE

    def _agent_cache_idle_ttl(self) -> float:
        """Effective idle TTL in seconds — configured override, else default."""
        from gateway.run import _AGENT_CACHE_IDLE_TTL_SECS
        return self._agent_cache_bounds().idle_ttl_secs or _AGENT_CACHE_IDLE_TTL_SECS

    def _sweep_agent_cache_under_pressure(self) -> int:
        """Shed cached transcripts once the gateway heap nears its budget; returns count evicted.

        The LRU cap counts entries and the idle sweep counts seconds; neither knows one cached agent
        pins a full ``_session_messages`` transcript (tens of MB), so RSS climbs until the cgroup
        throttles. Above the anonymous-RSS budget this soft-evicts LRU agents (transcript rebuilt
        from the persisted session next turn). Never touched: agents mid-turn, the most recently
        used sessions, and transcripts not yet on disk.

        A gateway serving many chats therefore holds every warm transcript indefinitely: agents that took a
        turn within the TTL are never idle-swept, and the sweep additionally defers finalizable sessions
        until they expire. RSS climbs until the cgroup throttles and SIGTERM can no longer flush inside
        systemd's stop timeout (#80764).
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        from gateway.agent_cache_pressure import (
            plan_pressure_evictions, read_anon_rss_mb, transcript_persistence_caught_up
        )
        bounds = self._agent_cache_bounds()
        _cache = getattr(self, "_agent_cache", None)
        _lock = getattr(self, "_agent_cache_lock", None)
        # Nothing cached — whatever is using the heap, it isn't us, and warning about it every tick
        # would point at the wrong subsystem.
        if not bounds.memory_high_mb or not _cache or _lock is None:
            return 0
        rss_mb = read_anon_rss_mb()
        if rss_mb is None or rss_mb < bounds.memory_high_mb:
            return 0
        running_ids = self._running_agent_ids()

        def _is_live(agent: Any) -> bool:
            return agent is not None and agent is not _AGENT_PENDING_SENTINEL and id(agent) not in running_ids

        def _is_evictable(key: str, agent: Any) -> bool:
            return _is_live(agent) and transcript_persistence_caught_up(agent)

        with _lock:
            ordered = [(key, _first_agent(entry)) for key, entry in _cache.items()]
            plan = plan_pressure_evictions(
                ordered, is_evictable=_is_evictable, max_evictions=bounds.max_evictions_per_pass,
                protect_recent=bounds.protect_recent,
            )
            for key, _ in plan:
                _cache.pop(key, None)
        if not plan:
            _mid_turn = sum(1 for _, a in ordered if a is not None and id(a) in running_ids)
            _unflushed = sum(1 for _, a in ordered if _is_live(a) and not transcript_persistence_caught_up(a))
            logger.warning(
                "Agent cache pressure: anon RSS %dMB over budget %dMB but no "
                "evictable session (%d cached, %d mid-turn, %d blocked on "
                "un-flushed persistence)%s",
                rss_mb, bounds.memory_high_mb, len(ordered), _mid_turn, _unflushed,
                (
                    " — transcripts are not reaching the session DB "
                    "(session persistence disabled or failing?); the memory "
                    "valve cannot shed sessions until they persist."
                    if _unflushed and not _mid_turn
                    else " — memory will keep climbing until those turns finish."
                ),
            )
            return 0
        evicted_count = len(plan)
        logger.warning(
            "Agent cache pressure: anon RSS %dMB over budget %dMB — evicting %d LRU session(s): %s",
            rss_mb, bounds.memory_high_mb, evicted_count, ", ".join(key for key, _ in plan),
        )
        try:
            threading.Thread(target=self._release_pressure_batch, args=(plan,), daemon=True,
                             name="agent-cache-pressure").start()
        except Exception:
            # Thread spawn failed (interpreter shutdown): release inline, unguarded (as on main).
            self._release_pressure_batch(plan)
        # _release_pressure_batch drains `plan` in place (so the trim runs with no lingering agent
        # refs) — len(plan) is 0 once the daemon thread finishes, hence the pre-captured count.
        return evicted_count

    def _release_pressure_batch(self, plan: List[tuple]) -> None:
        """Release a pressure-evicted batch sequentially on one daemon thread, then ``malloc_trim`` so
        RSS actually falls. The plan is drained (``pop`` + ``del``), not iterated, so no local
        reference pins evicted agents during ``gc.collect`` + trim (else the valve over-evicts)."""
        while plan:
            key, agent = plan.pop(0)  # FIFO — evict LRU-first order preserved
            try:
                self._commit_then_release_soft(agent, key)
            except Exception as _e:
                logger.debug("Pressure release failed for %s: %s", key, _e)
            del agent
        with suppress(Exception):
            from hermes_cli.mem_trim import trim_memory
            trim_memory(force=True, reason="agent_cache_pressure")

    def _enforce_agent_cache_cap(self) -> None:
        """Evict oldest cached agents past the LRU cap (requires _agent_cache_lock); cleanup on a
        daemon thread. Mid-turn agents are SKIPPED, so the cache may stay over cap until the next
        insert."""
        _cache = getattr(self, "_agent_cache", None)
        # OrderedDict.popitem(last=False) pops oldest; plain dict lacks the arg so skip enforcement
        # if a test fixture swapped the cache type.
        if _cache is None or not hasattr(_cache, "move_to_end"):
            return
        # Snapshot of agent instances mid-turn, keyed by id() so lookup is O(1) and independent of
        # AIAgent.__eq__ (which MagicMock overrides in tests).
        running_ids = self._running_agent_ids()
        # Walk LRU → MRU; only the first (size - cap) LRU positions are candidates. An active slot is
        # SKIPPED rather than evicting a newer entry — that would penalise a fresh session (no cache
        # history) to protect a long-running one. Cache may stay over cap until the next insert.
        cap = self._agent_cache_cap()
        candidates = [(key, _tuple_agent(_cache.get(key))) for key in list(_cache.keys())[:max(0, len(_cache) - cap)]]
        evict_plan = [(key, agent) for key, agent in candidates if agent is None or id(agent) not in running_ids]
        for key, _ in evict_plan:
            _cache.pop(key, None)
        remaining_over_cap = len(_cache) - cap
        if remaining_over_cap > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(_cache), cap, remaining_over_cap,
            )
        for key, agent in evict_plan:
            logger.info("Agent cache at cap; evicting LRU session=%s (cache_size=%d)", key, len(_cache))
            if agent is not None:
                # Commit end-of-session memory, then soft-release, both on the daemon thread so the
                # (possibly network-bound) provider call never blocks the held cache lock.
                self._spawn_release_thread(self._commit_then_release_soft, (agent, key), f"agent-cache-evict-{key[:24]}", inline_fallback=False)

    def _sweep_idle_cached_agents(self) -> int:
        """Evict cached agents idle past the idle TTL (lock acquired internally; cleanup on daemon
        threads; mid-turn agents SKIPPED); returns the number evicted."""
        _cache = getattr(self, "_agent_cache", None)
        _lock = getattr(self, "_agent_cache_lock", None)
        if _cache is None or _lock is None:
            return 0
        now = time.time()
        idle_ttl = self._agent_cache_idle_ttl()
        to_evict: List[tuple] = []
        running_ids = self._running_agent_ids()
        with _lock:
            for key, entry in list(_cache.items()):
                agent = _tuple_agent(entry)
                if agent is None or id(agent) in running_ids:
                    continue  # mid-turn — don't tear it down
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None or (now - last_activity) <= idle_ttl:
                    continue
                # Not yet expired in the store (daily-reset fires hours after the last message): keep
                # the agent so the expiry watcher can call on_session_end() with the live transcript.
                # Only defer when the watcher will EVER finalize it — for mode == "none" deferring pins
                # the agent for the gateway's lifetime (the leak this sweep relieves); those soft-evict
                # WITHOUT on_session_end, correctly.
                if self._finalizable_unexpired_session_entry(key) is not None:
                    continue
                to_evict.append((key, agent))
            for key, _ in to_evict:
                _cache.pop(key, None)
        for key, agent in to_evict:
            logger.info("Agent cache idle-TTL evict: session=%s (idle=%.0fs)", key, now - getattr(agent, "_last_activity_ts", now))
            self._spawn_release_thread(self._release_evicted_agent_soft, (agent,), f"agent-cache-idle-{key[:24]}", inline_fallback=False)
        return len(to_evict)
