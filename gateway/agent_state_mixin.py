"""Agent cache and per-session runtime state for the messaging gateway.

This mixin is a behavior-neutral extraction from gateway.run.GatewayRunner.
Runtime-owned globals that tests and integrations patch through gateway.run
are resolved lazily to preserve that compatibility surface.
"""

from __future__ import annotations

import inspect
import threading
from typing import Any, Dict, List, Optional

from gateway.config import Platform
from gateway.session import SessionSource, build_session_context_prompt
from hermes_cli.config import cfg_get

# Shared with gateway.run so identity checks across the gateway use one object.
_AGENT_PENDING_SENTINEL = object()


def _gateway_runtime():
    """Return the fully initialized gateway.run module at method call time."""
    import gateway.run as gateway_run

    return gateway_run


class _GatewayLoggerProxy:
    """Preserve the established ``gateway.run.logger`` patch surface."""

    def __getattr__(self, name: str):
        return getattr(_gateway_runtime().logger, name)


logger = _GatewayLoggerProxy()


class GatewayAgentStateMixin:
    """Cached-agent lifecycle, session overrides, and run-generation guards."""

    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    # Config keys whose values MUST invalidate the gateway's cached agent
    # when they change.  The agent bakes these into its compressor / context
    # handling at construction time, so a mid-running-gateway config edit
    # would otherwise be silently ignored until the user triggers a
    # different cache eviction (model switch, /reset, etc.).
    #
    # Each entry is a tuple of (section, key) read from the raw config dict.
    # Add more here as new baked-at-construction config settings are added.
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"),
        ("model", "max_tokens"),
        ("compression", "enabled"),
        ("compression", "threshold"),
        ("compression", "codex_gpt55_autoraise"),
        ("compression", "codex_app_server_auto"),
        ("compression", "target_ratio"),
        ("compression", "protect_last_n"),
        ("agent", "disabled_toolsets"),
        ("memory", "provider"),
        ("checkpoints", "enabled"),
        ("checkpoints", "max_snapshots"),
        ("checkpoints", "max_total_size_mb"),
        ("checkpoints", "max_file_size_mb"),
    )

    _HONCHO_CACHE_BUSTING_KEYS = (
        "honcho.peer_name",
        "honcho.ai_peer",
        "honcho.pin_peer_name",
        "honcho.runtime_peer_prefix",
        "honcho.user_peer_aliases",
    )
    _HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}

    @classmethod
    def _empty_honcho_cache_busting_config(cls) -> dict[str, Any]:
        return {key: None for key in cls._HONCHO_CACHE_BUSTING_KEYS}

    @classmethod
    def _extract_honcho_cache_busting_config(cls) -> dict[str, Any]:
        """Extract Honcho identity keys, memoized by honcho.json mtime."""
        try:
            from plugins.memory.honcho.client import (
                HonchoClientConfig,
                resolve_config_path,
            )

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
                "honcho.user_peer_aliases": sorted(aliases.items())
                if isinstance(aliases, dict)
                else [],
            }
            cls._HONCHO_CACHE_BUSTING_MEMO = {memo_key: values}
            return dict(values)
        except Exception:
            return cls._empty_honcho_cache_busting_config()

    @classmethod
    def _extract_cache_busting_config(cls, user_config: dict | None) -> dict:
        """Pull values that must bust the cached agent.

        Returns a flat dict keyed by 'section.key'.  Missing config keys and
        non-dict sections yield None values, which still contribute to the
        signature (so 'absent' vs 'present-and-null' differ).

        The live tool registry generation is included too.  MCP reloads and
        dynamic MCP tool-list changes mutate the registry without necessarily
        changing config.yaml.  Cached AIAgent instances freeze their tool
        schemas at construction time, so a registry generation change must
        rebuild the agent before the next turn.
        """
        out: Dict[str, Any] = {}
        cfg = user_config if isinstance(user_config, dict) else {}
        for section, key in cls._CACHE_BUSTING_CONFIG_KEYS:
            section_val = cfg.get(section)
            if section == "checkpoints" and isinstance(section_val, bool):
                # Preserve legacy ``checkpoints: true`` behavior. A live
                # toggle must still rebuild the cached agent.
                out[f"{section}.{key}"] = section_val if key == "enabled" else None
            elif isinstance(section_val, dict):
                out[f"{section}.{key}"] = section_val.get(key)
            else:
                out[f"{section}.{key}"] = None
        try:
            from tools.registry import registry

            out["tools.registry_generation"] = getattr(registry, "_generation", None)
        except Exception:
            out["tools.registry_generation"] = None

        # Honcho identity-mapping keys live in honcho.json, not user_config.
        # Only read that file when Honcho is the active memory provider.
        provider = cfg_get(cfg, "memory", "provider")
        if isinstance(provider, str) and provider.lower() == "honcho":
            out.update(cls._extract_honcho_cache_busting_config())
        else:
            out.update(cls._empty_honcho_cache_busting_config())

        return out

    @staticmethod
    def _agent_config_signature(
        model: str,
        runtime: dict,
        enabled_toolsets: list,
        ephemeral_prompt: str,
        cache_keys: dict | None = None,
        user_id: str | None = None,
        user_id_alt: str | None = None,
    ) -> str:
        """Compute a stable string key from agent config values.

        When this signature changes between messages, the cached AIAgent is
        discarded and rebuilt.  When it stays the same, the cached agent is
        reused — preserving the frozen system prompt and tool schemas for
        prompt cache hits.

        ``cache_keys`` is an optional flat dict of additional config values
        that should invalidate the cache when they change.  Callers pass
        the output of ``_extract_cache_busting_config(user_config)`` so
        edits to model.context_length / compression.* in config.yaml are
        picked up on the next gateway message without a manual restart.

        ``user_id`` and ``user_id_alt`` are the runtime user identities
        carried by the current message's gateway source.  They participate
        in the cache key because the Honcho memory provider freezes them
        into ``HonchoSessionManager`` at first-message init (see
        ``plugins/memory/honcho/__init__.py::_do_session_init``).  Without
        them in the signature, a shared-thread session_key (one in which
        ``build_session_key`` intentionally omits the participant ID,
        e.g. ``thread_sessions_per_user=False``) would reuse the cached
        AIAgent across distinct users, causing the second user's messages
        to be attributed to the first user's resolved Honcho peer.  This
        broke #27371's per-user-peer contract in multi-user gateways.
        Per-user agent rebuilds in shared threads trade prompt-cache
        warmth for correct memory attribution.
        """
        import hashlib, json as _j

        # Fingerprint the FULL credential string instead of using a short
        # prefix. OAuth/JWT-style tokens frequently share a common prefix
        # (e.g. "eyJhbGci"), which can cause false cache hits across auth
        # switches if only the first few characters are considered.
        _api_key = str(runtime.get("api_key", "") or "")
        _api_key_fingerprint = (
            hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""
        )

        _cache_keys_sorted = sorted((cache_keys or {}).items())

        blob = _j.dumps(
            [
                model,
                _api_key_fingerprint,
                runtime.get("base_url", ""),
                runtime.get("provider", ""),
                runtime.get("api_mode", ""),
                sorted(enabled_toolsets) if enabled_toolsets else [],
                # reasoning_config excluded — it's set per-message on the
                # cached agent and doesn't affect system prompt or tools.
                ephemeral_prompt or "",
                _cache_keys_sorted,
                str(user_id or ""),
                str(user_id_alt or ""),
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _rehydrate_session_model_override(self, session_key: str) -> None:
        """Lazily restore a persisted /model override after a gateway restart.

        ``_session_model_overrides`` is in-memory only, so before persistence
        a restart silently reverted every session to the global default model.
        The non-secret parts (model/provider/base_url) are written through to
        the session store when /model runs (and cleared on /new); here we read
        them back on first use and re-resolve credentials via the normal
        runtime provider resolution — api_key is never persisted to disk.

        No-op when an in-memory override already exists (live state wins) or
        when the store has nothing persisted (e.g. the user ran /new, which
        clears both the in-memory dict and the persisted field).
        """
        if session_key in self._session_model_overrides:
            return
        store = getattr(self, "session_store", None)
        if store is None:
            return
        try:
            persisted = store.get_model_override(session_key)
        except Exception:
            logger.debug(
                "Failed to read persisted session model override", exc_info=True
            )
            return
        if not persisted:
            return
        override: Dict[str, Any] = {
            "model": persisted.get("model"),
            "provider": persisted.get("provider"),
            "base_url": persisted.get("base_url"),
        }
        provider = persisted.get("provider")
        if provider:
            # Re-resolve credentials for the persisted provider. On failure
            # (e.g. credentials were removed since the switch) keep the
            # credential-less override — _resolve_session_agent_runtime falls
            # back to env-based resolution and applies model/provider on top.
            try:
                runtime = _gateway_runtime()._resolve_runtime_agent_kwargs_for_provider(
                    provider
                )
                override["api_key"] = runtime.get("api_key")
                override["api_mode"] = runtime.get("api_mode")
                override["credential_pool"] = runtime.get("credential_pool")
                if not override.get("base_url"):
                    override["base_url"] = runtime.get("base_url")
            except Exception:
                logger.debug(
                    "Credential re-resolution failed for persisted override "
                    "(provider=%s); using credential-less override",
                    provider,
                    exc_info=True,
                )
        self._session_model_overrides[session_key] = override
        logger.info(
            "Rehydrated persisted /model override for session=%s: model=%s provider=%s",
            session_key,
            override.get("model"),
            provider or "",
        )

    def _apply_session_model_override(
        self, session_key: str, model: str, runtime_kwargs: dict
    ) -> tuple:
        """Apply /model session overrides if present, returning (model, runtime_kwargs).

        The gateway /model command stores per-session overrides in
        ``_session_model_overrides``.  These must take precedence over
        config.yaml defaults so the switched model is actually used for
        subsequent messages.  Fields with ``None`` values are skipped so
        partial overrides don't clobber valid config defaults.
        """
        override = self._session_model_overrides.get(session_key)
        if not override:
            return model, runtime_kwargs
        model = override.get("model", model)
        for key in ("provider", "api_key", "base_url", "api_mode", "credential_pool"):
            val = override.get(key)
            if val is not None:
                runtime_kwargs[key] = val
        if (
            runtime_kwargs.get("api_key")
            and runtime_kwargs.get("credential_pool") is None
            and override.get("provider")
        ):
            runtime_kwargs["credential_pool"] = (
                _gateway_runtime()._credential_pool_for_provider(
                    override.get("provider")
                )
            )
        return model, runtime_kwargs

    def _snapshot_session_model_override(self, session_key: str) -> dict:
        """Capture a gateway session override before a one-turn switch."""
        override = self._session_model_overrides.get(session_key)
        return {
            "had_override": override is not None,
            "override": dict(override) if override is not None else None,
        }

    def _restore_session_model_override(self, session_key: str, snapshot: dict) -> None:
        """Restore the session override captured before a one-turn switch."""
        if not session_key:
            return
        if snapshot.get("had_override"):
            self._session_model_overrides[session_key] = dict(
                snapshot.get("override") or {}
            )
        else:
            self._session_model_overrides.pop(session_key, None)
        self._evict_cached_agent(session_key)

    def _is_intentional_model_switch(self, session_key: str, agent_model: str) -> bool:
        """Return True if *agent_model* matches an active /model session override."""
        override = self._session_model_overrides.get(session_key)
        return override is not None and override.get("model") == agent_model

    def _release_running_agent_state(
        self,
        session_key: str,
        *,
        run_generation: Optional[int] = None,
    ) -> bool:
        """Pop ALL per-running-agent state entries for ``session_key``.

        Replaces ad-hoc ``del self._running_agents[key]`` calls scattered
        across the gateway.  Those sites had drifted: some popped only
        ``_running_agents``; some also ``_running_agents_ts``; only one
        path also cleared ``_busy_ack_ts``.  Each missed entry was a
        small, persistent leak — a (str_key → float) tuple per session
        per gateway lifetime.

        Use this at every site that ends a running turn, regardless of
        cause (normal completion, /stop, /reset, /resume, sentinel
        cleanup, stale-eviction).  Per-session state that PERSISTS
        across turns (``_session_model_overrides``, ``_voice_mode``,
        ``_pending_approvals``, ``_update_prompt_pending``) is NOT
        touched here — those have their own lifecycles.

        When ``run_generation`` is provided, only clear the slot if that
        generation is still current for the session.  This prevents an
        older async run whose generation was bumped by /stop or /new from
        clobbering a newer run's state during its own unwind.  Returns
        True when the slot was cleared, False when an ownership guard
        blocked it.
        """
        if not session_key:
            return False
        if run_generation is not None and not self._is_session_run_current(
            session_key, run_generation
        ):
            return False
        lease = getattr(self, "_active_session_leases", {}).pop(session_key, None)
        if lease is not None:
            try:
                lease.release()
            except Exception:
                logger.debug("Failed to release active session slot", exc_info=True)
        self._running_agents.pop(session_key, None)
        self._running_agents_ts.pop(session_key, None)
        if hasattr(self, "_busy_ack_ts"):
            self._busy_ack_ts.pop(session_key, None)
        # Turn boundary: a running-agent slot was just released.  Persist the
        # new (lower) in-flight count so the dashboard readout stays current
        # between lifecycle transitions.  Preserves gateway_state (see
        # _persist_active_agents).
        self._persist_active_agents()
        return True

    def _release_turn_lease(self, session_key: str, run_generation: int) -> bool:
        """Release the turn lease acquired by (``session_key``, ``run_generation``).

        Companion to the acquisition in ``_handle_message_with_agent``
        (#64934). The token map is keyed by (routing key, run generation), so
        this can only ever free the lease its own turn acquired — a stale
        unwind whose generation was bumped by /stop or /new pops ITS token,
        and the registry's identity check refuses it if a newer turn already
        holds the lease. Idempotent and safe for bare test runners built via
        ``object.__new__`` (getattr defaults).
        """
        if not session_key:
            return False
        tokens = getattr(self, "_turn_lease_tokens", None)
        registry = getattr(self, "_turn_leases", None)
        if tokens is None or registry is None:
            return False
        token = tokens.pop((session_key, run_generation), None)
        if token is None:
            return False
        try:
            return registry.release(token)
        except Exception:
            logger.debug("Failed to release turn lease", exc_info=True)
            return False

    def _rebind_turn_lease(
        self, session_key: str, run_generation: int, new_session_id: str
    ) -> bool:
        """Follow a mid-turn session_id rotation with the held turn lease.

        Compression (session-hygiene pre-compression or the agent's own
        compressor) can rotate ``session_entry.session_id`` while this turn
        is in flight. The turn's flush targets the NEW id, so the
        serialization boundary must follow it — otherwise an alias routing
        key resolving the new id (topic tip-walk onto the fresh child) could
        start a concurrent turn the lease never sees (#64934 rotation-alias
        window). Call at every site that reassigns session_entry.session_id
        mid-turn. Fail-open no-op when there is no held token.
        """
        if not session_key or not new_session_id:
            return False
        tokens = getattr(self, "_turn_lease_tokens", None)
        registry = getattr(self, "_turn_leases", None)
        if tokens is None or registry is None:
            return False
        token = tokens.get((session_key, run_generation))
        if token is None:
            return False
        try:
            return registry.rebind(token, new_session_id)
        except Exception:
            logger.debug("Failed to rebind turn lease", exc_info=True)
            return False

    def _clear_conversation_scope(self, session_key: str, *, reason: str) -> None:
        """Clear ALL conversation-scoped per-session state for ``session_key``.

        THE single conversation-boundary funnel. Call this — and nothing
        else — whenever a session_key crosses a conversation boundary:
        /new, /resume, auto-reset (idle/daily/suspended), expiry
        finalization, and the compression-exhausted auto-reset.

        Why a funnel: these boundaries used to each carry a hand-copied
        pop-list of the per-session dicts, and the lists drifted every time
        a new dict was added (#48031, #58403, #10702, #35809 were all
        "boundary X forgot dict Y" bugs — e.g. /new cleared the /model
        override but not the /model --once restore snapshot). Adding a new
        conversation-scoped dict now means adding its attribute name to
        _CONVERSATION_SCOPED_STATE below; every boundary picks it up
        automatically.

        Scope rules:
        - Conversation-scoped (cleared here): model/reasoning overrides,
          one-turn restore snapshots, pending model notes, last-resolved
          model cache, queued follow-up events, and the boundary security
          state (approvals, /yolo, slash-confirm, update prompts).
        - Turn-scoped (NOT cleared here): _running_agents/_ts, slot leases,
          turn-lease tokens — owned by _release_running_agent_state and the
          dispatch finally.
        - Idle agent-cache eviction is NOT a conversation boundary: the
          session is still alive and a resumed turn rebuilds from these
          overrides. Only true boundaries call this.

        Safe on bare test runners built via ``object.__new__`` (every
        access is getattr-guarded).
        """
        if not session_key:
            return
        for attr in _gateway_runtime()._CONVERSATION_SCOPED_STATE:
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(session_key, None)
        self._clear_session_boundary_security_state(session_key)
        logger.debug("Cleared conversation scope for %s (%s)", session_key, reason)

    def _clear_session_boundary_security_state(self, session_key: str) -> None:
        """Clear per-session control state that must not survive a boundary switch."""
        if not session_key:
            return

        pending_skills_reload_notes = getattr(
            self, "_pending_skills_reload_notes", None
        )
        if isinstance(pending_skills_reload_notes, dict):
            pending_skills_reload_notes.pop(session_key, None)

        pending_approvals = getattr(self, "_pending_approvals", None)
        if isinstance(pending_approvals, dict):
            pending_approvals.pop(session_key, None)

        update_prompt_pending = getattr(self, "_update_prompt_pending", None)
        if isinstance(update_prompt_pending, dict):
            update_prompt_pending.pop(session_key, None)

        try:
            from tools import slash_confirm as _slash_confirm_mod
        except Exception:
            _slash_confirm_mod = None
        if _slash_confirm_mod is not None:
            try:
                _slash_confirm_mod.clear(session_key)
            except Exception as e:
                logger.debug(
                    "Failed to clear slash-confirm state for session boundary %s: %s",
                    session_key,
                    e,
                )

        try:
            from tools.approval import clear_session as _clear_approval_session
        except Exception:
            return

        try:
            _clear_approval_session(session_key)
        except Exception as e:
            logger.debug(
                "Failed to clear approval state for session boundary %s: %s",
                session_key,
                e,
            )

    def _begin_session_run_generation(self, session_key: str) -> int:
        """Claim a fresh run generation token for ``session_key``.

        Every top-level gateway turn gets a monotonically increasing token.
        If a later command like /stop or /new invalidates that token while the
        old worker is still unwinding, the late result can be recognized and
        dropped instead of bleeding into the fresh session.
        """
        if not session_key:
            return 0
        generations = self.__dict__.get("_session_run_generation")
        if generations is None:
            generations = {}
            self._session_run_generation = generations
        next_generation = int(generations.get(session_key, 0)) + 1
        generations[session_key] = next_generation
        return next_generation

    def _invalidate_session_run_generation(
        self, session_key: str, *, reason: str = ""
    ) -> int:
        """Invalidate any in-flight run token for ``session_key``."""
        generation = self._begin_session_run_generation(session_key)
        if reason:
            logger.info(
                "Invalidated run generation for %s → %d (%s)",
                session_key,
                generation,
                reason,
            )
        return generation

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        """Return True when ``generation`` is still current for ``session_key``."""
        if not session_key:
            return True
        generations = self.__dict__.get("_session_run_generation") or {}
        return int(generations.get(session_key, 0)) == int(generation)

    def _bind_adapter_run_generation(
        self,
        adapter: Any,
        session_key: str,
        generation: int | None,
    ) -> None:
        """Bind a gateway run generation to the adapter's active-session event."""
        if not adapter or not session_key or generation is None:
            return
        try:
            interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
            if interrupt_event is not None:
                setattr(interrupt_event, "_hermes_run_generation", int(generation))
        except Exception:
            pass

    async def _interrupt_and_clear_session(
        self,
        session_key: str,
        source: SessionSource,
        *,
        interrupt_reason: str,
        invalidation_reason: str,
        release_running_state: bool = True,
    ) -> None:
        """Interrupt the current run and clear queued session state consistently."""
        if not session_key:
            return
        running_agent = self._running_agents.get(session_key)
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            running_agent.interrupt(interrupt_reason)
        self._invalidate_session_run_generation(session_key, reason=invalidation_reason)
        adapter = self._adapter_for_source(source)
        interrupt_session_activity = getattr(
            type(adapter), "interrupt_session_activity", None
        )
        if adapter and callable(interrupt_session_activity):
            metadata = self._thread_metadata_for_source(source)
            try:
                params = inspect.signature(interrupt_session_activity).parameters
                accepts_metadata = "metadata" in params or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                )
            except (TypeError, ValueError):
                accepts_metadata = False
            if accepts_metadata:
                await adapter.interrupt_session_activity(
                    session_key, source.chat_id, metadata=metadata
                )
            else:
                await adapter.interrupt_session_activity(session_key, source.chat_id)
        if adapter and hasattr(adapter, "get_pending_message"):
            adapter.get_pending_message(session_key)  # consume and discard
        self._pending_messages.pop(session_key, None)
        if release_running_state:
            self._release_running_agent_state(session_key)
            # Evict the cached agent: ``_interrupt_requested`` is only
            # cleared by the turn finalizer, so on a hung or still-draining
            # run the flag survives the lock release and kills the session's
            # NEXT message at the top of the tool loop (interrupted=True,
            # api_calls=0, empty response — silently swallowed, #44212).
            # Evicting mirrors the /new and /model paths: the next message
            # rebuilds the agent from session history, while the old agent
            # object keeps its interrupt flag so a hung drain still dies
            # when it unblocks.
            self._evict_cached_agent(session_key)

    async def _refresh_agent_cache_message_count(
        self, session_key: str, session_id: Optional[str]
    ) -> None:
        """Re-baseline a cached agent's stored message_count after THIS turn.

        The cross-process coherence guard (#45966) compares the session's
        on-disk ``message_count`` against the count snapshotted next to the
        cached agent, and rebuilds the agent on a mismatch.  But the snapshot
        is taken at agent-BUILD time — before this turn writes its own user +
        assistant (+ tool) rows — and the cache entry is never rewritten on a
        reuse.  So without this re-baseline, THIS process's own turn would
        grow ``message_count`` and the very next turn would see a mismatch
        and rebuild the agent — every turn, for every conversation — silently
        destroying the per-conversation prompt caching the cache exists to
        protect.

        Call this once a turn has completed and the agent has flushed its
        rows to the SessionDB.  It snapshots the now-current count (which
        includes this process's own writes) so the guard only fires when a
        DIFFERENT process changes the transcript out from under us.  The
        ``_sig`` is left untouched; only the count element is refreshed, and
        only when the same agent is still cached (no rebuild/eviction raced
        in between).  Fail-safe: any DB error leaves the snapshot as-is, which
        at worst costs one unnecessary rebuild on the next turn.

        When the cache entry records a ``session_id`` (4-tuple form, #54947)
        that differs from the current ``session_id`` — meaning the cache
        was built for a DIFFERENT conversation under the same ``session_key``
        — the snapshot is intentionally left untouched.  Overwriting it with
        the current session's count would corrupt the original conversation's
        baseline and cause the next switch back to fire the cross-process
        guard spuriously.  Fail-safe: the legacy 3-tuple shape (no
        ``session_id``) is still re-baselined as before.
        """
        if self._session_db is None or not session_id:
            return
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if not _cache_lock or _cache is None:
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
            # Only re-baseline a live 3-tuple entry; skip pending sentinels,
            # legacy 2-tuples (they intentionally opt out of the guard), and
            # the case where the entry was evicted/rebuilt mid-turn.
            if (
                isinstance(cached, tuple)
                and len(cached) > 2
                and cached[0] is not _AGENT_PENDING_SENTINEL
            ):
                # If the snapshot was taken for a different session_id
                # (same session_key, different conversation), leave the
                # snapshot alone — the current session_id's count belongs
                # to a different DB row (#54947).
                _snapshot_sid = cached[3] if len(cached) > 3 else None
                if _snapshot_sid is not None and _snapshot_sid != session_id:
                    return
                if cached[2] != _live:
                    if _snapshot_sid is None:
                        # Legacy 3-tuple: preserve the original 3-element
                        # shape so existing entries stay compatible with
                        # callers that index ``cached[2]`` directly.
                        _cache[session_key] = (cached[0], cached[1], _live)
                    else:
                        _cache[session_key] = (
                            cached[0],
                            cached[1],
                            _live,
                            _snapshot_sid,
                        )

    def _set_pending_turn_sidecar_notes(
        self, session_key: str, notes: List[str]
    ) -> None:
        """Stage per-turn must-deliver notes for the next agent run (one-shot)."""
        if not session_key or not notes:
            return
        if not hasattr(self, "_pending_turn_sidecar_notes"):
            self._pending_turn_sidecar_notes = {}
        self._pending_turn_sidecar_notes[session_key] = list(notes)

    def _consume_pending_turn_sidecar_notes(self, session_key: str) -> List[str]:
        if not session_key:
            return []
        notes = getattr(self, "_pending_turn_sidecar_notes", None)
        if not isinstance(notes, dict):
            return []
        staged = notes.pop(session_key, None)
        return list(staged) if isinstance(staged, list) else []

    def _voice_channel_sidecar_note(
        self, event, source: SessionSource, session_key: str
    ) -> Optional[str]:
        """Return a ``[Voice channel now: ...]`` note when VC state changed.

        Compares the live Discord voice-channel context against the last
        value delivered for this session and returns a note only on change
        (including leaving the channel).  Unchanged state returns ``None`` so
        the per-turn member/speaking serialization cannot churn the prompt.
        """
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
        if not hasattr(self, "_session_vc_last"):
            self._session_vc_last = {}
        vc_prev = self._session_vc_last.get(session_key) if session_key else None
        if session_key:
            self._session_vc_last[session_key] = vc_now
        if vc_now == (vc_prev if vc_prev is not None else ""):
            return None
        if not vc_now:
            return "[Voice channel now: not connected to a voice channel]"
        return f"[Voice channel now: {vc_now}]"

    def _pinned_session_context_prompt(
        self, context, redact_pii: bool, session_key: Optional[str]
    ) -> str:
        """Return the session-context prompt, pinned per session.

        Key hit → the pinned bytes are reused VERBATIM (immunizes the
        composed system prompt against renderer nondeterminism); key miss →
        re-render ``build_session_context_prompt`` and re-pin (a legitimate
        cache bust: rename, topic edit, /sethome, redact_pii flip, ...).
        """
        if not hasattr(self, "_session_ephemeral_pin"):
            self._session_ephemeral_pin = {}
        _eph_key = self._ephemeral_change_key(context, redact_pii)
        _eph_pin = self._session_ephemeral_pin.get(session_key) if session_key else None
        if _eph_pin is not None and _eph_pin[0] == _eph_key:
            return _eph_pin[1]
        text = build_session_context_prompt(context, redact_pii=redact_pii)
        if session_key:
            self._session_ephemeral_pin[session_key] = (_eph_key, text)
        return text

    @staticmethod
    def _ephemeral_change_key(context, redact_pii: bool) -> str:
        """Hash the exact inputs ``build_session_context_prompt`` renders.

        This key decides when the pinned per-session context-prompt bytes are
        reused verbatim vs re-rendered.  The maintained invariant (guarded by
        the parity test in tests/gateway/test_prompt_tail_freeze.py): any
        input whose change alters the rendered bytes MUST appear here —
        omission means a stale pinned prompt (cosmetic staleness); inclusion
        of an extra field only costs a spurious re-render.
        """
        import hashlib

        src = context.source
        platform = src.platform.value if src.platform else ""

        discord_ids: tuple = ()
        discord_tools = ""
        if src.platform == Platform.DISCORD:
            from gateway.session import _discord_tools_loaded

            discord_tools = "1" if _discord_tools_loaded() else "0"
            discord_ids = (
                str(src.guild_id or ""),
                str(src.parent_chat_id or ""),
                str(src.thread_id or ""),
                str(src.chat_id or ""),
                # Only PRESENCE is rendered (the id itself is delivered
                # per-turn in the user message) — keying on the value would
                # re-render every message for zero byte change.
                "1" if src.message_id else "0",
            )

        try:
            from hermes_constants import display_hermes_home

            home_display = str(display_hermes_home())
        except Exception:
            home_display = ""

        key_tuple = (
            platform,
            str(src.chat_id or ""),
            str(src.thread_id or ""),
            str(src.chat_type or ""),
            str(src.chat_name or ""),
            str(src.chat_topic or ""),
            str(src.user_name or ""),
            str(src.user_id or ""),
            str(getattr(src, "profile", None) or ""),
            bool(context.shared_multi_user_session),
            discord_ids,
            discord_tools,
            tuple(p.value for p in context.connected_platforms),
            tuple(
                (
                    p.value,
                    str(getattr(hc, "name", "") or ""),
                    str(getattr(hc, "chat_id", "") or ""),
                )
                for p, hc in context.home_channels.items()
            ),
            bool(redact_pii),
            home_display,
        )
        return hashlib.sha256(repr(key_tuple).encode("utf-8")).hexdigest()

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent for a session (called on /new, /model, etc).

        Pops the entry AND soft-releases the evicted agent's LLM client
        pool so the httpx connection (sockets + held buffers) is freed
        promptly rather than waiting on CPython GC — AIAgent holds
        reference cycles (callbacks, tool state) that delay refcount
        collection, so a manual release is required to keep gateway RSS
        flat across many /new, /model, undo and reset operations (#29298,
        same leak class as #25315).

        The release is soft (``release_clients()``): it frees the client
        pool and per-turn child subagents but PRESERVES the session's
        terminal sandbox, browser daemon, and tracked bg processes (keyed
        on task_id), because the session may resume with a freshly-built
        agent.  Call sites that want a hard teardown (true conversation
        boundaries like /new) already call ``_cleanup_agent_resources``
        before evicting; ``release_clients`` is idempotent and safe to
        run again after that (the client is already None).

        Cleanup runs on a daemon thread so we never block holding
        ``_agent_cache_lock`` on slow socket teardown — mirrors the
        cap-enforcer and idle-sweeper paths.
        """
        # Prompt-stability state rides the agent-cache lifecycle: a fresh
        # agent must re-render its session-context bytes (the pin) and re-see
        # the current voice-channel state once.
        _pin_store = getattr(self, "_session_ephemeral_pin", None)
        if isinstance(_pin_store, dict):
            _pin_store.pop(session_key, None)
        _vc_store = getattr(self, "_session_vc_last", None)
        if isinstance(_vc_store, dict):
            _vc_store.pop(session_key, None)

        _lock = getattr(self, "_agent_cache_lock", None)
        evicted = None
        if _lock:
            with _lock:
                evicted = self._agent_cache.pop(session_key, None)
        else:
            _cache = getattr(self, "_agent_cache", None)
            if _cache is not None:
                evicted = _cache.pop(session_key, None)

        agent = evicted[0] if isinstance(evicted, tuple) and evicted else evicted
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return

        # Don't tear down an agent that's actively mid-turn — its client,
        # sandbox and child subagents are in use by the running request.
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }
        if id(agent) in running_ids:
            return

        try:
            threading.Thread(
                target=self._release_evicted_agent_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-evict-{str(session_key)[:24]}",
            ).start()
        except Exception:
            # If we can't spawn a thread (interpreter shutdown), release
            # inline as a best-effort fallback.
            try:
                self._release_evicted_agent_soft(agent)
            except Exception:
                pass

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.

        Both _last_activity_ts and _last_activity_desc are only reset for
        fresh external turns (depth 0); they are semantically paired —
        desc describes the activity *at* ts, so updating one without the
        other would make get_activity_summary() misleading.
        For interrupt-recursive turns both are preserved so the inactivity
        watchdog can accumulate stuck-turn idle time and fire the 30-min
        timeout (#15654).  The depth-0 reset is still needed: a session
        idle for 29 min would otherwise trip the watchdog before the new
        turn makes its first API call (#9051).
        """
        if interrupt_depth == 0:
            agent._last_activity_ts = _gateway_runtime().time.time()
            agent._last_activity_desc = "starting new turn (cached)"
            # Reset the SessionDB flush cursor so the new turn's messages are
            # fully persisted — a stale value from the previous turn would
            # cause `_flush_messages_to_session_db` to skip new rows (#44327).
            if hasattr(agent, "_last_flushed_db_idx"):
                agent._last_flushed_db_idx = 0
        agent._api_call_count = 0

    def _commit_memory_before_soft_evict(self, agent: Any, key: str) -> None:
        """Fire on_session_end extraction before soft-evicting a live agent.

        Soft eviction (``_release_evicted_agent_soft``) deliberately keeps the
        session resumable and does NOT fire ``on_session_end`` — that hook is
        reserved for the true session boundary, tear-down done by
        ``_session_expiry_watcher`` when the session finally expires.

        But the watcher tears down whatever agent it finds in ``_agent_cache``
        at expiry time.  If cache pressure (the LRU cap) soft-evicts a
        finalizable session's agent BEFORE it expires, the watcher later finds
        no cached agent and ``on_session_end`` is silently skipped — memory
        providers never see the transcript (#11205, LRU-cap variant).

        We hold the live, fully-scoped agent right now, so commit its
        end-of-session memory extraction here using the agent's own memory
        manager (correct per-user/chat scoping, no reconstruction).  This uses
        ``commit_memory_session`` — extraction WITHOUT provider teardown — so
        the eviction stays soft and a resumed turn keeps working.

        Only fires for sessions the expiry watcher will eventually finalize
        (finite reset policy).  For ``mode == "none"`` sessions the watcher
        never runs, so there is no missed-boundary to compensate for and we
        skip the commit (the agent is simply released).  Best-effort: any
        failure is swallowed so eviction still proceeds.
        """
        if agent is None or not hasattr(agent, "commit_memory_session"):
            return
        if getattr(agent, "_memory_manager", None) is None:
            return  # no external memory provider — nothing to commit
        try:
            _store = getattr(self, "session_store", None)
            if _store is None:
                return
            _store._ensure_loaded()
            entry = _store._entries.get(key)
            if entry is None:
                return
            # Only compensate when the watcher would otherwise expect to find
            # this agent at expiry (finite policy, not yet expired). Expired
            # sessions are torn down by the watcher directly; mode="none"
            # sessions are never finalized.
            if not _store.is_session_finalizable(entry):
                return
            if _store._is_session_expired(entry):
                return
            messages = getattr(agent, "_session_messages", None)
            agent.commit_memory_session(
                messages if isinstance(messages, list) else None
            )
            logger.debug(
                "Committed on_session_end extraction before soft-evicting "
                "finalizable session=%s (cache pressure, pre-expiry)",
                key,
            )
        except Exception as _e:
            logger.debug("Pre-evict memory commit failed for %s: %s", key, _e)

    def _commit_then_release_soft(self, agent: Any, key: str) -> None:
        """Commit end-of-session memory (if warranted), then soft-release.

        Runs on the daemon eviction thread so the memory-provider call and the
        client teardown never block the caller's held cache lock. Order matters:
        commit uses the live agent's memory manager before ``release_clients``
        drops the message buffer.
        """
        self._commit_memory_before_soft_evict(agent, key)
        self._release_evicted_agent_soft(agent)

    def _release_evicted_agent_soft(self, agent: Any) -> None:
        """Soft cleanup for cache-evicted agents — preserves session tool state.

        Called from _enforce_agent_cache_cap and _sweep_idle_cached_agents.
        Distinct from _cleanup_agent_resources (full teardown) because a
        cache-evicted session may resume at any time — its terminal
        sandbox, browser daemon, and tracked bg processes must outlive
        the Python AIAgent instance so the next agent built for the
        same task_id inherits them.
        """
        if agent is None:
            return
        try:
            if hasattr(agent, "release_clients"):
                agent.release_clients()
            else:
                # Older agent instance (shouldn't happen in practice) —
                # fall back to the legacy full-close path.
                self._cleanup_agent_resources(agent)
        except Exception:
            pass
        # Free conversation history memory — can be tens of MB with tool
        # outputs (file reads, terminal output, search results) on heavy
        # 100+-tool-call sessions. release_clients() deliberately preserves
        # session tool state for resume, but the message list is rebuilt from
        # persisted session JSON on the next turn, so dropping it here is safe.
        if hasattr(agent, "_session_messages"):
            agent._session_messages = []

    def _enforce_agent_cache_cap(self) -> None:
        """Evict oldest cached agents when cache exceeds _AGENT_CACHE_MAX_SIZE.

        Must be called with _agent_cache_lock held.  Resource cleanup
        (memory provider shutdown, tool resource close) is scheduled
        on a daemon thread so the caller doesn't block on slow teardown
        while holding the cache lock.

        Agents currently in _running_agents are SKIPPED — their clients,
        terminal sandboxes, background processes, and child subagents
        are all in active use by the running turn.  Evicting them would
        tear down those resources mid-turn and crash the request.  If
        every candidate in the LRU order is active, we simply leave the
        cache over the cap; it will be re-checked on the next insert.
        """
        _cache = getattr(self, "_agent_cache", None)
        if _cache is None:
            return
        # OrderedDict.popitem(last=False) pops oldest; plain dict lacks the
        # arg so skip enforcement if a test fixture swapped the cache type.
        if not hasattr(_cache, "move_to_end"):
            return

        # Snapshot of agent instances that are actively mid-turn.  Use id()
        # so the lookup is O(1) and doesn't depend on AIAgent.__eq__ (which
        # MagicMock overrides in tests).
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }

        # Walk LRU → MRU and evict excess-LRU entries that aren't mid-turn.
        # We only consider entries in the first (size - cap) LRU positions
        # as eviction candidates.  If one of those slots is held by an
        # active agent, we SKIP it without compensating by evicting a
        # newer entry — that would penalise a freshly-inserted session
        # (which has no cache history to retain) while protecting an
        # already-cached long-running one.  The cache may therefore stay
        # temporarily over cap; it will re-check on the next insert,
        # after active turns have finished.
        cache_max_size = _gateway_runtime()._AGENT_CACHE_MAX_SIZE
        excess = max(0, len(_cache) - cache_max_size)
        evict_plan: List[tuple] = []  # [(key, agent), ...]
        if excess > 0:
            ordered_keys = list(_cache.keys())
            for key in ordered_keys[:excess]:
                entry = _cache.get(key)
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is not None and id(agent) in running_ids:
                    continue  # active mid-turn; don't evict, don't substitute
                evict_plan.append((key, agent))

        for key, _ in evict_plan:
            _cache.pop(key, None)

        remaining_over_cap = len(_cache) - cache_max_size
        if remaining_over_cap > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(_cache),
                cache_max_size,
                remaining_over_cap,
            )

        for key, agent in evict_plan:
            logger.info(
                "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
                key,
                len(_cache),
            )
            if agent is not None:
                # Commit end-of-session memory extraction, then soft-release,
                # both on the daemon thread so the (possibly network-bound)
                # provider call never blocks the held cache lock. The commit
                # only fires for finalizable-not-yet-expired sessions whose
                # agent would otherwise vanish before the expiry watcher can
                # fire on_session_end (#11205, LRU-cap variant).
                threading.Thread(
                    target=self._commit_then_release_soft,
                    args=(agent, key),
                    daemon=True,
                    name=f"agent-cache-evict-{key[:24]}",
                ).start()

    def _sweep_idle_cached_agents(self) -> int:
        """Evict cached agents whose AIAgent has been idle > _AGENT_CACHE_IDLE_TTL_SECS.

        Safe to call from the session expiry watcher without holding the
        cache lock — acquires it internally.  Returns the number of entries
        evicted.  Resource cleanup is scheduled on daemon threads.

        Agents currently in _running_agents are SKIPPED for the same reason
        as _enforce_agent_cache_cap: tearing down an active turn's clients
        mid-flight would crash the request.
        """
        _cache = getattr(self, "_agent_cache", None)
        _lock = getattr(self, "_agent_cache_lock", None)
        if _cache is None or _lock is None:
            return 0
        runtime = _gateway_runtime()
        now = runtime.time.time()
        idle_ttl_secs = runtime._AGENT_CACHE_IDLE_TTL_SECS
        to_evict: List[tuple] = []
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }
        with _lock:
            for key, entry in list(_cache.items()):
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is None:
                    continue
                if id(agent) in running_ids:
                    continue  # mid-turn — don't tear it down
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None:
                    continue
                if (now - last_activity) > idle_ttl_secs:
                    # Check whether the session has actually expired in the
                    # session store.  If it hasn't (e.g. daily-reset mode
                    # where the reset fires hours after the user's last
                    # message), keep the agent in cache so the session-store
                    # expiry watcher can still find it and call
                    # on_session_end() with the live transcript.  Skipping
                    # eviction here means the agent stays alive until the
                    # session genuinely expires, at which point the watcher
                    # (gateway/run.py _session_expiry_watcher) tears it down
                    # properly.  (#11205 follow-up)
                    #
                    # BUT only defer when the watcher will EVER finalize this
                    # session.  For a mode == "none" session the watcher never
                    # fires (is_session_finalizable() is False), so deferring
                    # would pin the agent in cache for the gateway's entire
                    # lifetime — the exact leak this idle sweep exists to
                    # relieve.  Those sessions fall through to soft eviction
                    # WITHOUT on_session_end, and that is correct: a mode=="none"
                    # session never reaches a session-end boundary, so there is
                    # no missed on_session_end to compensate for.  (The finite
                    # case — a session evicted under LRU-cap pressure before it
                    # expires — is instead covered by _commit_memory_before_soft_
                    # evict on the cap path, which fires on_session_end via the
                    # live agent's memory manager before releasing it.)
                    session_entry = None
                    _store = getattr(self, "session_store", None)
                    try:
                        if _store is not None:
                            _store._ensure_loaded()
                            session_entry = _store._entries.get(key)
                    except Exception:
                        session_entry = None
                    if (
                        session_entry is not None
                        and _store is not None
                        and _store.is_session_finalizable(session_entry)
                        and not _store._is_session_expired(session_entry)
                    ):
                        continue  # keep agent — finite session hasn't expired
                    to_evict.append((key, agent))
            for key, _ in to_evict:
                _cache.pop(key, None)
        for key, agent in to_evict:
            logger.info(
                "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
                key,
                now - getattr(agent, "_last_activity_ts", now),
            )
            threading.Thread(
                target=self._release_evicted_agent_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-idle-{key[:24]}",
            ).start()
        return len(to_evict)
