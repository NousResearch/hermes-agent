"""Durable, silent post-reply transcript compression for gateway sessions."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
import logging
import time
from uuid import uuid4

logger = logging.getLogger("gateway.run")

_RETRY_SECONDS = 300
_SKIP_SECONDS = 3600
_LEASE_SECONDS = 300
_SUMMARY_TIMEOUT = 180


class GatewayPostReplyIdleWorkerMixin:
    async def _post_reply_idle_watcher(self, interval: float = 15.0) -> None:
        """Poll durable jobs on startup; two concurrent scans bound model work."""
        active: set[asyncio.Task] = set()

        async def scan():
            try:
                await self._process_due_post_reply_idle()
            except Exception:
                logger.warning("Post-reply idle watcher tick failed", exc_info=True)

        try:
            while self._running:
                if len(active) < 2:
                    task = asyncio.create_task(scan())
                    active.add(task)
                    task.add_done_callback(active.discard)
                remaining = interval
                while self._running and remaining > 0:
                    tick = min(1.0, remaining)
                    await asyncio.sleep(tick)
                    remaining -= tick
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

    async def _process_due_post_reply_idle(self) -> None:
        """Take at most one claim per served profile per tick; SQLite leases survive restarts."""
        from gateway.run import _resolve_handoff_watch_scopes, _async_profile_runtime_scope, _multiplex_profile_homes

        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            # Unlike handoff's root poll, compression must bind even the launch
            # profile's home and secret scope before resolving a model route.
            offload = getattr(self, "_run_in_executor_with_context", asyncio.to_thread)
            scopes = await offload(_multiplex_profile_homes, self.config)
        else:
            scopes = await _resolve_handoff_watch_scopes(self)
        for _name, home in scopes:
            if not self._running:
                return
            try:
                if home is None:
                    await self._post_reply_idle_claim_one()
                else:
                    async with _async_profile_runtime_scope(home):
                        await self._post_reply_idle_claim_one()
            except Exception:
                logger.warning("Post-reply idle profile scan failed (%s)", home, exc_info=True)

    async def _post_reply_idle_renew(self, db, sid, generation, holder) -> None:
        """Refresh the durable claim throughout setup and the provider request."""
        while True:
            await asyncio.sleep(max(0.01, _LEASE_SECONDS / 3))
            try:
                still_owned = await asyncio.to_thread(
                    db.renew_post_reply_idle, sid, generation, holder, lease_ttl=_LEASE_SECONDS)
            except Exception:
                logger.warning("Post-reply idle claim renewal failed for %s", sid, exc_info=True)
                continue  # An expired lease will fail closed at the compaction commit.
            if not still_owned:
                return

    async def _post_reply_idle_claim_one(self) -> None:
        from gateway.run import _load_gateway_config
        from gateway.post_reply_idle_policy import validate_post_reply_idle_policy
        config = _load_gateway_config()
        try:
            if config.get("compression", {}).get("enabled", True) is False or not validate_post_reply_idle_policy(config):
                return
        except ValueError as exc:
            logger.warning("Invalid post-reply idle policy; disabling it: %s", exc)
            return
        session_db = self._session_db
        db = getattr(session_db, "_db", session_db)
        if db is None or not callable(getattr(db, "claim_due_post_reply_idle", None)):
            return
        holder = uuid4().hex
        claim = await asyncio.to_thread(db.claim_due_post_reply_idle, holder, lease_ttl=_LEASE_SECONDS)
        if claim is None:
            return
        sid, generation, watermark = claim
        retry = _RETRY_SECONDS
        renewal = asyncio.create_task(self._post_reply_idle_renew(db, sid, generation, holder))
        try:
            retry = await self._post_reply_idle_run_claim(db, sid, generation, watermark, holder)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Post-reply idle compression failed for %s", sid, exc_info=True)
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
            # Successful in-place publication clears the claim atomically; release is then a no-op.
            await asyncio.to_thread(db.release_post_reply_idle, sid, generation, holder,
                                    retry_at=time.time() + retry)

    def _post_reply_idle_has_pending_input(self, key: str) -> bool:
        """Check unadmitted adapter buffers as well as runner-side turns."""
        if self._is_session_running(key) or key in (getattr(self, "_queued_events", None) or {}):
            return True
        adapters = (getattr(self, "adapters", {}), *getattr(self, "_profile_adapters", {}).values())
        return any(
            key in getattr(adapter, name, ())
            for group in adapters for adapter in tuple(group.values())
            for name in ("_active_sessions", "_pending_messages", "_pending_text_batches")
        )

    async def _post_reply_idle_run_claim(self, db, sid, generation, watermark, holder) -> float:
        row = await asyncio.to_thread(db.get_session, sid)
        if not row or row.get("ended_at") is not None:
            return _SKIP_SECONDS
        stored_prompt = row.get("system_prompt")
        if not isinstance(stored_prompt, str) or not stored_prompt.strip():
            logger.warning("Post-reply idle session %s has no restorable prompt; retrying", sid)
            return _RETRY_SECONDS
        key = row.get("session_key")
        if not key:
            return _SKIP_SECONDS
        entry = await asyncio.to_thread(self.session_store.lookup_by_session_key, key)
        if entry is None or entry.session_id != sid:
            return _SKIP_SECONDS
        if (self._post_reply_idle_has_pending_input(key)
                or await self._session_has_compression_in_flight(key)):
            return _RETRY_SECONDS
        history = await asyncio.to_thread(db.get_messages_as_conversation, sid, repair_alternation=True)
        if (len(history) < 8 or not any(m.get("role") == "user" for m in history)
                or sum(len(str(m.get("content") or "")) for m in history) < 1000):
            return _SKIP_SECONDS

        from agent.conversation_compression import CompressionCommitFence
        from gateway.run import _load_gateway_config
        from gateway.post_reply_idle_policy import resolve_post_reply_idle_rule, validate_post_reply_idle_policy
        from gateway.session_identity import canonical_identity, identity_of
        source = self._restored_source(entry)
        if source is None:
            return _SKIP_SECONDS
        config = _load_gateway_config()
        try:
            if not validate_post_reply_idle_policy(config):
                return _SKIP_SECONDS
        except ValueError as exc:
            logger.warning("Invalid post-reply idle policy; disabling it: %s", exc)
            return _SKIP_SECONDS
        identity = identity_of(source)
        if identity is None and not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            identity = canonical_identity(source, runner=self)
        try:
            rule = resolve_post_reply_idle_rule(config, source, identity)
            if rule is None:
                return _SKIP_SECONDS
        except ValueError as exc:
            logger.warning("Invalid post-reply idle policy; disabling it: %s", exc)
            return _SKIP_SECONDS
        from agent.model_metadata import estimate_messages_tokens_rough
        approx_tokens = estimate_messages_tokens_rough(history)
        if approx_tokens < rule.get("min_tokens", 0):
            return _SKIP_SECONDS
        model, runtime = self._resolve_session_agent_runtime(source=source, session_key=key, user_config=config)
        if runtime.get("api_mode") == "codex_app_server":
            # Codex owns a server-side thread; rewriting its DB mirror loses the live context.
            return _SKIP_SECONDS
        agent, sync_db = await self._hmwa_hygiene_build_agent(model, runtime, entry)
        agent._end_session_on_close = False  # Never close the live session on any cleanup path.
        if getattr(agent, "_cached_system_prompt", None) != stored_prompt:
            # Hygiene's fallback seed is empty when its second prompt read fails. Never
            # let that reduced-toolset agent overwrite the live pinned prompt.
            await self._cleanup_agent_resources_off_loop(agent, context="post-reply idle", session_key=key)
            return _RETRY_SECONDS
        agent.compression_in_place = True
        agent.context_compressor.abort_on_summary_failure = True
        compress_window = getattr(agent.context_compressor, "_compress_window", None)
        if callable(compress_window):
            start, end = compress_window(history)
            if start >= end:
                await self._cleanup_agent_resources_off_loop(agent, context="post-reply idle", session_key=key)
                return _SKIP_SECONDS
        agent._post_reply_idle_claim = (
            generation, watermark, holder, lambda: self._post_reply_idle_has_pending_input(key)
        )
        binder = getattr(getattr(agent, "context_compressor", None), "bind_session_state", None)
        if callable(binder):
            binder(sync_db, sid)
        agent._print_fn = lambda *args, **kwargs: None
        # Evict before publication is possible: a turn admitted immediately after commit
        # must never reuse the old cached system prompt or transcript.
        self._evict_cached_agent(key)
        fence = CompressionCommitFence(total_ceiling_seconds=_SUMMARY_TIMEOUT)
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, copy_context().run,
            lambda: agent._compress_context(history, "", approx_tokens=approx_tokens,
                                            commit_fence=fence, task_id=sid),
        )
        self._track_deferred_agent_worker(future, agent)
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=_SUMMARY_TIMEOUT)
            if not getattr(agent, "_last_compression_attempt_in_place", False):
                logger.info("Post-reply idle compression did not commit for %s", sid)
                return (_RETRY_SECONDS if getattr(getattr(agent, "context_compressor", None), "_last_summary_error", None)
                        or getattr(agent, "_compression_blocked_transient", None) else _SKIP_SECONDS)
            return _SKIP_SECONDS
        except asyncio.CancelledError:
            fence.try_cancel_before_commit()
            raise
        except asyncio.TimeoutError:
            fence.try_cancel_before_commit()
            logger.warning("Post-reply idle compression timed out for %s", sid)
            return _RETRY_SECONDS
        finally:
            if future.done():
                await self._cleanup_agent_resources_off_loop(agent, context="post-reply idle", session_key=key)
            else:
                async def _late_cleanup():
                    try:
                        await asyncio.shield(future)
                    except Exception:
                        pass
                    await self._cleanup_agent_resources_off_loop(agent, context="post-reply idle", session_key=key)
                cleanup_task = asyncio.create_task(_late_cleanup())
                # Shutdown must wait for resource cleanup as well as the model future.
                self._track_deferred_agent_worker(cleanup_task, agent)
