"""Owner-loop, local-only maintenance of an already-bound gateway session.

No adapter delivery, session creation, or policy/scheduler lives here. Callers supply the
exact routing key and session id as a compare-and-swap expectation, plus a threshold.
Threaded Telegram DMs require their topic binding and route to share one SQLite file;
cross-file topic maintenance fails closed. The checks around a provider call are not a
transaction over the external provider: a concurrent /topic restore immediately after
commit admission cannot be rolled back. The post-call check prevents clearing the new
binding's usage, not an already accepted provider compaction.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
from typing import Any

from agent.conversation_compression import CompressionCommitFence, finalize_context_engine_compression_notification
from agent.conversation_compression_manual import MIN_MESSAGES, CompressRequest, compress_now
from gateway.config import Platform
from gateway.session import build_session_key
from gateway.session_transcript import TranscriptReadError
from gateway.turn_lease import TurnLeaseTimeoutError


class _BindingFence(CompressionCommitFence):
    """Refuse commit if the route or Telegram topic binding changed while summarizing."""

    def __init__(self, store, key: str, session_id: str, *, profile=None, topic_db=None):
        super().__init__()
        self.store, self.key, self.session_id = store, key, session_id
        self.profile, self.topic_db = profile, topic_db

    def begin_commit(self, cancel_event=None):
        # The compressor calls this on its worker thread, not on the gateway loop.
        if not _bound(self.store, self.key, self.session_id, self.profile, self.topic_db):
            self.revoke_commit_admission()
        return super().begin_commit(cancel_event)


def _topic_bound(entry, profile, topic_db) -> bool:
    source = entry.origin
    if source is None:
        return False
    if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
        return True
    if topic_db is None or not source.chat_id or not source.user_id:
        return False
    try:
        mode = topic_db.is_telegram_topic_mode_enabled(
            chat_id=source.chat_id, user_id=source.user_id, profile_name=profile)
        if not source.thread_id:
            return not mode
        # Threaded DMs are ambiguous without their authoritative topic row: a
        # deleted last binding also disables mode, leaving the route unchanged.
        if not mode:
            return False
        binding = topic_db.get_telegram_topic_binding(
            chat_id=source.chat_id, thread_id=source.thread_id, profile_name=profile)
        return bool(binding and binding["profile_name"] == profile
                    and binding["user_id"] == str(source.user_id)
                    and binding["session_key"] == entry.session_key
                    and binding["session_id"] == entry.session_id)
    except Exception:
        return False


def _bound(store, key: str, session_id: str, profile=None, topic_db=None) -> bool:
    entry = store.lookup_by_session_key(key)
    return bool(entry and entry.session_id == session_id and not entry.active_turn_token
                and not entry.suspended and not entry.resume_pending
                and _topic_bound(entry, profile, topic_db))


def _valid_number(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


async def maintain_existing_session(runner: Any, params: dict) -> dict:
    """Inspect or compact ONE exact bound session on the gateway owner loop.

    The threshold is a caller policy, not a gateway default. A timeout from the socket
    means pending: it does not cancel this coroutine or release its turn lease.
    """
    if not isinstance(params, dict) or set(params) != {"action", "profile", "session_key", "session_id", "min_percent"}:
        return {"status": "invalid_request"}
    action, profile = params["action"], params["profile"]
    key, sid, threshold = params["session_key"], params["session_id"], params["min_percent"]
    if (action not in ("inspect", "compact") or not all(isinstance(v, str) and v.strip()
            for v in (profile, key, sid)) or not _valid_number(threshold) or not 0 < threshold <= 100):
        return {"status": "invalid_request"}
    # The owner declares served homes; do not resolve an arbitrary caller-supplied profile
    # via a filesystem path, or let an unscoped read fall back to the launch profile.
    homes = getattr(runner, "_served_profile_homes", None) or {}
    home = homes.get(profile)
    if (home is None and not homes and not runner.config.multiplex_profiles
            and profile == getattr(runner, "_primary_profile_name", None)):
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    if home is None:
        return {"status": "profile_not_served"}
    from gateway.run import _async_profile_runtime_scope
    try:
        async with _async_profile_runtime_scope(home):
            return await _maintain_scoped(runner, action, profile, key, sid, threshold)
    except Exception:
        # Provider exceptions may embed authorization headers or prompt text.
        return {"status": "error"}


async def _maintain_scoped(runner, action, profile, key, sid, threshold):
    store = runner.async_session_store
    sync_store = store._store
    topic_db = getattr(runner._session_db, "_db", None)

    async def current():
        entry = await store.lookup_by_session_key(key)  # never get_or_create_session
        if not entry or entry.session_id != sid or not entry.origin or entry.suspended or entry.resume_pending:
            return None
        # Validate the exact routing identity, including the namespace. A key supplied by
        # the caller is an expectation, never an instruction to create or switch a lane.
        source = entry.origin
        expected = build_session_key(
            source, group_sessions_per_user=runner.config.group_sessions_per_user,
            thread_sessions_per_user=runner.config.thread_sessions_per_user,
            profile=profile if runner.config.multiplex_profiles else None)
        if entry.session_key != key or key != expected:
            return None
        # A satellite profile can have its topic rows in a different state.db
        # than the process-wide routing index. No cross-file commit can fence
        # both, so reject threaded topic maintenance rather than guessing.
        if source.platform == Platform.TELEGRAM and source.chat_type == "dm" and source.thread_id:
            routing_db = sync_store._routing_db
            if (routing_db is None or topic_db is None or
                    routing_db.db_path != topic_db.db_path):
                return None
        return entry if await runner._run_in_executor_with_context(
            lambda: _topic_bound(entry, profile, topic_db)) else None

    entry = await current()
    if entry is None:
        return {"status": "stale_binding"}
    if runner._is_session_running(key) or entry.active_turn_token:
        return {"status": "busy"}
    try:
        token = await runner._turn_leases.acquire(sid, owner_key=key, generation=0, timeout=0.05)
    except TurnLeaseTimeoutError:
        return {"status": "busy"}
    if token is None:
        return {"status": "busy"}
    try:
        entry = await current()
        if entry is None:
            return {"status": "stale_binding"}
        if runner._is_session_running(key) or entry.active_turn_token:
            return {"status": "busy"}
        agent = runner._cached_agent_for(key)
        if agent is not None and getattr(agent, "session_id", None) != sid:
            agent = None
        ctx = getattr(agent, "context_compressor", None)
        used = entry.last_prompt_tokens  # last real turn, not transcript estimate or cumulative tokens
        if not isinstance(used, int) or isinstance(used, bool) or used <= 0:
            return {"status": "unknown_usage"}
        live_used = getattr(ctx, "last_prompt_tokens", 0)
        if _valid_number(live_used) and live_used > 0 and live_used != used:
            return {"status": "unknown_usage"}
        # /context's display fallback can use the configured model after a /model switch;
        # maintenance requires a positive window of the actual persisted session model.
        session_row = await runner._session_db.get_session(sid)
        model = session_row.get("model") if isinstance(session_row, dict) else None
        if not isinstance(model, str) or not model:
            return {"status": "unknown_usage"}
        if agent is not None and getattr(agent, "model", None) != model:
            return {"status": "unknown_usage"}
        # A metadata lookup may silently substitute a 256k default for an unknown
        # model. Require the matched live agent's actual compressor window instead.
        window = getattr(ctx, "context_length", 0) if ctx is not None else 0
        if not _valid_number(window) or window <= 0:
            return {"status": "unknown_usage"}
        percent = 100 * used / window
        figures = {"used_tokens": used, "context_window": window, "percent": round(percent, 2)}
        if percent <= threshold:
            return {"status": "below_threshold", **figures}
        if action == "inspect":
            return {"status": "above_threshold", **figures}
        if await current() is None or runner._is_session_running(key):
            return {"status": "stale_binding", **figures}
        source = entry.origin
        model, runtime = runner._resolve_session_agent_runtime(source=source, session_key=key)
        if model != session_row["model"]:
            return {"status": "unknown_usage", **figures}
        if agent is not None and getattr(agent, "_codex_session", None) is not None and str(runtime.get("api_mode") or "").lower() != "codex_app_server":
            return {"status": "unknown_usage", **figures}
        if str(runtime.get("api_mode") or "").lower() == "codex_app_server":
            if agent is None or getattr(agent, "_codex_session", None) is None or runner._cached_agent_for(key) is not agent:
                return {"status": "no_live_thread", **figures}
            count = getattr(ctx, "compression_count", None)
            if not isinstance(count, int) or isinstance(count, bool):
                return {"status": "unknown_usage"}
            fence = _BindingFence(sync_store, key, sid, profile=profile, topic_db=topic_db)
            await runner._run_in_executor_with_context(
                lambda: agent._compress_context([], "", force=True, task_id=sid,
                                                commit_fence=fence))
            if getattr(ctx, "compression_count", count) <= count:
                return {"status": "not_compacted", **figures}
            ctx.last_prompt_tokens = -1
            ctx.awaiting_real_usage_after_compression = True
        else:
            if not runtime.get("api_key"):
                return {"status": "provider_unavailable", **figures}
            try:
                history = await store.load_transcript(sid)
            except TranscriptReadError:
                return {"status": "history_unreadable", **figures}
            if len(history) < MIN_MESSAGES:
                return {"status": "not_enough_messages", **figures}
            # Reuse the manual compressor but force in-place; never rotate a binding or
            # rewrite the archived transcript. The compressor owns the durable row commit.
            from gateway.run import _platform_config_key
            runtime["platform"] = _platform_config_key(source.platform)
            runtime["gateway_session_key"] = key
            runtime["reasoning_config"] = runner._resolve_session_reasoning_config(source=source, model=model)
            tmp = await runner._build_manual_compression_agent(sid, model, runtime)
            try:
                tmp.compression_in_place = True
                messages = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
                fence = _BindingFence(sync_store, key, sid, profile=profile, topic_db=topic_db)
                result = await runner._run_in_executor_with_context(
                    lambda: compress_now(tmp, messages, CompressRequest(), system_message="",
                                         task_id=sid, skip_without_window=True, commit_fence=fence))
                if (result.status != "compressed" or tmp.session_id != sid
                        or tmp._last_compaction_in_place is not True):
                    return {"status": "not_compacted", **figures}
                finalize_context_engine_compression_notification(tmp, committed=True)
            finally:
                finalize_context_engine_compression_notification(tmp, committed=False)
                await runner._cleanup_agent_resources_off_loop(tmp, context="session maintenance")
            # The in-place flag is necessary but not sufficient: verify the durable
            # active transcript changed before clearing the previous prompt usage.
            after = await store.load_transcript(sid)
            if not after or after == history:
                return {"status": "readback_failed", **figures}
        if await current() is None or runner._is_session_running(key):
            return {"status": "stale_binding", **figures}
        if not await store.clear_prompt_usage_if_bound(key, sid):
            return {"status": "stale_binding", **figures}
        if await current() is None or (await store.lookup_by_session_key(key)).last_prompt_tokens != 0:
            return {"status": "readback_failed", **figures}
        if str(runtime.get("api_mode") or "").lower() != "codex_app_server":
            runner._evict_cached_agent(key)
        return {"status": "compacted", **figures}
    finally:
        runner._turn_leases.release(token)


def session_maintenance_verb(runner, loop):
    """Control socket executor -> owner loop. Timeout never cancels an in-flight commit."""
    def handler(params: dict) -> dict:
        future = asyncio.run_coroutine_threadsafe(maintain_existing_session(runner, params), loop)
        try:
            return future.result(timeout=120)
        except concurrent.futures.TimeoutError:
            return {"status": "pending"}
        except Exception:
            return {"status": "error"}
    return handler
