"""Non-blocking gateway transcript compaction.

The foreground turn remains the sole owner of response delivery. Once its
transcript flush is complete, this mixin may summarize an old, bounded prefix
on the gateway executor. The expensive model call never holds a turn or
database lease; publication uses a short prefix-CAS transaction so new turns
may append safely while the summary is being generated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BackgroundCompactionSettings:
    enabled: bool = True
    threshold: float = 0.65
    chunk_tokens: int = 96_000
    use_main_model: bool = True
    protect_tail_messages: int = 20

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "BackgroundCompactionSettings":
        compression = config.get("compression", {}) if isinstance(config, dict) else {}
        if not isinstance(compression, dict):
            compression = {}
        try:
            threshold = float(compression.get("background_threshold", 0.65))
        except (TypeError, ValueError):
            threshold = 0.65
        # The foreground 85% hygiene path is the final safety net. Keep the
        # background trigger strictly below it even when config is malformed.
        threshold = min(0.84, max(0.10, threshold))
        try:
            chunk_tokens = int(compression.get("background_chunk_tokens", 96_000))
        except (TypeError, ValueError):
            chunk_tokens = 96_000
        chunk_tokens = min(192_000, max(16_000, chunk_tokens))
        try:
            protect_tail = int(compression.get("protect_last_n", 20))
        except (TypeError, ValueError):
            protect_tail = 20
        return cls(
            enabled=(
                _as_bool(compression.get("enabled"), True)
                and _as_bool(compression.get("background_enabled"), True)
            ),
            threshold=threshold,
            chunk_tokens=chunk_tokens,
            use_main_model=_as_bool(
                compression.get("background_use_main_model"), True
            ),
            protect_tail_messages=max(4, protect_tail),
        )


@dataclass(frozen=True)
class CompactionPrefix:
    messages: List[Dict[str, Any]]
    last_state_message_id: int
    estimated_tokens: int


def select_compaction_prefix(
    messages: List[Dict[str, Any]],
    *,
    target_tokens: int,
    protect_tail_messages: int,
) -> Optional[CompactionPrefix]:
    """Choose the largest bounded old prefix ending at a turn boundary.

    The next conversational row must be a user message, which prevents
    splitting assistant tool calls from their tool results. Non-conversational
    rows are not sent to the summarizer; their IDs are still covered later by
    the database prefix selected through ``last_state_message_id``.
    """
    from agent.model_metadata import estimate_messages_tokens_rough

    conversational = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant", "tool", "function"}
        and message.get("_state_message_id") is not None
    ]
    max_prefix_len = len(conversational) - max(4, int(protect_tail_messages))
    if max_prefix_len < 4:
        return None

    running_tokens = 0
    best: Optional[CompactionPrefix] = None
    hard_ceiling = max(target_tokens, int(target_tokens * 1.25))
    for index, message in enumerate(conversational[:max_prefix_len]):
        clean = {
            key: value
            for key, value in message.items()
            if key != "_state_message_id"
        }
        running_tokens += estimate_messages_tokens_rough([clean])
        prefix_len = index + 1
        if prefix_len < 4:
            continue
        next_message = conversational[prefix_len]
        if next_message.get("role") != "user":
            if running_tokens > hard_ceiling and best is not None:
                break
            continue
        if running_tokens <= target_tokens:
            best = CompactionPrefix(
                messages=[
                    {
                        key: value
                        for key, value in item.items()
                        if key != "_state_message_id"
                    }
                    for item in conversational[:prefix_len]
                ],
                last_state_message_id=int(message["_state_message_id"]),
                estimated_tokens=running_tokens,
            )
            continue
        # A single unusually large turn may cross the target before any clean
        # boundary. Permit a small overshoot, but never feed an unbounded
        # transcript to the background summarizer.
        if best is None and running_tokens <= hard_ceiling:
            best = CompactionPrefix(
                messages=[
                    {
                        key: value
                        for key, value in item.items()
                        if key != "_state_message_id"
                    }
                    for item in conversational[:prefix_len]
                ],
                last_state_message_id=int(message["_state_message_id"]),
                estimated_tokens=running_tokens,
            )
        break
    if best is None:
        return None

    # The database commit replaces every active row through the selected
    # message ID.  Never let an unrecognised row (for example a legacy
    # persisted ``system``/``developer`` message) get swept into that range
    # without being summarized. Normal gateway transcripts contain only the
    # four roles above; unusual/legacy transcripts safely fall back to the
    # foreground compressor, which receives their complete message list.
    covered_roles = {
        message.get("role")
        for message in messages
        if isinstance(message, dict)
        and message.get("_state_message_id") is not None
        and int(message["_state_message_id"]) <= best.last_state_message_id
    }
    if not covered_roles.issubset({"user", "assistant", "tool", "function"}):
        return None
    return best


class GatewayBackgroundCompactionMixin:
    """GatewayRunner mixin implementing best-effort background compaction."""

    if TYPE_CHECKING:
        _session_db: Any
        async_session_store: Any
        _turn_leases: Any
        _background_tasks: set[asyncio.Task[Any]]

        def _resolve_session_agent_runtime(self, **kwargs: Any) -> tuple[str, dict]: ...

        async def _run_in_executor_with_context(
            self, func: Any, *args: Any
        ) -> Any: ...

        def _evict_cached_agent(self, session_key: str) -> None: ...

    def _init_background_compaction(self) -> None:
        self._background_compaction_tasks: Dict[str, asyncio.Task] = {}
        self._background_compaction_generation = 0

    def _background_compaction_settings(self) -> tuple[BackgroundCompactionSettings, dict]:
        # Runtime import avoids a module cycle: gateway.run imports this mixin.
        from gateway.run import _load_gateway_config

        config = _load_gateway_config()
        return BackgroundCompactionSettings.from_config(config), config

    def _maybe_schedule_background_compaction(
        self,
        *,
        source: Any,
        session_key: str,
        session_id: str,
        observed_prompt_tokens: int,
        context_length: int,
    ) -> bool:
        """Schedule one compactor for a session when the soft limit is crossed."""
        if not session_id or getattr(self, "_session_db", None) is None:
            return False
        settings, config = self._background_compaction_settings()
        if not settings.enabled:
            return False
        try:
            observed = int(observed_prompt_tokens or 0)
            context = int(context_length or 0)
        except (TypeError, ValueError):
            return False
        if observed <= 0 or context <= 0:
            return False
        if observed < int(context * settings.threshold):
            return False

        task_map = getattr(self, "_background_compaction_tasks", None)
        if not isinstance(task_map, dict):
            self._init_background_compaction()
            task_map = self._background_compaction_tasks
        existing = task_map.get(session_id)
        if existing is not None and not existing.done():
            return False

        self._background_compaction_generation += 1
        generation = self._background_compaction_generation
        task = asyncio.create_task(
            self._run_background_compaction(
                source=source,
                session_key=session_key,
                session_id=session_id,
                observed_prompt_tokens=observed,
                context_length=context,
                settings=settings,
                user_config=config,
                generation=generation,
            ),
            name=f"background-compaction:{session_id}",
        )
        task_map[session_id] = task
        background_tasks = getattr(self, "_background_tasks", None)
        if not isinstance(background_tasks, set):
            background_tasks = set()
            self._background_tasks = background_tasks
        background_tasks.add(task)

        def _discard(done: asyncio.Task) -> None:
            background_tasks.discard(done)
            if task_map.get(session_id) is done:
                task_map.pop(session_id, None)

        task.add_done_callback(_discard)
        logger.info(
            "Background compaction scheduled: session=%s tokens=%s/%s threshold=%d%%",
            session_id,
            f"{observed:,}",
            f"{context:,}",
            int(settings.threshold * 100),
        )
        return True

    async def _run_background_compaction(
        self,
        *,
        source: Any,
        session_key: str,
        session_id: str,
        observed_prompt_tokens: int,
        context_length: int,
        settings: BackgroundCompactionSettings,
        user_config: dict,
        generation: int,
    ) -> None:
        started = time.monotonic()
        try:
            snapshot = await self._session_db.get_compaction_snapshot(session_id)
            if snapshot.get("status") != "ready":
                logger.info(
                    "Background compaction skipped: session=%s snapshot=%s",
                    session_id,
                    snapshot.get("status"),
                )
                return

            messages = snapshot.get("messages") or []
            active_ids = snapshot.get("active_message_ids") or []
            chunk_target = min(
                settings.chunk_tokens,
                max(16_000, int(context_length * 0.35)),
            )
            prefix = select_compaction_prefix(
                messages,
                target_tokens=chunk_target,
                protect_tail_messages=settings.protect_tail_messages,
            )
            if prefix is None:
                logger.info(
                    "Background compaction skipped: session=%s has no safe bounded prefix",
                    session_id,
                )
                return
            expected_prefix_ids = [
                int(message_id)
                for message_id in active_ids
                if int(message_id) <= prefix.last_state_message_id
            ]
            if not expected_prefix_ids:
                return

            model, runtime = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
                user_config=user_config,
            )
            if str(runtime.get("api_mode") or "").lower() == "codex_app_server":
                # Codex owns the actual remote thread; rewriting only Hermes'
                # local transcript cannot shrink that context.
                logger.info(
                    "Background compaction skipped for Codex app-server session %s",
                    session_id,
                )
                return

            context_config = (
                user_config.get("context", {})
                if isinstance(user_config, dict)
                else {}
            )
            if isinstance(context_config, dict):
                context_engine = str(
                    context_config.get("engine") or "compressor"
                ).strip().lower()
            else:
                context_engine = "compressor"
            if context_engine != "compressor":
                # A plugin engine owns its own durable context state. Rewriting
                # the SQLite transcript behind it would violate that engine's
                # lifecycle contract.
                logger.info(
                    "Background compaction skipped for context engine %s: session=%s",
                    context_engine,
                    session_id,
                )
                return

            def _summarize_prefix():
                from agent.context_compressor import ContextCompressor

                compression_config = user_config.get("compression", {})
                if not isinstance(compression_config, dict):
                    compression_config = {}
                try:
                    threshold = float(compression_config.get("threshold", 0.50))
                except (TypeError, ValueError):
                    threshold = 0.50
                try:
                    protect_first = int(
                        compression_config.get("protect_first_n", 3)
                    )
                except (TypeError, ValueError):
                    protect_first = 3
                try:
                    target_ratio = float(
                        compression_config.get("target_ratio", 0.20)
                    )
                except (TypeError, ValueError):
                    target_ratio = 0.20
                compressor = ContextCompressor(
                    model=model,
                    threshold_percent=threshold,
                    protect_first_n=max(0, protect_first),
                    protect_last_n=settings.protect_tail_messages,
                    summary_target_ratio=target_ratio,
                    quiet_mode=True,
                    base_url=runtime.get("base_url") or "",
                    api_key=runtime.get("api_key") or "",
                    config_context_length=context_length,
                    provider=runtime.get("provider") or "",
                    api_mode=runtime.get("api_mode") or "",
                    # Background publication is optional. A failed summary
                    # must leave the original prefix untouched rather than
                    # committing a deterministic failure marker.
                    abort_on_summary_failure=True,
                    max_tokens=runtime.get("max_tokens"),
                )
                compressor.force_main_runtime = settings.use_main_model
                compressed = compressor.compress(
                    prefix.messages,
                    current_tokens=prefix.estimated_tokens,
                )
                aborted = bool(getattr(compressor, "_last_compress_aborted", False))
                return compressed, aborted

            compacted_prefix, aborted = (
                await self._run_in_executor_with_context(_summarize_prefix)
            )
            if aborted or not compacted_prefix:
                logger.warning(
                    "Background compaction summary aborted: session=%s",
                    session_id,
                )
                return

            from agent.model_metadata import estimate_messages_tokens_rough

            compacted_tokens = estimate_messages_tokens_rough(compacted_prefix)
            if compacted_tokens >= int(prefix.estimated_tokens * 0.95):
                logger.info(
                    "Background compaction made insufficient progress: session=%s "
                    "tokens=%s->%s",
                    session_id,
                    f"{prefix.estimated_tokens:,}",
                    f"{compacted_tokens:,}",
                )
                return

            lease_registry = getattr(self, "_turn_leases", None)
            lease_token = None
            if lease_registry is not None:
                lease_token = await lease_registry.acquire(
                    session_id,
                    owner_key=f"{session_key}:background-compaction",
                    generation=generation,
                    timeout=30.0,
                )
                if lease_token is not None and getattr(lease_token, "degraded", False):
                    logger.info(
                        "Background compaction commit skipped after turn-lease timeout: %s",
                        session_id,
                    )
                    return

            lock_holder = (
                f"background:{os.getpid()}:{generation}:{time.monotonic_ns()}"
            )
            lock_acquired = False
            try:
                lock_acquired = await self._session_db.try_acquire_compression_lock(
                    session_id,
                    lock_holder,
                    ttl_seconds=60.0,
                )
                if not lock_acquired:
                    logger.info(
                        "Background compaction commit skipped: compression lock busy for %s",
                        session_id,
                    )
                    return
                result = await self._session_db.commit_compaction_snapshot(
                    session_id,
                    expected_prefix_ids,
                    compacted_prefix,
                )
            finally:
                if lock_acquired:
                    await self._session_db.release_compression_lock(
                        session_id, lock_holder
                    )
                if lease_registry is not None and lease_token is not None:
                    lease_registry.release(lease_token)

            if result.get("status") != "committed":
                logger.info(
                    "Background compaction result discarded: session=%s reason=%s",
                    session_id,
                    result.get("reason", result.get("status")),
                )
                return

            await self.async_session_store.update_session(
                session_key,
                last_prompt_tokens=0,
            )
            self._evict_cached_agent(session_key)
            logger.info(
                "Background compaction committed: session=%s prefix=%d tail=%d "
                "tokens=%s->%s duration=%.1fs",
                session_id,
                result.get("archived_prefix_count", 0),
                result.get("preserved_tail_count", 0),
                f"{prefix.estimated_tokens:,}",
                f"{compacted_tokens:,}",
                time.monotonic() - started,
            )
        except asyncio.CancelledError:
            logger.info("Background compaction cancelled: session=%s", session_id)
            raise
        except Exception:
            logger.warning(
                "Background compaction failed: session=%s",
                session_id,
                exc_info=True,
            )


__all__ = [
    "BackgroundCompactionSettings",
    "CompactionPrefix",
    "GatewayBackgroundCompactionMixin",
    "select_compaction_prefix",
]
