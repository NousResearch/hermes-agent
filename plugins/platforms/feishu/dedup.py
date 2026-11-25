"""Persistent Feishu message-id claims and safe release before gateway admission."""

import asyncio
import json
import logging
import time

logger = logging.getLogger("plugins.platforms.feishu.adapter")
_DEDUP_TTL_SECONDS = 24 * 60 * 60


class _Claim:
    def __init__(self, adapter, message_id, token):
        self.adapter, self.message_id, self.token = adapter, message_id, token
        self.dispatched = False

    async def release(self):
        try:
            await self.adapter._release_message_claim(self.message_id, self.token)
        except Exception:
            logger.exception("[Feishu] Failed to release an unadmitted message claim")


class FeishuDedupMixin:
    async def handle_message(self, event):
        event._mark_ingress_dispatched()
        await super().handle_message(event)

    def adopt_inbound_dedup(self, previous):
        """Share claims with a reconnecting instance while the old instance still has work."""
        if (getattr(previous, "platform", None) != self.platform
                or getattr(previous, "_dedup_state_path", None) != self._dedup_state_path):
            return
        with previous._dedup_lock:
            self._seen_message_ids = previous._seen_message_ids
            self._seen_message_order = previous._seen_message_order
            self._dedup_claim_tokens = previous.__dict__.setdefault("_dedup_claim_tokens", {})
            self._dedup_persist_lock = previous._dedup_persist_lock_or_create()
            self._dedup_lock = previous._dedup_lock

    def _load_seen_message_ids(self) -> None:
        try:
            payload = json.loads(self._dedup_state_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning("[Feishu] Failed to load persisted dedup state from %s", self._dedup_state_path, exc_info=True)
            return
        seen_data = payload.get("message_ids", {}) if isinstance(payload, dict) else {}
        now = time.time()
        if isinstance(seen_data, list):  # legacy format: plain list of IDs, no timestamps
            entries = {str(item).strip(): 0.0 for item in seen_data if str(item).strip()}
        elif isinstance(seen_data, dict):
            entries = {}
            for key, value in seen_data.items():
                if isinstance(key, str) and key.strip() and isinstance(value, (int, float, str)):
                    try:
                        entries[key] = float(value)
                    except ValueError:
                        pass
        else:
            return
        valid = {m: ts for m, ts in entries.items() if ts == 0.0 or _DEDUP_TTL_SECONDS <= 0 or now - ts < _DEDUP_TTL_SECONDS}
        sorted_ids = sorted(valid, key=lambda k: valid[k], reverse=True)[:self._dedup_cache_size]
        self._seen_message_order = list(reversed(sorted_ids))
        self._seen_message_ids = {k: valid[k] for k in sorted_ids}

    async def _persist_seen_async(self):
        async with self._dedup_persist_lock_or_create():
            await self._run_blocking(self._persist_seen_message_ids)

    async def _claim_message_id(self, message_id):
        now, token = time.time(), object()
        with self._dedup_lock:
            seen_at = self._seen_message_ids.get(message_id)
            if seen_at is not None and (_DEDUP_TTL_SECONDS <= 0 or now - seen_at < _DEDUP_TTL_SECONDS):
                return None
            owners = self.__dict__.setdefault("_dedup_claim_tokens", {})
            if seen_at is not None:
                self._seen_message_order[:] = [item for item in self._seen_message_order if item != message_id]
            self._seen_message_ids[message_id] = now
            owners[message_id] = token
            self._seen_message_order.append(message_id)
            while len(self._seen_message_order) > self._dedup_cache_size:
                oldest = self._seen_message_order.pop(0)
                self._seen_message_ids.pop(oldest, None)
                owners.pop(oldest, None)
        claim = _Claim(self, message_id, token)
        # Keep the lock and the write alive after cancellation: a cancelled executor future
        # can still finish its write, so release must wait until that write has settled.
        writing = asyncio.create_task(self._persist_seen_async())
        try:
            await asyncio.shield(writing)
        except BaseException:
            async def release_after_write():
                try:
                    await writing
                except Exception:
                    logger.exception("[Feishu] Persisting the inbound claim failed")
                await claim.release()

            cleanup = asyncio.create_task(release_after_write())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass  # the shielded cleanup still owns the pending write
            raise
        return claim

    async def _is_duplicate(self, message_id: str) -> bool:
        return await self._claim_message_id(message_id) is None

    async def _release_message_claim(self, message_id, token):
        async with self._dedup_persist_lock_or_create():
            with self._dedup_lock:
                owners = self.__dict__.setdefault("_dedup_claim_tokens", {})
                if owners.get(message_id) is not token:
                    return
                owners.pop(message_id, None)
                self._seen_message_ids.pop(message_id, None)
                self._seen_message_order[:] = [item for item in self._seen_message_order if item != message_id]
            # A retired adapter cannot restart its SDK executor. Its replacement shares this
            # state and will persist the released ID when a redelivery claims it again.
            if not getattr(self, "_sdk_executor_closing", False):
                await self._run_blocking(self._persist_seen_message_ids)

    def _dedup_persist_lock_or_create(self) -> asyncio.Lock:
        lock = getattr(self, "_dedup_persist_lock", None)
        if lock is None:
            lock = self._dedup_persist_lock = asyncio.Lock()
        return lock

    async def _release_buffered_ingress(self):
        for event in [*self._pending_text_batches.values(), *self._pending_media_batches.values()]:
            await event._release_unadmitted_claims()
