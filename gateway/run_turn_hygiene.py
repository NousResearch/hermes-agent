"""Gateway turn hygiene phases; inherited by GatewayTurnMixin."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import inspect
import json
import os
import queue
import threading
import time
from agent.i18n import t
from agent.session_activity import format_iteration_progress
from contextlib import nullcontext, suppress
from contextvars import copy_context
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import (
    SessionSource, _session_key_namespace, build_channel_continuity_note,
    build_session_context,
)
from gateway.session_transcript import TranscriptReadError
from gateway.turn_context import TurnContext
from gateway.turn_lease import DEFAULT_LEASE_WAIT, TurnLeaseTimeoutError
from hermes_constants import get_hermes_home_override
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import base_url_hostname

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnHygieneMixin:
    @dataclasses.dataclass
    class _HygienePlan:
        """Hygiene pre-check outcome for one turn."""

        needs_compress: bool
        approx_tokens: int
        msg_count: int
        warn_token_threshold: int

    @staticmethod
    def _hmwa_hygiene_read_config(hs, data):
        """Apply model / compression knobs from the gateway config onto ``hs`` (invalid values keep the defaults)."""
        # Resolve model name (same logic as run_sync)
        _model_cfg = data.get("model", {})
        if isinstance(_model_cfg, str):
            hs.model = _model_cfg
        elif isinstance(_model_cfg, dict):
            hs.model = _model_cfg.get("default") or _model_cfg.get("model") or hs.model
            _raw_ctx = _model_cfg.get("context_length")
            if _raw_ctx is not None:
                with suppress(TypeError, ValueError):
                    hs.config_context_length = int(_raw_ctx)
            hs.provider = _model_cfg.get("provider") or None
            hs.base_url = _model_cfg.get("base_url") or None

        # Only the enabled flag is shared with the agent's compression config (hygiene runs higher).
        _comp_cfg = data.get("compression", {})
        if not isinstance(_comp_cfg, dict):
            return
        hs.compression_enabled = str(_comp_cfg.get("enabled", True)).lower() in {"true", "1", "yes"}

        def _knob(key, current, cast, allow_zero=False):
            raw = _comp_cfg.get(key)
            if raw is None:
                return current
            try:
                parsed = cast(raw)
            except (TypeError, ValueError):
                return current
            return parsed if (parsed >= 0 if allow_zero else parsed > 0) else current

        hs.hard_msg_limit = _knob("hygiene_hard_message_limit", hs.hard_msg_limit, int)
        hs.timeout_seconds = _knob("hygiene_timeout_seconds", hs.timeout_seconds, float)
        hs.total_ceiling_seconds = _knob("hygiene_total_ceiling_seconds", hs.total_ceiling_seconds, float)
        # The ceiling can never be tighter than one idle window, or the extension loop would be dead code.
        hs.total_ceiling_seconds = max(hs.total_ceiling_seconds, hs.timeout_seconds)
        hs.max_turn_hold_seconds = _knob("hygiene_max_turn_hold_seconds", hs.max_turn_hold_seconds, float)
        hs.failure_cooldown_seconds = _knob(
            "hygiene_failure_cooldown_seconds", hs.failure_cooldown_seconds, float, allow_zero=True,
        )

    async def _hmwa_hygiene_settings(self, source, session_key):
        """Resolve model/provider/context-length + hygiene knobs (fail-soft: errors keep defaults).

        The 0.85 threshold is deliberately HIGHER than the agent's compressor (0.50): a safety net
        for sessions that grew between turns. ``max_turn_hold_seconds`` bounds the TURN wait
        (compressor keeps running detached, commit fenced); kept below transport idle-timeouts."""
        from gateway.run import _load_gateway_config
        hs = self._HygieneSettings(
            model="anthropic/claude-sonnet-4.6", threshold_pct=0.85, compression_enabled=True,
            hard_msg_limit=5000, timeout_seconds=30.0, total_ceiling_seconds=600.0,
            max_turn_hold_seconds=10.0, failure_cooldown_seconds=300.0, config_context_length=None,
            provider=None, base_url=None, api_key=None, data={},
        )
        try:
            hs.data = _load_gateway_config()
            if hs.data:
                self._hmwa_hygiene_read_config(hs, hs.data)
            configured_model, configured_provider, configured_base_url = hs.model, hs.provider, hs.base_url

            with suppress(Exception):
                hs.model, _hyg_runtime = self._resolve_session_agent_runtime(
                    source=source, session_key=session_key,
                    user_config=hs.data if isinstance(hs.data, dict) else None,
                )
                hs.provider = _hyg_runtime.get("provider") or hs.provider
                hs.base_url = _hyg_runtime.get("base_url") or hs.base_url
                hs.api_key = _hyg_runtime.get("api_key") or hs.api_key

            if hs.config_context_length is not None:
                try:
                    from hermes_cli.route_identity import should_clear_context_pin_async

                    if await should_clear_context_pin_async(
                        configured_model, hs.model, configured_base_url, hs.base_url,
                        configured_provider, hs.provider,
                    ):
                        hs.config_context_length = None
                except Exception:
                    hs.config_context_length = None

            # custom_providers per-model context_length fallback (as in run_agent.py); needs base_url.
            if hs.config_context_length is None and hs.base_url:
                with suppress(TypeError, ValueError):
                    try:
                        from hermes_cli.config import (
                            get_compatible_custom_providers as _gw_gcp,
                            get_custom_provider_context_length as _gw_gccl,
                        )
                        _hyg_custom_providers = _gw_gcp(hs.data)
                    except Exception:
                        _hyg_custom_providers = hs.data.get("custom_providers")
                        if not isinstance(_hyg_custom_providers, list):
                            _hyg_custom_providers = []
                    _hyg_custom_ctx = _gw_gccl(
                        model=hs.model, base_url=hs.base_url, custom_providers=_hyg_custom_providers,
                    )
                    if _hyg_custom_ctx:
                        hs.config_context_length = int(_hyg_custom_ctx)
        except Exception:
            pass
        return hs

    async def _hmwa_hygiene_plan(self, hs, history, session_entry, session_key):
        """Decide whether hygiene compression fires this turn (token/message thresholds, DB-backed
        failure cooldown, in-flight compression)."""
        from agent.model_metadata import estimate_messages_tokens_rough, get_model_context_length_async
        _hyg_context_length = await get_model_context_length_async(
            hs.model, base_url=hs.base_url or "", api_key=hs.api_key or "",
            config_context_length=hs.config_context_length, provider=hs.provider or "",
        )
        _compress_token_threshold = int(_hyg_context_length * hs.threshold_pct)
        _warn_token_threshold = int(_hyg_context_length * 0.95)
        _msg_count = len(history)

        # Real usage decides: the API-reported prompt count, else the anchor persisted on the session
        # row (real count + delta of what was appended since, survives gateway restarts), else the
        # rough estimate (runs 30-50% high, which only fires hygiene early — safe). Do NOT compensate
        # with a threshold multiplier.
        from agent.image_token_cost import image_cost_context, learned_image_token_cost
        _anchored = None
        # Images in any local delta/estimate are priced at the cost learned from this model's usage.
        with image_cost_context(learned_image_token_cost(hs.model, hs.base_url)):
            if session_entry.last_prompt_tokens <= 0:
                from agent.usage_anchor import persisted_anchor_tokens
                _session_db = getattr(self, "_session_db", None)
                _anchored = persisted_anchor_tokens(
                    getattr(_session_db, "_db", _session_db), session_entry.session_id, history,
                )
            if session_entry.last_prompt_tokens > 0:
                _approx_tokens, _token_source = session_entry.last_prompt_tokens, "actual"
            elif _anchored is not None:
                _approx_tokens, _token_source = _anchored, "anchored"
            else:
                _approx_tokens, _token_source = estimate_messages_tokens_rough(history), "estimated"

        # Hard safety valve: force compression at an extreme message count regardless of tokens,
        # breaking the disconnect → no token data → no compression spiral. 5000 clears 1M+ sessions.
        _needs_compress = _approx_tokens >= _compress_token_threshold or _msg_count >= hs.hard_msg_limit

        if _needs_compress:
            # DB-backed cooldown (shared with context_compressor.py): survives gateway restarts, so a
            # failing compression is not re-triggered on every restart.
            # The in-memory dict was reset on every restart, re-triggering the same failing compression and
            # wedging session storage (#74136).
            _session_db = getattr(self, "_session_db", None)
            _getter = getattr(getattr(_session_db, "_db", _session_db), "get_compression_failure_cooldown", None)
            if _getter is not None:
                _cooldown_state = None
                with suppress(Exception):
                    _cooldown_state = _getter(session_entry.session_id)
                if _cooldown_state and _cooldown_state.get("remaining_seconds", 0) > 0:
                    logger.info(
                        "Session hygiene: skipping compression for %s; "
                        "previous failure cooldown active for %.1fs",
                        session_entry.session_id, _cooldown_state["remaining_seconds"],
                    )
                    _needs_compress = False

        if _needs_compress and await self._session_has_compression_in_flight(session_key):
            # A prior compression still holds the durable lock (e.g. a shielded worker left by /stop):
            # another attempt would wait up to 600s behind a commit the fence will refuse.
            logger.info(
                "Session hygiene: skipping compression for %s; "
                "another compression is already in flight", session_entry.session_id,
            )
            _needs_compress = False

        if _needs_compress:
            logger.info(
                "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                "(threshold: %s%% of %s = %s tokens)",
                _msg_count, f"{_approx_tokens:,}", _token_source,
                int(hs.threshold_pct * 100), f"{_hyg_context_length:,}", f"{_compress_token_threshold:,}",
            )
        return self._HygienePlan(_needs_compress, _approx_tokens, _msg_count, _warn_token_threshold)

    async def _hmwa_hygiene_wait_for_summary(self, attempt, hs, session_entry):
        """Progress-aware inline wait for the detached hygiene compressor. Returns the compressed
        transcript; raises ``HygieneTurnHoldExceeded`` (turn-hold budget) or
        ``asyncio.TimeoutError`` (idle/ceiling/fence cancel) for the caller's handlers.

        Idle timeout (fence ticks per streamed token) + hard ceiling + turn-hold cap."""
        from gateway.run import HygieneTurnHoldExceeded, hygiene_wait_should_extend
        fence = attempt.commit_fence
        while True:
            if fence.is_cancelled:
                raise asyncio.TimeoutError
            # Charge the idle budget from the LAST PROGRESS event, else silence can approach 2x timeout.
            _hyg_waited = time.monotonic() - attempt.wait_started
            _slice = min(
                max(hs.timeout_seconds - fence.seconds_since_progress(), 0.005),
                max(hs.total_ceiling_seconds - _hyg_waited, 0.005),
            )
            # Cap the slice at the remaining turn-hold budget so a continuously-streaming worker can't
            # hold the turn until the ceiling. Budget exhausted → immediate timeout → abandonment.
            _turn_hold_remaining = hs.max_turn_hold_seconds - (time.monotonic() - attempt.wait_started)
            _slice = 0.005 if _turn_hold_remaining <= 0 else min(_slice, max(_turn_hold_remaining, 0.005))
            # Short poll so a /stop or /restart cancel is not stuck behind a full idle window.
            _idle_left = max(hs.timeout_seconds - fence.seconds_since_progress(), 0.005)
            _slice = min(_slice, 0.25)
            try:
                _compressed, _ = await asyncio.wait_for(asyncio.shield(attempt.future), timeout=_slice)
                return _compressed
            except asyncio.TimeoutError:
                if fence.is_cancelled:
                    raise
                _hyg_waited = time.monotonic() - attempt.wait_started
                _idle = fence.seconds_since_progress()
                # Never hold the TURN past the budget even while the summary streams: proceed on the
                # uncompressed transcript so the wire never trips a transport idle-timeout.
                if _hyg_waited >= hs.max_turn_hold_seconds:
                    logger.info(
                        "Session hygiene compression for session %s exceeded the turn-hold "
                        "budget (%.1fs >= %.1fs) — abandoning inline wait, proceeding "
                        "without compression this turn",
                        session_entry.session_id, _hyg_waited, hs.max_turn_hold_seconds,
                    )
                    raise HygieneTurnHoldExceeded(
                        f"turn-hold budget {hs.max_turn_hold_seconds:.1f}s "
                        f"elapsed after {_hyg_waited:.1f}s"
                    )
                if hygiene_wait_should_extend(
                    idle=_idle, timeout=hs.timeout_seconds, waited=_hyg_waited,
                    ceiling=hs.total_ceiling_seconds, fence_cancelled=fence.is_cancelled,
                ):
                    if _slice >= _idle_left - 1e-9:
                        logger.info(
                            "Session hygiene compression for session %s still streaming after "
                            "%.0fs (last progress %.1fs ago) — extending wait (ceiling %.0fs)",
                            session_entry.session_id, _hyg_waited, _idle, hs.total_ceiling_seconds,
                        )
                    continue
                raise

    async def _hmwa_hygiene_cancel_or_adopt(self, attempt, context):
        """Cancel the worker at the commit fence; on success release its lease and defer agent
        cleanup, returning ``None``. When the worker already crossed into its commit, consume and
        return the compressed transcript instead (a successful compaction is never a timeout; the
        turn may be held past the budget by up to the commit duration — by design). The lock-free
        ``commit_in_flight`` marker keeps the poll from spinning on a hung commit."""
        fence = attempt.commit_fence
        while not fence.commit_in_flight:
            cancelled = fence.try_cancel_before_commit()
            if cancelled is None:
                await asyncio.sleep(0.025)
            elif cancelled:
                fence.release_cancelled_compression_lock()
                self._hmwa_hygiene_defer_cleanup(attempt, context)
                return None
            else:
                break
        _compressed, _ = await attempt.future
        return _compressed

    def _hmwa_hygiene_defer_cleanup(self, attempt, context):
        """Hand the agent's cleanup to the still-running worker future and mark it deferred."""
        self._defer_agent_cleanup_until_future_done(attempt.future, attempt.agent, context=context)
        attempt.cleanup_deferred = True

    @staticmethod
    def _hmwa_hygiene_stamp(agent, desc, provenance_name, debug_label):
        from agent.session_activity import ActivityProvenance
        from gateway.run import _stamp_hygiene_compression_provenance
        _stamp_hygiene_compression_provenance(agent, desc, getattr(ActivityProvenance, provenance_name), debug_label)

    async def _hmwa_hygiene_notify(self, source, meta, message, what):
        """Best-effort user notice on the hygiene thread; failure is logged, never raised."""
        try:
            _adapter = self._adapter_for_source(source)
            if _adapter and source.chat_id:
                await _adapter.send(source.chat_id, message, metadata=meta)
        except Exception as _werr:
            logger.warning("Failed to deliver %s to user: %s", what, _werr)

    async def _hmwa_hygiene_record_failure_cooldown(self, hs, session_key, session_id, reason):
        """Escalate the failure streak (off-loop) and persist the cooldown, when enabled."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        if hs.failure_cooldown_seconds < 0:
            return
        _hyg_cooldown = await asyncio.to_thread(
            _hygiene_cooldown_for_failure, self, session_key, hs.failure_cooldown_seconds,
        )
        _record_hygiene_cooldown(self, session_id, _hyg_cooldown, reason)

    async def _hmwa_hygiene_on_turn_hold(self, attempt, hs, session_entry, session_key, source):
        """``except HygieneTurnHoldExceeded`` body: keep or cancel the worker's commit admission,
        notify the user, and re-raise; returns the compressed transcript only when the worker
        was already committing.

        Turn-hold expiry is an availability boundary, not a failure: the streak must NOT advance,
        only flat retry spacing is recorded. A watermark-fenced commit (rows appended after
        compression start survive as cloned tail) KEEPS admission: the turn proceeds uncompressed
        now and the summary is adopted at the worker's fenced commit — always cancelling burned
        every attempt for thinking summary models. Without the fence a late commit could clobber
        newer turns, so cancel."""
        from gateway.run import (
            _HYGIENE_TURNHOLD_RETRY_SECONDS, _record_hygiene_cooldown, _reset_hygiene_failure_streak
        )
        fence = attempt.commit_fence
        _hyg_keep_admission = bool(getattr(fence, "commit_watermark_fenced", False)) and not fence.is_cancelled
        if _hyg_keep_admission:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene turn-hold")
            # NO retry-after here (it would also block the agent-side preflight compressor); spacing
            # comes from the durable compression lock. The done-callback records the flat retry-after
            # ONLY if the worker ends without committing anything.
            _sid, _skey, _agent = session_entry.session_id, session_key, attempt.agent

            def _hyg_adopt_or_space_retry(_fut, _gw=self, _sid=_sid, _skey=_skey, _agent=_agent):
                try:
                    _exc = _fut.exception()
                except (asyncio.CancelledError, Exception):
                    _committed = False
                else:
                    _committed = _exc is None and (
                        bool(getattr(_agent, "_last_compaction_in_place", False))
                        or getattr(_agent, "session_id", _sid) != _sid
                    )
                if _committed:
                    logger.info(
                        "Session hygiene compression for session %s finished after the "
                        "turn-hold was released — summary adopted at the watermark-fenced "
                        "commit boundary (#97963)", _sid,
                    )
                    try:
                        _reset_hygiene_failure_streak(_gw, _skey)
                    except Exception as _rs_err:
                        logger.debug("hygiene streak reset after deferred adoption failed: %s", _rs_err)
                else:
                    # Nothing to adopt (summary failed / fence refused / superseded): flat spacing so
                    # sustained traffic doesn't spawn and abandon a compressor every turn.
                    _record_hygiene_cooldown(
                        _gw, _sid, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                        "hygiene compression deferred: turn-hold budget expired and the "
                        "detached attempt did not commit",
                    )

            attempt.future.add_done_callback(_hyg_adopt_or_space_retry)
            _log_suffix = (
                " — the watermark-fenced worker keeps its commit admission and the summary "
                "will be adopted when it finishes"
            )
        else:
            _adopted = await self._hmwa_hygiene_cancel_or_adopt(attempt, "session hygiene turn-hold")
            if _adopted is not None:
                return _adopted
            # Short flat retry-after, else every turn re-spawns, holds and cancels a compressor.
            _record_hygiene_cooldown(
                self, session_entry.session_id, _HYGIENE_TURNHOLD_RETRY_SECONDS,
                "hygiene compression deferred: turn-hold budget expired while the "
                "summary was still streaming",
            )
            _log_suffix = ""
        self._hmwa_hygiene_stamp(
            attempt.agent, "session hygiene compression turn-hold",
            "AGENT_COMPRESSION_TURNHOLD", "hygiene compression turn-hold activity stamp failed",
        )
        logger.info(
            "Session hygiene compression for session %s exceeded turn-hold budget (%.1fs); "
            "proceeding without compression this turn%s",
            session_entry.session_id, time.monotonic() - attempt.wait_started, _log_suffix,
        )
        await self._hmwa_hygiene_notify(
            source, attempt.meta, t("gateway.compress.turnhold_deferred"), "compression-turnhold notice",
        )
        raise

    async def _hmwa_hygiene_on_timeout(self, attempt, hs, session_entry, session_key, source):
        """``except asyncio.TimeoutError`` body: cancel at the commit fence, record the failure
        cooldown, warn the user, and re-raise; returns the compressed transcript only when the
        worker crossed the commit boundary first."""
        from gateway.run import _hygiene_compression_timeout_message
        fence = attempt.commit_fence
        _hyg_waited = time.monotonic() - attempt.wait_started
        _hyg_total_exhausted = _hyg_waited >= hs.total_ceiling_seconds or fence.deadline_exceeded
        if _hyg_total_exhausted:
            # The worker checks this deadline between digest calls; keep its lease until it exits so
            # an unchanged session cannot overlap a retry (the release below is then a no-op).
            fence.retain_compression_lock_until_worker_done()
        # Capture fence state BEFORE try_cancel (which itself sets is_cancelled).
        _hyg_fence_cancelled = fence.is_cancelled
        _adopted = await self._hmwa_hygiene_cancel_or_adopt(attempt, "session hygiene timeout")
        if _adopted is not None:
            return _adopted
        await self._hmwa_hygiene_record_failure_cooldown(
            hs, session_key, session_entry.session_id,
            "session hygiene compression " + (
                "cancelled at commit fence" if _hyg_fence_cancelled
                else "total ceiling exhausted" if _hyg_total_exhausted
                else "timed out with no output from the summary model"
            ),
        )
        self._hmwa_hygiene_stamp(
            attempt.agent,
            "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
            else "session hygiene compression timed out",
            "AGENT_COMPRESSION_TIMEOUT", "hygiene compression timeout activity stamp failed",
        )
        if _hyg_fence_cancelled:
            logger.warning(
                "Session hygiene compression for session %s was cancelled at the "
                "commit fence; continuing without compression", session_entry.session_id,
            )
            raise
        _hyg_elapsed = time.monotonic() - attempt.wait_started
        if _hyg_total_exhausted:
            logger.warning(
                "Session hygiene compression for session %s reached its total ceiling after "
                "%.1fs (progress observed=%s); continuing without compression",
                session_entry.session_id, _hyg_elapsed, fence.progress_observed,
            )
        else:
            logger.warning(
                "Session hygiene compression for session %s made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without compression",
                session_entry.session_id, fence.seconds_since_progress(), _hyg_elapsed, hs.total_ceiling_seconds,
            )
        await self._hmwa_hygiene_notify(
            source, attempt.meta,
            _hygiene_compression_timeout_message(
                total_exhausted=_hyg_total_exhausted, elapsed=_hyg_elapsed,
                idle_timeout=hs.timeout_seconds, progress_observed=fence.progress_observed,
            ),
            "compression-timeout warning",
        )
        raise

    def _hmwa_hygiene_on_unwind(self, attempt, hs, session_entry, session_key):
        """``except BaseException`` body (caller re-raises): revoke commit admission BEFORE the host
        unwinds so the detached worker can never commit later, and record a cooldown — otherwise
        the next turn re-arms hygiene and waits up to 600s behind a fence that refuses again."""
        from gateway.run import _hygiene_cooldown_for_failure, _record_hygiene_cooldown
        attempt.commit_fence.revoke_commit_admission()
        if not attempt.cleanup_deferred:
            self._hmwa_hygiene_defer_cleanup(attempt, "session hygiene unwind")
        if hs.failure_cooldown_seconds >= 0:
            try:
                _record_hygiene_cooldown(
                    self, session_entry.session_id,
                    _hygiene_cooldown_for_failure(self, session_key, hs.failure_cooldown_seconds),
                    "session hygiene compression cancelled at commit fence",
                )
            except Exception as _cd_err:
                logger.debug("hygiene unwind cooldown record failed: %s", _cd_err)

    async def _hmwa_hygiene_adopt_transcript(
        self, attempt, _compressed, history, plan, *, session_entry, source, _quick_key, run_generation,
    ):
        """Adopt a finished compression (rotation / in-place / refused); publishes the transcript to
        continue with on ``attempt.history``. Returns ``(rotated, in_place, new_count, new_tokens)``.

        Rewrite only on rotation (NEW session id): in-place compaction already soft-archived the
        old rows and rewrite_transcript() would DELETE them; neither rotation nor in-place signals
        FAILURE and an unconditional rewrite would leave only the summary. Write-before-repoint:
        a repoint-then-failed-rewrite would point the live entry at an empty session."""
        from agent.model_metadata import estimate_messages_tokens_rough
        _hyg_agent = attempt.agent
        # _compress_context rotates to a NEW session_id so the old transcript stays intact/searchable.
        _hyg_new_sid = _hyg_agent.session_id
        _hyg_rotated = _hyg_new_sid != session_entry.session_id
        _hyg_in_place = bool(getattr(_hyg_agent, "_last_compaction_in_place", False))
        # Anti-growth guard: refuse a compression that did not shrink the transcript (seen 427K→598K).
        _hyg_in_toks = estimate_messages_tokens_rough(history)
        _hyg_out_toks = estimate_messages_tokens_rough(_compressed)
        if _hyg_rotated and _hyg_out_toks > _hyg_in_toks:
            logger.warning(
                "Gateway hygiene compression for session %s would grow transcript (~%s -> ~%s "
                "tokens); keeping the original transcript unchanged",
                session_entry.session_id, f"{_hyg_in_toks:,}", f"{_hyg_out_toks:,}",
            )
            _hyg_rotated = False
            _compressed = history
        # Only rewrite the transcript when rotation produced a NEW session id. In-place compaction does NOT
        # need a rewrite: archive_and_compact() has already soft-archived the previous active rows and
        # inserted the compacted messages as the new active set inside _compress_context(). Calling
        # rewrite_transcript() after in-place compaction would invoke replace_messages(active_only=False)
        # which DELETEs ALL rows — including the archived turns that archive_and_compact() deliberately
        # preserved (silent data loss, #61145). The danger this guards against (mirrors the /compress fix
        # #44794/#39704): if _compress_context returns a summary but neither rotates nor completes
        # archive_and_compact(), the session_id is unchanged for a FAILURE reason, and an unconditional
        # rewrite_transcript() would DELETE the original messages and replace them with only the compressed
        # summary (permanent data loss, #21301). Write-before-repoint (mirrors manual /compress): if we
        # repointed session_entry onto the child SID and rewrite_transcript then failed (lock/ENOSPC), the
        # live entry would already reference a brand-new empty session while the turn continues — the
        # conversation silently vanishes. Persist the child transcript first; only then rebind the live
        # entry.
        if _hyg_rotated:
            if not await self.async_session_store.rewrite_transcript(_hyg_new_sid, _compressed):
                logger.error(
                    "Session hygiene: failed to persist compressed transcript for rotated session "
                    "%s → %s; keeping the live entry on the original session so the "
                    "conversation is not dropped", session_entry.session_id, _hyg_new_sid,
                )
                # Fail closed: treat like no rotation.
                _hyg_rotated = False
                _hyg_in_place = False
            else:
                session_entry.session_id = _hyg_new_sid
                # The held turn lease follows the rotation (alias keys still serialize on this turn).
                self._rebind_turn_lease(_quick_key, run_generation, _hyg_new_sid)
                await self.async_session_store._save()
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding, source, session_entry, reason="hygiene-compression",
                )

        if _hyg_rotated or _hyg_in_place:
            # Rewritten (rotation) or persisted by archive_and_compact() (in-place): reset token count.
            session_entry.last_prompt_tokens = 0
            attempt.history = _compressed
            _new_count = len(_compressed)
            _new_tokens = estimate_messages_tokens_rough(_compressed)
        else:
            # No rewrite happened — post-compression counts equal the pre-compression ones.
            _new_count = plan.msg_count
            _new_tokens = plan.approx_tokens
            logger.warning(
                "Gateway hygiene compression for session %s did not rotate or compact in place (no "
                "session_db on the hygiene agent) — preserving the original transcript instead "
                "of overwriting it with the summary (#21301).", session_entry.session_id,
            )

        logger.info(
            "Session hygiene: compressed %s → %s msgs, ~%s → ~%s tokens",
            plan.msg_count, _new_count, f"{plan.approx_tokens:,}", f"{_new_tokens:,}",
        )
        if _new_tokens >= plan.warn_token_threshold:
            logger.warning("Session hygiene: still ~%s tokens after compression", f"{_new_tokens:,}")
        return _hyg_rotated, _hyg_in_place, _new_count, _new_tokens

    async def _hmwa_hygiene_apply_result(
        self, attempt, hs, _compressed, history, plan, *,
        session_entry, session_key, source, _quick_key, run_generation,
    ):
        """Adopt a finished hygiene compression, rebind the session + turn lease, record
        streak/cooldown, and warn the user on abort."""
        from gateway.run import _reset_hygiene_failure_streak, hygiene_compaction_recovered
        _hyg_rotated, _hyg_in_place, _new_count, _new_tokens = await self._hmwa_hygiene_adopt_transcript(
            attempt, _compressed, history, plan, session_entry=session_entry, source=source,
            _quick_key=_quick_key, run_generation=run_generation,
        )
        # Summary failure aborts the compressor (nothing dropped). Warn the user visibly — agent.log
        # is invisible on TG/Discord — so they know the chat is "frozen" and can /compress or /reset.
        _comp = getattr(attempt.agent, "context_compressor", None)
        _hyg_aborted = _comp is not None and getattr(_comp, "_last_compress_aborted", False)
        # A fence-cancelled _compress_context returns the original transcript with
        # _last_compress_aborted False: treat that no-op as an abort so hygiene records a cooldown
        # instead of retrying into the 600s wait. A committed rotate/in-place is never an abort.
        _hyg_fence_cancelled = bool(attempt.commit_fence.is_cancelled and not _hyg_rotated and not _hyg_in_place)
        if _hyg_fence_cancelled:
            _hyg_aborted = True
        # Recovery decision lives in the unit-tested predicate: the "neither rotated nor in place"
        # path reuses pre-compression counts, so a numbers-only check would read a no-op as success.
        if not _hyg_aborted and hygiene_compaction_recovered(
            aborted=_hyg_aborted, rotated=_hyg_rotated, in_place=_hyg_in_place,
            msg_count=plan.msg_count, new_count=_new_count, approx_tokens=plan.approx_tokens,
            new_tokens=_new_tokens,
        ):
            await asyncio.to_thread(_reset_hygiene_failure_streak, self, session_key)
        if _hyg_aborted:
            await self._hmwa_hygiene_record_failure_cooldown(
                hs, session_key, session_entry.session_id,
                "session hygiene compression cancelled at commit fence" if _hyg_fence_cancelled
                else getattr(_comp, "_last_summary_error", None),
            )
            self._hmwa_hygiene_stamp(
                attempt.agent, "session hygiene compression aborted",
                "AGENT_COMPRESSION_COOLDOWN", "hygiene compression abort activity stamp failed",
            )
            if not _hyg_fence_cancelled:
                # Force-redact: provider exception text may contain credentials; this reaches users.
                from agent.redact import redact_sensitive_text
                _err = redact_sensitive_text(getattr(_comp, "_last_summary_error", None) or "unknown error", force=True)
                await self._hmwa_hygiene_notify(
                    source, attempt.meta, "⚠️ Context compression aborted "
                    f"({_err}). No messages were dropped — "
                    "conversation is unchanged. Run /compress to retry, /reset for a clean "
                    "session, or check your auxiliary.compression model configuration.",
                    "compression-failure warning",
                )
        # Configured aux model failed, recovered on the main model: only the user can fix that config.
        elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
            _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
            _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
            await self._hmwa_hygiene_notify(
                source, attempt.meta, f"ℹ️ Configured compression model `{_aux_model}` "
                f"failed ({_aux_err}). Recovered using your main "
                "model — context is intact — but you may want to "
                "check `auxiliary.compression.model` in config.yaml.",
                "aux-model-fallback notice",
            )

    async def _hmwa_hygiene_codex_compaction(self, hs, plan, history, session_entry, session_key, _hyg_runtime):
        """codex app-server runtime: the real context is the server-side thread, not the transcript
        mirror. The detached-agent path would only rewrite the mirror and its finally-eviction
        would destroy the live thread (next turn starts blank), so use the cached agent's
        thread/compact/start and KEEP it cached."""
        from gateway.run import run_codex_hygiene_compaction
        # codex app-server runtime: the model's real context is the app-server's server-side thread, not the
        # transcript mirror. See #73503.
        _hyg_codex_auto = "native"
        _hyg_comp_cfg = hs.data.get("compression") if isinstance(hs.data, dict) else None
        if isinstance(_hyg_comp_cfg, dict):
            _hyg_codex_auto = str(_hyg_comp_cfg.get("codex_app_server_auto", "native") or "native")
        _hyg_codex_outcome = await run_codex_hygiene_compaction(
            self, session_key, session_entry.session_id, auto_mode=_hyg_codex_auto, history=history,
            approx_tokens=plan.approx_tokens, timeout_seconds=hs.total_ceiling_seconds,
            failure_cooldown_seconds=hs.failure_cooldown_seconds,
        )
        logger.info(
            "Session hygiene (codex app-server): %s (session=%s, mode=%s, ~%s tokens)",
            _hyg_codex_outcome, session_entry.session_id, _hyg_codex_auto, f"{plan.approx_tokens:,}",
        )

    async def _hmwa_hygiene_build_agent(self, _hyg_model, _hyg_runtime, session_entry):
        """Build the detached hygiene ``AIAgent`` with the live session's system prompt. Returns
        ``(agent, sync_session_db)``."""
        from gateway.run import _GATEWAY_HYGIENE_PLATFORM, _seed_hygiene_system_prompt
        from run_agent import AIAgent
        try:
            _hyg_session_row = await self._session_db.get_session(session_entry.session_id)
        except Exception as exc:
            _hyg_session_row = None
            logger.warning(
                "Session hygiene could not restore the system prompt for session %s: %s. "
                "Preserving an empty prompt so the live turn rebuilds it with its "
                "configured providers.", session_entry.session_id, exc, exc_info=True,
            )
        _hyg_session_db = getattr(self._session_db, "_db", self._session_db)
        # With compression.checkpoint_required on, load the memory provider so the checkpoint exists
        # before any mutation; otherwise keep the fast path (no provider init).
        from hermes_cli.config import load_config as _load_cfg
        from utils import is_truthy_value as _is_truthy

        _hyg_checkpoint_required = _is_truthy(
            ((_load_cfg() or {}).get("compression") or {}).get("checkpoint_required"), default=False,
        )
        _hyg_agent = AIAgent(
            **_hyg_runtime, model=_hyg_model, max_iterations=4, quiet_mode=True,
            skip_memory=not _hyg_checkpoint_required, enabled_toolsets=["memory"],
            session_id=session_entry.session_id, session_db=_hyg_session_db,
        )
        _seed_hygiene_system_prompt(_hyg_agent, _hyg_session_row)
        # A rebuilt (not retained) prompt is deliberately stale for every real gateway surface.
        _hyg_agent.platform = _GATEWAY_HYGIENE_PLATFORM
        return _hyg_agent, _hyg_session_db

    async def _hmwa_hygiene_detached_attempt(
        self, attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
        source, session_entry, session_key, _quick_key, run_generation,
    ):
        """Run one detached hygiene compression attempt end to end; publishes the transcript to
        continue with (compressed or original) on ``attempt.history``."""
        from gateway.run import HygieneTurnHoldExceeded
        from agent.conversation_compression import CompressionCommitFence
        _hyg_agent, _hyg_session_db = await self._hmwa_hygiene_build_agent(_hyg_model, _hyg_runtime, session_entry)
        attempt.agent = _hyg_agent
        try:
            # Hygiene owns the session binding, so prefer in-place compaction over minting a
            # continuation child. Without a SessionDB this stays False.
            _hyg_agent.compression_in_place = True
            _bind_hyg_state = getattr(getattr(_hyg_agent, "context_compressor", None), "bind_session_state", None)
            if callable(_bind_hyg_state):
                _bind_hyg_state(_hyg_session_db, session_entry.session_id)
            # Never finalize on close() — that would end the live gateway session row.
            _hyg_agent._end_session_on_close = False
            _hyg_agent._print_fn = lambda *a, **kw: None

            loop = asyncio.get_running_loop()
            _hyg_commit_fence = CompressionCommitFence(total_ceiling_seconds=hs.total_ceiling_seconds)
            # Default executor (NOT self._get_executor): a hung summary must never occupy an
            # agent-work slot. MUST run in the caller's contextvars (multiplex secret scope).
            attempt.commit_fence = _hyg_commit_fence
            attempt.future = loop.run_in_executor(
                None,
                # But it MUST run inside the caller's contextvars: under multiplex_profiles the profile
                # secret scope / HERMES_HOME override live in ContextVars, and a bare run_in_executor worker
                # starts with an empty Context — the summary model's get_secret(<PROVIDER>_API_KEY) then
                # fails closed (UnscopedSecretError) and every hygiene compaction silently degrades to a
                # lossy truncation (#100849 bundle).
                copy_context().run,
                lambda: _hyg_agent._compress_context(
                    _hyg_msgs, "", approx_tokens=plan.approx_tokens, commit_fence=_hyg_commit_fence,
                ),
            )
            attempt.wait_started = time.monotonic()
            try:
                _compressed = await self._hmwa_hygiene_wait_for_summary(attempt, hs, session_entry)
            except HygieneTurnHoldExceeded:
                _compressed = await self._hmwa_hygiene_on_turn_hold(attempt, hs, session_entry, session_key, source)
            except asyncio.TimeoutError:
                _compressed = await self._hmwa_hygiene_on_timeout(attempt, hs, session_entry, session_key, source)
            except BaseException:
                self._hmwa_hygiene_on_unwind(attempt, hs, session_entry, session_key)
                raise

            await self._hmwa_hygiene_apply_result(
                attempt, hs, _compressed, history, plan, session_entry=session_entry,
                session_key=session_key, source=source, _quick_key=_quick_key,
                run_generation=run_generation,
            )
        finally:
            # Evict the cached agent so the next turn rebuilds its system prompt.
            self._evict_cached_agent(session_key)
            if not attempt.cleanup_deferred:
                await self._cleanup_agent_resources_off_loop(_hyg_agent, context="session hygiene")

    async def _hmwa_run_session_hygiene(
        self, event, source, session_entry, session_key, history, _quick_key, run_generation,
    ):
        """Auto-compress pathologically large transcripts before the agent starts so oversized
        histories don't cause repeated truncation/context failures. Token source: the API's
        prompt_tokens from the last turn, else a char/4 estimate."""
        from gateway.run import HygieneTurnHoldExceeded
        if not history or len(history) < 4:
            return history

        hs = await self._hmwa_hygiene_settings(source, session_key)
        if not hs.compression_enabled:
            return history
        plan = await self._hmwa_hygiene_plan(hs, history, session_entry, session_key)
        if not plan.needs_compress:
            return history

        attempt = self._HygieneAttempt(agent=None, meta=self._event_thread_metadata(event, source), history=history)
        try:
            _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                source=source, session_key=session_key,
                user_config=hs.data if isinstance(hs.data, dict) else None,
            )
            if str(_hyg_runtime.get("api_mode") or "").lower() == "codex_app_server":
                await self._hmwa_hygiene_codex_compaction(hs, plan, history, session_entry, session_key, _hyg_runtime)
            elif _hyg_runtime.get("api_key"):
                # Pass the FULL transcript (tool results included) as the agent loop does: filtering
                # to user/assistant starved the compressor (tool results are the bulk of context).
                _hyg_msgs = [m for m in history if m.get("role") in {"user", "assistant", "tool"}]
                if len(_hyg_msgs) >= 4:
                    await self._hmwa_hygiene_detached_attempt(
                        attempt, hs, plan, history, _hyg_msgs, _hyg_model, _hyg_runtime,
                        source, session_entry, session_key, _quick_key, run_generation,
                    )
        except HygieneTurnHoldExceeded:
            # Availability boundary, not a failure — already logged at INFO by the turn-hold handler.
            # Must not hit the generic "auto-compress failed" warning below: that log is how thinking-model
            # deployments read as permanently broken (#97963; surfaced by @686f6c61 in PR #99657).
            pass
        except Exception as e:
            logger.warning("Session hygiene auto-compress failed: %s", e)
        return attempt.history
