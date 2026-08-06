"""/compress slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

import asyncio

from agent.i18n import t
from gateway.platforms.base import MessageEvent

from gateway.slash_commands._shared import logger

class CompressCommandsMixin:
    """/compress handlers."""

    async def _handle_compress_command(self, event: MessageEvent) -> str:
        """Profile-scoping wrapper around manual /compress.

        Multiplexed gateways resolve credentials through the fail-closed
        per-profile secret scope (``agent.secret_scope``, Workstream A). The
        agent turn installs it via ``_run_agent``'s wrapper, but slash-command
        dispatch does not — so manual /compress reached the compressor's
        provider resolution unscoped and died with ``UnscopedSecretError``
        (``get_secret('OPENROUTER_BASE_URL') called with no profile secret
        scope active``). Install the source profile's scope around the whole
        handler, mirroring ``_run_agent``. Single-profile gateways skip this
        — zero behavior change.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._handle_compress_command_inner(event)

        from gateway.run import _profile_runtime_scope

        profile_home = self._resolve_profile_home_for_source(event.source)
        with _profile_runtime_scope(profile_home):
            return await self._handle_compress_command_inner(event)

    async def _handle_compress_command_inner(self, event: MessageEvent) -> str:
        """Handle /compress command -- manually compress conversation context.

        Accepts an optional focus topic: ``/compress <focus>`` guides the
        summariser to preserve information related to *focus* while being
        more aggressive about discarding everything else.

        Also accepts the boundary-aware form ``/compress here [N]``:
        summarize everything except the most recent ``N`` exchanges
        (default 2), kept verbatim. Inspired by Claude Code's Rewind
        "Summarize up to here" action (v2.1.139, May 2026,
        https://code.claude.com/docs/en/whats-new/2026-w20).
        """
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)

        if not history or len(history) < 4:
            return t("gateway.compress.not_enough")

        # Parse args: either a focus topic (full compress) or the
        # boundary-aware "here [N]" form (partial compress).
        from hermes_cli.partial_compress import (
            extract_compress_flags,
            parse_partial_compress_args,
            rejoin_compressed_head_and_tail,
            split_history_for_partial_compress,
            summarize_compress_preview,
        )
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification,
        )
        _raw_args = (event.get_command_args() or "").strip()
        # Strip --preview/--dry-run/--aggressive before positional parsing
        # so the flags coexist with 'here [N]' / focus-topic forms.
        _raw_args, _preview, _aggressive = extract_compress_flags(_raw_args)
        partial, keep_last, focus_topic = parse_partial_compress_args(_raw_args)

        _agg_note = ""
        if _aggressive:
            # LLM-free hard truncation is not supported on this surface —
            # it would need its own transcript-persistence branch outside
            # the guarded _compress_context rotation machinery (#44794).
            _agg_note = t("gateway.compress.aggressive_unsupported")
            if not _preview:
                return _agg_note

        if _preview:
            # Report what WOULD be compressed — no agent, no writes.
            from agent.model_metadata import estimate_request_tokens_rough
            _pv_msgs = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            approx_tokens = estimate_request_tokens_rough(_pv_msgs)
            report = summarize_compress_preview(
                _pv_msgs, partial, keep_last, focus_topic, approx_tokens
            )
            lines = [f"🗜️ {line}" for line in report["lines"]]
            if _aggressive:
                lines.append(_agg_note)
            return "\n".join(lines)

        try:
            from run_agent import AIAgent
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough

            session_key = self._session_key_for_source(source)
            # Preserve the same platform + stable gateway session identity that a
            # normal gateway turn passes (gateway/run.py main turn), so external
            # context engines bind this temporary compression agent to the
            # original platform conversation instead of falling back to an
            # unbound/default "cli" host source — see #50422. _platform_config_key
            # maps LOCAL->"cli" exactly like the live turn, avoiding a new
            # "local" vs "cli" mismatch.
            from gateway.run import (
                _GATEWAY_HYGIENE_PLATFORM,
                _platform_config_key,
                _seed_hygiene_system_prompt,
            )
            platform_key = (
                _platform_config_key(source.platform) if source.platform else None
            )
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
            if not runtime_kwargs.get("api_key"):
                return t("gateway.compress.no_provider")

            # Pass the FULL transcript (tool results included) — same
            # rationale as the session-hygiene auto-compress in
            # gateway/run.py (#3854): filtering to user/assistant-only
            # starves the compressor's tool-result pruning and can trip the
            # protect-first/last early-return on short filtered histories.
            msgs = [
                m for m in history
                if m.get("role") in {"user", "assistant", "tool"}
            ]

            # Boundary-aware split: only the head is summarized; the most
            # recent `keep_last` exchanges are preserved verbatim. The
            # split snaps the tail to a user-turn start so the rejoined
            # transcript keeps role alternation valid.
            tail: list = []
            head = msgs
            if partial:
                head, tail = split_history_for_partial_compress(msgs, keep_last)
                if not tail:
                    # Degenerate split — fall back to full compression.
                    partial = False
                    head = msgs

            # Bind the temporary compression agent to the originating source's
            # platform + stable gateway session key. These are *authoritative*
            # identity invariants (derived from `source`), so assign them into
            # runtime_kwargs directly rather than via setdefault: a value already
            # present there from the resolver would be a placeholder/stale
            # identity and must not win. Assigning (vs passing a second explicit
            # kwarg) also keeps each key single-valued, avoiding a "got multiple
            # values for keyword argument" TypeError. platform is only set when
            # known: for a source without platform metadata we leave it unset so
            # AIAgent's default (platform=None -> source "cli") applies, exactly
            # the prior behavior. _resolve_session_agent_runtime does not set
            # either key today, so in practice this just adds them.
            if platform_key is not None:
                runtime_kwargs["platform"] = platform_key
            runtime_kwargs["gateway_session_key"] = session_key

            # The manual compression helper skips memory-provider initialization,
            # but _compress_context may persist its cached system prompt. Restore
            # the exact live-session prompt so provider blocks are retained.
            session_row = None
            get_session = getattr(self._session_db, "get_session", None)
            if callable(get_session):
                try:
                    session_row = await get_session(session_entry.session_id)
                except Exception as exc:
                    logger.warning(
                        "Manual compression could not restore the system prompt "
                        "for session %s: %s. Preserving an empty prompt so the "
                        "live turn rebuilds it with its configured providers.",
                        session_entry.session_id,
                        exc,
                        exc_info=True,
                    )

            tmp_agent = AIAgent(
                **runtime_kwargs,
                model=model,
                max_iterations=4,
                quiet_mode=True,
                skip_memory=True,
                enabled_toolsets=["memory"],
                session_id=session_entry.session_id,
                session_db=getattr(self._session_db, "_db", self._session_db),
            )
            _seed_hygiene_system_prompt(tmp_agent, session_row)
            # Keep the real source platform during construction so external
            # context engines bind correctly. If compression has to rebuild the
            # prompt, stamp that provider-less fallback as stale for the next
            # real gateway turn.
            tmp_agent.platform = _GATEWAY_HYGIENE_PLATFORM
            try:
                tmp_agent._print_fn = lambda *a, **kw: None
                # Prevent close() from ending the newly rotated session —
                # the gateway session entry now points at the new id and
                # must remain open for the next user turn.
                tmp_agent._end_session_on_close = False

                # Estimate with system prompt + tool schemas included so the
                # figure reflects real request pressure, not a transcript-only
                # underestimate (#6217). Must be computed after tmp_agent is
                # built so _cached_system_prompt/tools are populated.
                _sys_prompt = getattr(tmp_agent, "_cached_system_prompt", "") or ""
                _tools = getattr(tmp_agent, "tools", None) or None
                approx_tokens = estimate_request_tokens_rough(
                    msgs, system_prompt=_sys_prompt, tools=_tools
                )

                compressor = tmp_agent.context_compressor
                if not compressor.has_content_to_compress(head):
                    return t("gateway.compress.nothing_to_do")

                # _run_in_executor_with_context (not a bare run_in_executor):
                # the profile secret scope installed by the wrapper is a
                # contextvar, and the default-executor hop would drop it —
                # the compressor's aux-client provider resolution would then
                # read credentials unscoped and fail closed under
                # multiplexing.
                compressed, _ = await self._run_in_executor_with_context(
                    lambda: tmp_agent._compress_context(
                        head,
                        "",
                        approx_tokens=approx_tokens,
                        focus_topic=focus_topic,
                        force=True,
                        defer_context_engine_notification=True,
                    )
                )

                # If _compress_context returned unchanged because a
                # concurrent compression lock is held, tell the user
                # clearly instead of showing the misleading
                # "No changes from compression" no-op text. The wording
                # distinguishes a confirmed holder from an unconfirmed
                # acquisition failure (describe_compression_lock_skip).
                # The deferred context-engine notification is discarded by
                # the finally block below (finalize committed=False).
                _lock_skipped = getattr(tmp_agent, "_compression_skipped_due_to_lock", None)
                if _lock_skipped is True or isinstance(_lock_skipped, str):
                    from agent.manual_compression_feedback import (
                        describe_compression_lock_skip,
                    )
                    return describe_compression_lock_skip(_lock_skipped)

                if partial and tail:
                    compressed = rejoin_compressed_head_and_tail(compressed, tail)

                # _compress_context either rotated (legacy: ended the old
                # session, created a continuation id — write compressed messages
                # into the NEW session so the original stays searchable) or
                # compacted in place (compression.in_place / #38763: same id,
                # transcript replaced with the compacted set).
                new_session_id = tmp_agent.session_id
                rotated = new_session_id != session_entry.session_id
                _in_place = bool(getattr(tmp_agent, "_last_compaction_in_place", False))

                # Persist the compressed transcript BEFORE repointing the live
                # session onto the new session_id. Order matters: if we
                # repointed first and the canonical DB write then failed (lock
                # contention under concurrent writes, ENOSPC, a disk/IO error),
                # the session entry would already reference a brand-new, empty
                # session_id while the handler still reported success — the
                # user's active conversation would silently vanish from view.
                # Writing first, and treating a write failure as fatal, keeps
                # the old history reachable (on rotation the entry still points
                # at it; in place the original transcript is untouched) and lets
                # the outer handler surface a "compress failed" banner instead.
                #
                # Only rewrite the transcript when rotation produced a NEW
                # session id.  In-place compaction does NOT need a rewrite:
                # archive_and_compact() has already soft-archived the previous
                # active rows and inserted the compacted messages as the new
                # active set inside _compress_context().  Calling
                # rewrite_transcript() after in-place compaction would invoke
                # replace_messages(active_only=False) which DELETEs ALL rows —
                # including the archived turns that archive_and_compact()
                # deliberately preserved (silent data loss, #61145).
                #
                # The third case: _compress_context could NOT rotate AND was
                # not in-place (e.g. legacy mode but _session_db unavailable /
                # the DB split raised) — there session_id is unchanged for a
                # FAILURE reason, and rewrite_transcript() would DELETE the
                # original messages and replace them with only the compressed
                # summary (permanent data loss #44794, #39704).
                if rotated:
                    if not await self.async_session_store.rewrite_transcript(
                        new_session_id, compressed
                    ):
                        raise RuntimeError(
                            f"failed to persist compressed transcript for "
                            f"session {new_session_id}"
                        )
                    session_entry.session_id = new_session_id
                    await self.async_session_store._save()
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compress-command",
                    )
                elif _in_place:
                    # archive_and_compact() already persisted the compacted
                    # transcript inside _compress_context — nothing to do.
                    pass
                else:
                    logger.warning(
                        "Manual /compress: session rotation did not occur "
                        "(session_id unchanged) and in-place mode is off — "
                        "preserving original transcript instead of overwriting "
                        "it (#44794)."
                    )
                # Reset stored token count — transcript changed, old value is stale
                await self.async_session_store.update_session(
                    session_entry.session_key, last_prompt_tokens=0
                )
                finalize_context_engine_compression_notification(
                    tmp_agent,
                    committed=True,
                )
                new_tokens = estimate_request_tokens_rough(
                    compressed, system_prompt=_sys_prompt, tools=_tools
                )
                summary = summarize_manual_compression(
                    msgs,
                    compressed,
                    approx_tokens,
                    new_tokens,
                    compression_state=compressor,
                )
                # Detect summary-generation failure so we can surface a
                # visible warning to the user even on the manual /compress
                # path (otherwise the failure is silently logged).
                # _last_compress_aborted means the aux LLM returned no
                # usable summary and the compressor preserved messages
                # unchanged (no drop, no placeholder).  force=True was
                # passed above so any active cooldown is bypassed.
                _summary_aborted = bool(getattr(compressor, "_last_compress_aborted", False))
                _summary_err = getattr(compressor, "_last_summary_error", None)
                # Force-redact provider exception text at this UI boundary
                # even when global redaction is disabled.
                if _summary_err:
                    from agent.redact import redact_sensitive_text
                    _summary_err = redact_sensitive_text(_summary_err, force=True)
                # Separately: did the user's CONFIGURED aux model fail
                # and we recovered via main?  Surface that as an info
                # note so they can fix their config.
                _aux_fail_model = getattr(compressor, "_last_aux_model_failure_model", None)
                _aux_fail_err = getattr(compressor, "_last_aux_model_failure_error", None)
            finally:
                finalize_context_engine_compression_notification(
                    tmp_agent,
                    committed=False,
                )
                # Evict cached agent so next turn rebuilds system prompt
                # from current files (SOUL.md, memory, etc.).
                self._evict_cached_agent(session_key)
                # Off-loop + bounded: temporary-agent teardown can block on
                # subprocess/network/SQLite work. Running it inline freezes the
                # gateway loop and stalls platform polling / heartbeat, the same
                # wedge class fixed for /new (#35994) and hygiene/shutdown
                # (#53175).
                await self._cleanup_agent_resources_off_loop(
                    tmp_agent, context="manual compression"
                )
            lines = [f"🗜️ {summary['headline']}"]
            if focus_topic:
                lines.append(t("gateway.compress.focus_line", topic=focus_topic))
            lines.append(summary["token_line"])
            if summary["note"]:
                lines.append(summary["note"])
            if _summary_aborted:
                lines.append(
                    t(
                        "gateway.compress.aborted",
                        error=(_summary_err or "unknown error"),
                    )
                )
            elif _aux_fail_model:
                lines.append(
                    t(
                        "gateway.compress.aux_failed",
                        model=_aux_fail_model,
                        error=(_aux_fail_err or "unknown error"),
                    )
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Manual compress failed: %s", e)
            return t("gateway.compress.failed", error=e)
