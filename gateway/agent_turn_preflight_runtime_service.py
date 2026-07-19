"""Production pre-agent preflight for gateway foreground runs.

Promoted from ``GatewayRunner._handle_message_with_agent`` (post-bootstrap):
session hygiene compression, first-contact / home / voice-channel notes,
inbound message preparation + timestamps, and turn-sidecar staging.

Callers must pass a live runner; this module does not reimplement compression
or message enrichment internals.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.config import Platform


def _load_gateway_config() -> dict:
    """Load the same config dict production uses (call-time import avoids cycle)."""
    from gateway.run import _load_gateway_config as _lgc
    return _lgc()


def _home_target_env_var(platform_name: str) -> str:
    from gateway.run import _home_target_env_var as _htev
    return _htev(platform_name)


def _message_timestamps_enabled(user_config: Optional[dict]) -> bool:
    from gateway.run import _message_timestamps_enabled as _mte
    return _mte(user_config)


def _hermes_home_path():
    """Prefer gateway.run._hermes_home so tests monkeypatching it still apply."""
    from gateway import run as _gateway_run
    return getattr(_gateway_run, "_hermes_home")


@dataclass(slots=True)
class GatewayPreparedAgentTurnPreflight:
    """Result of production pre-agent preflight (hygiene → message stage)."""

    history: list = field(default_factory=list)
    turn_sidecar_notes: list[str] = field(default_factory=list)
    message_text: str | None = None
    persist_user_message: str | None = None
    persist_user_timestamp: Any = None
    aborted: bool = False


async def prepare_gateway_agent_turn_preflight(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_entry: Any,
    session_key: str,
    history: list,
    turn_sidecar_notes: list[str] | None,
    quick_key: str,
    run_generation: int,
    logger: Any,
) -> GatewayPreparedAgentTurnPreflight:
    """Run hygiene, sidecar notes, message enrichment, and stage notes.

    Side effects on ``runner`` / ``session_entry`` (hygiene rotation, lease
    rebind, staged sidecar map) match the former inline block in
    ``_handle_message_with_agent``.
    """
    _quick_key = quick_key
    turn_sidecar_notes = list(turn_sidecar_notes or [])
    persist_user_message = None
    persist_user_timestamp = None
    history = list(history) if history is not None else []

    # Session hygiene: auto-compress pathologically large transcripts
    #
    # Long-lived gateway sessions can accumulate enough history that
    # every new message rehydrates an oversized transcript, causing
    # repeated truncation/context failures.  Detect this early and
    # compress proactively — before the agent even starts.  (#628)
    #
    # Token source priority:
    # 1. Actual API-reported prompt_tokens from the last turn
    #    (stored in session_entry.last_prompt_tokens)
    # 2. Rough char-based estimate (str(msg)//4). Overestimates
    #    by 30-50% on code/JSON-heavy sessions, but that just
    #    means hygiene fires a bit early — safe and harmless.
    # -----------------------------------------------------------------
    if history and len(history) >= 4:
        from agent.model_metadata import (
            estimate_messages_tokens_rough,
            get_model_context_length_async,
        )

        # Read model + compression config from config.yaml.
        # NOTE: hygiene threshold is intentionally HIGHER than the agent's
        # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
        # sessions that grew too large between turns — it fires pre-agent
        # to prevent API failures.  The agent's own compressor handles
        # normal context management during its tool loop with accurate
        # real token counts.  Having hygiene at 0.50 caused premature
        # compression on every turn in long gateway sessions.
        _hyg_model = "anthropic/claude-sonnet-4.6"
        _hyg_threshold_pct = 0.85
        _hyg_compression_enabled = True
        _hyg_hard_msg_limit = 5000
        _hyg_config_context_length = None
        _hyg_provider = None
        _hyg_base_url = None
        _hyg_api_key = None
        _hyg_data = {}
        try:
            _hyg_data = _load_gateway_config()
            if _hyg_data:
                # Resolve model name (same logic as run_sync)
                _model_cfg = _hyg_data.get("model", {})
                if isinstance(_model_cfg, str):
                    _hyg_model = _model_cfg
                elif isinstance(_model_cfg, dict):
                    _hyg_model = _model_cfg.get("default") or _model_cfg.get("model") or _hyg_model
                    # Read explicit context_length override from model config
                    # (same as run_agent.py lines 995-1005)
                    _raw_ctx = _model_cfg.get("context_length")
                    if _raw_ctx is not None:
                        try:
                            _hyg_config_context_length = int(_raw_ctx)
                        except (TypeError, ValueError):
                            pass
                    # Read provider for accurate context detection
                    _hyg_provider = _model_cfg.get("provider") or None
                    _hyg_base_url = _model_cfg.get("base_url") or None

                # Read compression settings — only use enabled flag.
                # The threshold is intentionally separate from the agent's
                # compression.threshold (hygiene runs higher).
                _comp_cfg = _hyg_data.get("compression", {})
                if isinstance(_comp_cfg, dict):
                    _hyg_compression_enabled = str(
                        _comp_cfg.get("enabled", True)
                    ).lower() in {"true", "1", "yes"}
                    _raw_hard_limit = _comp_cfg.get("hygiene_hard_message_limit")
                    if _raw_hard_limit is not None:
                        try:
                            _parsed = int(_raw_hard_limit)
                            if _parsed > 0:
                                _hyg_hard_msg_limit = _parsed
                        except (TypeError, ValueError):
                            pass

            try:
                _hyg_model, _hyg_runtime = runner._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                    user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                )
                _hyg_provider = _hyg_runtime.get("provider") or _hyg_provider
                _hyg_base_url = _hyg_runtime.get("base_url") or _hyg_base_url
                _hyg_api_key = _hyg_runtime.get("api_key") or _hyg_api_key
            except Exception:
                pass

            # Check custom_providers per-model context_length
            # (same fallback as run_agent.py lines 1171-1189).
            # Must run after runtime resolution so _hyg_base_url is set.
            if _hyg_config_context_length is None and _hyg_base_url:
                try:
                    try:
                        from hermes_cli.config import get_compatible_custom_providers as _gw_gcp
                        _hyg_custom_providers = _gw_gcp(_hyg_data)
                    except Exception:
                        _hyg_custom_providers = _hyg_data.get("custom_providers")
                        if not isinstance(_hyg_custom_providers, list):
                            _hyg_custom_providers = []
                    for _cp in _hyg_custom_providers:
                        if not isinstance(_cp, dict):
                            continue
                        _cp_url = (_cp.get("base_url") or "").rstrip("/")
                        if _cp_url and _cp_url == _hyg_base_url.rstrip("/"):
                            _cp_models = _cp.get("models", {})
                            if isinstance(_cp_models, dict):
                                _cp_model_cfg = _cp_models.get(_hyg_model, {})
                                if isinstance(_cp_model_cfg, dict):
                                    _cp_ctx = _cp_model_cfg.get("context_length")
                                    if _cp_ctx is not None:
                                        _hyg_config_context_length = int(_cp_ctx)
                            break
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

        if _hyg_compression_enabled:
            _hyg_context_length = await get_model_context_length_async(
                _hyg_model,
                base_url=_hyg_base_url or "",
                api_key=_hyg_api_key or "",
                config_context_length=_hyg_config_context_length,
                provider=_hyg_provider or "",
            )
            _compress_token_threshold = int(
                _hyg_context_length * _hyg_threshold_pct
            )
            _warn_token_threshold = int(_hyg_context_length * 0.95)

            _msg_count = len(history)

            # Prefer actual API-reported tokens from the last turn
            # (stored in session entry) over the rough char-based estimate.
            _stored_tokens = session_entry.last_prompt_tokens
            if _stored_tokens > 0:
                _approx_tokens = _stored_tokens
                _token_source = "actual"
            else:
                _approx_tokens = estimate_messages_tokens_rough(history)
                _token_source = "estimated"
                # Note: rough estimates overestimate by 30-50% for code/JSON-heavy
                # sessions, but that just means hygiene fires a bit early — which
                # is safe and harmless.  The 85% threshold already provides ample
                # headroom (agent's own compressor runs at 50%).  A previous 1.4x
                # multiplier tried to compensate by inflating the threshold, but
                # 85% * 1.4 = 119% of context — which exceeds the model's limit
                # and prevented hygiene from ever firing for ~200K models (GLM-5).

            # Hard safety valve: force compression if message count is
            # extreme, regardless of token estimates.  This breaks the
            # death spiral where API disconnects prevent token data
            # collection, which prevents compression, which causes more
            # disconnects.  5000 messages is far above any normal session
            # but catches truly runaway growth before it becomes
            # unrecoverable.  Set well clear of legitimate large-context
            # (1M+) sessions doing thousands of short turns — those
            # compress on the token threshold, not this count-based floor.
            # Threshold is configurable via
            # compression.hygiene_hard_message_limit.
            # (#2153)
            _HARD_MSG_LIMIT = _hyg_hard_msg_limit
            _needs_compress = (
                _approx_tokens >= _compress_token_threshold
                or _msg_count >= _HARD_MSG_LIMIT
            )

            if _needs_compress:
                logger.info(
                    "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                    "(threshold: %s%% of %s = %s tokens)",
                    _msg_count, f"{_approx_tokens:,}", _token_source,
                    int(_hyg_threshold_pct * 100),
                    f"{_hyg_context_length:,}",
                    f"{_compress_token_threshold:,}",
                )

                _hyg_meta = runner._thread_metadata_for_source(source, runner._reply_anchor_for_event(event))

                try:
                    from run_agent import AIAgent

                    _hyg_model, _hyg_runtime = runner._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                    )
                    if _hyg_runtime.get("api_key"):
                        # Pass the FULL transcript (tool results included).
                        # Filtering to user/assistant-only starved the
                        # compressor: tool results are usually the bulk of
                        # the context, _prune_old_tool_results never saw
                        # them, and short filtered histories tripped the
                        # protect-first/last early-return so nothing was
                        # compressed at all (#3854). The agent loop passes
                        # its full message list to _compress_context — the
                        # gateway now matches.
                        _hyg_msgs = [
                            m for m in history
                            if m.get("role") in {"user", "assistant", "tool"}
                        ]

                        if len(_hyg_msgs) >= 4:
                            _hyg_session_db = getattr(runner._session_db, "_db", runner._session_db)
                            _hyg_agent = AIAgent(
                                **_hyg_runtime,
                                model=_hyg_model,
                                max_iterations=4,
                                quiet_mode=True,
                                skip_memory=True,
                                enabled_toolsets=["memory"],
                                session_id=session_entry.session_id,
                                session_db=_hyg_session_db,
                            )
                            try:
                                # Gateway hygiene runs before the user turn
                                # starts and already owns the session binding.
                                # Prefer in-place compaction here: it archives
                                # old rows under the same session id instead of
                                # minting a continuation child that then has to
                                # be published back to SessionStore/topic
                                # bindings.  If no SessionDB is available,
                                # compress_context leaves this flag false and
                                # the guard below preserves the transcript.
                                _hyg_agent.compression_in_place = True
                                _bind_hyg_state = getattr(
                                    getattr(_hyg_agent, "context_compressor", None),
                                    "bind_session_state",
                                    None,
                                )
                                if callable(_bind_hyg_state):
                                    _bind_hyg_state(
                                        _hyg_session_db,
                                        session_entry.session_id,
                                    )
                                # It must never finalize on close() — close()
                                # would end the live gateway session row.
                                _hyg_agent._end_session_on_close = False
                                _hyg_agent._print_fn = lambda *a, **kw: None

                                loop = asyncio.get_running_loop()
                                _compressed, _ = await loop.run_in_executor(
                                    None,
                                    lambda: _hyg_agent._compress_context(
                                        _hyg_msgs, "",
                                        approx_tokens=_approx_tokens,
                                    ),
                                )

                                # _compress_context ends the old session and creates
                                # a new session_id.  Write compressed messages into
                                # the NEW session so the old transcript stays intact
                                # and searchable via session_search.
                                _hyg_new_sid = _hyg_agent.session_id
                                _hyg_rotated = _hyg_new_sid != session_entry.session_id
                                _hyg_in_place = bool(
                                    getattr(_hyg_agent, "_last_compaction_in_place", False)
                                )
                                if _hyg_rotated:
                                    session_entry.session_id = _hyg_new_sid
                                    # The held turn lease follows the
                                    # rotation so an alias key resolving
                                    # the fresh child still serializes
                                    # against this turn (#64934).
                                    runner._rebind_turn_lease(
                                        _quick_key, run_generation, _hyg_new_sid
                                    )
                                    await runner.async_session_store._save()
                                    await asyncio.to_thread(
                                        runner._sync_telegram_topic_binding,
                                        source, session_entry,
                                        reason="hygiene-compression",
                                    )

                                # Only rewrite the transcript when rotation produced
                                # a NEW session id.  In-place compaction does NOT
                                # need a rewrite: archive_and_compact() has already
                                # soft-archived the previous active rows and inserted
                                # the compacted messages as the new active set inside
                                # _compress_context().  Calling rewrite_transcript()
                                # after in-place compaction would invoke
                                # replace_messages(active_only=False) which DELETEs
                                # ALL rows — including the archived turns that
                                # archive_and_compact() deliberately preserved
                                # (silent data loss, #61145).
                                #
                                # The danger this guards against (mirrors the
                                # /compress fix #44794/#39704): if _compress_context
                                # returns a summary but neither rotates nor completes
                                # archive_and_compact(), the session_id is unchanged
                                # for a FAILURE reason, and an unconditional
                                # rewrite_transcript() would DELETE the original
                                # messages and replace them with only the compressed
                                # summary (permanent data loss, #21301).
                                if _hyg_rotated:
                                    await runner.async_session_store.rewrite_transcript(
                                        session_entry.session_id, _compressed
                                    )
                                    # Reset stored token count — transcript rewritten
                                    session_entry.last_prompt_tokens = 0
                                    history = _compressed
                                    _new_count = len(_compressed)
                                    _new_tokens = estimate_messages_tokens_rough(
                                        _compressed
                                    )
                                elif _hyg_in_place:
                                    # archive_and_compact() already persisted the
                                    # compacted transcript inside _compress_context.
                                    # Reset counts to match the new active set.
                                    session_entry.last_prompt_tokens = 0
                                    history = _compressed
                                    _new_count = len(_compressed)
                                    _new_tokens = estimate_messages_tokens_rough(
                                        _compressed
                                    )
                                else:
                                    # No rewrite happened — transcript preserved
                                    # unchanged, so the post-compression counts equal
                                    # the pre-compression ones.
                                    _new_count = _msg_count
                                    _new_tokens = _approx_tokens
                                    logger.warning(
                                        "Gateway hygiene compression for session %s "
                                        "did not rotate or compact in place "
                                        "(no session_db on the hygiene agent) — "
                                        "preserving the original transcript instead "
                                        "of overwriting it with the summary (#21301).",
                                        session_entry.session_id,
                                    )

                                logger.info(
                                    "Session hygiene: compressed %s → %s msgs, "
                                    "~%s → ~%s tokens",
                                    _msg_count, _new_count,
                                    f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                )

                                if _new_tokens >= _warn_token_threshold:
                                    logger.warning(
                                        "Session hygiene: still ~%s tokens after "
                                        "compression",
                                        f"{_new_tokens:,}",
                                    )

                                # If summary generation failed, the
                                # compressor aborts entirely and returns
                                # messages unchanged — nothing is dropped.
                                # Surface a visible warning to the gateway
                                # user — agent.log alone is invisible on
                                # TG/Discord/etc. — so they know the chat
                                # is "frozen" at the current size and can
                                # /compress to retry or /reset to start
                                # fresh.
                                _comp = getattr(_hyg_agent, "context_compressor", None)
                                if _comp is not None and getattr(_comp, "_last_compress_aborted", False):
                                    _err = getattr(_comp, "_last_summary_error", None) or "unknown error"
                                    # Force-redact: provider exception text
                                    # may contain credentials; this message
                                    # reaches gateway users directly.
                                    from agent.redact import redact_sensitive_text
                                    _err = redact_sensitive_text(_err, force=True)
                                    _warn_msg = (
                                        "⚠️ Context compression aborted "
                                        f"({_err}). No messages were dropped — "
                                        "conversation is unchanged. Run /compress "
                                        "to retry, /reset for a clean session, or "
                                        "check your auxiliary.compression model "
                                        "configuration."
                                    )
                                    try:
                                        _adapter = runner._adapter_for_source(source)
                                        if _adapter and source.chat_id:
                                            await _adapter.send(source.chat_id, _warn_msg, metadata=_hyg_meta)
                                    except Exception as _werr:
                                        logger.warning(
                                            "Failed to deliver compression-failure warning to user: %s",
                                            _werr,
                                        )
                                # Separately: if the user's CONFIGURED aux
                                # model failed and we recovered by falling
                                # back to the main model, tell them — a
                                # misconfigured auxiliary.compression.model
                                # is something only they can fix, and
                                # silent recovery would hide it.
                                elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
                                    _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
                                    _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
                                    _aux_msg = (
                                        f"ℹ️ Configured compression model `{_aux_model}` "
                                        f"failed ({_aux_err}). Recovered using your main "
                                        "model — context is intact — but you may want to "
                                        "check `auxiliary.compression.model` in config.yaml."
                                    )
                                    try:
                                        _adapter = runner._adapter_for_source(source)
                                        if _adapter and source.chat_id:
                                            await _adapter.send(source.chat_id, _aux_msg, metadata=_hyg_meta)
                                    except Exception as _werr:
                                        logger.warning(
                                            "Failed to deliver aux-model-fallback notice to user: %s",
                                            _werr,
                                        )
                            finally:
                                # Evict the cached agent so the next turn
                                # rebuilds its system prompt from current
                                # SOUL.md, memory, and skills.
                                runner._evict_cached_agent(session_key)
                                await runner._cleanup_agent_resources_off_loop(
                                    _hyg_agent, context="session hygiene"
                                )

                except Exception as e:
                    logger.warning(
                        "Session hygiene auto-compress failed: %s", e
                    )

    # First-message onboarding -- only on the very first interaction ever.
    # Delivered on the current user message (sidecar), NOT the ephemeral
    # system prompt: present-on-turn-1/absent-on-turn-2 was a guaranteed
    # system-prompt diff and agent rebuild.
    if not history and not await runner.async_session_store.has_any_sessions():
        # Default first-contact note: a brief self-introduction.
        _intro_note = (
            "[System note: This is the user's very first message ever. "
            "Briefly introduce yourself and mention that /help shows available commands. "
            "Keep the introduction concise -- one or two sentences max.]"
        )
        # Opt-in structured profile-build path. When enabled (default
        # "ask") and not yet offered on this install, swap the plain intro
        # for a consent-gated directive that offers to build a user
        # profile and persists confirmed facts via memory(target="user").
        # The offer fires at most once (onboarding.seen flag); set
        # onboarding.profile_build: off in config.yaml to disable.
        try:
            from agent.onboarding import (
                PROFILE_BUILD_FLAG,
                is_seen,
                mark_seen,
                profile_build_directive,
                profile_build_mode,
            )
            _onb_cfg = _load_gateway_config()
            if (
                profile_build_mode(_onb_cfg) == "ask"
                and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG)
            ):
                turn_sidecar_notes.append(profile_build_directive().strip())
                mark_seen(_hermes_home_path() / "config.yaml", PROFILE_BUILD_FLAG)
            else:
                turn_sidecar_notes.append(_intro_note)
        except Exception as _pb_err:
            logger.debug(
                "Profile-build onboarding directive failed, using plain intro: %s",
                _pb_err,
            )
            turn_sidecar_notes.append(_intro_note)

    # One-time prompt if no home channel is set for this platform
    # Skip for webhooks - they deliver directly to configured targets (github_comment, etc.)
    if not history and source.platform and source.platform != Platform.LOCAL and source.platform != Platform.WEBHOOK:
        platform_name = source.platform.value
        env_key = _home_target_env_var(platform_name)
        # Multiplex: home channel may live only in the profile secret
        # scope / PlatformConfig, not process os.environ.
        home_env = ""
        try:
            from agent.secret_scope import get_secret

            home_env = (get_secret(env_key) or "").strip() if env_key else ""
        except Exception:
            home_env = ""
        if not home_env:
            home_env = (os.getenv(env_key) or "").strip() if env_key else ""
        # Also honor in-memory / yaml home_channel on this platform.
        try:
            if not home_env and runner.config.get_home_channel(source.platform):
                home_env = "set"
        except Exception:
            pass
        # Secondary-profile platforms (e.g. Slack on yolo) may only exist
        # under that profile's loaded config — check after scope install.
        if not home_env:
            try:
                from hermes_cli.profiles import get_profile_dir
                from gateway.config import load_gateway_config as _lgc
                prof = (getattr(source, "profile", None) or "").strip()
                if prof and prof != "default":
                    # Already inside profile scope for secondary handlers;
                    # re-read live config for home_channel.
                    _pcfg = _lgc()
                    if _pcfg.get_home_channel(source.platform):
                        home_env = "set"
            except Exception:
                pass
        if not home_env:
            # Slack dispatches all Hermes commands through a single
            # parent slash command `/hermes`; bare `/sethome` is not
            # registered and would fail with "app did not respond".
            sethome_cmd = (
                "/hermes sethome"
                if source.platform == Platform.SLACK
                else "/sethome"
            )
            notice = (
                f"📬 No home channel is set for {platform_name.title()}. "
                f"A home channel is where Hermes delivers cron job results "
                f"and cross-platform messages.\n\n"
                f"Type {sethome_cmd} to make this chat your home channel, "
                f"or ignore to skip."
            )
            await runner._deliver_platform_notice(source, notice)

    # -----------------------------------------------------------------
    # Voice channel awareness — deliver current voice channel state so
    # the agent knows who is in the channel and who is speaking, without
    # needing a separate tool call.  Delivered on the current user
    # message and ONLY when it changed since the previous turn: the
    # member/speaking serialization differs essentially every turn, and
    # appending it to the ephemeral system prompt forced a full agent
    # rebuild + prompt-cache re-key per message.  The system prompt
    # carries a static pointer line instead (gateway/session.py).
    # -----------------------------------------------------------------
    _vc_note = runner._voice_channel_sidecar_note(event, source, session_key)
    if _vc_note:
        turn_sidecar_notes.append(_vc_note)

    # -----------------------------------------------------------------
    # Auto-analyze images sent by the user
    #
    # If the user attached image(s), we run the vision tool eagerly so
    # the conversation model always receives a text description.  The
    # local file path is also included so the model can re-examine the
    # image later with a more targeted question via vision_analyze.
    #
    # We filter to image paths only (by media_type) so that non-image
    # attachments (documents, audio, etc.) are not sent to the vision
    # tool even when they appear in the same message.
    # -----------------------------------------------------------------
    message_text = await runner._prepare_profile_scoped_inbound_message_text(
        event=event,
        source=source,
        history=history,
        session_key=session_key,
    )
    if message_text is None:
        return GatewayPreparedAgentTurnPreflight(
            history=history,
            turn_sidecar_notes=list(turn_sidecar_notes),
            message_text=None,
            persist_user_message=persist_user_message,
            persist_user_timestamp=persist_user_timestamp,
            aborted=True,
        )

    # Capture the platform event time as message metadata and keep the
    # persisted transcript clean (strip any leading timestamp prefix).
    # This runs regardless of the toggle so storage stays clean and the
    # send-time is preserved. Only the in-context RENDER (prepending the
    # human-readable prefix the model sees) is gated behind
    # gateway.message_timestamps.enabled — default OFF.
    try:
        from hermes_time import get_timezone as _get_evt_tz
        from gateway.message_timestamps import (
            coerce_message_timestamp as _coerce_msg_ts,
            render_user_content_with_timestamp as _render_msg_ts,
            strip_leading_message_timestamps as _strip_msg_ts,
        )
        _evt_tz = _get_evt_tz()
        _evt_ts = getattr(event, "timestamp", None)
        if message_text and isinstance(message_text, str):
            _clean_message_text, _embedded_ts = _strip_msg_ts(
                message_text, tz=_evt_tz)
            persist_user_message = _clean_message_text
            _event_epoch = _coerce_msg_ts(_evt_ts, tz=_evt_tz)
            persist_user_timestamp = (
                _event_epoch if _event_epoch is not None else _embedded_ts
            )
            if _message_timestamps_enabled(_load_gateway_config()):
                message_text = _render_msg_ts(
                    _clean_message_text,
                    persist_user_timestamp,
                    tz=_evt_tz,
                )
            else:
                # Toggle off: model sees the clean message; the timestamp
                # is still stored as metadata for later opt-in.
                message_text = _clean_message_text
    except Exception as _ts_err:
        logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)

    # Stage the collected must-deliver notes for this turn's agent run
    # (one-shot; consumed in run_sync).  Staged AFTER the message_text
    # early-out above so an aborted turn cannot leak its notes into the
    # next turn's user message.
    if turn_sidecar_notes and session_key:
        runner._set_pending_turn_sidecar_notes(session_key, turn_sidecar_notes)

    return GatewayPreparedAgentTurnPreflight(
        history=history,
        turn_sidecar_notes=list(turn_sidecar_notes),
        message_text=message_text,
        persist_user_message=persist_user_message,
        persist_user_timestamp=persist_user_timestamp,
        aborted=False,
    )
