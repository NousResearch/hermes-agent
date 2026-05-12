"""CompressionMixin -- context compression methods extracted from AIAgent.

Encapsulates compression model feasibility checking, warning replay,
and context compression.  Previously inline in AIAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import os

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class CompressionMixin:
    """Mixin providing context compression methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    def _check_compression_model_feasibility(self) -> None:
        """Warn at session start if the auxiliary compression model's context
        window is smaller than the main model's compression threshold.

        When the auxiliary model cannot fit the content that needs summarising,
        compression will either fail outright (the LLM call errors) or produce
        a severely truncated summary.

        Called during ``__init__`` so CLI users see the warning immediately
        (via ``_vprint``).  The gateway sets ``status_callback`` *after*
        construction, so ``_replay_compression_warning()`` re-sends the
        stored warning through the callback on the first
        ``run_conversation()`` call.
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import (
                _resolve_task_provider_model,
                get_text_auxiliary_client,
            )
            from agent.model_metadata import (
                MINIMUM_CONTEXT_LENGTH,
                get_model_context_length,
            )

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            # Best-effort aux provider label for the warning message. The
            # configured provider may be "auto", in which case we fall back
            # to the client's base_url hostname so the user can still tell
            # where the compression model is actually being called.
            try:
                _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
            except Exception:
                _aux_cfg_provider = ""
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=getattr(self, "_aux_compression_context_length_config", None),
                # Each model must be resolved with its own provider so that
                # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
                # are invoked for the correct client, not inherited from the main model.
                provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(self, "provider", "")),
            )

            # Hard floor: the auxiliary compression model must have at least
            # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
            # is already required to meet this floor (checked earlier in
            # __init__), so the compression model must too — otherwise it
            # cannot summarise a full threshold-sized window of main-model
            # content.  Mirrors the main-model rejection pattern.
            if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
                raise ValueError(
                    f"Auxiliary compression model {aux_model} has a context "
                    f"window of {aux_context:,} tokens, which is below the "
                    f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                    f"Agent.  Choose a compression model with at least "
                    f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                    f"auxiliary.compression.model in config.yaml), or set "
                    f"auxiliary.compression.context_length to override the "
                    f"detected value if it is wrong."
                )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # Auto-correct: lower the live session threshold so
                # compression actually works this session.  The hard floor
                # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
                # so the new threshold is always >= 64K.
                #
                # The compression summariser sends a single user-role
                # prompt (no system prompt, no tools) to the aux model, so
                # new_threshold == aux_context is safe: the request is
                # the raw messages plus a small summarisation instruction.
                old_threshold = threshold
                new_threshold = aux_context
                self.context_compressor.threshold_tokens = new_threshold
                # Keep threshold_percent in sync so future main-model
                # context_length changes (update_model) re-derive from a
                # sensible number rather than the original too-high value.
                main_ctx = self.context_compressor.context_length
                if main_ctx:
                    self.context_compressor.threshold_percent = (
                        new_threshold / main_ctx
                    )
                safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
                # Build human-readable "model (provider)" labels for both
                # the main model and the compression model so users can
                # tell at a glance which provider each side is actually
                # using. When the configured provider is empty or "auto",
                # fall back to the client's base_url hostname.
                _main_model = getattr(self, "model", "") or "?"
                _main_provider = getattr(self, "provider", "") or ""
                _aux_provider_label = (
                    _aux_cfg_provider
                    if _aux_cfg_provider and _aux_cfg_provider != "auto"
                    else ""
                )
                if not _aux_provider_label:
                    try:
                        from urllib.parse import urlparse
                        _aux_provider_label = (
                            urlparse(aux_base_url).hostname or aux_base_url
                        )
                    except Exception:
                        _aux_provider_label = aux_base_url or "auto"
                _main_label = (
                    f"{_main_model} ({_main_provider})"
                    if _main_provider
                    else _main_model
                )
                _aux_label = f"{aux_model} ({_aux_provider_label})"
                msg = (
                    f"⚠ Compression model {_aux_label} context is "
                    f"{aux_context:,} tokens, but the main model "
                    f"{_main_label}'s compression threshold was "
                    f"{old_threshold:,} tokens. "
                    f"Auto-lowered this session's threshold to "
                    f"{new_threshold:,} tokens so compression can run.\n"
                    f"  To make this permanent, edit config.yaml — either:\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — auto-lowered session threshold to %d to "
                    "keep compression working.",
                    aux_model,
                    aux_context,
                    old_threshold,
                    new_threshold,
                )
        except ValueError:
            # Hard rejections (aux below minimum context) must propagate
            # so the session refuses to start.
            raise
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )


    def _replay_compression_warning(self) -> None:
        """Re-send the compression warning through ``status_callback``.

        During ``__init__`` the gateway's ``status_callback`` is not yet
        wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
        method is called once at the start of the first
        ``run_conversation()`` — by then the gateway has set the callback,
        so every platform (Telegram, Discord, Slack, etc.) receives the
        warning.
        """
        msg = getattr(self, "_compression_warning", None)
        if msg and self.status_callback:
            try:
                self.status_callback("lifecycle", msg)
            except Exception:
                pass


    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """Compress conversation context and split the session in SQLite.

        Args:
            focus_topic: Optional focus string for guided compression — the
                summariser will prioritise preserving information related to
                this topic.  Inspired by Claude Code's ``/compact <focus>``.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )

        # Notify external memory provider before compression discards context
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        try:
            compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)
        except TypeError:
            # Plugin context engine with strict signature that doesn't accept
            # focus_topic — fall back to calling without it.
            compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)

        summary_error = getattr(self.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(self, "_last_compression_summary_warning", None) != summary_error:
                self._last_compression_summary_warning = summary_error
                self._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # No hard failure — but did the configured aux model error out
            # and get recovered by retrying on main?  Surface that so users
            # know their auxiliary.compression.model setting is broken even
            # though compression succeeded.
            _aux_fail_model = getattr(self.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(self.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(self, "_last_aux_fallback_warning_key", None) != _aux_key:
                    self._last_aux_fallback_warning_key = _aux_key
                    self._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                # Trigger memory extraction on the old session before it rotates.
                self.commit_memory_session(messages)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db_created = False
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config=self._session_init_model_config,
                    parent_session_id=old_session_id,
                )
                self._session_db_created = True
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Notify the context engine that the session_id rotated because of
        # compression (not a fresh /new). Plugin engines (e.g. hermes-lcm) use
        # boundary_reason="compression" to preserve DAG lineage across the
        # rollover instead of re-initializing fresh per-session state.
        # See hermes-lcm#68. Built-in ContextCompressor ignores kwargs.
        try:
            _old_sid = locals().get("old_session_id")
            if _old_sid and hasattr(self.context_compressor, "on_session_start"):
                self.context_compressor.on_session_start(
                    self.session_id or "",
                    boundary_reason="compression",
                    old_session_id=_old_sid,
                )
        except Exception as _ce_err:
            logger.debug("context engine on_session_start (compression): %s", _ce_err)

        # Notify memory providers of the compression-driven session_id rotation
        # so provider-cached per-session state (Hindsight's _document_id,
        # accumulated turn buffers, counters) refreshes. reset=False because
        # the logical conversation continues; only the id and DB row rolled
        # over. See #6672.
        try:
            _old_sid = locals().get("old_session_id")
            if _old_sid and self._memory_manager:
                self._memory_manager.on_session_switch(
                    self.session_id or "",
                    parent_session_id=_old_sid,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # Warn on repeated compressions (quality degrades with each pass)
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # Update token estimate after compaction so pressure calculations
        # use the post-compression count, not the stale pre-compression one.
        # Use estimate_request_tokens_rough() so tool schemas are included —
        # with 50+ tools enabled, schemas alone can add 20-30K tokens, and
        # omitting them delays the next compression cycle far past the
        # configured threshold (issue #14695).
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=self.tools or None,
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt
