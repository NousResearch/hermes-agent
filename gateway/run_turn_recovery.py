"""Durable compression recovery and turn-progress routing."""
from __future__ import annotations

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Optional
from gateway.config import Platform
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")

class GatewayTurnRecoveryMixin:
    @staticmethod
    async def _await_session_policy_commit(awaitable):
        """Keep admission owned until a storage worker and its publication tail settle."""
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    logger.warning("Session policy worker failed during cancellation", exc_info=True)
                    break
            try:
                task.result()
            except BaseException:
                logger.warning("Session policy write failed during cancellation", exc_info=True)
            raise

    async def _compression_exhaustion_action(self, source) -> Optional[str]:
        """Read only the explicit policy leaf; uncertainty is never permission to clear a pause."""
        def read():
            from gateway.run import _gateway_config_home
            from hermes_cli.config import _expand_env_vars
            from hermes_cli.managed_scope import get_managed_dir

            home = (self._resolve_profile_home_for_source(source)
                    if getattr(self.config, "multiplex_profiles", False) else _gateway_config_home())
            if home is None:
                raise RuntimeError("Source profile home is unavailable")
            managed = get_managed_dir()
            paths = [Path(home) / "config.yaml"]
            if managed is not None:
                paths.append(managed / "config.yaml")
            choice = None
            for path in paths:
                try:
                    raw = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    continue
                layer = yaml.safe_load(raw)
                if layer is None:
                    continue
                if not isinstance(layer, dict):
                    raise ValueError("Configuration must be a mapping")
                if "compression" not in layer:
                    continue
                compression = layer["compression"]
                if not isinstance(compression, dict):
                    raise ValueError("compression must be a mapping")
                if "exhaustion_action" in compression:
                    choice = compression["exhaustion_action"]
            # The adapter installs the source profile scope; to_thread preserves it.
            # Expand only the winning leaf after both layers pass the strict guards.
            if isinstance(choice, str):
                choice = _expand_env_vars(choice)
            return choice if isinstance(choice, str) and choice in {"reset", "pause"} else None
        try:
            return await asyncio.to_thread(read)
        except (OSError, ValueError, RuntimeError, ImportError, TypeError, yaml.YAMLError):
            logger.warning("Could not establish explicit compression.exhaustion_action; existing pause retained", exc_info=True)
            return None

    @staticmethod
    def _compression_pause_notice(*, persistence_failed=False):
        if persistence_failed:
            return ("Compression recovery could not be persisted. Retry /compress or /new after fixing "
                    "storage, or /resume an authorized session. An initial pause held only in memory "
                    "cannot be guaranteed after a restart while storage is unavailable.")
        return ("Session paused: the conversation could not be compressed further. History is preserved. "
                "Use /compress, /new, or /resume another authorized session. Setting "
                "compression.exhaustion_action: reset explicitly resets this pause on your next message.")

    def _sync_compression_recovery_binding(self, source, entry, expected_id, *, reason):
        """Do not let a settled old tail rebind a route replaced by /new or /resume."""
        with self.session_store._lock:
            current = self.session_store._entries.get(entry.session_key)
            if current is entry and current.session_id == expected_id:
                self._sync_telegram_topic_binding(source, entry, reason=reason)

    async def _reset_session_after_compression_exhaustion(
        self, session_key, session_entry, source, *, require_primary=True, event=None,
    ):
        expected_id = session_entry.session_id
        async def commit():
            new_entry = await self.async_session_store.reset_session(
                session_key, require_primary=require_primary, expected_session_id=expected_id,
            )
            if new_entry is None:
                raise RuntimeError("Session route changed before compression recovery")
            from gateway.session_automation import move_route_wakes
            await move_route_wakes(self, session_key, expected_id)
            self._evict_cached_agent(session_key)
            queued_events = self._overflow_queue(session_key)
            self._clear_conversation_scope(session_key, reason="compression_exhausted_reset")
            if queued_events:
                self._session_state(session_key).conversation.queued_events = queued_events
            try:
                await asyncio.to_thread(
                    self._sync_compression_recovery_binding, source, new_entry, new_entry.session_id,
                    reason="compression-exhausted-reset",
                )
            except Exception:
                logger.warning("Compression recovery committed but topic binding sync failed", exc_info=True)
            if event is not None:
                event._gateway_preserve_pending = False
            return new_entry
        return await self._await_session_policy_commit(commit())

    def _run_agent_progress_threading(
        self, source: SessionSource, event_message_id: Optional[str], _native_slack_task_cards: bool
    ) -> tuple[Optional[dict], Optional[str], Optional[dict]]:
        """Resolve where progress bubbles are threaded (platform-specific).

        Returns ``(progress_metadata, progress_reply_to, status_thread_metadata)``; the latter is
        for status / approval / stream sends (Feishu topics need the triggering message id via the
        reply API, so carry it as a fallback). Slack and Buzz honour the user's reply_in_thread
        opt-out: never synthesise a thread for progress, or every later reply inherits it."""
        from gateway.run import _non_conversational_metadata, _resolve_progress_thread_id
        is_buzz = str(getattr(source.platform, "value", source.platform) or "").lower() == "buzz"
        _progress_reply_in_thread = True
        _adapter = self._delivery_adapter_for(source) if source.platform == Platform.SLACK or is_buzz else None
        if _adapter is not None:
            try:
                if is_buzz:
                    _progress_reply_in_thread = getattr(_adapter, "_reply_to_mode", "first") != "off"
                else:
                    # Relay lane: the adapter owns mode resolution; native lane: flat extra key.
                    _mode_fn = getattr(_adapter, "_effective_reply_in_thread", None)
                    _progress_reply_in_thread = bool(
                        _mode_fn() if callable(_mode_fn) else _adapter.config.extra.get("reply_in_thread", True)
                    )
            except Exception:
                _progress_reply_in_thread = True
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id, reply_in_thread=_progress_reply_in_thread,
        )
        # Relay Discord auto-thread lane: the connector stamps prospective_thread_id at ingest.
        _relay_prospective_thread_id = (
            str(getattr(source, "prospective_thread_id", None))
            if source.platform == Platform.DISCORD
            and getattr(source, "delivered_via_upstream_relay", False)
            and getattr(source, "prospective_thread_id", None)
            and not source.thread_id
            else None
        )
        _progress_metadata = _non_conversational_metadata(
            self._thread_metadata_for_progress(
                source, event_message_id, _progress_thread_id, _relay_prospective_thread_id,
            ),
            platform=source.platform,
        )
        if _native_slack_task_cards:
            # chat.startStream in channels requires the recipient team/user pair; harmless elsewhere.
            _progress_metadata = dict(_progress_metadata or {})
            if source.scope_id:
                _progress_metadata.setdefault("recipient_team_id", source.scope_id)
                _progress_metadata.setdefault("slack_team_id", source.scope_id)
            if source.user_id:
                _progress_metadata.setdefault("recipient_user_id", source.user_id)
        # Buzz has no native thread_id: thread via reply-to unless the user opted out.
        _progress_reply_to = (
            event_message_id
            if (source.platform in (Platform.FEISHU, Platform.MATTERMOST) and source.thread_id and event_message_id)
            or (is_buzz and event_message_id and _progress_reply_in_thread)
            or _relay_prospective_thread_id
            else None
        )
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            _status_thread_metadata = {"thread_id": _progress_thread_id, "reply_to_message_id": event_message_id}
        else:
            _status_thread_metadata = self._thread_metadata_for_progress(
                source, event_message_id, _progress_thread_id, _relay_prospective_thread_id,
            )
        return _progress_metadata, _progress_reply_to, _status_thread_metadata
