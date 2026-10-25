"""Progress-lane threading for gateway turns; inherited by ``GatewayTurnMixin``.

Split out of ``gateway/run_turn.py`` (FILE_LINES ratchet): these two methods only resolve WHERE
progress / status bubbles are threaded. They use only helper methods the mixin chain already
provides (``_thread_metadata_for_source``, ``_thread_metadata_for_target``,
``_delivery_adapter_for``), so they compose as a mixin without new state.
"""

from __future__ import annotations

from typing import Any, Optional

from gateway.config import Platform
from gateway.session import SessionSource


class GatewayTurnProgressThreadingMixin:
    def _thread_metadata_for_progress(
        self, source: SessionSource, event_message_id: Optional[str], _progress_thread_id: Any,
        _relay_prospective_thread_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Thread metadata for a progress-lane send; relay Discord auto-thread lane falls back to the reply anchor.

        The connector will auto-thread on the reply anchor (thread is born on its FIRST send), so
        carrying it routes progress / status bubbles into the same thread as the final reply."""
        if not _progress_thread_id:
            metadata = None
        elif _progress_thread_id == source.thread_id:
            metadata = self._thread_metadata_for_source(source, event_message_id)
        else:
            metadata = self._thread_metadata_for_target(
                source.platform, source.chat_id, _progress_thread_id,
                chat_type=getattr(source, "chat_type", None), reply_to_message_id=event_message_id,
            )
        if metadata is None and _relay_prospective_thread_id:
            metadata = {"reply_to_message_id": event_message_id}
        return metadata

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
