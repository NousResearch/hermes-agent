"""Session, transcript and inbound preparation before a gateway agent turn."""

from __future__ import annotations

import dataclasses
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, List, Optional

from agent.i18n import t
from gateway.response_filters import display_kind_for_event
from gateway.session import build_session_context
from gateway.session_transcript import TranscriptReadError


if TYPE_CHECKING:
    from gateway.run import GatewayRunner


class GatewayTurnPreparationMixin:
    @dataclasses.dataclass
    class _PreparedTurn:
        """Inputs to the agent run assembled by ``_hmwa_prepare_turn``."""

        history: Any
        context_prompt: str
        message_text: Any
        persist_user_message: Any
        persist_user_timestamp: Any
        persist_user_display_kind: Optional[str]
        persistence_session_id: Optional[str] = None
        persistence_owner: Optional[str] = None
        title_user_message: Optional[str] = None

    async def _hmwa_prepare_turn(self: GatewayRunner, event, source, session_entry, session_key, _quick_key, run_generation):
        """Everything between session resolution and the agent run: session open, task-local env,
        context prompt, sidecar notes, turn lease, transcript load + hygiene, inbound text. Returns
        ``(_PreparedTurn, env_tokens)``; a ``str`` first element is a reply to send instead of
        running (history unreadable); ``None`` drops the turn (inbound text rejected)."""
        from gateway.run import _load_gateway_config
        from tools.approval_yolo import restore_session_yolo
        _was_auto_reset, _is_new_session = await self._hmwa_open_session(session_entry, session_key, source)
        restore_session_yolo(session_key, session_entry.yolo is True)  # a restarted gateway's set starts empty
        context = build_session_context(source, self.config, session_entry)
        # Session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)
        # Self-injected turns (internal=True) persist with a DB-only display_kind: timeline notices, not user bubbles.
        persist_user_display_kind = display_kind_for_event(event)
        _redact_pii = False  # privacy.redact_pii, re-read per message
        with suppress(Exception):
            _redact_pii = bool((_load_gateway_config().get("privacy") or {}).get("redact_pii", False))

        # The context prompt render is pinned per session, keyed by a hash of the renderer inputs, so
        # the system prompt cannot drift turn-over-turn; a miss (thread rename, /sethome) re-renders.
        if event.internal and session_key:
            await self._rehydrate_prompt_pins(session_key, session_entry.session_id)
        context_prompt = self._pinned_session_context_prompt(
            self._prompt_session_context(context, session_entry), _redact_pii, session_key,
            internal=event.internal,
        )

        # Per-turn notes ride the user message via the api_content sidecar, NOT context_prompt
        # (appending to the ephemeral system prompt forced a full agent rebuild).
        turn_sidecar_notes: list[str] = []
        if _was_auto_reset:
            await self._hmwa_deliver_auto_reset_notice(session_entry, source, turn_sidecar_notes)

        # Keep human text separate from skills, sender metadata, and other model context.
        # With no text (e.g. voice-only), retain the existing enriched-message title fallback.
        title_user_message = event.text or None

        # Auto-load bound skill(s) only on NEW sessions; ongoing ones carry the content in history.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            self._hmwa_auto_load_skills(event, _auto, _quick_key, session_key)

        await self._hmwa_acquire_turn_lease(_quick_key, run_generation, session_entry, _session_env_tokens)

        # A turn becomes durable recovery work only after it owns the per-session lease; marking
        # earlier would falsely recover a message that never began processing.
        await self._mark_durable_active_turn(event, session_entry.session_key)

        # An unreadable store is not an empty conversation: stop before the agent invents continuity
        # from []. Restore task-local context here (before the broad cleanup finally).
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
            history = await self._hmwa_run_session_hygiene(
                event, source, session_entry, session_key, history, _quick_key, run_generation,
            )
        except TranscriptReadError:
            self._clear_session_env(_session_env_tokens)
            return t("gateway.errors.history_unavailable"), _session_env_tokens

        await self._hmwa_first_contact_notes(source, history, turn_sidecar_notes, event.text, internal=event.internal)

        # Voice channel state rides the user message ONLY when changed (in the system prompt it
        # forced a rebuild + prompt-cache re-key per message).
        _vc_note = self._voice_channel_sidecar_note(event, source, session_key)
        if _vc_note:
            turn_sidecar_notes.append(_vc_note)

        # Auto-analyze user images so the model gets a description plus the local path.
        message_text = await self._prepare_profile_scoped_inbound_message_text(
            event=event, source=source, history=history, session_key=session_key,
        )
        if message_text is None:
            return None, _session_env_tokens

        message_text, persist_user_message, persist_user_timestamp = (
            self._hmwa_apply_message_timestamp(event, message_text)
        )

        # Stage the notes (one-shot; consumed in run_sync) AFTER the early-out so an aborted turn
        # cannot leak them into the next turn.
        if turn_sidecar_notes and session_key:
            self._set_pending_turn_sidecar_notes(session_key, turn_sidecar_notes)

        # Bind this run generation to the adapter so deferred post-delivery callbacks are released
        # by the run that registered them.
        self._bind_adapter_run_generation(self._delivery_adapter_for(source), session_key, run_generation)
        # Delivery IDs are only unique in their transport namespace. Keyless turns
        # need their own identity, even when another process writes to this session.
        import uuid
        namespace = [source.platform.value, source.profile, source.scope_id,
                     source.chat_id, source.thread_id, str(event.message_id)]
        owner = (str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(namespace)))
                 if event.message_id else str(uuid.uuid4()))
        from gateway.platforms.base_pending import bind_pending_dispatch_input
        bind_pending_dispatch_input(session_entry.session_id, owner)
        return self._PreparedTurn(
            history, context_prompt, message_text, persist_user_message, persist_user_timestamp,
            persist_user_display_kind, session_entry.session_id, owner,
            title_user_message=title_user_message,
        ), _session_env_tokens
