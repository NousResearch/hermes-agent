"""Process completion consumption, silence provenance, and pending FIFO phases."""
from __future__ import annotations
import asyncio
import logging
from typing import Sequence
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")

class GatewayCompletionConsumptionMixin:

    @staticmethod
    def _completion_silence_metadata(event: MessageEvent) -> dict:
        """Authorize only an unmerged, gateway-injected completion at the turn boundary."""
        injected_text = getattr(event, "_completion_silence_text", None)
        if (
            injected_text is not None and event.internal is True
            and event.message_type == MessageType.TEXT and not event.media_urls
            and event.text == injected_text
            and not event.is_command()
        ):
            return {"completion_silence_allowed": True}
        return {}

    @staticmethod
    def _process_completion_consumed(evt: dict) -> bool:
        """An ID-only consumed marker is authoritative only for the known spawn incarnation."""
        from tools.process_registry import process_registry
        if evt.get("type") != "completion" or not evt.get("started_at") or not evt.get("session_id"):
            return False
        try:
            session = process_registry.get(evt["session_id"])
            return bool(
                session is not None and session.id == evt["session_id"]
                and session.started_at == evt["started_at"] and session.exited
                and session.session_key == evt.get("session_key")
                and process_registry.is_completion_consumed(session.id)
            )
        except Exception:
            logger.warning("Could not verify process completion consumption; retaining notification", exc_info=True)
            return False

    def _refresh_process_completion_event(self, event: MessageEvent) -> bool:
        """Revalidate admitted producers; retain unknown work and rebuild only changed batches."""
        from gateway.run_notifications import _mark_internal_notification
        metadata = getattr(event, "metadata", None)
        if not getattr(event, "internal", False) or not isinstance(metadata, dict):
            return True
        entries = metadata.get("process_completion_entries")
        if not entries:
            return True
        if not isinstance(entries, (list, tuple)) or any(
            not isinstance(entry, (list, tuple)) or len(entry) != 2
            or not isinstance(entry[0], str) or not isinstance(entry[1], dict)
            for entry in entries
        ):
            return True
        remaining = [(text, evt) for text, evt in entries if not self._process_completion_consumed(evt)]
        if len(remaining) == len(entries):
            return True
        if not remaining:
            return False
        silence_allowed = bool(self._completion_silence_metadata(event))
        event.metadata["process_completion_entries"] = remaining
        event.text = remaining[0][0] if len(remaining) == 1 else self._format_coalesced_process_completions(
            [(text, evt, None) for text, evt in remaining]
        )
        event.text = _mark_internal_notification(event.text)
        if silence_allowed:
            event._completion_silence_text = event.text
        return True

    @staticmethod
    def _format_coalesced_process_completions(entries: Sequence[tuple[str, dict, asyncio.Future | None]]) -> str:
        """Build one bounded synthetic event from several redacted completions."""
        from gateway.run import _redact_gateway_user_facing_secrets
        lines = [
            f"[IMPORTANT: {len(entries)} background processes completed for this session.",
            "Treat these results as one completion batch and send at most one "
            "consolidated user-facing response.",
        ]
        shown = entries[:10]
        for _text, evt, _future in shown:
            session_id = str(evt.get("session_id") or "unknown")
            exit_code = evt.get("exit_code")
            reason = str(evt.get("completion_reason") or "exited")
            # Unconditional gateway redaction floor (the producer-seam redactor is configurable). Redact
            # BEFORE slicing: truncating first can leave a credential fragment the patterns miss.
            output = _redact_gateway_user_facing_secrets(str(evt.get("output") or "")).strip()
            if len(output) > 800:
                output = f"[… truncated …]\n{output[-800:]}"
            lines.append(f"\n- {session_id}: exit_code={exit_code}, reason={reason}")
            if output:
                lines.append(output)
        omitted = len(entries) - len(shown)
        if omitted:
            lines.append(
                f"\n- … and {omitted} more completion(s); inspect them with "
                "the process tool if they affect the conclusion."
            )
        lines.append("If a result does not change the current conclusion, absorb it silently.]")
        return "\n".join(lines)

    def _hm_rescue_orphaned_fifo(
        self, event: MessageEvent, source: SessionSource, is_internal: bool, _quick_key: str
    ) -> tuple[MessageEvent, SessionSource, bool]:
        """FIFO orphan rescue: a session that went idle with a populated overflow (post-turn drain
        never promoted, e.g. a compression-demoted follow-up) silently orphaned those events. The
        oldest orphan runs as THIS turn and the incoming event is parked behind the chain. Skipped
        for control commands and internal events."""
        try:
            # ── FIFO orphan rescue (#99882) ──────────────────────────────── If this session went idle with
            # a populated overflow (queued during a busy window whose post-turn drain never promoted — e.g.
            # a compression-demoted follow-up after the compression window ended through an exit that
            # skipped the promotion site), those events were silently orphaned. We are starting the next
            # turn for this session NOW: re-stage the orphans in FIFO order and enqueue the incoming event
            # behind them, so arrival order (#28503) holds: oldest orphan runs as this turn, the rest drain
            # in order, the new message last.
            _orphan_adapter = self._delivery_adapter_for(source)
            if _orphan_adapter is None or getattr(event, "internal", False) or event.get_command():
                return event, source, is_internal
            _rescued = self._rescue_orphaned_overflow(_quick_key, _orphan_adapter)
            if _rescued is None:
                return event, source, is_internal
            # Into the slot when the chain was a single orphan (post-turn drain picks it up),
            # otherwise into overflow behind the already-staged next orphan.
            self._enqueue_fifo(_quick_key, event, _orphan_adapter)
            # Same session key by construction; carry the orphan's own source so reply anchors /
            # thread metadata point at the message actually being answered.
            _rescued_source = getattr(_rescued, "source", None)
            source = _rescued_source if _rescued_source is not None else source
            return _rescued, source, bool(getattr(_rescued, "internal", False))
        except Exception:
            logger.debug("FIFO orphan rescue pre-claim failed for %s", _quick_key, exc_info=True)
            return event, source, is_internal

    def _next_unconsumed_completion_event(self, event):
        """Skip consumed cold wakes while promoting their queued siblings."""
        from gateway.run import _dequeue_pending_event
        while not self._refresh_process_completion_event(event):
            adapter = self._delivery_adapter_for(event.source)
            key = self._session_key_for_source(event.source)
            event = self._promote_queued_event(
                key, adapter, _dequeue_pending_event(adapter, key) if adapter else None,
            )
            if event is None:
                return None
        return event

    def _next_unconsumed_pending_completion(self, event, adapter, session_key):
        """Skip consumed pending wakes without stranding FIFO siblings."""
        from gateway.run import _dequeue_pending_event
        while event and not self._refresh_process_completion_event(event):
            event = self._promote_queued_event(
                session_key, adapter, _dequeue_pending_event(adapter, session_key),
            )
        return event
