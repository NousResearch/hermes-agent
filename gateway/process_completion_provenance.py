"""Process-local proof for discarding an unchanged synthetic completion at adapter drain."""
from dataclasses import dataclass
from math import isfinite

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import build_session_key


@dataclass(frozen=True)
class _CompletionProvenance:
    text: str
    route: str
    identities: tuple[tuple[str, float, str], ...]


def stamp_completion_provenance(event: MessageEvent, entries: list[dict]) -> None:
    """Keep immutable producer identity off ingress metadata and durable admission payloads.

    Reconstructed/replayed events intentionally lose this optimization and remain deliverable.
    A missing spawn epoch cannot establish which incarnation an ID-only consumed marker names.
    """
    identities = []
    for entry in entries:
        process_id, started_at, session_key = (entry.get(key) for key in ("session_id", "started_at", "session_key"))
        if (entry.get("type") != "completion" or not isinstance(process_id, str) or not process_id.strip()
                or type(started_at) not in (int, float) or not isfinite(started_at) or started_at <= 0
                or not isinstance(session_key, str) or not session_key
                or session_key != event.metadata.get("gateway_session_key")):
            return
        identities.append((process_id, started_at, session_key))
    if identities:
        event._gateway_process_completion_provenance = _CompletionProvenance(
            event.text, build_session_key(event.source), tuple(identities))


def consumed_completion_only(event: MessageEvent) -> bool:
    """Fail open on mixed input, rewritten content, unknown identity or a stale process instance."""
    proof = getattr(event, "_gateway_process_completion_provenance", None)
    if (not isinstance(proof, _CompletionProvenance) or event.internal is not True
            or event.message_type != MessageType.TEXT or event.media_urls or event.media_types
            or event.prompt_response or event.channel_context or event.reply_to_text or event.raw_message is not None
            or event.text != proof.text or build_session_key(event.source) != proof.route
            or event.metadata.get("notification_origin") != "process_registry_synthetic"
            or event.metadata.get("gateway_process_session_ids") != list(dict.fromkeys(i[0] for i in proof.identities))
            or any(i[2] != event.metadata.get("gateway_session_key") for i in proof.identities)):
        return False
    from tools.process_registry import process_registry
    for process_id, started_at, session_key in proof.identities:
        session = process_registry.get(process_id)
        if (session is None or session.id != process_id or session.started_at != started_at
                or session.session_key != session_key or not session.exited
                or not process_registry.is_completion_consumed(process_id)):
            return False
    return True
