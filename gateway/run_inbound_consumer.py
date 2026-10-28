"""``post_gateway_admission``: let a plugin consume an admitted inbound message (#129958).

``GatewayInboundMixin._handle_message`` fires it once per non-internal message that survived
authorization, bot admission, pause/drain, pending-reply intercepts, the running-session lane and
idle slash-command dispatch, inside the claimed session slot. It runs under the message's ROUTED
profile scope: the receiving bot's handler binds the transport home (auth reads its ``.env``), which
differs from the runtime home when that bot serves a chat for another profile.

Fail-open: a callback that raises, times out or returns anything other than
``{"action": "handled"}`` is logged/ignored and the message proceeds to the agent, so one buggy
plugin can never blackhole a profile's traffic. The payload is a plain snapshot, never live
runner/session-store handles.

``post_transcription_admission`` (#145125) is the SAME contract fired once more, after speech-to-text
has run, for the messages whose text is only known then (a voice note with no caption). It is a
separate hook rather than a flag on the first one so the existing ordering — and every plugin
already written against it — stays exactly as it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

POST_ADMISSION_HOOK = "post_gateway_admission"
POST_TRANSCRIPTION_ADMISSION_HOOK = "post_transcription_admission"


def _routed_scope(runner: Any, source: Any):
    if not getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        return contextlib.nullcontext()
    from gateway.run import _async_profile_runtime_scope
    return _async_profile_runtime_scope(runner._resolve_profile_home_for_source(source))


async def _fire_consume_hook(
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    hook_name: str,
    text: str,
    *,
    source_kind: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Return ``(handled, reply)``; ``handled=False`` means run the ordinary agent turn."""
    try:
        async with _routed_scope(runner, source):
            from hermes_cli.lifecycle import ainvoke_hook, has_hook
            if not has_hook(hook_name):
                return False, None
            # Built INSIDE the fail-open boundary: a payload that cannot be built
            # (a source snapshot that raises) must let the agent turn run, exactly as
            # before this hook was factored out.
            payload: dict[str, Any] = dict(
                session_key=session_key,
                platform=source.platform.value if source.platform else "",
                source=source.to_dict(), message_id=event.message_id, text=text,
            )
            # Additive: only the transcription hook carries it, so the payload of the existing
            # ``post_gateway_admission`` hook is byte-for-byte what it always was.
            if source_kind is not None:
                payload["source_kind"] = source_kind
            results = await ainvoke_hook(hook_name, **payload)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("%s failed; continuing with the agent turn", hook_name, exc_info=True)
        return False, None
    for result in results:
        if isinstance(result, dict) and result.get("action") == "handled":
            reply = result.get("reply")
            logger.info("%s: a plugin handled the message for session %s", hook_name, session_key)
            return True, reply if isinstance(reply, str) and reply.strip() else None
    return False, None


async def run_post_admission_hook(
    runner: Any, event: Any, source: Any, session_key: str
) -> tuple[bool, Optional[str]]:
    """Return ``(handled, reply)``; ``handled=False`` means run the ordinary agent turn."""
    return await _fire_consume_hook(
        runner, event, source, session_key, POST_ADMISSION_HOOK, event.text or "",
    )


async def run_post_transcription_admission_hook(
    runner: Any, event: Any, source: Any, session_key: str, transcript: str
) -> tuple[bool, Optional[str]]:
    """Fire ``post_transcription_admission`` with the spoken text of a voice message.

    Same fail-open contract and same ``{"action": "handled", "reply": ...}`` reply shape as
    ``post_gateway_admission``; the payload adds ``source_kind="voice"`` so a plugin can be
    stricter with a spoken command than with a typed one.
    """
    return await _fire_consume_hook(
        runner, event, source, session_key, POST_TRANSCRIPTION_ADMISSION_HOOK, transcript,
        source_kind="voice",
    )
