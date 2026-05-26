from __future__ import annotations

import logging
import uuid
from typing import Any

from .ports import (
    AuthorizeSource,
    NativeCallInvitation,
    NativeCallResult,
    NativeCallSignalingPort,
    NativeMediaStartRequest,
    WebRTCMediaPort,
)

logger = logging.getLogger(__name__)


def _is_dm(source: Any) -> bool:
    return str(getattr(source, "chat_type", "dm") or "dm").lower() == "dm"


async def _reject_if_possible(
    signaling: NativeCallSignalingPort,
    contact_id: str,
    reason_code: str,
) -> None:
    try:
        await signaling.reject(contact_id, reason_code)
    except Exception:
        logger.warning(
            "SimpleX native call reject failed",
            extra={"contact_id": contact_id, "reason_code": reason_code},
            exc_info=True,
        )


class NativeCallApplication:
    def __init__(
        self,
        *,
        signaling: NativeCallSignalingPort,
        media: WebRTCMediaPort,
        is_authorized: AuthorizeSource,
    ) -> None:
        self.signaling = signaling
        self.media = media
        self.is_authorized = is_authorized

    async def handle_incoming_invitation(
        self,
        source: Any,
        invitation: NativeCallInvitation,
    ) -> NativeCallResult:
        contact_id = str(invitation.contact_id or "").strip()
        if not contact_id:
            logger.warning("SimpleX native call invitation missing contact id")
            return NativeCallResult(
                ok=False,
                code="call_simplex_native_signaling_failed",
                message="SimpleX-native call setup failed: missing contact id.",
            )

        if not _is_dm(source):
            code = "call_private_chat_required"
            logger.info(
                "SimpleX native call rejected",
                extra={"contact_id": contact_id, "reason_code": code},
            )
            await _reject_if_possible(self.signaling, contact_id, code)
            return NativeCallResult(
                ok=False,
                code=code,
                message="Calls are private-only. DM me /call to create a private room.",
            )

        try:
            authorized = bool(self.is_authorized(source))
        except Exception:
            logger.exception("SimpleX native call authorization check failed")
            authorized = False

        if not authorized:
            code = "call_auth_denied"
            logger.info(
                "SimpleX native call rejected",
                extra={"contact_id": contact_id, "reason_code": code},
            )
            await _reject_if_possible(self.signaling, contact_id, code)
            return NativeCallResult(
                ok=False,
                code=code,
                message="SimpleX-native call rejected.",
            )

        call_id = f"call_{uuid.uuid4().hex}"
        try:
            media_result = await self.media.start_incoming(
                NativeMediaStartRequest(
                    call_id=call_id,
                    contact_id=contact_id,
                    media=invitation.media,
                    encrypted=invitation.encrypted,
                    shared_key=invitation.shared_key,
                )
            )
        except Exception as exc:
            code = "call_simplex_native_media_failed"
            logger.warning(
                "SimpleX native call media start raised",
                extra={
                    "call_id": call_id,
                    "contact_id": contact_id,
                    "exception_type": type(exc).__name__,
                    "reason_code": code,
                },
            )
            await _reject_if_possible(self.signaling, contact_id, code)
            return NativeCallResult(
                ok=False,
                code=code,
                message="SimpleX-native call media setup failed.",
                call_id=call_id,
            )
        if not media_result.ok or media_result.offer is None:
            code = media_result.code or "call_simplex_native_media_failed"
            logger.warning(
                "SimpleX native call media start failed: call_id=%s code=%s",
                call_id,
                code,
                extra={"call_id": call_id, "contact_id": contact_id, "reason_code": code},
            )
            await _reject_if_possible(self.signaling, contact_id, code)
            return NativeCallResult(
                ok=False,
                code=code,
                message=media_result.message or "SimpleX-native call media setup failed.",
                call_id=call_id,
            )

        try:
            await self.signaling.send_offer(contact_id, media_result.offer)
            await self.signaling.send_status(contact_id, "connecting")
        except Exception:
            code = "call_simplex_native_signaling_failed"
            logger.warning(
                "SimpleX native call signaling failed",
                extra={"call_id": call_id, "contact_id": contact_id, "reason_code": code},
                exc_info=True,
            )
            try:
                await self.media.stop(call_id)
            except Exception:
                logger.warning(
                    "SimpleX native call media stop failed after signaling failure",
                    extra={"call_id": call_id, "contact_id": contact_id, "reason_code": code},
                    exc_info=True,
                )
            await _reject_if_possible(self.signaling, contact_id, code)
            return NativeCallResult(
                ok=False,
                code=code,
                message="SimpleX-native call signaling failed.",
                call_id=call_id,
            )

        logger.info("SimpleX native call offer sent: call_id=%s", call_id)
        return NativeCallResult(
            ok=True,
            code="call_simplex_native_connecting",
            message="SimpleX-native call is connecting.",
            call_id=call_id,
        )
