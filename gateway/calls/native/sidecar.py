from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from .ports import NativeMediaOffer, NativeMediaStartRequest, NativeMediaStartResult

logger = logging.getLogger(__name__)

_PROTOCOL_FAILED = "call_sidecar_protocol_failed"
_MALFORMED_RESPONSE_MESSAGE = "call sidecar returned malformed protocol response"
_FAILED_RESPONSE_MESSAGE = "call sidecar returned a failed response"


class _SidecarProtocolError(Exception):
    def __init__(self, reason: str, message: str = _MALFORMED_RESPONSE_MESSAGE) -> None:
        super().__init__(reason)
        self.reason = reason
        self.message = message


class SidecarMediaPort:
    def __init__(self, command: Sequence[str], timeout_seconds: float = 10.0) -> None:
        self._command = list(command)
        self._timeout_seconds = timeout_seconds

    async def start_incoming(self, request: NativeMediaStartRequest) -> NativeMediaStartResult:
        if not self._command:
            return NativeMediaStartResult(
                ok=False,
                code="call_sidecar_start_failed",
                message="sidecar command is not configured",
            )

        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("sidecar stdio pipes are unavailable")

            payload = {
                "type": "start_incoming",
                "callId": request.call_id,
                "contactId": request.contact_id,
                "media": request.media,
                "encrypted": request.encrypted,
                "sharedKey": request.shared_key,
            }
            process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await process.stdin.drain()
            process.stdin.close()

            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                return NativeMediaStartResult(
                    ok=False,
                    code="call_simplex_native_timeout",
                    message="timed out waiting for call sidecar response",
                )

            if not line:
                return self._protocol_failure(
                    "missing_response",
                    "call sidecar exited before sending a response",
                )

            try:
                response = json.loads(line.decode("utf-8"))
                return self._result_from_response(response)
            except UnicodeDecodeError:
                return self._protocol_failure("invalid_response_encoding")
            except json.JSONDecodeError:
                return self._protocol_failure("invalid_response")
            except _SidecarProtocolError as exc:
                return self._protocol_failure(exc.reason, exc.message)
        except Exception:
            logger.exception("SimpleX native call sidecar failed")
            return NativeMediaStartResult(
                ok=False,
                code="call_sidecar_start_failed",
                message="call sidecar failed to start or returned invalid protocol data",
            )
        finally:
            if process is not None:
                await self._cleanup_process(process)

    async def stop(self, call_id: str) -> None:
        return None

    def _result_from_response(self, response: Any) -> NativeMediaStartResult:
        if not isinstance(response, dict):
            raise _SidecarProtocolError("invalid_response")

        ok = response.get("ok")
        if not isinstance(ok, bool):
            raise _SidecarProtocolError("invalid_ok")

        if ok is True:
            offer = response.get("offer")
            if not isinstance(offer, dict):
                raise _SidecarProtocolError("invalid_offer")

            rtc_session = offer.get("rtcSession")
            if not isinstance(rtc_session, str):
                raise _SidecarProtocolError("invalid_rtc_session")

            rtc_ice_candidates = offer.get("rtcIceCandidates")
            if not isinstance(rtc_ice_candidates, str):
                raise _SidecarProtocolError("invalid_rtc_ice_candidates")

            capabilities = offer.get("capabilities", {})
            if "capabilities" in offer and not isinstance(capabilities, dict):
                raise _SidecarProtocolError("invalid_capabilities")

            if not isinstance(capabilities, dict):
                capabilities = {}
            return NativeMediaStartResult(
                ok=True,
                offer=NativeMediaOffer(
                    rtc_session=rtc_session,
                    rtc_ice_candidates=rtc_ice_candidates,
                    capabilities=capabilities,
                ),
            )

        code = response.get("code")
        message = response.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise _SidecarProtocolError(
                "invalid_error_response",
                _FAILED_RESPONSE_MESSAGE,
            )

        return NativeMediaStartResult(
            ok=False,
            code=code,
            message=message,
        )

    def _protocol_failure(
        self,
        reason: str,
        message: str = _MALFORMED_RESPONSE_MESSAGE,
    ) -> NativeMediaStartResult:
        logger.warning(
            "SimpleX native call sidecar protocol failure",
            extra={"reason": reason},
        )
        return NativeMediaStartResult(
            ok=False,
            code=_PROTOCOL_FAILED,
            message=message,
        )

    async def _cleanup_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            process.kill()
            await process.wait()
