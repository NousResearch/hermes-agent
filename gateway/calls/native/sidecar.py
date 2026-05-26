from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from .ports import NativeMediaOffer, NativeMediaStartRequest, NativeMediaStartResult

logger = logging.getLogger(__name__)


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
                return NativeMediaStartResult(
                    ok=False,
                    code="call_sidecar_protocol_failed",
                    message="call sidecar exited before sending a response",
                )

            response = json.loads(line.decode("utf-8"))
            return self._result_from_response(response)
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

    def _result_from_response(self, response: dict[str, Any]) -> NativeMediaStartResult:
        if response.get("ok") is True:
            offer = response.get("offer")
            if not isinstance(offer, dict):
                raise ValueError("sidecar response is missing offer")
            capabilities = offer.get("capabilities", {})
            if not isinstance(capabilities, dict):
                capabilities = {}
            return NativeMediaStartResult(
                ok=True,
                offer=NativeMediaOffer(
                    rtc_session=offer["rtcSession"],
                    rtc_ice_candidates=offer["rtcIceCandidates"],
                    capabilities=capabilities,
                ),
            )

        return NativeMediaStartResult(
            ok=False,
            code=response.get("code") or "call_sidecar_protocol_failed",
            message=response.get("message") or "call sidecar returned an invalid response",
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
