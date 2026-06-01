from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from .ports import (
    NativeMediaAnswer,
    NativeMediaAnswerRequest,
    NativeMediaAnswerResult,
    NativeMediaOffer,
    NativeMediaStartRequest,
    NativeMediaStartResult,
)

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
    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 10.0,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._command = list(command)
        self._timeout_seconds = timeout_seconds
        self._on_event = on_event
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._stderr_tasks: dict[int, asyncio.Task] = {}
        self._stdout_tasks: dict[int, asyncio.Task] = {}
        self._response_waiters: dict[str, asyncio.Future] = {}

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
                stderr=asyncio.subprocess.PIPE,
            )
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("sidecar stdio pipes are unavailable")
            if process.stderr is not None:
                self._stderr_tasks[id(process)] = asyncio.create_task(
                    self._drain_stderr(process.stderr, request.call_id)
                )

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
                result = self._result_from_response(response)
                if result.ok:
                    previous = self._processes.pop(request.call_id, None)
                    if previous is not None:
                        await self._cleanup_process(previous)
                    self._processes[request.call_id] = process
                    self._stdout_tasks[id(process)] = asyncio.create_task(
                        self._drain_stdout(process, request.call_id)
                    )
                    process = None
                return result
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

    async def start_outgoing_answer(
        self,
        request: NativeMediaAnswerRequest,
    ) -> NativeMediaAnswerResult:
        if not self._command:
            return NativeMediaAnswerResult(
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
                stderr=asyncio.subprocess.PIPE,
            )
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("sidecar stdio pipes are unavailable")
            if process.stderr is not None:
                self._stderr_tasks[id(process)] = asyncio.create_task(
                    self._drain_stderr(process.stderr, request.call_id)
                )

            payload = {
                "type": "start_outgoing_answer",
                "callId": request.call_id,
                "contactId": request.contact_id,
                "media": request.media,
                "encrypted": request.encrypted,
                "sharedKey": request.shared_key,
                "offer": self._json_payload(request.offer),
            }
            process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await process.stdin.drain()

            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                return NativeMediaAnswerResult(
                    ok=False,
                    code="call_simplex_native_timeout",
                    message="timed out waiting for call sidecar response",
                )

            if not line:
                return self._answer_protocol_failure(
                    "missing_response",
                    "call sidecar exited before sending a response",
                )

            try:
                response = json.loads(line.decode("utf-8"))
                result = self._answer_result_from_response(response)
                if result.ok:
                    previous = self._processes.pop(request.call_id, None)
                    if previous is not None:
                        await self._cleanup_process(previous)
                    self._processes[request.call_id] = process
                    self._stdout_tasks[id(process)] = asyncio.create_task(
                        self._drain_stdout(process, request.call_id)
                    )
                    process = None
                return result
            except UnicodeDecodeError:
                return self._answer_protocol_failure("invalid_response_encoding")
            except json.JSONDecodeError:
                return self._answer_protocol_failure("invalid_response")
            except _SidecarProtocolError as exc:
                return self._answer_protocol_failure(exc.reason, exc.message)
        except Exception:
            logger.exception("SimpleX native call sidecar failed")
            return NativeMediaAnswerResult(
                ok=False,
                code="call_sidecar_start_failed",
                message="call sidecar failed to start or returned invalid protocol data",
            )
        finally:
            if process is not None:
                await self._cleanup_process(process)

    async def stop(self, call_id: str) -> None:
        process = self._processes.pop(call_id, None)
        if process is None:
            return

        stdin = process.stdin
        if stdin is not None and not stdin.is_closing():
            try:
                payload = {"type": "stop", "callId": call_id}
                stdin.write(
                    (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
                )
                await stdin.drain()
                stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                logger.debug(
                    "SimpleX native call sidecar stop command failed",
                    extra={"call_id": call_id},
                    exc_info=True,
                )

        await self._cleanup_process(process, graceful_timeout=1.0)

    async def apply_answer(self, call_id: str, answer: Any) -> bool:
        return await self._send_control(call_id, "apply_answer", "answer", answer)

    async def add_extra_ice(self, call_id: str, extra: Any) -> bool:
        return await self._send_control(call_id, "add_extra_ice", "extra", extra)

    async def debug_process_audio(self, call_id: str, audio_path: str) -> dict[str, Any]:
        process = self._processes.get(call_id)
        if process is None or process.returncode is not None:
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_no_sidecar",
                "message": "no active native call sidecar is available",
            }

        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None or stdin.is_closing():
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_no_sidecar",
                "message": "native call sidecar stdio is unavailable",
            }

        waiter: asyncio.Future | None = None
        try:
            loop = asyncio.get_running_loop()
            waiter = loop.create_future()
            self._response_waiters[call_id] = waiter
            payload = {
                "type": "debug_process_audio",
                "callId": call_id,
                "audioPath": str(audio_path),
            }
            stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await stdin.drain()
            try:
                response = await asyncio.wait_for(
                    waiter,
                    timeout=self._timeout_seconds,
                )
            except TimeoutError:
                return {
                    "ok": False,
                    "code": "call_simplex_native_debug_audio_timeout",
                    "message": "timed out waiting for debug audio response",
                }
            if isinstance(response, dict) and isinstance(response.get("ok"), bool):
                return response
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_failed",
                "message": "sidecar returned malformed debug audio response",
            }
        except (BrokenPipeError, ConnectionResetError):
            self._processes.pop(call_id, None)
            await self._cleanup_process(process)
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_failed",
                "message": "native call sidecar pipe closed",
            }
        except Exception:
            logger.debug(
                "SimpleX native call sidecar debug audio command failed",
                extra={"call_id": call_id},
                exc_info=True,
            )
            return {
                "ok": False,
                "code": "call_simplex_native_debug_audio_failed",
                "message": "native call sidecar debug audio command failed",
            }
        finally:
            if waiter is not None and self._response_waiters.get(call_id) is waiter:
                self._response_waiters.pop(call_id, None)

    async def _send_control(
        self,
        call_id: str,
        control_type: str,
        payload_key: str,
        payload_value: Any,
    ) -> bool:
        process = self._processes.get(call_id)
        if process is None:
            return False
        if process.returncode is not None:
            self._processes.pop(call_id, None)
            await self._cleanup_process(process)
            return False

        stdin = process.stdin
        if stdin is None or stdin.is_closing():
            return False

        try:
            payload = {
                "type": control_type,
                "callId": call_id,
                payload_key: self._json_payload(payload_value),
            }
            stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            await stdin.drain()
            return True
        except (BrokenPipeError, ConnectionResetError):
            self._processes.pop(call_id, None)
            await self._cleanup_process(process)
            return False
        except Exception:
            logger.debug(
                "SimpleX native call sidecar control command failed",
                extra={"call_id": call_id, "control_type": control_type},
                exc_info=True,
            )
            return False

    def _json_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        rtc_session = getattr(value, "rtc_session", None)
        rtc_ice_candidates = getattr(value, "rtc_ice_candidates", None)
        call_dh_pub_key = getattr(value, "call_dh_pub_key", None)
        if (
            rtc_session is not None
            or rtc_ice_candidates is not None
            or call_dh_pub_key is not None
        ):
            payload: dict[str, Any] = {}
            if rtc_session is not None:
                payload["rtcSession"] = rtc_session
            if rtc_ice_candidates is not None:
                payload["rtcIceCandidates"] = rtc_ice_candidates
            if call_dh_pub_key:
                payload["callDhPubKey"] = call_dh_pub_key
            capabilities = getattr(value, "capabilities", None)
            if isinstance(capabilities, dict):
                payload["capabilities"] = capabilities
            return payload
        return value

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

            call_dh_pub_key = offer.get("callDhPubKey")
            if call_dh_pub_key is not None and not isinstance(call_dh_pub_key, str):
                raise _SidecarProtocolError("invalid_call_dh_pub_key")

            if not isinstance(capabilities, dict):
                capabilities = {}
            return NativeMediaStartResult(
                ok=True,
                offer=NativeMediaOffer(
                    rtc_session=rtc_session,
                    rtc_ice_candidates=rtc_ice_candidates,
                    capabilities=capabilities,
                    call_dh_pub_key=call_dh_pub_key,
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

    def _answer_result_from_response(self, response: Any) -> NativeMediaAnswerResult:
        if not isinstance(response, dict):
            raise _SidecarProtocolError("invalid_response")

        ok = response.get("ok")
        if not isinstance(ok, bool):
            raise _SidecarProtocolError("invalid_ok")

        if ok is True:
            answer = response.get("answer")
            if not isinstance(answer, dict):
                raise _SidecarProtocolError("invalid_answer")

            rtc_session = answer.get("rtcSession")
            if not isinstance(rtc_session, str):
                raise _SidecarProtocolError("invalid_rtc_session")

            rtc_ice_candidates = answer.get("rtcIceCandidates")
            if not isinstance(rtc_ice_candidates, str):
                raise _SidecarProtocolError("invalid_rtc_ice_candidates")

            return NativeMediaAnswerResult(
                ok=True,
                answer=NativeMediaAnswer(
                    rtc_session=rtc_session,
                    rtc_ice_candidates=rtc_ice_candidates,
                ),
            )

        code = response.get("code")
        message = response.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise _SidecarProtocolError(
                "invalid_error_response",
                _FAILED_RESPONSE_MESSAGE,
            )

        return NativeMediaAnswerResult(
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

    def _answer_protocol_failure(
        self,
        reason: str,
        message: str = _MALFORMED_RESPONSE_MESSAGE,
    ) -> NativeMediaAnswerResult:
        logger.warning(
            "SimpleX native call sidecar protocol failure",
            extra={"reason": reason},
        )
        return NativeMediaAnswerResult(
            ok=False,
            code=_PROTOCOL_FAILED,
            message=message,
        )

    async def _cleanup_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        graceful_timeout: float = 0.0,
        skip_stdout_task: bool = False,
    ) -> None:
        try:
            if process.returncode is None:
                if graceful_timeout > 0:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
                        return
                    except TimeoutError:
                        pass
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
        finally:
            if not skip_stdout_task:
                await self._finish_stdout_task(process)
            await self._finish_stderr_task(process)

    async def _finish_stdout_task(self, process: asyncio.subprocess.Process) -> None:
        task = self._stdout_tasks.pop(id(process), None)
        if task is None:
            return
        if not task.done():
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except TimeoutError:
                task.cancel()
            except Exception:
                logger.debug("SimpleX native call sidecar stdout drain failed", exc_info=True)
        if task.done() and not task.cancelled():
            try:
                task.result()
            except Exception:
                logger.debug("SimpleX native call sidecar stdout drain failed", exc_info=True)

    async def _finish_stderr_task(self, process: asyncio.subprocess.Process) -> None:
        task = self._stderr_tasks.pop(id(process), None)
        if task is None:
            return
        if not task.done():
            try:
                await asyncio.wait_for(task, timeout=0.5)
            except TimeoutError:
                task.cancel()
            except Exception:
                logger.debug("SimpleX native call sidecar stderr drain failed", exc_info=True)
        if task.done() and not task.cancelled():
            try:
                task.result()
            except Exception:
                logger.debug("SimpleX native call sidecar stderr drain failed", exc_info=True)

    async def _drain_stderr(
        self,
        stream: asyncio.StreamReader,
        call_id: str,
    ) -> None:
        from agent.redact import redact_sensitive_text

        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            logger.info(
                "SimpleX native call sidecar stderr: %s",
                redact_sensitive_text(text[:2000], force=True),
                extra={"call_id": call_id},
            )

    async def _drain_stdout(
        self,
        process: asyncio.subprocess.Process,
        call_id: str,
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            return
        while True:
            line = await stdout.readline()
            if not line:
                waiter = self._response_waiters.pop(call_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(
                        {
                            "ok": False,
                            "code": "call_simplex_native_sidecar_exited",
                            "message": "native SimpleX sidecar exited",
                        }
                    )
                return
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning(
                    "Ignoring malformed SimpleX native sidecar stdout",
                    extra={"call_id": call_id},
                )
                continue
            if not isinstance(payload, dict):
                logger.warning(
                    "Ignoring non-object SimpleX native sidecar stdout",
                    extra={"call_id": call_id},
                )
                continue
            if payload.get("type") == "event" or payload.get("type") in {
                "audio",
                "ice",
                "status",
            }:
                await self._handle_event(process, call_id, payload)
                continue
            waiter = self._response_waiters.get(call_id)
            if waiter is not None and not waiter.done():
                waiter.set_result(payload)
                continue
            logger.debug(
                "Ignoring unexpected SimpleX native sidecar response",
                extra={"call_id": call_id},
            )

    async def _handle_event(
        self,
        process: asyncio.subprocess.Process,
        call_id: str,
        event: dict[str, Any],
    ) -> None:
        event.setdefault("callId", call_id)
        event.setdefault("event", event.get("type"))
        if self._on_event is not None:
            try:
                result = self._on_event(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning(
                    "SimpleX native call sidecar event handler failed",
                    extra={"call_id": call_id},
                    exc_info=True,
                )
        if event.get("event") == "status" and event.get("status") in {
            "ended",
            "failed",
        }:
            if self._processes.get(call_id) is process:
                self._processes.pop(call_id, None)
                self._stdout_tasks.pop(id(process), None)
                await self._cleanup_process(process, skip_stdout_task=True)
