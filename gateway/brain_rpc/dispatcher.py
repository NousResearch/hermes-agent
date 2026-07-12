"""Brain RPC dispatcher — parse request envelope, auth, dispatch, result.

Entry point used by the relay WebSocket transport when it receives a
``brain_rpc_request`` frame on an authenticated session.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from gateway.brain_rpc.auth import verify_auth
from gateway.brain_rpc.config import (
    DEFAULT_TIMEOUT_MS,
    MAX_IN_FLIGHT,
    MAX_TIMEOUT_MS,
    BrainRpcHostConfig,
    is_brain_rpc_enabled,
)
from gateway.brain_rpc.errors import (
    INTERNAL,
    METHOD_NOT_FOUND,
    RATE_LIMITED,
    TIMEOUT,
    UNAVAILABLE,
    VERSION_UNSUPPORTED,
    BrainRpcError,
)
from gateway.brain_rpc.handlers import BRAIN_RPC_CONTRACT_VERSION, HANDLERS

logger = logging.getLogger(__name__)

# Re-export for package consumers
__all__ = [
    "BRAIN_RPC_CONTRACT_VERSION",
    "BrainRpcDispatcher",
    "handle_brain_rpc_request",
    "is_brain_rpc_enabled",
]


def _host_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _result_frame(
    request_id: str,
    *,
    ok: bool,
    result: Any = None,
    error: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": "brain_rpc_result",
        "contract_version": BRAIN_RPC_CONTRACT_VERSION,
        "request_id": request_id,
        "ok": ok,
        "result": result if ok else None,
        "error": error if not ok else None,
        "meta": meta or {},
    }


class BrainRpcDispatcher:
    """Stateful dispatcher with in-flight concurrency bound (contract §7)."""

    def __init__(
        self,
        host: Optional[BrainRpcHostConfig] = None,
        *,
        max_in_flight: int = MAX_IN_FLIGHT,
    ) -> None:
        self.host = host or BrainRpcHostConfig.from_env()
        self._max_in_flight = max(1, max_in_flight)
        # Event-loop-local counter (handle() is async; no threads share it).
        self._in_flight = 0

    async def handle(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """Process one ``brain_rpc_request`` frame → ``brain_rpc_result`` frame."""
        if not is_brain_rpc_enabled():
            rid = str((frame or {}).get("request_id") or "")
            return _result_frame(
                rid,
                ok=False,
                error={
                    "code": UNAVAILABLE,
                    "message": "brain rpc disabled on host",
                    "retryable": True,
                    "details": {},
                },
                meta={"duration_ms": 0, "host_time": _host_time()},
            )

        # Bound concurrency: refuse rather than queue unbounded work.
        if self._in_flight >= self._max_in_flight:
            rid = str((frame or {}).get("request_id") or "")
            return _result_frame(
                rid,
                ok=False,
                error={
                    "code": RATE_LIMITED,
                    "message": "too many in-flight brain rpc calls",
                    "retryable": True,
                    "details": {},
                },
                meta={"duration_ms": 0, "host_time": _host_time()},
            )

        self._in_flight += 1
        try:
            return await self._handle_locked(frame)
        finally:
            self._in_flight -= 1

    async def _handle_locked(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        started = time.monotonic()
        request_id = str((frame or {}).get("request_id") or "")
        profile_for_meta: Optional[str] = None

        def _meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            duration_ms = int((time.monotonic() - started) * 1000)
            m: Dict[str, Any] = {
                "duration_ms": duration_ms,
                "host_time": _host_time(),
            }
            if profile_for_meta:
                m["hermes_profile"] = profile_for_meta
            if extra:
                m.update(extra)
            return m

        try:
            if not isinstance(frame, dict):
                raise BrainRpcError(INTERNAL, "invalid frame")

            if not request_id:
                # Still return a result so the caller can see the failure.
                request_id = ""

            version = frame.get("contract_version", BRAIN_RPC_CONTRACT_VERSION)
            try:
                version_i = int(version)
            except (TypeError, ValueError):
                version_i = -1
            if version_i != BRAIN_RPC_CONTRACT_VERSION:
                raise BrainRpcError(
                    VERSION_UNSUPPORTED,
                    f"contract_version {version!r} unsupported; host supports {BRAIN_RPC_CONTRACT_VERSION}",
                    details={"supported": BRAIN_RPC_CONTRACT_VERSION},
                )

            method = str(frame.get("method") or "").strip()
            if not method:
                raise BrainRpcError(METHOD_NOT_FOUND, "missing method")

            timeout_ms = frame.get("timeout_ms", DEFAULT_TIMEOUT_MS)
            try:
                timeout_ms = int(timeout_ms)
            except (TypeError, ValueError):
                timeout_ms = DEFAULT_TIMEOUT_MS
            if timeout_ms < 1:
                timeout_ms = DEFAULT_TIMEOUT_MS
            if timeout_ms > MAX_TIMEOUT_MS:
                timeout_ms = MAX_TIMEOUT_MS

            auth_ctx = verify_auth(frame.get("auth"), method=method, host=self.host)
            profile_for_meta = auth_ctx.subject.hermes_profile

            handler = HANDLERS.get(method)
            if handler is None:
                raise BrainRpcError(
                    METHOD_NOT_FOUND,
                    f"unknown method {method}",
                    details={"method": method},
                )

            params = frame.get("params") or {}
            if not isinstance(params, dict):
                params = {}

            # Run sync handlers off the event loop; respect timeout.
            loop = asyncio.get_running_loop()

            def _call() -> Dict[str, Any]:
                return handler(params, auth_ctx, self.host)

            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _call),
                    timeout=timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError as exc:
                raise BrainRpcError(
                    TIMEOUT,
                    "host exceeded budget",
                    retryable=True,
                    details={"timeout_ms": timeout_ms},
                ) from exc

            logger.info(
                "brain_rpc ok method=%s request_id=%s tenant=%s instance=%s user=%s duration_ms=%s",
                method,
                request_id,
                auth_ctx.tenant_id,
                auth_ctx.instance_id,
                auth_ctx.subject.portal_user_id,
                int((time.monotonic() - started) * 1000),
            )
            return _result_frame(request_id, ok=True, result=result, meta=_meta())

        except BrainRpcError as err:
            logger.info(
                "brain_rpc error code=%s request_id=%s message=%s",
                err.code,
                request_id,
                err.message,
            )
            return _result_frame(
                request_id,
                ok=False,
                error=err.to_dict(),
                meta=_meta(),
            )
        except Exception as exc:  # noqa: BLE001 - never leak stack to wire
            logger.exception("brain_rpc internal error request_id=%s", request_id)
            return _result_frame(
                request_id,
                ok=False,
                error={
                    "code": INTERNAL,
                    "message": "internal host error",
                    "retryable": True,
                    "details": {},
                },
                meta=_meta(),
            )


# Process-wide default dispatcher (lazy).
_default_dispatcher: Optional[BrainRpcDispatcher] = None


def get_default_dispatcher() -> BrainRpcDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = BrainRpcDispatcher()
    return _default_dispatcher


async def handle_brain_rpc_request(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level convenience used by the relay transport."""
    return await get_default_dispatcher().handle(frame)


def reset_default_dispatcher() -> None:
    """Test helper: drop the process-wide dispatcher so host config reloads."""
    global _default_dispatcher
    _default_dispatcher = None
