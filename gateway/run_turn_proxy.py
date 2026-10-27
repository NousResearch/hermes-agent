"""Forward a gateway turn through the authenticated remote API route."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any, Dict, Optional
from agent.i18n import t
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayProxyTurnMixin:
    async def _run_agent_via_proxy(
        self, message: str, context_prompt: str, history: list[dict[str, Any]],
        source: SessionSource, session_id: str, session_key: str | None = None,
        run_generation: int | None = None, event_message_id: str | None = None,
        scheduled_heartbeat: bool = False,
    ) -> dict[str, Any]:
        """Forward the message to a remote Hermes API server instead of running a local AIAgent.

        Lets a Docker container handle Matrix E2EE while the agent runs on the host with full
        access to local files, memory, skills, and a unified session store. Native gateway
        sources use the dedicated authenticated route so current-chat recall cannot downgrade
        to the profile-global OpenAI surface."""
        from gateway.run import _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS
        from aiohttp import ClientError, ClientSession as _AioClientSession, ClientTimeout

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return self._proxy_error_result(t("gateway.proxy.url_missing"))

        # The proxy key is a per-profile credential read through the installed secret scope:
        # ``_run_agent`` binds the source profile's scope (or the launch profile's once the process
        # multiplexes), and an unscoped single-profile process reads os.environ inside get_secret.
        # Any get_secret() error propagates rather than degrading to the ambient key, which may hold
        # another profile's credential.
        from agent.secret_scope import get_secret_str

        proxy_key = get_secret_str("GATEWAY_PROXY_KEY").strip()

        if not proxy_key:
            return self._proxy_error_result(
                "⚠️ Gateway proxy mode requires GATEWAY_PROXY_KEY so the remote API can "
                "authenticate and preserve chat scope.")

        from gateway.proxy_request import build_proxy_request
        try:
            request = build_proxy_request(proxy_url, proxy_key, source, session_id,
                                          session_key, message, context_prompt, history)
        except (TypeError, ValueError) as exc:
            return self._proxy_error_result(f"⚠️ Gateway proxy context rejected: {exc}")

        _run_still_current = self._run_still_current_fn(session_key, run_generation)

        def _stale_result(what: str) -> dict[str, Any]:
            logger.info(
                "Discarding stale proxy %s for %s — generation %d is no longer current",
                what, session_key or "?", run_generation or 0,
            )
            return {
                "final_response": "", "messages": [], "api_calls": 0, "tools": [],
                "history_offset": len(history), "session_id": session_id, "response_previewed": False,
            }

        _thread_metadata: dict[str, Any] | None = self._thread_metadata_for_source(source, event_message_id)
        _stream_consumer = (
            None if scheduled_heartbeat
            else self._proxy_stream_consumer(source, event_message_id, _thread_metadata, _run_still_current)
        )
        stream_task = asyncio.create_task(_stream_consumer.run()) if _stream_consumer else None

        _adapter = self._delivery_adapter_for(source)
        if _adapter and not scheduled_heartbeat:
            with suppress(Exception):
                await _adapter.send_typing(source.chat_id, metadata=_thread_metadata)

        full_response = ""
        _start = time.time()
        saw_done = False

        def _consume_sse_line(line: str) -> bool:
            """Parse one SSE line into full_response; True when the terminal ``[DONE]`` was seen.

            Malformed frames (bad JSON, ``choices: [null]``, non-dict deltas) are skipped —
            one bad chunk must not abort the whole stream."""
            nonlocal full_response
            line = line.strip()
            if not line.startswith("data: "):
                return False
            data = line[6:]
            if data.strip() == "[DONE]":
                return True
            try:
                choices = json.loads(data).get("choices") or []
                content = choices[0].get("delta", {}).get("content", "") if choices else ""
            except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
                return False
            if content:
                full_response += content
                if _stream_consumer:
                    _stream_consumer.on_delta(content)
            return False

        try:
            # sock_connect bounds the TCP connect phase so an unreachable proxy host
            # (DNS fail, firewall, remote down) fails fast instead of hanging on the OS default.
            _timeout = ClientTimeout(total=0, sock_read=1800, sock_connect=30)
            async with _AioClientSession(timeout=_timeout) as session:
                async with session.post(request.url, json=request.body, headers=request.headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning("Proxy error (%d) from %s: %s", resp.status, proxy_url, error_text[:500])
                        return self._proxy_error_result(t("gateway.proxy.http_error", status=resp.status, error=error_text[:300]))

                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not _run_still_current():
                            return _stale_result("stream")
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if _consume_sse_line(line):
                                saw_done = True
                                break
                        # Exit before the iterator requests another network chunk.
                        if saw_done:
                            break
                        if len(buffer) > _GATEWAY_PROXY_SSE_BUFFER_MAX_CHARS:
                            raise ValueError("Proxy SSE stream exceeded max buffer size without a line boundary")
                    # The final SSE frame may not be newline-terminated: flush the residual
                    # buffer after EOF instead of silently dropping its content.
                    if not saw_done and buffer:
                        saw_done = _consume_sse_line(buffer)
                    if not saw_done:
                        # Clean EOF without [DONE] — the upstream dropped the response
                        # mid-stream. Keep any partial text but say so instead of
                        # presenting the truncation as a complete answer.
                        logger.warning(
                            "Proxy SSE stream from %s ended without [DONE] — response may be truncated "
                            "(%d chars received)", proxy_url, len(full_response),
                        )
                        if not full_response:
                            return self._proxy_error_result(t("gateway.proxy.closed_early"))
        except (ClientError, TimeoutError) as e:
            logger.error("Proxy connection error to %s: %s", proxy_url, e)
            if not full_response:
                return self._proxy_error_result(t("gateway.proxy.connection_error", error=e))
            # Partial response — return what we got
        finally:
            if _stream_consumer:
                _stream_consumer.finish()
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()

        _elapsed = time.time() - _start
        if not _run_still_current():
            return _stale_result("result")
        logger.info(
            "proxy response: url=%s session=%s time=%.1fs response=%d chars",
            proxy_url, (session_id or "")[:20], _elapsed, len(full_response),
        )
        return {
            "final_response": full_response or t("gateway.proxy.no_response"),
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            "api_calls": 1,
            "tools": [],
            "history_offset": len(history),
            "session_id": session_id,
            "response_previewed": _stream_consumer is not None and bool(full_response),
        }

