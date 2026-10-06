"""Stream diagnostics — per-attempt counters, exception chains, retry logging.

When a streaming request dies mid-response these helpers record WHY (which CF
edge / OpenRouter downstream served it, bytes+chunks before the drop, HTTP
status, underlying httpx error class) to ``agent.log`` in full and to the
user-facing status line compactly. ``run_agent`` keeps thin forwarders so
existing call sites and tests patching ``run_agent.<helper>`` keep working.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lowercased upstream headers captured per attempt for post-hoc analysis.
STREAM_DIAG_HEADERS = (
    "cf-ray", "cf-cache-status", "x-openrouter-provider", "x-openrouter-model", "x-openrouter-id",
    "x-request-id", "x-vercel-id", "via", "server", "x-forwarded-for",
)


def stream_diag_init() -> Dict[str, Any]:
    """Fresh per-attempt diagnostic dict; mutated in place by the streaming functions and read by the retry block."""
    return {
        "started_at": time.time(), "first_chunk_at": None, "chunks": 0, "bytes": 0,
        "headers": {}, "http_status": None, "serving_provider": None,
        # True once a terminal finish_reason is seen on this attempt: tells a mid-flight
        # transport failure (False) from an error raised after completion (#102766).
        "finish_reason_seen": False,
    }


def stream_diag_capture_response(agent: Any, diag: Dict[str, Any], http_response: Any) -> None:
    """Snapshot headers + HTTP status at stream open (so they survive a drop before the first chunk). Best-effort."""
    if http_response is None or not isinstance(diag, dict):
        return
    try:
        diag["http_status"] = getattr(http_response, "status_code", None)
    except Exception:
        pass
    try:
        headers = getattr(http_response, "headers", None) or {}
        captured: Dict[str, str] = {}
        for name in STREAM_DIAG_HEADERS:
            try:
                if val := headers.get(name):
                    captured[name] = str(val)[:120]  # keep log lines bounded
            except Exception:
                continue
        diag["headers"] = captured
    except Exception:
        pass


def flatten_exception_chain(error: BaseException) -> str:
    """Compact ``Outer(msg) <- Inner(msg) <- ...`` rendering, walking ``__cause__`` then ``__context__``
    (deduped, max 4 deep): the OpenAI SDK wraps httpx errors so only the wrapper class is visible at
    the catch site; the inner RemoteProtocolError/ConnectError/ReadError says WHY the stream died."""
    seen: List[BaseException] = []
    link: Optional[BaseException] = error
    while link is not None and len(seen) < 4 and link not in seen:
        seen.append(link)
        nxt = getattr(link, "__cause__", None) or getattr(link, "__context__", None)
        if nxt is None or nxt is link:
            break
        link = nxt

    def render(e: BaseException) -> str:
        msg = str(e).strip().replace("\n", " ")
        msg = msg[:140] + "…" if len(msg) > 140 else msg
        return f"{type(e).__name__}({msg})" if msg else type(e).__name__

    return " <- ".join(render(e) for e in seen) if seen else type(error).__name__


def _diag_fields(diag: Optional[Dict[str, Any]]) -> tuple:
    """(http_status, bytes, chunks, elapsed, ttfb, serving_provider, finish_reason_seen, upstream)
    for the retry log line; ``-`` when unknown."""
    _bytes = _chunks = 0
    _elapsed = 0.0
    _ttfb = _headers_repr = _http_status = _serving_provider = "-"
    _finish_reason_seen = False
    if isinstance(diag, dict):
        try:
            _now = time.time()
            _bytes = int(diag.get("bytes") or 0)
            _chunks = int(diag.get("chunks") or 0)
            _started = float(diag.get("started_at") or _now)
            _elapsed = max(0.0, _now - _started)
            _first = diag.get("first_chunk_at")
            if _first is not None:
                _ttfb = f"{max(0.0, float(_first) - _started):.2f}s"
            headers = diag.get("headers") or {}
            if isinstance(headers, dict) and headers:
                _headers_repr = " ".join(f"{k}={v}" for k, v in headers.items())
            if diag.get("http_status") is not None:
                _http_status = str(diag.get("http_status"))
            if diag.get("serving_provider"):
                _serving_provider = str(diag["serving_provider"])
            _finish_reason_seen = bool(diag.get("finish_reason_seen"))
        except Exception:
            pass
    return _http_status, _bytes, _chunks, _elapsed, _ttfb, _serving_provider, _finish_reason_seen, _headers_repr


def log_stream_retry(
    agent: Any, *, kind: str, error: BaseException, attempt: int, max_attempts: int,
    mid_tool_call: bool, diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Structured WARNING to ``agent.log`` for a transient stream drop + retry, always logged regardless of
    UI verbosity. With *diag*, also records upstream headers, HTTP status, bytes/chunks, elapsed and TTFB on
    the dying attempt — enough to tell "one CF edge / downstream provider" from "random across runs".

    The ``serving_provider`` field names the downstream that actually served the attempt (relays re-roll it
    per request and report it only in the chunk body), so a drop can be attributed to a provider even when
    the response carried no ``x-openrouter-provider`` header."""
    try:
        try:
            _summary = agent._summarize_api_error(error)
        except Exception:
            _summary = str(error)
        if _summary and len(_summary) > 240:
            _summary = _summary[:240] + "…"
        try:
            _chain = flatten_exception_chain(error)
        except Exception:
            _chain = type(error).__name__

        logger.warning(
            "Stream %s on attempt %s/%s — retrying. subagent_id=%s depth=%s provider=%s base_url=%s "
            "error_type=%s error=%s chain=%s http_status=%s bytes=%d chunks=%d elapsed=%.2fs ttfb=%s "
            "serving_provider=%s finish_reason_seen=%s upstream=[%s]",
            kind, attempt, max_attempts,
            getattr(agent, "_subagent_id", None) or "-", getattr(agent, "_delegate_depth", 0),
            agent.provider or "-", agent.base_url or "-",
            type(error).__name__, _summary, _chain, *_diag_fields(diag),
            extra={"mid_tool_call": mid_tool_call},
        )
    except Exception:
        logger.debug("stream-retry log emit failed", exc_info=True)


def emit_stream_drop(
    agent: Any, *, error: BaseException, attempt: int, max_attempts: int,
    mid_tool_call: bool, diag: Optional[Dict[str, Any]] = None,
) -> None:
    """One compact user-visible status line for a stream drop+retry, plus the full WARNING via log_stream_retry.
    ``after Xs`` distinguishes "couldn't connect" (0s) from "died mid-stream" (idle-kill / proxy timeout)."""
    kind = "drop mid tool-call" if mid_tool_call else "drop"
    log_stream_retry(
        agent, kind=kind, error=error, attempt=attempt, max_attempts=max_attempts, mid_tool_call=mid_tool_call, diag=diag
    )
    provider = agent.provider or "provider"
    _suffix = ""
    try:
        started = diag.get("started_at") if isinstance(diag, dict) else None
        if started is not None:
            _suffix = f" after {max(0.0, time.time() - float(started)):.1f}s"
    except Exception:
        pass
    try:
        agent._buffer_diagnostic_status(
            f"⚠️ {provider} stream {kind} ({type(error).__name__}){_suffix} "
            f"— attempt {attempt}/{max_attempts} dropped, reconnecting"
        )
        agent._touch_activity(f"stream retry {attempt}/{max_attempts} after {type(error).__name__}")
    except Exception:
        pass


# Above this size a refused/reset connect is as likely a body-size limit on the endpoint or a proxy
# in front of it as an outage, so the notice says so (#97548: an ~829 KB request failed twice
# before the stream opened while short chats went through).
LARGE_REQUEST_HINT_BYTES = 256 * 1024


def _failed_request(error: BaseException) -> Any:
    """The buffered ``httpx.Request`` carried by the exception chain (SDK wrapper or httpx error), else None."""
    link: Optional[BaseException] = error
    for _ in range(8):
        if link is None:
            return None
        try:
            request = getattr(link, "request", None)
        except RuntimeError:  # httpx raises when the error was built without a request
            request = None
        if request is not None:
            return request
        link = link.__cause__ or link.__context__
    return None


def _request_body_bytes(request: Any) -> Optional[int]:
    content = getattr(request, "content", None)
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, (bytes, bytearray, memoryview)):
        return len(content)
    return None


def connect_exhausted_notice(error: BaseException, *, attempts: int, base_url: Any) -> str:
    """The one user-facing line for \"no stream event ever arrived and the connect retries are spent\":
    names the host, the attempt count and the serialized request size so a body-size limit is
    distinguishable from an outage without reading agent.log."""
    from urllib.parse import urlparse
    request = _failed_request(error)
    host = urlparse(str(getattr(request, "url", None) or base_url or "")).hostname or "the endpoint"
    size = _request_body_bytes(request)
    line = f"❌ Could not open a stream to {host} after {attempts} attempt{'s' if attempts != 1 else ''}"
    if size is None:
        return line + "; the endpoint looks unreachable — try again in a moment."
    line += f" (request {max(1, round(size / 1024))} KB)"
    if size >= LARGE_REQUEST_HINT_BYTES:
        return line + "; the endpoint or a proxy in front of it may reject requests this large."
    return line + "; the endpoint looks unreachable — try again in a moment."


def buffer_connect_exhausted_notice(agent: Any, error: BaseException, *, attempts: int, base_url: Any) -> None:
    """Buffer :func:`connect_exhausted_notice` once per turn: the outer retry/fallback loop re-enters the
    stream call several times, and every pass exhausting the same budget must not add another copy."""
    text = connect_exhausted_notice(error, attempts=attempts, base_url=base_url)
    if any(str(msg) == text for _kind, msg in getattr(agent, "_retry_status_buffer", None) or ()):
        return
    agent._buffer_diagnostic_status(text)


__all__ = [
    "STREAM_DIAG_HEADERS",
    "connect_exhausted_notice",
    "buffer_connect_exhausted_notice",
    "stream_diag_init",
    "stream_diag_capture_response",
    "flatten_exception_chain",
    "log_stream_retry",
    "emit_stream_drop",
]
