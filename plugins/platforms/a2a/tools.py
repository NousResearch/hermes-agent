"""A2A client tools (``a2a`` toolset): a2a_discover/call/list/history/orchestrate talk to *other*
agents. Peers come from config.yaml ``a2a_agents: {name: {url, auth: {type: bearer, token}, timeout,
capabilities}}``. Stdlib urllib; wire format is A2A v1.0 ``SendMessage`` (v0.3 replies still parse)."""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from gateway.platforms._shared import coerce_port as _coerce_int

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
_ORCHESTRATE_MAX_WORKERS = 6  # max parallel peers for fan-out


def _load_config() -> dict:
    """Read-only view of config.yaml; peers are only read, never mutated (cache-safe)."""
    from hermes_cli.config import load_config_readonly
    return load_config_readonly() or {}


def _configured_peers() -> dict:
    return _load_config().get("a2a_agents") or {}


def _peer_from_entry(entry: dict, **extra: Any) -> dict:
    return {"url": entry.get("url", ""), "auth": entry.get("auth", {}) or {},
            "headers": entry.get("headers", {}) or {},
            "timeout": int(entry.get("timeout", _DEFAULT_TIMEOUT)), **extra}


def _resolve_peer(agent: str) -> Optional[dict]:
    """Peer name -> {url, auth, timeout, capabilities, tenant}, or treat ``agent`` as a URL."""
    if agent.startswith(("http://", "https://")):
        return {"url": agent, "auth": {}, "timeout": _DEFAULT_TIMEOUT, "capabilities": []}
    entry = _configured_peers().get(agent)
    return _peer_from_entry(entry, capabilities=entry.get("capabilities", []) or [], tenant=entry.get("tenant", ""),
                            idempotency=bool(entry.get("idempotency", False)),
                            allowed_rpc_origins=entry.get("allowed_rpc_origins") or []) if entry else None


def _auth_header(auth: dict) -> dict:
    return {"Authorization": f"Bearer {auth['token']}"} if auth and auth.get("type") == "bearer" and auth.get("token") else {}


class _NoCredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail-closed redirect policy for credential-bearing requests.

    A redirect hop is followed with the full header map only when its target
    is same-origin with the ORIGINAL request URL or is one of the operator's
    pinned allowed origins. Any other hop is refused (HTTPError), never
    followed — urllib's built-in cross-host Authorization stripping is
    partial (scheme/port changes, custom headers); we enforce it uniformly.
    """

    def __init__(self, allowed_origins: tuple[str, ...] = ()):
        self.allowed_origins = allowed_origins
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        original = req.full_url
        if _url_same_origin(newurl, original) or any(
                _url_same_origin(newurl, o) for o in self.allowed_origins):
            return super().redirect_request(req, fp, code, msg, headers, newurl)
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"A2A redirect to cross-origin {newurl} refused (not same-origin, not in allowed_rpc_origins)",
            headers, fp)


def _open_url_no_redirect_leak(req: urllib.request.Request, timeout: int,
                               allowed_origins: tuple[str, ...] = ()) -> Any:
    """urlopen with fail-closed cross-origin redirect handling."""
    opener = urllib.request.build_opener(_NoCredentialRedirectHandler(allowed_origins))
    return opener.open(req, timeout=timeout)


def _http_get_json(url: str, headers: dict, timeout: int,
                   allowed_origins: tuple[str, ...] = ()) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-A2A/1.0", **headers}, method="GET")
    with _open_url_no_redirect_leak(req, timeout, allowed_origins) as resp:  # noqa: S310 (configured peers)
        return json.loads(resp.read().decode("utf-8"))


# Retry budget for Cloudflare 524 (origin timeout) on blocking POST, when the
# peer has opted in. A 524 means the proxy gave up on the response, NOT that
# the origin failed — the peer may have already executed the task. Retrying a
# mutating send on a 524 is only safe when the peer deduplicates requests,
# which the operator asserts per peer via `idempotency: true` in the peer
# config. Default: no retry — the indeterminate outcome propagates.
_POST_MAX_RETRIES = 3


def _http_post_json(url: str, body: dict, headers: dict, timeout: int,
                    retry_524: bool = False,
                    allowed_origins: tuple[str, ...] = ()) -> dict:
    data = json.dumps(body).encode("utf-8")
    # Custom peer headers are operator-controlled but Content-Type and
    # A2A-Version are protocol-owned and must not be clobbered.
    # User-Agent stays overridable (some proxies filter user agents).
    hdrs = {
        "User-Agent": "Hermes-A2A/1.0",
        **headers,
        "Content-Type": "application/json",
        "A2A-Version": protocol.PROTOCOL_VERSION,
    }
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    attempts = _POST_MAX_RETRIES if retry_524 else 1
    for attempt in range(attempts):
        try:
            with _open_url_no_redirect_leak(req, timeout, allowed_origins) as resp:  # noqa: S310 (configured peers)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if (retry_524 and e.code == 524 and attempt < attempts - 1):
                # Exponential backoff: 1s, 2s, 4s. Safe as a blocking sleep:
                # plugin tools run on worker threads, never the event loop.
                logger.warning(
                    "A2A: 524 from %s (origin may have completed the task); "
                    "peer opted into idempotent retry, backing off %.0fs",
                    url, 2 ** attempt)
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("A2A retry loop exited without a result")  # pragma: no cover


def _fetch_card(base_url: str, headers: dict, timeout: int,
                allowed_origins: tuple[str, ...] = ()) -> dict:
    """GET the v1.0 agent-card.json; on 404 fall back to the v0.2 agent.json alias."""
    base = base_url.rstrip("/")
    try:
        return _http_get_json(base + "/.well-known/agent-card.json", headers, timeout, allowed_origins)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return _http_get_json(base + "/.well-known/agent.json", headers, timeout, allowed_origins)


def _select_jsonrpc_interface(card: Optional[dict]) -> Optional[dict]:
    if isinstance(card, dict):
        for iface in card.get("supportedInterfaces", []) or []:
            if isinstance(iface, dict) and iface.get("protocolBinding") == "JSONRPC" and iface.get("url"):
                return iface
    return None


def _rpc_url(base_url: str, card: Optional[dict]) -> str:
    """Card's JSONRPC interface (v1.0 supportedInterfaces) > card's legacy top-level url > base."""
    if iface := _select_jsonrpc_interface(card):
        return str(iface["url"])
    if isinstance(card, dict) and isinstance(card.get("url"), str) and card["url"]:
        return card["url"]
    return base_url.rstrip("/")


def _url_origin(url: str) -> tuple[str, str]:
    """(scheme, host:port) of a URL, lowercased; port defaulted per scheme."""
    parsed = urllib.parse.urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    # parsed.port is None when absent; explicit :0 is a real (if unroutable)
    # port and must not be silently defaulted.
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), f"{host}:{port}"


def _url_same_origin(candidate: str, configured: str) -> bool:
    """True when candidate and configured share scheme + host + port."""
    try:
        return _url_origin(candidate) == _url_origin(configured)
    except ValueError:
        return False


def _allowed_rpc_origins(peer: dict) -> list[str]:
    """Operator-pinned cross-origin RPC URLs exempt from the origin check.

    Entries are compared by ORIGIN (scheme + host + port), so an entry pins
    the whole service, not one exact path.
    """
    raw = peer.get("allowed_rpc_origins") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(u).rstrip("/") for u in raw if str(u).strip()]


def _origin_allowed(candidate: str, peer: dict) -> bool:
    """True when candidate's origin is the configured origin or a pinned
    allowed origin (origin-level match, not exact string)."""
    try:
        cand = _url_origin(candidate)
    except ValueError:
        return False
    # Same origin as the configured base URL (any path) is always allowed —
    # a card may move the RPC within its own service.
    try:
        if _url_same_origin(candidate, peer.get("url", "")):
            return True
    except ValueError:
        pass
    for entry in _allowed_rpc_origins(peer):
        try:
            if _url_origin(entry) == cand:
                return True
        except ValueError:
            continue
    return False


def _send_task(agent_label: str, peer: dict, message: str, context_id: str) -> tuple[str, str, str]:
    """One SendMessage to a peer -> (reply_text, context_id, state). Raises urllib errors /
    ValueError for the caller to format; handles redaction, audit, persistence, metrics."""
    base_url = peer.get("url", "")
    # Per-peer custom headers over the derived auth header; a custom
    # Authorization is operator-controlled proxy auth and wins, with a
    # warning so the override is never silent.
    headers = {**_auth_header(peer.get("auth", {}) or {}), **(peer.get("headers", {}) or {})}
    timeout = int(peer.get("timeout", _DEFAULT_TIMEOUT))
    auth = peer.get("auth", {}) or {}
    if auth and any(k.lower() == "authorization" for k in (peer.get("headers", {}) or {})):
        logger.warning(
            "A2A: peer '%s' custom headers override the derived Authorization "
            "header — deliberate proxy auth schemes only",
            agent_label)
    # Operator asserts this peer dedupes on message identity, making 524
    # retries and stream-fallback resends safe. Without it, a 524 (origin
    # may have completed the task) is surfaced, never retried.
    idempotency = bool(peer.get("idempotency", False))
    allowed = tuple(_allowed_rpc_origins(peer))
    try:
        card = _fetch_card(base_url, headers, min(timeout, 30), allowed)  # best-effort, to learn the rpc URL
    except Exception:
        card = None
    ctx = context_id or protocol.new_context_id()
    safe_message = security.redact_outbound(message)
    # Deterministic task_id for idempotent retries: the same (context, message)
    # pair always produces the same task ID, so a 524/stream-fallback retry
    # hits the peer with an ID it has already seen and can deduplicate against.
    task_id = protocol.deterministic_task_id(ctx, safe_message)
    # v1.0: contextId lives inside the Message, not at the params top level.
    rpc_body = {"jsonrpc": "2.0", "id": task_id, "method": "SendMessage",
                "params": {"taskId": task_id,
                           "message": protocol.text_message(protocol.ROLE_USER, safe_message, context_id=ctx)}}
    iface = _select_jsonrpc_interface(card)
    tenant = str(iface["tenant"]) if iface and iface.get("tenant") else str(peer.get("tenant") or "")
    if tenant:
        rpc_body["params"]["tenant"] = tenant
    security.audit("outbound", agent_label, rpc_body["id"], safe_message)
    protocol.persist_message(ctx, "user", safe_message, rpc_body["id"])
    protocol.metrics.outbound_total += 1
    rpc_url = _rpc_url(base_url, card)
    if not _origin_allowed(rpc_url, peer):
        # The card advertised an RPC interface on a different origin than the
        # configured base URL. Sending there would forward operator secrets
        # (bearer tokens, proxy service tokens) to a card-controlled host.
        # Refuse: fall back to the configured origin, never follow the card.
        logger.warning(
            "A2A: peer '%s' card advertised cross-origin RPC URL %s; not in "
            "peer's allowed_rpc_origins — using configured origin %s instead",
            agent_label, rpc_url, base_url)
        rpc_url = base_url.rstrip("/")

    # Streaming path: if the peer advertises streaming, SendStreamingMessage
    # keeps bytes flowing (SSE keepalives) so proxies with idle timeouts
    # (e.g. Cloudflare's ~100s) do not kill long-running turns. The vetted
    # rpc_url above is reused for both paths (never re-derived from the card).
    if isinstance(card, dict) and (card.get("capabilities") or {}).get("streaming"):
        try:
            return _send_task_stream(agent_label, rpc_url, rpc_body,
                                     headers, timeout, ctx, rpc_body["id"])
        except _A2aTransportError as exc:
            # Zero-frame transport failure: task provably never reached the
            # peer's engine -> message/send is a clean first dispatch.
            # Frames-received death: outcome INDETERMINATE (peer may have
            # executed). Fall back only when the operator asserted
            # idempotency for this peer (same contract as 524 retry);
            # otherwise return an explicit indeterminate result.
            frames_seen = getattr(exc, "frames_received", False)
            if frames_seen and not idempotency:
                logger.warning(
                    "A2A: streaming send for %s died after frames (%s); "
                    "outcome indeterminate — NOT falling back to message/send",
                    agent_label, exc)
                return _indeterminate_outcome(agent_label, exc, ctx, rpc_body["id"])
            if frames_seen:
                logger.warning(
                    "A2A: streaming send for %s died after frames (%s); "
                    "peer asserted idempotency — falling back to message/send",
                    agent_label, exc)
            else:
                logger.debug(
                    "A2A: streaming send failed for %s (%s); falling back to message/send",
                    agent_label, exc)
        except urllib.error.HTTPError as exc:
            if exc.code not in _STREAM_FALLBACK_HTTP_CODES:
                raise
            logger.debug(
                "A2A: streaming endpoint returned %s for %s; falling back to message/send",
                exc.code, agent_label)
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
            # Connection-level failures (DNS, refused, reset, bad framing,
            # read timeout) before any frame -> clean fallback.
            logger.debug(
                "A2A: streaming connection failed for %s (%s); falling back to message/send",
                agent_label, exc)

    resp = _http_post_json(rpc_url, rpc_body, headers, timeout, retry_524=idempotency)
    if "error" in resp:
        raise ValueError(f"Peer '{agent_label}' returned an error: {resp['error'].get('message', resp['error'])}")
    payload = protocol.unwrap_send_message_response(resp.get("result", {}))
    reply = _reply_text_from_result(payload)
    reply_ctx, state = ctx, ""
    if isinstance(payload, dict):
        reply_ctx = payload.get("contextId", ctx)
        state = (payload.get("status") or {}).get("state", "")
    protocol.persist_message(reply_ctx, "agent", reply, rpc_body["id"])
    protocol.metrics.inbound_total += 1
    return reply, reply_ctx, state


# HTTP statuses where the peer effectively does not serve the streaming
# endpoint (card advertised streaming, endpoint disagrees) -> fall back.
_STREAM_FALLBACK_HTTP_CODES = frozenset({404, 405, 501})

# SSE per-read idle cap. Server keepalives arrive every ~5s; 30s of silence
# means the stream is starved, not merely slow.
_STREAM_READ_TIMEOUT_S = 30.0


class _A2aTransportError(ValueError):
    """Transport-level failure of the A2A streaming path.

    Distinct from application-level JSON-RPC errors so _send_task can fall
    back to message/send for transport problems (endpoint missing, stream
    died, truncated response) WITHOUT resubmitting a task the peer already
    processed and rejected at the application level.

    ``frames_received`` records whether the dead stream had produced any
    frames before the failure. A zero-frame failure is a provably clean
    first dispatch (message/send fallback is safe). A frames-received
    failure is an INDETERMINATE outcome: it falls back only when the peer
    config asserts idempotency (same contract as 524 retry); otherwise the
    caller must not resubmit.

    ``seen_ctx`` is the last contextId the stream established;
    ``frame_count`` is how many frames arrived before the stream died.
    """

    def __init__(self, *args: object, frames_received: bool = False,
                 seen_ctx: str = "", frame_count: int = 0) -> None:
        super().__init__(*args)
        self.frames_received = frames_received
        self.seen_ctx = seen_ctx
        self.frame_count = frame_count


def _http_post_sse(url: str, body: dict, headers: dict, timeout: int):
    """POST with Accept: text/event-stream and yield decoded SSE data payloads.

    Yields each ``data:`` frame's parsed JSON. Malformed data lines are
    skipped silently. Comment lines (keepalives) and event/id fields are
    consumed without yielding. Raises urllib errors / _A2aTransportError for
    the caller to format.

    Timeout semantics: the socket's per-read timeout is capped at
    ``_STREAM_READ_TIMEOUT_S`` (keepalive-starvation detection — server
    keepalives arrive every ~5s), and the *total* turn is bounded by a
    wall-clock deadline of ``timeout + _STREAM_READ_TIMEOUT_S``. Without the
    deadline, a peer that keeps sending keepalives could hold the stream
    open indefinitely, since a per-read timeout resets on every byte.
    """
    data = json.dumps(body).encode("utf-8")
    # Accept is protocol-owned for this streaming request, alongside
    # Content-Type/A2A-Version (same precedence policy as _http_post_json).
    hdrs = {
        "User-Agent": "Hermes-A2A/1.0",
        **headers,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "A2A-Version": protocol.PROTOCOL_VERSION,
    }
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    deadline = time.monotonic() + timeout + _STREAM_READ_TIMEOUT_S
    with urllib.request.urlopen(req, timeout=min(timeout, _STREAM_READ_TIMEOUT_S)) as resp:  # noqa: S310 (configured peers)
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("text/event-stream"):
            # Peer ignored the stream request; body is a plain JSON-RPC response.
            # A 200 with a non-JSON body (HTML error page, empty body behind
            # a misbehaving proxy) is a transport problem, not an application
            # answer — wrap it so _send_task falls back to message/send
            # instead of hard-failing on an unhandled JSONDecodeError.
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                yield json.loads(raw)
            except (json.JSONDecodeError, RecursionError):
                raise _A2aTransportError(
                    f"non-SSE response was not valid JSON-RPC "
                    f"(Content-Type: {ctype!r}, first bytes: {raw[:64]!r})"
                ) from None
            return
        for raw in resp:
            if time.monotonic() > deadline:
                raise _A2aTransportError(
                    f"stream exceeded total deadline of {timeout}s")
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue  # keepalive comments, event/id fields
            payload = line[len("data:"):].strip()
            try:
                yield json.loads(payload)
            except (json.JSONDecodeError, RecursionError):
                continue  # malformed/hostile frame — skip, do not abort


def _send_task_stream(agent_label: str, rpc_url: str, rpc_body: dict, headers: dict,
                      timeout: int, ctx: str, task_id: str) -> tuple[str, str, str]:
    """Send one SendStreamingMessage and collect the terminal StreamResponse.

    Frames are JSON-RPC-wrapped StreamResponse objects (A2A v1.0 §9.4); the
    stream closes on the terminal state. Returns (reply_text, context_id, state).

    Raises _A2aTransportError when the stream cannot produce a result
    (closed before any result frame, or closed without a terminal state).
    A zero-frame failure is safe to retry via message/send; a frames-received
    failure carries ``frames_received=True`` (plus ``seen_ctx`` and
    ``frame_count``) so the caller can treat it as an indeterminate outcome
    rather than resubmit. Mid-stream connection deaths (wall-clock deadline,
    read timeout, reset) are re-qualified the same way: the reader raises
    them without bookkeeping, so the loop re-tags them against the frames it
    actually consumed before the failure. Raises ValueError for
    application-level JSON-RPC errors — the peer processed and rejected the
    task, so retrying would duplicate side effects.
    """
    rpc_body = dict(rpc_body, method="SendStreamingMessage")
    result = None
    saw_terminal = False
    artifact_parts: list = []
    task_id_out = task_id
    # Peers may establish the context in an early frame and omit it from
    # later ones (spec-legal). Track the latest non-empty value so history
    # lands under the peer's own addressing.
    seen_ctx = ctx
    frame_count = 0
    try:
        for frame in _http_post_sse(rpc_url, rpc_body, headers, timeout):
            if not isinstance(frame, dict):
                continue
            frame_count += 1
            if frame.get("error"):
                # Application-level rejection on a healthy stream: return it
                # to the caller, never fall back (would resubmit the task).
                raise ValueError(f"Peer '{agent_label}' returned an error: "
                                 f"{frame['error'].get('message', frame['error'])}")
            candidate = frame.get("result")
            if not isinstance(candidate, dict):
                continue
            # StreamResponse is member-discriminated (A2A v1.0): task snapshots,
            # statusUpdate events, artifactUpdate events, bare messages.
            if isinstance(candidate.get("artifactUpdate"), dict):
                art = candidate["artifactUpdate"].get("artifact") or {}
                if art.get("parts"):
                    artifact_parts.extend(art["parts"])
                continue
            if isinstance(candidate.get("statusUpdate"), dict):
                upd = candidate["statusUpdate"]
                cand = {"status": upd.get("status") or {},
                        "contextId": upd.get("contextId", ""),
                        "taskId": upd.get("taskId", "")}
            elif isinstance(candidate.get("task"), dict):
                cand = candidate["task"]
            elif isinstance(candidate.get("message"), dict):
                cand = {"status": {"state": "", "message": candidate["message"]},
                        "contextId": candidate["message"].get("contextId", "")}
            else:
                cand = candidate
            result = cand
            if cand.get("taskId"):
                task_id_out = cand["taskId"]  # peer-assigned id keys the history
            frame_ctx = str(cand.get("contextId") or "")
            if frame_ctx:
                seen_ctx = frame_ctx
            state = (cand.get("status") or {}).get("state", "")
            if state in protocol.TERMINAL_STATES:
                saw_terminal = True
                break
    except _A2aTransportError as exc:
        # The SSE reader raises without frame bookkeeping (wall-clock
        # deadline exceeded, non-SSE/non-JSON body guard). Re-qualify it
        # here: once frames have arrived the death is an indeterminate
        # outcome, not a clean zero-frame failure.
        if not exc.frames_received:
            exc.frames_received = frame_count > 0
            exc.seen_ctx = seen_ctx
            exc.frame_count = frame_count
        raise
    except urllib.error.HTTPError:
        # HTTP status errors keep their existing semantics (404/405/501 ->
        # fallback, others propagate) and are handled by _send_task.
        raise
    except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
        # Mid-stream connection death after _http_post_sse may have already
        # yielded frames. Zero-frame stays a clean first dispatch;
        # frames-received is an indeterminate outcome.
        raise _A2aTransportError(
            f"stream failed mid-read: {exc}",
            frames_received=(frame_count > 0), seen_ctx=seen_ctx,
            frame_count=frame_count) from exc
    if result is None:
        raise _A2aTransportError(
            f"Peer '{agent_label}' stream closed without a result",
            frames_received=(frame_count > 0), seen_ctx=seen_ctx,
            frame_count=frame_count)
    if not saw_terminal:
        # Truncated stream (peer died after WORKING): indistinguishable from
        # success if we returned here, so fail loud.
        raise _A2aTransportError(
            f"Peer '{agent_label}' stream closed without a terminal state",
            frames_received=True, seen_ctx=seen_ctx, frame_count=frame_count)
    if artifact_parts:
        # Accumulate every artifact part (multi-artifact/chunked streams);
        # artifacts carry the final output ahead of status messages.
        result = dict(result, artifacts=[{"parts": artifact_parts}])
    reply = _reply_text_from_result(result)
    reply_ctx = result.get("contextId") or seen_ctx
    state = (result.get("status") or {}).get("state", "")
    protocol.persist_message(reply_ctx, "agent", reply, task_id_out)
    protocol.metrics.inbound_total += 1
    return reply, reply_ctx, state


def _indeterminate_outcome(agent_label: str, exc: _A2aTransportError,
                           ctx: str, task_id: str) -> tuple[str, str, str]:
    """Return an explicit indeterminate result instead of resubmitting.

    The stream produced frames but died without a terminal state: the peer
    may have already executed the task, and it has no proven retry-safe
    idempotency contract, so a message/send fallback could run it twice.
    """
    frame_count = getattr(exc, "frame_count", 0) or 0
    reply_ctx = getattr(exc, "seen_ctx", "") or ctx
    reply = (
        f"Peer '{agent_label}' stream died mid-task after {frame_count} "
        f"frame{'s' if frame_count != 1 else ''} — outcome unknown; the task "
        f"may have already run. No automatic retry (a replay could run it "
        f"twice); use a2a_call again with a new message if intended."
    )
    protocol.persist_message(reply_ctx, "agent", reply, task_id)
    protocol.metrics.inbound_total += 1
    return reply, reply_ctx, ""


def _reply_text_from_result(result: Any) -> str:
    result = protocol.unwrap_send_message_response(result)
    if not isinstance(result, dict):
        return str(result)
    # Artifacts first (final output), then status message (interim/clarify), else bare Message.
    for artifact in result.get("artifacts", []) or []:
        txt = protocol.extract_text(artifact)
        if txt:
            return txt
    return protocol.extract_text((result.get("status", {}) or {}).get("message") or result)


_AUTH_ERR = "Error: peer '{agent}' rejected auth (HTTP {code}). Check the configured token."
_HTTP_CALL_ERRORS = {401: _AUTH_ERR, 403: _AUTH_ERR, 429: "Error: peer '{agent}' rate limited us (HTTP 429). Retry later."}

def a2a_discover(args: dict, **_: Any) -> str:
    """Fetch and summarize the Agent Card at ``url``."""
    url = str(args.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required (e.g. http://localhost:9999)."
    try:
        card = _fetch_card(url, {}, _DEFAULT_TIMEOUT)
    except urllib.error.HTTPError as e:
        return f"Error: discovery failed — HTTP {e.code} from {url}."
    except Exception as e:
        return f"Error: could not reach {url} — {e}."
    caps = card.get("capabilities", {}) or {}
    skills = card.get("skills", []) or []
    auth = "yes" if card.get("security") else "no"
    proto = ", ".join(
        f"{i.get('protocolBinding', '?')} v{i.get('protocolVersion', '?')}"
        for i in (card.get("supportedInterfaces", []) or []) if isinstance(i, dict)
    ) or f"v{card.get('protocolVersion', '?')} (pre-1.0 card)"
    lines = [f"Agent: {card.get('name', '?')}", f"Description: {card.get('description', '')}", f"URL: {_rpc_url(url, card)}",
             f"Protocol: {proto}",
             f"Streaming: {bool(caps.get('streaming'))}  Push: {bool(caps.get('pushNotifications'))}  Auth required: {auth}",
             f"Skills ({len(skills)}):"]
    lines.extend(f"  - {s.get('name', s.get('id', '?'))}: {s.get('description', '')}" for s in skills[:20])
    return "\n".join(lines)


def a2a_call(args: dict, **_: Any) -> str:
    """Send a task to a peer (configured name or direct URL); ``context_id`` continues a prior exchange."""
    # Accept common aliases models reach for (observed live: 'agent_name').
    agent = str(args.get("agent") or args.get("agent_name") or args.get("name") or "").strip()
    message = str(args.get("message") or args.get("text") or args.get("task") or "").strip()
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    if not agent or not message:
        return "Error: both 'agent' and 'message' are required."
    peer = _resolve_peer(agent)
    if not peer or not peer.get("url"):
        return f"Error: unknown agent '{agent}'. Configure it under 'a2a_agents' in config.yaml or pass a full http(s):// URL."
    try:
        reply, reply_ctx, state = _send_task(agent, peer, message, context_id)
    except urllib.error.HTTPError as e:
        return _HTTP_CALL_ERRORS.get(e.code, "Error: call to '{agent}' failed — HTTP {code}.").format(agent=agent, code=e.code)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error: call to '{agent}' failed — {e}."
    short_state = state.replace("TASK_STATE_", "").replace("_", "-").lower()  # v0.3 states pass through
    header = f"[{agent} · context {reply_ctx}" + (f" · {short_state}" if state else "") + "]"
    body = reply or "(no text reply)"
    if state == protocol.STATE_INPUT_REQUIRED:
        body += f"\n\n(The peer needs more input — answer by calling a2a_call again with context_id '{reply_ctx}'.)"
    return f"{header}\n{body}"


def a2a_list(args: dict | None = None, **_: Any) -> str:
    """List configured A2A peers, persisted conversations, and metrics."""
    peers = _configured_peers()
    lines = []
    if peers:
        lines.append(f"Configured peers ({len(peers)}):")
        for name, entry in peers.items():
            caps = entry.get("capabilities", [])
            lines.append(f"  - {name}: {entry.get('url', '?')} (auth: {(entry.get('auth') or {}).get('type', 'none')})"
                         + (f" caps: {', '.join(caps)}" if caps else ""))
    else:
        lines.append("No peers configured. Add them under 'a2a_agents' in config.yaml.")
    if convos := protocol.list_conversations():
        lines.append("")
        lines.append(f"Persisted conversations ({len(convos)}) — recall with a2a_history:")
        lines.extend(f"  - {c}" for c in convos[:25])
    m = protocol.metrics.snapshot()
    lines.append("")
    lines.append(f"Metrics: {m['inbound_total']} in / {m['outbound_total']} out, {m['tasks_completed']} completed, "
                 f"{m['tasks_failed']} failed, {m['streams_started']} streams, {m['push_sent']} push sent, "
                 f"{m['anti_loop_triggers']} anti-loop, {m['rate_limit_triggers']} rate-limited, avg {m['avg_latency_ms']}ms")
    return "\n".join(lines)


def a2a_history(args: dict, **_: Any) -> str:
    """Recall a persisted A2A conversation (survives compaction/restarts)."""
    context_id = str(args.get("context_id") or args.get("contextId") or "").strip()
    if not context_id:
        return "Error: 'context_id' is required (see a2a_list for known conversations)."
    limit = max(1, min(_coerce_int(args.get("limit") or 50, 50), 200))
    messages = protocol.load_conversation(context_id, limit=limit)
    if not messages:
        return f"No persisted conversation for context '{context_id}'."
    lines = [f"Conversation {context_id} (last {len(messages)} messages):"]
    for m in messages:
        text = (m.get("text") or "").strip()
        lines.append(f"[{m.get('role', '?')}] {text[:1000] + ' …[truncated]' if len(text) > 1000 else text}")
    return "\n".join(lines)


def _match_peers_by_capability(capability: str) -> list[tuple[str, dict]]:
    """Configured peers that advertise the capability ('*' matches all)."""
    return [(name, entry) for name, entry in _configured_peers().items()
            if capability in (entry.get("capabilities", []) or []) or capability == "*"]


def _call_peer_sync(agent_name: str, peer_entry: dict, message: str, context_id: str = "") -> tuple[str, str]:
    """Call a single peer synchronously -> (agent_name, reply_text)."""
    try:
        reply, _ctx, _state = _send_task(agent_name, _peer_from_entry(peer_entry), message, context_id)
        return (agent_name, reply or "(no reply)")
    except Exception as e:
        return (agent_name, f"Error: {e}")


def a2a_orchestrate(args: dict, **_: Any) -> str:
    """Fan-out a task to peers matching a capability. Modes: ``all``, ``first`` (first successful),
    ``best`` (longest successful — coarse; use ``all`` to judge yourself)."""
    capability = str(args.get("capability") or "").strip()
    message = str(args.get("message") or args.get("task") or "").strip()
    mode = str(args.get("mode") or "all").strip().lower()
    mode = mode if mode in ("all", "first", "best") else "all"
    context_id = str(args.get("context_id") or "").strip()
    if not message:
        return "Error: 'message' is required."
    if not capability:
        return "Error: 'capability' is required (or use '*' for all peers)."
    if not (matches := _match_peers_by_capability(capability)):
        return f"Error: no configured peers advertise capability '{capability}'."
    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(len(matches), _ORCHESTRATE_MAX_WORKERS)) as pool:
        futures = {pool.submit(_call_peer_sync, name, entry, message, context_id): name for name, entry in matches}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
                if mode == "first" and not results[-1][1].startswith("Error:"):
                    for f in futures:  # good reply; cancel peers that haven't started
                        f.cancel()
                    break
            except Exception as e:
                results.append((name, f"Error: {e}"))
    results.sort(key=lambda r: r[0])  # deterministic output
    successes = [(name, reply) for name, reply in results if not reply.startswith("Error:")]
    if mode in ("best", "first"):
        if not successes:
            return "\n".join(["All peers failed:"] + [f"  {name}: {reply}" for name, reply in results])
        name, reply = max(successes, key=lambda r: len(r[1])) if mode == "best" else successes[0]
        return f"[{mode}: {name}]\n{reply}"
    return "\n".join([f"Orchestrated '{capability}' to {len(matches)} peer(s):"]
                     + [line for name, reply in results for line in (f"\n--- {name} ---", reply)])


def _str(description: str) -> dict:
    return {"type": "string", "description": description}


# name -> (handler, description, properties, required)
_TOOLS: dict[str, tuple[Any, str, dict, list[str]]] = {
    "a2a_discover": (a2a_discover,
                     "Fetch and summarize another agent's A2A Agent Card from a URL (its name, description, "
                     "capabilities, and skills). Use this to find out what a remote agent can do before calling it.",
                     {"url": _str("Base URL of the remote A2A agent, e.g. http://localhost:9999")}, ["url"]),
    "a2a_call": (a2a_call,
                 "Send a natural-language task to a remote A2A agent and return its reply. The agent is a peer "
                 "(any A2A-compliant framework), not a sub-agent you control. Pass 'context_id' from a previous "
                 "reply to continue a multi-turn exchange.",
                 {"agent": _str("Configured peer name (from a2a_agents) or a full http(s):// URL."),
                  "message": _str("The task / message to send the peer, in natural language."),
                  "context_id": _str("Optional: context id from a prior reply, to continue the conversation.")},
                 ["agent", "message"]),
    "a2a_list": (a2a_list, "List configured A2A peer agents, persisted A2A conversations, and metrics.", {}, []),
    "a2a_history": (a2a_history,
                    "Recall a persisted A2A conversation transcript by context_id (survives restarts and "
                    "context compaction). Use a2a_list to see known context ids.",
                    {"context_id": _str("Context id of the conversation to recall."),
                     "limit": {"type": "integer", "description": "Max messages to return (default 50, max 200)."}},
                    ["context_id"]),
    "a2a_orchestrate": (a2a_orchestrate,
                        "Fan-out a task to multiple peer agents by capability. Peers are matched from config.yaml "
                        "a2a_agents.*.capabilities. Modes: 'all' (return all replies), 'first' (first successful), "
                        "'best' (longest successful reply).",
                        {"capability": _str("Capability to match (e.g. 'research', 'code') or '*' for all peers."),
                         "message": _str("The task to send to all matching peers."),
                         "mode": {"type": "string", "enum": ["all", "first", "best"], "description": "How to aggregate results. Default: 'all'."},
                         "context_id": _str("Optional: shared context id for all peers.")},
                        ["capability", "message"]),
}


def _a2a_tools_available() -> bool:
    """check_fn: serve the client tools ONLY when the operator opted into A2A (peers under
    ``a2a_agents``, inbound platform enabled, or A2A_PORT set). Fail closed.

    Maintainer-directed (#95681): these registered unconditionally, so every session on every install paid
    ~561 tok/call for tools whose only possible output without config is 'no peers configured'. A2A is
    unrelated to Bot Mode (bots talk over gateway RPCs) — for most installs this toolset is foreign-agent
    plumbing they never enabled. Config adds mid-session surface at the next compaction (#97073).
    """
    cfg = {}
    with contextlib.suppress(Exception):
        cfg = _load_config()
        if cfg.get("a2a_agents"):
            return True
    try:
        if os.getenv("A2A_PORT"):
            return True
        a2a_cfg = (cfg.get("platforms") or {}).get("a2a") or {}
        return bool(isinstance(a2a_cfg, dict) and a2a_cfg.get("enabled"))
    except Exception:  # noqa: BLE001
        return False


def register_tools(ctx) -> None:
    """Register the client tools in the ``a2a`` toolset (config-gated)."""
    for name, (handler, description, properties, required) in _TOOLS.items():
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        ctx.register_tool(name=name, toolset="a2a", handler=handler, description=description,
                          schema={"name": name, "description": description, "parameters": parameters},
                          emoji="\U0001f9e9", check_fn=_a2a_tools_available)  # puzzle piece


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import TypedDict  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
