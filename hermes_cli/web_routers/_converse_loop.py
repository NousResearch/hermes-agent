"""Dashboard-specific glue for ``WS /api/audio/converse``.

The framework-agnostic VAD/STT/mic-shim core (:class:`_NetworkMicStream`,
:class:`ConverseSession`) now lives in :mod:`tools.voice_converse_loop` so the
aiohttp gateway can reuse it; they are re-exported here so existing importers
(and the dashboard handler in :mod:`hermes_cli.web_routers.audio`) keep working.

What stays here is the tui_gateway/dashboard-specific turn driver:

* :class:`_CaptureTransport` — a :class:`tui_gateway.transport.Transport` that
  funnels the agent's ``message.delta`` text into a callback and signals when
  ``message.complete`` lands, so a REAL main turn can be driven in-process.
* :func:`create_voice_session` / :func:`run_voice_turn` — create/use an ephemeral
  tui_gateway session and dispatch ``prompt.submit`` through
  :func:`tui_gateway.server.dispatch`, streaming assistant deltas via ``on_delta``
  and blocking until the turn ends.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable, Optional

# Re-exported for backwards-compatible imports (tests + the dashboard handler
# import these names from this module); the definitions now live in the neutral
# shared module so the aiohttp gateway can reuse them without a dashboard import.
from tools.voice_converse_loop import (  # noqa: F401
    ConverseSession,
    _NetworkMicStream,
)

_log = logging.getLogger("hermes_cli.web_server")


class _CaptureTransport:
    """A :class:`tui_gateway.transport.Transport` that captures one turn's output.

    ``dispatch`` binds this transport for the request AND ``prompt.submit`` copies
    it onto ``session["transport"]``, so every ``message.*`` event for the turn is
    written here.  ``message.delta`` text feeds ``on_delta``; ``message.complete``
    (or a terminal ``error``) sets :attr:`done`.
    """

    __slots__ = ("_on_delta", "_sid", "done", "error")

    def __init__(self, on_delta: Callable[[str], None], sid: str) -> None:
        self._on_delta = on_delta
        self._sid = sid
        self.done = threading.Event()
        self.error: Optional[str] = None

    def write(self, obj: dict) -> bool:
        """Consume one JSON frame; always reports success (never a gone peer)."""
        try:
            if not isinstance(obj, dict) or obj.get("method") != "event":
                return True
            params = obj.get("params") or {}
            # Only this session's events (dispatch may fan session-less globals here).
            if params.get("session_id") not in (self._sid, ""):
                return True
            etype = params.get("type")
            payload = params.get("payload") or {}
            if etype == "message.delta":
                text = payload.get("text")
                if isinstance(text, str) and text:
                    self._on_delta(text)
            elif etype == "message.complete":
                self.done.set()
            elif etype == "error":
                self.error = str(payload.get("message") or "turn error")
                self.done.set()
        except Exception:  # noqa: BLE001 - a capture bug must not break the turn
            _log.debug("converse capture transport write failed", exc_info=True)
        return True

    def close(self) -> None:
        return None


def create_voice_session(model: Optional[str] = None) -> str:
    """Create a fresh ephemeral tui_gateway session; return its live session id.

    A dedicated session per WebSocket keeps the spoken conversation's history
    isolated (and persisted, like the dashboard chat) without colliding with a
    typed session.
    """
    from tui_gateway.server import dispatch

    # source="voice" tags the durable session row as a chat sub-kind the dashboard
    # labels "Voice" (vs. the default "tui"/platform, which buckets under Automations).
    # _resolve_session_source never rewrites an explicit source, and every subsequent
    # prompt.submit re-binds HERMES_SESSION_SOURCE from this stored value via
    # _set_session_context, so all turns persist as source="voice". The agent PLATFORM
    # is unchanged, leaving toolset resolution untouched.
    params: dict = {"title": "Voice conversation", "source": "voice"}
    if model:
        params["model"] = model
    req = {"jsonrpc": "2.0", "id": f"converse-new-{uuid.uuid4().hex[:8]}",
           "method": "session.create", "params": params}
    resp = dispatch(req, None)
    if not isinstance(resp, dict) or resp.get("error"):
        err = (resp or {}).get("error") if isinstance(resp, dict) else None
        raise RuntimeError(f"could not create voice session: {err}")
    sid = ((resp.get("result") or {}).get("session_id"))
    if not sid:
        raise RuntimeError("session.create returned no session_id")
    return str(sid)


def run_voice_turn(
    session_id: str, text: str, on_delta: Callable[[str], None],
    *, interrupted: bool = False, timeout: float = 300.0,
) -> Optional[str]:
    """Run one main turn for *text* through ``prompt.submit``, streaming deltas.

    Dispatches with a :class:`_CaptureTransport` bound so the agent's streaming
    ``message.delta`` text reaches ``on_delta`` and blocks until the turn
    completes (``message.complete``).  Returns an error string on failure, else
    ``None``.  *interrupted* prepends the barge-in note to the model-bound
    message (client-side barge-in parity).
    """
    from tui_gateway.server import dispatch

    capture = _CaptureTransport(on_delta, session_id)
    params: dict = {"session_id": session_id, "text": text}
    if interrupted:
        params["interrupted"] = True
    req = {"jsonrpc": "2.0", "id": f"converse-turn-{uuid.uuid4().hex[:8]}",
           "method": "prompt.submit", "params": params}
    resp = dispatch(req, capture)
    # prompt.submit replies {"status": "streaming"} inline and runs the turn on a
    # thread that writes message.* through session["transport"] (== capture).
    if isinstance(resp, dict) and resp.get("error"):
        return str((resp.get("error") or {}).get("message") or "prompt.submit failed")
    if not capture.done.wait(timeout=timeout):
        return "voice turn timed out"
    return capture.error
