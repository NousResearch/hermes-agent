"""LitCo turn server: the HTTP + SSE contract a LitKit instance uses to drive one matter's agent.

Contract (LitKit survey, Appendix D; execution ledger decision 5)::

    POST /turn                 -> SSE of agent2-shaped events
    POST /interrupt/{turnId}   -> stop that turn
    GET  /health               -> version, uptime, active turns, matterId
    GET  /deliverables/{id}    -> bytes of a file listed in final.deliverables

Every event is one SSE frame, ``event: <type>`` plus a JSON ``data`` line carrying
``type``, ``turnId``, ``stepId`` (monotonic per turn) and ``ts`` (ms since epoch) beside
the event's own fields. Event order for a turn: ``goal_accepted`` first, ``final`` last,
and in between ``assistant_delta`` / ``assistant_reset`` / ``tool_*`` / ``error_classified``
/ ``loop_halted`` as the agent produces them.

Sessions: one LitKit thread = one ``sessionId`` = one Hermes session. Turns on the same
``sessionId`` run one at a time in arrival order; turns on different ``sessionId``s run
concurrently. There is no cumulative token ceiling; ``budgetMs`` (if given) interrupts the
turn when the wall clock runs out.

The agent itself is behind the :class:`TurnRunner` interface so the server can be tested
with a fake runner; :mod:`litco.hermes_runner` is the production runner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from aiohttp import ClientSession, ClientTimeout, web

from litco import __version__
from litco.assertion import secret_matches, verify_user_assertion
from litco.homes import (changed_deliverables, inbox_dir, matter_home, pop_registered, resolve_deliverable,
                         safe_segment, snapshot_deliverables, thread_home)

logger = logging.getLogger("litco.turn_server")

HOST_SECRET_HEADER = "X-Host-Secret"
ACTING_USER_HEADER = "X-LitKit-Acting-User"
ASSERTION_HEADER = "X-LitKit-User-Assertion"
CHANNELS = ("slack", "web", "telegram")
KINDS = ("channel", "dm")
KEEPALIVE_SECONDS = 15.0
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 512 * 1024 * 1024
FINISHED_TURN_RETENTION = 256


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------

@dataclass
class TurnRequest:
    matter_id: str
    user_id: str
    session_id: str
    text: str
    attachments: List[Dict[str, Any]]
    channel: str
    kind: str
    budget_ms: Optional[int] = None
    acting_user: Optional[str] = None  # set only when a user assertion verified


@dataclass
class TurnOutcome:
    """What a runner returns. ``halted`` names a ``loop_halted`` reason; ``error`` a failure message."""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    model_used: Optional[str] = None
    halted: Optional[str] = None
    halted_explanation: Optional[str] = None
    error: Optional[str] = None
    error_category: str = "unknown"


class TurnContext:
    """Everything a runner needs for one turn. ``emit`` and ``interrupt`` are thread-safe."""

    def __init__(self, *, turn_id: str, request: TurnRequest, home: Path, cwd: Path,
                 emit: Callable[[str, Dict[str, Any]], None]):
        self.turn_id = turn_id
        self.request = request
        self.home = home
        self.cwd = cwd
        self.local_attachments: List[Dict[str, Any]] = []
        self._emit = emit
        self._lock = threading.Lock()
        self._interrupted = threading.Event()
        self.interrupt_reason: Optional[str] = None
        self._interrupt_hooks: List[Callable[[str], None]] = []

    # -- events ----------------------------------------------------------------
    def emit(self, event_type: str, **fields: Any) -> None:
        self._emit(event_type, fields)

    # -- interruption ------------------------------------------------------------
    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def on_interrupt(self, hook: Callable[[str], None]) -> None:
        """Register ``hook(reason)``; runs immediately when the turn is already interrupted."""
        with self._lock:
            if not self._interrupted.is_set():
                self._interrupt_hooks.append(hook)
                return
            reason = self.interrupt_reason or "interrupted"
        _safe(hook, reason)

    def interrupt(self, reason: str = "interrupted") -> bool:
        with self._lock:
            if self._interrupted.is_set():
                return False
            self.interrupt_reason = reason
            self._interrupted.set()
            hooks, self._interrupt_hooks = self._interrupt_hooks, []
        for hook in hooks:
            _safe(hook, reason)
        return True

    def wait_interrupted(self, timeout: Optional[float] = None) -> bool:
        return self._interrupted.wait(timeout)


class TurnRunner(Protocol):
    def run(self, ctx: TurnContext) -> TurnOutcome:  # pragma: no cover - interface
        """Run one turn synchronously (called on a worker thread) and return its outcome."""


def _safe(fn: Callable, *args: Any) -> None:
    try:
        fn(*args)
    except Exception:
        logger.debug("litco hook raised", exc_info=True)


# ---------------------------------------------------------------------------
# Turn bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    turn_id: str
    request: TurnRequest
    started_at: float
    queue: "asyncio.Queue[Optional[bytes]]"
    ctx: Optional[TurnContext] = None
    step: int = 0
    done: bool = False
    task: Optional["asyncio.Task"] = None
    pending_interrupt: Optional[str] = None
    events: List[str] = field(default_factory=list)
    # Runner-thread events wait here until the loop drains them; deque append/popleft are atomic.
    inbox: "deque" = field(default_factory=deque)


class TurnServer:
    """aiohttp application implementing the turn contract for one matter."""

    def __init__(self, runner: TurnRunner, *, host_secret: Optional[str] = None, matter_id: Optional[str] = None,
                 home: Optional[Path] = None, agent_token: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None):
        env = dict(os.environ) if env is None else env
        self.runner = runner
        self.host_secret = host_secret if host_secret is not None else env.get("LITCO_HOST_SECRET", "")
        self.matter_id = matter_id if matter_id is not None else env.get("LITCO_MATTER_ID", "")
        self.home = Path(home).resolve() if home is not None else matter_home(env)
        self.agent_token = agent_token if agent_token is not None else env.get("LITCO_AGENT_TOKEN", "")
        self.started_at = time.time()
        self._turns: Dict[str, _Turn] = {}
        self._finished: List[str] = []
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._runner_site: Optional[web.AppRunner] = None
        self._background: set = set()

    # -- app ---------------------------------------------------------------------
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY_BYTES)
        app.router.add_post("/turn", self.handle_turn)
        app.router.add_post("/interrupt/{turn_id}", self.handle_interrupt)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/deliverables/{file_id}", self.handle_deliverable)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def start(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._runner_site = web.AppRunner(self.build_app())
        await self._runner_site.setup()
        site = web.TCPSite(self._runner_site, host, port)
        await site.start()
        logger.info("litco turn server listening on http://%s:%s for matter %s", host, port, self.matter_id or "?")

    async def stop(self) -> None:
        if self._runner_site is not None:
            await self._runner_site.cleanup()
            self._runner_site = None

    async def _on_shutdown(self, _app: web.Application) -> None:
        for turn in list(self._turns.values()):
            if not turn.done:
                self._interrupt_turn(turn, "shutdown")

    @property
    def active_turn_count(self) -> int:
        return sum(1 for t in self._turns.values() if not t.done)

    # -- auth --------------------------------------------------------------------
    def _authorized(self, request: web.Request) -> bool:
        return secret_matches(request.headers.get(HOST_SECRET_HEADER), self.host_secret)

    @staticmethod
    def _error(status: int, code: str, message: str) -> web.Response:
        return web.json_response({"error": {"code": code, "message": message}}, status=status)

    def _verify_acting_user(self, request: web.Request, matter_id: str) -> tuple:
        """``(acting_user or None, error response or None)``. Both headers or neither."""
        acting = request.headers.get(ACTING_USER_HEADER)
        assertion = request.headers.get(ASSERTION_HEADER)
        if not acting and not assertion:
            return None, None
        if not acting or not assertion:
            return None, self._error(401, "assertion_incomplete",
                                     f"{ACTING_USER_HEADER} and {ASSERTION_HEADER} must be sent together")
        result = verify_user_assertion(assertion, matter_id=matter_id, secret=self.host_secret)
        if not result.ok:
            return None, self._error(401, f"assertion_{result.reason}", "user assertion rejected")
        if result.user_id != acting:
            return None, self._error(401, "assertion_user_mismatch", "acting user does not match the assertion")
        return acting, None

    # -- handlers ----------------------------------------------------------------
    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True, "version": __version__, "hermesVersion": _hermes_version(),
            "uptimeSeconds": round(time.time() - self.started_at, 3),
            "activeTurns": self.active_turn_count, "matterId": self.matter_id or None})

    async def handle_interrupt(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return self._error(401, "unauthorized", "missing or wrong host secret")
        turn = self._turns.get(request.match_info["turn_id"])
        if turn is None:
            return self._error(404, "turn_not_found", "no such turn")
        if turn.done:
            return web.json_response({"ok": True, "turnId": turn.turn_id, "state": "finished"})
        self._interrupt_turn(turn, "interrupted")
        return web.json_response({"ok": True, "turnId": turn.turn_id, "state": "interrupting"}, status=202)

    async def handle_deliverable(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return self._error(401, "unauthorized", "missing or wrong host secret")
        path = resolve_deliverable(self.home, request.match_info["file_id"])
        if path is None:
            return self._error(404, "file_not_found", "no such deliverable")
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{path.name}"'})

    async def handle_turn(self, request: web.Request) -> web.StreamResponse:
        if not self._authorized(request):
            return self._error(401, "unauthorized", "missing or wrong host secret")
        try:
            body = await request.json()
        except Exception:
            return self._error(400, "bad_json", "body must be JSON")
        parsed, err = self._parse_turn(body)
        if err is not None:
            return err
        acting, err = self._verify_acting_user(request, parsed.matter_id)
        if err is not None:
            return err
        if acting is not None and parsed.user_id != acting:
            return self._error(403, "user_mismatch", "userId does not match the asserted user")
        parsed.acting_user = acting

        turn = _Turn(turn_id=f"turn_{uuid.uuid4().hex}", request=parsed, started_at=time.time(),
                     queue=asyncio.Queue())
        self._turns[turn.turn_id] = turn
        loop = asyncio.get_running_loop()
        self._push(turn, "goal_accepted", {"sessionId": parsed.session_id})
        turn.task = loop.create_task(self._execute(turn))
        self._background.add(turn.task)
        turn.task.add_done_callback(self._background.discard)

        response = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
            "X-Turn-Id": turn.turn_id})
        await response.prepare(request)
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(turn.queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if frame is None:
                    break
                await response.write(frame)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # The turn keeps running: its transcript lands in the session, and /interrupt still works.
            logger.info("litco turn %s: client went away; turn continues", turn.turn_id)
            return response
        with _suppress():
            await response.write_eof()
        return response

    # -- parsing -----------------------------------------------------------------
    def _parse_turn(self, body: Any) -> tuple:
        if not isinstance(body, dict):
            return None, self._error(400, "bad_body", "body must be a JSON object")

        def _s(key: str) -> str:
            value = body.get(key)
            return value.strip() if isinstance(value, str) else ""

        matter_id, user_id, session_id, text = _s("matterId"), _s("userId"), _s("sessionId"), body.get("text")
        if not matter_id or not session_id:
            return None, self._error(400, "missing_field", "matterId and sessionId are required")
        if self.matter_id and matter_id != self.matter_id:
            return None, self._error(403, "matter_mismatch", "this host serves a different matter")
        if not isinstance(text, str):
            return None, self._error(400, "missing_field", "text must be a string")
        channel = _s("channel") or "web"
        if channel not in CHANNELS:
            return None, self._error(400, "bad_channel", f"channel must be one of {', '.join(CHANNELS)}")
        kind = _s("kind") or "channel"
        if kind not in KINDS:
            return None, self._error(400, "bad_kind", "kind must be channel or dm")
        if kind == "dm" and not user_id:
            return None, self._error(400, "missing_field", "a dm turn needs userId")
        attachments = body.get("attachments") or []
        if not isinstance(attachments, list) or not all(isinstance(a, dict) for a in attachments):
            return None, self._error(400, "bad_attachments", "attachments must be a list of objects")
        if not text.strip() and not attachments:
            return None, self._error(400, "empty_turn", "text or attachments required")
        budget = body.get("budgetMs")
        if budget is not None and (not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0):
            return None, self._error(400, "bad_budget", "budgetMs must be a positive number")
        return TurnRequest(matter_id=matter_id, user_id=user_id, session_id=session_id, text=text,
                           attachments=attachments, channel=channel, kind=kind,
                           budget_ms=int(budget) if budget is not None else None), None

    # -- event plumbing ----------------------------------------------------------
    def _push(self, turn: _Turn, event_type: str, fields: Dict[str, Any]) -> None:
        """Queue one SSE frame (event-loop thread only)."""
        if turn.done:
            return
        turn.step += 1
        payload = {"type": event_type, "turnId": turn.turn_id, "stepId": turn.step,
                   "ts": int(time.time() * 1000), **fields}
        turn.events.append(event_type)
        data = json.dumps(payload, ensure_ascii=False, default=str)
        turn.queue.put_nowait(f"event: {event_type}\ndata: {data}\n\n".encode("utf-8"))

    def _threadsafe_emitter(self, turn: _Turn, loop: asyncio.AbstractEventLoop) -> Callable[[str, Dict[str, Any]], None]:
        def emit(event_type: str, fields: Dict[str, Any]) -> None:
            if event_type in ("goal_accepted", "final"):
                return  # owned by the server
            turn.inbox.append((event_type, fields))
            try:
                loop.call_soon_threadsafe(self._drain, turn)
            except RuntimeError:
                pass  # loop closed during shutdown
        return emit

    def _drain(self, turn: _Turn) -> None:
        """Move runner events onto the SSE queue in emission order (event-loop thread only)."""
        while turn.inbox:
            event_type, fields = turn.inbox.popleft()
            self._push(turn, event_type, fields)

    def _interrupt_turn(self, turn: _Turn, reason: str) -> None:
        if turn.ctx is not None:
            turn.ctx.interrupt(reason)
        elif turn.pending_interrupt is None:
            turn.pending_interrupt = reason

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = self._session_locks[session_id] = asyncio.Lock()
        return lock

    # -- execution ---------------------------------------------------------------
    async def _execute(self, turn: _Turn) -> None:
        req = turn.request
        loop = asyncio.get_running_loop()
        outcome = TurnOutcome()
        deliverables: List[dict] = []
        budget_handle = None
        lock = self._session_lock(req.session_id)
        try:
            async with lock:
                cwd = thread_home(self.home, req.kind, req.user_id)
                ctx = TurnContext(turn_id=turn.turn_id, request=req, home=self.home, cwd=cwd,
                                  emit=self._threadsafe_emitter(turn, loop))
                turn.ctx = ctx
                if turn.pending_interrupt:
                    ctx.interrupt(turn.pending_interrupt)
                if req.budget_ms:
                    remaining = req.budget_ms / 1000.0 - (time.time() - turn.started_at)
                    budget_handle = loop.call_later(max(remaining, 0.0), ctx.interrupt, "budget_exhausted")
                if ctx.interrupted:
                    outcome = TurnOutcome()
                else:
                    ctx.local_attachments = await self._fetch_attachments(turn, ctx)
                    before = snapshot_deliverables(cwd)
                    outcome = await loop.run_in_executor(None, self.runner.run, ctx)
                    deliverables = changed_deliverables(self.home, cwd, before, pop_registered(turn.turn_id))
        except Exception as exc:  # the runner should not raise; classify if it does
            logger.exception("litco turn %s failed", turn.turn_id)
            outcome = TurnOutcome(error=_short(str(exc)) or exc.__class__.__name__, error_category="unknown")
        finally:
            pop_registered(turn.turn_id)  # never leak a failed turn's registrations
            if budget_handle is not None:
                budget_handle.cancel()
        self._finish(turn, outcome, deliverables)

    def _finish(self, turn: _Turn, outcome: TurnOutcome, deliverables: List[dict]) -> None:
        self._drain(turn)  # every runner event precedes the terminal frames, however the loop scheduled them
        ctx = turn.ctx
        if outcome.error:
            self._push(turn, "error_classified", {"category": outcome.error_category, "message": outcome.error,
                                                  "recovery": "surface"})
        halted = outcome.halted
        if not halted and ctx is not None and ctx.interrupted:
            halted = ctx.interrupt_reason or "interrupted"
        if not halted and turn.pending_interrupt:
            halted = turn.pending_interrupt
        if halted:
            self._push(turn, "loop_halted", {"reason": halted,
                                             "explanation": outcome.halted_explanation or _halt_text(halted)})
        usage: Dict[str, Any] = {"inputTokens": int(outcome.input_tokens or 0),
                                 "outputTokens": int(outcome.output_tokens or 0)}
        if outcome.cache_read_tokens is not None:
            usage["cacheReadTokens"] = int(outcome.cache_read_tokens)
        if outcome.cache_write_tokens is not None:
            usage["cacheWriteTokens"] = int(outcome.cache_write_tokens)
        final: Dict[str, Any] = {"text": outcome.text or "", "citations": [], "usage": usage,
                                 "durationMs": int((time.time() - turn.started_at) * 1000)}
        if outcome.model_used:
            final["modelUsed"] = outcome.model_used
        if deliverables:
            final["deliverables"] = deliverables
        self._push(turn, "final", final)
        turn.done = True
        turn.queue.put_nowait(None)
        self._finished.append(turn.turn_id)
        while len(self._finished) > FINISHED_TURN_RETENTION:
            self._turns.pop(self._finished.pop(0), None)

    async def _fetch_attachments(self, turn: _Turn, ctx: TurnContext) -> List[Dict[str, Any]]:
        """Download attachments that carry a ``url`` into ``inbox/<turnId>/``.

        The request is authenticated with the host's LitKit agent token and, when the turn
        carries one, the acting user's assertion (so the app applies that lawyer's walls).
        Failures are reported to the agent in the prompt, never fatal.
        """
        out: List[Dict[str, Any]] = []
        items = turn.request.attachments
        if not items:
            return out
        folder = inbox_dir(self.home, turn.turn_id)
        headers = {}
        if self.agent_token:
            headers["Authorization"] = f"Bearer {self.agent_token}"
        timeout = ClientTimeout(total=300)
        async with ClientSession(timeout=timeout) as session:
            for index, item in enumerate(items):
                entry = {"fileId": item.get("fileId"), "filename": item.get("filename"), "mime": item.get("mime"),
                         "path": None, "error": None}
                url = item.get("url")
                name = safe_segment(item.get("filename") or item.get("fileId") or f"attachment-{index}")
                if not url:
                    entry["error"] = "no url supplied"
                    out.append(entry)
                    continue
                target = folder / name
                try:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        size = 0
                        with open(target, "wb") as fh:
                            async for chunk in resp.content.iter_chunked(1 << 16):
                                size += len(chunk)
                                if size > MAX_ATTACHMENT_BYTES:
                                    raise RuntimeError("attachment too large")
                                fh.write(chunk)
                    entry["path"] = str(target)
                except Exception as exc:
                    with _suppress():
                        target.unlink()
                    entry["error"] = _short(str(exc)) or exc.__class__.__name__
                out.append(entry)
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


def _short(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _halt_text(reason: str) -> str:
    return {
        "interrupted": "The turn was stopped on request.",
        "budget_exhausted": "The turn ran out of its time budget.",
        "shutdown": "The matter host is shutting down.",
    }.get(reason, "The turn stopped early.")


def _hermes_version() -> Optional[str]:
    """The running Hermes release, as Hermes itself reports it.

    The package metadata says ``0.0.0`` in a source checkout, so ask Hermes's own identity
    resolver (install stamp, then git): e.g. ``0.21.5`` or ``0.21.5+3720.g3754997`` for a fork
    commit past the release tag. Falls back to the release date when neither is known.
    """
    try:
        from hermes_cli.version_info import get_version_info
        info = get_version_info()
        if info.base_version and info.base_version not in ("unknown", "0.0.0"):
            return info.derived_version or info.base_version
    except Exception:
        pass
    try:
        from hermes_cli import __release_date__
        return str(__release_date__)
    except Exception:
        return None


def build_user_message(ctx: TurnContext) -> str:
    """The text handed to the agent: the lawyer's words plus where any attachments landed."""
    req = ctx.request
    lines = [req.text.strip()] if req.text.strip() else []
    if ctx.local_attachments:
        lines.append("")
        lines.append("Attached files:")
        for att in ctx.local_attachments:
            label = att.get("filename") or att.get("fileId") or "file"
            if att.get("path"):
                lines.append(f"- {label} ({att.get('mime') or 'unknown type'}): {att['path']}")
            else:
                lines.append(f"- {label} (LitKit file {att.get('fileId')}; not downloaded: {att.get('error')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sidecar entry point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover - exercised on the host, not in unit tests
    """Run the turn server standalone (sidecar mode) with the Hermes runner."""
    import argparse

    parser = argparse.ArgumentParser(description="LitCo turn server (sidecar mode)")
    parser.add_argument("--host", default=os.environ.get("LITCO_TURN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LITCO_TURN_PORT", "8765")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from litco.hermes_runner import HermesTurnRunner

    server = TurnServer(HermesTurnRunner())
    web.run_app(server.build_app(), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
