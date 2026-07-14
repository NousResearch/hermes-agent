"""AG-UI HTTP/SSE server for Hermes.

Exposes a single POST endpoint that accepts an AG-UI ``RunAgentInput`` and
streams AG-UI protocol events (SSE). It runs the synchronous Hermes
``AIAgent`` on a worker thread; assistant text, reasoning, AND server-tool
calls stream live via ``AGUIEventBridge`` (Hermes' ``stream_delta_callback`` /
``reasoning_callback`` / ``tool_progress_callback`` / ``step_callback``).

Because server-tool events stream live, a server tool card renders IN ORDER —
after any preamble text, before the model's follow-up narration — without
buffering text (live token streaming is preserved). The live callbacks do not
carry the model's real tool_call_id, so live server-tool events use generated
ids; that is fine because server tools self-correlate (no client round-trip).

Frontend (client-executed) tools use Hermes' interrupt mechanism — see
``agui_adapter/session.py``. When the model calls one, the run unwinds; the
adapter emits the frontend tool call WITHOUT a result and finishes the run.
Any server-side tools called in the same turn ran first (the batch is
sequential) and streamed their results live. The client executes the frontend
tool and starts a new run with the result appended to ``messages``. Frontend
tools need the model's REAL id (for resume correlation), so they are emitted
POST-run from the message list — the bridge skips them in the live path.

Run framing:

    RUN_STARTED
      -> (live)  TEXT_MESSAGE_* / REASONING_MESSAGE_* / TOOL_CALL_* (server)
      -> (post)  STATE_SNAPSHOT for state-writer tools (message order)
      -> (post)  TOOL_CALL_* for frontend tools (real ids, no result)
    RUN_FINISHED   (or RUN_ERROR on failure)

State-writer tools (declared via ``forwarded_props``) are INTERNAL: their
authoritative UI is the state card driven by ``StateSnapshotEvent``, not a
chatty tool chip. The bridge therefore SUPPRESSES their live ``TOOL_CALL_*`` /
``TOOL_CALL_RESULT`` events; the server emits only the snapshot the call
produced, post-run, in message order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from concurrent.futures import InvalidStateError
from typing import Any, Deque, Dict, List, Optional, Set

from ag_ui.core import (
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from agui_adapter import resume_shim, translate
from agui_adapter.events import AGUIEventBridge
from agui_adapter.session import (
    AgentConfig,
    RunState,
    build_run_agent,
    reset_current_agent,
    reset_current_state,
    set_current_agent,
    set_current_state,
)

logger = logging.getLogger(__name__)

# Install the resume shim (no-op unless a resume run sets the flag).
resume_shim.install()

_FORWARD_HEADERS = ("x-aimock-context", "x-test-id", "x-aimock-strict")


def _new_message_id(prefix: str = "msg") -> str:
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _collect_forward_headers(headers) -> Dict[str, str]:
    return {name: headers.get(name) for name in _FORWARD_HEADERS if headers.get(name)}


def _input_tool_call_ids(messages) -> Set[str]:
    """All tool-call ids already present in the inbound AG-UI messages, so we
    can emit events only for tool calls produced by *this* run."""
    ids: Set[str] = set()
    for m in messages:
        if getattr(m, "role", None) == "assistant":
            for tc in getattr(m, "tool_calls", None) or []:
                ids.add(tc.id)
        elif getattr(m, "role", None) == "tool":
            ids.add(m.tool_call_id)
    return ids


def _new_tool_calls(hermes_messages: List[dict], known_ids: Set[str]):
    """Yield (id, name, arguments) for assistant tool calls not already known."""
    for m in hermes_messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                tcid = tc.get("id")
                if tcid and tcid not in known_ids:
                    fn = tc.get("function") or {}
                    yield tcid, fn.get("name", ""), fn.get("arguments", "{}")


def _server_tool_results_by_name(
    messages: List[dict], known_ids: Set[str], skip_names: Set[str]
) -> Dict[str, Deque[str]]:
    """Build ``{server_tool_name: FIFO of result strings}`` from a run's messages.

    Used to close any live server-tool card whose ``step_callback`` never fired
    (the run unwound via interrupt before the next iteration). Walks NEW tool
    calls in message order, pairs each with its ``role:"tool"`` result, and
    groups the results by name so the bridge's per-name FIFO pairs them with the
    dangling starts. ``skip_names`` excludes frontend + state-writer tools (they
    are not emitted live)."""
    results_by_id: Dict[str, str] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid:
                results_by_id[tcid] = m.get("content", "")
    out: Dict[str, Deque[str]] = {}
    for tcid, name, _args in _new_tool_calls(messages, known_ids):
        if not name or name in skip_names:
            continue
        out.setdefault(name, deque()).append(results_by_id.get(tcid, ""))
    return out


def _run_turn(run_input: RunAgentInput, config: AgentConfig, bridge: AGUIEventBridge,
              fwd_headers: Dict[str, str], approval_cb=None, on_agent=None) -> Dict[str, Any]:
    """Build + configure the agent and run one turn (on a worker thread).

    ``on_agent`` (if given) is called with the constructed agent before the turn
    runs, so the stream can hold a handle to ``agent.interrupt()`` and abort the
    turn if the client disconnects mid-run.
    """
    frontend_schemas = translate.agui_tools_to_openai(run_input.tools)
    frontend_names = translate.frontend_tool_names(run_input.tools)
    context_text = translate.context_to_text(run_input.context)
    # forwarded_props (agent config) and inbound shared state are each injected
    # as their own read-only system message, exactly like context_text.
    props_text = translate.forwarded_props_to_text(run_input.forwarded_props)
    state_text = translate.state_to_text(run_input.state)
    prep = translate.prepare_run(
        run_input.messages,
        context_text=context_text,
        system_texts=[props_text, state_text],
    )

    # Shared state: seed the run-scoped store from inbound state so snapshots
    # carry UI-set keys (e.g. preferences) alongside agent-written keys. Declare
    # which server-executed tools mutate which state key (from forwarded_props).
    state_specs, state_schemas = translate.parse_state_writer_props(run_input.forwarded_props)
    inbound_state = run_input.state if isinstance(run_input.state, dict) else {}
    run_state = RunState(state=dict(inbound_state), specs=state_specs)

    agent = build_run_agent(
        config,
        frontend_tool_schemas=frontend_schemas,
        frontend_tool_names=frontend_names,
        state_writer_specs=state_specs or None,
        state_writer_schemas=state_schemas or None,
        default_headers=fwd_headers or None,
    )
    if on_agent is not None:
        on_agent(agent)
    # Text, reasoning, AND server-tool calls stream live. Classify the tools so
    # the bridge emits live TOOL_CALL_* only for server tools: frontend tools
    # are emitted post-run (real ids), state-writer tools via a snapshot.
    bridge.set_tool_classes(
        frontend_names=frontend_names,
        state_writer_names=set(state_specs),
    )
    agent.stream_delta_callback = bridge.on_text_delta
    agent.reasoning_callback = bridge.on_reasoning_delta
    agent.tool_progress_callback = bridge.on_tool_progress
    agent.step_callback = bridge.on_step
    agent.thinking_callback = None

    token = set_current_agent(agent)
    state_token = set_current_state(run_state)
    resume_token = resume_shim.set_resume(prep.is_resume)

    # Approval bootstrap, established INSIDE the worker thread (the thread-local
    # callback and the interactive-routing contextvar must be set on the thread
    # that actually runs the tool, not the event-loop thread).
    from tools.approval import set_hermes_interactive_context, reset_hermes_interactive_context
    from tools import terminal_tool as _terminal_tool
    from gateway.session_context import set_session_vars, clear_session_vars

    prev_cb = _terminal_tool._get_approval_callback()
    interactive_token = None
    session_tokens = None
    if approval_cb is not None:
        _terminal_tool.set_approval_callback(approval_cb)
        interactive_token = set_hermes_interactive_context(True)
        # session_key (per-thread approval cache/scope keying) + interactive
        # context, but NO `platform`. Setting a platform
        # makes tools.approval._is_gateway_approval_context() return True, which
        # routes check_all_command_guards() into the gateway branch — that
        # branch needs a registered gateway notify callback (which this adapter
        # has none) and otherwise falls through to submit_pending(), so the
        # thread-local interactive approval callback below (the PARK/interrupt
        # mechanism this whole feature is built on) would NEVER fire. Only the
        # CLI-interactive branch consults set_approval_callback(). The env-flag
        # sibling of this same failure class (HERMES_GATEWAY_SESSION /
        # HERMES_EXEC_ASK, read from os.environ, not the contextvar) is
        # neutralized once at create_app(). async_delivery stays False: AG-UI
        # has no background push channel (like the API server).
        session_tokens = set_session_vars(session_key=run_input.thread_id,
                                          async_delivery=False)
    try:
        result = agent.run_conversation(prep.user_message, conversation_history=prep.conversation_history)
    finally:
        if interactive_token is not None:
            reset_hermes_interactive_context(interactive_token)
        if approval_cb is not None:
            _terminal_tool.set_approval_callback(prev_cb)
        if session_tokens is not None:
            clear_session_vars(session_tokens)
        resume_shim.reset_resume(resume_token)
        reset_current_state(state_token)
        reset_current_agent(token)
    return {
        "result": result or {},
        "frontend_names": frontend_names,
        "state_writer_names": set(state_specs),
        "run_state": run_state,
    }


async def _drain(queue):
    """Yield queued events until the DONE sentinel (exclusive)."""
    from agui_adapter import approvals

    while True:
        item = await queue.get()
        if item is approvals.DONE:
            return
        yield item


async def _consume_queue(queue, encoder: EventEncoder, run_input: RunAgentInput):
    """Drain a run's event queue and yield encoded SSE frames, including the one
    terminal frame. Shared by the fresh-run and resume drain paths:

    * a ``(PARK, interrupt)`` marker  -> RUN_FINISHED{outcome:interrupt}, stop;
    * an ``(ERROR, message)`` marker  -> RUN_ERROR, stop (no trailing success);
    * DONE (end of ``_drain``)        -> RUN_FINISHED{outcome:success}.
    """
    from ag_ui.core import RunFinishedSuccessOutcome, RunFinishedInterruptOutcome
    from agui_adapter import approvals

    async for item in _drain(queue):
        if isinstance(item, tuple) and item and item[0] is approvals.PARK:
            logger.info("AG-UI run parked on approval interrupt (thread=%s run=%s)",
                        run_input.thread_id, run_input.run_id)
            yield encoder.encode(RunFinishedEvent(
                thread_id=run_input.thread_id, run_id=run_input.run_id,
                outcome=RunFinishedInterruptOutcome(interrupts=[item[1]])))
            return
        if isinstance(item, tuple) and item and item[0] is approvals.ERROR:
            logger.info("AG-UI run errored (thread=%s run=%s)",
                        run_input.thread_id, run_input.run_id)
            yield encoder.encode(RunErrorEvent(message=item[1]))
            return
        yield encoder.encode(item)
    logger.info("AG-UI run finished (thread=%s run=%s)",
                run_input.thread_id, run_input.run_id)
    yield encoder.encode(RunFinishedEvent(
        thread_id=run_input.thread_id, run_id=run_input.run_id,
        outcome=RunFinishedSuccessOutcome()))


def _approval_timeout() -> float:
    try:
        from tools.approval import _get_approval_timeout

        return float(_get_approval_timeout())
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("Could not read approvals.timeout (%s); using 60s default", exc)
        return 60.0


# Running agents keyed by thread_id, so a client disconnect can interrupt the
# turn. The SAME worker lives across a park->resume (it blocks in the approval
# callback and continues inline on resume), so its agent stays reachable by
# thread_id for BOTH the fresh-run stream and the resume stream. Registered when
# the agent is built, popped in the worker's finally. Single process (one uvicorn
# worker). ASSUMES one active run per thread_id (the AG-UI/CopilotKit model:
# turns on a thread are sequential). Concurrent same-thread runs (client misuse)
# would alias this slot; the compare-and-swap in _unregister_run_agent keeps a
# finishing worker from evicting a newer worker's entry, and a mis-targeted
# best-effort interrupt is bounded (it can never approve anything).
_run_agents: Dict[str, Any] = {}
_run_agents_lock = threading.Lock()


def _register_run_agent(thread_id: str, agent: Any) -> None:
    with _run_agents_lock:
        _run_agents[thread_id] = agent


def _unregister_run_agent(thread_id: str, agent: Any) -> None:
    # Compare-and-swap: only clear the slot if it still holds OUR agent, so a
    # finishing worker never evicts a newer same-thread worker's registration.
    with _run_agents_lock:
        if _run_agents.get(thread_id) is agent:
            _run_agents.pop(thread_id, None)


def _interrupt_run(thread_id: str, run_id: str) -> None:
    """Best-effort: ask the running agent for *thread_id* to interrupt.

    Called when a client disconnects mid-stream (fresh OR resume path) so the
    worker unwinds at its next interrupt check instead of running the whole turn
    for a gone client. Cooperative: a worker blocked in the approval callback
    won't observe it until the approval timeout, and a tool in non-cooperative
    I/O runs to its own timeout.
    """
    with _run_agents_lock:
        agent = _run_agents.get(thread_id)
    if agent is None:
        return
    logger.info("AG-UI client disconnected; interrupting run (thread=%s run=%s)",
                thread_id, run_id)
    try:
        agent.interrupt("AG-UI client disconnected")
    except Exception:  # noqa: BLE001 - interrupt is best-effort
        logger.debug("agent.interrupt() on disconnect failed", exc_info=True)


async def _event_stream(run_input: RunAgentInput, encoder: EventEncoder,
                        config: AgentConfig, fwd_headers: Dict[str, str]):
    from agui_adapter import approvals

    loop = asyncio.get_running_loop()

    # ---- Resume run: re-attach to a parked worker and resolve its decision ----
    if getattr(run_input, "resume", None):
        parked = approvals.take(run_input.thread_id)
        yield encoder.encode(RunStartedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
        if parked is None:
            logger.info("AG-UI resume with no parked approval (thread=%s run=%s)",
                        run_input.thread_id, run_input.run_id)
            yield encoder.encode(RunErrorEvent(
                message="No pending approval for this thread (expired, unknown, or server restarted)."))
            return
        logger.info("AG-UI resume re-attaching parked worker (thread=%s run=%s)",
                    run_input.thread_id, run_input.run_id)
        queue = parked.queue
        # Resolve the decision to unblock the parked worker, then drain the SAME
        # queue the worker resumes writing to as it continues inline.
        try:
            decision = approvals.resume_to_decision(run_input.resume, parked.pending.interrupt_id)
        except Exception:
            logger.exception("AG-UI resume decode failed")
            # Unblock the parked worker (deny) so it can't leak, then report.
            try:
                if not parked.pending.decision.done():
                    parked.pending.decision.set_result("deny")
            except Exception:
                pass
            yield encoder.encode(RunErrorEvent(message="Invalid resume payload."))
            return
        if not parked.pending.allow_permanent and decision == "always":
            decision = "session"
        try:
            parked.pending.decision.set_result(decision)
        except InvalidStateError:
            # The worker already claimed the future on timeout (deny). Report cleanly.
            logger.info("AG-UI resume arrived after approval timeout (thread=%s run=%s)",
                        run_input.thread_id, run_input.run_id)
            yield encoder.encode(RunErrorEvent(message="Approval already timed out."))
            return
        # The resumed turn runs inline on the ORIGINAL (parked) worker, whose
        # agent is still registered under this thread_id — so a disconnect here
        # (often the longest leg: the approved command actually executes) can
        # interrupt it, same as a fresh run.
        try:
            async for frame in _consume_queue(queue, encoder, run_input):
                yield frame
        except (asyncio.CancelledError, GeneratorExit):
            _interrupt_run(run_input.thread_id, run_input.run_id)
            raise
        return

    # ---- Fresh run ----
    if approvals.is_parked(run_input.thread_id):
        logger.info("AG-UI fresh run rejected; approval already parked (thread=%s run=%s)",
                    run_input.thread_id, run_input.run_id)
        yield encoder.encode(RunStartedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
        yield encoder.encode(RunErrorEvent(
            message="A pending approval must be resolved before starting a new run on this thread."))
        return

    logger.info("AG-UI run start (thread=%s run=%s)", run_input.thread_id, run_input.run_id)
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            logger.debug("AG-UI emit after loop close; dropping event", exc_info=True)
            return

    bridge = AGUIEventBridge(emit)
    known_ids = _input_tool_call_ids(run_input.messages)

    approval_cb = approvals.make_approval_callback(
        thread_id=run_input.thread_id,
        emit=emit, queue=queue,
        last_tool_call_id=bridge.last_tool_call_id,
        new_id=_new_message_id,
        timeout=_approval_timeout(),
    )

    worker_agent: Dict[str, Any] = {"agent": None}

    def _on_agent(a) -> None:
        worker_agent["agent"] = a
        _register_run_agent(run_input.thread_id, a)

    def worker() -> None:
        try:
            out = _run_turn(run_input, config, bridge, fwd_headers, approval_cb=approval_cb,
                            on_agent=_on_agent)
            result = out["result"]
            frontend_names = out["frontend_names"]
            state_writer_names = out["state_writer_names"]
            run_state: RunState = out["run_state"]
            messages = result.get("messages") or []

            # Snapshots recorded (in call order) by the state-writer handlers.
            # Consumed FIFO as we walk the state-writer tool calls in message
            # order, so each StateSnapshotEvent is emitted right after the tool
            # call that produced it and carries the full merged state as of then.
            snapshots = list(run_state.snapshots)
            snap_idx = 0

            # Server-tool TOOL_CALL_* / TOOL_CALL_RESULT events already streamed
            # LIVE (in order) via the bridge's tool_progress/step callbacks, so
            # we do NOT re-emit them here (that would double the cards). The
            # post-run pass owns only what the live path deliberately skips:
            #   * state-writer tools -> the StateSnapshotEvent (chip suppressed),
            #   * frontend tools      -> the client handoff (real model ids).
            # Close any assistant text still streaming before these post-run
            # events so a snapshot/handoff never lands inside an open message.
            bridge.finish()

            # Close any live server-tool card whose completion never arrived via
            # step_callback because the run unwound via interrupt right after the
            # tool batch (the mixed server+frontend case). Pairs dangling starts
            # with their message results by name/FIFO; no-op on a normal finish
            # (step_callback already emitted every END/RESULT).
            bridge.flush_pending_server_tools(
                _server_tool_results_by_name(
                    messages, known_ids, frontend_names | state_writer_names
                )
            )

            handed_off = False
            deferred_frontend = []
            for tcid, name, args in _new_tool_calls(messages, known_ids):
                if name in frontend_names:
                    deferred_frontend.append((tcid, name, args))
                    continue
                # After a state-writer tool call, emit the snapshot it produced
                # so the frontend re-renders with the new full shared state. The
                # visible chip was already suppressed in the live path.
                if name in state_writer_names and snap_idx < len(snapshots):
                    emit(StateSnapshotEvent(snapshot=snapshots[snap_idx]))
                    snap_idx += 1
            for tcid, name, args in deferred_frontend:
                handed_off = True
                # Anchor the client-side tool call to a fresh assistant message.
                # The AG-UI → CopilotKit conversion maps TOOL_CALL_START →
                # ActionExecutionStart with parentMessageId; a distinct parent
                # per run keeps multi-turn frontend-tool calls (e.g. the D5
                # 3-pill sequence) individually reconcilable so each turn's
                # handler re-applies instead of a later run reverting to an
                # earlier historical tool call's state.
                parent_id = _new_message_id()
                emit(ToolCallStartEvent(tool_call_id=tcid, tool_call_name=name, parent_message_id=parent_id))
                emit(ToolCallArgsEvent(tool_call_id=tcid, delta=args if isinstance(args, str) else json.dumps(args)))
                emit(ToolCallEndEvent(tool_call_id=tcid))

            # Final assistant text: only on a normal finish (not a client-tool
            # handoff, whose final_response is an interrupt placeholder). Emit a
            # fallback only if nothing streamed live.
            if not handed_off:
                final = result.get("final_response") or ""
                if final and not bridge.emitted_any_text:
                    bridge.on_text_delta(final)
            bridge.finish()
        except Exception:  # noqa: BLE001 - surfaced as RUN_ERROR
            logger.exception("AG-UI run failed")
            # Close any AG-UI lifecycle the failed run left open (streaming
            # text/reasoning, dangling tool-call START) BEFORE the stream emits
            # the terminal RUN_ERROR — otherwise the client sees a message/tool
            # card stuck perpetually "streaming". Cleanup must never mask the
            # original error, so it is itself guarded. It also runs BEFORE the
            # ERROR marker is enqueued so the draining stream never returns
            # (racing the client loop's closure) before this cleanup completes.
            try:
                bridge.close_open()
            except Exception:  # noqa: BLE001
                logger.exception("AG-UI bridge cleanup after run failure failed")
            # Routed through the guarded emit() (the single worker->loop enqueue
            # primitive) so a closed loop — client gone / server shutdown racing a
            # failing run — never crashes this worker thread with an unhandled
            # RuntimeError and silently drops the RUN_ERROR.
            emit((approvals.ERROR, "The agent run failed."))
        finally:
            # No discard here: the parked entry's lifecycle is owned by take()
            # (popped on resume) and the callback's own timeout path (identity-
            # scoped discard). A thread-scoped discard in this finally could
            # delete a newer, unrelated entry parked for the same thread_id.
            # Same guarded emit() path as every other worker->loop enqueue.
            emit(approvals.DONE)
            _unregister_run_agent(run_input.thread_id, worker_agent["agent"])

    threading.Thread(target=worker, name="hermes-agui-run", daemon=True).start()

    yield encoder.encode(RunStartedEvent(thread_id=run_input.thread_id, run_id=run_input.run_id))
    try:
        async for frame in _consume_queue(queue, encoder, run_input):
            yield frame
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected (or the server is shutting down) mid-stream. Ask
        # the running agent to interrupt so the worker unwinds at its next
        # interrupt check instead of running the whole turn for a gone client.
        # Best-effort + cooperative: the daemon worker still drains to DONE onto
        # a queue nobody reads (emit() is guarded against the closed loop).
        _interrupt_run(run_input.thread_id, run_input.run_id)
        raise


def create_app(config: Optional[AgentConfig] = None, *,
               session_token: Optional[str] = None,
               bound_host: str = "127.0.0.1") -> FastAPI:
    """Build the AG-UI FastAPI app.

    ``bound_host`` MUST match the interface the app is actually served on
    (e.g. the ``uvicorn.run(host=...)`` value). The fail-closed token
    requirement (``require_token_or_refuse``) and the DNS-rebind Host guard
    (``host_accepted`` in the ``_security`` middleware) are both enforced
    against ``bound_host``, not against whatever interface uvicorn is told to
    bind to separately -- this function has no way to observe the actual bind
    address. A caller that serves the app on a different interface than the
    ``bound_host`` it passes here is responsible for that mismatch; nothing
    downstream can detect or correct it. ``entry.main()`` is what couples the
    two values today (it derives ``bound_host`` from the same host it passes
    to uvicorn) -- any new entry point MUST preserve that coupling.
    """
    config = config or AgentConfig()

    from agui_adapter.auth import require_token_or_refuse
    require_token_or_refuse(bound_host, session_token)

    # Defense-in-depth for the same failure class as the session-platform bug:
    # inherited HERMES_GATEWAY_SESSION / HERMES_EXEC_ASK route dangerous-command
    # approval into the gateway/ask branch (read straight from os.environ, not
    # overridable by the worker's interactive contextvar), silently disabling
    # the native interrupt-approval flow. An AG-UI process is never a
    # gateway/ask context, so clear them here (once, before any worker runs).
    from agui_adapter import approvals
    _cleared_env = approvals.neutralize_interrupt_preempting_env()
    if _cleared_env:
        logger.warning(
            "Ignoring inherited %s in the AG-UI server process: they would "
            "divert dangerous-command approval into the gateway/ask branch and "
            "silently disable the native interrupt-approval flow. Cleared for "
            "this process.", ", ".join(_cleared_env))

    app = FastAPI(title="Hermes AG-UI Adapter")

    from fastapi.responses import JSONResponse
    from agui_adapter.auth import host_accepted, token_valid

    @app.middleware("http")
    async def _security(request: Request, call_next):
        if not host_accepted(request.headers.get("host", ""), bound_host):
            return JSONResponse(status_code=400, content={"detail": "Invalid Host header."})
        if request.method == "POST" and not request.headers.get("content-type", "").lower().startswith("application/json"):
            # A cross-site JSON POST is no longer a CORS "simple request", so the
            # browser preflights it and (absent CORS headers) blocks it. Legit
            # AG-UI/CopilotKit clients already send application/json.
            return JSONResponse(status_code=415, content={"detail": "Content-Type must be application/json."})
        # /health needs no token (the Host guard above still applies); everything
        # else is token-gated when a session token is configured.
        if session_token and request.url.path != "/health":
            if not token_valid(request, session_token):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/")
    async def run_agent_endpoint(request: Request) -> Response:
        # Returns a StreamingResponse (SSE) on the happy path, or a JSONResponse
        # on a malformed body — hence the Response supertype annotation.
        try:
            body = await request.json()
            run_input = RunAgentInput.model_validate(body)
        except Exception:  # noqa: BLE001 - malformed JSON or schema violation
            return JSONResponse(status_code=400, content={"detail": "Invalid request body."})
        encoder = EventEncoder(accept=request.headers.get("accept"))
        fwd_headers = _collect_forward_headers(request.headers)
        return StreamingResponse(
            _event_stream(run_input, encoder, config, fwd_headers),
            media_type=encoder.get_content_type(),
        )

    return app
