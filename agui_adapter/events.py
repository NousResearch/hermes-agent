"""Translate Hermes ``AIAgent`` callbacks into AG-UI protocol events.

The Hermes agent runs synchronously in a worker thread and communicates
progress through a handful of push callbacks (see ``run_agent.AIAgent``):

    stream_delta_callback(text)          # assistant text deltas; None = flush
    reasoning_callback(text)             # provider reasoning/thinking deltas
    tool_progress_callback(evt, name, preview, args, **kw)  # "tool.started" ...
    step_callback(api_call_count, prev_tools)  # prev_tools carry results

This module owns the *translation* only. It converts those callbacks into
AG-UI ``BaseEvent`` objects and hands them to an ``emit`` callable. Thread
safety and transport (SSE) are the server's concern: ``emit`` is expected to
be safe to call from the agent worker thread (e.g. it pushes onto a queue the
async SSE generator drains).

Translates Hermes' streaming callbacks (text deltas, reasoning, tool
progress/steps) into AG-UI protocol events.

Notable behaviors:

* Assistant text is a *lifecycle*: AG-UI needs TEXT_MESSAGE_START before the
  first delta and TEXT_MESSAGE_END after the last. The bridge tracks whether a
  text (or reasoning) message is currently open and opens/closes it lazily.
  Reasoning uses the REASONING_MESSAGE_* lifecycle (not THINKING_*), because
  only that builds a ``role:"reasoning"`` message on the client.
* Tool completion + results arrive one step late via ``step_callback`` (the
  ``prev_tools`` of the *next* iteration). A FIFO queue of
  tool-call ids per tool name matches a completion to the right start, so
  parallel/repeated same-name calls stay correctly paired.

Server-tool events stream LIVE and IN ORDER: ``tool.started`` fires while the
tool executes (same iteration the model requested it), so the TOOL_CALL_START/
ARGS land right after any preamble text and BEFORE the model's next-iteration
follow-up narration. That interleaving is the whole point — it keeps live token
streaming (text is never buffered) while putting the tool card ahead of the
narration that comments on its result, matching the north-star DOM order.

The live path is gated to SERVER tools only (:meth:`set_tool_classes`):

* **Frontend/client tools** carry no result on this run — the model call
  unwinds via interrupt and the client executes them. The server's post-run
  handoff owns them (real model-issued ids, ``parent_message_id``), so the
  bridge SKIPS them entirely in the live path.
* **State-writer tools** are INTERNAL: their authoritative UI is the state
  card driven by a ``StateSnapshotEvent``, not a chatty chip. The bridge
  SUPPRESSES their live TOOL_CALL_* events; the server emits the snapshot.

Live server-tool events use GENERATED ids (Hermes' live callbacks do not carry
the model's real tool_call_id). That is fine for server tools: START/ARGS/END/
RESULT self-correlate via the per-name FIFO and there is no client round-trip
that would need id matching.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set

from ag_ui.core import (
    BaseEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

logger = logging.getLogger(__name__)

Emit = Callable[[BaseEvent], None]


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class AGUIEventBridge:
    """Stateful translator from Hermes agent callbacks to AG-UI events.

    One instance per agent run. Not thread-safe on its own — the Hermes agent
    fires all of these callbacks from a single worker thread, so mutations to
    the open-message / tool-queue state are serialized by construction. The
    injected ``emit`` is the only thing that crosses the thread boundary.
    """

    def __init__(self, emit: Emit, *, id_factory: Callable[[str], str] = _new_id):
        self._emit = emit
        self._id = id_factory
        # Whether any assistant text has been emitted this run (lets the
        # server decide if it must emit a final-response fallback message when
        # a provider path produced no text deltas).
        self.emitted_any_text: bool = False
        # Tool names whose live TOOL_CALL_* events the bridge must NOT emit.
        # Frontend/client tools are owned by the server's post-run handoff
        # (real model ids); state-writer tools are represented by a state card
        # (StateSnapshotEvent), not a chip. Both are suppressed in the live path.
        # Populated per run via ``set_tool_classes``; empty means "every tool is
        # a server tool" (the smoke-script / default case).
        self._frontend_names: Set[str] = set()
        self._state_writer_names: Set[str] = set()
        # Assistant text message currently streaming (None when closed).
        self._text_message_id: Optional[str] = None
        # Thinking message currently streaming (None when closed).
        self._thinking_message_id: Optional[str] = None
        # FIFO of AG-UI tool_call_ids per Hermes tool name, so a completion
        # arriving via step_callback pairs with the right start event.
        self._tool_ids: Dict[str, Deque[str]] = {}
        # Real (model-issued) id overrides, keyed by tool name — used for
        # client-side tools whose id must match the assistant message so the
        # returned ToolMessage correlates. See server-side client-tool path.
        self._id_overrides: Dict[str, Deque[str]] = {}
        # Most recent server-tool call id started this run — the binding target
        # for an approval interrupt. For the case that matters (a `terminal`
        # command hitting the approval gate) this is exact: `terminal` is not in
        # _PARALLEL_SAFE_TOOLS, so any batch containing it runs SEQUENTIALLY
        # (agent/tool_dispatch_helpers._should_parallelize_tool_batch), and on
        # the sequential path Hermes fires tool.started immediately before each
        # tool executes — so when terminal's approval callback fires,
        # _last_tool_call_id already holds terminal's own id. (Only a
        # hypothetical parallel-safe *dangerous* tool — none exists today — could
        # make this the wrong id, since the parallel path starts every call in
        # the batch before any executes.) None until the first server tool.
        self._last_tool_call_id: Optional[str] = None

    # -- per-run tool classification ---------------------------------------

    def set_tool_classes(
        self,
        *,
        frontend_names: Optional[Set[str]] = None,
        state_writer_names: Optional[Set[str]] = None,
    ) -> None:
        """Declare which tool names are frontend (client-executed) and which are
        state-writers, so the live ``on_tool_progress`` / ``on_step`` path emits
        TOOL_CALL_* events ONLY for server tools (names that are neither). See
        the module docstring for why each class is skipped or suppressed."""
        self._frontend_names = set(frontend_names or ())
        self._state_writer_names = set(state_writer_names or ())

    def _is_live_server_tool(self, name: Optional[str]) -> bool:
        """True when *name* is a plain server tool — the only class whose
        TOOL_CALL_* events stream live. Frontend and state-writer tools are
        handled elsewhere (post-run handoff / state snapshot)."""
        return bool(name) and name not in self._frontend_names and name not in self._state_writer_names

    # -- public callback surface (assign these onto the AIAgent) -----------

    def on_text_delta(self, text: Optional[str]) -> None:
        """``stream_delta_callback``. ``None`` flushes/closes the message."""
        if text is None:
            self._close_text()
            return
        if not text:
            return
        # Text and thinking can't interleave in one AG-UI stream — close any
        # open thinking message before assistant text starts.
        self._close_thinking()
        if self._text_message_id is None:
            self._text_message_id = self._id("msg")
            self._emit(TextMessageStartEvent(message_id=self._text_message_id, role="assistant"))
        self.emitted_any_text = True
        self._emit(TextMessageContentEvent(message_id=self._text_message_id, delta=text))

    def on_reasoning_delta(self, text: Optional[str]) -> None:
        """``reasoning_callback``. Provider reasoning/thinking deltas.

        Emits AG-UI ``REASONING_MESSAGE_*`` events (not ``THINKING_*``): only
        the reasoning-message lifecycle is turned into a ``role:"reasoning"``
        message by ``@ag-ui/client`` / CopilotKit, so this is what the client
        actually renders in its reasoning slot.
        """
        if not text:
            return
        # A model that emits reasoning then answer text is the norm; keep them
        # in separate AG-UI message lifecycles.
        self._close_text()
        if self._thinking_message_id is None:
            self._thinking_message_id = self._id("think")
            self._emit(
                ReasoningMessageStartEvent(
                    message_id=self._thinking_message_id, role="reasoning"
                )
            )
        self._emit(
            ReasoningMessageContentEvent(
                message_id=self._thinking_message_id, delta=text
            )
        )

    def on_tool_progress(
        self,
        event_type: str,
        name: Optional[str] = None,
        preview: Optional[str] = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        """``tool_progress_callback``. Only ``tool.started`` is bridged here;
        completion + results come through ``on_step`` where the result payload
        is available. Emission is gated to server tools via
        ``_is_live_server_tool`` (frontend/state-writer tools are handled
        elsewhere)."""
        if event_type != "tool.started" or not name:
            return
        if not self._is_live_server_tool(name):
            return
        args_dict = _coerce_args(args)
        # Close any open assistant/thinking message before a tool call block.
        self._close_text()
        self._close_thinking()

        tc_id = self._next_id_for(name)
        self._tool_ids.setdefault(name, deque()).append(tc_id)
        self._last_tool_call_id = tc_id

        self._emit(ToolCallStartEvent(tool_call_id=tc_id, tool_call_name=name))
        self._emit(ToolCallArgsEvent(tool_call_id=tc_id, delta=json.dumps(args_dict)))

    def on_step(self, api_call_count: int, prev_tools: Any = None) -> None:
        """``step_callback``. ``prev_tools`` lists the tools completed in the
        previous iteration, each carrying its result.

        Only server tools that ``on_tool_progress`` emitted a START for have a
        FIFO queue entry, so frontend / state-writer tools appearing here find
        no queue and are skipped — the END/RESULT gating falls out of that."""
        if not isinstance(prev_tools, list):
            return
        for info in prev_tools:
            name, result = _tool_info(info)
            queue = self._tool_ids.get(name or "")
            if not name or not queue:
                continue
            tc_id = queue.popleft()
            self._emit(ToolCallEndEvent(tool_call_id=tc_id))
            self._emit(
                ToolCallResultEvent(
                    message_id=self._id("res"),
                    tool_call_id=tc_id,
                    content="" if result is None else str(result),
                )
            )
            if not queue:
                self._tool_ids.pop(name, None)

    def flush_pending_server_tools(self, results_by_name: Dict[str, Deque[str]]) -> None:
        """Emit END + RESULT for any live server-tool START still unpaired.

        ``on_step`` pairs a completion when the NEXT iteration begins, but a run
        that unwinds via interrupt right after a tool batch (the mixed
        server+frontend case: the frontend tool interrupts, so there is no next
        iteration) never fires that ``step_callback``. This closes those dangling
        server-tool cards from the post-run message results.

        ``results_by_name`` maps a server-tool name to a FIFO of ALL that name's
        result strings in call order (built by the server from the run's
        messages). Only the still-open starts are closed here; because a start is
        removed from its per-name queue as soon as ``on_step`` pairs it, the ``k``
        pending ids for a name are the LAST ``k`` calls of that name — so we align
        against the TAIL of the results FIFO. Missing entries fall back to an
        empty result so a card is always closed (never left open on the client)."""
        for name, queue in list(self._tool_ids.items()):
            all_results = list(results_by_name.get(name) or ())
            # Pending ids correspond to the last len(queue) calls of this name.
            results = deque(all_results[-len(queue):]) if queue else deque()
            while queue:
                tc_id = queue.popleft()
                result = results.popleft() if results else None
                self._emit(ToolCallEndEvent(tool_call_id=tc_id))
                self._emit(
                    ToolCallResultEvent(
                        message_id=self._id("res"),
                        tool_call_id=tc_id,
                        content="" if result is None else str(result),
                    )
                )
            self._tool_ids.pop(name, None)

    def finish(self) -> None:
        """Close any messages left open when the run ends."""
        self._close_text()
        self._close_thinking()

    def last_tool_call_id(self) -> Optional[str]:
        return self._last_tool_call_id

    def close_open(self) -> None:
        """Close EVERY still-open AG-UI lifecycle — streaming text, reasoning,
        AND any dangling tool-call START — for the error path.

        Unlike :meth:`finish` (which closes only text/reasoning and leaves tool
        starts open for ``on_step`` / the post-run flush to pair with results),
        this force-closes open tool calls by emitting a bare ``TOOL_CALL_END``
        (no result — the run failed, there is none). Used when the run raises
        mid-stream so a terminal ``RUN_ERROR`` is never preceded by an unclosed
        START/message the client would render as perpetually streaming."""
        self._close_text()
        self._close_thinking()
        for name in list(self._tool_ids):
            queue = self._tool_ids.pop(name)
            while queue:
                self._emit(ToolCallEndEvent(tool_call_id=queue.popleft()))

    # -- client-side tool id correlation -----------------------------------

    def bind_client_tool_id(self, name: str, tool_call_id: str) -> None:
        """Force the next ``tool.started`` for *name* to use *tool_call_id*
        (the model-issued id) instead of a generated one, so the AG-UI event
        the client sees matches the assistant message and the returned
        ToolMessage correlates on resume."""
        self._id_overrides.setdefault(name, deque()).append(tool_call_id)

    # -- internals ---------------------------------------------------------

    def _next_id_for(self, name: str) -> str:
        overrides = self._id_overrides.get(name)
        if overrides:
            tc_id = overrides.popleft()
            if not overrides:
                self._id_overrides.pop(name, None)
            return tc_id
        return self._id("tc")

    def _close_text(self) -> None:
        if self._text_message_id is not None:
            self._emit(TextMessageEndEvent(message_id=self._text_message_id))
            self._text_message_id = None

    def _close_thinking(self) -> None:
        if self._thinking_message_id is not None:
            self._emit(ReasoningMessageEndEvent(message_id=self._thinking_message_id))
            self._thinking_message_id = None


def _coerce_args(args: Any) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"raw": args}
    return args if isinstance(args, dict) else {}


def _tool_info(info: Any) -> tuple[Optional[str], Any]:
    """Extract (name, result) from a step_callback prev_tools entry."""
    if isinstance(info, dict):
        name = info.get("name") or info.get("function_name")
        result = info.get("result") if "result" in info else info.get("output")
        return name, result
    if isinstance(info, str):
        return info, None
    return None, None
