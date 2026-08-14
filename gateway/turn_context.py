"""Per-turn context shared between ``GatewayRunner._run_agent_inner`` and the
``TurnRunner`` collaborator (gateway/run.py).

``_run_agent_inner`` historically defined its tool-progress plumbing as nested
closures (``progress_callback`` ~250 LOC, ``send_progress_messages`` ~353 LOC)
that closed over ~20 enclosing locals.  ``TurnContext`` is the extraction seam:
each closed-over local becomes a field on this dataclass, so the closure bodies
can move onto ``TurnRunner`` methods unchanged modulo ``name`` -> ``ctx.name``
rewrites.

Field notes:

- All fields are written once by ``_run_agent_inner`` while wiring up the turn
  (a few — ``_progress_metadata``, ``_progress_reply_to``, ``agent_holder`` —
  are computed slightly later than construction and assigned onto the ctx as
  soon as the original locals were bound).  None of the original closures
  *rebound* their captured names (no ``nonlocal``); mutable state uses the
  same single-element-list containers as before (``last_progress_msg``,
  ``repeat_count``, ...), so mutation stays visible to the outer body through
  the shared objects exactly as it did through the shared closure cells.
- ``_run_still_current`` stays a callable (it captures ``self``/
  ``session_key``/``run_generation``); carrying the callable keeps the
  extracted bodies byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class TurnContext:
    """Closed-over locals of ``_run_agent_inner`` needed by ``TurnRunner``."""

    # --- read-only turn identity / wiring -------------------------------
    source: Any = None
    _run_still_current: Callable[[], bool] = None  # type: ignore[assignment]
    _live_status_adapter: Any = None
    _live_status_mode: str = "off"
    _thinking_enabled: bool = False
    progress_mode: str = "off"
    progress_grouping: str = "grouped"
    tool_progress_enabled: bool = False

    # --- queues ----------------------------------------------------------
    progress_queue: Any = None
    log_queue: Any = None

    # --- mutable single-element containers (shared with the outer body) --
    last_progress_msg: list = field(default_factory=lambda: [None])
    last_tool: list = field(default_factory=lambda: [None])
    last_was_terminal_block: list = field(default_factory=lambda: [False])
    repeat_count: list = field(default_factory=lambda: [0])
    long_tool_hint_fired: list = field(default_factory=lambda: [False])
    agent_holder: list = field(default_factory=lambda: [None])

    # --- constants / cleanup bookkeeping ---------------------------------
    _LONG_TOOL_THRESHOLD_S: float = 30.0
    _cleanup_progress: bool = False
    _cleanup_msg_ids: List[str] = field(default_factory=list)

    # --- progress threading metadata (assigned after construction, before
    #     send_progress_messages is scheduled) ----------------------------
    _progress_metadata: Optional[dict] = None
    _progress_reply_to: Optional[Any] = None

    # ------------------------------------------------------------------
    # run_sync extraction (second wave of the seam): the closed-over locals
    # of ``_run_agent_inner`` that ``run_sync`` (and the four sibling bridge
    # callbacks) captured.  Same rules as above: read-only snapshots or
    # shared mutable containers; ``message`` is the ONE exception — the old
    # closure rebound it via ``nonlocal``, so the rebind sites now write
    # ``ctx.message`` and the outer body reads ``ctx.message`` afterwards.
    # ------------------------------------------------------------------

    # --- the ex-``nonlocal`` turn message (rebindable) --------------------
    message: Optional[str] = None

    # --- turn parameters / config snapshots (read-only in run_sync) -------
    history: Any = None
    context_prompt: Optional[str] = None
    channel_prompt: Optional[str] = None
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    run_generation: Optional[int] = None
    process_task_id: str = ""
    process_baseline: frozenset[str] = field(default_factory=frozenset)
    _interrupt_depth: int = 0
    event_message_id: Optional[str] = None
    moa_config: Optional[dict] = None
    persist_user_message: Optional[Any] = None
    persist_user_timestamp: Optional[float] = None
    # display_kind stamped on the persisted user row at turn start when this
    # turn was self-injected (MessageEvent.internal), e.g.
    # "internal_notification" for async-delegation/background notifications
    # (#82888). DB-only presentation metadata; never sent to the provider.
    persist_user_display_kind: Optional[str] = None
    user_config: Any = None
    enabled_toolsets: Any = None
    disabled_toolsets: Any = None
    log_mode_enabled: bool = False
    interim_assistant_messages_enabled: bool = False
    needs_progress_queue: bool = False

    # --- lazy-imported callables captured from the outer body -------------
    AIAgent: Any = None
    resolve_display_setting: Any = None

    # --- mutable holder cells (shared-list pattern; outer body + the
    #     post-executor closures read mutations through the same objects) --
    result_holder: list = field(default_factory=lambda: [None])
    tools_holder: list = field(default_factory=lambda: [None])
    stream_consumer_holder: list = field(default_factory=lambda: [None])
    streaming_tts_consumer_holder: list = field(default_factory=lambda: [None])

    # --- voice-ack wiring --------------------------------------------------
    _voice_ack_fired: list = field(default_factory=lambda: [False])
    _voice_ack_guild: list = field(default_factory=lambda: [None])
    _voice_ack_loop: Any = None

    # --- hook / status bridge wiring (published at original binding sites) -
    _loop_for_step: Any = None
    _hooks_ref: Any = None
    _status_adapter: Any = None
    _status_chat_id: Any = None
    _status_thread_metadata: Optional[dict] = None

    # --- extracted sibling callbacks (bound TurnRunner methods; run_sync
    #     reads them through the ctx exactly where it used to close over
    #     the sibling closures) ---------------------------------------------
    progress_callback: Optional[Callable] = None
    voice_ack_callback: Optional[Callable] = None
    _step_callback_sync: Optional[Callable] = None
    _event_callback_sync: Optional[Callable] = None
    _status_callback_sync: Optional[Callable] = None

    # --- Slack-native task-card progress (opt-in; #29483) ------------------
    # True when the Slack adapter's ``native_task_cards_enabled()`` opt-in is
    # set for this turn's platform. The ID-bearing lifecycle callbacks are
    # published by TurnRunner (like voice_ack_callback above) so tool starts
    # and completions correlate by real tool-call ID instead of tool name.
    _native_slack_task_cards: bool = False
    native_tool_start_callback: Optional[Callable] = None
    native_tool_complete_callback: Optional[Callable] = None
