"""Per-turn context shared between ``GatewayRunner._run_agent_inner`` and ``TurnRunner``.

Each ex-closure local is a field, written once by ``_run_agent_inner`` while wiring the turn.
``message`` (ex-``nonlocal``) is the only rebindable field; other mutable state uses
single-element lists so mutation stays visible to the outer body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class TurnContext:
    # read-only turn identity / wiring
    source: Any = None
    _run_still_current: Callable[[], bool] = None  # type: ignore[assignment]
    _live_status_adapter: Any = None
    _live_status_mode: str = "off"
    _thinking_enabled: bool = False
    progress_mode: str = "off"
    progress_grouping: str = "grouped"
    tool_progress_enabled: bool = False
    progress_queue: Any = None
    log_queue: Any = None
    # mutable single-element containers (shared with the outer body)
    last_progress_msg: list = field(default_factory=lambda: [None])
    last_tool: list = field(default_factory=lambda: [None])
    last_was_terminal_block: list = field(default_factory=lambda: [False])
    repeat_count: list = field(default_factory=lambda: [0])
    long_tool_hint_fired: list = field(default_factory=lambda: [False])
    agent_holder: list = field(default_factory=lambda: [None])
    _LONG_TOOL_THRESHOLD_S: float = 30.0
    _cleanup_progress: bool = False
    _cleanup_msg_ids: List[str] = field(default_factory=list)
    _progress_metadata: Optional[dict] = None
    _progress_reply_to: Optional[Any] = None
    message: Optional[str] = None  # the only rebindable field
    # turn parameters / config snapshots (read-only in run_sync)
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
    # Raw inbound platform id (not the event_message_id reply anchor); stamped on the user turn.
    inbound_message_id: Optional[str] = None
    moa_config: Optional[dict] = None
    persist_user_message: Optional[Any] = None
    persist_user_timestamp: Optional[float] = None
    # display_kind of the persisted user row for a self-injected turn; DB-only, never sent.
    # "internal_notification" for async-delegation/background notifications (#82888).
    persist_user_display_kind: Optional[str] = None
    user_config: Any = None
    enabled_toolsets: Any = None
    disabled_toolsets: Any = None
    log_mode_enabled: bool = False
    interim_assistant_messages_enabled: bool = False
    needs_progress_queue: bool = False
    AIAgent: Any = None
    resolve_display_setting: Any = None
    result_holder: list = field(default_factory=lambda: [None])
    tools_holder: list = field(default_factory=lambda: [None])
    stream_consumer_holder: list = field(default_factory=lambda: [None])
    streaming_tts_consumer_holder: list = field(default_factory=lambda: [None])
    # voice-ack wiring
    _voice_ack_fired: list = field(default_factory=lambda: [False])
    _voice_ack_guild: list = field(default_factory=lambda: [None])
    _voice_ack_loop: Any = None
    # hook / status bridge wiring
    _loop_for_step: Any = None
    _hooks_ref: Any = None
    _status_adapter: Any = None
    _status_chat_id: Any = None
    _status_thread_metadata: Optional[dict] = None
    # bound TurnRunner callbacks read via ctx
    progress_callback: Optional[Callable] = None
    voice_ack_callback: Optional[Callable] = None
    _step_callback_sync: Optional[Callable] = None
    _event_callback_sync: Optional[Callable] = None
    _status_callback_sync: Optional[Callable] = None
    # Slack-native task cards (opt-in); ID-bearing callbacks correlate start/complete by call ID
    # --- Slack-native task-card progress (opt-in; #29483) ------------------ True when the Slack adapter's
    # ``native_task_cards_enabled()`` opt-in is set for this turn's platform. The ID-bearing lifecycle
    # callbacks are published by TurnRunner (like voice_ack_callback above) so tool starts and completions
    # correlate by real tool-call ID instead of tool name.
    _native_slack_task_cards: bool = False
    native_tool_start_callback: Optional[Callable] = None
    native_tool_complete_callback: Optional[Callable] = None

    # Exact provenance for the synthetic startup recovery turn. Kept last to
    # preserve positional construction compatibility for existing fields.
    startup_resume: bool = False
