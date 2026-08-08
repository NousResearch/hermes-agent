"""
Domain State Classes — decomposed state for AppState.

These dataclasses partition AppState's 100+ fields into focused, single-responsibility
domain objects. Each class owns its fields and provides a clear namespace.

Domain organization:
- IdentityState:      model, provider, base_url, platform, session, ACP routing
- RuntimeState:       iteration control, retry counters, tool delay
- FeatureState:       feature flags, pressure thresholds
- CallbackState:      all event callbacks (stream, tool, thinking, etc.)
- OutputState:        print/output configuration
- APIClientState:     HTTP client, auth tokens, API config, prompt caching
- ProviderState:      provider routing, sorting, model settings
- ToolState:          tool registry, toolsets, enforcement
- MemoryState:        memory subsystem state, nudge intervals
- SessionState:       session metadata, logs, messages, checkpoints
- DatabaseState:      SQLite session DB, parent/child session tracking
- TaskState:          todo store, output queue
- TokenState:         token counters, cost tracking, session totals
- InterruptState:     interrupt, approval, delegation, thread locks
- ContextState:       context compressor, compression settings, user turns
- FallbackState:      fallback chain, model selection
- StreamState:        stream callback, break flag
- CredentialState:    credential pool, file handler
- ActivityState:      last activity timestamp and description
- ExtraState:         extra kwargs, private state, image cache
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set

try:
    from agent.context_types import ContextScope
except ImportError:
    class ContextScope:
        LONG_TERM = "long_term"
        MEDIUM_TERM = "medium_term"
        SHORT_TERM = "short_term"


# ─── 1. Identity ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class IdentityState:
    """Identity & configuration — the "who am I" layer."""

    model: str = ""
    provider: str = ""
    base_url: str = ""
    api_key: str = ""
    platform: str = ""
    session_id: str = ""
    agent_name: str = ""

    # ACP command routing (CLI/gateway)
    acp_command: str = ""
    acp_args: List[str] = field(default_factory=list)

    # API mode: "chat_completions", "codex_responses", "anthropic_messages"
    api_mode: str = "chat_completions"

    # Internal base_url variants
    _base_url: str = ""
    _base_url_lower: str = ""


# ─── 2. Runtime ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RuntimeState:
    """Iteration control, retry state, tool delay — the "how do I run" layer."""

    max_iterations: int = 90
    iteration_budget: Any = None  # IterationBudget instance; Any avoids circular import
    tool_delay: float = 1.0
    tool_delay_type: str = "fixed"
    max_tool_call_iterations: int = 90

    # Retry counters (reset per turn)
    _invalid_tool_retries: int = 0
    _invalid_json_retries: int = 0
    _empty_content_retries: int = 0
    _incomplete_scratchpad_retries: int = 0
    _codex_incomplete_retries: int = 0

    # Last content state
    _last_content_with_tools: Any = None
    _mute_post_response: bool = False
    _surrogate_sanitized: bool = False

    # Streaming checkpoint recovery
    _recovered_streaming_checkpoint: Any = None

    # Return context flag
    return_context: bool = False

    # Extra kwargs for model calls
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ─── 3. Features ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FeatureState:
    """Feature flags and pressure thresholds."""

    save_trajectories: bool = False
    verbose_logging: bool = False
    quiet_mode: bool = False
    ephemeral_system_prompt: str = ""
    skip_context_files: bool = False
    pass_session_id: bool = False
    persist_session: bool = True
    compression_enabled: bool = True
    checkpoints_enabled: bool = False

    # Budget pressure thresholds
    _budget_caution_threshold: float = 0.7
    _budget_warning_threshold: float = 0.9
    _budget_pressure_enabled: bool = True
    _context_pressure_warned: bool = False

    # Additional feature flags
    no_prompt_override: bool = False
    enable_mention_suggestions: bool = True
    use_progressive_summarization: bool = False
    enable_flask_agent: bool = False

    # MCP server configurations
    mcpServers: Any = None


# ─── 4. Callbacks ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CallbackState:
    """All event-callback references — stream, tool, thinking, step, etc."""

    stream_delta_callback: Callable = None
    tool_progress_callback: Callable = None
    tool_start_callback: Callable = None
    tool_complete_callback: Callable = None
    clarify_callback: Callable = None
    reasoning_callback: Callable = None
    thinking_callback: Callable = None
    step_callback: Callable = None
    status_callback: Callable = None
    tool_gen_callback: Callable = None
    background_review_callback: Callable = None
    message_callback: Callable = None

    # Internal callback state
    _reasoning_deltas_fired: bool = False


# ─── 5. Output ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class OutputState:
    """Print, output, and log-prefix configuration."""

    _print_fn: Callable = None
    log_prefix: str = ""
    log_prefix_chars: int = 100
    print_callback: Callable = None
    print_color: bool = True
    user_message_color: str = "yellow"
    agent_message_color: str = "green"


# ─── 6. API Client ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class APIClientState:
    """HTTP client, auth tokens, and API-level configuration."""

    client: Any = None  # OpenAI-compatible client
    _client_kwargs: Dict[str, Any] = field(default_factory=dict)
    _anthropic_client: Any = None
    _anthropic_api_key: str = ""
    _anthropic_base_url: str = ""
    _is_anthropic_oauth: bool = False
    _use_prompt_caching: bool = False
    _cache_ttl: str = "5m"
    max_tokens: int = None
    reasoning_config: Dict[str, Any] = None
    prefill_messages: List[Dict[str, Any]] = field(default_factory=list)
    _cached_system_prompt: str = ""


# ─── 7. Provider ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ProviderState:
    """Provider routing, model sorting, and discovery settings."""

    providers_allowed: List[str] = None
    providers_ignored: List[str] = None
    providers_order: List[str] = None
    provider_sort: str = None
    provider_require_parameters: bool = False
    provider_data_collection: str = None

    # Thinking budget
    thinking_budget_tokens: int = 0
    reasoning_settings: Dict[str, Any] = None


# ─── 8. Tools ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolState:
    """Tool registry, toolsets, and enforcement state."""

    tools: List[Dict[str, Any]] = field(default_factory=list)
    valid_tool_names: Set[str] = field(default_factory=set)
    enabled_toolsets: List[str] = None
    disabled_toolsets: List[str] = None
    _tool_use_enforcement: Any = "auto"
    _last_reported_tool: str = ""
    _executing_tools: bool = False
    _task_router: Any = None  # TaskRouter instance


# ─── 9. Memory ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MemoryState:
    """Memory subsystem — store, manager, and nudge intervals."""

    _memory_store: Any = None
    _memory_enabled: bool = False
    _user_profile_enabled: bool = False
    _memory_nudge_interval: int = 10
    _memory_flush_min_turns: int = 6
    _turns_since_memory: int = 0
    _iters_since_skill: int = 0
    _memory_manager: Any = None
    _skill_nudge_interval: int = 10


# ─── 10. Session ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SessionState:
    """Session metadata, file logs, message buffer, checkpoints, trajectories."""

    session_start: Any = None  # datetime
    logs_dir: Any = None  # Path
    session_log_file: Any = None  # Path
    _session_messages: List[Dict[str, Any]] = field(default_factory=list)
    _persist_user_message_idx: Any = None
    _persist_user_message_override: str = None
    _checkpoint_mgr: Any = None
    _checkpoint_enabled: bool = False
    trajectory_logger: Any = None
    _session_start_time: float = 0.0


# ─── 11. Database ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DatabaseState:
    """SQLite session store and session hierarchy."""

    _session_db: Any = None
    _parent_session_id: str = ""
    _last_flushed_db_idx: int = 0


# ─── 12. Task ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TaskState:
    """Todo store and task output queue."""

    _todo_store: Any = None
    _task_output_queue: Any = None


# ─── 13. Token & Cost ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TokenState:
    """Token counters and cost tracking."""

    total_cost: float = 0.0
    _total_tokens: int = 0
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _estimated_usage: Dict[str, Any] = field(default_factory=dict)

    session_total_tokens: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_cache_read_tokens: int = 0
    session_cache_write_tokens: int = 0
    session_reasoning_tokens: int = 0
    session_api_calls: int = 0
    session_estimated_cost_usd: float = 0.0
    session_cost_status: str = "unknown"
    session_cost_source: str = "none"

    # Cost budget configuration (loaded from cost.* in config.yaml)
    cost_budget_enabled: bool = True
    cost_max_usd: float = 0.0
    cost_alert_thresholds: tuple = (0.5, 0.8, 1.0)


# ─── 14. Interrupt & Control ──────────────────────────────────────────────────


@dataclass(slots=True)
class InterruptState:
    """Interrupt, approval, delegation, and threading primitives."""

    _interrupt_requested: bool = False
    _interrupt_message: str = ""
    _waiting_for_user_input: bool = False
    _waiting_for_approval: bool = False
    _user_confirmed: bool = False
    _last_approval_response: Any = None

    _client_lock: Any = None  # threading.RLock
    _delegate_depth: int = 0
    _active_children: List[Any] = field(default_factory=list)
    _active_children_lock: Any = None  # threading.Lock


# ─── 15. Context & Compression ───────────────────────────────────────────────


@dataclass(slots=True)
class ContextState:
    """Context compression, user turns, primary runtime."""

    context_compressor: Any = None
    compression_enabled: bool = True
    _context_compressor: Any = None
    _subdirectory_hints: Any = None
    _user_turn_count: int = 0
    _primary_runtime: Dict[str, Any] = field(default_factory=dict)
    # Context lifecycle tracking
    _context_scope: str = "short_term"  # Current active scope: "long_term", "medium_term", "short_term"


# ─── 16. Fallback ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FallbackState:
    """Fallback chain, index, and model selection."""

    _fallback_chain: List[Dict[str, Any]] = field(default_factory=list)
    _fallback_index: int = 0
    _fallback_activated: bool = False
    _fallback_model: Dict[str, Any] = None


# ─── 17. Stream ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class StreamState:
    """Streaming callback and break flag."""

    _stream_callback: Callable = None
    _stream_needs_break: bool = False


# ─── 18. Credentials ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class CredentialState:
    """Credential pool and file handler."""

    _credential_pool: Any = None
    _credential_file_handler: Any = None


# ─── 19. Activity ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ActivityState:
    """Last activity timestamp and description."""

    _last_activity_time: float = 0.0
    _last_activity_ts: float = 0.0
    _last_activity_desc: str = "initializing"


# ─── 20. Extra ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ExtraState:
    """Private state, image fallback cache, current tool tracking."""

    _private_state: Dict[str, Any] = field(default_factory=dict)
    _anthropic_image_fallback_cache: Dict[str, str] = field(default_factory=dict)
    _current_tool: str = None
    _api_call_count: int = 0
