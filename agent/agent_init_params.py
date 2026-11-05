"""Names of the ``AIAgent`` constructor params that ``init_agent`` copies onto the agent."""

# Constructor params stored verbatim under the same name.
_PASSTHROUGH_PARAMS = (
    "model", "max_iterations", "save_trajectories", "verbose_logging", "quiet_mode",
    "tool_progress_mode", "ephemeral_system_prompt", "preloaded_skills_prompt", "platform", "skip_context_files",
    "load_soul_identity", "pass_session_id", "log_prefix_chars",
    # OpenRouter provider preferences
    "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
    "provider_require_parameters", "provider_data_collection", "openrouter_min_coding_score",
    # Toolset filtering
    "enabled_toolsets", "disabled_toolsets",
    # Model response configuration (None = provider/model default)
    "max_tokens", "reasoning_config", "service_tier",
    "side_agent",
)
# Gateway identity params stored as ``agent._<name>``. gateway_session_key is the stable
# per-chat key (e.g. agent:main:telegram:dm:123).
_GATEWAY_IDENTITY_PARAMS = (
    "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
    "gateway_session_key",
)
_CALLBACK_PARAMS = (
    "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
    "tool_result_metadata_callback",
    "thinking_callback", "reasoning_callback", "clarify_callback",
    "read_terminal_callback", "read_preview_callback", "drive_preview_callback",
    "read_window_below_callback", "connection_callback", "tour_callback", "theme_callback",
    "setup_beat_callback", "step_callback", "stream_delta_callback",
    "interim_assistant_callback",
    "status_callback", "notice_callback", "notice_clear_callback",
    "event_callback", "reaction_callback", "tool_gen_callback",
)
