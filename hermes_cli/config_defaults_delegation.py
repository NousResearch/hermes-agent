"""Default delegation settings, shared by every config loader."""

DELEGATION_DEFAULTS = {
    "model": "",  # e.g. "google/gemini-3-flash-preview" (empty = inherit parent)
    "provider": "",  # e.g. "openrouter" (empty = inherit parent provider + credentials)
    # Fallback chain for delegated children (same entry format as the top-level list).
    # For an unpinned child, null = inherit the parent chain; [] = disable fallback.
    # A child pinned by provider, endpoint, or model gets no fallback unless this
    # setting declares one explicitly.
    "fallback_providers": None,
    "base_url": "",  # direct OpenAI-compatible endpoint for subagents
    "api_key": "",  # key for delegation.base_url (falls back to OPENAI_API_KEY)
    # Wire protocol for delegation.base_url: "chat_completions" | "codex_responses" |
    # "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix); set
    # explicitly for non-standard endpoints.
    "api_mode": "",
    # Per-child request settings on every delegation call (all resolution branches). Top-level
    # keys = API kwargs (e.g. service_tier); "extra_body" sub-dict merges into extra_body, e.g.
    # {"extra_body": {"provider": {"sort": "throughput"}}}. Explicit values win OVER
    # runtime/parent overrides (extra_body deep-merged 1 level).
    "request_overrides": {},
    # compression_threshold_tokens: optional absolute cap on a subagent's compaction TRIGGER
    # (not the request payload), applied as the lower of this and the child's ratio threshold.
    # 0 (default) = no subagent-specific cap; children compact where the parent does — the lower
    # of 0.50 x window and the global compression.threshold_tokens cap. A replay of a 1,393-agent
    # run showed 200K-400K caps within 5% of each other in cost once cache prefixes are intact,
    # and every compaction is a chance to lose detail, so the default stays off. A token count >= 16000 enables it;
    # other values (true, "200k") are config errors: warned and ignored.
    "compression_threshold_tokens": 0,
    # When delegate_task narrows child toolsets, keep the parent's enabled MCP toolsets (so
    # toolsets=["web"] doesn't strip MCP). false = strict intersection.
    "inherit_mcp_toolsets": True,
    # Per-subagent iteration cap (own budget, independent of the parent's).
    "max_iterations": 250,
    # Hard per-summary char ceiling on subagent results, layered on the dynamic budget (each
    # summary is sized to the parent's remaining context headroom; trimmed text spills to
    # ~/.hermes/cache/delegation/ with a head+tail window + read_file offset footer, nothing
    # lost). 0 disables the ceiling; the dynamic budget still applies.
    "max_summary_chars": 24000,
    # Inactivity cap per child (seconds, floor 30) — time with NO progress, not total runtime. 0 = no cap:
    # children fail only from real errors (API, tools, iteration budget). A progressing child (including one
    # waiting on a multi-minute completion) restarts the window; a frozen one is caught.
    "child_timeout_seconds": 0,
    # Subagent effort: "ultra" | "max" | "xhigh" | "high" | "medium" | "low" | "minimal" |
    # "none" (empty = inherit)
    "reasoning_effort": "",
    # Max parallel children per batch AND max concurrent background delegation units; async
    # dispatches beyond it run synchronously. Floor 1, no ceiling.
    "max_concurrent_children": 10,
    # Background fan-outs return as ONE message when the whole call finishes. true = each task
    # (or `group`) returns on its own as it finishes — more new turns for the orchestrator.
    "independent_completions": False,
    # Total child sessions a parent may create across repeated
    # delegate_task calls. Unlike max_concurrent_children, completed
    # children still consume this budget. Set to 0 to disable the cap.
    "max_children_per_session": 10,
    # Orchestrator role controls. Depth floored at 1, no ceiling; each level multiplies cost.
    "max_spawn_depth": 1,  # 1 = flat, 2 = orchestrator→leaf, 3+ = deeper
    "orchestrator_enabled": True,  # kill switch for role="orchestrator"
    # Total subagents a finite one-shot run (hermes chat -q / --oneshot) may spawn; 0 = unlimited.
    # Each child re-pays a cold system prompt and re-explores the repo, and one-shot spawns are mostly
    # "review my own work" rather than parallel work (agent/oneshot_footprint.py).
    "oneshot_max_children": 2,
    # Subagent threads ALWAYS resolve approvals non-interactively (the parent TUI owns stdin;
    # input() from a worker would deadlock). false = auto-deny, true = auto-approve "once"; both
    # log a warning audit line. true only for trusted batch work.
    "subagent_auto_approve": False,
    # Subagent background processes (task_id "sa-...") route notify_on_complete / watch_pattern
    # notifications to the PARENT; false suppresses them (the child's result is the
    # deliverable). Async-delegation results are NEVER suppressed.
    "surface_child_process_notifications": False,
}
