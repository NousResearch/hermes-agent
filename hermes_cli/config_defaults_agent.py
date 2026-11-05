"""The `agent` section of DEFAULT_CONFIG, split out of config_defaults.py: that facade is
over its line budget and may only shrink, and agent-level keys deserve a topical home.

Pure-data leaf module — must not import from hermes_cli.config. Comments are the user-facing
docs of config.yaml.
"""

AGENT_DEFAULTS = {
    # Turn cap. null = unlimited (default; caps caused silent mid-task truncation). Positive int
    # caps; "none"/"unlimited"/"inf"/0/-1 also mean unlimited (resolve_turn_limit).
    "max_turns": None,
    # Optional one-time model-visible checkpoint warning before a finite turn cap is exhausted.
    # null = off; set a ratio strictly between 0 and 1 (for example, 0.75).
    "budget_warning_ratio": None,
    # Wall-clock budget (seconds) per run. null = off. When set: one-time wrap-up notice at 80%
    # elapsed; implicit provider stale timeouts capped to remaining budget. CLI equivalent:
    # `hermes chat --run-budget N`.
    "run_budget_seconds": None,
    # Gateway inactivity timeout (seconds). Only fires when the agent is completely idle — not
    # while calling tools or receiving API responses. 0 = unlimited.
    "gateway_timeout": 1800,
    # Max seconds an alias routing key waits for the active turn holding the same session lease;
    # on expiry the message is rejected with a resend notice. Keep short: Telegram dispatches
    # sequentially, so a waiter delays unrelated topics. Non-positive -> 5s.
    "gateway_turn_lease_timeout": 5,
    # Per-session AIAgent cache in the gateway. Each entry keeps a warm prompt prefix AND the
    # full transcript: too small re-pays uncached prompts, too large fills the heap.
    "agent_cache": {
        "max_size": 128,  # LRU entry cap
        "idle_ttl_secs": 3600,  # evict agents idle this long
        # Anonymous-RSS budget (MB) above which LRU transcripts are shed (reloaded from disk
        # next turn). "auto" = derive from cgroup memory limit (or total RAM); number =
        # explicit; 0/off = disable the pass.
        "memory_high_mb": "auto",
        # Max sessions shed per pass (teardown bursts can't stall the gateway) and the number of
        # most-recently-used sessions the pass never touches.
        "max_evictions_per_pass": 16,
        "protect_recent": 8,
    },
    # Force-interrupt budget (seconds) once gateway stop()/drain has begun (SIGTERM, and the
    # final phase of in-band restart). 0 = interrupt immediately. Keep under systemd
    # TimeoutStopSec or risk SIGKILL mid-cleanup; for /restart prefer restart_after_turn_timeout
    # so turns finish BEFORE stop().
    "restart_drain_timeout": 0,
    # Floor under the stop()/drain wait (seconds) for cron jobs and api_server runs. Interrupted
    # chat turns resume on the next message, but an interrupted cron run is recorded as a
    # permanent failure and an interrupted /v1 run fails its waiting caller, so neither may
    # inherit restart_drain_timeout's 0 (#82161, #132989). Clamped to the shutdown-watchdog
    # leash minus teardown headroom (~50s unless TimeoutStopSec is raised). 0 = opt out.
    "cron_drain_timeout": 30,
    # In-band restart (/restart, SIGUSR1): refuse new work, then wait up to this many seconds
    # for in-flight agents/cron/api runs to finish before stop(). 0 = enter stop() at once. 30
    # min is a safety valve for wedged agents, not a target; raise for long unattended turns.
    # Default 30 min is a safety valve for wedged agents, not a target latency — an interactive `hermes
    # gateway restart` must never block for hours on a turn that wedged (#79133).
    "restart_after_turn_timeout": 1800,
    # Max seconds a submitted prompt waits for the deferred agent build (MCP discovery, model
    # metadata, skills scan) before failing visibly. The prompt is delivered as soon as the
    # build completes (progress notice past 30s), so this only fires on a hung build. Raise for
    # many slow/unreachable MCP servers.
    # See #63078.
    "build_wait_timeout": 600,
    # Hermes-level retry attempts for API errors (connection drops, timeouts, 5xx) wrapping the
    # whole call; the OpenAI SDK also retries transient errors (max_retries=2). Set 1 for fast
    # failover to fallback providers; raise to tolerate longer provider hiccups.
    "api_max_retries": 3,
    # Routes that are the user's own machine (host or host:port, e.g. a home LLM box behind a
    # tunnel). A 5xx from one of these is read as a refusal of that box — its GPU is held by a
    # local app, the engine is parked, or it never came back from Wake-on-LAN — so the retry
    # loop is skipped and the fallback chain is used at once, instead of burning backoff plus
    # the host's own startup wait. Empty (default) keeps today's retry-then-fallback for every
    # route; phrase/code matching still identifies the well-known local-engine refusals.
    "local_refusal_hosts": [],
    # Once api_max_retries AND the fallback chain are spent on a transient outage (5xx,
    # overloaded/529, connect/read timeouts) with nothing delivered yet, wait and retry this many
    # more cycles (jittered 15/30/60/60/60s; a provider Retry-After wins up to 120s) with a
    # visible "retrying automatically" countdown instead of ending the turn. Esc/interrupt stops
    # the wait; auth/format/billing/policy errors never enter. 0 disables.
    "auto_recovery_cycles": 5,
    # Seconds the Codex/Responses stream may keep reading after its terminal frame so the relay
    # finalizer can run. Relays that never close the SSE socket after response.completed would
    # otherwise wedge the turn until the idle watchdog discards the already-billed response
    # (#103864). 0 skips the drain. Well-behaved endpoints close immediately and never wait this long.
    "stream_drain_timeout": 2.0,
    # Empty-response retry guard. Empty retries re-send the full input at full price; this stops
    # re-billing deterministic empties (unsignaled refusals, zero output tokens) while failing
    # open on ambiguous evidence (missing usage, any tokens, model/provider change).
    "empty_response_guard": {
        "enabled": True,  # False = legacy fixed 3 retries unconditionally
        # When one empty attempt's estimated input cost >= this USD, the streak's retry budget
        # drops from 3 to 1. Unknown pricing / missing usage leaves it untouched.
        "cost_threshold_usd": 0.25,
    },
    # Fast mode: "" / "normal" (off), "fast" (always), "auto" (first fast_auto_seconds of every
    # turn), "cold" (first turn of a session only).
    "service_tier": "",
    "fast_auto_seconds": 60,
    # Responses-only `text.verbosity` (never chat_completions): "" = not sent, or low | medium | high.
    "text_verbosity": "",
    "daybreak": False,  # opt-in; only catalog-listed Codex models (agent/codex_service_tiers.py)
    # System-prompt guidance telling the model to call tools instead of describing actions.
    # "auto" = gpt/codex models; true/false = force for all models; or a list of model-name
    # substrings (e.g. ["gpt", "codex", "gemini", "qwen"]).
    "tool_use_enforcement": "auto",
    # Execution-discipline prompt block (tool persistence, tools for arithmetic/system facts,
    # read-back after external writes, count reconciliation, literal identifiers,
    # verification-gated completion). Chosen once per session by model name (byte-stable).
    # "auto" = gpt/codex/grok/deepseek/kimi/qwen/glm/minimax/mimo/mistral; true/false = force;
    # or a list of model-name substrings.
    "execution_guidance": "auto",
    # When the model narrates an action ("I'll go check the logs...") but emits no tool call,
    # inject a "continue now, execute the tools" nudge and loop (max 2 nudges/turn). Corrective
    # sibling of tool_use_enforcement. "auto" = codex_responses api_mode only; true = all
    # api_modes (fixes Gemini/Claude "stops after stating intent"); false = never; or a list of
    # model-name substrings.
    "intent_ack_continuation": "auto",
    # Anti-stall guards: (1) identical-call loop breaker appends a notice when the same tool is
    # called 3+ times with identical args AND results (never blocks; pollers like `process`
    # exempt); (2) continue-intent extension of empty-response recovery re-prompts once when the
    # model says it will continue but takes no action. False disables both.
    "stall_guards": True,
    # "Finish the job" prompt block for all models: don't stop at a stub, never fabricate output
    # when the real path is blocked. ~80 cached tokens. False disables.
    "task_completion_guidance": True,
    # Prompt block for all models steering independent tool calls (reads, searches, fetches,
    # read-only commands) into one batched turn; the runtime already runs them concurrently. ~70
    # cached tokens. False disables.
    "parallel_tool_call_guidance": True,
    # Toolchain probe: surfaces Python/pip/uv/PEP-668 state in the system prompt only when
    # something non-default is detected (no pip module, pip/python mismatch, PEP 668 without
    # uv); zero tokens when clean. Skipped for docker/modal/ssh backends (own probe).
    "environment_probe": True,
    # Bot Mode teammate-messaging protocol section (silent unless desktop Bot Mode manages it).
    "bot_mode_protocol": True,
    # Embedder-supplied text appended to the system prompt's environment-hints block, so a host
    # wrapping Hermes (sandbox runner, managed platform) can describe proxy/credential/ mount
    # layout without editing SOUL.md. Env HERMES_ENVIRONMENT_HINT overrides it.
    "environment_hint": "",
    # Coding posture: on interactive coding surfaces (CLI, TUI, desktop, ACP) in a code
    # workspace, add a coding brief + live git/workspace snapshot to the system prompt
    # (agent/coding_context.py). "auto" = prompt-only when interactive AND cwd is a code
    # workspace (toolsets untouched, messaging platforms unaffected); "focus" = auto + collapse
    # toolset to the lean coding set (+ enabled MCP servers) + demote non-coding skill
    # categories to names-only (explicit opt-in); "on" = force everywhere; "off" = disable.
    "coding_context": "auto",
    # Standing operator instructions (string or list) appended to the coding brief as an extra
    # stable system block — project-wide workflow rules, e.g. "Don't run tsc/lint until I
    # approve." Cache-safe: takes effect next session.
    "coding_instructions": "",
    # When verify-on-stop finds edits without fresh verification evidence, add guidance for
    # creative UI work (no broad tsc/lint/test before visual approval) and clean-diff
    # expectations. false = keep the evidence nudge terse.
    "verify_guidance": True,
    # Max consecutive `pre_verify` "continue" nudges per turn (hooks can't trap the loop).
    "max_verify_nudges": 3,
    # Verification closure: after code edits in a workspace, refuse a final answer until fresh
    # verification evidence exists or the agent explains why it can't check (bounded loop,
    # passive ledger). False (default) because the nudges proved more noise than signal; true =
    # force on everywhere; "auto" = on for interactive coding surfaces and programmatic callers,
    # off for messaging surfaces. Doc/markdown/skill-only edits never fire.
    "verify_on_stop": False,
    # Inactivity warning (seconds), once per run before gateway_timeout; no interrupt. 0 = off.
    "gateway_timeout_warning": 900,
    # Max seconds any surface (CLI, TUI/Desktop, messaging gateway) blocks an agent awaiting a
    # clarify-tool reply; then it unblocks with "[user did not respond within Xm]". 0 or less =
    # unlimited. Resolved by tools/clarify_gateway.py::resolve_clarify_timeout (a legacy
    # top-level ``clarify.timeout`` still wins when explicitly set).
    # 1h because users step away and a shorter value evicted the entry mid-think so a later
    # button tap hit a dead entry. Tradeoff: a higher value holds the gateway's running-agent
    # guard longer for a genuinely abandoned prompt — lower it to free the guard sooner. See #32762.
    "clarify_timeout": 3600,
    # "Still working" status interval (seconds); 0 = off. Lower = faster feedback, more noise;
    # 180 catches spinning weak-model runs before users /restart.
    "gateway_notify_interval": 180,
    # Session stall watchdog (seconds): RECOVERY notifier for an in-process AIAgent with an
    # adapter-queued follow-up while its activity clock is stale — NOT a general stall detector
    # (ignores startup restore, build sentinels, leases, debounce, other processes; scan cadence
    # per AIAgent). Notify-only: tells the user to try /new. Distinct from gateway_timeout
    # (kills the turn) and gateway_notify_interval. 0 = disable.
    # See #76354.
    "session_stall_timeout": 300,
    # Transcript-sanitiser heal escalation: after this many pre-send heal passes within a
    # 10-minute window, log one ERROR and queue a ONE-TIME out-of-band notice pointing at /debug
    # share or `hermes doctor` (status channel only; prompt cache untouched). 0 = no escalation
    # (per-window WARNINGs still fire).
    # See #96870.
    "sanitizer_heal_escalation_threshold": 3,
    # Seconds of continuous reconnect failure before a platform gets needs_attention flagged in
    # gateway status (`hermes status` / fleet monitoring). Retries never stop — a signal, not a
    # circuit breaker. 0 = disable.
    "reconnect_attention_after": 7200,
    # Freshness window (seconds) for the auto-continue note. After a crash/restart mid-run the
    # next user message gets "[System note: your previous turn was interrupted...]" prepended;
    # only when the last persisted transcript row is younger than this, so stale markers don't
    # revive an unrelated old task. Covers gateway_timeout (1800) plus slack. 0 = always inject.
    "gateway_auto_continue_freshness": 3600,
    # Max seconds the gateway waits for boot auto-resume turns before releasing the
    # startup-restore inbound gate (all inbound is QUEUED while shut, so one long resumed turn
    # would leave every channel unanswered). On timeout the gate opens and the resume keeps
    # running in the background; duplicate-agent protection is unaffected because the resume
    # slot is claimed synchronously first. 0 = wait forever.
    "gateway_startup_restore_drain_timeout": 30,
    # Max seconds the boot turn-machinery warm-up (run_agent import graph, tool schemas +
    # availability probes, context-file tier) may hold the inbound gate shut, so an early
    # message isn't served with a skeleton system prompt. On timeout the gate opens and warm-up
    # finishes in the background. 0 = disable warm-up (lazy init).
    "gateway_startup_warmup_timeout": 20,
    # Stale-stream ceiling (s) for local providers (Ollama, oMLX, llama-cpp) when the base stale
    # timeout is at its 180s default, so a wedged local server eventually trips the detector
    # instead of hanging forever. Env HERMES_LOCAL_STREAM_STALE_TIMEOUT overrides.
    "local_stream_stale_timeout": 900,
    "stream_max_lifetime": 0,  # opt-in whole-stream cap in s, 0 = off (agent/stream_max_lifetime.py)
    # How user-attached images reach the main model (gateway, TUI, CLI /attach). "auto" = native
    # when the model reports supports_vision=True AND auxiliary.vision.provider is not
    # explicitly set, else text; "native" = always attach (non-vision models error at the
    # provider or get a last-chance text fallback); "text" = always pre-analyze with
    # vision_analyze and prepend the description. vision_analyze stays a tool regardless.
    "image_input_mode": "auto",
    "disabled_toolsets": [],
    # Model name (any reasonable spelling) -> effort level; overrides agent.reasoning_effort
    # when the current model matches. Edit in config.yaml (no CLI support: dots in keys).
    "reasoning_overrides": {},
    # Stored reasoning is replayed to every provider that can read it by default. This flag
    # only adds the must-echo " " pad on reasoning-less tool-call turns, which DeepSeek,
    # Kimi/Moonshot and Xiaomi MiMo require (auto-detected by provider name/host). Set
    # `reasoning_echo: true` on a `model:` or `fallback_providers:` entry for a custom gateway
    # proxying one of them. Strict providers (Mistral, Groq, Cerebras) never get the field.
    "reasoning_echo": False,
    # Turn liveness watchdog: a turn with no observable progress for `timeout_s` seconds is
    # logged, force-interrupted so the UI can retry, and its lease stops renewing so stale-turn
    # cleanup can reclaim the session even if the interrupt can't unwind a wedged frame.
    # timeout_s <= 0 disables; poll_s = sampling interval. Invalid values (NaN, Inf,
    # non-positive poll) warn and fall back to defaults. See agent/turn_liveness.py.
    "turn_liveness": {"timeout_s": 600.0, "poll_s": 15.0},
}
