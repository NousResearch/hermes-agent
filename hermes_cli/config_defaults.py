"""Default configuration data for Hermes Agent.

Pure-data leaf module: DEFAULT_CONFIG and OPTIONAL_ENV_VARS, extracted
verbatim from hermes_cli/config.py. Must not import from hermes_cli.config.
"""

DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
    "credential_pool_strategies": {},
    "toolsets": ["hermes-cli"],
    # SQLite journal mode used by every Hermes database opener. WAL is the
    # normal default; set DELETE for weak-fsync/shared filesystems where WAL is
    # not crash-safe (for example macOS virtiofs, NFS, or SMB).
    "database": {
        "journal_mode": "wal",
        # Optional WAL sizing pragmas, applied when set to integers.
        # None = SQLite defaults (autocheckpoint 1000 pages, no size limit).
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
    },
    # Soft file-descriptor limit for long-running Hermes server processes.
    # Clamped to the OS hard limit; 0/false/null disables the adjustment.
    "runtime": {
        "nofile_soft_limit": 4096,
    },
    # Global active chat session cap across CLI, TUI/dashboard, and messaging.
    # None/0 = unbounded.
    "max_concurrent_sessions": None,
    # Soft LRU cap on in-memory TUI/desktop/dashboard sessions. When more than
    # this many are live, the gateway evicts the least-recently-active DETACHED
    # sessions (no live client) so accumulated agents don't pile up under memory
    # pressure. Reopening one re-resumes it from disk. 0/null disables.
    "max_live_sessions": 16,
    "session": {
        # Per-terminal `hermes -c`: each CLI session drops a breadcrumb file
        # under $HERMES_HOME/terminal-sessions/<terminal-id>, and a bare
        # -c/--continue resumes THIS terminal's session (tmux pane, kitty
        # window, wezterm pane, plain tty, ...) instead of the globally
        # most-recent one. Set false to restore the old latest-session
        # behavior everywhere.
        "terminal_continue": True,
    },
    "agent": {
        # Unlimited by default. The agent turn cap caused more problems than
        # it solved (silent mid-task truncation). null = unlimited; set a
        # positive integer to cap, or use "none"/"unlimited"/"inf"/0/-1 —
        # all normalized by hermes_cli.config.resolve_turn_limit.
        "max_turns": None,
        # Optional wall-clock budget in seconds per conversation run.
        # null/absent = feature fully off (zero behavior change). When set,
        # the agent gets a one-time wrap-up notice at 80% elapsed and
        # implicit provider stale timeouts are capped to the remaining
        # budget. CLI one-shot equivalent: `hermes chat --run-budget N`.
        "run_budget_seconds": None,
        # Inactivity timeout for gateway agent execution (seconds).
        # The agent can run indefinitely as long as it's actively calling
        # tools or receiving API responses.  Only fires when the agent has
        # been completely idle for this duration.  0 = unlimited.
        "gateway_timeout": 1800,
        # Maximum time an alias routing key waits for the active turn holding
        # the same resolved session lease. On expiry the inbound message is
        # rejected with a resend notice rather than run without serialization.
        # Non-positive values fall back to 1800 seconds.
        "gateway_turn_lease_timeout": 1800,
        # Per-session AIAgent cache in the gateway. Each cached agent keeps a
        # warm prompt prefix AND the session's full transcript, so the cache
        # trades memory for cost: too small and every turn re-pays an uncached
        # prompt, too large and tool-heavy transcripts fill the heap.
        "agent_cache": {
            # LRU entry cap.
            "max_size": 128,
            # Evict an agent that has been idle this long (seconds).
            "idle_ttl_secs": 3600,
            # Anonymous-RSS budget (MB) above which the gateway starts shedding
            # least-recently-used transcripts, which reload from the persisted
            # session on the next turn. "auto" derives the budget from the
            # cgroup memory limit the gateway runs under (or total RAM when
            # uncapped); a number sets it explicitly; 0/off disables the pass
            # and lets memory grow to whatever the two bounds above allow.
            "memory_high_mb": "auto",
            # Upper bound on how many sessions one pressure pass sheds, so a
            # burst of teardowns cannot stall the gateway.
            "max_evictions_per_pass": 16,
            # Most-recently-used sessions the pressure pass never touches —
            # they are the ones actively paying for a warm prompt cache.
            "protect_recent": 8,
        },
        # Force-interrupt budget once gateway stop()/drain has begun
        # (seconds). Applies to SIGTERM/external stop and to the final
        # phase of in-band restart after any after-turn wait. 0 = interrupt
        # immediately (the default).
        #
        # Keep this short and under systemd TimeoutStopSec — a long value
        # here invites SIGKILL-mid-cleanup. For in-band restart
        # (/restart, SIGUSR1), prefer restart_after_turn_timeout below so
        # active turns finish *before* stop() begins (#77184).
        "restart_drain_timeout": 0,
        # Cron-only floor under the stop()/drain wait (seconds). A chat turn
        # interrupted by a restart is announced to the user and resumed on
        # their next message; an interrupted cron run is written to jobs.json
        # as a permanent failure that nobody is waiting on, so it must not
        # inherit restart_drain_timeout's 0 (#82161). Clamped at runtime to
        # the shutdown-watchdog leash minus teardown headroom, so raising it
        # past ~50s has no effect unless TimeoutStopSec is raised too.
        # 0 = opt out (cron drains on restart_drain_timeout, legacy).
        "cron_drain_timeout": 30,
        # In-band restart wait for active turns to finish before stop()
        # (seconds). /restart and SIGUSR1 refuse new work, then wait up to
        # this cap for in-flight agents/cron/api runs to complete naturally
        # so the requesting turn is not amputated by restart_drain_timeout.
        # 0 = legacy behaviour (enter stop()/drain immediately). Default
        # 30 min is a safety valve for wedged agents, not a target latency —
        # an interactive `hermes gateway restart` must never block for hours
        # on a turn that wedged (#79133). Long unattended turns can raise
        # this in config.yaml.
        "restart_after_turn_timeout": 1800,
        # Upper bound (seconds) a submitted prompt waits for the deferred
        # agent build (MCP discovery, model metadata, skills scan) before
        # failing with a visible error (#63078). The gateway's wait is
        # patient — the prompt is delivered the moment the build completes
        # and a progress notice is emitted past 30s — so this cap only fires
        # on a genuinely hung build. Raise it for deployments with many slow
        # or unreachable MCP servers.
        "build_wait_timeout": 600,
        # Max app-level retry attempts for API errors (connection drops,
        # provider timeouts, 5xx, etc.) before the agent surfaces the
        # failure.  The OpenAI SDK already does its own low-level retries
        # (max_retries=2 default) for transient network errors; this is
        # the Hermes-level retry loop that wraps the whole call.  Lower
        # this to 1 if you use fallback providers and want fast failover
        # on flaky primaries; raise it if you prefer to tolerate longer
        # provider hiccups on a single provider.
        "api_max_retries": 3,
        # Empty-response retry guard (NS-503).  The empty-retry loop
        # re-sends the full conversation input at full price on every
        # attempt; these settings stop it from re-billing *deterministic*
        # empties (unsignaled provider refusals with zero output tokens)
        # while failing open on any ambiguous evidence (missing usage,
        # any generated tokens, model/provider change mid-streak).
        "empty_response_guard": {
            # Master switch for both guards below. False restores the
            # legacy fixed 3-retry behaviour unconditionally.
            "enabled": True,
            # When the estimated input cost of a single empty attempt
            # meets or exceeds this many USD, the retry budget for the
            # streak drops from 3 to 1. Unknown pricing or missing usage
            # leaves the budget untouched.
            "cost_threshold_usd": 0.25,
        },
        "service_tier": "",
        # Tool-use enforcement: injects system prompt guidance that tells the
        # model to actually call tools instead of describing intended actions.
        # Values: "auto" (default — applies to gpt/codex models), true/false
        # (force on/off for all models), or a list of model-name substrings
        # to match (e.g. ["gpt", "codex", "gemini", "qwen"]).
        "tool_use_enforcement": "auto",
        # Execution-discipline guidance: injects a system prompt block covering
        # tool persistence, mandatory tool use for arithmetic/system facts,
        # external-write read-back, count reconciliation, literal preservation
        # of identifiers, and verification-gated completion.  Chosen once at
        # session start keyed on model name (prompt stays byte-stable).
        # Values: "auto" (default — applies to gpt/codex/grok/deepseek/kimi/
        # qwen/glm/minimax/mimo/mistral models), true/false (force on/off for
        # all models), or a list of model-name substrings to match.
        "execution_guidance": "auto",
        # Intent-ack continuation: when the model opens a turn by narrating an
        # action it will take ("I'll go check the logs...") but emits no tool
        # call, intercept the turn-end, inject a "continue now, execute the
        # tools" nudge, and loop instead of ending the turn (capped at 2 nudges
        # per turn). This is the corrective sibling of tool_use_enforcement (the
        # preventive prompt-side guard). Values: "auto" (default — fires only on
        # the codex_responses api_mode, the historical behavior), true (all
        # api_modes — fixes the Gemini/Claude "stops after stating intent" case),
        # false (never), or a list of model-name substrings to match.
        "intent_ack_continuation": "auto",
        # Runtime anti-stall guards. When True (default), two conservative
        # guards run: (1) an identical-call loop breaker that appends a short
        # notice to the tool result when the same tool is called 3+ consecutive
        # times with identical arguments AND identical results (never blocks;
        # pollers like `process` are exempt), and (2) a continue-intent
        # extension of the empty-response recovery that re-prompts once when
        # the model ends its turn saying it will continue but takes no action.
        # Set False to disable both.
        "stall_guards": True,
        # Universal "finish the job" guidance — short prompt block applied to
        # all models that targets two cross-family failure modes: (1) stopping
        # after a stub instead of finishing the artifact, (2) fabricating
        # plausible-looking output when a real path is blocked.  Costs ~80
        # tokens in the cached system prompt.  Set False to disable globally.
        "task_completion_guidance": True,
        # Universal parallel-tool-call guidance — short prompt block applied to
        # all models that tells the model to batch independent tool calls
        # (reads, searches, web fetches, read-only commands) into one turn
        # instead of one call per turn.  The runtime already runs independent
        # calls concurrently, so this just steers the model to produce the
        # batch — cutting round-trips and the resent-context cost that
        # compounds over a long conversation.  Costs ~70 tokens in the cached
        # system prompt.  Set False to disable globally.
        "parallel_tool_call_guidance": True,
        # Local-environment toolchain probe — surfaces Python/pip/uv/PEP-668
        # state in the system prompt when something non-default is detected
        # (e.g. python3 has no pip module, pip→python version mismatch, PEP
        # 668 enforcement without uv).  Costs zero tokens when the env is
        # clean (probe emits nothing).  Skipped for remote terminal backends
        # (docker/modal/ssh — they have their own probe).  Set False to
        # disable entirely.
        "environment_probe": True,
        # Bot Mode teammate-messaging protocol section (silent unless a
        # profile is managed by the desktop's Bot Mode).
        "bot_mode_protocol": True,
        # Embedder-supplied environment description appended to the system
        # prompt's environment-hints block. Lets a host that wraps Hermes
        # (sandbox runner, managed platform) explain the runtime environment
        # — proxy, credential handling, mount layout — without editing the
        # identity slot (SOUL.md). Empty by default. The HERMES_ENVIRONMENT_HINT
        # env var overrides this (build-time/container mechanism).
        "environment_hint": "",
        # Coding posture — on interactive coding surfaces (CLI, TUI, desktop
        # app, ACP) in a code workspace, Hermes adds a coding operating brief
        # + a live git/workspace snapshot to the system prompt. See
        # agent/coding_context.py.
        #   "auto" (default) — prompt-only posture when the surface is
        #                      interactive AND cwd is a code workspace.
        #                      Toolsets are never touched; messaging platforms
        #                      unaffected.
        #   "focus"          — auto + collapse the toolset to the lean coding
        #                      set (+ enabled MCP servers) + demote non-coding
        #                      skill categories to names-only in the prompt's
        #                      skill index. Explicit opt-in.
        #   "on"             — force the prompt posture everywhere.
        #   "off"            — disable entirely.
        "coding_context": "auto",
        # Standing operator instructions for the coding posture. A string (or
        # list of strings) appended to the coding brief as an extra stable
        # system block — pin project-wide workflow rules here instead of editing
        # the shipped brief, e.g. "For UI work, don't run tsc/lint until I
        # approve. Clean the diff before you commit and push." Cache-safe:
        # takes effect next session. Empty by default.
        "coding_instructions": "",
        # When verify-on-stop finds edited code without fresh verification
        # evidence, append guidance for creative UI work (avoid broad
        # tsc/lint/test before visual approval) and clean-diff expectations.
        # Set false to keep the evidence nudge terse.
        "verify_guidance": True,
        # Upper bound on consecutive `pre_verify` "continue" nudges in a single
        # turn, so a user/plugin hook can never trap the loop.
        "max_verify_nudges": 3,
        # Verification closure: after the agent edits files in a code workspace,
        # do not accept a final answer until fresh verification evidence exists
        # or the agent explains why it cannot run checks. The loop is bounded
        # and uses the passive verification ledger. Default is False (opt-in):
        # the v31/v32 config migrations already switch existing installs off
        # because the verification narrative proved more noise than signal,
        # and the docs tell users to treat off as the effective default — a
        # fresh install must not be the one population that still gets the
        # nudges. Set true to force on everywhere, or "auto" for the legacy
        # surface-aware behavior (on for interactive coding surfaces — CLI,
        # TUI, desktop — and programmatic callers, off for conversational
        # messaging surfaces). Doc/markdown/skill-only edits never fire it.
        "verify_on_stop": False,
        # Staged inactivity warning: send a warning to the user at this
        # threshold before escalating to a full timeout.  The warning fires
        # once per run and does not interrupt the agent.  0 = disable warning.
        "gateway_timeout_warning": 900,
        # Maximum time (seconds) the gateway will block an agent waiting for
        # a clarify-tool response from the user.  Hit this and the agent
        # unblocks with "[user did not respond within Xm]" so it can adapt
        # rather than pinning the running-agent guard forever.  CLI clarify
        # blocks indefinitely (input() is synchronous) and ignores this.
        # Default 3600 (1h): real users step away (meetings, AFK) and the
        # old 600s default evicted the entry mid-think, so a later button
        # tap landed on a dead entry (#32762).  Tradeoff: a higher value
        # holds the gateway's running-agent guard longer for a genuinely
        # abandoned prompt — lower it if a single session must free up the
        # guard sooner.
        "clarify_timeout": 3600,
        # Periodic "still working" notification interval (seconds).
        # Sends a status message every N seconds so the user knows the
        # agent hasn't died during long tasks.  0 = disable notifications.
        # Lower values mean faster feedback on slow tasks but more chat
        # noise; 180s is a compromise that catches spinning weak-model runs
        # (60+ tool iterations with tiny output) before users assume the
        # bot is dead and /restart.
        "gateway_notify_interval": 180,
        # Session stall watchdog (seconds). Scope (#76354): this is a
        # RECOVERY notifier for an in-process AIAgent that has an
        # adapter-queued follow-up (pending inbound / queued event) while its
        # activity clock is stale — NOT a general gateway/session stall
        # detector. It does not observe startup restoration, build sentinels,
        # turn leases, debounce state, or work owned by another process; the
        # scan cadence is per AIAgent instance, not globally coordinated per
        # durable session. Notify-only: warns the user to try /new. Distinct
        # from gateway_timeout (which kills the turn) and
        # gateway_notify_interval ("still working" heartbeats). 0 = disable.
        "session_stall_timeout": 300,
        # Long-lived reconnect-loop escalation (seconds). A platform that has
        # been continuously failing/reconnecting for this long gets
        # needs_attention flagged in gateway runtime status (visible in
        # `hermes status` / fleet monitoring). Retries never stop — this is a
        # signal, not a circuit breaker. 0 = disable.
        "reconnect_attention_after": 7200,
        # Freshness window for the gateway auto-continue note (seconds).
        # After a gateway crash/restart/SIGTERM mid-run, the next user
        # message gets a "[System note: your previous turn was
        # interrupted — process the unfinished tool result(s) first]"
        # prepended so the model picks up where it left off.  That's the
        # right behaviour while the interruption is fresh, but stale
        # markers (transcript last touched hours or days ago) can revive
        # an unrelated old task when the user's next message starts new
        # work.  This window is the max age of the last persisted
        # transcript row for which we still inject the continue note.
        # Default 3600s comfortably covers a long turn (gateway_timeout
        # default is 1800s) plus runtime slack.  Set to 0 to disable the
        # gate and restore pre-fix behaviour (always inject).
        "gateway_auto_continue_freshness": 3600,
        # Max seconds the gateway waits for boot auto-resume turns to finish
        # before it releases the startup-restore inbound gate.  While startup
        # restore is in progress the gateway QUEUES every inbound message
        # instead of replying, so no channel gets an answer until this gate
        # opens.  Without a bound, one pathologically long resumed turn holds
        # the gate shut and every channel's inbound piles up unanswered for as
        # long as that turn runs.  On timeout the gate releases and the slow
        # resume turn keeps running in the background; duplicate-agent
        # protection is unaffected because the resume slot is claimed
        # synchronously before the gate runs.  Set to 0 to disable the bound
        # (historical "wait forever" behaviour).
        "gateway_startup_restore_drain_timeout": 30,
        # Stale-stream ceiling for local providers (Ollama, oMLX, llama-cpp) in
        # seconds. When the base stale timeout is at its default (180s) and a
        # local endpoint is detected, this finite ceiling replaces the former
        # infinite disable so a wedged local server eventually trips the
        # detector instead of hanging forever. The env var
        # ``HERMES_LOCAL_STREAM_STALE_TIMEOUT`` overrides for escape-hatch use.
        "local_stream_stale_timeout": 900,
        # How user-attached images are presented to the main model on each turn.
        #   "auto"   — attach natively when the active model reports
        #              supports_vision=True AND the user hasn't explicitly
        #              configured auxiliary.vision.provider.  Otherwise fall
        #              back to text (vision_analyze pre-analysis).
        #   "native" — always attach natively; non-vision models will either
        #              error at the provider or get a last-chance text fallback
        #              (see run_agent._prepare_messages_for_api).
        #   "text"   — always pre-analyze with vision_analyze and prepend the
        #              description as text; the main model never sees pixels.
        # Affects gateway platforms, the TUI, and CLI /attach.  vision_analyze
        # remains available as a tool regardless of this setting — the routing
        # only controls how inbound user images are presented.
        "image_input_mode": "auto",
        "disabled_toolsets": [],

        # Per-model reasoning effort overrides (spelling-tolerant).
        # Dict mapping model names (any reasonable spelling) to effort levels.
        # Takes precedence over agent.reasoning_effort when the current model
        # matches a key in this dict.
        # Edit directly in config.yaml (no CLI support due to dots in keys).
        "reasoning_overrides": {},

        # Per-provider opt-in to preserve assistant ``reasoning_content``
        # when replaying history.  The built-in echo families (DeepSeek,
        # Kimi/Moonshot, Xiaomi MiMo) are auto-detected by provider name
        # and base-URL host.  Custom providers and OpenAI-compatible
        # gateways that proxy those same models (or other thinking-mode
        # backends) are not covered by the host-based rules.
        #
        # Set ``reasoning_echo: true`` on a ``model:`` entry (primary) or a
        # ``fallback_providers:`` entry (per-fallback) to preserve
        # ``reasoning_content`` on replay for that provider only.  Default
        # ``false`` keeps the historical strict-provider behavior (Mistral,
        # Groq, Cerebras reject the field with HTTP 400).
        "reasoning_echo": False,
    },

    "terminal": {
        "backend": "local",
        "modal_mode": "auto",
        # Remote-backend graceful degradation: when a connection-class
        # infrastructure failure occurs (SSH host unreachable, Docker daemon
        # down), "warn" (default) returns a structured degraded tool result
        # with a reason + retry hint so the model can act on it; "fail"
        # preserves the historical error + traceback behavior.
        "degraded_mode": "warn",
        "cwd": ".",  # Use current directory
        # Terminal font family for the desktop app's embedded xterm.js terminal.
        # When set (e.g. "'CaskaydiaCoveNerdFont', 'JetBrains Mono', monospace"),
        # the desktop terminal uses this as the CSS font-family value, with the
        # built-in default ("'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo,
        # Consolas, monospace") as fallback when the field is empty or unset.
        # This lets users install a Nerd Font (or any custom font) and configure
        # it here without patching the built desktop app.
        "font_family": "",
        "timeout": 180,
        # Bounded grace period (seconds) between SIGTERM and an escalated
        # SIGKILL when terminating a host process tree (browser daemons, etc.).
        # A daemon that stalls in its SIGTERM handler is force-killed after this
        # window so it can't leak indefinitely. 0 disables escalation (SIGTERM
        # only — the historical behavior). Floored internally at 0.
        "daemon_term_grace_seconds": 2.0,
        # Environment variables to pass through to sandboxed execution
        # (terminal and execute_code).  Skill-declared required_environment_variables
        # are passed through automatically; this list is for non-skill use cases.
        "env_passthrough": [],
        # HOME handling for host tool subprocesses:
        #   auto    — host keeps the real OS-user HOME; containers use
        #             HERMES_HOME/home for persistent state (default)
        #   real    — force the real OS-user HOME
        #   profile — force HERMES_HOME/home when it exists (old strict
        #             per-profile CLI config isolation)
        "home_mode": "auto",
        # Extra files to source in the login shell when building the
        # per-session environment snapshot.  Use this when tools like nvm,
        # pyenv, asdf, or custom PATH entries are registered by files that
        # a bash login shell would skip — most commonly ``~/.bashrc``
        # (bash doesn't source bashrc in non-interactive login mode) or
        # zsh-specific files like ``~/.zshrc`` / ``~/.zprofile``.
        # Paths support ``~`` / ``${VAR}``. Missing files are silently
        # skipped. When empty, Hermes auto-sources ``~/.profile``,
        # ``~/.bash_profile``, and ``~/.bashrc`` (in that order) if the
        # snapshot shell is bash (this is the ``auto_source_bashrc``
        # behaviour — disable with that key if you want strict login-only
        # semantics).
        "shell_init_files": [],
        # When true (default), Hermes sources the user's shell rc files
        # (``~/.profile``, ``~/.bash_profile``, ``~/.bashrc``) in the
        # login shell used to build the environment snapshot. This
        # captures PATH additions, shell functions, and aliases — which a
        # plain ``bash -l -c`` would otherwise miss because bash skips
        # bashrc in non-interactive login mode, and because a default
        # Debian/Ubuntu ``~/.bashrc`` short-circuits on non-interactive
        # sources. ``~/.profile`` and ``~/.bash_profile`` are tried first
        # because ``n`` / ``nvm`` / ``asdf`` installers typically write
        # their PATH exports there without an interactivity guard. Turn
        # this off if your rc files misbehave when sourced
        # non-interactively (e.g. one that hard-exits on TTY checks).
        "auto_source_bashrc": True,
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "docker_forward_env": [],
        # Explicit environment variables to set inside Docker containers.
        # Unlike docker_forward_env (which reads values from the host process),
        # docker_env lets you specify exact key-value pairs — useful when Hermes
        # runs as a systemd service without access to the user's shell environment.
        # Example: {"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock"}
        "docker_env": {},
        "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        # Vercel Sandbox runtime (vercel_sandbox backend only).
        # Supported: node24, node22, python3.13.
        "vercel_runtime": "node24",
        # Container resource limits (docker, singularity, modal, daytona, vercel_sandbox — ignored for local/ssh)
        "container_cpu": 1,
        "container_memory": 5120,       # MB (default 5GB)
        "container_disk": 51200,        # MB (default 50GB)
        "container_persistent": True,   # Persist filesystem across sessions
        # Docker volume mounts — share host directories with the container.
        # Each entry is "host_path:container_path" (standard Docker -v syntax).
        # Example:
        # ["/home/user/projects:/workspace/projects",
        #  "/home/user/.hermes/cache/documents:/output"]
        # For gateway MEDIA delivery, write inside Docker to /output/... and emit
        # the host-visible path in MEDIA:, not the container path.
        "docker_volumes": [],
        # Explicit opt-in: mount the host cwd into /workspace for Docker sessions.
        # Default off because passing host directories into a sandbox weakens isolation.
        "docker_mount_cwd_to_workspace": False,
        # Opt-in egress lockdown for Docker terminal sessions. When false,
        # Docker runs with --network=none so commands cannot reach the network.
        "docker_network": True,
        "docker_extra_args": [],        # Extra flags passed verbatim to docker run
        # /dev/shm size for the Docker sandbox. Docker's 64 MB default silently
        # breaks Chromium/Playwright and PyTorch DataLoader workers; tmpfs is
        # lazily allocated so the higher ceiling costs nothing until used.
        # Set to "" (or "0") to omit the flag and use Docker's default.
        "docker_shm_size": "1g",
        # Explicit opt-in: run the Docker container as the host user's uid:gid
        # (via `--user`).  When enabled, files written into bind-mounted dirs
        # (docker_volumes, the persistent workspace, or the auto-mounted cwd)
        # are owned by your host user instead of root, which avoids needing
        # `sudo chown` after container runs. Default off to preserve behavior
        # for images whose entrypoints expect to start as root (e.g. the
        # bundled Hermes image, which drops to the `hermes` user via
        # s6-setuidgid inside each supervised service).
        # When on, SETUID/SETGID caps are omitted from the container since
        # no privilege drop is needed.
        "docker_run_as_host_user": False,
        # Persistent shell — keep a long-lived bash shell across execute() calls
        # so cwd/env vars/shell variables survive between commands.
        # Enabled by default for non-local backends (SSH); local is always opt-in
        # via TERMINAL_LOCAL_PERSISTENT env var.
        "persistent_shell": True,
    },

    "web": {
        "backend": "",           # shared fallback — applies to both search and extract
        "search_backend": "",    # per-capability override for web_search (e.g. "searxng")
        "extract_backend": "",   # per-capability override for web_extract (e.g. "native")
        "extract_char_limit": 15000,  # per-page char budget for web_extract; larger pages truncate + store full text in cache/web
        # Keyless free-tier ring: with NO web backend configured or keyed,
        # web_search/web_extract rotate round-robin across five vendors'
        # public free tiers (exa, parallel, tavily, firecrawl, keenable),
        # failing over to the next ring vendor on rate limits. Never
        # pre-empts a configured or keyed backend. Set false to disable.
        "keyless_fallback": True,
        # One-shot keyless rescue: when the chosen/keyed backend fails a
        # web_search/web_extract call, THAT call retries once on the keyless
        # free-tier ring — the next call attempts the chosen backend again
        # (no sticky failover). Off when keyless_fallback is false.
        "keyless_rescue": True,
        # Per-provider tier selection for ring vendors with both a keyless
        # free endpoint and a keyed paid path (exa, parallel, tavily,
        # firecrawl, keenable). Set by the `hermes tools` picker's
        # "Free (keyless)" / "Paid (API key)" rows.
        #   free  — always use the anonymous free endpoint (even with a key)
        #   paid  — always use the keyed path (missing key = error; vendor
        #           is also excluded from the keyless ring)
        #   unset — auto: keyed when the API key is present, else the ring
        "provider_tier": {},
    },

    "browser": {
        # Browser tool implementation.
        # ""            — DEFAULT: Browser Use mode when the browser-use CLI
        #                 (or uvx) is available; otherwise the built-in
        #                 browser tools. Camofox setups always keep the
        #                 built-in tools (no CDP surface).
        # "browser-use" — force Browser Use mode: one browser_exec tool
        #                 driving the Browser Use CLI 3.0 over any CDP
        #                 backend (local Chrome, cloud browsers)
        # "off"         — force the built-in browser tools
        #                 (browser_navigate, browser_click, …)
        "backend": "",
        "inactivity_timeout": 120,
        "command_timeout": 30,  # Timeout for browser commands in seconds (screenshot, navigate, etc.)
        "record_sessions": False,  # Auto-record browser sessions as WebM videos
        "headed": False,  # Local mode: launch Chromium with a visible window (also skips per-turn cleanup so the window persists between turns; idle reaper still applies)
        "allow_private_urls": False,  # Allow navigating to private/internal IPs (localhost, 192.168.x.x, etc.)
        # Browser engine for local mode.  Passed as ``--engine <value>`` to
        # agent-browser v0.25.3+.
        # "auto"       — use Chrome (default, don't pass --engine at all)
        # "lightpanda" — use Lightpanda (1.3-5.8x faster navigation, no screenshots)
        # "chrome"     — explicitly request Chrome
        # Also settable via AGENT_BROWSER_ENGINE env var.
        "engine": "auto",
        "auto_local_for_private_urls": True,  # When a cloud provider is set, auto-spawn local Chromium for LAN/localhost URLs instead of sending them to the cloud
        "cdp_url": "",  # Optional persistent CDP endpoint for attaching to an existing Chromium/Chrome
        "allow_unsafe_evaluate": False,  # Legacy override: when true, browser_console(expression=...) bypasses the restrict_evaluate denylist entirely
        "restrict_evaluate": False,  # Opt-in denylist blocking sensitive JS primitives (cookies/storage/clipboard/network/form values) in browser_console(expression=...)
        # CDP supervisor — dialog + frame detection via a persistent WebSocket.
        # Active only when a CDP-capable backend is attached (Browserbase or
        # local Chrome via /browser connect). See
        # website/docs/developer-guide/browser-supervisor.md.
        "dialog_policy": "must_respond",  # must_respond | auto_dismiss | auto_accept
        "dialog_timeout_s": 300,  # Safety auto-dismiss after N seconds under must_respond
        "camofox": {
            # When true, Hermes sends a stable profile-scoped userId to Camofox
            # so the server maps it to a persistent Firefox profile automatically.
            # When false (default), each session gets a random userId (ephemeral).
            "managed_persistence": False,
            # Optional externally managed Camofox identity. Useful when another
            # app owns the visible browser and Hermes should operate in it.
            "user_id": "",
            "session_key": "",
            # Rehydrate tab_id from Camofox before creating a new tab.
            "adopt_existing_tab": False,
            # Docker Camofox opens page URLs from inside the container. Enable
            # this to rewrite loopback page URLs (localhost/127.0.0.1/::1) to a
            # host alias while leaving CAMOFOX_URL itself unchanged.
            "rewrite_loopback_urls": False,
            "loopback_host_alias": "host.docker.internal",
        },
    },

    # Filesystem checkpoints — automatic snapshots before destructive file ops.
    # When enabled, the agent takes a snapshot of the working directory once
    # per conversation turn (on first write_file/patch call).  Use /rollback
    # to restore.
    #
    # Defaults changed in v2 (single shared shadow store, real pruning):
    #   - enabled: True -> False   (opt-in; most users never use /rollback)
    #   - max_snapshots: 50 -> 20  (now actually enforced via ref rewrite)
    #   - auto_prune:   False -> True (orphans/stale pruned automatically)
    # Opt in via ``hermes chat --checkpoints`` or set enabled=True here.
    "checkpoints": {
        "enabled": False,
        # Max checkpoints to keep per working directory.  Pre-v2 this only
        # limited the `/rollback` listing; v2 actually rewrites the ref and
        # garbage-collects older commits.
        "max_snapshots": 20,
        # Hard ceiling on total ``~/.hermes/checkpoints/`` size (MB).  When
        # exceeded, the oldest checkpoint per project is dropped in a
        # round-robin pass until total size falls under the cap.
        # 0 disables the size cap.
        "max_total_size_mb": 500,
        # Skip any single file larger than this when staging a checkpoint.
        # Prevents accidental snapshotting of datasets, model weights, and
        # other large generated assets.  0 disables the filter.
        "max_file_size_mb": 10,
        # Auto-maintenance: hermes sweeps the checkpoint base at startup
        # (at most once per ``min_interval_hours``) and:
        #   * deletes project entries whose last_touch is older than
        #     ``retention_days``
        #   * GCs the single shared store to reclaim unreachable objects
        #   * enforces ``max_total_size_mb`` across remaining projects
        #   * deletes ``legacy-*`` archives older than ``retention_days``
        #
        # NOTE: this automatic sweep never deletes "orphan" entries (workdir
        # no longer found on disk). A missing workdir at startup is
        # ambiguous — it can mean the project was deleted, or that an
        # external volume / network share / VPN is simply not mounted yet —
        # and this sweep runs unattended, so it must never guess. Orphan
        # cleanup is only available via the explicit
        # ``hermes checkpoints prune`` command (add ``--keep-orphans`` to
        # skip it), where a human is looking at the output.
        "auto_prune": True,
        "retention_days": 7,
        "min_interval_hours": 24,
    },

    # Hard cap (chars) for a single automatic context file such as SOUL.md,
    # AGENTS.md, CLAUDE.md, .hermes.md, or .cursorrules before Hermes applies
    # head/tail truncation. ``null`` (the default) lets the cap scale with the
    # model's context window (floor 20K, ceiling 500K) so large-context models
    # rarely truncate a project doc. Set a positive integer to pin a fixed cap
    # and override the dynamic behavior. Separate from read_file tool limits.
    "context_file_max_chars": None,

    # Maximum characters returned by a single read_file call.  Reads that
    # exceed this are rejected with guidance to use offset+limit.
    # 100K chars ≈ 25–35K tokens across typical tokenisers.
    "file_read_max_chars": 100_000,

    # Seconds to wait at agent-build time for in-flight MCP server discovery
    # to finish before the agent snapshots its tool list.  MCP discovery runs
    # in a background thread so a slow/dead server can't freeze startup; this
    # bounds how long the first agent build blocks on it.  The wait returns
    # the INSTANT discovery completes, so users with no MCP servers (the common
    # case) or fast servers pay ~0s regardless of this value — the bound is
    # only reached when a server is genuinely still connecting.  The old 0.75s
    # default was a touch short for HTTP/OAuth servers on a cold connect; a
    # modest bump lets more of them land in the FIRST turn's snapshot.  This is
    # only a turn-1 latency/UX knob: a server that misses this window is still
    # picked up automatically on the next turn by the between-turns refresh
    # (see agent/turn_context.py), so correctness never depends on it.  Keep it
    # small so a slow/dead server adds little to first-response latency.
    "mcp_discovery_timeout": 1.5,

    # Single-query (``hermes -q/-z "..."``) variant of mcp_discovery_timeout.
    # In one-shot mode there is only ONE turn, so the between-turns late-binding
    # refresh never runs: a server that misses the small interactive bound is
    # invisible to the LLM for the whole session.  This larger bound gives slow
    # cold-start servers (npx, uvx, remote HTTP) a chance to land in the one
    # tool snapshot.  ``thread.join(timeout)`` returns the instant discovery
    # completes, so reachable servers only wait for their real handshake time
    # while unavailable servers remain bounded.
    "mcp_single_query_discovery_timeout": 15.0,

    # MCP runtime behavior (distinct from the per-server definitions in
    # mcp_servers: and from the auxiliary.mcp side-LLM task settings).
    "mcp": {
        # Auto-reload MCP connections when config.yaml's mcp_servers section
        # changes at runtime (CLI file watcher, default on).
        # Set to false to stop the automatic reload: every automatic reload
        # rebuilds the agent tool surface and INVALIDATES the provider
        # prompt cache (the next message re-sends the full input prefix),
        # which is expensive on long-context / high-reasoning models.
        # When disabled, the watcher still detects the change and prints
        # guidance to apply it deliberately via /reload-mcp.
        "auto_reload_on_config_change": True,
    },

    # Tool-output truncation thresholds. When terminal output or a
    # single read_file page exceeds these limits, Hermes truncates the
    # payload sent to the model (keeping head + tail for terminal,
    # enforcing pagination for read_file). Tuning these trades context
    # footprint against how much raw output the model can see in one
    # shot. Ported from anomalyco/opencode PR #23770.
    #
    # - max_bytes:       terminal_tool output cap, in chars
    #                    (default 50_000 ≈ 12-15K tokens).
    # - max_lines:       read_file pagination cap — the maximum `limit`
    #                    a single read_file call can request before
    #                    being clamped (default 2000).
    # - max_line_length: per-line cap applied when read_file emits a
    #                    line-numbered view (default 2000 chars).
    "tool_output": {
        "max_bytes": 50_000,
        "max_lines": 2000,
        "max_line_length": 2000,
    },

    # Tool loop guardrails nudge models when they repeat failed or
    # non-progressing tool calls. Soft warnings are always-on by default;
    # hard stops are opt-in so interactive CLI/TUI sessions keep flowing.
    "tool_loop_guardrails": {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
        "warn_after": {
            "exact_failure": 2,
            "same_tool_failure": 3,
            "idempotent_no_progress": 2,
        },
        "hard_stop_after": {
            "exact_failure": 5,
            "same_tool_failure": 8,
            "idempotent_no_progress": 5,
        },
        # Per-turn runaway-loop caps (inspired by Claude Code v2.1.212,
        # Week 29, July 2026). Hard ceilings on how many times a runaway-prone
        # tool may be called within a SINGLE agent loop (turn); the counters
        # reset at the start of every turn, so a legitimate multi-turn session
        # is never starved. They are always-on and fire regardless of the
        # warn/hard-stop thresholds above. A single turn issuing dozens of web
        # searches or spawning dozens of subagents is already pathological, so
        # the defaults are low. Set either to 0 to disable that cap (unlimited).
        "loop_caps": {
            "max_web_searches": 50,   # max web_search calls per turn (0 = unlimited)
            "max_subagents": 50,      # max subagents spawned per turn (0 = unlimited)
        },
    },

    "compression": {
        "enabled": True,
        "progress_notices": False,    # opt-in (#52995): when True, routine compression
                                      # progress statuses (compacting/preflight/pre-API/
                                      # idle/retry) are delivered to chat gateway
                                      # platforms instead of being suppressed by the
                                      # gateway noise filter. Default False keeps
                                      # routine compression silent-by-design on chat
                                      # surfaces (server-side logging only). Failure
                                      # notices and manual /compress feedback are
                                      # always visible regardless of this setting.
        "threshold": 0.50,            # compress when context usage exceeds this ratio.
                                      # Models with context windows below 512K are
                                      # floored at 0.75 (raise-only) so compaction
                                      # doesn't fire with half the window still free;
                                      # set this above 0.75 to override the floor.
        "threshold_tokens": None,     # absolute token cap — when set, compression
                                      # triggers at the lower of the ratio-based
                                      # threshold and this token count. Clamped to
                                      # the model's context length at apply-time.
        "target_ratio": 0.20,         # fraction of threshold to preserve as recent tail
        "tail_mode": "legacy",        # tail retention policy (#87326):
                                      #   "legacy" — 0.20×window verbatim tail (default)
                                      #   "lean"   — clamped 2.5%-of-window tail
                                      #              (10K floor / 25K cap) plus chunked
                                      #              digests, a mechanical anchor index,
                                      #              verbatim user messages, and
                                      #              session_search recovery pointers in
                                      #              the summary. ~3x fewer retained
                                      #              tokens after compaction; costs a few
                                      #              extra summarizer calls at the
                                      #              compaction boundary.
        "protect_last_n": 20,         # minimum recent messages to keep uncompressed
        "min_tail_user_messages": 1,  # REAL (actionable) user messages guaranteed to
                                      # survive in the uncompressed tail. 1 = existing
                                      # single last-user anchor (default, behavior-
                                      # preserving); raise to e.g. 3 to keep the last
                                      # 3 real user turns verbatim when bulky tool
                                      # outputs fill the tail token budget.
        "max_attempts": 3,            # compression retry rounds before a turn gives up
                                      # with "max compression attempts reached". Raise
                                      # (e.g. 6) for tool-schema-heavy sessions where 3
                                      # rounds cannot clear the request estimate.
                                      # Validated >= 1, hard-capped at 10.
        "proactive_prune_tokens": 0,  # opt-in trigger (tokens) for the deterministic,
                                      # no-LLM tool-result prune, run independently of
                                      # `threshold` above. On large-window models
                                      # `threshold` (≈50% of the window) rarely fires,
                                      # so old tool output otherwise rides in history
                                      # and is re-sent every turn; a low value like
                                      # 48000 reclaims it early. 0 = off. Recent tail
                                      # protected by `protect_last_n`. Built-in
                                      # compressor only (other engines inherit a no-op).
                                      # NOTE: each committed prune rewrites already-sent
                                      # history, breaking the provider prompt-cache
                                      # prefix — the min_reclaim gate below keeps those
                                      # breaks episodic rather than per-turn.
        "proactive_prune_min_result_chars": 8000,  # the prune's summarize pass only
                                      # touches tool results larger than this (chars);
                                      # clamped to >= 200 so a generated summary can't
                                      # itself be re-summarized.
        "proactive_prune_min_reclaim_tokens": 4096,  # a proactive prune only commits
                                      # when it reclaims at least this many tokens
                                      # (measured on the pruned output), then waits
                                      # for a full trigger-sized token runway to
                                      # regrow before rearming. Keeps prompt-cache
                                      # breaks episodic. 0 = no minimum-savings gate.
        "micro_compact": False,       # opt-in: after each completed turn, fold the
                                      # oldest un-absorbed exchange into a rolling
                                      # summary, amortizing compression cost instead
                                      # of paying it in one batch stall. Default False
                                      # because a pass rewrites already-sent history
                                      # and so breaks the provider prompt-cache prefix
                                      # EVERY turn — the per-turn cache break that
                                      # `proactive_prune_min_reclaim_tokens` above
                                      # exists to avoid. Enable only when you have
                                      # measured that the amortized stall is worth
                                      # more to you than the cached-prefix discount.
                                      # See docs/micro-compaction.md.
        "micro_compact_every_n_turns": 1,  # cadence: run a pass every Nth completed
                                      # turn. Since each pass costs one prompt-cache
                                      # break, this is the dial for how often that
                                      # cost is paid — 1 reclaims most aggressively
                                      # at one break per turn, 5 trades reclaim rate
                                      # for a fifth of the breaks. Clamped to >= 1.
                                      # Ignored unless `micro_compact` is true.
        "micro_compact_defrag_threshold_tokens": 2000,  # once the rolling summary
                                      # exceeds this many tokens, the next pass
                                      # re-summarizes the summary itself instead of
                                      # letting it grow without bound.
        "hygiene_hard_message_limit": 5000,  # gateway session-hygiene force-compress threshold by message count
        "hygiene_timeout_seconds": 30,  # max seconds gateway waits for pre-agent hygiene compression
                                      # WITHOUT forward progress. The summary call streams, so
                                      # this is an inactivity budget: a slow model still
                                      # producing tokens keeps extending the wait; only a
                                      # silent/hung call is cut off.
        "hygiene_total_ceiling_seconds": 600,  # absolute cap on the hygiene compression wait even
                                      # while tokens are still moving — bounds a degenerate
                                      # trickle stream. Clamped to >= hygiene_timeout_seconds.
        "hygiene_failure_cooldown_seconds": 300,  # skip repeated failed hygiene attempts for this session
        "context_timeout_seconds": 120,  # inactivity budget for in-agent compress_context
                                      # (conversation loop, /compress, preflight, etc.).
                                      # Same progress-aware semantics as hygiene_timeout_seconds:
                                      # streamed summary tokens extend the wait; only a silent
                                      # worker is cut off. 0 = disable the owned wrapper
                                      # (callers that already pass commit_fence, e.g. gateway
                                      # hygiene, never use this path).
        "context_total_ceiling_seconds": 600,  # absolute cap on the *pre-commit*
                                      # in-agent compress_context wait (summary /
                                      # stream phase) even while tokens are still
                                      # moving. Clamped to >= context_timeout_seconds
                                      # when the idle budget is > 0. Guarantee:
                                      # the summary phase is bounded by this
                                      # ceiling; an already-started SessionDB
                                      # commit is never abandoned mid-flight —
                                      # if the commit itself runs past the
                                      # ceiling it is logged (WARNING, then
                                      # ERROR) and surfaced to the user via the
                                      # warning channel while the host keeps
                                      # waiting in bounded increments for the
                                      # commit to finish.
        "protect_first_n": 3,         # non-system head messages always preserved
                                      # verbatim, in ADDITION to the system prompt
                                      # (which is always implicitly protected). Set to
                                      # 0 for long-running rolling-compaction sessions
                                      # where you want nothing pinned except the
                                      # system prompt + rolling summary + recent tail.
        "abort_on_summary_failure": False,  # When True, auto-compression that fails
                                      # to generate a summary (aux LLM errored / returned
                                      # non-JSON / timed out) aborts entirely instead of
                                      # dropping the middle window with a static
                                      # "summary unavailable" placeholder.  Messages are
                                      # preserved unchanged and the session "freezes" at
                                      # its current size until the user runs /compress
                                      # (which bypasses the failure cooldown) or /new.
                                      # Default False matches historical behavior; set to
                                      # True if you'd rather pause than silently lose
                                      # context turns when your aux model is flaky.
        "codex_gpt55_autoraise": True,  # Historical key name kept for compatibility.
                                      # When True, gpt-5.4 / gpt-5.5 / gpt-5.6 on the
                                      # ChatGPT Codex OAuth route raise their compaction
                                      # trigger to 85% (vs the global `threshold` above).
                                      # Codex hard-caps these families at a 272K window, so
                                      # the default 50% would compact at ~136K and waste half
                                      # the usable context. Set to False to opt back down to
                                      # the global threshold (e.g. 0.50) for those Codex
                                      # sessions. Only this exact route is affected —
                                      # gpt-5.4 / 5.5 / 5.6 on OpenAI's direct API,
                                      # OpenRouter, and Copilot keep the global threshold
                                      # regardless.
        "codex_gpt55_autoraise_notice": True,  # Display the one-time Codex gpt-5.4/5.5/5.6
                                      # autoraise banner. Set False to keep the
                                      # 85% threshold autoraise but suppress the
                                      # user-facing notice in CLI/gateway output.
        "codex_app_server_auto": "native",  # Codex app-server (codex CLI runtime) thread
                                      # compaction mode. The codex agent owns the real
                                      # thread context, so Hermes' summarizer cannot
                                      # shrink it (#36801). native = codex decides when
                                      # to compact its own thread (default); hermes =
                                      # Hermes' compression threshold triggers
                                      # thread/compact/start; off = never auto-trigger
                                      # (codex may still compact natively).
        "codex_responses_native": False,  # Opt in to OpenAI's server-side compaction
                                      # on the Responses API. Engages ONLY for
                                      # gpt-5.6-family models on api.openai.com or
                                      # the ChatGPT Codex backend; every other
                                      # route/model is unaffected. Hermes' local
                                      # compression stays armed as the fallback.
        "codex_responses_compact_threshold": 200000,  # Server-side compaction trigger
                                      # (input tokens). Clamped below the local
                                      # compression threshold at request time so
                                      # the server compacts before Hermes does.
        "in_place": True,             # When True, compaction rewrites the message
                                      # list and rebuilds the system prompt WITHOUT
                                      # rotating the session id — the conversation
                                      # keeps one durable id for its whole life
                                      # (no parent_session_id chain, no `name #N`
                                      # renumbering). Eliminates the session-rotation
                                      # bug cluster (#33618 /goal loss, #14238 lost
                                      # response, #33907 orphans, #45117 search gaps,
                                      # #42228 null cwd) — see #38763. Non-destructive:
                                      # the live context is compacted (lossy for what
                                      # the model reloads), but the pre-compaction
                                      # turns are soft-archived under the same id
                                      # (active=0, compacted=1) — still searchable via
                                      # session_search and recoverable, not deleted.
                                      # Default True since 2107b86024; set False to
                                      # restore the legacy rotating-compaction path.
        "model_thresholds": {},       # Per-model threshold overrides. Keys are
                                      # substring-matched against the model name
                                      # (longest match wins); values replace the
                                      # global `threshold` for that model, e.g.
                                      #   model_thresholds:
                                      #     "glm-5.2": 0.40
                                      #     "claude-sonnet": 0.35
                                      # The small-context floor (0.75 for <512K
                                      # models) still applies on top of overrides
                                      # (raise-only: an override above the floor
                                      # wins; one below it is raised to the floor).
        "idle_compact_after_seconds": 0,  # Opt-in idle compaction (0 = disabled).
                                      # When > 0, a session that resumes after at
                                      # least this many seconds of inactivity
                                      # compacts its accumulated history up front,
                                      # before the first reply — so a long-lived
                                      # thread resumed hours later doesn't re-read
                                      # its full stale context on every turn.
                                      # Time-based; complements (does not replace)
                                      # the size-based `threshold` above. Skipped
                                      # when the context is already at/below the
                                      # post-compression target (threshold ×
                                      # target_ratio) and it honors the same
                                      # failure-cooldown / anti-thrash / per-session
                                      # lock guards as every automatic compaction.
                                      # Example: 1800 = compact after 30 min idle.
    },

    # Anthropic prompt caching (Claude via OpenRouter or native Anthropic API).
    # cache_ttl: "5m" or "1h" (Anthropic-supported tiers). Other non-falsy
    # values are silently ignored. Falsy values (false, null, "off",
    # "disabled", "no", "none") disable prompt caching entirely.
    "prompt_caching": {
        "cache_ttl": "5m",
    },

    # OpenRouter-specific settings.
    # response_cache: enable OpenRouter response caching (X-OpenRouter-Cache header).
    #   When enabled, identical requests return cached responses for free (zero billing).
    #   This is separate from Anthropic prompt caching and works alongside it.
    #   See: https://openrouter.ai/docs/guides/features/response-caching
    # response_cache_ttl: how long cached responses remain valid, in seconds (1-86400).
    #   Default 300 (5 minutes). Only used when response_cache is enabled.
    # min_coding_score: knob for the openrouter/pareto-code router (0.0-1.0).
    #   Only applied when model.model is "openrouter/pareto-code". Higher
    #   values route to stronger (more expensive) coders; lower values open
    #   up cheaper, faster options. Default 0.65 lands on the mid-tier
    #   coder on the current Pareto frontier. Empty string = let OpenRouter
    #   pick the strongest available coder (router's documented default
    #   when the plugins block is omitted).
    #   See: https://openrouter.ai/docs/guides/routing/routers/pareto-router
    "openrouter": {
        "response_cache": True,
        "response_cache_ttl": 300,
        "min_coding_score": 0.65,
    },

    # AWS Bedrock provider configuration.
    # Only used when model.provider is "bedrock".
    "bedrock": {
        "region": "",  # AWS region for Bedrock API calls (empty = AWS_REGION env var → us-east-1)
        "discovery": {
            "enabled": True,           # Auto-discover models via ListFoundationModels
            "provider_filter": [],     # Only show models from these providers (e.g. ["anthropic", "amazon"])
            "refresh_interval": 3600,  # Cache discovery results for this many seconds
        },
        "guardrail": {
            # Amazon Bedrock Guardrails — content filtering and safety policies.
            # Create a guardrail in the Bedrock console, then set the ID and version here.
            # See: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
            "guardrail_identifier": "",  # e.g. "abc123def456"
            "guardrail_version": "",     # e.g. "1" or "DRAFT"
            "stream_processing_mode": "async",  # "sync" or "async"
            "trace": "disabled",         # "enabled", "disabled", or "enabled_full"
        },
    },

    # Auxiliary model config — provider:model for each side task.
    # Format: provider is the provider name, model is the model slug.
    # "auto" for provider = auto-detect best available provider.
    # Empty model = use provider's default auxiliary model.
    # All tasks fall back to openrouter:google/gemini-3-flash-preview if
    # the configured provider is unavailable.
    #
    # extra_body: forwarded verbatim as request body fields on every aux call
    # for that task. Use this to set provider-specific knobs (independent of
    # main-agent settings). On OpenRouter you can set provider routing prefs
    # and the Pareto Code coding-score floor here. Example:
    #
    #   auxiliary:
    #     compression:
    #       provider: openrouter
    #       model: openrouter/pareto-code
    #       extra_body:
    #         provider:           # OpenRouter provider routing
    #           order: [anthropic, google]
    #           sort: throughput  # or price | latency
    #         plugins:            # OpenRouter Pareto Code router
    #           - id: pareto-router
    #             min_coding_score: 0.5
    #
    # Each aux task is independent — main-agent provider_routing and
    # openrouter.min_coding_score do NOT propagate to aux calls by design.
    "auxiliary": {
        # Same-provider retries for a transient transport blip (connection
        # reset / timeout / 5xx / 408) on ANY auxiliary call before falling
        # back. Default 2 (→ 3 total attempts), clamped [0,6]. Matters most for
        # pinned calls like MoA reference advisors, where provider fallback is
        # not a meaningful recovery, so an unretried blip silently loses the
        # call.
        "transient_retries": 2,
        # Restrict the auxiliary auto-chain's OpenRouter fallback to free
        # (:free) SKUs. When true, the OpenRouter step is skipped entirely
        # unless the resolved fallback model ends in ":free" — a PAID lane
        # is never engaged for background auxiliary traffic (compression,
        # title generation, session search, vision, web extract) even when
        # OPENROUTER_API_KEY is present. Default false keeps the historical
        # paid fallback for users who want it.
        "free_only": False,
        # Override the auxiliary auto-chain's OpenRouter fallback model
        # (default: google/gemini-3.6-flash, a PAID model). Set e.g.
        # "nvidia/nemotron-3-ultra-550b-a55b:free" together with
        # free_only: true to keep auxiliary traffic free-only. A one-time
        # WARNING is logged whenever a non-":free" model is engaged.
        "openrouter_model": "",
        # Endpoints that reject NON-streaming chat requests outright (e.g.
        # Tencent Copilot returns HTTP 400 "Non-stream chat request is
        # currently not supported"). Auxiliary calls to a matching endpoint
        # are sent with stream=True and aggregated client-side. Entries are
        # case-insensitive substrings matched against the endpoint URL;
        # copilot.tencent.com is always treated as stream-only.
        "stream_only_base_urls": [],
        "vision": {
            "provider": "auto",    # auto | openrouter | nous | codex | custom
            "model": "",           # e.g. "google/gemini-2.5-flash", "gpt-4o"
            "base_url": "",        # direct OpenAI-compatible endpoint (takes precedence over provider)
            "api_key": "",         # API key for base_url (falls back to OPENAI_API_KEY)
            "timeout": 120,        # seconds — LLM API call timeout; vision payloads need generous timeout
            "extra_body": {},      # OpenAI-compatible provider-specific request fields
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "download_timeout": 30,  # seconds — image HTTP download timeout; increase for slow connections
        },
        "web_extract": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 360,        # seconds (6min) — per-attempt LLM summarization timeout; increase for slow local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "compression": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,        # seconds — compression summarises large contexts; increase for local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Note: session_search no longer uses an auxiliary LLM (PR #27590 —
        # single-shape tool returns DB content directly). The old
        # ``auxiliary.session_search.*`` block was removed here. Existing
        # values in user config.yaml files are harmless leftovers and ignored.
        "skills_hub": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "approval": {
            "provider": "auto",
            "model": "",           # fast/cheap model recommended (e.g. gemini-flash, haiku)
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "mcp": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "title_generation": {
            "enabled": True,
            "provider": "auto",
            "model": "",
            "prefer_fast_model": False,  # opt in to provider fast tier; auto otherwise uses the main model
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "language": "",
        },
        "memory_query_rewrite": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 8,
            "extra_body": {},
        },
        "tts_audio_tags": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Triage specifier — flesh out a rough one-liner in the Kanban
        # Triage column into a concrete spec, then promote it to ``todo``.
        # Invoked by ``hermes kanban specify`` (single id or --all). Set a
        # cheap, capable model here (gemini-flash works well); the main
        # model is overkill for short spec expansion.
        "triage_specifier": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Kanban decomposer — decomposes a triage task into a graph of
        # child tasks routed to specialist profiles by description.
        # Invoked by ``hermes kanban decompose`` and the kanban
        # auto-decompose dispatcher tick. Returns a JSON task graph;
        # uses more tokens than the specifier so allow more headroom.
        "kanban_decomposer": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 180,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Profile describer — auto-generates a 1-2 sentence description
        # of what a profile is good at. Invoked by
        # ``hermes profile describe <name> --auto`` and the dashboard's
        # auto-generate button. Short, cheap call.
        "profile_describer": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Goal judge — evaluates whether a /goal run's latest response
        # satisfies the goal/contract, and drafts goal contracts. Short
        # structured-JSON calls; a fast cheap model is fine.
        "goal_judge": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Curator — skill-usage review fork. Timeout is generous because the
        # review pass can take several minutes on reasoning models (umbrella
        # building over hundreds of candidate skills). "auto" = use main chat
        # model; override via `hermes model` → auxiliary → Curator to route
        # to a cheaper aux model (e.g. openrouter google/gemini-3-flash-preview).
        "curator": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 600,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Monitor — urgency/importance classifier used by the important-mail
        # monitor catalog automation (cron/scripts/classify_items.py). Scores
        # candidate items 0-10 against the user's criteria so only above-
        # threshold items get delivered. "auto" = main chat model; override to
        # a cheap fast model (e.g. openrouter google/gemini-3-flash-preview,
        # haiku) since per-item scoring is high-volume and a small model is fine.
        "monitor": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Background review — the post-turn self-improvement fork that decides
        # whether to save a memory / patch a skill. "auto" (default) = run on
        # the main chat model, replaying the full conversation, which is already
        # warm in the prompt cache (cheap cache reads) — unchanged, optimal.
        # Set provider/model to a cheaper model (e.g. openrouter
        # google/gemini-3-flash-preview) to run the review there for ~3-5x lower
        # cost. A different model can't reuse the main prompt cache anyway, so
        # the fork automatically replays a compact digest instead of the full
        # transcript when routed (minimises the cold-write). Same model = full
        # replay; different model = digest. Quality holds (memory capture
        # identical, skill near-identical in benchmarks).
        "background_review": {
            # Master switch for automatic post-turn memory/skill review forks.
            # false = skip automatic spawns (manual /refine still works).
            "enabled": True,
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "moa_reference": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 900,
            "extra_body": {},
            # NOTE: no reasoning_effort here by design — MoA reasoning depth is
            # configured PER SLOT in the MoA preset (moa.presets.<name>.
            # reference_models[].reasoning_effort / aggregator.reasoning_effort),
            # not at the auxiliary-task level.
        },
        "moa_aggregator": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 900,
            "extra_body": {},
            # NOTE: no reasoning_effort here by design — see moa_reference above.
        },
    },
    
    "display": {
        "compact": False,
        "personality": "",
        "resume_display": "full",
        # Recap tuning for /resume and startup resume. The defaults match the
        # historical hardcoded values; expose them as config so power users can
        # widen or tighten the snapshot to taste.
        "resume_exchanges": 10,            # max user+assistant pairs to show
        "resume_max_user_chars": 300,      # truncate user message text
        "resume_max_assistant_chars": 200, # truncate non-last assistant text
        "resume_max_assistant_lines": 3,   # truncate non-last assistant lines
        # When True (default), assistant entries that are *only* tool calls
        # (no visible text) are skipped in the recap. This prevents the recap
        # from being dominated by `[2 tool calls: terminal, read_file]` lines
        # when an exchange was tool-heavy. Set False to restore the legacy
        # behavior of showing tool-call summaries inline.
        "resume_skip_tool_only": True,
        "busy_input_mode": "interrupt",  # interrupt | queue | steer
        # When busy_input_mode="steer", suppress only the visible
        # "Steered into current run" confirmation bubble by setting this false.
        # The mid-turn steering itself still happens.
        "busy_steer_ack_enabled": True,
        # Classic CLI multiline fallbacks beyond Alt+Enter.
        # Default true matches Claude Code / Codex / OpenCode: Ctrl+J inserts
        # a newline, a trailing backslash followed by Enter continues the draft,
        # and supported terminals are asked to report Shift+Enter distinctly.
        # Set false to restore the legacy c-j submit fallback on unusual POSIX
        # PTYs whose plain Enter arrives as LF instead of CR.
        "cli_multiline_shortcuts": True,
        # Which interface bare `hermes` (and `hermes chat`) launches by default:
        #   "cli" — the classic prompt_toolkit REPL (default, preserves prior behavior)
        #   "tui" — the modern Ink TUI (same as passing `--tui`)
        # Explicit flags always win over this setting: `--cli` forces the classic
        # REPL and `--tui` (or HERMES_TUI=1) forces the TUI regardless of config.
        "interface": "cli",
        # When true, `hermes --tui` auto-resumes the most recent human-
        # facing session on launch instead of forging a fresh one.
        # Mirrors `hermes -c` muscle memory.  Default off so existing
        # users aren't surprised.  HERMES_TUI_RESUME=<id> always wins.
        "tui_auto_resume_recent": False,
        # When true (default), `hermes --tui` drops a one-time hint
        # ("subagents working · /agents to watch live") the first time a turn
        # starts delegating, nudging the user toward the live spawn-tree
        # dashboard. Set false to suppress the hint.
        "tui_agents_nudge": True,
        "bell_on_complete": False,
        # Stream the model's reasoning/thinking live before the response.
        # Default ON: on thinking models the reasoning phase can run tens of
        # seconds, and with this off the user stares at a spinner the whole
        # time even though tokens are streaming. Set false for quiet output.
        "show_reasoning": True,
        # When reasoning display is on, the post-response "Reasoning" recap box
        # collapses long thinking to the first 10 lines. Set true to print the
        # complete thinking text uncollapsed (live streaming is always full).
        "reasoning_full": False,
        # Background self-improvement review notifications surfaced in chat.
        #   "off"     — no chat notification (the review still runs and writes)
        #   "on"      — generic "💾 Memory updated" line (default)
        #   "verbose" — include a compact content preview of what changed
        # Per-platform overrides via display.platforms.<platform>.memory_notifications.
        "memory_notifications": "on",
        # Gateway notifications when a terminal(background=true) process
        # finishes:
        #   "concise" — one-line status message; failures append a short
        #               output tail (default)
        #   "all"     — running-output updates + final raw-output message
        #   "result"  — final raw-output message only
        #   "error"   — final raw-output message only on non-zero exit
        #   "off"     — no watcher messages at all
        "background_process_notifications": "concise",
        "streaming": False,
        "timestamps": False,      # Show message timestamps (CLI labels, TUI rows, desktop transcript)
        "timestamp_format": "%H:%M",  # strftime format for timestamps (e.g. "%b-%d %H:%M")
        "final_response_markdown": "strip",  # render | strip | raw
        # Preserve recent classic CLI output across Ctrl+L, /redraw, and
        # terminal resize full-screen clears. Disable if a terminal emulator
        # behaves badly with replayed scrollback.
        "persistent_output": True,
        "persistent_output_max_lines": 200,
        # Clear terminal scrollback as well as the visible viewport when the
        # classic CLI performs a full redraw/resize recovery. Disabled by
        # default because some users prefer preserving terminal history;
        # enable when a terminal/tmux stack stamps stale prompt chrome into
        # scrollback during fullscreen/restore window transitions.
        "cli_rebuild_scrollback_on_redraw": False,
        # Print a one-line summary of resolved modal prompts (approval /
        # clarify) into scrollback so the question and decision survive the
        # panel repaint. Set false to keep scrollback untouched.
        "persist_prompts": True,
        "inline_diffs": True,     # Show inline diff previews for write actions (write_file, patch, skill_manage)
        # File-mutation verifier footer.  When true (default), the agent
        # appends a one-line advisory to its final response whenever a
        # write_file / patch call failed during the turn and was never
        # superseded by a successful write to the same path.  This catches
        # the "batch of parallel patches, half fail, model claims success"
        # class of over-claim that otherwise forces users to run
        # `git status` to verify edits landed.  Set false to suppress.
        "file_mutation_verifier": True,
        # Nous credits status-bar notices (usage bands, grant-spent, depleted /
        # restored).  When false, no credits notices are emitted — balance data
        # is still captured and /usage keeps working.  Off switch for sub +
        # top-up users who find the gauge noisy.
        "credits_notices": True,
        # Turn-completion explainer.  When true (default), the agent appends a
        # one-line explanation to its final response whenever a turn ends
        # abnormally with no usable reply — empty content after retries, a
        # partial/truncated stream, a still-pending tool result, or an
        # iteration/budget limit.  Replaces the bare "(empty)" sentinel so the
        # failure isn't silent from the UI's perspective.  Set false to suppress.
        "turn_completion_explainer": True,
        "show_cost": False,       # Show $ cost in the status bar (off by default)
        # Show a color-coded battery read-out as the first status-bar element in
        # the CLI/TUI (off by default). No-op on machines without a battery.
        "battery": False,
        # Focus view (/focus): display-only reduced-output mode. When true the
        # CLI/TUI pins tool_progress to "off" (reusing the existing suppression
        # path), reports a per-turn hidden-line count with a recovery hint, and
        # pins a "focus" segment in the status bar. focus_saved_tool_progress
        # holds the mode /focus off restores. Never affects what is sent to the
        # model — see hermes_cli/focus_view.py.
        "focus_view": False,
        "focus_saved_tool_progress": "all",
        "skin": "default",
        # UI language for static user-facing messages (approval prompts, a
        # handful of gateway slash-command replies).  Does NOT affect agent
        # responses, log lines, tool outputs, or slash-command descriptions.
        # Supported: en, zh, ja, de, es, fr, tr, uk.  Unknown values fall back to en.
        "language": "en",
        # TUI busy indicator style: kaomoji (default), emoji, unicode (braille
        # spinner), or ascii.  Live-swappable via `/indicator <style>`.
        "tui_status_indicator": "kaomoji",
        # Seconds between prompt_toolkit redraws in the classic CLI when idle.
        # Default 1.0 keeps the wall-clock status-bar read-outs (idle-since-
        # last-turn) ticking and keeps the bottom chrome alive during idle —
        # without it prompt_toolkit stops repainting the status bar after a
        # turn and it can go stale/disappear (#45592).
        # Set 0 to disable the background refresh if it fights terminal
        # auto-scroll in non-fullscreen mode on some emulators (#48309).
        "cli_refresh_interval": 1.0,
        "user_message_preview": {  # CLI: how many submitted user-message lines to echo back in scrollback
            "first_lines": 2,
            "last_lines": 2,
        },
        "interim_assistant_messages": True,  # Gateway: send natural mid-turn assistant status messages. Desktop: keep mid-turn narration between tool calls instead of collapsing to the final message.
        # Codex Responses models narrate progress in a dedicated commentary
        # channel. When true (default), completed commentary messages are
        # delivered as visible mid-turn updates via the interim message path.
        # When false, commentary falls back to the reasoning channel and is
        # only visible when show_reasoning is enabled.
        "show_commentary": True,
        "tool_progress_command": False,  # Enable /verbose command in messaging gateway
        # NOTE: display.tool_progress_overrides is deprecated and no longer
        # seeded here — use display.platforms. A user-set value is still
        # honored at runtime (gateway display_config back-compat read) and
        # folded into display.platforms by the v15→16 migration.
        "tool_preview_length": 0,  # Max chars for tool call previews (0 = no limit, show full paths/commands)
        # Human-phrased tool status labels for built-in tools: "Searching the
        # web for ...", "Reading <file>", "Browsing <url>" instead of the raw
        # tool name. Applies to CLI spinner + gateway/desktop tool-progress.
        # Custom/plugin/MCP tools always fall back to the raw preview.
        "friendly_tool_labels": True,
        # CLI-only post-turn accounting line printed after each interactive turn:
        # "⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands".
        # Observed from the tool-progress feed the CLI already receives; never
        # printed in quiet/non-interactive paths or in gateway/messaging
        # surfaces (those have their own runtime footer).
        "turn_summary": True,
        # CLI-only: append cumulative turn output tokens to the live spinner
        # timer ("⚡ Reading file  ( 2.3s · ↓ 1.2k tok)"). Updates as each API
        # call in the turn reports usage.
        "spinner_token_flow": True,
        # How gateway tool-progress is grouped on platforms that support message
        # editing: "accumulate" (default) edits one bubble in place; "separate"
        # sends one message per tool (the pre-v0.9 behavior, noisier). Only
        # applies where tool_progress is already enabled. Per-platform override
        # via display.platforms.<platform>.tool_progress_grouping.
        "tool_progress_grouping": "accumulate",
        # Optional custom phrases for generic long-running status messages.
        # Built-in defaults live in gateway/assets/status_phrases.yaml. Users
        # can set `path`/`paths` to HERMES_HOME-relative YAML files/directories
        # (or rely on conventional status_phrases.yaml / status_phrases/*.yaml).
        # Keys: status, generic. Use
        # mode: "append" (default) to add phrases, or "replace" to fully
        # replace configured surfaces. Per-platform overrides live under
        # display.platforms.<platform>.status_phrases.
        "status_phrases": {},
        # How a reasoning/thinking summary renders when show_reasoning is on.
        # "code" (default) = 💭 fenced code block; "blockquote" = "> " lines;
        # "subtext" = "-# " lines (Discord small grey metadata text). Discord
        # defaults to "subtext"; override per-platform via
        # display.platforms.<platform>.reasoning_style.
        "reasoning_style": "code",
        # Auto-delete system-notice replies (e.g. "✨ New session started!",
        # "♻ Restarting gateway…", "⚡ Stopped…") after N seconds on platforms
        # that support message deletion (currently Telegram; other platforms
        # ignore and leave the message in place).  Only affects slash-command
        # replies wrapped with gateway.platforms.base.EphemeralReply — agent
        # responses and content messages are never touched.  Default 0
        # (disabled) preserves prior behavior.
        "ephemeral_system_ttl": 0,
        # Per-platform display/streaming overrides. Each key is a gateway
        # platform ("telegram", "discord", "slack", …) mapping to a dict of
        # display settings that override the global value for that platform
        # only. A setting left unset here falls through to the global default.
        #
        # Shipped defaults encode the streaming experience that works best
        # per platform:
        #   - Telegram has native animated draft streaming (sendMessageDraft),
        #     which is smooth, so streaming is on by default there.
        #   - Discord and Slack only have edit-based streaming (repeated
        #     editMessage), which flickers and is noticeably jankier, so
        #     streaming is off by default for both.
        # These are gap-fillers: a user who explicitly sets, e.g.,
        # display.platforms.discord.streaming: true keeps their value
        # (config deep-merge has user values win over defaults). The global
        # streaming.enabled master switch still gates everything — these
        # per-platform flags only take effect once streaming is enabled.
        "platforms": {
            "telegram": {"streaming": True},
            "discord": {"streaming": False},
            "slack": {"streaming": False},
        },
        # Gateway runtime-metadata footer appended to the FINAL message of a turn
        # (disabled by default to keep replies minimal). When enabled, renders
        # e.g. `model · 68% · ~/projects/hermes`. Per-platform overrides go under
        # display.platforms.<platform>.runtime_footer.
        "runtime_footer": {
            "enabled": False,
            "fields": ["model", "context_pct", "cwd"],  # Order shown; drop any to hide
        },
        "copy_shortcut": "auto",  # "auto" (platform default) | "ctrl_c" | "ctrl_shift_c" | "disabled"
        # Petdex animated mascot (https://github.com/crafter-station/petdex).
        # A purely cosmetic sprite that reacts to agent activity across the
        # CLI, TUI, and desktop app. Manage with `hermes pets`. Disabled until
        # a pet is installed + selected (no effect on prompt caching — this is
        # a display concern only).
        "pet": {
            "enabled": False,
            # Active pet slug; resolved against installed pets in
            # get_hermes_home()/pets/. Empty → first installed pet.
            "slug": "",
            # Terminal render protocol for CLI/TUI:
            #   auto  — detect kitty/iTerm2/sixel, else unicode half-blocks
            #   kitty | iterm | sixel | unicode | off
            "render_mode": "auto",
            # Master size scalar (relative to native 192×208 frames). One knob
            # shrinks every surface: the desktop canvas scales its pixels by it
            # and the CLI/TUI derive their terminal column width from it. The
            # half-block fallback clamps to a legibility floor (it can't shrink
            # as far as true-pixel kitty/GUI without turning to mush).
            "scale": 0.33,
            # Hard override for terminal column width. 0 = auto (derive from
            # scale); set a positive int only to pin the half-block/kitty width
            # independently of scale.
            "unicode_cols": 0,
        },
    },

    # Web dashboard settings
    "dashboard": {
        "theme": "default",  # Dashboard visual theme: "default", "midnight", "ember", "mono", "cyberpunk", "rose"
        # Process-isolation rollout controls. Runtime reads these through the
        # raw config loader, so tui_gateway.server also owns explicit defaults.
        "turn_isolation": False,
        "compute_host_heartbeat_secs": 15,
        "compute_host_respawn_max": 3,
        # Hide the token/cost analytics surfaces (Analytics page, token bars and
        # cost figures on the Models page) by default.  The numbers shown there
        # are a local debug estimate: they only count successful main-agent
        # responses with a usable ``response.usage``, and silently exclude every
        # auxiliary call (context compression, title generation, vision,
        # session search, web extract, smart approval, MCP routing, plugin LLM
        # access) plus provider-side retries, fallback attempts, and any call
        # whose usage block didn't come back.  Cache writes are also missing
        # from the API response.  On models with heavy auxiliary traffic
        # (Kimi K2.6, MiniMax M2.7) the local total can be 10x-100x lower than
        # the provider bill, which is worse than hiding the numbers entirely
        # because they look precise enough to compare against the provider.
        # Set this to True to re-enable the surfaces with the understanding
        # that the numbers are a local lower-bound estimate, not billing.
        "show_token_analytics": False,
        # OAuth gate configuration (engaged when ``--host`` is set and
        # ``--insecure`` is not). The bundled Nous Portal plugin reads
        # both keys at startup; they are the canonical surface for these
        # settings. Each can be overridden by an environment variable —
        # ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` and
        # ``HERMES_DASHBOARD_PORTAL_URL`` respectively — and the env var
        # wins when set to a non-empty value. The override path is what
        # Fly.io's platform-secret injection uses to push the per-deploy
        # client_id at provisioning time without operators needing to
        # touch config.yaml. Local dev / non-Fly deploys can set either
        # surface; missing values fall through to the plugin's defaults
        # (no provider registered when ``client_id`` is empty;
        # ``portal_url`` defaults to https://portal.nousresearch.com).
        "oauth": {
            "client_id": "",  # agent:{instance_id} — Portal provisions this
            "portal_url": "",  # blank → use plugin default (production Portal)
        },
        # Username/password gate configuration — read by the bundled
        # ``dashboard_auth/basic`` plugin (a self-hosted "just put a
        # password on my dashboard" provider that needs no OAuth IDP).
        # The plugin registers a password provider when ``username`` plus
        # either ``password_hash`` (preferred — no plaintext at rest) or
        # ``password`` (plaintext, hashed in-memory at load) are set. Each
        # key is overridable by an env var
        # (``HERMES_DASHBOARD_BASIC_AUTH_USERNAME`` /
        # ``_PASSWORD_HASH`` / ``_PASSWORD`` / ``_SECRET`` /
        # ``_TTL_SECONDS``), env winning when non-empty. Leave ``username``
        # empty (the default) to keep the plugin a no-op — loopback /
        # ``--insecure`` operators and OAuth users are unaffected.
        #
        # ``secret`` is the HMAC key used to sign the stateless session
        # tokens this provider mints. When empty, a random per-process key
        # is generated — fine for a single process, but sessions then
        # don't survive a restart or span multiple workers. Set an
        # explicit ``secret`` (32+ random bytes, base64/hex/raw) for
        # stable multi-worker / restart-surviving sessions. Compute a
        # ``password_hash`` with
        # ``python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"``.
        "basic_auth": {
            "username": "",  # blank → plugin no-op (no password provider)
            "password_hash": "",  # scrypt$... (preferred — no plaintext at rest)
            "password": "",  # plaintext fallback (hashed in-memory at load)
            "secret": "",  # token-signing key; blank → random per-process
            "session_ttl_seconds": 0,  # 0 → plugin default (12h)
        },
        # Drain-control service-credential configuration — read by the
        # bundled ``dashboard_auth/drain`` plugin (the first consumer of the
        # generic non-interactive token-auth capability). The SECRET itself
        # is a credential and is NOT configured here: it is provisioned by
        # nous-account-service at deploy time via the
        # ``HERMES_DASHBOARD_DRAIN_SECRET`` env var (the .env-is-for-secrets
        # rule). These are the behavioural knobs only. The plugin is a no-op
        # unless that env var is set to a >=256-bit secret; a weak secret is
        # rejected at registration (fail-closed) and the drain endpoint stays
        # disabled. ``scope`` is the capability label attached to the verified
        # principal; ``min_secret_chars`` is the entropy bar (url-safe-b64
        # chars; 43 ~= 256 bits).
        "drain_auth": {
            "scope": "drain",
            "min_secret_chars": 43,
        },
        # Public URL override (env: ``HERMES_DASHBOARD_PUBLIC_URL``).
        # When set, this is the complete authority — scheme + host +
        # optional path prefix (e.g. ``https://example.com/hermes``) —
        # the OAuth ``redirect_uri`` is built from. Set this for deploys
        # behind reverse proxies that don't reliably forward
        # ``X-Forwarded-Host`` / ``X-Forwarded-Proto`` / ``X-Forwarded-Prefix``
        # (manual nginx setups, on-prem ingresses, custom-domain Fly
        # deploys without proper proxy headers). When set,
        # ``X-Forwarded-Prefix`` is IGNORED on the OAuth path because
        # the operator has declared the public URL — we no longer need
        # to guess from proxy headers, and stacking the prefix on top
        # would double-prefix the common case where the prefix is
        # already baked into ``public_url``. Leave empty to use the
        # existing proxy-header reconstruction (the default).
        #
        # Validation: rejects values without ``http(s)://`` scheme or
        # without a host, and any string containing quote / angle /
        # whitespace / control characters. A malformed value silently
        # falls through to request reconstruction rather than breaking
        # the login flow.
        "public_url": "",
    },

    # Privacy settings
    "privacy": {
        "redact_pii": False,  # When True, hash user IDs and strip phone numbers from LLM context
    },

    # Text-to-speech configuration
    # Each provider supports an optional `max_text_length:` override for the
    # per-request input-character cap. Omit it to use the provider's documented
    # limit (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k model-aware,
    # Gemini 32000, Edge 5000, Mistral 4000, NeuTTS/KittenTTS 2000).
    "tts": {
        # Set explicitly to pin a backend:
        # "edge" (free) | "elevenlabs" (premium) | "openai" | "xai" | "minimax" | "mistral" | "gemini" | "deepinfra" | "neutts" (local) | "kittentts" (local) | "piper" (local)
        "provider": "edge",
        "edge": {
            "voice": "en-US-AriaNeural",
            # Popular: AriaNeural, JennyNeural, AndrewNeural, BrianNeural, SoniaNeural
        },
        "elevenlabs": {
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam
            "model_id": "eleven_multilingual_v2",
        },
        "openai": {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            # Voices: alloy, ash, ballad, cedar, coral, echo, fable, marin,
            # nova, onyx, sage, shimmer, verse (gpt-4o-mini-tts; the tts-1
            # era stopped at alloy/echo/fable/onyx/nova/shimmer)
        },
        "gemini": {
            "model": "gemini-2.5-flash-preview-tts",
            "voice": "Kore",
            # When true, Gemini 3.1 TTS uses a hidden auxiliary-model rewrite
            # pass to insert freeform square-bracket audio tags into the TTS
            # script. Visible chat replies are unchanged.
            "audio_tags": False,
            # Optional local Markdown/text file with Gemini TTS performance
            # direction. It may include AUDIO PROFILE, SCENE, DIRECTOR'S NOTES,
            # SAMPLE CONTEXT, and either a `{transcript}` placeholder or no
            # transcript section; Hermes appends the live transcript when absent.
            "persona_prompt_file": "",
        },
        "xai": {
            "voice_id": "eve",  # or custom voice ID — see https://docs.x.ai/developers/model-capabilities/audio/custom-voices
            "language": "en",  # BCP-47 code ("en", "pt-BR") or "auto"
            "speed": 1.0,  # 0.7–1.5, playback speed
            "auto_speech_tags": False,  # insert expressive audio tags via LLM rewrite
            "optimize_streaming_latency": 0,  # 0–2, trades quality for lower latency
            "sample_rate": 24000,  # 22050 / 24000 / 44100 / 48000
            "bit_rate": 128000,  # MP3 bitrate; only applies when codec=mp3
        },
        "mistral": {
            "model": "voxtral-mini-tts-2603",
            "voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",  # Paul - Neutral
        },
        "minimax": {
            "model": "speech-02-hd",
            "voice_id": "English_expressive_narrator",
        },
        "kittentts": {
            "model": "KittenML/kitten-tts-nano-0.8-int8",  # nano 25MB; micro 41MB; mini 80MB
            "voice": "Jasper",
        },
        "neutts": {
            "ref_audio": "",  # Path to reference voice audio (empty = bundled default)
            "ref_text": "",   # Path to reference voice transcript (empty = bundled default)
            "model": "neuphonic/neutts-air-q4-gguf",  # HuggingFace model repo
            "device": "cpu",  # cpu, cuda, or mps
        },
        "piper": {
            # Voice name (e.g. "en_US-lessac-medium") downloaded on first
            # use, OR an absolute path to a pre-downloaded .onnx file.
            # Full voice list: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md
            "voice": "en_US-lessac-medium",
            # "voices_dir": "",        # Override voice cache dir; default = ~/.hermes/cache/piper-voices/
            # "use_cuda": False,       # Requires onnxruntime-gpu
            # "length_scale": 1.0,     # 2.0 = twice as slow
            # "noise_scale": 0.667,
            # "noise_w_scale": 0.8,
            # "volume": 1.0,
            # "normalize_audio": True,
        },
        "deepinfra": {
            "model": "",  # empty = first tts-tagged model from the live catalog
            "voice": "default",
            # "base_url": "",  # override DEEPINFRA_BASE_URL for TTS only
        },
    },

    "stt": {
        "enabled": True,
        # When true, gateway voice messages are transcribed for the agent and
        # the raw transcript is also echoed back to the user as a 🎙️ message.
        # Set false to keep STT for the agent while suppressing that user-facing echo.
        "echo_transcripts": True,
        # NOTE: no seeded "provider" key. Strict selection semantics treat a
        # stored stt.provider as an explicit user pick; seeding "local" here
        # made a fresh install indistinguishable from a user choice. The
        # autodetect ladder covers unset. Valid values when set:
        # "local" (free, faster-whisper) | "groq" | "openai" (Whisper API) | "mistral" (Voxtral Transcribe) | "elevenlabs" (Scribe) | "deepinfra"
        # Global language hint applied to EVERY provider unless a per-provider
        # language overrides it. Defaults to "en" — Whisper auto-detection
        # frequently misidentifies short/accented clips, which reads as
        # "STT transcribed the wrong language". Set to "" to restore
        # auto-detect, or to your language code ("es", "zh", "uk", ...).
        "language": "en",
        # Pre-upload silence trim for cloud providers (groq/openai/mistral/
        # xai/elevenlabs/deepinfra). Local whisper gets Silero VAD; cloud
        # endpoints otherwise receive raw audio — silence inflates upload
        # time, per-audio-minute billing, and hallucination risk. Collapses
        # pauses with ffmpeg client-side; any failure uploads the original.
        "cloud_trim_silence": True,
        "cloud_trim_threshold_db": -40,  # audio quieter than this counts as silence
        "cloud_trim_keep_ms": 300,  # how much of each pause survives (keeps natural pacing)
        "local": {
            "model": "base",  # tiny, base, small, medium, large-v3
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
            "initial_prompt": "",
            # Anti-hallucination hardening (faster-whisper decodes junk tokens
            # from silence/noise without these):
            "vad": True,  # Silero VAD filter — silence never reaches whisper. false = old raw behavior (music/ambient).
            "vad_min_silence_ms": 500,  # min silence (ms) that splits speech chunks when vad is on
            "no_speech_prob_threshold": 0.6,  # drop a segment only if no_speech_prob is ABOVE this...
            "logprob_threshold": -1.0,  # ...AND its avg_logprob is BELOW this (both must hit)
            "unload_after_idle_seconds": 0,  # 0=never (default); e.g. 300 releases the model after 5min idle
        },
        "groq": {
            "model": "whisper-large-v3-turbo",  # whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
        "openai": {
            "model": "whisper-1",  # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
        "mistral": {
            "model": "voxtral-mini-latest",  # voxtral-mini-latest, voxtral-mini-2602
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
        "xai": {
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
        "elevenlabs": {
            "model_id": "scribe_v2",  # scribe_v2, scribe_v1
            "language_code": "",  # auto-detect by default; set to "eng", "spa", "fra", etc. to force
            "tag_audio_events": False,
            "diarize": False,
        },
        "deepinfra": {
            "model": "",  # empty = first stt-tagged model from the live catalog
            # "base_url": "",  # override DEEPINFRA_BASE_URL for STT only
        },
    },

    "voice": {
        "record_key": "ctrl+b",
        "submit_mode": "direct",       # TUI: direct submits immediately; draft leaves an editable transcript
        "max_recording_seconds": 120,
        "auto_tts": False,
        "beep_enabled": True,         # Play record start/stop beeps in CLI voice mode
        "beep_volume": 0.3,           # Beep amplitude multiplier (0.0-1.0, default keeps prior hardcoded value)
        "thinking_sound": True,       # Calm ambient bubble sound while the agent works in voice chat (volume follows beep_volume)
        "silence_threshold": 200,     # RMS below this = silence (0-32767)
        "silence_duration": 3.0,      # Seconds of silence before auto-stop
        "barge_in": True,             # Interrupt the agent / stop TTS when the user starts talking
        "barge_in_grace_seconds": 0.5,  # Trip suppression right after TTS playback starts (onset transient); the mic itself is live for the whole turn
        "barge_in_threshold_multiplier": 3.0,  # Speech trigger = quiet-room floor x this (floor is calibrated BEFORE playback, never against speaker bleed)
        # Saying EXACTLY one of these phrases (and nothing else) ends the
        # voice chat instead of being sent to the agent. Case-insensitive,
        # surrounding punctuation ignored. Set [] to disable.
        "stop_phrases": ["stop"],
    },

    # "Hey Hermes" hands-free wake word. Always-on, on-device hotword
    # detection that starts a fresh voice session — the "Hey Siri" pattern.
    # Off by default; toggle with /wake or `wake_word.enabled: true`.
    "wake_word": {
        "enabled": False,
        "surface": "auto",            # eligible surface: "auto" (first claimant) | "cli" | "tui" | "gui"
        "input_device": None,          # PortAudio input device index/name; null uses the process default
        "capture": "auto",            # auto | local | client — where PCM is captured (client = desktop streams mic via wake.feed)
        "provider": "openwakeword",   # "openwakeword" (free, local) | "sherpa" (free, ANY phrase, no training) | "porcupine" (premium; needs PORCUPINE_ACCESS_KEY)
        "phrase": "hey hermes",       # for "sherpa" this IS the detected phrase (any text works); for other engines it's a cosmetic label — detection is keyed by the model/keyword below
        "sensitivity": 0.6,           # 0.0-1.0 detection threshold, consistent across engines (higher = stricter, fewer false triggers)
        "confirmation_frames": 3,     # openWakeWord only: consecutive over-threshold frames required to fire (higher = fewer false triggers on ambient speech, slightly more latency; 1 = old single-frame behavior)
        "start_new_session": True,    # start a fresh session on wake vs. continue the current one
        "profile_routing": True,      # sherpa only: also listen for every wake-enabled profile's phrase and route the wake to the matching profile
        "openwakeword": {
            # "hey_hermes" (the bundled, works-out-of-the-box default) OR a
            # built-in openWakeWord name ("hey_jarvis", "alexa", "hey_mycroft",
            # ...) OR a path to a custom .onnx/.tflite model for another phrase.
            # See the wake-word docs for the custom-model training guide.
            "model": "hey_hermes",
            # "" (auto — tflite on macOS ARM64, onnx elsewhere) | "onnx" | "tflite".
            # openWakeWord's onnx backend scores near-zero on macOS ARM64
            # (dscripka/openWakeWord#336), so auto avoids a listener that arms
            # but never fires. Set explicitly only to override that choice.
            "inference_framework": "",
        },
        "sherpa": {
            # Optional path to a sherpa-onnx KWS model directory. Empty =
            # auto-download the small English zipformer model on first use.
            "model_dir": "",
        },
        "porcupine": {
            # Built-in keyword ("jarvis", "computer", "bumblebee", ...) or a path
            # to a custom .ppn from the Picovoice Console.
            "keyword": "jarvis",
        },
    },
    
    "human_delay": {
        "mode": "off",
        "min_ms": 800,
        "max_ms": 2500,
    },
    
    # Context engine -- controls how the context window is managed when
    # approaching the model's token limit.
    # "compressor" = built-in lossy summarization (default).
    # Set to a plugin name to activate an alternative engine (e.g. "lcm"
    # for Lossless Context Management).  The engine must be installed as
    # a plugin in plugins/context_engine/<name>/ or ~/.hermes/plugins/.
    "context": {
        "engine": "compressor",
        # Return freed glibc allocator pages after long-running agent/TUI
        # cleanup boundaries. Unsupported platforms are safe no-ops.
        "memory_trim": {
            "enabled": True,
            "cooldown_seconds": 60.0,
            # Successful trim calls are INFO logged every Nth periodic call;
            # force paths always log so process-close behavior is visible.
            "log_every_n": 1,
            # Suppress INFO logs only when a readable RSS change is smaller.
            # 0 reports every successful configured trim.
            "info_log_min_delta_mb": 0.0,
        },
    },

    # Persistent memory -- bounded curated memory injected into system prompt
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        # Approval gate for memory writes (add/replace/remove), applied to BOTH
        # foreground agent turns and the background self-improvement review fork
        # (the source of unprompted "wrong assumption" saves users reported).
        #   false (default) — write freely; the gate is off (pre-gate behaviour)
        #   true            — require approval: foreground writes prompt inline
        #                     (entries are small enough to review in a chat
        #                     bubble); background-review writes are staged
        #                     instead of committed (a daemon thread cannot block
        #                     on a prompt). Review staged entries with
        #                     /memory pending, /memory approve <id>,
        #                     /memory reject <id>.
        # To disable memory entirely, use memory_enabled: false instead.
        "write_approval": False,
        "memory_char_limit": 2200,   # ~800 tokens at 2.75 chars/token
        "user_char_limit": 1375,     # ~500 tokens at 2.75 chars/token
        # Periodic built-in memory review. External providers with automatic
        # turn/session extraction can set this to 0 and keep the small local
        # store reserved for explicit high-frequency operational facts.
        "nudge_interval": 10,
        # External memory provider plugin (empty = built-in only).
        # Set to a provider name to activate: "openviking", "mem0",
        # "hindsight", "holographic", "retaindb", "byterover".
        # Only ONE external provider is allowed at a time.
        "provider": "",
    },

    # Subagent delegation — override the provider:model used by delegate_task
    # so child agents can run on a different (cheaper/faster) provider and model.
    # Uses the same runtime provider resolution as CLI/gateway startup, so all
    # configured providers (OpenRouter, Nous, Z.ai, Kimi, etc.) are supported.
    "delegation": {
        "model": "",       # e.g. "google/gemini-3-flash-preview" (empty = inherit parent model)
        "provider": "",    # e.g. "openrouter" (empty = inherit parent provider + credentials)
        "base_url": "",    # direct OpenAI-compatible endpoint for subagents
        "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        "api_mode": "",    # wire protocol for delegation.base_url: "chat_completions",
                           # "codex_responses", or "anthropic_messages". Empty = auto-detect
                           # from URL (e.g. /anthropic suffix → anthropic_messages). Set this
                           # explicitly for non-standard endpoints the heuristic can't detect.
        # When delegate_task narrows child toolsets explicitly, preserve any
        # MCP toolsets the parent already has enabled. On by default so
        # narrowing (e.g. toolsets=["web","browser"]) expresses "I want these
        # extras" without silently stripping MCP tools the parent already has.
        # Set to false for strict intersection.
        "inherit_mcp_toolsets": True,
        "max_iterations": 250,  # per-subagent iteration cap (each subagent gets its own budget,
                               # independent of the parent's max_iterations)
        # Subagent summaries return to the parent's context verbatim. A batch
        # fan-out (N children) returns N summaries at once, which can exceed
        # the parent's context window and trigger a compression/429 death
        # spiral. delegate_task sizes each summary against the parent's
        # remaining context headroom (split across the batch); when it must
        # trim, the full text is spilled to ~/.hermes/cache/delegation/
        # (mounted into remote backends) and the in-context summary becomes a
        # head+tail window plus a footer with the exact read_file offset to
        # page the omitted middle — the same convention web_extract uses for
        # large pages. Nothing is lost. max_summary_chars is a hard per-summary
        # character ceiling layered on top of that dynamic budget
        # (belt-and-suspenders for models that ignore the "be concise"
        # instruction). 0 disables the hard ceiling; the dynamic headroom
        # budget still applies.
        "max_summary_chars": 24000,

        "child_timeout_seconds": 0,  # optional wall-clock cap per child agent. 0 (default)
                                     # = no timeout: children fail only from real errors
                                     # (API, tools, iteration budget), never a delegation
                                     # stopwatch. Set a positive number of seconds
                                     # (floor 30s) to enforce a hard cap.
        "reasoning_effort": "",  # subagent effort: "ultra", "max", "xhigh", "high",
                                 # "medium", "low", "minimal", "none" (empty = inherit)
        "max_concurrent_children": 10,  # unified concurrency cap: max parallel children per batch
                                       # AND max concurrent background (background=true)
                                       # delegation units. New async dispatches beyond the cap
                                       # fall back to synchronous execution. Floor of 1, no ceiling.
                                       # (Replaces the deprecated max_async_children.)
        # Orchestrator role controls (see tools/delegate_tool.py:_get_max_spawn_depth
        # and _get_orchestrator_enabled).  Floored at 1, no upper ceiling —
        # raise deliberately, each level multiplies API cost.
        "max_spawn_depth": 1,        # depth (1 = flat [default], 2 = orchestrator→leaf, 3+ = deeper)
        "orchestrator_enabled": True,  # kill switch for role="orchestrator"
        # When a subagent hits a dangerous-command approval prompt, the parent's
        # prompt_toolkit TUI owns stdin — a thread-local input() call from the
        # subagent worker would deadlock the parent UI. To avoid the deadlock,
        # subagent threads ALWAYS resolve approvals non-interactively:
        #   false (default) → auto-deny with a logger.warning audit line (safe)
        #   true             → auto-approve "once" with a logger.warning audit line
        # Flip to true only if you trust delegated work to run dangerous cmds
        # without human review (cron pipelines, batch automation, etc.).
        "subagent_auto_approve": False,
    },

    # Ephemeral prefill messages file — JSON list of {role, content} dicts
    # injected at the start of every API call for few-shot priming.
    # Never saved to sessions, logs, or trajectories.
    "prefill_messages_file": "",

    # Goals — persistent cross-turn goals (Ralph-style loop).
    # After every turn, a lightweight judge call asks the auxiliary model
    # whether the active /goal is satisfied by the assistant's last
    # response. If not, Hermes feeds a continuation prompt back into the
    # same session and keeps working until the goal is done, the turn
    # budget is exhausted, or the user pauses/clears it. Judge failures
    # fail OPEN (continue) so a flaky judge never wedges progress — the
    # turn budget is the real backstop.
    "goals": {
        # Max continuation turns before Hermes auto-pauses the goal and
        # asks the user to /goal resume. Protects against judge false
        # negatives (goal actually done but judge says continue) and
        # unbounded model spend on fuzzy / unachievable goals.
        "max_turns": 20,
    },


    # Loops — /loop recurring in-session wakeups (Claude Code parity).
    # A loop re-runs a prompt (or slash command) on a cadence inside the
    # live session. Fixed-interval mode fires on the user's clock;
    # self-paced mode (no interval given) starts at the floor and backs
    # off exponentially while the agent's replies stop changing.
    "loops": {
        # Smallest fixed interval accepted (seconds). Tighter cadences are
        # raised to this floor — each tick is a full agent turn.
        "min_interval_seconds": 30,
        # Backstop tick budget: the loop auto-pauses after this many
        # wakeups unless the user set --times. 0 = unlimited.
        "max_ticks": 100,
        # Self-paced cadence bounds (seconds).
        "self_paced_floor_seconds": 60,
        "self_paced_ceiling_seconds": 900,
    },

    # Mixture of Agents — named presets used by /moa. A preset is an execution
    # mode around the main model, not a provider/model itself: references +
    # aggregator synthesize private guidance before each main-model iteration.
    "moa": {
        "default_preset": "default",
        "active_preset": "",
        # When true, every MoA turn that runs the reference fan-out writes the
        # FULL turn (each reference's exact input messages + output + usage/cost,
        # and the aggregator's exact input + output) to a JSONL file at
        # <hermes_home>/moa-traces/<session_id>.jsonl. Off by default — turn it
        # on to audit / improve MoA behavior from real runs. Set trace_dir to
        # override the output directory.
        "save_traces": False,
        "trace_dir": "",
        # Privacy redaction filter for advisor (reference) outputs. Advisors
        # can echo PII from the conversation (emails, formatted phone numbers)
        # and credential shapes into reference blocks, traces, and the
        # aggregator prompt. Modes ('' = off, the default):
        #   "display" — redact user-visible surfaces only (reference blocks
        #               shown in the UI + saved MoA trace records); the
        #               aggregator still sees raw advisor text.
        #   "full"    — additionally redact the advisor text injected into
        #               the aggregator prompt (issue #59959).
        "privacy_filter": "",
        "presets": {
            "default": {
                "reference_models": [
                    {"provider": "openai-codex", "model": "gpt-5.5"},
                    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                ],
                "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                "max_tokens": 4096,
                "enabled": True,
            }
        },
    },

    # Skills — external skill directories for sharing skills across tools/agents.
    # Each path is expanded (~, ${VAR}) and resolved.  Read-only — skill creation
    # always goes to ~/.hermes/skills/.
    "skills": {
        "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
        # Project-local skill discovery: when a session starts inside a git
        # checkout, ``<root>/.hermes/skills/`` and ``<root>/.agents/skills/``
        # are sourced as the highest-precedence skill tier — but ONLY when the
        # project root is listed in trusted_project_dirs below. Trust a repo
        # with ``hermes skills trust`` (run from inside it). Set to false to
        # disable discovery entirely (no scan, no untrusted-skills notice).
        "project_discovery": True,
        # Absolute paths of project roots whose repo-local skills may load.
        # Managed by ``hermes skills trust`` / ``hermes skills untrust``.
        "trusted_project_dirs": [],
        # Substitute ${HERMES_SKILL_DIR} and ${HERMES_SESSION_ID} in SKILL.md
        # content with the absolute skill directory and the active session id
        # before the agent sees it.  Lets skill authors reference bundled
        # scripts without the agent having to join paths.
        "template_vars": True,
        # Pre-execute inline shell snippets written as !`cmd` in SKILL.md
        # body.  Their stdout is inlined into the skill message before the
        # agent reads it, so skills can inject dynamic context (dates, git
        # state, detected tool versions, …).  Off by default because any
        # content from the skill author runs on the host without approval;
        # only enable for skill sources you trust.
        "inline_shell": False,
        # Timeout (seconds) for each !`cmd` snippet when inline_shell is on.
        "inline_shell_timeout": 10,
        # Run the keyword/pattern security scanner on skills the agent
        # writes via skill_manage (create/edit/patch).  Off by default
        # because the agent can already execute the same code paths via
        # terminal() with no gate, so the scan adds friction (blocks
        # skills that mention risky keywords in prose) without meaningful
        # security.  Turn on if you want the belt-and-suspenders — a
        # dangerous verdict will then surface as a tool error to the
        # agent, which can retry with the flagged content removed.
        # External hub installs (trusted/community sources) are always
        # scanned regardless of this setting.
        "guard_agent_created": False,
        # Advisory NVIDIA SkillEvaluator Tier 1 scan on hub installs
        # (`hermes skills install`). Runs ALONGSIDE the built-in skills
        # guard (which stays the enforcement layer) and only when the
        # optional `skillevaluator` binary is on PATH:
        #   uv tool install --python 3.13 \
        #     "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.1.0"
        # Findings are informational — shown with file/line before the
        # install confirmation, never blocking. Secrets-class findings
        # (private keys, tokens, credentialed connection strings) are
        # highlighted in red. On by default because it is a no-op
        # without the binary installed.
        "tier1_advisory": True,
        # Approval gate for skill_manage (create/edit/patch/write_file/delete/
        # remove_file), applied to BOTH foreground agent turns and the
        # background self-improvement review fork.
        #   false (default) — write freely; the gate is off (pre-gate behaviour)
        #   true            — require approval: stage the write for review
        #                     instead of committing (a SKILL.md is too large to
        #                     review inline, so skills always stage rather than
        #                     prompt). List with /skills pending, inspect with
        #                     /skills diff <id> (full diff — CLI/dashboard/file,
        #                     never crammed into a chat bubble), apply with
        #                     /skills approve <id> or drop with /skills reject <id>.
        "write_approval": False,
        # Per-mutation audit ledger (tracker #79686 P3). Every skill mutation
        # — curator, agent, or user — appends one JSONL entry to
        # ~/.hermes/skills/.curator_ledger.jsonl with before/after file
        # hashes; file contents are stored content-addressed (deduped) under
        # ~/.hermes/.curator_backups/blobs/. Enables `hermes curator ledger`
        # and single-mutation `hermes curator rollback <entry-id>`.
        # Telemetry, never a gate: ledger failures cannot block a mutation.
        "ledger": True,
    },

    # Curator — background skill maintenance.
    #
    # Periodically reviews AGENT-CREATED skills (never bundled or
    # hub-installed) and keeps the collection tidy: marks long-unused skills
    # as stale, archives genuinely obsolete ones (archive only, never
    # deletes), and spawns a forked aux-model agent to consolidate overlaps
    # and patch drift. Runs inactivity-triggered from session start — no
    # cron daemon.
    #
    # See `hermes curator status` for the last run summary.
    "curator": {
        "enabled": True,
        # How long to wait between curator runs (hours).  Default: 7 days.
        "interval_hours": 24 * 7,
        # Only run when the agent has been idle at least this long (hours).
        "min_idle_hours": 2,
        # Mark a skill as "stale" after this many days without use.
        "stale_after_days": 30,
        # Archive a skill (move to skills/.archive/) after this many days
        # without use. Archived skills are recoverable — no auto-deletion.
        "archive_after_days": 90,
        # Run the LLM consolidation (umbrella-building) pass. OFF by default.
        # When off, a curator run does ONLY the deterministic inactivity prune
        # (mark stale / archive long-unused skills) and skips the forked
        # aux-model review entirely — no umbrella-building, no aux-model cost.
        # Set to true to opt back into merging overlapping skills into
        # class-level umbrellas. `hermes curator run --consolidate` overrides
        # this for a single invocation.
        "consolidate": False,
        # Also prune (archive) bundled built-in skills after the inactivity
        # period, not just agent-created ones. ON by default. Built-ins are
        # normally restored on every `hermes update`, so pruning them only
        # sticks because a suppression list tells the re-seeder to leave them
        # archived. Hub-installed skills are NEVER pruned here — they have an
        # external upstream owner. Built-ins accrue usage telemetry and their
        # inactivity clock starts the first time the curator sees them, so a
        # long-unused built-in is archived only after archive_after_days of
        # genuine non-use (never a mass-prune on the first run). Set to false
        # to keep all bundled built-ins permanently.
        "prune_builtins": True,
        # TTL purge of skills/.archive/. 0 (default) = never purge — archived
        # skills are kept forever. When > 0, `hermes curator purge` deletes
        # archived skills older than this many days (explicit command only,
        # never automatic; every purge is recorded in the audit ledger).
        "archive_ttl_days": 0,
        # Pre-run backup: before every real curator pass (dry-run is
        # skipped), snapshot ~/.hermes/skills/ into
        # ~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz so the
        # user can roll back with `hermes curator rollback`.
        "backup": {
            "enabled": True,
            "keep": 5,  # retain last N regular snapshots
        },
    },

    # Honcho AI-native memory -- reads ~/.honcho/config.json as single source of truth.
    # This section is only needed for hermes-specific overrides; everything else
    # (apiKey, workspace, peerName, sessions, enabled) comes from the global config.
    "honcho": {},

    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York").
    # Empty string means use server-local time.
    "timezone": "",

    # Slack platform settings (gateway mode)
    "slack": {
        "require_mention": True,       # Require @mention to respond in channels
        "free_response_channels": "",  # Comma-separated channel IDs where bot responds without mention
        "allowed_channels": "",        # If set, bot ONLY responds in these channel IDs (whitelist)
        # Channel IDs where @mention is ALWAYS required, even when
        # require_mention is false globally (per-channel force-mention override).
        "require_mention_channels": "",
        # Ignore a channel/thread message addressed to another user (first token
        # @mentions someone other than the bot) unless the bot is also mentioned.
        # Opt-in; default off keeps existing behaviour. Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
        "ignore_other_user_mentions": False,
        # If True, require @mention in Slack thread replies too.
        "thread_require_mention": False,
        "channel_prompts": {},         # Per-channel ephemeral system prompts
    },

    # Discord platform settings (gateway mode)
    "discord": {
        "require_mention": True,       # Require @mention to respond in server channels
        "free_response_channels": "",  # Comma-separated channel IDs where bot responds without mention
        "allowed_channels": "",        # If set, bot ONLY responds in these channel IDs (whitelist)
        "auto_thread": True,           # Auto-create threads on @mention in channels (like Slack)
        "thread_require_mention": False,  # If True, require @mention in threads too (multi-bot threads)
        "bots_require_inline_mention": False,  # Multi-bot rooms: if True, another bot must type @thisbot in its message to trigger a reply; a Discord reply/quote alone won't. Prevents two bots auto-replying to each other forever. Does not affect humans.
        "history_backfill": True,         # If True, prepend recent channel scrollback when bot is triggered (recovers messages missed while require_mention gated them out)
        "history_backfill_limit": 50,     # Max number of recent messages to scan when assembling the backfill block
        "missed_message_backfill": {
            "enabled": False,             # Replay missed Discord messages after reconnect/startup
            "channels": "",               # Comma-separated channel IDs; empty uses free_response_channels
            "window_seconds": 21600,      # Only inspect messages from the last 6 hours
            "limit": 100,                 # Global cap on messages scanned per reconnect
            "max_dispatches": 10,         # Cap on recovered messages dispatched per reconnect
        },
        "reactions": True,             # Add 👀/✅/❌ reactions to messages during processing
        # Discord Gateway transport health. These settings inspect the active
        # WebSocket's ready/open/heartbeat state; they never use Discord REST as
        # proof that Gateway events are still arriving. Set any value to 0 to
        # disable this compatibility-safe probe during a rollback.
        "websocket_liveness_interval_seconds": 15,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 60,
        "websocket_max_latency_seconds": 30,
        "channel_prompts": {},         # Per-channel ephemeral system prompts (forum parents apply to child threads)
        # Opt-in DM role-based auth (#12136). By default, DISCORD_ALLOWED_ROLES
        # authorizes only guild messages in the role's own guild — DMs require
        # DISCORD_ALLOWED_USERS. Set dm_role_auth_guild to a guild ID to also
        # authorize DMs from members of that one trusted guild holding the
        # allowed role. Unset / empty / 0 = secure default (DM role-auth off).
        "dm_role_auth_guild": "",
        # discord / discord_admin tools: restrict which actions the agent may call.
        # Default (empty) = all actions allowed (subject to bot privileged intents).
        # Accepts comma-separated string ("list_guilds,list_channels,fetch_messages")
        # or YAML list. Unknown names are dropped with a warning at load time.
        # Actions: list_guilds, server_info, list_channels, channel_info,
        # list_roles, member_info, search_members, fetch_messages, list_pins,
        # pin_message, unpin_message, create_thread, add_role, remove_role.
        "server_actions": "",
        # DEPRECATED / no-op. Any uploaded file is now always cached and
        # surfaced to the agent regardless of file type — authorization to
        # message the agent is the gate, not the extension. Kept so existing
        # configs that set it do not error. Env override:
        # DISCORD_ALLOW_ANY_ATTACHMENT.
        "allow_any_attachment": False,
        # Maximum bytes per attachment the gateway will cache. The whole file
        # is held in memory while being written, so unlimited uploads carry a
        # real memory cost. Default 32 MiB matches the historical hardcoded
        # cap. Set to 0 for no cap. Env override: DISCORD_MAX_ATTACHMENT_BYTES.
        "max_attachment_bytes": 33554432,
        # When True, Discord approval prompts mention numeric allowed users so
        # owners notice approval requests in shared channels/threads. Env
        # override: DISCORD_APPROVAL_MENTIONS. Default false avoids surprise
        # pings.
        "approval_mentions": False,
        # Discord voice-channel inactivity timeout, in seconds. Set to 0 to
        # keep the bot in VC until an explicit `/voice leave` / disconnect.
        "voice_channel_inactivity_timeout_seconds": 300,
        # Minimum seconds to wait for a VC playback before force-stopping it.
        # The adapter also probes clip duration and extends this floor by a
        # padding window, so long TTS readbacks are not cut at exactly 120s.
        "voice_playback_timeout_seconds": 120,
        # Voice-channel audio effects (the continuous mixer). OFF by default.
        # When enabled, the bot installs a software mixer on the outgoing voice
        # stream so a low ambient "thinking" bed, verbal acknowledgements, and
        # TTS replies can OVERLAP (ducking the ambient under speech) instead of
        # stop-and-swap — the Grok-voice-mode feel. discord.py ships no mixer;
        # this is implemented in plugins/platforms/discord/voice_mixer.py.
        "voice_fx": {
            "enabled": False,         # master switch for the mixer subsystem
            "ambient_enabled": True,  # play the idle "thinking" bed while tools run
            "ambient_path": "",       # custom loop audio file; "" = synthesised pad
            "ambient_gain": 0.18,     # idle bed loudness, 0.0–1.0
            "duck_gain": 0.06,        # ambient loudness while speech plays
            "speech_gain": 1.0,       # TTS / ack loudness, 0.0–1.0
            "ack_enabled": True,      # speak a short phrase before the first tool call
            "ack_phrases": [          # picked at random; set [] to disable phrases
                "Let me look into that.",
                "One moment.",
                "Checking on that now.",
                "Give me a sec.",
                "On it.",
            ],
        },
    },

    # WhatsApp platform settings (gateway mode)
    "whatsapp": {
        # Reply prefix prepended to every outgoing WhatsApp message.
        # Default (None) uses the built-in "⚕ *Hermes Agent*" header.
        # Set to "" (empty string) to disable the header entirely.
        # Supports \n for newlines, e.g. "🤖 *My Bot*\n──────\n"
    },

    # Telegram platform settings (gateway mode)
    "telegram": {
        "reactions": False,            # Add 👀/✅/❌ reactions to messages during processing
        "channel_prompts": {},         # Per-chat/topic ephemeral system prompts (topics inherit from parent group)
        "allowed_chats": "",           # If set, bot ONLY responds in these group/supergroup chat IDs (whitelist)
        "extra": {
            "rich_messages": False,     # Bot API 10.1 rich messages (tables/task lists/details/math) render natively; set True to opt in. Default stays legacy MarkdownV2 because rich messages can be hard to copy as plain text in Telegram clients.
            "rich_drafts": False,       # Experimental Bot API 10.1 rich draft previews during Telegram DM streaming. Default off because Telegram Desktop/macOS can visually overlay rich draft frames until the chat redraws.
        },
    },

    # Mattermost platform settings (gateway mode)
    "mattermost": {
        "require_mention": True,       # Require @mention to respond in channels
        "free_response_channels": "",  # Comma-separated channel IDs where bot responds without mention
        "allowed_channels": "",        # If set, bot ONLY responds in these channel IDs (whitelist)
        "channel_prompts": {},         # Per-channel ephemeral system prompts
    },

    # Matrix platform settings (gateway mode)
    "matrix": {
        "require_mention": True,       # Require @mention to respond in rooms
        "free_response_rooms": "",     # Comma-separated room IDs where bot responds without mention
        "allowed_rooms": "",           # If set, bot ONLY responds in these room IDs (whitelist)
    },

    # Approval mode for dangerous commands:
    #   manual — always prompt the user
    #   smart  — use auxiliary LLM to auto-approve low-risk commands (default)
    #   off    — skip all approval prompts (equivalent to --yolo)
    #
    # cron_mode — what to do when a cron job hits a dangerous command:
    #   deny    — block the command and let the agent find another way (default, safe)
    #   approve — auto-approve all dangerous commands in cron jobs
    #
    # single_query_mode — what to do when a single-query (-q) session hits a
    # dangerous command. -q runs export HERMES_INTERACTIVE=1 (for interactive
    # sudo prompts) but have NO user waiting to answer approval prompts — an
    # unanswered prompt just waits the full timeout then fails closed, so the
    # agent is forced to work around the block (often via execute_code). This
    # setting makes that intent explicit:
    #   deny    — block the command and let the agent find another way (default,
    #             safe; mirrors cron_mode deny)
    #   approve — auto-approve all dangerous commands in single-query mode
    #
    # timeout — seconds to wait for the user's approve/deny before failing
    # closed (deny). Shared by the CLI prompt and gateway/messaging waits.
    # Messaging approvals arrive as a push notification the user may not see
    # immediately — 60s proved too tight on Telegram/Discord (the prompt
    # expired before the user reached their phone), so the default is 300.
    "approvals": {
        "mode": "smart",
        "timeout": 300,
        "cron_mode": "deny",
        "single_query_mode": "deny",
        # Operator-customizable policy text for smart approvals. When
        # non-empty, this is appended to the smart-approval guardian's
        # SYSTEM prompt (trusted channel) as additional rules — e.g.
        # "Always ESCALATE commands touching /etc" or "APPROVE docker
        # compose restarts under ~/deploys". Inspired by ChatGPT Work's
        # customizable auto-review guardian policy.
        "smart_policy": "",
        # Consecutive-denial circuit breaker for smart approvals: after this
        # many guardian DENY verdicts in a row within one session, the deny
        # message returned to the model escalates to a hard-stop instruction
        # (report to the user / ask for manual run or /approve) instead of a
        # plain "Do NOT retry". Any approval resets the count. 0 disables.
        # Inspired by ChatGPT Work's auto-review circuit breaker.
        "denial_breaker_threshold": 3,
        # User-defined deny rules: fnmatch globs matched against terminal
        # commands. A match blocks the command unconditionally — BEFORE the
        # --yolo / /yolo / mode=off bypass — making this the user-editable
        # counterpart to the code-shipped hardline blocklist. Patterns are
        # case-insensitive and must be quoted in YAML when they start with
        # * or contain {}/!/: sequences. Example:
        #   deny:
        #     - "git push --force*"
        #     - "*curl*|*sh*"
        "deny": [],
        # When true, /reload-mcp asks the user to confirm before rebuilding
        # the MCP tool set for the active session.  Reloading invalidates
        # the provider prompt cache (tool schemas are baked into the system
        # prompt), so the next message re-sends full input tokens — this can
        # be expensive on long-context or high-reasoning models.  Users click
        # "Always Approve" to silence the prompt permanently; that flips
        # this key to false.
        "mcp_reload_confirm": True,
        # When true, destructive session slash commands (/clear, /new, /reset,
        # /undo) ask the user to confirm before discarding conversation state.
        # Three-option prompt (Approve Once / Always Approve / Cancel) routed
        # through tools.slash_confirm — native yes/no buttons on Telegram,
        # Discord, and Slack; text fallback elsewhere.  Users click "Always
        # Approve" to silence the prompt permanently; that flips this key to
        # false.  TUI also honors this setting for its /clear, /new, and /reset
        # modal; HERMES_TUI_NO_CONFIRM=1 force-skips that modal regardless of
        # the configured value.
        "destructive_slash_confirm": True,
    },

    # Permanently allowed dangerous command patterns (added via "always" approval)
    "command_allowlist": [],
    # User-defined quick commands that bypass the agent loop (type: exec only)
    "quick_commands": {},

    # Per-platform system-prompt hint overrides. Lets an admin append to or
    # replace Hermes' built-in platform hint for a single messaging platform
    # (WhatsApp, Slack, Telegram, ...) without affecting other platforms.
    # Useful for enterprise/managed profiles that ship platform-aware skills.
    # Each key is a platform name; the value is either:
    #   { "append": "extra text" }   — keep the default hint, append text
    #   { "replace": "full text" }   — substitute the default hint entirely
    #   "extra text"                 — shorthand for { "append": ... }
    # `replace` wins over `append` if both are given. Example:
    #   platform_hints:
    #     whatsapp:
    #       append: >
    #         When tabular output would be useful, invoke the
    #         table_formatting skill instead of emitting a Markdown table.
    "platform_hints": {},

    # Shell-script hooks — declarative bridge that invokes shell scripts
    # on plugin-hook events (pre_tool_call, post_tool_call, pre_llm_call,
    # subagent_stop, etc.).  Each entry maps an event name to a list of
    # {matcher, command, timeout} dicts.  First registration of a new
    # command prompts the user for consent; subsequent runs reuse the
    # stored approval from ~/.hermes/shell-hooks-allowlist.json.
    # See `website/docs/user-guide/features/hooks.md` for schema + examples.
    "hooks": {},

    # Auto-accept shell-hook registrations without a TTY prompt.  Also
    # toggleable per-invocation via --accept-hooks or HERMES_ACCEPT_HOOKS=1.
    # Gateway / cron / non-interactive runs need this (or one of the other
    # channels) to pick up newly-added hooks.
    "hooks_auto_accept": False,
    # Custom personalities — add your own entries here
    # Supports string format: {"name": "system prompt"}
    # Or dict format: {"name": {"description": "...", "system_prompt": "...", "tone": "...", "style": "..."}}
    "personalities": {},

    # Pre-exec security scanning via tirith
    "security": {
        "allow_private_urls": False,  # Allow requests to private/internal IPs (for OpenWrt, proxies, VPNs)
        "redact_secrets": True,
        # Persisted acknowledgement for unattended model overrides whose tier
        # lets the vendor train on prompts/completions. The startup guard still
        # prints the full warning on every run and never bypasses cost guards.
        "allow_data_training_tiers_noninteractive": False,
        # Human approval presentation transport. "builtin" preserves the
        # current CLI/TUI/gateway/ACP surfaces. A plugin transport is used only
        # when named explicitly here. Transport timeout/error/invalid response
        # denies unless transport_fallback is explicitly set to "builtin".
        # This is presentation only: plugins cannot detect, suppress, or
        # auto-approve commands outside a correlated human response.
        "approval": {
            "transport": "builtin",
            "transport_fallback": "deny",
        },
        # Writes to agent-instruction files (AGENTS.md/CLAUDE.md/SOUL.md/
        # .cursorrules, project-local .hermes config) always require human
        # approval — even under auto-approve/yolo. Extra patterns are
        # fnmatch globs matched against the basename (e.g. "*.mdc").
        "protected_instruction_files": True,
        "protected_instruction_extra_patterns": [],
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
        "website_blocklist": {
            "enabled": False,
            "domains": [],
            "shared_files": [],
        },
        # Acknowledged supply-chain security advisories. Each entry is the
        # ID of an advisory the user has read and acted on (uninstalled the
        # compromised package, rotated credentials). Acked advisories no
        # longer trigger the startup banner. Add via `hermes doctor --ack
        # <id>`; remove by editing the list directly. See
        # ``hermes_cli/security_advisories.py`` for the catalog.
        "acked_advisories": [],
        # Allow Hermes to lazy-install opt-in backend packages from PyPI
        # the first time the user enables a backend that needs them
        # (e.g. installing ``elevenlabs`` when the user picks ElevenLabs as
        # their TTS provider). Set to false to require explicit
        # ``pip install`` for everything beyond the base set — appropriate
        # for restricted networks, audited environments, or air-gapped
        # systems where any runtime install is unacceptable.
        "allow_lazy_installs": True,
    },

    "cron": {
        # Allow cron-spawned agents to use the cronjob toolset (create/edit/
        # remove scheduled jobs from within a cron run — the "cron-librarian"
        # pattern). Off by default: the cronjob toolset is policy-denied in
        # cron context to prevent unattended scheduling loops. Jobs created
        # this way are user-owned in the same flat jobs table as every other
        # job. Interactive toolsets (messaging/clarify) stay denied in cron
        # context regardless of this setting.
        "allow_agent_scheduling": False,
        # Pre-dispatch configuration validation (T1-26): before constructing
        # any agent machinery for a job, verify the provider API key resolves
        # (unless a fallback chain is configured), attached skills are ready
        # (required env/commands present), and delivery platforms are
        # configured. A failing job is recorded as last_status=blocked_config
        # with ONE alert (no re-alert every tick) and NO LLM call is made.
        # Set to false to restore the old behavior (fail during the run).
        "preflight": True,
        # Fail closed when an unpinned job's current global model/provider
        # differs from its creation-time snapshot. This prevents unattended
        # jobs from silently inheriting a paid default. Set to false only when
        # jobs should deliberately track changing global inference defaults.
        "model_drift_guard": True,
        # Default inference model for cron jobs (Axis A — WHAT model an
        # agent job runs on). Resolution at fire time: per-job user pin >
        # cron.model > global model.default. When set, unpinned jobs follow
        # this deliberately, so the #44585 model-drift fail-closed guard does
        # not engage for the model axis — cron spend no longer shadows chat
        # `/model` switches. Empty string = fall through to model.default.
        "model": "",
        # Inference provider paired with cron.model (NOT the scheduler
        # provider below). Empty string = resolve from global config.
        "model_provider": "",
        # Active cron SCHEDULER provider (Axis B — the trigger that decides
        # WHEN a due job fires). Empty string = the built-in in-process 60s
        # ticker (default). Name an installed provider (plugins/cron_providers/<name>/ or
        # $HERMES_HOME/plugins/<name>/) to relocate the trigger — e.g. "chronos",
        # the NAS-mediated managed-cron provider for scale-to-zero deployments.
        # An unknown or unavailable provider falls back to the built-in, so cron
        # never loses its trigger.
        "provider": "",
        # Chronos (NAS-mediated managed cron) settings. Only consulted when
        # provider == "chronos". All non-secret (URLs + the JWT audience): the
        # agent holds NO external-scheduler credentials. For hosted agents, NAS
        # sets these at provision time. The outbound provision call reuses the
        # agent's existing Nous Portal token — there is no token key here.
        "chronos": {
            # NAS / portal base URL the agent calls to arm/cancel one-shots
            # and that mints the inbound fire JWT (used as the expected issuer).
            "portal_url": "https://portal.nousresearch.com",
            # The agent's OWN publicly-reachable base URL for NAS→agent fires
            # (NAS POSTs {callback_url}/api/cron/fire). Empty → Chronos is
            # unavailable and the resolver falls back to the built-in ticker.
            "callback_url": "",
            # This agent's expected JWT audience (e.g. "agent:{instance_id}").
            "expected_audience": "",
            # NAS JWKS URL for verifying the inbound fire JWT's signature.
            # Empty → the fire endpoint refuses all tokens (no unsigned decode).
            "nas_jwks_url": "",
        },
        # Wrap delivered cron responses with a header (task name) and footer
        # ("The agent cannot see this message").  Set to false for clean output.
        "wrap_response": True,
        # Make cron deliveries CONTINUABLE: a user can reply to a cron brief
        # and the agent has it in context (no "what is Task #2?" amnesia).
        # Default False preserves the historical isolation guarantee (cron
        # deliveries live only in the cron job's own session). Per-job
        # `attach_to_session` overrides this for a single job.
        #
        # Behaviour is THREAD-PREFERRED, scoped to the job's origin chat:
        #   - Thread-capable platforms (Telegram forum/DM topics, Discord
        #     threads, Slack threads): a dedicated thread is opened for the job
        #     via the adapter's create_handoff_thread, the brief is delivered
        #     into it, and that thread's session is seeded so the user's reply
        #     in-thread continues with full context. Each continuable job gets
        #     its own scrollback, isolated from the parent channel.
        #   - DM-only platforms (WhatsApp / Signal / SMS): no threads exist, so
        #     the brief is mirrored into the origin DM session instead — the
        #     DM itself is the continuation surface.
        # Both paths ride the shipped gateway.mirror.mirror_to_session and are
        # alternation- and cache-safe (appended at a turn boundary, never
        # mid-loop, never mutating the cached system prompt). Only the origin
        # chat is ever touched — fan-out / broadcast targets are never mirrored.
        "mirror_delivery": False,
        # Maximum number of due jobs to run in parallel per tick.
        # null/0 = unbounded (limited only by thread count).
        # 1 = serial (pre-v0.9 behaviour).
        # Also overridable via HERMES_CRON_MAX_PARALLEL env var.
        "max_parallel_jobs": None,
        # Per-job output-file retention: save_job_output keeps the N most
        # recent .md files and prunes older ones. 0 or negative disables
        # pruning (for operators who manage cleanup externally). Default 50.
        "output_retention": 50,
        # Timeout (seconds) for a no-agent cron script. Also overridable via
        # HERMES_CRON_SCRIPT_TIMEOUT. Keep this in sync with
        # cron.scheduler._DEFAULT_SCRIPT_TIMEOUT so config set recognizes the
        # same setting the scheduler reads.
        "script_timeout_seconds": 3600,
        # Timeout (seconds) for SessionDB() init inside cron jobs.
        # SessionDB opens/migrates state.db synchronously and has no timeout
        # of its own against a wedged sqlite3.connect. An unbounded hang here
        # wedges the job's dispatch guard forever. Also overridable via
        # HERMES_CRON_SESSION_DB_TIMEOUT env var. 0 = unlimited (skip the bound).
        "session_db_timeout_seconds": 10,
        # Timeout (seconds) for each media attachment send during cron
        # delivery via a live gateway adapter. Large attachments (long TTS
        # audio, big exports) can exceed the old fixed 30s window. Also
        # overridable via HERMES_CRON_MEDIA_SEND_TIMEOUT env var. Keep in
        # sync with cron.scheduler._DEFAULT_MEDIA_SEND_TIMEOUT.
        "media_send_timeout_seconds": 300,
    },

    # Kanban multi-agent coordination — controls the dispatcher loop that
    # spawns workers for ready tasks. The dispatcher ticks every N seconds
    # (default 60), reclaims stale claims, promotes dependency-satisfied
    # todos to ready, and fires `hermes -p <assignee> chat -q ...` for
    # each claimable ready task. One dispatcher per profile is sufficient;
    # running more than one on the same kanban.db will race for claims.
    "kanban": {
        # Auto-subscribe the originating gateway/TUI session to task
        # completion + block events when ``kanban_create`` is called from
        # inside a session that has a persistent delivery channel. The
        # agent that dispatched the task will get notified automatically
        # instead of having to poll. Disable to mirror pre-feature
        # behaviour — e.g. for a profile that prefers explicit
        # ``kanban_notify-subscribe`` calls per task.
        "auto_subscribe_on_create": True,
        # Run the dispatcher inside the gateway process. On by default —
        # the cost is ~300µs every `dispatch_interval_seconds` when idle,
        # and gateway is the supervisor users already have. Set to false
        # only if you run the dispatcher as a separate systemd unit or
        # don't want the gateway to spawn workers.
        "dispatch_in_gateway": True,
        # Automatically claim tasks in the first-class review column and spawn
        # the assigned profile with the bundled sdlc-review skill. Disable for
        # boards where every review is performed manually from the dashboard.
        "review_dispatch": True,
        # Seconds between dispatcher ticks (idle or not). Lower = snappier
        # pickup of newly-ready tasks; higher = less SQL pressure.
        "dispatch_interval_seconds": 60,
        # Auto-block after this many consecutive non-success attempts for the
        # same task/profile (spawn_failed, timed_out, or crashed). Reassignment
        # resets the streak for the new profile.
        "failure_limit": 2,
        # Worker stdout/stderr logs rotate at spawn time. Defaults preserve
        # the historical 2 MiB + one-backup behavior; long-running workers can
        # raise these to keep more early failure evidence.
        "worker_log_rotate_bytes": 2 * 1024 * 1024,
        "worker_log_backup_count": 1,
        # Profile assigned to the root/orchestration task after Triage
        # decomposition. When unset, falls back to the default profile (the
        # one `hermes` launches with no -p flag). This does not control the
        # decomposer prompt, model, or skills; configure that LLM path under
        # auxiliary.kanban_decomposer.
        "orchestrator_profile": "",
        # Where a child task lands if the orchestrator can't match an
        # assignee to any installed profile. When unset, falls back to the
        # default profile. A task never ends up with assignee=None.
        "default_assignee": "",
        # Global concurrency cap (#33488): when set to a positive int, the
        # HOST never has more than N tasks in 'running' at once — counted
        # across every active board and across both the ready and review
        # dispatch lanes (workers are OS processes sharing one machine's
        # memory, so the cap bounds the machine, not each board; OOF-30).
        # Unset (None) means
        # "derive from system memory" (OOF-30/OOF-77): the dispatcher caps
        # concurrency at roughly MemTotal / 512 MiB, clamped to [2, 8] —
        # e.g. 2 workers on a 1 GiB VM. On hosts where total memory can't
        # be read (macOS/Windows), unset falls back to no cap. Set an
        # explicit value to override the derived default in either
        # direction.
        "max_in_progress": None,
        # Per-profile concurrency cap (#21582). When set to a positive int,
        # no single profile can have more than N workers running at once,
        # even if the global max_in_progress / max_spawn caps would allow
        # it. Tasks blocked this way defer to the next dispatcher tick.
        # Unset (None) means "no per-profile cap" — backward-compatible
        # with existing installs. Useful for fan-out workflows that would
        # otherwise saturate one profile's local model / API quota /
        # browser pool while leaving other profiles idle.
        "max_in_progress_per_profile": None,
        # When true, the kanban dispatcher auto-runs the decomposer on
        # tasks that land in Triage (every dispatcher tick). When false,
        # decomposition is manual via `hermes kanban decompose <id>` or
        # the dashboard's Decompose button.
        "auto_decompose": True,
        # Max triage tasks to decompose per dispatcher tick. Prevents a
        # large bulk-load of triage tasks from spending a burst of aux
        # LLM calls in one tick. Excess tasks defer to the next tick.
        "auto_decompose_per_tick": 3,
        # Stale detection: running tasks that have exceeded this many
        # seconds without a heartbeat (since ``last_heartbeat_at``) are
        # auto-reclaimed to ``ready`` on the next dispatcher tick. The
        # worker process (if still running host-locally) is terminated
        # before the reclaim.  0 disables stale detection entirely.
        "dispatch_stale_timeout_seconds": 14400,
        # Orphaned-card reconciliation: each dispatcher tick, requeue
        # 'running' cards whose claim bookkeeping is broken (claim_lock or
        # claim_expires NULL with a dead/gone worker) — zombies invisible
        # to the TTL/crash/stale recovery paths. Set false to keep orphans
        # frozen for manual forensics.
        "reconcile_orphans": True,
        # Notify subscriptions survive a task reaching ``done`` (completion
        # is reversible — controllers reopen done work for review
        # corrections), and are normally removed on archive. On boards that
        # never archive, the notifier GC purges subscriptions for tasks
        # that have been ``done`` with no new activity for this many days,
        # so stale rows don't accumulate and get scanned on every notifier
        # tick forever. Set 0 to disable the sweep.
        "done_sub_retention_days": 30,
    },

    # execute_code settings — controls the tool used for programmatic tool calls.
    "code_execution": {
        # Execution mode:
        #   project (default) — scripts run in the session's working directory
        #     with the active virtualenv/conda env's python, so project deps
        #     (pandas, torch, project packages) and relative paths resolve.
        #   strict            — scripts run in an isolated temp directory with
        #     hermes-agent's own python (sys.executable). Maximum isolation
        #     and reproducibility; project deps and relative paths won't work.
        # Env scrubbing (strips *_API_KEY, *_TOKEN, *_SECRET, ...) and the
        # tool whitelist apply identically in both modes.
        "mode": "project",
    },

    # Tool Search (progressive disclosure for large tool surfaces).
    # When the model is connected to many MCP servers or non-core plugin
    # tools, their JSON schemas can consume a substantial fraction of the
    # context window on every turn. When enabled, those tools are replaced
    # in the model-facing tools array with three bridge tools —
    # tool_search / tool_describe / tool_call — and surfaced on demand.
    #
    # Core Hermes tools (terminal, read_file, write_file, patch,
    # search_files, todo, memory, browser_*, etc.) are NEVER deferred.
    # See tools/tool_search.py for full design notes and the
    # openclaw-tool-search-report PDF in this PR for the rationale.
    "tools": {
        "tool_search": {
            # Tiered disclosure: any deferrable (MCP/plugin) tool activates
            # the bridge; the listing then scales with catalog size.
            #   Tier 0 — no MCP/plugin tools: everything stays eager.
            #   Tier 1 — catalog listing fits the budget: bridge + skills-style
            #     name+description manifest (degrades to names-only).
            #   Tier 2 — per-tool listing over budget even names-only (e.g.
            #     Cloudflare's ~3,300-tool flat API surface): bare bridge +
            #     a one-line-per-server summary (name + tool count) so the
            #     model knows which domains are reachable; individual tools
            #     discoverable through tool_search only.
            # "auto"/"on" — activate when at least one deferrable tool exists.
            # "off" — disable entirely. Tools-array assembly is a pass-through.
            "enabled": "auto",
            # Listing budget as a percentage of the active model's context
            # length. Effective budget = min(this % of context,
            # listing_max_tokens). Range 0..100.
            "threshold_pct": 5,
            # When the model calls tool_search without a ``limit`` argument,
            # how many hits to return. Range 1..max_search_limit.
            "search_default_limit": 5,
            # Hard upper bound the model can request via ``limit``. Range 1..50.
            "max_search_limit": 20,
            # Skills-style catalog listing embedded in the tool_search bridge
            # description: every deferred tool's name + first sentence of its
            # description (≤60 chars), grouped by MCP server / toolset. Keeps
            # capabilities discoverable while schemas stay deferred.
            # "auto" (default) — include when the listing fits the budget
            #   (falls back to names-only, then to the bare tier-2 bridge).
            # "on"  — same rendering, but explicit intent to always list.
            # "off" — always the bare bridge (tier 2 for every catalog).
            "listing": "auto",
            # Absolute cap on the embedded listing in tokens (chars/4
            # estimate), regardless of context size. Range 200..60000.
            "listing_max_tokens": 4000,
        },
    },

    # Logging — controls file logging to ~/.hermes/logs/.
    # agent.log captures INFO+ (all agent activity); errors.log captures WARNING+.
    "logging": {
        "level": "INFO",       # Minimum level for agent.log: DEBUG, INFO, WARNING
        "max_size_mb": 5,      # Max size per log file before rotation
        "backup_count": 3,     # Number of rotated backup files to keep
    },

    # Remotely-hosted model catalog manifest.  When enabled, the CLI fetches
    # curated model lists for OpenRouter and Nous Portal from this URL,
    # falling back to the in-repo snapshot on network failure.  Lets us
    # update model picker lists without shipping a hermes-agent release.
    # The default URL is served by the docs site GitHub Pages deploy.
    "model_catalog": {
        "enabled": True,
        "url": "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json",
        # Disk cache TTL in hours.  Beyond this, the CLI refetches on the
        # next /model or `hermes model` invocation; network failures
        # silently fall back to the stale cache.
        "ttl_hours": 1,
        # Optional per-provider override URLs for third parties that want
        # to self-host their own curation list using the same schema.
        # Example:
        #   providers:
        #     openrouter:
        #       url: https://example.com/my-curation.json
        "providers": {},
    },

    # Per-model metadata overrides — manually declare context_window,
    # max_output_tokens, capabilities, or model family for any
    # provider+model. Recognized fields: context_window,
    # max_output_tokens, supports_tools, supports_vision,
    # supports_reasoning, model_family.
    #
    # Semantics:
    #   1. Explicit (model_overrides.<provider>.<model_id>): wins over
    #      models.dev, OpenRouter, and hardcoded defaults for the fields
    #      it sets. NOTE: an explicit model.context_length (global) and a
    #      custom_providers per-model context_length are user settings at
    #      other layers and are consulted in the resolution chain order
    #      documented in agent/model_metadata.py.
    #   2. Fill-gap defaults (model_overrides.<provider>._default and
    #      model_overrides._default): apply ONLY to models the catalog
    #      does not know. They never displace catalog data for known
    #      models, so a _default cannot accidentally clamp every model
    #      of a provider.
    #
    # An unknown model id (not in models.dev) starts from safe defaults
    # (200K context, tools on, vision/reasoning off) and the override
    # patches the fields it sets — overriding a model the catalog
    # doesn't know yet is the supported self-unblock path (#84482,
    # #8731).
    #
    # Provider keys accept the Hermes provider id (as used elsewhere in
    # this file) or the models.dev provider id; model ids match
    # case-insensitively.
    #
    # Example:
    #   model_overrides:
    #     upstage:
    #       solar-pro4:
    #         context_window: 524288
    #       syn-pro:
    #         context_window: 65536
    #     custom:my-local-vllm:
    #       my-llava-model:
    #         context_window: 8192
    #         supports_vision: true
    #         supports_reasoning: false
    #         supports_tools: true
    #     _default:            # fill-gap only: models not in the catalog
    #       context_window: 128000
    "model_overrides": {},

    # models.dev registry — provider/model metadata (context windows,
    # capabilities, pricing, modalities).  The agent fetches this on startup
    # and serves from cache; a background daemon refreshes stale data.
    # Override ``url`` to point at a mirror (e.g. a self-hosted copy behind
    # a corporate proxy).  ETag conditional GET ensures refreshes are
    # cheap (304 = no download).
    "models_dev": {
        "url": "",  # empty = default https://models.dev/api.json
    },

    # Network settings — workarounds for connectivity issues.
    "network": {
        # Force IPv4 connections.  On servers with broken or unreachable IPv6,
        # Python tries AAAA records first and hangs for the full TCP timeout
        # before falling back to IPv4.  Set to true to skip IPv6 entirely.
        "force_ipv4": False,
    },

    # Gateway monitoring — Service Health Monitoring plus redacted Operational
    # Diagnostics for the gateway daemon, exported over OTLP to an
    # operator-configured endpoint (OTEL Collector, DataDog, ...). Content-free
    # by construction: no prompts, messages, tool args/results, session
    # history, usage analytics, audit logs, or trajectories. Off by default;
    # nothing is collected or sent until an operator enables it and sets an
    # endpoint.
    "monitoring": {
        # Stable install identifier attached to exported health signals so an
        # operator can tell instances apart in their collector. Empty string
        # means "mint a fresh UUID on first use"; clear it to rotate. Carries
        # no account identity.
        "install_id": "",
        # Gateway health & diagnostics export.
        "gateway_health_export": {
            "enabled": False,
            "metrics_enabled": True,
            "diagnostic_events_enabled": True,
            "warning_error_events_enabled": True,
            "export_interval_seconds": 60,
            "logs_export_interval_seconds": 5,
            "resource_attributes": {
                "service.name": "hermes-gateway",
                "deployment.environment.name": "production",
            },
        },
        # OTLP destination. headers_env maps header names to ENVIRONMENT
        # VARIABLE NAMES (never secret values); values are read from the
        # environment at export time.
        "export": {
            "otlp": {
                "enabled": False,
                "endpoint": "",
                "headers_env": {},
            },
        },
    },

    # Gateway settings — control how messaging platforms (Telegram, Discord,
    # Slack, etc.) deliver agent-produced files as native attachments.
    "gateway": {
        # Optional named-profile allowlist for multiplex mode. None preserves
        # the historical serve-all behavior; [] serves only the default.
        "multiplex_profile_allowlist": None,

        # Durable delivery-obligation ledger: final agent responses are
        # recorded in state.db around the platform send, and a gateway that
        # died between finalize and platform ACK redelivers the stored
        # response on the next boot (ambiguous cases carry a visible
        # "recovered reply — may be a duplicate" marker; honest
        # at-least-once). Disable to lose in-flight final responses on
        # crash/restart, as before.
        "delivery_ledger": True,

        # Seconds the gateway waits for a single messaging platform to finish
        # connecting during startup (and on reconnect). Discord in particular
        # can blow past the old fixed 30s when an account has many slash
        # commands to sync (#19776: 90-173 skills → ~28-31s sync). Raise this
        # if your gateway hits "discord connect timed out" / "Timeout waiting
        # for connection to Discord" restart loops. ``0`` or negative disables
        # the timeout entirely (wait indefinitely). Bridged at startup to the
        # internal HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT env var, which still
        # works as a manual override and wins if set explicitly.
        "platform_connect_timeout": 30,

        # In-process event-loop liveness watchdog (#69089). A daemon OS thread
        # probes the gateway asyncio loop; after consecutive missed probes it
        # dumps all-thread stacks and hard-exits with the service-restart exit
        # code so the supervisor (systemd/launchd) revives the process instead
        # of leaving a wedged-but-alive zombie. Set to false to disable.
        "loop_watchdog": True,

        # Whether the gateway keeps writing the legacy sessions.json mirror of
        # its routing index. The primary copy lives in state.db (the
        # gateway_routing table). Default True for backward compatibility with
        # external tooling and downgrade safety; set to false to stop
        # producing ~/.hermes/sessions/sessions.json entirely.
        "write_sessions_json": True,

        # Scale-to-zero idle detection (Phase 0). The gateway watches for idle
        # and, when an instance is opted in via the NAS "Labs" toggle (carried as
        # the HERMES_SCALE_TO_ZERO env stamp) AND messaging is relay-only/absent
        # AND a wakeUrl is registered, drives the relay transport dormant so the
        # platform (e.g. Fly autostop:"suspend") can suspend the now-idle machine;
        # it wakes on the connector's wakeUrl poke. This is the idle TIMEOUT only
        # — whether the feature is enabled at all is the Labs toggle, never a
        # config key (decisions.md D2/D11). 0/negative falls back to the default.
        "scale_to_zero": {
            "idle_timeout_minutes": 5,
        },

        # Auto-resume restart-loop breaker (#30719, defense-3). When the
        # gateway is killed mid-turn (SIGTERM) and revived by a supervisor
        # (launchd KeepAlive / systemd Restart=), it auto-resumes the
        # restart-interrupted session on the next boot. If the resumed turn
        # keeps triggering another kill (e.g. the agent runs a raw
        # `launchctl kickstart ai.hermes.gateway` that defenses 1-2 don't
        # cover), the result is a tight SIGTERM-respawn loop. This breaker
        # chains restart-interrupted boots together and, once `max_restarts`
        # of them chain up, SKIPS auto-resume for that boot — the gateway
        # still starts and serves real inbound messages, it just stops
        # replaying the session that keeps killing it. Set `max_restarts` to
        # 0 to disable the breaker.
        # Two boots belong to the same chain when they are no more than
        # `max_gap_seconds` apart (floored by `window_seconds`). Chaining on
        # the GAP rather than on a fixed window is what makes the breaker see
        # SLOW crash cycles: a loop whose period exceeds the window used to
        # prune its own history on every boot, so the counter never left 1 and
        # the breaker never tripped — e.g. the ~150s wedged-event-loop cycle in
        # #81642 (stall -> ~90s liveness-watchdog hard-exit -> respawn ->
        # auto-resume replays the same session), which also makes
        # `hermes update` hang because it can never drain the gateway.
        "restart_loop_guard": {
            "max_restarts": 3,
            "window_seconds": 60,
            "max_gap_seconds": 300,
        },

        # Portable respawn-storm circuit breaker (complements
        # ``restart_loop_guard`` above). Counts gateway (re)starts in a sliding
        # window and, when too many land, sleeps an exponential backoff before
        # booting so a crash-looping supervisor (launchd KeepAlive, systemd
        # Restart=always) can't hammer the process into a respawn storm.
        # ``max_starts <= 0`` disables the breaker. The env vars
        # ``HERMES_GATEWAY_MAX_STARTS`` / ``HERMES_GATEWAY_START_WINDOW_S``
        # override these defaults for escape-hatch use.
        "respawn_storm": {
            "max_starts": 5,
            "window_seconds": 120,
        },

        # Inject a human-readable timestamp prefix (e.g.
        # "[Tue 2026-04-28 13:40:53 CEST]") onto user messages IN THE MODEL'S
        # CONTEXT so the agent has temporal awareness of when each message was
        # sent. Off by default — when off, the model sees clean message text.
        # Persisted transcripts always stay clean (the timestamp is stored as
        # message metadata regardless of this toggle), so turning it on later
        # surfaces send-times for past messages too.
        "message_timestamps": {
            "enabled": False,
        },

        # Maximum bytes for an inbound image / audio / video payload the
        # gateway will buffer into memory and cache to disk. Inbound media is
        # read fully into RAM before being written, so an unbounded upload
        # (Discord Nitro allows 500 MB) or a remote media URL pointing at a
        # huge file can spike memory and OOM-kill the gateway on constrained
        # deployments. Enforced in the shared cache helpers
        # (gateway/platforms/base.py), so the cap holds across every platform
        # adapter. ``0`` disables the cap. Default 128 MiB.
        "max_inbound_media_bytes": 134217728,

        # When false (default), any file path the agent emits is delivered
        # as a native attachment as long as it isn't under the credential /
        # system-path denylist (/etc, /proc, ~/.ssh, ~/.aws, ~/.hermes/.env,
        # auth.json, etc.). This matches the symmetry of inbound delivery
        # — we accept any document type the user uploads, and the agent
        # can hand back any file that isn't a credential.
        #
        # When true, fall back to the older allowlist+recency-window
        # behavior: files must live under the Hermes cache, under
        # ``media_delivery_allow_dirs``, or be freshly produced inside the
        # ``trust_recent_files_seconds`` window. Recommended for
        # public-facing gateways where prompt injection from one user
        # shouldn't be able to exfiltrate the host's secrets to that same
        # user. Bridged to HERMES_MEDIA_DELIVERY_STRICT.
        "strict": False,
        # Extra directories from which model-emitted bare file paths may be
        # uploaded as native gateway attachments. Files inside the Hermes
        # cache (~/.hermes/cache/{documents,images,audio,video,screenshots})
        # are always trusted; this list adds operator-controlled roots
        # (project dirs, scratch dirs, mounted shares). Accepts a list of
        # absolute paths or a single os.pathsep-separated string. Bridged
        # to HERMES_MEDIA_ALLOW_DIRS at gateway startup. Tilde paths are
        # expanded. Honored in both default and strict mode.
        "media_delivery_allow_dirs": [],
        # When true, files whose mtime is within ``trust_recent_files_seconds``
        # of "now" are trusted for native delivery even outside the cache /
        # operator allowlist — useful for ``pandoc -o /tmp/report.pdf`` or
        # PDFs the agent writes into a working directory. System paths
        # (/etc, /proc, ~/.ssh, ~/.aws, etc.) remain blocked regardless.
        # Disable to fall back to pure-allowlist mode. Bridged to
        # HERMES_MEDIA_TRUST_RECENT_FILES. Only consulted when ``strict``
        # is true; in default mode the denylist alone gates delivery.
        "trust_recent_files": True,
        # Recency window in seconds. 600 (10 min) comfortably covers a
        # multi-tool agent turn. Bridged to HERMES_MEDIA_TRUST_RECENT_SECONDS.
        # Only consulted when ``strict`` is true.
        "trust_recent_files_seconds": 600,

        # OpenAI-compatible API server platform
        # (gateway/platforms/api_server.py).
        "api_server": {
            # Maximum number of agent runs the API server will service
            # concurrently. Requests to /v1/chat/completions, /v1/responses,
            # and /v1/runs that arrive while this many runs are already
            # in flight are rejected with HTTP 429 + a Retry-After header,
            # bounding CPU / memory / upstream-LLM-quota exhaustion from a
            # request flood. Set to 0 to disable the cap entirely.
            "max_concurrent_runs": 10,
        },
    },

    # Real-time token streaming to messaging platforms (Telegram, Discord,
    # Slack, etc.). Read at the top level by the gateway; absent this block the
    # gateway falls back to these same defaults, so adding it here only makes
    # the feature discoverable in config.yaml — it does not change behavior.
    #
    # Disabled by default: streaming costs extra edit/draft API calls per
    # response. Set ``enabled: true`` and restart the gateway to turn it on.
    "streaming": {
        # Master switch. When false, each response is delivered as a single
        # final message (no progressive updates).
        "enabled": False,
        # Transport selection:
        #   "auto"  — prefer native draft streaming where the platform
        #             supports it (Telegram DMs via sendMessageDraft,
        #             Bot API 9.5+) and fall back to edit-based elsewhere.
        #             Safe global default: platforms without draft support
        #             (Discord, Slack, Matrix, Telegram groups) transparently
        #             use the edit path, so "auto" only upgrades chats that
        #             can render the smoother native preview.
        #   "draft" — explicitly request native drafts; falls back to edit
        #             when the platform/chat doesn't support them.
        #   "edit"  — progressive editMessageText only (legacy behavior).
        #   "off"   — disable streaming entirely (same as enabled: false).
        "transport": "auto",
        # Minimum seconds between progressive edits — tuned for Telegram's
        # ~1 edit/s flood envelope.
        "edit_interval": 0.8,
        # Flush the buffer to the platform once this many characters have
        # accumulated, so short replies feel near-instant.
        "buffer_threshold": 24,
        # Cursor glyph appended to the in-progress message while streaming.
        "cursor": " \u2589",
        # When >0, the final edit for a long-running streamed response is
        # delivered as a fresh message if the preview has been visible at
        # least this many seconds, so the platform timestamp reflects
        # completion time. Telegram only; other platforms ignore it.
        "fresh_final_after_seconds": 0.0,
    },

    # Session storage — controls automatic cleanup of ~/.hermes/state.db.
    # state.db accumulates every session, message, tool call, and FTS5 index
    # entry forever.  Without auto-pruning, a heavy user (gateway + cron)
    # reports 384MB+ databases with 68K+ messages, which slows down FTS5
    # inserts, /resume listing, and insights queries.
    "sessions": {
        # When true, prune ended sessions inactive for retention_days once
        # per (roughly) min_interval_hours at CLI/gateway/cron startup.
        # Activity is the latest message timestamp, falling back to creation
        # time for empty sessions. Active sessions are always preserved.
        # Default false: session history is valuable for search recall, and
        # silently deleting it could surprise users.  Opt in explicitly.
        "auto_prune": False,
        # How many inactive days of ended-session history to keep. Matches
        # the default of ``hermes sessions prune``.
        "retention_days": 90,
        # When true, auto-archive (soft-hide, never delete) sessions that
        # haven't been touched in ``auto_archive_days`` days, once per
        # (roughly) min_interval_hours.  "Touched" is last activity, not
        # creation, so an old-but-recently-used session is spared.  Pinned
        # sessions are always exempt.  Off by default — opt in explicitly.
        "auto_archive": False,
        # Idle threshold (days of no activity) before auto-archive hides a
        # session.  Only applies when auto_archive is true.
        "auto_archive_days": 3,
        # VACUUM after a prune that actually deleted rows.  SQLite does not
        # reclaim disk space on DELETE — freed pages are just reused on
        # subsequent INSERTs — so without VACUUM the file stays bloated
        # even after pruning.  VACUUM blocks writes for a few seconds per
        # 100MB, so it only runs at startup, and only when prune deleted
        # ≥1 session.
        "vacuum_after_prune": True,
        # Minimum days between successful VACUUM rewrites. Pruning can still
        # run on its normal cadence while SQLite reuses the freed pages.
        "min_vacuum_interval_days": 30,
        # Minimum hours between auto-maintenance runs (avoids repeating
        # the sweep on every CLI invocation).  Tracked via state_meta in
        # state.db itself, so it's shared across all processes.
        "min_interval_hours": 24,
        # Legacy per-session JSON snapshot writer.  When true, the agent
        # rewrites ``~/.hermes/sessions/session_{sid}.json`` on every turn
        # boundary with the full message list.  state.db is canonical and
        # has every field the snapshot stored (plus per-message timestamps
        # and token counts), so this is off by default — the snapshots had
        # no consumer outside their own overwrite guard and accumulated
        # GBs of disk on heavy users.  Opt in only if you have an external
        # tool that consumes the JSON files directly.
        "write_json_snapshots": False,
        # Search-index (FTS) storage optimization — the compact v23 layout
        # that drops duplicate content copies and stops trigram-indexing tool
        # output (typically reclaims ~60%+ of state.db on heavy users). It is
        # OPT-IN: existing databases keep their working legacy index until the
        # user runs `hermes sessions optimize-storage`, because the rebuild is
        # disk-heavy and long on large DBs (see that command's disk preflight).
        #
        #   "advise" (default): `hermes update` prints a one-line notice with
        #     the reclaimable size and the command, when a legacy index is
        #     detected. Nothing is changed automatically.
        #   "require": the notice is shown as a REQUIRED upgrade (firmer copy),
        #     and future tooling may gate on it. Flip this default in a future
        #     release when we're ready to make the v23 layout mandatory — the
        #     command, progress bar, and resumability are already in place, so
        #     enforcement is a copy/gating change, not new migration code.
        #   "off": suppress the notice entirely.
        "fts_optimize_notice": "advise",
        # CJK-bigram search index (messages_fts_cjk, cjk_unicode61 loadable
        # tokenizer). When the extension is built (native/fts5_cjk/build.sh →
        # ~/.hermes/lib/libfts5_cjk.so), 1-2 char CJK terms (일본, 项目, ...)
        # get index-speed exact matching instead of LIKE full-table scans.
        # True (default): use the index when the extension is present; the
        # setting is inert when it isn't. False: never load the extension or
        # serve the cjk index. Bridged to HERMES_CJK_FTS (internal carrier).
        "cjk_fts": True,
        # Slow session-search log threshold in milliseconds: searches at or
        # above it log one INFO line with the routing path taken (fts_cjk /
        # fts5 / trigram / like_scan) so latency regressions stay
        # attributable per query shape. 0 logs every search. Bridged to
        # HERMES_SEARCH_SLOW_MS (internal carrier).
        "search_slow_ms": 1000,
        # Transcript safety limits. A runaway session (hundreds of thousands
        # of rows) can exhaust memory when its transcript is materialized in
        # one shot, so interactive resume and in-memory export are guarded by
        # bounded row counts. Set a limit to 0 to disable that guard.
        # Max active messages (across the full compression lineage) a session
        # may hold and still be resumed interactively (CLI/TUI/desktop).
        "max_resume_messages": 20000,
        # Max active messages a single session may hold for an in-memory
        # (non-streaming) export such as `hermes sessions export`. Checked
        # per session, so full-DB backups of many small sessions still work.
        "max_export_messages": 20000,
    },

    # Contextual first-touch onboarding hints (see agent/onboarding.py).
    # Each hint is shown once per install and then latched here so it
    # never fires again.  Users can wipe the section to re-see all hints.
    "onboarding": {
        "seen": {},
        # Structured profile-build path offered on the very first gateway
        # message ever. "ask" (default) -> offer to build a user profile
        # (opt-in, consent-gated; the agent asks before any lookup and never
        # reads connected accounts silently). "off" -> plain intro only.
        # The offer fires at most once (latched under onboarding.seen).
        "profile_build": "ask",
    },

    # Privacy-safe aggregate metrics written only to this profile's local
    # telemetry directory. Collection is opt-in and no remote sink exists.
    "telemetry": {
        "shared_metrics": {
            "enabled": False,
        },
    },

    # ``hermes doctor`` behaviour.
    "doctor": {
        # Per-probe timeout (seconds) for the opt-in `hermes doctor --live`
        # real-call backend probes (Firecrawl/FAL/browser/MCP/TTS/STT).
        "live_probe_timeout": 10,
    },

    # ``hermes update`` behaviour.
    "updates": {
        # Pre-update safety backup — ONE consolidated mechanism, three modes:
        #
        #   quick (default) — snapshot critical small state files (pairing
        #     JSONs, cron jobs, config.yaml, .env, auth.json, per-profile
        #     DBs) into <HERMES_HOME>/state-snapshots/ before the update.
        #     Files over 1 GiB (e.g. a bloated state.db) are skipped with a
        #     warning so the snapshot stays fast. Restore via ``/snapshot``.
        #     This is the #15733 (lost pairing data) / #34600 (emptied cron
        #     jobs) safety net.
        #   full — the quick snapshot PLUS a full ``hermes backup``-style zip
        #     of HERMES_HOME into <HERMES_HOME>/backups/, restorable with
        #     ``hermes import``. Can add minutes on large homes. This is the
        #     #48200 (wrong-path wipe) safety net. ``--backup`` forces this
        #     for a single run.
        #   off — no pre-update backup of any kind. ``--no-backup`` forces
        #     this for a single run.
        #
        # Legacy boolean values are honored: true -> full, false -> off.
        "pre_update_backup": "quick",
        # How many full pre-update backup zips to retain (mode ``full``).
        # Older ones are pruned automatically after each successful backup.
        # Values below 1 are floored to 1 — the backup just created is
        # always preserved. The quick snapshot always keeps exactly 1.
        "backup_keep": 5,
        # What `hermes update` does with uncommitted local changes to the
        # source tree when it runs NON-interactively — i.e. triggered from
        # the desktop/chat app or the gateway, where there's no TTY to answer
        # a restore prompt. Interactive (terminal) updates are unaffected:
        # they always stash the changes and ask whether to restore, exactly
        # as they always have.
        #   "stash"   — auto-stash the changes, pull, then auto-restore them
        #               on top of the updated code (the safe default; nothing
        #               is ever lost — conflicts are preserved in a git stash).
        #   "discard" — auto-stash the changes and throw the stash away after
        #               the pull. Use this only if you never intend to keep
        #               local edits to the source tree on this machine.
        #               Stash-and-drop (not `reset --hard` + `clean -fd`) so
        #               ignored paths — node_modules, venv, build outputs —
        #               are never touched.
        "non_interactive_local_changes": "stash",
        # When `hermes update` finds the source checkout parked on a feature
        # branch (left behind by tooling or a manual checkout), switch back
        # to the update target automatically — but only when the branch is
        # clean and every commit on it is already merged into the target.
        # When it is not safe, the code update is SKIPPED with a loud
        # warning instead of pretending success (2026-08-17 incident:
        # "✓ Code updated!" printed while the checkout stayed days behind
        # main on a stale branch). Set false to never auto-switch.
        "auto_switch_parked_branch": True,
        # Refresh an already-installed cua-driver during `hermes update`.
        # The refresh is best-effort and macOS-only. Turn this off if the
        # upstream installer is not appropriate for the machine, for example
        # on non-admin accounts where `/Applications` is not writable.
        "refresh_cua_driver": True,
    },

    # Language Server Protocol — semantic diagnostics from real
    # language servers (pyright, gopls, rust-analyzer, etc.) wired
    # into the post-write lint check used by ``write_file`` and
    # ``patch``.
    #
    # LSP is gated on git-workspace detection: when the agent's
    # cwd (or the file being edited) is inside a git worktree, LSP
    # runs against that workspace.  When neither is in a git repo,
    # LSP stays dormant and the in-process syntax check is the only
    # tier — handy for Telegram/Discord chats where the cwd is the
    # user's home directory.
    "lsp": {
        # Master toggle.  Setting this to false disables the entire
        # subsystem — no servers spawn, no background event loop, no
        # cost.
        "enabled": True,

        # Diagnostic-wait mode for the post-write check.
        # ``"document"`` waits up to ``wait_timeout`` seconds for the
        # current file's diagnostics; ``"full"`` additionally requests
        # workspace-wide diagnostics (slower).
        "wait_mode": "document",
        "wait_timeout": 5.0,

        # How to handle missing server binaries.
        # ``"auto"`` — try to install via npm/go/pip into
        #              ``<HERMES_HOME>/lsp/bin/`` on first use.
        # ``"manual"`` — only use binaries already on PATH.
        # ``"off"`` — alias for ``manual``.
        "install_strategy": "auto",

        # Idle language servers are shut down automatically after this
        # many seconds with no file activity, then respawned on demand.
        # Prevents long-running gateway/CLI processes from accumulating
        # stale pyright/gopls/tsserver children (hundreds of MB each,
        # plus pipe FDs) as the agent moves across worktrees.  Set to 0
        # to disable idle reaping and keep servers for process lifetime.
        "idle_timeout": 600.0,

        # Per-server overrides.  Each key is a server_id from the
        # registry (``pyright``, ``typescript``, ``gopls``,
        # ``rust-analyzer``, etc.) and accepts:
        #   disabled: true
        #     — skip this server even when its extensions match
        #   command: ["full/path/to/server", "--stdio"]
        #     — pin a custom binary path; bypasses auto-install
        #   env: {"KEY": "value"}
        #     — extra env vars passed to the spawned process
        #   initialization_options: {...}
        #     — merged into the LSP ``initializationOptions``
        # Empty by default; the registry defaults work for typical
        # setups.
        "servers": {},
    },


    # X (Twitter) Search via xAI's built-in x_search Responses tool.
    # The tool registers when xAI credentials are available (SuperGrok
    # OAuth or XAI_API_KEY) AND the x_search toolset is enabled in
    # `hermes tools`. These settings tune the backing Responses API call.
    "x_search": {
        # xAI model used for the Responses call. grok-4.5 is the
        # recommended default; any Grok model with x_search tool
        # access works.
        "model": "grok-4.5",
        # Optional reasoning effort sent to xAI Responses API models that
        # support it. Leave null to preserve the selected model's default.
        "reasoning_effort": None,
        # Request timeout in seconds (minimum 30). x_search can take
        # 60-120s for complex queries — the default is generous.
        "timeout_seconds": 180,
        # Number of automatic retries on 5xx / ReadTimeout / ConnectionError.
        # Each retry backs off (1.5x attempt seconds, capped at 5s).
        "retries": 2,
    },

    # =========================================================================
    # External secret sources
    # =========================================================================
    # Pull credentials from external secret managers at process startup
    # rather than storing them in ~/.hermes/.env.
    "secrets": {
        # Optional explicit ordering of enabled secret sources.  When
        # omitted, sources run in registration order (bundled first,
        # then plugin-registered).  Regardless of this list, "mapped"
        # sources (explicit VAR→ref bindings, e.g. a future 1Password
        # env: map) always take precedence over "bulk" sources
        # (project dumps like Bitwarden BSM), and the first source to
        # claim a var wins — later claims are skipped with a warning.
        # Example: sources: [onepassword, bitwarden]
        # "sources": [],
        "bitwarden": {
            # Master switch.  When false, BSM is never contacted and the
            # bws binary is never auto-installed — same as not having
            # this section at all.
            "enabled": False,
            # Name of the env var that holds the Bitwarden machine-account
            # access token.  This is the one bootstrap secret; it lives
            # in ~/.hermes/.env (or your shell) and never in config.yaml.
            "access_token_env": "BWS_ACCESS_TOKEN",
            # UUID of the BSM project to sync from.
            "project_id": "",
            # Seconds to reuse a fresh disk/memory cache entry before contacting
            # Bitwarden again. 0 disables normal fresh-cache reuse.
            "cache_ttl_seconds": 300,
            # Optional encrypted last-good fallback for network/timeout outages.
            # When enabled, successful BWS fetches write AES-GCM encrypted cache
            # material under ~/.hermes/cache/. If a later startup cannot reach
            # Bitwarden due to NETWORK/TIMEOUT, Hermes may use this encrypted
            # cache for up to max_stale_seconds. Auth failures do not fall back.
            "encrypted_cache": {
                "enabled": False,
                "max_stale_seconds": 0,
            },
            # When True, BSM values overwrite existing env vars.  Default
            # True because the point of using BSM is centralized rotation —
            # if .env had the final say, rotating in Bitwarden wouldn't
            # take effect until you also cleared the matching .env line.
            "override_existing": True,
            # When True, the bws binary is auto-downloaded into
            # ~/.hermes/bin/ on first use.  When False you must install
            # bws yourself and have it on PATH.
            "auto_install": True,
            # Bitwarden region / self-hosted endpoint.  Empty string
            # means use the bws CLI default (US Cloud,
            # https://vault.bitwarden.com).  Set to
            # https://vault.bitwarden.eu for EU Cloud, or your own URL
            # for self-hosted Bitwarden.  Plumbed into the bws subprocess
            # as BWS_SERVER_URL.  Prompted for during
            # `hermes secrets bitwarden setup`.
            "server_url": "",
        },
        "onepassword": {
            # Master switch.  When false, the op CLI is never invoked —
            # same as not having this section at all.
            "enabled": False,
            # Mapping of env-var name → 1Password secret reference
            # (op://vault/item/field).  Each entry is resolved with a
            # single `op read` at startup.
            "env": {},
            # Optional account shorthand / sign-in address passed as
            # `op read --account <account>`.  Empty = op's default account.
            "account": "",
            # Name of the env var holding a 1Password service-account token
            # for headless auth.  Sourced from ~/.hermes/.env (or the shell)
            # and exported to the op child as OP_SERVICE_ACCOUNT_TOKEN.
            # Leave the var unset to use an interactive/desktop op session.
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
            # Optional absolute path to the op binary.  When set it is used
            # verbatim (PATH is not consulted) — pin this to avoid trusting
            # whatever `op` appears first on PATH.  Empty = resolve via PATH.
            "binary_path": "",
            # Seconds to cache resolved values in-process and on disk.  0
            # disables BOTH cache layers (no values are written to disk).
            "cache_ttl_seconds": 300,
            # When True (default), resolved values overwrite existing env
            # vars so rotating a secret in 1Password takes effect on next
            # start.  Flip to false to let .env / shell exports win locally.
            "override_existing": True,
        },
    },

    # Paste collapse thresholds (TUI + CLI).
    #
    # paste_collapse_threshold (default 5)
    #   Bracketed-paste handler. Pastes with this many newlines or more
    #   collapse to a file reference. Set 0 to disable.
    #
    # paste_collapse_threshold_fallback (default 5)
    #   Fallback heuristic for terminals without bracketed paste support.
    #   Same line count test but heuristically gated by chars-added /
    #   newlines-added to avoid false positives from normal typing.
    #   Set 0 to disable.
    #
    # paste_collapse_char_threshold (default 2000)
    #   Long single-line paste guard. Pastes whose total char length
    #   reaches this value collapse to a file reference even if line
    #   count is below the line threshold. Catches the "8000 chars of
    #   minified JSON / log output on one line" case. Set 0 to disable.
    "paste_collapse_threshold": 5,
    "paste_collapse_threshold_fallback": 5,
    "paste_collapse_char_threshold": 2000,

    # Computer Use (cua-driver) toolset settings.
    "computer_use": {
        # cua-driver ships with anonymous usage telemetry (PostHog) ENABLED
        # by default upstream. Hermes disables it for our users unless they
        # explicitly opt in here. When false (default), Hermes sets
        # CUA_DRIVER_RS_TELEMETRY_ENABLED=0 in the cua-driver child env for
        # every invocation (MCP backend, status, doctor, install). Set true
        # to let cua-driver use its own default (telemetry on).
        "cua_telemetry": False,
        # Cap driver screenshot longest edge (pixels) via set_config on
        # session start. Shrinks SOM multimodal payloads; 0 disables.
        "max_image_dimension": 1456,
        # Mode for capture_after follow-ups: som (screenshot + overlays —
        # default), ax (elements only, no PNG — faster), vision (pixels only).
        "capture_after_mode": "som",
        # Disable the cursor overlay rendered by cua-driver. The overlay
        # shows where agent actions land but can peg a core when idle
        # (macOS vImage redraw loop #47032; Linux/WSL2 idle spin #28152).
        # cua-driver ≥ 0.6.x supports --no-overlay; Hermes also calls
        # set_agent_cursor_enabled(false) after start_session when this is on.
        #   None  = auto-detect (off on macOS + headless/WSL2 Linux; on elsewhere)
        #   True  = always disable the overlay
        #   False = always enable the overlay
        "no_overlay": None,
        # cua-driver permission mode for each Hermes computer-use runtime.
        #   standard (default) — cua-driver's own approval boundary. Protected
        #     operations (e.g. attaching to an existing signed-in browser
        #     profile) fail closed unless grant_existing_profile is enabled
        #     below.
        #   bounded — repeatable automation under a user-reviewed session
        #     capability manifest (set capability_manifest below). No runtime
        #     prompts; anything outside the manifest fails closed inside
        #     cua-driver.
        # `unrestricted` is intentionally NOT accepted here: it stays bound to
        # the explicit per-session YOLO toggle so a config line can never
        # silently bypass approvals.
        "permission_mode": "standard",
        # Absolute or ~ path to the reviewed cua-driver capability
        # manifest used when permission_mode is "bounded". Hermes passes the
        # canonical --capability-manifest and --approve-capability-manifest
        # flags when it launches the runtime. See
        # https://cua.ai/docs/reference/cua-driver/permission-modes
        "capability_manifest": "",
        # Pre-authorize existing-profile browser attachment in standard mode
        # (cua-driver's trusted-launcher `--grant existing-profile`). When
        # true, the agent can attach to your already-running, signed-in
        # Chrome/Edge window — exposing that profile's live pages, cookies,
        # and storage to the browser protocol — without a per-use prompt.
        # Leave false to keep existing-profile attachment failing closed;
        # isolated driver-owned profiles work either way.
        "grant_existing_profile": False,
    },

    # =========================================================================
    # Egress credential-injection proxy (iron-proxy)
    # =========================================================================
    # When enabled, outbound traffic from remote terminal sandboxes (Docker
    # today; Modal/SSH in follow-ups) is routed through a managed iron-proxy
    # subprocess.  The sandbox sees opaque proxy tokens; iron-proxy swaps in
    # real API credentials at the egress boundary.  Compromising the sandbox
    # leaks tokens that only work behind the configured trusted proxy boundary
    # (CA private key + proxy endpoint integrity are part of that boundary).
    #
    # Configure with `hermes egress setup`.  Disabled by default — the rest of
    # Hermes works exactly as before with `enabled: false`.
    "proxy": {
        # Master switch.  When false, iron-proxy is never started, no docker
        # mounts are added, no binaries are auto-installed — feature is a
        # complete no-op.
        "enabled": False,
        # Tunnel listener port.  Sandboxes get `HTTPS_PROXY=http://<host>:<port>`.
        # 9090 is the default; collide-aware setup wizard can reassign.
        "tunnel_port": 9090,
        # Auto-download the pinned iron-proxy binary into ~/.hermes/bin/ on
        # first use.  When false, you must place `iron-proxy` on PATH yourself.
        "auto_install": True,
        # Where iron-proxy looks up the real upstream secrets at egress time.
        # "env"        — process env (default; what bitwarden integration
        #                already populates if you use it)
        # "bitwarden"  — refetch via `bws secret list` on each proxy restart;
        #                rotation in the Bitwarden web app propagates without
        #                touching .env (requires `secrets.bitwarden.enabled`).
        "credential_source": "env",
        # When true, the Docker backend refuses to start a sandbox if the
        # proxy is enabled but not running.  False = fall back to direct
        # outbound with real credentials in the sandbox (the legacy posture).
        "enforce_on_docker": True,
        # NOTE: ``fail_on_uncovered_providers`` was removed.  It gated a
        # refuse-start when Anthropic / Azure OpenAI / Gemini env vars were
        # present — those providers are now first-class swapped providers
        # via per-provider match_headers rules (x-api-key, api-key,
        # x-goog-api-key), so the fail-closed tier is empty.  A leftover
        # key in existing user configs is ignored harmlessly.
        # When credential_source is bitwarden but the BWS access token /
        # project_id is missing OR the bws fetch returns no values for
        # mapped providers, the daemon raises by default.  Set this to
        # True to opt back in to the legacy "silently fall back to host
        # env" behaviour — useful for migrations where the operator wants
        # to switch credential_source to bitwarden but hasn't fully wired
        # BWS yet.  Defaults to false (strict).
        "allow_env_fallback": False,
        # SSRF deny list applied to outbound traffic.  Omit / leave empty
        # to use the safe default: loopback, link-local (incl. cloud
        # metadata IPs at 169.254.169.254), and RFC1918.  Set to an
        # explicit ``[]`` to opt out entirely (only sensible in hermetic
        # tests that need to reach a loopback upstream).
        "upstream_deny_cidrs": None,
        # Extra allowed upstream hosts beyond the bundled defaults (which
        # cover OpenRouter, OpenAI, Anthropic, Google, xAI, Mistral, Groq,
        # Together, DeepSeek, Nous).  Wildcards (`*.foo.com`) are supported.
        "extra_allowed_hosts": [],
    },

    # Hermes Desktop (Electron app) launch options. These only affect
    # `hermes desktop`; they do not touch the CLI/gateway.
    "desktop": {
        # Git repository discovery for the Desktop Projects sidebar. Empty
        # roots preserve the historical bounded scan of the user's home.
        "repo_scan_enabled": True,
        "repo_scan_roots": [],
        "repo_scan_exclude_paths": [],
        # Extra Electron command-line flags appended to every desktop launch,
        # e.g. ["--ozone-platform=x11"] on headless/VM X11 hosts that need an
        # explicit ozone backend, or GPU workaround flags. A list of strings;
        # a single string is also accepted and shell-split.
        "electron_flags": [],
        # GPU hardware acceleration policy for the desktop app:
        #   "auto"  - let the app detect remote displays (SSH/VNC/RDP) and
        #             disable GPU only then (default; current behavior).
        #   true    - always disable GPU acceleration (software rendering).
        #             Use on no-GPU VMs / Proxmox hosts where the GPU path hangs.
        #   false   - always keep GPU acceleration on, even over a remote display.
        # Bridged to the HERMES_DESKTOP_DISABLE_GPU env var the Electron app reads.
        "disable_gpu": "auto",
        # Linux keychain backend for secure token storage (Chromium's
        # --password-store switch, which safeStorage needs before it can
        # encrypt remote gateway tokens):
        #   "auto"  - detect the session keychain: KWallet via KDE session env
        #             vars, GNOME Keyring / any org.freedesktop.secrets
        #             provider (e.g. KeePassXC) via D-Bus (default).
        #   "gnome-libsecret" / "kwallet" / "kwallet5" / "kwallet6" / "basic"
        #           - force a specific backend ("basic" = unencrypted store).
        # Ignored on macOS/Windows. Bridged to the HERMES_DESKTOP_PASSWORD_STORE
        # env var the Electron app reads, so an explicit env var still wins.
        "password_store": "auto",
        # macOS only: optional persistent code-signing identity (a cert in the
        # login keychain — a self-signed "Code Signing" cert from Keychain
        # Access works; no Apple Developer account needed) used to re-sign
        # locally rebuilt desktop apps. A certificate-anchored Designated
        # Requirement stays stable across rebuilds, so TCC grants (Full Disk
        # Access, Desktop/Downloads/Documents, Accessibility, Automation,
        # microphone) survive every update. Empty keeps the default stable
        # ad-hoc signing (identifier-pinned requirement).
        "macos_signing_identity": "",
        # Auto-continue a turn that was killed mid-run by an app/backend/machine
        # crash: resuming that session re-submits the interrupted prompt (shown
        # as a "resumed interrupted turn" event) if the interruption is fresh.
        # A stale interruption just shows the recovered partial transcript.
        "auto_continue": {
            "enabled": True,
            # How recent the interruption must be to auto-continue (minutes).
            "freshness_minutes": 15,
            # Crash-loop breaker: max automatic re-runs of one interrupted turn.
            "max_attempts": 2,
        },
    },


    # Google Vertex AI provider (Gemini via the OpenAI-compatible endpoint).
    # Auth is OAuth2 (short-lived access tokens minted from a service-account
    # JSON or Application Default Credentials) — NOT a static API key. The
    # credential *path* is a secret-adjacent pointer and lives in .env
    # (VERTEX_CREDENTIALS_PATH / GOOGLE_APPLICATION_CREDENTIALS); these two
    # settings are non-secret routing config and live here. Both are bridged to
    # the VERTEX_PROJECT_ID / VERTEX_REGION env vars the adapter reads, so an
    # explicit env var still wins over config.yaml.
    "vertex": {
        # GCP project ID. Empty → use the project_id embedded in the service
        # account JSON (or ADC-resolved project).
        "project_id": "",
        # Vertex region. "global" is required for the Gemini 3.x preview models
        # (regional endpoints silently 404 them). Override to a regional value
        # (e.g. "us-central1") only if your models are pinned to a region.
        "region": "global",
    },

    # Config schema version - bump this when adding new required fields
    "_config_version": 38,
}

# Optional environment variables that enhance functionality
OPTIONAL_ENV_VARS = {
    # ── Provider (handled in provider selection, not shown in checklists) ──
    "NOUS_BASE_URL": {
        "description": "Nous Portal base URL override",
        "prompt": "Nous Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENROUTER_API_KEY": {
        "description": "OpenRouter API key (for vision, web scraping helpers, and MoA)",
        "prompt": "OpenRouter API key",
        "url": "https://openrouter.ai/keys",
        "password": True,
        "tools": ["vision_analyze"],
        "category": "provider",
        "advanced": True,
    },
    "GOOGLE_API_KEY": {
        "description": "Google AI Studio API key (also recognized as GEMINI_API_KEY)",
        "prompt": "Google AI Studio API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_API_KEY": {
        "description": "Google AI Studio API key (alias for GOOGLE_API_KEY)",
        "prompt": "Gemini API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_BASE_URL": {
        "description": "Google AI Studio base URL override",
        "prompt": "Gemini base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "VERTEX_CREDENTIALS_PATH": {
        "description": "Path to a Google Cloud service account JSON for Vertex AI (Gemini). "
                       "Vertex uses OAuth2, not a static API key — this points at the "
                       "credentials Hermes mints short-lived tokens from. Falls back to "
                       "GOOGLE_APPLICATION_CREDENTIALS, then to ADC (gcloud auth "
                       "application-default login). Set project/region under vertex: in config.yaml.",
        "prompt": "Vertex service account JSON path (leave empty to use ADC / GOOGLE_APPLICATION_CREDENTIALS)",
        "url": "https://cloud.google.com/iam/docs/keys-create-delete",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XAI_API_KEY": {
        "description": "xAI API key",
        "prompt": "xAI API key",
        "url": "https://console.x.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "XAI_BASE_URL": {
        "description": "xAI base URL override",
        "prompt": "xAI base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_API_KEY": {
        "description": "NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)",
        "prompt": "NVIDIA NIM API key",
        "url": "https://build.nvidia.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_BASE_URL": {
        "description": "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)",
        "prompt": "NVIDIA NIM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "LM_API_KEY": {
        "description": "LM Studio bearer token for auth-enabled local servers",
        "prompt": "LM Studio API key / bearer token",
        "url": None,
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "LM_BASE_URL": {
        "description": "LM Studio base URL override",
        "prompt": "LM Studio base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "GLM_API_KEY": {
        "description": "Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)",
        "prompt": "Z.AI / GLM API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ZAI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "Z_AI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GLM_BASE_URL": {
        "description": "Z.AI / GLM base URL override",
        "prompt": "Z.AI / GLM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_API_KEY": {
        "description": "Kimi / Moonshot API key",
        "prompt": "Kimi API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_BASE_URL": {
        "description": "Kimi / Moonshot base URL override",
        "prompt": "Kimi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_CN_API_KEY": {
        "description": "Kimi / Moonshot China API key",
        "prompt": "Kimi (China) API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "STEPFUN_API_KEY": {
        "description": "StepFun Step Plan API key",
        "prompt": "StepFun Step Plan API key",
        "url": "https://platform.stepfun.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "STEPFUN_BASE_URL": {
        "description": "StepFun Step Plan base URL override",
        "prompt": "StepFun Step Plan base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "ARCEEAI_API_KEY": {
        "description": "Arcee AI API key",
        "prompt": "Arcee AI API key",
        "url": "https://chat.arcee.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ARCEE_BASE_URL": {
        "description": "Arcee AI base URL override",
        "prompt": "Arcee base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "GMI_API_KEY": {
        "description": "GMI Cloud API key",
        "prompt": "GMI Cloud API key",
        "url": "https://www.gmicloud.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GMI_BASE_URL": {
        "description": "GMI Cloud base URL override",
        "prompt": "GMI Cloud base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "ACTUAL_API_KEY": {
        "description": "Actual Computer inference key (ac_...)",
        "prompt": "Actual Computer inference key",
        "url": "https://actual.inc/user/keys",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ACTUAL_BASE_URL": {
        "description": "Actual Computer base URL override (set to http://127.0.0.1:8080 for the local offline daemon)",
        "prompt": "Actual Computer base URL (leave empty for hosted relay)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "FIREWORKS_API_KEY": {
        "description": "Fireworks AI API key",
        "prompt": "Fireworks AI API key",
        "url": "https://app.fireworks.ai/settings/users/api-keys",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_API_KEY": {
        "description": "MiniMax API key (international)",
        "prompt": "MiniMax API key",
        "url": "https://www.minimax.io/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_BASE_URL": {
        "description": "MiniMax base URL override",
        "prompt": "MiniMax base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_API_KEY": {
        "description": "MiniMax API key (China endpoint)",
        "prompt": "MiniMax (China) API key",
        "url": "https://www.minimaxi.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_BASE_URL": {
        "description": "MiniMax (China) base URL override",
        "prompt": "MiniMax (China) base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek API key for direct DeepSeek access",
        "prompt": "DeepSeek API Key",
        "url": "https://platform.deepseek.com/api_keys",
        "password": True,
        "category": "provider",
    },
    "DEEPSEEK_BASE_URL": {
        "description": "Custom DeepSeek API base URL (advanced)",
        "prompt": "DeepSeek Base URL",
        "url": "",
        "password": False,
        "category": "provider",
    },
    "DASHSCOPE_API_KEY": {
        "description": "Alibaba Cloud DashScope API key (Qwen + multi-provider models)",
        "prompt": "DashScope API Key",
        "url": "https://modelstudio.console.alibabacloud.com/",
        "password": True,
        "category": "provider",
    },
    "DASHSCOPE_BASE_URL": {
        "description": "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)",
        "prompt": "DashScope Base URL",
        "url": "",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_QWEN_BASE_URL": {
        "description": "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)",
        "prompt": "Qwen Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_API_KEY": {
        "description": "OpenCode Zen API key (pay-as-you-go access to curated models)",
        "prompt": "OpenCode Zen API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "COMMANDCODE_API_KEY": {
        "description": "CommandCode API key (GOAT/Pro/Max/Provider plans — 30+ models via one key)",
        "prompt": "CommandCode API key",
        "url": "https://commandcode.ai/studio/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_BASE_URL": {
        "description": "OpenCode Zen base URL override",
        "prompt": "OpenCode Zen base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_API_KEY": {
        "description": "OpenCode Go API key ($10/month subscription for open models)",
        "prompt": "OpenCode Go API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_BASE_URL": {
        "description": "OpenCode Go base URL override",
        "prompt": "OpenCode Go base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HF_TOKEN": {
        "description": "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)",
        "prompt": "Hugging Face Token",
        "url": "https://huggingface.co/settings/tokens",
        "password": True,
        "category": "provider",
    },
    "HF_BASE_URL": {
        "description": "Hugging Face Inference Providers base URL override",
        "prompt": "HF base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_API_KEY": {
        "description": "Ollama Cloud API key (ollama.com — cloud-hosted open models)",
        "prompt": "Ollama Cloud API key",
        "url": "https://ollama.com/settings",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_BASE_URL": {
        "description": "Ollama Cloud base URL override (default: https://ollama.com/v1)",
        "prompt": "Ollama base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XIAOMI_API_KEY": {
        "description": "Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, mimo-v2-omni, mimo-v2-flash)",
        "prompt": "Xiaomi MiMo API Key",
        "url": "https://platform.xiaomimimo.com",
        "password": True,
        "category": "provider",
    },
    "XIAOMI_BASE_URL": {
        "description": "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)",
        "prompt": "Xiaomi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "UPSTAGE_API_KEY": {
        "description": "Upstage API key for Solar LLM models",
        "prompt": "Upstage API Key",
        "url": "https://console.upstage.ai/api-keys",
        "password": True,
        "category": "provider",
    },
    "UPSTAGE_BASE_URL": {
        "description": "Upstage base URL override (default: https://api.upstage.ai/v1)",
        "prompt": "Upstage base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_REGION": {
        "description": "AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)",
        "prompt": "AWS Region",
        "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_PROFILE": {
        "description": "AWS named profile for Bedrock authentication (from ~/.aws/credentials)",
        "prompt": "AWS Profile",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AZURE_FOUNDRY_API_KEY": {
        "description": "Azure Foundry API key for custom Azure endpoints",
        "prompt": "Azure Foundry API Key",
        "url": "https://ai.azure.com/",
        "password": True,
        "category": "provider",
    },
    "AZURE_FOUNDRY_BASE_URL": {
        "description": "Azure Foundry base URL (set via 'hermes model' for endpoint-specific config)",
        "prompt": "Azure Foundry base URL",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    # ── Tool API keys ──
    "EXA_API_KEY": {
        "description": "Exa API key for AI-native web search and contents",
        "prompt": "Exa API key",
        "url": "https://exa.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "PARALLEL_API_KEY": {
        "description": "Parallel API key for AI-native web search and extract",
        "prompt": "Parallel API key",
        "url": "https://parallel.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_KEY": {
        "description": "Firecrawl API key for web search and scraping",
        "prompt": "Firecrawl API key",
        "url": "https://firecrawl.dev/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_URL": {
        "description": "Firecrawl API URL for self-hosted instances (optional)",
        "prompt": "Firecrawl API URL (leave empty for cloud)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "FIRECRAWL_GATEWAY_URL": {
        "description": "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)",
        "prompt": "Firecrawl gateway URL (leave empty to derive from domain)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_DOMAIN": {
        "description": "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com",
        "prompt": "Tool-gateway domain suffix",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_SCHEME": {
        "description": "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts (`https` by default, set `http` for local gateway testing)",
        "prompt": "Tool-gateway URL scheme",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_USER_TOKEN": {
        "description": "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise read from the Hermes auth store)",
        "prompt": "Tool-gateway user token",
        "url": None,
        "password": True,
        "category": "tool",
        "advanced": True,
    },
    "TAVILY_API_KEY": {
        "description": "Tavily API key for AI-native web search and extract (optional — keyless works without it)",
        "prompt": "Tavily API key",
        "url": "https://app.tavily.com/home",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "KEENABLE_API_KEY": {
        "description": "Keenable API key for fast independent-index web search and page fetch (optional — keyless free tier works without it)",
        "prompt": "Keenable API key",
        "url": "https://keenable.ai",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "SEARXNG_URL": {
        "description": "URL of your SearXNG instance for free self-hosted web search",
        "prompt": "SearXNG URL (e.g. http://localhost:8080)",
        "url": "https://searxng.github.io/searxng/",
        "tools": ["web_search"],
        "password": False,
        "category": "tool",
    },
    "BRAVE_SEARCH_API_KEY": {
        "description": "Brave Search API subscription token (free tier: 2,000 queries/mo)",
        "prompt": "Brave Search subscription token",
        "url": "https://brave.com/search/api/",
        "tools": ["web_search"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_API_KEY": {
        "description": "Browserbase API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browserbase API key",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_PROJECT_ID": {
        "description": "Browserbase project ID (optional — only needed for cloud browser)",
        "prompt": "Browserbase project ID",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "BROWSER_USE_API_KEY": {
        "description": "Browser Use API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browser Use API key",
        "url": "https://browser-use.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_BROWSER_TTL": {
        "description": "Firecrawl browser session TTL in seconds (optional, default 300)",
        "prompt": "Browser session TTL (seconds)",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "AGENT_BROWSER_ENGINE": {
        "description": "Browser engine for local mode: auto (default Chrome), lightpanda (faster, no screenshots), chrome",
        "prompt": "Browser engine (auto/lightpanda/chrome)",
        "url": "https://github.com/vercel-labs/agent-browser",
        "tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_vision"],
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "CAMOFOX_URL": {
        "description": "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)",
        "prompt": "Camofox server URL",
        "url": "https://github.com/jo-inc/camofox-browser",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "CAMOFOX_API_KEY": {
        "description": "Optional bearer token sent as Authorization header to a remote/authenticated Camofox server",
        "prompt": "Camofox API key",
        "url": "https://github.com/jo-inc/camofox-browser",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
        "advanced": True,
    },
    "FAL_KEY": {
        "description": "FAL API key for image and video generation",
        "prompt": "FAL API key",
        "url": "https://fal.ai/",
        "tools": ["image_generate", "video_generate"],
        "password": True,
        "category": "tool",
    },
    "KREA_API_KEY": {
        "description": "Krea API key for Krea 2 image generation (Medium + Large)",
        "prompt": "Krea API key",
        "url": "https://www.krea.ai/settings/api-tokens",
        "tools": ["image_generate"],
        "password": True,
        "category": "tool",
    },
    "VOICE_TOOLS_OPENAI_KEY": {
        "description": "OpenAI API key for voice transcription (Whisper) and OpenAI TTS",
        "prompt": "OpenAI API Key (for Whisper STT + TTS)",
        "url": "https://platform.openai.com/api-keys",
        "tools": ["voice_transcription", "openai_tts"],
        "password": True,
        "category": "tool",
    },
    "ELEVENLABS_API_KEY": {
        "description": "ElevenLabs API key for premium text-to-speech voices and Scribe transcription",
        "prompt": "ElevenLabs API key",
        "url": "https://elevenlabs.io/",
        "tools": ["elevenlabs_tts", "voice_transcription"],
        "password": True,
        "category": "tool",
    },
    "MISTRAL_API_KEY": {
        "description": "Mistral API key for Voxtral TTS and transcription (STT)",
        "prompt": "Mistral API key",
        "url": "https://console.mistral.ai/",
        "password": True,
        "category": "tool",
    },
    "PORCUPINE_ACCESS_KEY": {
        "description": "Picovoice access key for the Porcupine 'Hey Hermes' wake word engine (optional; openWakeWord is the free default)",
        "prompt": "Picovoice access key",
        "url": "https://console.picovoice.ai/",
        "password": True,
        "category": "tool",
    },
    "GITHUB_TOKEN": {
        "description": "GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "prompt": "GitHub Token",
        "url": "https://github.com/settings/tokens",
        "password": True,
        "category": "tool",
    },

    # ── Bundled skills (opt-in: only needed if the user uses that skill) ──
    # These use category="skill" (distinct from "tool") so the sandbox
    # env blocklist in tools/environments/local.py does NOT rewrite them —
    # skills legitimately need these passed through to curl via
    # tools/env_passthrough.py when the user's skill calls out.
    "NOTION_API_KEY": {
        "description": "Notion integration token (used by the `notion` skill)",
        "prompt": "Notion API key",
        "url": "https://www.notion.so/my-integrations",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "LINEAR_API_KEY": {
        "description": "Linear personal API key (used by the `linear` skill)",
        "prompt": "Linear API key",
        "url": "https://linear.app/settings/account/security",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "AIRTABLE_API_KEY": {
        "description": "Airtable personal access token (used by the `airtable` skill)",
        "prompt": "Airtable API key",
        "url": "https://airtable.com/create/tokens",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "TENOR_API_KEY": {
        "description": "Tenor API key for GIF search (used by the `gif-search` skill)",
        "prompt": "Tenor API key",
        "url": "https://developers.google.com/tenor/guides/quickstart",
        "password": True,
        "category": "skill",
        "advanced": True,
    },

    # ── Honcho ──
    "HONCHO_API_KEY": {
        "description": "Honcho API key for AI-native persistent memory",
        "prompt": "Honcho API key",
        "url": "https://app.honcho.dev",
        "tools": ["honcho_context"],
        "password": True,
        "category": "tool",
    },
    "HONCHO_BASE_URL": {
        "description": "Base URL for self-hosted Honcho instances (no API key needed)",
        "prompt": "Honcho base URL (e.g. http://localhost:8000)",
        "category": "tool",
    },

    # ── Hindsight ──
    "HINDSIGHT_API_KEY": {
        "description": "Hindsight API key for graph-aware persistent memory",
        "prompt": "Hindsight API key",
        "url": "https://hindsight.vectorize.io",
        "tools": ["hindsight_recall"],
        "password": True,
        "category": "tool",
    },
    "HINDSIGHT_API_URL": {
        "description": "Base URL for the Hindsight API (default: https://api.hindsight.vectorize.io)",
        "prompt": "Hindsight API URL",
        "category": "tool",
        "advanced": True,
    },

    # ── Supermemory ──
    "SUPERMEMORY_API_KEY": {
        "description": "Supermemory API key for conversation-scoped persistent memory",
        "prompt": "Supermemory API key",
        "url": "https://supermemory.ai",
        "tools": ["supermemory_search"],
        "password": True,
        "category": "tool",
    },

    # ── Mem0 ──
    "MEM0_API_KEY": {
        "description": "Mem0 Platform API key for semantic persistent memory",
        "prompt": "Mem0 API key",
        "url": "https://app.mem0.ai",
        "tools": ["mem0_search"],
        "password": True,
        "category": "tool",
    },

    # ── RetainDB ──
    "RETAINDB_API_KEY": {
        "description": "RetainDB API key for persistent memory",
        "prompt": "RetainDB API key",
        "url": "https://retaindb.com",
        "tools": ["retaindb_search"],
        "password": True,
        "category": "tool",
    },
    "RETAINDB_BASE_URL": {
        "description": "Base URL for self-hosted RetainDB instances (default: https://api.retaindb.com)",
        "prompt": "RetainDB base URL",
        "category": "tool",
        "advanced": True,
    },

    # ── ByteRover ──
    "BRV_API_KEY": {
        "description": "ByteRover API key (optional, for cloud sync — local-first by default)",
        "prompt": "ByteRover API key",
        "url": "https://app.byterover.dev",
        "tools": ["brv_query"],
        "password": True,
        "category": "tool",
    },

    # ── OpenViking ──
    "OPENVIKING_API_KEY": {
        "description": "OpenViking API key (leave blank for local dev mode)",
        "prompt": "OpenViking API key",
        "tools": ["viking_search"],
        "password": True,
        "category": "tool",
    },
    "OPENVIKING_ENDPOINT": {
        "description": "OpenViking server URL (default: http://127.0.0.1:1933)",
        "prompt": "OpenViking endpoint",
        "category": "tool",
        "advanced": True,
    },

    # ── Langfuse observability ──
    "HERMES_LANGFUSE_PUBLIC_KEY": {
        "description": "Langfuse project public key (pk-lf-...)",
        "prompt": "Langfuse public key",
        "url": "https://cloud.langfuse.com",
        "password": False,
        "category": "tool",
    },
    "HERMES_LANGFUSE_SECRET_KEY": {
        "description": "Langfuse project secret key (sk-lf-...)",
        "prompt": "Langfuse secret key",
        "url": "https://cloud.langfuse.com",
        "password": True,
        "category": "tool",
    },
    "HERMES_LANGFUSE_BASE_URL": {
        "description": "Langfuse server URL (default: https://cloud.langfuse.com)",
        "prompt": "Langfuse server URL (leave empty for cloud.langfuse.com)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },

    # ── Messaging platforms ──
    "TELEGRAM_BOT_TOKEN": {
        "description": "Complete Telegram bot token created by @BotFather (numeric bot ID followed by a colon and secret)",
        "prompt": "Telegram bot token",
        "url": "https://t.me/BotFather",
        "password": True,
        "category": "messaging",
    },
    "TELEGRAM_ALLOWED_USERS": {
        "description": "Optional comma-separated numeric Telegram user IDs allowed immediately; leave blank to approve new users through DM pairing",
        "prompt": "Allowed Telegram user IDs (comma-separated)",
        "url": "https://t.me/userinfobot",
        "password": False,
        "category": "messaging",
    },
    "TELEGRAM_PROXY": {
        "description": "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, https://, socks5://",
        "prompt": "Telegram proxy URL (optional)",
        "password": False,
        "category": "messaging",
    },
    "DISCORD_BOT_TOKEN": {
        "description": "Discord bot token from Developer Portal",
        "prompt": "Discord bot token",
        "url": "https://discord.com/developers/applications",
        "password": True,
        "category": "messaging",
    },
    "DISCORD_ALLOWED_USERS": {
        "description": "Comma-separated Discord user IDs allowed to use the bot",
        "prompt": "Allowed Discord user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "DISCORD_REPLY_TO_MODE": {
        "description": "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first message only, default), 'all' (reply on every chunk)",
        "prompt": "Discord reply mode (off/first/all)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "SLACK_BOT_TOKEN": {
        "description": "Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. "
                       "Required scopes: chat:write, app_mentions:read, channels:history, groups:history, "
                       "im:history, im:read, im:write, mpim:history, mpim:read, users:read, files:read, files:write",
        "prompt": "Slack Bot Token (xoxb-...)",
        "help": "In your Slack app, add the required bot scopes, install the app to the workspace, then copy OAuth & Permissions > Bot User OAuth Token.",
        "url": "https://api.slack.com/apps",
        "password": True,
        "category": "messaging",
    },
    "SLACK_APP_TOKEN": {
        "description": "Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → "
                       "App-Level Tokens. Also ensure Event Subscriptions include: message.im, "
                       "message.channels, message.groups, message.mpim, app_mention",
        "prompt": "Slack App Token (xapp-...)",
        "help": "In your Slack app, enable Socket Mode, then create Basic Information > App-Level Tokens with the connections:write scope.",
        "url": "https://api.slack.com/apps",
        "password": True,
        "category": "messaging",
    },
    "SLACK_ALLOWED_USERS": {
        "description": "Comma-separated Slack member IDs allowed to use Hermes, e.g. U01ABC2DEF3. Without this, Slack may connect but deny messages by default.",
        "prompt": "Allowed Slack member IDs",
        "help": "In Slack, open your profile, choose More or the three-dot menu, then Copy member ID. Add multiple IDs comma-separated.",
        "url": "https://api.slack.com/apps",
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_URL": {
        "description": "Mattermost server URL (e.g. https://mm.example.com)",
        "prompt": "Mattermost server URL",
        "url": "https://mattermost.com/deploy/",
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_TOKEN": {
        "description": "Mattermost bot token or personal access token",
        "prompt": "Mattermost bot token",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "MATTERMOST_ALLOWED_USERS": {
        "description": "Comma-separated Mattermost user IDs allowed to use the bot",
        "prompt": "Allowed Mattermost user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_REQUIRE_MENTION": {
        "description": "Require @mention in Mattermost channels (default: true). Set to false to respond to all messages.",
        "prompt": "Require @mention in channels",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATTERMOST_FREE_RESPONSE_CHANNELS": {
        "description": "Comma-separated Mattermost channel IDs where bot responds without @mention",
        "prompt": "Free-response channel IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_HOMESERVER": {
        "description": "Matrix homeserver URL (e.g. https://matrix.example.org)",
        "prompt": "Matrix homeserver URL",
        "url": "https://matrix.org/ecosystem/servers/",
        "password": False,
        "category": "messaging",
    },
    "MATRIX_ACCESS_TOKEN": {
        "description": "Matrix access token (preferred over password login)",
        "prompt": "Matrix access token",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "MATRIX_USER_ID": {
        "description": "Matrix user ID (e.g. @hermes:example.org)",
        "prompt": "Matrix user ID (@user:server)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_ALLOWED_USERS": {
        "description": "Comma-separated Matrix user IDs allowed to use the bot (@user:server format)",
        "prompt": "Allowed Matrix user IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "MATRIX_REQUIRE_MENTION": {
        "description": "Require @mention in Matrix rooms (default: true). Set to false to respond to all messages.",
        "prompt": "Require @mention in rooms (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_FREE_RESPONSE_ROOMS": {
        "description": "Comma-separated Matrix room IDs where bot responds without @mention",
        "prompt": "Free-response room IDs (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_AUTO_THREAD": {
        "description": "Auto-create threads for messages in Matrix rooms (default: true)",
        "prompt": "Auto-create threads in rooms (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_DM_AUTO_THREAD": {
        "description": "Auto-create threads for DM messages in Matrix (default: false)",
        "prompt": "Auto-create threads in DMs (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_DEVICE_ID": {
        "description": "Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)",
        "prompt": "Matrix device ID (stable across restarts)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "MATRIX_RECOVERY_KEY": {
        "description": "Matrix recovery key for cross-signing verification after device key rotation (from Element: Settings → Security → Recovery Key)",
        "prompt": "Matrix recovery key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "BLUEBUBBLES_SERVER_URL": {
        "description": "BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)",
        "prompt": "BlueBubbles server URL",
        "url": "https://bluebubbles.app/",
        "password": False,
        "category": "messaging",
    },
    "BLUEBUBBLES_PASSWORD": {
        "description": "BlueBubbles server password (from BlueBubbles Server → Settings → API)",
        "prompt": "BlueBubbles server password",
        "url": None,
        "password": True,
        "category": "messaging",
    },
    "BLUEBUBBLES_ALLOWED_USERS": {
        "description": "Comma-separated iMessage addresses (email or phone) allowed to use the bot",
        "prompt": "Allowed iMessage addresses (comma-separated)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "BLUEBUBBLES_ALLOW_ALL_USERS": {
        "description": "Allow all BlueBubbles users without allowlist",
        "prompt": "Allow All BlueBubbles Users",
        "category": "messaging",
    },
    "QQ_APP_ID": {
        "description": "QQ Bot App ID from QQ Open Platform (q.qq.com)",
        "prompt": "QQ App ID",
        "url": "https://q.qq.com",
        "category": "messaging",
    },
    "QQ_CLIENT_SECRET": {
        "description": "QQ Bot Client Secret from QQ Open Platform",
        "prompt": "QQ Client Secret",
        "password": True,
        "category": "messaging",
    },
    "QQ_ALLOWED_USERS": {
        "description": "Comma-separated QQ user IDs allowed to use the bot",
        "prompt": "QQ Allowed Users",
        "category": "messaging",
    },
    "QQ_GROUP_ALLOWED_USERS": {
        "description": "Comma-separated QQ group IDs allowed to interact with the bot",
        "prompt": "QQ Group Allowed Users",
        "category": "messaging",
    },
    "QQ_ALLOW_ALL_USERS": {
        "description": "Allow all QQ users without an allowlist (true/false)",
        "prompt": "Allow All QQ Users",
        "category": "messaging",
    },
    "QQBOT_HOME_CHANNEL": {
        "description": "Default QQ channel/group for cron delivery and notifications",
        "prompt": "QQ Home Channel",
        "category": "messaging",
    },
    "QQBOT_HOME_CHANNEL_NAME": {
        "description": "Display name for the QQ home channel",
        "prompt": "QQ Home Channel Name",
        "category": "messaging",
    },
    "QQ_SANDBOX": {
        "description": "Enable QQ sandbox mode for development testing (true/false)",
        "prompt": "QQ Sandbox Mode",
        "category": "messaging",
    },
    "IRC_SERVER": {
        "description": "IRC server hostname (e.g. irc.libera.chat)",
        "prompt": "IRC server",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "IRC_CHANNEL": {
        "description": "IRC channel to join (e.g. #hermes)",
        "prompt": "IRC channel",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "IRC_NICKNAME": {
        "description": "Bot nickname on IRC (default: hermes-bot)",
        "prompt": "IRC nickname",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "IRC_SERVER_PASSWORD": {
        "description": "IRC server password (if required)",
        "prompt": "IRC server password",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "IRC_NICKSERV_PASSWORD": {
        "description": "NickServ password for nick identification",
        "prompt": "NickServ password",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_ALLOW_ALL_USERS": {
        "description": "Allow all users to interact with messaging bots (true/false). Default: false.",
        "prompt": "Allow all users (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_ENABLED": {
        "description": "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open WebUI, LobeChat, etc. to connect.",
        "prompt": "Enable API server (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_KEY": {
        "description": "Bearer token for API server authentication. Required whenever the API server is enabled; server refuses to start without it.",
        "prompt": "API server auth key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_PORT": {
        "description": "Port for the API server (default: 8642).",
        "prompt": "API server port",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_HOST": {
        "description": "Host/bind address for the API server (default: 127.0.0.1). API_SERVER_KEY is still required even on loopback binds.",
        "prompt": "API server host",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_MODEL_NAME": {
        "description": "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' for the default profile). Useful for multi-user setups with OpenWebUI.",
        "prompt": "API server model name",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_URL": {
        "description": "URL of a remote Hermes API server to forward messages to (proxy mode). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Use for Docker E2EE containers that relay to a host agent. Also configurable via gateway.proxy_url in config.yaml.",
        "prompt": "Remote Hermes API server URL (e.g. http://192.168.1.100:8642)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_KEY": {
        "description": "Bearer token for authenticating with the remote Hermes API server (proxy mode). Must match the API_SERVER_KEY on the remote host.",
        "prompt": "Remote API server auth key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "WEBHOOK_ENABLED": {
        "description": "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.",
        "prompt": "Enable webhooks (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_PORT": {
        "description": "Port for the webhook HTTP server (default: 8644).",
        "prompt": "Webhook port",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_SECRET": {
        "description": "Global HMAC secret for webhook signature validation (overridable per route in config.yaml).",
        "prompt": "Webhook secret",
        "url": None,
        "password": True,
        "category": "messaging",
    },

    # ── Agent settings ──
    # NOTE: MESSAGING_CWD was removed here — use terminal.cwd in config.yaml
    # instead.  The gateway reads TERMINAL_CWD (bridged from terminal.cwd).
    "SUDO_PASSWORD": {
        "description": "Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting",
        "prompt": "Sudo password",
        "url": None,
        "password": True,
        "category": "setting",
    },
    # HERMES_TOOL_PROGRESS_MODE is deprecated — tool progress is configured via
    # display.tool_progress in config.yaml (off|new|all|verbose|log). The
    # gateway still falls back to HERMES_TOOL_PROGRESS_MODE for backward
    # compatibility, so it lives in _EXTRA_ENV_KEYS (known to reload and
    # compatibility paths) but is intentionally NOT listed here:
    # OPTIONAL_ENV_VARS feeds user-facing surfaces (dashboard keys page, setup
    # checklists) and deprecated knobs shouldn't be offered there. The boolean
    # HERMES_TOOL_PROGRESS is fully unsupported since the v12 config support
    # floor retired its only consumer (the v3→4 migration).
    "HERMES_PREFILL_MESSAGES_FILE": {
        "description": "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "prompt": "Prefill messages file path",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "HERMES_EPHEMERAL_SYSTEM_PROMPT": {
        "description": "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "prompt": "Ephemeral system prompt",
        "url": None,
        "password": False,
        "category": "setting",
    },
}
