"""Default configuration data for Hermes Agent: DEFAULT_CONFIG and OPTIONAL_ENV_VARS.

Pure-data leaf module — must not import from hermes_cli.config. Comments are the user-facing
docs of config.yaml.
"""


def _aux(timeout, *, reasoning_effort=True, **extra):
    """Standard auxiliary-task model block (see DEFAULT_CONFIG["auxiliary"]).

    reasoning_effort=False omits that key (MoA blocks configure depth per slot);
    ``extra`` keys are appended after the standard ones.
    """
    d = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        d["reasoning_effort"] = ""
    d.update(extra)
    return d


DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
    "credential_pool_strategies": {},
    "toolsets": ["hermes-cli"],
    # journal_mode: SQLite journal mode for every Hermes DB. "wal" default; use "delete" on
    # weak-fsync/shared filesystems where WAL is not crash-safe (macOS virtiofs, NFS, SMB).
    "database": {
        "journal_mode": "wal",
        # WAL sizing pragmas (ints). None = SQLite defaults (autocheckpoint 1000 pages, no limit).
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
    },
    # Soft fd limit for long-running server processes; clamped to OS hard limit. 0/false/null = off.
    "runtime": {"nofile_soft_limit": 4096},
    # Global active chat session cap across CLI, TUI/dashboard, and messaging. None/0 = unbounded.
    "max_concurrent_sessions": None,
    # Soft LRU cap on in-memory TUI/desktop/dashboard sessions. Above it the gateway evicts the
    # least-recently-active DETACHED sessions (no live client); reopening re-resumes from disk.
    # 0/null disables.
    "max_live_sessions": 16,
    "session": {
        # Per-terminal `hermes -c`: each CLI session writes a breadcrumb under
        # $HERMES_HOME/terminal-sessions/<terminal-id>, so bare -c/--continue resumes THIS
        # terminal's session (tmux/kitty/wezterm pane, tty). false = resume globally most-recent.
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
        # Cron-only floor under the stop()/drain wait (seconds). Interrupted chat turns resume on
        # the next message, but an interrupted cron run is recorded as a permanent failure, so it
        # must not inherit restart_drain_timeout's 0. Clamped to the shutdown-watchdog leash minus
        # teardown headroom (~50s unless TimeoutStopSec is raised). 0 = opt out.
        # A chat turn interrupted by a restart is announced to the user and resumed on their next message;
        # an interrupted cron run is written to jobs.json as a permanent failure that nobody is waiting on,
        # so it must not inherit restart_drain_timeout's 0 (#82161).
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
        # Empty-response retry guard. Empty retries re-send the full input at full price; this stops
        # re-billing deterministic empties (unsignaled refusals, zero output tokens) while failing
        # open on ambiguous evidence (missing usage, any tokens, model/provider change).
        "empty_response_guard": {
            "enabled": True,  # False = legacy fixed 3 retries unconditionally
            # When one empty attempt's estimated input cost >= this USD, the streak's retry budget
            # drops from 3 to 1. Unknown pricing / missing usage leaves it untouched.
            "cost_threshold_usd": 0.25,
        },
        # Fast mode: "" / "normal" (off), "fast" (always), "auto" (first
        # fast_auto_seconds of every turn), "cold" (first turn of a session only).
        "service_tier": "",
        "fast_auto_seconds": 60,
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
        # Max seconds the gateway blocks an agent awaiting a clarify-tool reply; then it unblocks
        # with "[user did not respond within Xm]". CLI clarify blocks indefinitely and ignores this.
        # 1h because users step away and a shorter value evicted the entry mid-think so a later
        # button tap hit a dead entry. Lower it to free the running-agent guard sooner.
        # Maximum time (seconds) the gateway will block an agent waiting for a clarify-tool response from
        # the user. Tradeoff: a higher value holds the gateway's running-agent guard longer for a genuinely
        # abandoned prompt — lower it if a single session must free up the guard sooner. See #32762.
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
        # Transcript-sanitiser repeated-heal escalation threshold (#96870).
        # After this many pre-send heal passes within a 10-minute session
        # window, log one ERROR (session id + heal pattern) and queue a
        # ONE-TIME out-of-band user notice pointing at /debug share or
        # `hermes doctor`. Delivered via the status channel only —
        # conversation context / prompt caching untouched. 0 = disable
        # escalation (per-window WARNINGs still fire).
        "sanitizer_heal_escalation_threshold": 3,
        # Long-lived reconnect-loop escalation (seconds). A platform that has
        # been continuously failing/reconnecting for this long gets
        # needs_attention flagged in gateway runtime status (visible in
        # `hermes status` / fleet monitoring). Retries never stop — this is a
        # signal, not a circuit breaker. 0 = disable.
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
        # Stale-stream ceiling (seconds) for local providers (Ollama, oMLX, llama-cpp). Applied when
        # the base stale timeout is at its 180s default and a local endpoint is detected, so a
        # wedged local server eventually trips the detector instead of hanging forever. Env
        # HERMES_LOCAL_STREAM_STALE_TIMEOUT overrides.
        "local_stream_stale_timeout": 900,
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
        # Turn liveness watchdog (#95548): a turn that shows no observable
        # progress (activity-clock idle, never touched by lease renewal) for
        # `timeout_s` seconds is logged loudly, force-interrupted so the UI
        # can retry it, and its durable turn lease stops renewing so TTL
        # expiry lets stale-turn cleanup reclaim the session even when the
        # hard interrupt cannot unwind a wedged frame. `timeout_s` <= 0
        # disables the watchdog; `poll_s` is the sampling interval. Invalid
        # values (typo, NaN, Inf, non-positive poll) warn and fall back to
        # the default instead of crashing startup or silently disabling the
        # watchdog. See agent/turn_liveness.py.
        "turn_liveness": {
            "timeout_s": 600.0,
            "poll_s": 15.0,
        },
    },

    "terminal": {
        "backend": "local",
        "modal_mode": "auto",
        # Remote-backend connection-class failures (SSH host unreachable, Docker daemon down):
        # "warn" = structured degraded tool result with reason + retry hint; "fail" = raise error +
        # traceback.
        "degraded_mode": "warn",
        "cwd": ".",  # Use current directory
        # Root directory for Hermes' terminal session temp files (background
        # logs/pid/exit files, code-execution sandboxes, etc.). When empty,
        # Hermes uses TMPDIR/TMP/TEMP if set, otherwise a managed dir on real
        # storage at HERMES_HOME/cache/terminal (auto-pruned after 72h) — NOT
        # tmpfs /tmp, which is RAM-capped and small on many distros (e.g.
        # Arch-based setups) and fills up under load. Set this to redirect
        # session temp anywhere else; must be an absolute POSIX path that
        # exists. User-set paths are never auto-pruned.
        "temp_dir": "",
        # Terminal font family for the desktop app's embedded xterm.js terminal.
        # When set (e.g. "'CaskaydiaCoveNerdFont', 'JetBrains Mono', monospace"),
        # the desktop terminal uses this as the CSS font-family value, with the
        # built-in default ("'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo,
        # Consolas, monospace") as fallback when the field is empty or unset.
        # This lets users install a Nerd Font (or any custom font) and configure
        # it here without patching the built desktop app.
        "font_family": "",
        "timeout": 180,
        # Seconds between SIGTERM and escalated SIGKILL for host process trees (browser daemons). 0
        # = SIGTERM only.
        "daemon_term_grace_seconds": 2.0,
        # Bounded linger (seconds) for one-shot CLI runs (-q/-Q/-z) that exit
        # while background processes spawned with notify_on_complete=true are
        # still running. The dying parent owns those children's stdout pipes,
        # so exiting immediately kills the delivery a few seconds later —
        # destroying Bot Mode handoff replies dispatched via message_agent /
        # bot_relay from a short-lived `hermes -p <bot> chat -Q` recipient
        # (#90879). The parent instead waits (up to this bound) for tracked
        # notify_on_complete processes to finish before exiting. Plain
        # background processes without notify_on_complete (servers, daemons)
        # are never waited on. 0 disables the linger.
        "oneshot_completion_wait_seconds": 600.0,
        # Environment variables to pass through to sandboxed execution
        # (terminal and execute_code).  Skill-declared required_environment_variables
        # are passed through automatically; this list is for non-skill use cases.
        "env_passthrough": [],
        # HOME for host tool subprocesses: "auto" = host keeps the real OS-user HOME, containers use
        # HERMES_HOME/home; "real" = force real HOME; "profile" = force HERMES_HOME/home when it
        # exists (strict per-profile isolation).
        "home_mode": "auto",
        # Extra files sourced in the login shell when building the per-session env snapshot — for
        # nvm/pyenv/asdf/PATH entries registered by files a bash login shell skips (~/.bashrc,
        # ~/.zshrc, ~/.zprofile). Supports ~ and ${VAR}; missing files skipped. When empty and the
        # shell is bash, ~/.profile, ~/.bash_profile, ~/.bashrc are auto-sourced in that order (see
        # auto_source_bashrc).
        "shell_init_files": [],
        # Source ~/.profile, ~/.bash_profile, ~/.bashrc in the snapshot login shell to capture PATH
        # additions, functions, and aliases that `bash -l -c` misses (bash skips bashrc when
        # non-interactive; Debian/Ubuntu ~/.bashrc short-circuits). ~/.profile and ~/.bash_profile
        # go first because n/nvm/asdf write PATH exports there without an interactivity guard. Turn
        # off if an rc file misbehaves when sourced non-interactively (exits on TTY check).
        "auto_source_bashrc": True,
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "docker_forward_env": [],
        # Exact key-value env pairs set inside Docker containers (unlike docker_forward_env, which
        # reads host values) — useful under systemd without the user's shell env. Example:
        # {"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock"}
        "docker_env": {},
        "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "vercel_runtime": "node24",  # vercel_sandbox backend only: node24 | node22 | python3.13
        # Container limits (docker, singularity, modal, daytona, vercel_sandbox; not local/ssh).
        "container_cpu": 1,
        "container_memory": 5120,       # MB (default 5GB)
        "container_disk": 51200,        # MB (default 50GB)
        "container_persistent": True,   # Persist filesystem across sessions
        # Docker volume mounts, "host_path:container_path" (docker -v syntax), e.g.
        # ["/home/user/.hermes/cache/documents:/output"]. For gateway MEDIA delivery, write to
        # /output/... inside Docker and emit the host-visible path in MEDIA:, not the container one.
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,  # mount host cwd at /workspace (weakens isolation)
        "docker_network": True,  # false = --network=none, no network access from commands
        "docker_extra_args": [],        # Extra flags passed verbatim to docker run
        # /dev/shm size for the Docker sandbox. Docker's 64 MB default silently breaks
        # Chromium/Playwright and PyTorch DataLoader workers; tmpfs is lazily allocated so the
        # higher ceiling is free until used. "" or "0" = omit the flag (Docker default).
        "docker_shm_size": "1g",
        # Run the container as the host uid:gid (`--user`) so files written to bind mounts
        # (docker_volumes, persistent workspace, mounted cwd) are owned by you, not root. Off by
        # default for images whose entrypoints must start as root (e.g. the bundled Hermes image,
        # which drops to `hermes` via s6-setuidgid). When on, SETUID/SETGID caps are omitted.
        "docker_run_as_host_user": False,
        # Explicit opt-in for trusted profiles to reuse the same Docker
        # container identity. Empty preserves the active-profile boundary.
        "docker_shared_container_key": "",
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
        # web_search/web_extract rotate round-robin across four vendors'
        # public free tiers (exa, parallel, firecrawl, keenable),
        # failing over to the next ring vendor on rate limits. Never
        # pre-empts a configured or keyed backend. Set false to disable.
        "keyless_fallback": True,
        # One-shot keyless rescue: when the chosen/keyed backend fails a
        # web_search/web_extract call, THAT call retries once on the keyless
        # free-tier ring — the next call attempts the chosen backend again
        # (no sticky failover). Off when keyless_fallback is false.
        "keyless_rescue": True,
        # Per-provider tier selection for vendors with both a keyless
        # free endpoint and a keyed paid path (exa, parallel,
        # firecrawl, keenable on the ring; tavily is opt-in keyless via
        # `hermes tools`, not a ring member). Set by the `hermes tools`
        # picker's "Free (keyless)" / "Paid (API key)" rows.
        #   free  — always use the anonymous free endpoint (even with a key)
        #   paid  — always use the keyed path (missing key = error; vendor
        #           is also excluded from the keyless ring)
        #   unset — auto: keyed when the API key is present, else the ring
        "provider_tier": {},
        # TTL result caching for web_search + web_extract. Repeat searches
        # (same query, same provider) within the TTL are served from an
        # in-process memo; repeat extracts of the same URL are served from
        # the cache/web full-text store. Concurrent identical searches
        # (parallel subagents) coalesce into one vendor request. Only
        # successful responses are cached.
        "cache_enabled": True,
        "cache_ttl_minutes": 20,
        # Hosts whose pages must always be fetched live, never from the
        # extract cache — sites you're actively developing but testing over
        # the public internet (staging deploys, tunnel URLs, preview
        # builds). Entries match exactly, as "*.wildcard", or as a domain
        # suffix ("mysite.dev" also covers "preview.mysite.dev").
        # localhost/private-IP URLs are always exempt automatically.
        #   cache_exempt_hosts: ["mysite.vercel.app", "*.ngrok-free.app"]
        "cache_exempt_hosts": [],
    },

    "browser": {
        # "" = Browser Use mode when the browser-use CLI (or uvx) is available, else built-in tools
        # (Camofox setups always keep built-in tools: no CDP surface); "browser-use" = force one
        # browser_exec tool driving the Browser Use CLI over any CDP backend (local Chrome, cloud);
        # "off" = force the built-in browser_navigate/browser_click/... tools.
        "backend": "",
        "inactivity_timeout": 120,
        "command_timeout": 30,  # Timeout for browser commands in seconds (screenshot, navigate, etc.)
        "snapshot_threshold": 15000,  # Max chars before snapshot truncate-and-store (min 1000)
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
        # Consent to browse with the user's REAL logins for local browsing.
        # When true, local browsing (the Browser Use CLI, or the built-in
        # browser tools) runs on a Hermes-managed SNAPSHOT of the user's
        # ACTIVE default-Chromium profile (Local State -> profile.last_used) —
        # its cookies, logins and preferences copied in and re-synced when a
        # fresh session launches — driven by Hermes' packaged Chromium. Only
        # the active profile is copied. The snapshot is a non-default dir, so it
        # sidesteps Chrome 136+'s block on debugging the default profile and
        # never contends with the user's running browser. Turning this back off
        # deletes the snapshot store (~/.hermes/browser-profile/) so copied
        # credentials don't outlive consent. Only Chromium-family default
        # browsers are supported (Chrome, Edge, Brave, Brave Origin, Chromium); a non-Chromium
        # default (e.g. Firefox) fails closed with a clear message. Default
        # false. Also gates the browser_exec ``local`` argument, which forces a
        # real-profile local session even under a cloud browser backend. Toggle
        # in the desktop Settings → Browser section.
        "use_real_profile": False,
        # When real-profile browsing needs the browser closed (Windows: a
        # running Chrome/Edge/Brave locks its cookie DB deny-all, so it must be
        # fully quit before its profile can be copied), arm the "offer to close
        # it" flow. This does NOT auto-kill: when the profile is locked the
        # snapshot always blocks and the agent asks the user first; only on
        # approval does it run `hermes browser close-profile` (terminates the
        # browser process tree bound to that profile, losing unsaved tabs) and
        # retry. Still locked afterward → stays blocked, no loop, no auto-kill.
        # OFF by default. No effect on macOS/Linux (copy-while-running works).
        "real_profile_autoclose": False,
        # Pin WHICH source browser profile directory gets snapshotted for
        # real-profile browsing (e.g. "Profile 2"). Unset/empty: follows the
        # browser's last-used profile (Local State → profile.last_used). On a
        # machine with several profiles (work + personal), last-used roulette
        # can silently hand the agent the wrong identity; a pin locks it. A pin
        # naming a directory that doesn't exist FAILS CLOSED with a fixable
        # message rather than falling back to last-used.
        "real_profile_pin": "",
        "allow_unsafe_evaluate": False,  # Legacy override: when true, browser_console(expression=...) bypasses the restrict_evaluate denylist entirely
        "restrict_evaluate": False,  # Opt-in denylist blocking sensitive JS primitives (cookies/storage/clipboard/network/form values) in browser_console(expression=...)
        # CDP supervisor — dialog + frame detection via a persistent WebSocket.
        # Active only when a CDP-capable backend is attached (Browserbase or
        # local Chrome via /browser connect). See
        # website/docs/developer-guide/browser-supervisor.md.
        "dialog_policy": "must_respond",  # must_respond | auto_dismiss | auto_accept
        "dialog_timeout_s": 300,  # safety auto-dismiss after N seconds under must_respond
        "camofox": {
            # true = send a stable profile-scoped userId so Camofox maps it to a persistent Firefox
            # profile; false = random ephemeral userId per session.
            "managed_persistence": False,
            # Externally managed Camofox identity, for when another app owns the visible browser.
            "user_id": "",
            "session_key": "",
            "adopt_existing_tab": False,  # rehydrate tab_id from Camofox before creating a tab
            # Docker Camofox opens page URLs from inside the container: rewrite loopback page URLs
            # (localhost/127.0.0.1/::1) to the host alias; CAMOFOX_URL itself is unchanged.
            "rewrite_loopback_urls": False,
            "loopback_host_alias": "host.docker.internal",
        },
        # Authenticated browser-extension controller lane. When enabled, an
        # extension that registers through the gateway can become the exact
        # controller for a session's browser_* tools (fail-closed once bound).
        # Local API registration additionally requires the API server bearer
        # key. developer_mode gates the privileged capabilities
        # (browser_cdp / browser_evaluate) — never negotiable without it.
        "extension_control": {
            "enabled": False,
            "developer_mode": False,
        },
    },
    # Filesystem checkpoints: snapshot the working directory once per turn (on the first
    # write_file/patch call); restore with /rollback. Opt-in via `hermes chat --checkpoints` or
    # enabled=True (most users never use /rollback). Single shared shadow store with real pruning.
    "checkpoints": {
        "enabled": False,
        # Max checkpoints per working directory; enforced by ref rewrite + GC of older commits.
        "max_snapshots": 20,
        # Hard ceiling on total ~/.hermes/checkpoints/ size (MB); the oldest checkpoint per project
        # is dropped round-robin until under the cap. 0 disables.
        "max_total_size_mb": 500,
        # Skip files larger than this (MB) when staging (datasets, model weights). 0 = no filter.
        "max_file_size_mb": 10,
        # Startup sweep (at most once per min_interval_hours): deletes projects whose last_touch is
        # older than retention_days, GCs the shared store, enforces max_total_size_mb, deletes
        # legacy-* archives older than retention_days. It NEVER deletes orphans (workdir missing on
        # disk) — a missing workdir may just be an unmounted volume/VPN, and an unattended sweep
        # must not guess. Orphans: `hermes checkpoints prune` (`--keep-orphans` to skip).
        "auto_prune": True,
        "retention_days": 7,
        "min_interval_hours": 24,
    },
    # Hard cap (chars) for one auto-loaded context file (SOUL.md, AGENTS.md, CLAUDE.md, .hermes.md,
    # .cursorrules) before head/tail truncation. null = scale with the model's context window (floor
    # 20K, ceiling 500K); a positive int pins a fixed cap. Separate from read_file limits.
    "context_file_max_chars": None,

    # Seconds to wait for a single context file read before skipping it with a
    # warning. Guards startup against network-backed filesystems (iCloud Drive,
    # OneDrive, NFS) that can block a cold read on an evicted file.
    "context_file_read_timeout": 5.0,

    # Maximum characters returned by a single read_file call.  Reads that
    # exceed this are rejected with guidance to use offset+limit.
    # 100K chars ≈ 25–35K tokens across typical tokenisers.
    "file_read_max_chars": 100_000,
    # Seconds the first agent build waits for background MCP discovery before snapshotting its tool
    # list. Returns the instant discovery completes (no MCP servers → ~0s); the bound only bites
    # when a server is still connecting. Turn-1 latency knob only: a server that misses it is picked
    # up by the between-turns refresh (agent/turn_context.py), so keep it small — a dead server adds
    # this much to first-response latency.
    "mcp_discovery_timeout": 1.5,
    # Same bound for single-query mode (``hermes -q/-z``). With only ONE turn there is no
    # between-turns refresh, so a server that misses the window is invisible for the whole session;
    # the larger bound lets slow cold-start servers (npx, uvx, remote HTTP) land. Reachable servers
    # still only wait their real handshake time.
    "mcp_single_query_discovery_timeout": 15.0,
    "mcp": {  # MCP runtime behavior (distinct from mcp_servers: definitions and auxiliary.mcp).
        # Auto-reload MCP connections when config.yaml's mcp_servers changes (CLI watcher). Every
        # reload rebuilds the tool surface and INVALIDATES the provider prompt cache (next message
        # re-sends the full prefix) — costly on long-context models. When false the watcher still
        # detects the change and prints /reload-mcp guidance.
        "auto_reload_on_config_change": True,
    },
    # Tool-output truncation. max_bytes: terminal_tool output cap in chars (head+tail kept; 50_000 ≈
    # 12-15K tokens). max_lines: max `limit` one read_file call may request before clamping.
    # max_line_length: per-line cap in read_file's line-numbered view (chars).
    "tool_output": {"max_bytes": 50000, "max_lines": 2000, "max_line_length": 2000},
    # Tool loop guardrails nudge models that repeat failed/non-progressing tool calls. Soft warnings
    # are always on; hard stops are opt-in so interactive sessions keep flowing.
    "tool_loop_guardrails": {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
        # Unattended gateway/cron platforms get hard stops by default (nobody
        # is present to /stop a model that ignores loop warnings); interactive
        # cli/tui/desktop/acp stay warning-only unless hard_stop_enabled.
        "non_interactive_hard_stop_enabled": True,
        "warn_after": {
            "exact_failure": 2,
            "same_tool_failure": 3,
            "idempotent_no_progress": 2,
        },
        "hard_stop_after": {
            "exact_failure": 5, "same_tool_failure": 8, "idempotent_no_progress": 5
        },
        # Per-turn hard ceilings for runaway-prone tools; counters reset every turn, always on
        # regardless of the thresholds above. Dozens of searches/subagents in ONE turn is already
        # pathological, hence low defaults. 0 = unlimited.
        "loop_caps": {
            "max_web_searches": 50,   # web_search calls per turn
            "max_subagents": 50,      # subagents spawned per turn
        },
    },

    "compression": {
        "enabled": True,
        "checkpoint_required": False, # Fail closed before lossy compaction unless an
                                      # active memory provider confirms checkpoint API
                                      # compatibility and completes the checkpoint.
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
        # tail_mode: "lean" = clamped 2.5%-of-window tail (10K floor / 25K cap) plus chunked
        # digests, anchor index, verbatim user messages and session_search pointers in the summary
        # (~3x fewer retained tokens; a few extra summarizer calls at the boundary). "legacy" =
        # 0.20×threshold verbatim tail (100-240K tokens on big windows).
        "tail_mode": "lean",
        "protect_last_n": 20,         # minimum recent messages kept uncompressed
        # min_tail_user_messages: REAL (actionable) user messages guaranteed to survive in the tail.
        # 1 = single last-user anchor; raise (e.g. 3) when bulky tool outputs fill the tail budget.
        "min_tail_user_messages": 1,
        # max_attempts: retry rounds before a turn gives up with "max compression attempts reached".
        # Raise (e.g. 6) for tool-schema-heavy sessions. Validated >= 1, cap 10.
        "max_attempts": 3,
        # proactive_prune_tokens: opt-in trigger (tokens) for the deterministic no-LLM tool-result
        # prune, independent of `threshold` (which rarely fires on large windows, so old tool output
        # is re-sent every turn); e.g. 48000 reclaims early. 0 = off. Tail protected by
        # `protect_last_n`. Built-in compressor only. Each committed prune rewrites sent history and
        # breaks the prompt-cache prefix — the min_reclaim gate below keeps those breaks episodic.
        "proactive_prune_tokens": 0,
        # Prune's summarize pass only touches tool results larger than this (chars); clamped >= 200
        # so a generated summary can't be re-summarized.
        "proactive_prune_min_result_chars": 8000,
        # A prune only commits when it reclaims at least this many tokens, then waits for a
        # trigger-sized runway to regrow before rearming. 0 = no minimum-savings gate.
        "proactive_prune_min_reclaim_tokens": 4096,
        # micro_compact: opt-in — after each turn fold the oldest un-absorbed exchange into a
        # rolling summary, amortizing compression cost. Off by default because every pass rewrites
        # sent history and breaks the prompt-cache prefix EVERY turn; enable only if the amortized
        # stall beats the cached-prefix discount. See website/docs/developer-guide/micro-compaction.md.
        "micro_compact": False,
        # Cadence: run a pass every Nth completed turn (1 = one cache break per turn, 5 = a fifth of
        # the breaks). Clamped >= 1; ignored unless micro_compact is true.
        "micro_compact_every_n_turns": 1,
        # Once the rolling summary exceeds this many tokens, the next pass re-summarizes it.
        "micro_compact_defrag_threshold_tokens": 2000,
        # Gateway session-hygiene force-compress threshold, by message count.
        "hygiene_hard_message_limit": 5000,
        # Max seconds the gateway waits for pre-agent hygiene compression WITHOUT forward progress.
        # Inactivity budget: a slow model still streaming tokens extends the wait.
        "hygiene_timeout_seconds": 30,
        # Absolute cap on the hygiene wait even while tokens are moving (bounds a trickle stream).
        # Clamped >= hygiene_timeout_seconds.
        "hygiene_total_ceiling_seconds": 600,
        "hygiene_failure_cooldown_seconds": 300,  # skip repeated failed hygiene attempts
        # Max seconds an ARRIVING user turn is held while a streaming hygiene summary finishes;
        # bounds user-visible latency (keep under chat idle timeouts, Telegram ~30s). On expiry the
        # turn proceeds uncompressed; the detached worker keeps its watermark-fenced commit, so the
        # summary is adopted at the next safe boundary.
        "hygiene_max_turn_hold_seconds": 10,
        # Inactivity budget for in-agent compress_context (loop, /compress, preflight); same
        # progress-aware semantics as hygiene_timeout_seconds. 0 = disable the owned wrapper
        # (callers passing commit_fence, e.g. gateway hygiene, never use it).
        "context_timeout_seconds": 120,
        # Absolute cap on the *pre-commit* compress_context wait (summary/stream phase) even while
        # tokens move. Clamped >= context_timeout_seconds when that is > 0. A started SessionDB
        # commit is never abandoned: past the ceiling it is logged (WARNING, then ERROR) and
        # surfaced on the warning channel while the host keeps waiting.
        "context_total_ceiling_seconds": 600,
        # Non-system head messages always kept verbatim, in ADDITION to the (always protected)
        # system prompt. 0 = pin nothing but system prompt + summary + tail.
        "protect_first_n": 3,
        # When True, auto-compression whose summary fails (aux error / non-JSON / timeout) aborts
        # instead of dropping the middle with a "summary unavailable" placeholder; the session
        # freezes at its size until /compress (bypasses the cooldown) or /new.
        "abort_on_summary_failure": False,
        # (Historical key name.) When True, gpt-5.4/5.5/5.6 and gpt-6 Astra (any slug containing
        # "astra" without "900k") on the ChatGPT Codex OAuth route raise their compaction trigger to
        # 85%: Codex hard-caps them at a 272K window, so the global 50% would compact at ~136K. False = global `threshold`. Only that route; the same models via
        # OpenAI direct, OpenRouter or Copilot keep the global value.
        "codex_gpt55_autoraise": True,
        # Show the one-time autoraise banner; False keeps the autoraise, hides the notice.
        "codex_gpt55_autoraise_notice": True,
        # Codex app-server thread compaction mode. The codex agent owns the thread context, so
        # Hermes' summarizer cannot shrink it. native = codex decides; hermes = Hermes' threshold
        # triggers thread/compact/start; off = never auto-trigger.
        "codex_app_server_auto": "native",
        # Opt in to OpenAI server-side compaction on the Responses API. Only gpt-5.6-family on
        # api.openai.com or the Codex backend; local compression stays as fallback.
        "codex_responses_native": False,
        # Absolute server compaction trigger (input tokens). None follows the local trigger with a
        # safety margin; explicit values only clamp downward so the server goes first.
        "codex_responses_compact_threshold": None,
        # in_place: compaction rewrites the message list and system prompt WITHOUT rotating the
        # session id (no parent_session_id chain, no `name #N` renumbering), avoiding the
        # session-rotation bug cluster. Pre-compaction turns are soft-archived under the same id
        # (active=0, compacted=1) — still session_search-able. False = legacy rotating-compaction
        # path.
        "in_place": True,
        # Per-model threshold overrides: keys substring-match the model name (longest wins), values
        # replace the global `threshold`, e.g. {"glm-5.2": 0.40}. Prefix a key with "<provider>:" to
        # scope it to one route ({"openai-codex:astra": 0.85} leaves Astra on OpenRouter/Nous at the
        # global value). The <512K floor (0.75) still applies raise-only on top.
        "model_thresholds": {},
        # Opt-in idle compaction (0 = off): a session resuming after this many idle seconds compacts
        # up front, before the first reply. Time-based complement to `threshold`; skipped when
        # already at/below threshold × target_ratio; honors the same cooldown/ anti-thrash/lock
        # guards. Example: 1800 = 30 min.
        "idle_compact_after_seconds": 0,
    },
    # Anthropic prompt caching (Claude via OpenRouter or native API). cache_ttl: "5m" | "1h"; other
    # non-falsy values are ignored; falsy (false, null, "off", "disabled", "no", "none") disables
    # caching.
    "prompt_caching": {"cache_ttl": "5m"},
    # OpenRouter settings. response_cache: X-OpenRouter-Cache header — identical requests return
    # cached responses at zero billing; independent of Anthropic prompt caching. response_cache_ttl:
    # seconds (1-86400), only used when response_cache is on. min_coding_score (0.0-1.0):
    # pareto-code router knob, applied only when model.model is "openrouter/pareto-code"; higher =
    # stronger/pricier coders, 0.65 = mid-tier, "" = let OpenRouter pick the strongest. Docs:
    # openrouter.ai/docs/guides/routing/routers/pareto-router
    "openrouter": {"response_cache": True, "response_cache_ttl": 300, "min_coding_score": 0.65},
    "bedrock": {  # AWS Bedrock; only used when model.provider is "bedrock".
        "region": "",  # empty = AWS_REGION env var → us-east-1
        "discovery": {
            "enabled": True,           # auto-discover models via ListFoundationModels
            "provider_filter": [],     # restrict to these providers, e.g. ["anthropic", "amazon"]
            "refresh_interval": 3600,  # cache discovery results (seconds)
        },
        # Bedrock Guardrails: create one in the console, then set ID and version.
        # https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
        "guardrail": {
            "guardrail_identifier": "",  # e.g. "abc123def456"
            "guardrail_version": "",     # e.g. "1" or "DRAFT"
            "stream_processing_mode": "async",  # "sync" | "async"
            "trace": "disabled",         # "enabled" | "disabled" | "enabled_full"
        },
    },
    # Auxiliary model config — provider/model per side task. provider "auto" = auto-detect;
    # empty model = provider's default aux model; all tasks fall back to
    # openrouter:google/gemini-3-flash-preview when the configured provider is unavailable.
    # extra_body is forwarded verbatim as request body fields for that task, e.g. OpenRouter
    # routing prefs / Pareto Code floor:
    #   auxiliary:
    #     compression:
    #       extra_body:
    #         provider: {order: [anthropic, google], sort: throughput}  # or price | latency
    #         plugins: [{id: pareto-router, min_coding_score: 0.5}]
    # Each task is independent — main-agent provider_routing and openrouter.min_coding_score
    # do NOT propagate to aux calls by design.
    "auxiliary": {
        # Same-provider retries for a transient blip (reset/timeout/5xx/408) on ANY aux call before
        # falling back; clamped [0,6]. Matters for pinned calls (MoA advisors) where provider
        # fallback is not meaningful recovery.
        "transient_retries": 2,
        # When true, the auto-chain's OpenRouter step is skipped unless the fallback model ends in
        # ":free" — a PAID lane is never used for background aux traffic even with
        # OPENROUTER_API_KEY set.
        "free_only": False,
        # Override the auto-chain's OpenRouter fallback model (default google/gemini-3.6-flash,
        # PAID). Pair e.g. "nvidia/nemotron-3-ultra-550b-a55b:free" with free_only: true. A one-time
        # WARNING is logged whenever a non-":free" model is engaged.
        "openrouter_model": "",
        # Endpoints that reject NON-streaming chat (HTTP 400): aux calls are sent with stream=True
        # and aggregated. Case-insensitive URL substrings; copilot.tencent.com is always
        # stream-only.
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
        # Note: web_extract no longer uses an auxiliary LLM — pages are
        # truncate-and-stored with a read_file pointer (no summarization),
        # and browser snapshots follow the same pattern. The old
        # ``auxiliary.web_extract.*`` block was removed here. Existing
        # values in user config.yaml files are harmless leftovers and ignored.
        "compression": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,        # seconds — compression summarises large contexts; increase for local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            # Guarded fast lane: only honored with a concrete provider/model
            # and an explicit ``reasoning_effort: none`` certification.
            # Zero preserves the historic uncapped compression request.
            "max_output_tokens": 0,
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
        # /review — the independent reviewer subagent's model. Unlike other
        # aux tasks this is not a single LLM call: the reviewer is a full
        # subagent (all normal subagent tools) spawned on the async
        # delegation rail. provider/model/base_url/api_key/api_mode are
        # resolved through the same credential system as delegation.provider
        # pins. Leave provider "auto" + model empty to run the reviewer on
        # the main agent's model.
        "review": {
            "provider": "auto",    # auto (= inherit main model) | openrouter | nous | anthropic | ...
            "model": "",           # e.g. "anthropic/claude-opus-4.6" — a strong reviewer model
            "base_url": "",        # direct OpenAI-compatible endpoint (takes precedence over provider)
            "api_key": "",         # API key for base_url / provider override
            "api_mode": "",        # force transport: chat_completions | anthropic_messages | codex_responses
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
            # Note: session_search no longer uses an auxiliary LLM (PR #27590 — single-shape tool returns DB
            # content directly). The old ``auxiliary.session_search.*`` block was removed here. Existing
            # values in user config.yaml files are harmless leftovers and ignored.
            "provider": "auto",
            "model": "",
            "prefer_fast_model": False,
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",
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
            # Aggregate INPUT-token budget for one review fork (issue #93057).
            # The fork's FIRST request replays the full snapshot as a warm
            # prompt-cache read (compaction is deferred until the first
            # provider response arrives); after that it compacts an oversized
            # snapshot in memory before further provider calls. This caps the
            # SUM of input tokens replayed across the whole review tool loop
            # (iterations are separately capped at 16). The loop stops before
            # the provider call that would cross the budget. 0 or a negative
            # value = unlimited.
            "max_input_tokens": 600000,
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
        # Recap tuning for /resume and startup resume.
        "resume_exchanges": 10,            # max user+assistant pairs to show
        "resume_max_user_chars": 300,      # truncate user message text
        "resume_max_assistant_chars": 200, # truncate non-last assistant text
        "resume_max_assistant_lines": 3,   # truncate non-last assistant lines
        # Skip tool-call-only assistant entries in the recap so it isn't dominated by `[2 tool
        # calls: ...]` lines; False shows them inline.
        "resume_skip_tool_only": True,
        "busy_input_mode": "interrupt",  # interrupt | queue | steer
        # steer mode: false hides only the "Steered into current run" bubble; steering itself still
        # happens.
        "busy_steer_ack_enabled": True,
        # Classic CLI multiline beyond Alt+Enter: Ctrl+J newline, trailing backslash+Enter
        # continues, Shift+Enter reported distinctly. False restores the c-j submit fallback for
        # POSIX PTYs whose plain Enter arrives as LF.
        "cli_multiline_shortcuts": True,
        # Interface bare `hermes`/`hermes chat` launches: "cli" (prompt_toolkit REPL) | "tui" (Ink).
        # Flags win: `--cli` forces the REPL, `--tui` / HERMES_TUI=1 forces the TUI.
        "interface": "cli",
        # `hermes --tui` auto-resumes the most recent human-facing session (like `hermes -c`).
        # HERMES_TUI_RESUME=<id> always wins.
        "tui_auto_resume_recent": False,
        # When true (default), the Desktop app reopens the last chat (or
        # last page) on cold start. Set false to always land on a fresh
        # new chat. Also a switch in Desktop Settings → Appearance.
        "resume_last_session": True,
        # When true (default), `hermes --tui` drops a one-time hint
        # ("subagents working · /agents to watch live") the first time a turn
        # starts delegating, nudging the user toward the live spawn-tree
        # dashboard. Set false to suppress the hint.
        "tui_agents_nudge": True,
        "bell_on_complete": False,
        # Bell when a blocking prompt opens (clarify/approval/sudo/secret).
        "bell_on_prompt": False,
        # Stream the model's reasoning/thinking live before the response.
        # Default ON: on thinking models the reasoning phase can run tens of
        # seconds, and with this off the user stares at a spinner the whole
        # time even though tokens are streaming. Set false for quiet output.
        "show_reasoning": True,
        # Post-response "Reasoning" recap collapses to 10 lines; true prints it all (live streaming
        # is always full).
        "reasoning_full": False,
        # Background self-improvement notices in chat: "off" (review still runs) | "on" (generic "💾
        # Memory updated") | "verbose" (content preview). Per-platform via
        # display.platforms.<platform>.memory_notifications.
        "memory_notifications": "on",
        # Gateway notices when a terminal(background=true) process finishes: "concise" (one line;
        # failures append an output tail) | "all" (running updates + final raw output) | "result"
        # (final raw only) | "error" (raw only on non-zero exit) | "off".
        "background_process_notifications": "concise",
        "streaming": False,
        "timestamps": False,      # message timestamps (CLI labels, TUI rows, desktop transcript)
        "timestamp_format": "%H:%M",  # strftime format, e.g. "%b-%d %H:%M"
        "final_response_markdown": "strip",  # render | strip | raw
        # Preserve recent classic-CLI output across Ctrl+L, /redraw and resize clears; disable if an
        # emulator misbehaves with replayed scrollback.
        "persistent_output": True,
        "persistent_output_max_lines": 200,
        # Also clear terminal scrollback on classic-CLI full redraw/resize recovery; enable when a
        # terminal/tmux stack stamps stale prompt chrome into scrollback.
        "cli_rebuild_scrollback_on_redraw": False,
        # Print a one-line summary of resolved modal prompts (approval/clarify) to scrollback.
        "persist_prompts": True,
        "inline_diffs": True,     # inline diff previews for write_file/patch/skill_manage
        # Append a one-line advisory to the final response when a write_file/patch failed this turn
        # and was never superseded by a successful write to the same path (catches "half the
        # parallel patches failed, model claims success").
        "file_mutation_verifier": True,
        # Nous credits status-bar notices (usage bands, grant-spent, depleted/restored). False mutes
        # them; balance data and /usage keep working.
        "credits_notices": True,
        # Append a one-line explanation when a turn ends with no usable reply (empty after retries,
        # truncated stream, pending tool result, iteration/budget limit) instead of the bare
        # "(empty)" sentinel.
        "turn_completion_explainer": True,
        "show_cost": False,       # $ cost in the status bar
        "battery": False,         # battery read-out first in status bar; no-op w/o battery
        # Focus view (/focus): display-only. Pins tool_progress to "off", reports per-turn
        # hidden-line count, pins a "focus" status segment. focus_saved_tool_progress holds the mode
        # /focus off restores. Never affects what the model sees (focus_view.py).
        "focus_view": False,
        "focus_saved_tool_progress": "all",
        "skin": "default",
        # UI language for static messages (approval prompts, some gateway slash replies); not agent
        # responses/logs/tool outputs. en, zh, ja, de, es, fr, tr, uk; unknown → en.
        "language": "en",
        # TUI busy indicator: kaomoji | emoji | unicode (braille) | ascii. `/indicator <style>`.
        "tui_status_indicator": "kaomoji",
        # Seconds between idle prompt_toolkit redraws in the classic CLI; keeps wall-clock
        # status-bar read-outs ticking and the bottom chrome from going stale. 0 disables it if it
        # fights terminal auto-scroll in non-fullscreen mode.
        # See #45592.
        "cli_refresh_interval": 1.0,
        # Vi/vim keybindings in the CLI input composer (config-only, no slash command).
        # Off by default, preserving prompt_toolkit's standard emacs bindings.
        "vim_mode": False,
        "user_message_preview": {  # CLI: submitted user-message lines echoed to scrollback
            "first_lines": 2,
            "last_lines": 2,
        },
        # Gateway: natural mid-turn assistant status messages. Desktop: keep mid-turn narration
        # between tool calls instead of collapsing to the final message.
        "interim_assistant_messages": True,
        # Codex Responses commentary channel: true delivers completed commentary as mid-turn interim
        # updates; false routes it to reasoning (visible only with show_reasoning).
        "show_commentary": True,
        "tool_progress_command": False,  # enable /verbose command in messaging gateway
        # display.tool_progress_overrides is deprecated (use display.platforms); a user-set value is
        # still honored at runtime and folded into platforms by migration.
        "tool_preview_length": 0,  # max chars for tool call previews (0 = no limit)
        # Human-phrased status labels for built-in tools ("Reading <file>") in CLI spinner and
        # gateway/desktop tool-progress; custom/plugin/MCP tools use the raw preview.
        "friendly_tool_labels": True,
        # CLI-only post-turn line: "⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3
        # commands". Never in quiet/non-interactive or gateway surfaces (own footer).
        "turn_summary": True,
        # CLI-only: cumulative turn output tokens on the live spinner ("· ↓ 1.2k tok").
        "spinner_token_flow": True,
        # Gateway tool-progress grouping where edits are supported: "accumulate" edits one bubble |
        # "separate" one message per tool (noisier). Needs tool_progress enabled. Per-platform:
        # display.platforms.<platform>.tool_progress_grouping.
        "tool_progress_grouping": "accumulate",
        # Custom long-running status phrases. Defaults: gateway/assets/status_phrases.yaml.
        # `path`/`paths` = HERMES_HOME-relative YAML files/dirs (or conventional status_phrases.yaml
        # / status_phrases/*.yaml). Keys: status, generic. mode: "append" (default) | "replace".
        # Per-platform: display.platforms.<platform>.status_phrases.
        "status_phrases": {},
        # Reasoning summary rendering: "code" (💭 fenced block) | "blockquote" ("> ") | "subtext"
        # ("-# " Discord small grey text; Discord's default). Per-platform via
        # display.platforms.<platform>.reasoning_style.
        "reasoning_style": "code",
        # Auto-delete EphemeralReply system notices ("✨ New session started!", …) after N seconds
        # where deletion is supported (Telegram; others ignore). Agent responses are never touched.
        # 0 = disabled.
        "ephemeral_system_ttl": 0,
        # Per-platform display/streaming overrides; unset keys fall through to the global. Telegram
        # has smooth native draft streaming (on); Discord/Slack only edit-based streaming, which
        # flickers (off). Gap-fillers only: explicit user values win, and the global
        # streaming.enabled master switch still gates everything.
        "platforms": {
            "telegram": {"streaming": True},
            "discord": {"streaming": False},
            "slack": {"streaming": False},
            # WeCom uses native streaming (msgtype: "stream") via
            # aibot_respond_msg — opt in by default so the WeCom client
            # renders the typing animation and cumulative content updates.
            "wecom": {"streaming": True},
        },
        # Gateway runtime footer on the FINAL message, e.g. `model · 68% · ~/projects/hermes`.
        # Per-platform: display.platforms.<platform>.runtime_footer.
        "runtime_footer": {
            "enabled": False,
            "fields": ["model", "context_pct", "cwd"],  # order shown; drop any to hide
        },
        # CLI/TUI status bar fields. Non-empty = only listed fields show (built-in order kept,
        # config controls visibility not ordering); empty = default set. Available: model,
        # context_detail, context_pct, cache_hit, latency, tps, compressions, bg_tasks,
        # bg_processes, bg_subagents, goal, git_branch (⎇ current branch, opt-in only), duration,
        # prompt_elapsed, idle_since, focus, yolo,
        # stash, battery, title, total_tokens (session Σ, opt-in only). Narrow terminals still drop
        # context_detail/prompt_elapsed/idle_since.
        "status_bar": {
            "fields": [],  # empty = built-in defaults (all fields)
        },
        "copy_shortcut": "auto",  # "auto" (platform default) | "ctrl_c" | "ctrl_shift_c" | "disabled"
        # Petdex animated mascot (https://github.com/crafter-station/petdex).
        # A purely cosmetic sprite that reacts to agent activity across the
        # CLI, TUI, and desktop app. Manage with `hermes pets`. Disabled until
        # a pet is installed + selected (no effect on prompt caching — this is
        # a display concern only).
        "pet": {
            "enabled": False,
            "slug": "",   # active pet slug in get_hermes_home()/pets/; empty → first installed
            # auto (detect kitty/iTerm2/sixel, else unicode half-blocks) | kitty | iterm | sixel |
            # unicode | off
            "render_mode": "auto",
            # Size scalar relative to native 192×208 frames, shared by desktop canvas and CLI/TUI
            # column width. Half-block fallback clamps to a legibility floor.
            "scale": 0.33,
            "unicode_cols": 0,  # Hard override for terminal column width; 0 = derive from scale.
        },
    },

    "dashboard": {
        # Visual theme: "default" | "midnight" | "ember" | "mono" | "cyberpunk" | "rose"
        "theme": "default",
        # Process-isolation rollout controls. Read via the raw config loader, so tui_gateway.server
        # also owns explicit defaults.
        "turn_isolation": False,
        "compute_host_heartbeat_secs": 15,
        "compute_host_respawn_max": 3,
        # Token/cost analytics surfaces are hidden by default: the numbers are a local LOWER-BOUND
        # estimate, not billing — only successful main-agent responses with a response.usage count;
        # auxiliary calls, retries, fallbacks and cache writes are missed, so the total can be
        # 10x-100x under the provider bill.
        "show_token_analytics": False,
        # IP addresses or bounded CIDR networks of reverse proxies allowed to
        # supply X-Forwarded-Proto / X-Forwarded-For. Loopback remains trusted
        # automatically. Wildcards and /0 networks are rejected so arbitrary
        # clients cannot spoof their scheme or source address.
        "trusted_proxies": [],
        # WebSocket keepalive for the dashboard/desktop web server (#79635).
        # Applied to NON-loopback binds only: loopback always disables the
        # protocol ping (see hermes_cli/web_server.py — an event-loop stall
        # must never kill a healthy local connection). Values are seconds.
        "ws_ping_interval": 20.0,
        "ws_ping_timeout": 20.0,
        # Grace window (seconds) before a WS-orphaned gateway session is
        # interrupted/reaped after its client disconnects (#79635). The
        # HERMES_TUI_WS_ORPHAN_REAP_GRACE_S env var remains an internal
        # override for backward compatibility. 0 disables the reap
        # (park forever).
        "ws_orphan_reap_grace_s": 20.0,
        # Activity-staleness threshold (seconds) gating the WS-orphan
        # interrupt of a detached RUNNING turn (#98028/#100325). A
        # client-absent turn is only interrupted once its agent activity
        # clock (the same one the agent.turn_liveness watchdog samples —
        # stamped by API waits, stream tokens, tool heartbeats) has been
        # idle at least this long; an actively-working detached turn runs
        # to completion. Default matches agent.turn_liveness.timeout_s.
        # 0 restores the old interrupt-at-grace-regardless behavior.
        "ws_orphan_activity_stale_s": 600.0,
        # Startup sweep of session rows orphaned by a dead gateway process
        # (#65194).  The ws-orphan grace timer above is in-process, so a
        # gateway restart (update, crash, systemd) leaves disconnected
        # sessions ``ended_at IS NULL`` forever — phantom "active" rows in
        # /resume and dashboards.  On every gateway boot (stdio TUI *and*
        # the desktop/dashboard WS sidecar), tui/desktop/subagent rows whose
        # start time AND newest message are both older than the session TTL
        # (HERMES_TUI_SESSION_TTL_S, default 6h) are closed with
        # end_reason='startup_orphan_reap'.  Messaging-gateway sessions
        # (telegram, discord, ...) are never touched; live in-memory
        # sessions are excluded; swept sessions stay resumable.
        "startup_orphan_sweep": True,
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
            "portal_url": "",
        },
        # Username/password gate (dashboard_auth/basic plugin, no OAuth IDP). Active when username
        # plus password_hash (preferred) or password (hashed in-memory) are set; empty username =
        # no-op. Env HERMES_DASHBOARD_BASIC_AUTH_USERNAME / _PASSWORD_HASH / _PASSWORD / _SECRET /
        # _TTL_SECONDS win when non-empty. secret signs session tokens; empty = random per-process
        # key (sessions die on restart, no multi-worker) — set 32+ random bytes. Hash:
        # plugins.dashboard_auth.basic.hash_password('PW').
        "basic_auth": {
            "username": "",
            "password_hash": "",  # scrypt$...
            "password": "",
            "secret": "",
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
        # the OAuth ``redirect_uri`` is built from. Its exact hostname is also
        # trusted by the HTTP Host / WebSocket Origin guards and engages the
        # auth gate when it is non-loopback, even if the backend binds to
        # loopback. Set this for deploys behind reverse proxies that don't
        # reliably forward
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

    "privacy": {
        "redact_pii": False,  # hash user IDs and strip phone numbers from LLM context
    },
    # Text-to-speech. Each provider accepts an optional `max_text_length:` override for the
    # per-request input-character cap; omit to use the provider's documented limit (OpenAI 4096, xAI
    # 15000, MiniMax 10000, ElevenLabs 5k-40k model-aware, Gemini 32000, Edge 5000, Mistral 4000,
    # NeuTTS/KittenTTS 2000).
    "tts": {
        # "edge" (free) | "elevenlabs" (premium) | "openai" | "xai" | "minimax" | "mistral" |
        # "gemini" | "deepinfra" | "neutts" (local) | "kittentts" (local) | "piper" (local)
        "provider": "edge",
        "edge": {
            # Popular: AriaNeural, JennyNeural, AndrewNeural, BrianNeural, SoniaNeural
            "voice": "en-US-AriaNeural",
        },
        "elevenlabs": {
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam
            "model_id": "eleven_multilingual_v2",
        },
        "openai": {
            "model": "gpt-4o-mini-tts",
            # gpt-4o-mini-tts voices: alloy, ash, ballad, cedar, coral, echo, fable, marin, nova,
            # onyx, sage, shimmer, verse
            "voice": "alloy",
        },
        "gemini": {
            "model": "gemini-2.5-flash-preview-tts",
            "voice": "Kore",
            # Gemini 3.1: aux-model rewrite inserts [audio tags] into the TTS script only.
            "audio_tags": False,
            # Optional local text file with performance direction; may include a `{transcript}`
            # placeholder, else the live transcript is appended.
            "persona_prompt_file": "",
        },
        "xai": {
            "voice_id": "eve",  # or a custom voice ID (docs.x.ai custom voices)
            "language": "en",  # BCP-47 code ("en", "pt-BR") or "auto"
            "speed": 1.0,  # 0.7–1.5
            "auto_speech_tags": False,  # insert expressive audio tags via LLM rewrite
            "optimize_streaming_latency": 0,  # 0–2, trades quality for lower latency
            "sample_rate": 24000,  # 22050 / 24000 / 44100 / 48000
            "bit_rate": 128000,  # MP3 bitrate; only applies when codec=mp3
        },
        "mistral": {
            "model": "voxtral-mini-tts-2603",
            "voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",  # Paul - Neutral
        },
        "minimax": {"model": "speech-02-hd", "voice_id": "English_expressive_narrator"},
        "kittentts": {
            "model": "KittenML/kitten-tts-nano-0.8-int8",  # nano 25MB; micro 41MB; mini 80MB
            "voice": "Jasper",
        },
        "neutts": {
            "ref_audio": "",  # path to reference voice audio (empty = bundled default)
            "ref_text": "",   # path to reference voice transcript (empty = bundled default)
            "model": "neuphonic/neutts-air-q4-gguf",  # HuggingFace model repo
            "device": "cpu",  # cpu, cuda, or mps
        },
        "piper": {
            # Voice name (downloaded on first use) or absolute path to a .onnx file; list:
            # github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md. Optional keys: voices_dir
            # (~/.hermes/cache/piper-voices/), use_cuda, length_scale (2.0 = twice as slow),
            # noise_scale, noise_w_scale, volume, normalize_audio.
            "voice": "en_US-lessac-medium",
        },
        "deepinfra": {
            "model": "",  # empty = first tts-tagged model from the live catalog
            "voice": "default",
            # optional "base_url" key overrides DEEPINFRA_BASE_URL for TTS only
        },
    },

    "stt": {
        "enabled": True,
        # Echo the raw transcript of gateway voice messages back as a 🎙️ message.
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
        # Client-side ffmpeg silence trim before cloud upload (local whisper uses VAD): silence
        # inflates upload time, billing and hallucinations. Failure = raw upload.
        "cloud_trim_silence": True,
        "cloud_trim_threshold_db": -40,  # quieter than this counts as silence
        "cloud_trim_keep_ms": 300,  # how much of each pause survives (natural pacing)
        "local": {
            "model": "base",  # tiny, base, small, medium, large-v3
            "language": "",  # auto-detect; set "en", "es", ... to force
            "initial_prompt": "",
            # Anti-hallucination (faster-whisper decodes junk from silence). vad: Silero filter
            # (false = raw audio, for music/ambient). A segment is dropped only if no_speech_prob
            # ABOVE no_speech_prob_threshold AND avg_logprob BELOW logprob_threshold.
            "vad": True,
            "vad_min_silence_ms": 500,  # min silence (ms) that splits speech chunks
            "no_speech_prob_threshold": 0.6,
            "logprob_threshold": -1.0,
            "unload_after_idle_seconds": 0,  # 0 = never; e.g. 300 frees the model after 5min
        },
        "groq": {
            # whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en
            "model": "whisper-large-v3-turbo",
            "language": "",  # auto-detect; set "en", "es", ... to force
        },
        "openai": {
            # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe
            "model": "whisper-1",
            "language": "",  # auto-detect; set "en", "es", ... to force
        },
        "mistral": {
            "model": "voxtral-mini-latest",  # voxtral-mini-latest, voxtral-mini-2602
            "language": "",  # auto-detect; set "en", "es", ... to force
        },
        "xai": {
            "language": "",  # auto-detect; set "en", "es", ... to force
        },
        "elevenlabs": {
            "model_id": "scribe_v2",  # scribe_v2, scribe_v1
            "language_code": "",  # auto-detect; set "eng", "spa", ... to force
            "tag_audio_events": False,
            "diarize": False,
        },
        "deepinfra": {
            "model": "",  # empty = first stt-tagged model from the live catalog
            # optional "base_url" key overrides DEEPINFRA_BASE_URL for STT only
        },
    },

    "voice": {
        # How the Desktop voice conversation is wired:
        #   chained  — STT → Hermes turn → TTS (the stt.* / tts.* providers below)
        #   gpt-live — one full-duplex voice model (OpenAI GPT-Live) owns the mic and speaker and
        #              DELEGATES every real request to Hermes (any model / provider you have
        #              selected); needs an OpenAI API key. $0.05/min voice layer billing.
        "voice_chat_mode": "chained",
        "gpt_live": {
            "model": "gpt-live-1",
            "voice": "marin",  # marin | quartz | ripple | vesper | willow | stone | gleam | meridian | ...
            # Extra sentences appended to the live model's conversation persona (tone, pacing, language).
            "instructions": "",
            # optional "api_key" / "base_url" keys override the OpenAI audio credentials for this mode only
        },
        "record_key": "ctrl+b",
        "submit_mode": "direct",  # TUI: direct submits immediately; draft = editable transcript
        "max_recording_seconds": 120,
        "auto_tts": False,
        # Desktop remote clients call the profile's STT/TTS providers
        # DIRECTLY (config + key fetched over the authenticated REST channel
        # at voice-session start) instead of relaying audio through the
        # gateway — lowest-hop path in both directions. false = always relay.
        "client_direct": True,
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
    # "Hey Hermes" hands-free wake word: always-on, on-device hotword detection that starts a fresh
    # voice session. Off by default; toggle with /wake.
    "wake_word": {
        "enabled": False,
        "surface": "auto",  # eligible surface: "auto" (first claimant) | "cli" | "tui" | "gui"
        "input_device": None,  # PortAudio input device index/name; null = process default
        "capture": "auto",  # auto | local | client (desktop streams mic via wake.feed)
        # "openwakeword" (free, local) | "sherpa" (free, ANY phrase, no training) | "porcupine"
        # (premium; needs PORCUPINE_ACCESS_KEY)
        "provider": "openwakeword",
        # sherpa: this IS the detected phrase; other engines: cosmetic label (detection is keyed by
        # the model/keyword below)
        "phrase": "hey hermes",
        "sensitivity": 0.6,  # 0.0-1.0 threshold, consistent across engines (higher = stricter)
        # openWakeWord only: consecutive over-threshold frames to fire (higher = fewer false
        # triggers, more latency; 1 = single-frame)
        "confirmation_frames": 3,
        "start_new_session": True,  # fresh session on wake vs. continue the current one
        # sherpa only: listen for every wake-enabled profile's phrase and route to it
        "profile_routing": True,
        "openwakeword": {
            # "hey_hermes" | built-in openWakeWord name ("hey_jarvis", "alexa", ...) | path to a
            # custom .onnx/.tflite model
            "model": "hey_hermes",
            # "" (auto: tflite on macOS ARM64, onnx elsewhere) | "onnx" | "tflite" — onnx scores
            # near-zero on macOS ARM64 (arms but never fires)
            "inference_framework": "",
        },
        "sherpa": {
            # sherpa-onnx KWS model dir; empty = auto-download the small English zipformer
            "model_dir": "",
        },
        "porcupine": {
            # built-in keyword ("jarvis", "computer", ...) or path to a custom .ppn
            "keyword": "jarvis",
        },
    },

    "human_delay": {"mode": "off", "min_ms": 800, "max_ms": 2500},

    # Context engine — how the context window is managed near the token limit. "compressor" =
    # built-in lossy summarization; or a plugin name (e.g. "lcm") installed in
    # plugins/context_engine/<name>/ or ~/.hermes/plugins/.
    "context": {
        "engine": "compressor",
        # Return freed glibc pages at agent/TUI cleanup boundaries (no-op elsewhere).
        "memory_trim": {
            "enabled": True,
            "cooldown_seconds": 60.0,
            "log_every_n": 1,  # INFO-log every Nth periodic trim; force paths always log.
            # Suppress INFO logs when the readable RSS delta is smaller; 0 = log all.
            "info_log_min_delta_mb": 0.0,
        },
    },
    "memory": {  # Persistent memory — bounded curated memory injected into the system prompt
        "memory_enabled": True,
        "user_profile_enabled": True,
        # Approval gate for memory writes on BOTH foreground turns and the background review fork.
        # true = foreground writes prompt inline; background writes are staged (/memory
        # pending|approve <id>|reject <id>). To disable memory: memory_enabled.
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
    # Subagent delegation — override the provider:model used by delegate_task so children run on a
    # cheaper/faster model. Uses the same runtime provider resolution as CLI/gateway startup, so
    # every configured provider is supported.
    "delegation": {
        "model": "",       # e.g. "google/gemini-3-flash-preview" (empty = inherit parent model)
        "provider": "",    # e.g. "openrouter" (empty = inherit parent provider + credentials)
        "base_url": "",    # direct OpenAI-compatible endpoint for subagents
        "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        "api_mode": "",    # wire protocol for delegation.base_url: "chat_completions",
                           # "codex_responses", or "anthropic_messages". Empty = auto-detect
                           # from URL (e.g. /anthropic suffix → anthropic_messages). Set this
                           # explicitly for non-standard endpoints the heuristic can't detect.
        # Per-child request settings sent on every delegation API call, on all
        # three resolution branches (direct base_url, named provider, and
        # parent-inherit). Top-level keys are API kwargs (e.g. service_tier);
        # an "extra_body" sub-dict is merged into the request's extra_body —
        # e.g. {"extra_body": {"provider": {"sort": "throughput"}}} routes
        # OpenRouter delegation children to the fastest provider. Precedence:
        # these explicit values merge OVER runtime/parent-derived overrides
        # (explicit keys win; extra_body deep-merged one level).
        "request_overrides": {},
        # When delegate_task narrows child toolsets explicitly, preserve any
        # MCP toolsets the parent already has enabled. On by default so
        # narrowing (e.g. toolsets=["web","browser"]) expresses "I want these
        # extras" without silently stripping MCP tools the parent already has.
        # Set to false for strict intersection.
        "inherit_mcp_toolsets": True,
        # Per-subagent iteration cap (own budget, independent of the parent's).
        "max_iterations": 250,
        # Hard per-summary char ceiling on subagent results, layered on the dynamic budget (each
        # summary is sized to the parent's remaining context headroom; trimmed text spills to
        # ~/.hermes/cache/delegation/ with a head+tail window + read_file offset footer, nothing
        # lost). 0 disables the ceiling; the dynamic budget still applies.
        "max_summary_chars": 24000,
        # Wall-clock cap per child (seconds, floor 30). 0 = no timeout: children fail only from real
        # errors (API, tools, iteration budget).
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
        # Orchestrator role controls. Depth floored at 1, no ceiling; each level multiplies cost.
        "max_spawn_depth": 1,  # 1 = flat, 2 = orchestrator→leaf, 3+ = deeper
        "orchestrator_enabled": True,  # kill switch for role="orchestrator"
        # Subagent threads ALWAYS resolve approvals non-interactively (the parent TUI owns stdin;
        # input() from a worker would deadlock). false = auto-deny, true = auto-approve "once"; both
        # log a warning audit line. true only for trusted batch work.
        "subagent_auto_approve": False,
        # Background processes started by subagents (task_id "sa-...") route
        # their notify_on_complete / watch_pattern notifications to the PARENT
        # conversation (children consume their own waits; anything outliving
        # the child needs a durable consumer). By default those parent-facing
        # notifications are SUPPRESSED — the child's consolidated delegation
        # result is the deliverable, and "npm ci finished" walls mid-chat are
        # noise. Async-delegation results themselves are NEVER suppressed.
        # Set to true to restore delivery of child process notifications
        # (with subagent attribution lines).
        "surface_child_process_notifications": False,
    },
    # Ephemeral prefill messages file — JSON list of {role, content} dicts injected at the start of
    # every API call for few-shot priming. Never saved to sessions/logs/trajectories.
    "prefill_messages_file": "",
    # Goals — persistent cross-turn /goal loop: after each turn an aux-model judge checks if the
    # goal is satisfied, else a continuation prompt re-enters the session until done, budget
    # exhausted, or paused. Judge failures fail OPEN; the budget is the backstop.
    "goals": {
        # Max continuation turns before auto-pause (/goal resume) — guards against judge false
        # negatives and unbounded spend.
        "max_turns": 20,
    },
    # Loops — /loop re-runs a prompt or slash command on a cadence in-session. Fixed interval fires
    # on the user's clock; self-paced (no interval) starts at the floor and backs off exponentially
    # while replies stop changing.
    "loops": {
        "min_interval_seconds": 30,  # smallest fixed interval; tighter cadences raised to it
        "max_ticks": 100,  # auto-pause after this many wakeups unless --times set; 0 = unlimited
        "self_paced_floor_seconds": 60,  # Self-paced cadence bounds (seconds).
        "self_paced_ceiling_seconds": 900,
    },
    # Mixture of Agents — named presets used by /moa. A preset is an execution mode around the main
    # model, not a model itself: references + aggregator synthesize private guidance before each
    # main-model iteration.
    "moa": {
        "default_preset": "default",
        "active_preset": "",
        # Write each MoA turn (reference + aggregator exact input/output/usage) as JSONL to
        # <hermes_home>/moa-traces/<session_id>.jsonl (or trace_dir) for auditing.
        "save_traces": False,
        "trace_dir": "",
        # PII/credential redaction of advisor outputs: "" off | "display" (UI reference blocks +
        # traces only; aggregator sees raw) | "full" (also the aggregator prompt).
        # Advisors can echo PII from the conversation (emails, formatted phone numbers) and credential
        # shapes into reference blocks, traces, and the aggregator prompt. Modes ('' = off, the default):
        # "display" — redact user-visible surfaces only (reference blocks shown in the UI + saved MoA trace
        # records); the aggregator still sees raw advisor text. "full"    — additionally redact the advisor
        # text injected into the aggregator prompt (issue #59959).
        "privacy_filter": "",
        "presets": {
            "default": {
                "reference_models": [
                    {"provider": "openai-codex", "model": "gpt-5.5"},
                    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                ],
                "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},

                "enabled": True,
            }
        },
    },

    # Skills — external skill directories for sharing skills across tools/agents.
    # Each path is expanded (~, ${VAR}) and resolved.  Read-only — skill creation
    # goes to ~/.hermes/skills/ unless create_dir (below) redirects it.
    "skills": {
        "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
        # Where agent-created skills (skill_manage action=create) are written.
        # Empty = the profile-local skills dir (~/.hermes/skills/). When set,
        # new skills land here AND every agent-facing instruction that names
        # the creation path (tool schema text, prompts) renders this directory
        # instead of the default. Expanded (~, ${VAR}); relative paths resolve
        # against HERMES_HOME. The directory is scanned for skills alongside
        # the local dir. e.g. "/opt/brain/skills"
        "create_dir": "",
        # Project-local skill discovery: when a session starts inside a git
        # checkout, ``<root>/.hermes/skills/`` and ``<root>/.agents/skills/``
        # are sourced as the highest-precedence skill tier — but ONLY when the
        # project root is listed in trusted_project_dirs below. Trust a repo
        # with ``hermes skills trust`` (run from inside it). Set to false to
        # disable discovery entirely (no scan, no untrusted-skills notice).
        "project_discovery": True,
        # Trusted project roots; managed by `hermes skills trust` / `untrust`.
        "trusted_project_dirs": [],
        # Skill names pinned as fully loaded in every new session (CLI, TUI, gateway, cron, API).
        # Resolved once when the agent's prompt is first built; missing/disabled names warn and
        # skip; HERMES_IGNORE_RULES suppresses the list like the other auto-injected context.
        "auto_load": [],
        # Substitute ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} in SKILL.md content.
        "template_vars": True,
        # Pre-execute !`cmd` snippets in SKILL.md, inlining stdout (dates, git state...). Off:
        # skill-author content would run on the host unapproved — trusted sources only.
        "inline_shell": False,
        "inline_shell_timeout": 10,  # seconds per !`cmd` snippet
        # Security-scan skills the agent writes via skill_manage. Off: the agent can run the same
        # code via terminal() ungated, so it mostly blocks prose with risky keywords. On: a
        # dangerous verdict is a tool error the agent can retry. Hub installs are always scanned.
        "guard_agent_created": False,
        # Advisory NVIDIA SkillEvaluator Tier 1 scan on `hermes skills install` (alongside the
        # enforcing built-in guard), only if `skillevaluator` is on PATH (uv tool install
        # "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git"). Informational, never
        # blocking; secrets-class findings shown red. No-op without it.
        "tier1_advisory": True,
        # Approval gate for skill_manage mutations on BOTH foreground turns and the background
        # review fork. true = ALWAYS stage (SKILL.md too large for an inline prompt): /skills
        # pending, /skills diff <id>, /skills approve|reject <id>.
        "write_approval": False,
        # Audit ledger: every skill mutation appends to ~/.hermes/skills/.curator_ledger.jsonl with
        # before/after hashes (blobs under ~/.hermes/.curator_backups/blobs/); powers `hermes
        # curator ledger` / `rollback <entry-id>`. Never a gate — failures can't block.
        # See #79686.
        "ledger": True,
    },

    # Curator — background maintenance of AGENT-CREATED skills (never hub-installed): marks
    # long-unused skills stale, archives (never deletes) obsolete ones, optionally consolidates
    # overlaps via a forked aux-model agent. Inactivity-triggered from session start, no cron
    # daemon. `hermes curator status` shows the last run.
    "curator": {
        "enabled": True,
        "interval_hours": 24 * 7,  # hours between runs
        "min_idle_hours": 2,  # only run after the agent has been idle this long
        "stale_after_days": 14,  # mark "stale" after this many unused days
        "archive_after_days": 30,  # move to skills/.archive/ (recoverable) after this many
        # LLM consolidation (umbrella-building) pass. OFF = deterministic inactivity prune only, no
        # aux-model cost. `hermes curator run --consolidate` overrides once.
        "consolidate": False,
        # Also prune bundled built-ins (a suppression list stops `hermes update` restoring them);
        # hub-installed skills are NEVER pruned. A built-in's clock starts when the curator first
        # sees it, so never a mass-prune on the first run. false = keep all.
        "prune_builtins": True,
        # TTL purge of skills/.archive/: 0 = never; > 0 lets the explicit `hermes curator purge`
        # delete older archived skills (never automatic; logged in the ledger).
        "archive_ttl_days": 0,
        # Before every real (non-dry-run) pass, snapshot ~/.hermes/skills/ to
        # ~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz (`hermes curator rollback`).
        "backup": {
            "enabled": True,
            "keep": 5,  # retain last N regular snapshots
        },
    },
    # Honcho AI-native memory — ~/.honcho/config.json is the source of truth (apiKey, workspace,
    # peerName, sessions, enabled); hermes-specific overrides only here.
    "honcho": {},
    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York"). Empty = server-local time.
    "timezone": "",

    "slack": {
        "require_mention": True,  # require @mention to respond in channels
        "free_response_channels": "",  # comma-separated channel IDs answered without mention
        "allowed_channels": "",  # if set, ONLY respond in these channel IDs (whitelist)
        "require_mention_channels": "",  # channel IDs where @mention is ALWAYS required
        # Ignore messages whose first token @mentions another user unless the bot is also mentioned.
        # Env: SLACK_IGNORE_OTHER_USER_MENTIONS.
        "ignore_other_user_mentions": False,
        "thread_require_mention": False,  # require @mention in thread replies too
        "channel_prompts": {},  # per-channel ephemeral system prompts
    },

    "discord": {
        "require_mention": True,  # require @mention to respond in server channels
        "free_response_channels": "",  # comma-separated channel IDs answered without mention
        "allowed_channels": "",  # if set, ONLY respond in these channel IDs (whitelist)
        "auto_thread": True,  # auto-create threads on @mention in channels (like Slack)
        "thread_require_mention": False,  # require @mention in threads too (multi-bot threads)
        # Multi-bot rooms: another bot must type @thisbot (a reply/quote alone won't) to trigger a
        # reply — stops two bots replying to each other forever. Humans unaffected.
        "bots_require_inline_mention": False,
        # Prepend recent channel scrollback when triggered (recovers messages gated out by
        # require_mention); limit = max messages scanned.
        "history_backfill": True,
        "history_backfill_limit": 50,
        # Replay messages missed while offline, after reconnect/startup.
        "missed_message_backfill": {
            "enabled": False,
            "channels": "",  # comma-separated channel IDs; empty uses free_response_channels
            "window_seconds": 21600,  # only inspect messages from the last 6 hours
            "limit": 100,  # global cap on messages scanned per reconnect
            "max_dispatches": 10,  # cap on recovered messages dispatched per reconnect
        },
        "reactions": True,  # add 👀/✅/❌ reactions to messages during processing
        # Gateway transport health probe: inspects the WebSocket's ready/open/heartbeat state (never
        # REST) as proof events still arrive. Any value 0 disables it.
        "websocket_liveness_interval_seconds": 15,
        "websocket_liveness_failure_threshold": 2,
        "websocket_heartbeat_ack_max_age_seconds": 60,
        "websocket_max_latency_seconds": 30,
        # Dispatch-side dimension: a socket that ACKs heartbeats but delivers no events for this
        # long is treated as deaf. 4 h absorbs a quiet server overnight; 0 disables it.
        "websocket_event_max_silence_seconds": 14400,
        # per-channel ephemeral system prompts (forum parents apply to child threads)
        "channel_prompts": {},
        # Opt-in DM role auth: DISCORD_ALLOWED_ROLES normally authorizes guild messages only (DMs
        # need DISCORD_ALLOWED_USERS). A guild ID here also authorizes DMs from that guild's members
        # holding the allowed role. Unset / "" / 0 = off.
        # See #12136.
        "dm_role_auth_guild": "",
        # discord / discord_admin tools: allowed actions (comma string or YAML list; empty = all,
        # subject to bot intents; unknown names dropped with a warning): list_guilds, server_info,
        # list_channels, channel_info, list_roles, member_info, search_members, fetch_messages,
        # list_pins, pin_message, unpin_message, create_thread, add_role, remove_role.
        "server_actions": "",
        # DEPRECATED no-op (uploads are always cached; messaging auth is the gate). Kept so existing
        # configs don't error. Env: DISCORD_ALLOW_ANY_ATTACHMENT.
        "allow_any_attachment": False,
        # Max bytes per cached attachment (held in memory while written); 0 = no cap. Env:
        # DISCORD_MAX_ATTACHMENT_BYTES.
        "max_attachment_bytes": 33554432,
        # Mention allowed users on approval prompts so owners notice them in shared channels. Env:
        # DISCORD_APPROVAL_MENTIONS.
        "approval_mentions": False,
        # Voice-channel inactivity timeout (seconds); 0 = stay until `/voice leave`.
        "voice_channel_inactivity_timeout_seconds": 300,
        # Minimum seconds before force-stopping a VC playback; the adapter probes clip duration and
        # extends this floor so long TTS isn't cut off.
        "voice_playback_timeout_seconds": 120,
        # Voice-channel software mixer (plugins/platforms/discord/voice_mixer.py): ambient
        # "thinking" bed, verbal acks and TTS OVERLAP (ambient ducked) vs stop-and-swap.
        "voice_fx": {
            "enabled": False,  # master switch for the mixer subsystem
            "ambient_enabled": True,  # play the idle "thinking" bed while tools run
            "ambient_path": "",  # custom loop audio file; "" = synthesised pad
            "ambient_gain": 0.18,  # idle bed loudness, 0.0–1.0
            "duck_gain": 0.06,  # ambient loudness while speech plays
            "speech_gain": 1.0,  # TTS / ack loudness, 0.0–1.0
            "ack_enabled": True,  # speak a short phrase before the first tool call
            "ack_phrases": [  # picked at random; [] disables phrases
                "Let me look into that.",
                "One moment.",
                "Checking on that now.",
                "Give me a sec.",
                "On it.",
            ],
        },
    },

    "whatsapp": {
        # reply_prefix: None = built-in "☤ *Hermes Agent*" header; "" disables; \n allowed.
    },

    "telegram": {
        "reactions": False,  # add 👀/✅/❌ reactions to messages during processing
        # per-chat/topic ephemeral system prompts (topics inherit from parent group)
        "channel_prompts": {},
        "allowed_chats": "",  # if set, ONLY respond in these group/supergroup chat IDs
        "extra": {
            # Bot API 10.1 native rich messages (tables/task lists/math). Off = legacy MarkdownV2,
            # since rich messages are hard to copy as plain text.
            "rich_messages": False,
            # Experimental rich draft previews while streaming DMs; off because Telegram
            # Desktop/macOS can overlay draft frames until the chat redraws.
            "rich_drafts": False,
            # CJK stays on legacy MarkdownV2 (Telegram Desktop/macOS garbles rich CJK, #47653);
            # set True on an unaffected client to get native rich tables for CJK.
            "allow_cjk_rich_messages": False,
        },
    },

    "mattermost": {
        "require_mention": True,  # require @mention to respond in channels
        "free_response_channels": "",  # comma-separated channel IDs answered without mention
        "allowed_channels": "",  # if set, ONLY respond in these channel IDs (whitelist)
        "channel_prompts": {},  # per-channel ephemeral system prompts
    },

    "matrix": {
        "require_mention": True,  # require @mention to respond in rooms
        "free_response_rooms": "",  # comma-separated room IDs answered without mention
        "allowed_rooms": "",  # if set, ONLY respond in these room IDs (whitelist)
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
    # unattended_mode — what to do when a session on an unattended
    # programmatic platform (webhook, msgraph_webhook, api_server) hits a
    # dangerous command. These surfaces bind a session platform like chat
    # gateways do, but have no send_exec_approval and no /approve channel —
    # a pending approval there just blocks for the full timeout with nobody
    # to answer (#37284, #87509):
    #   deny    — block the command instantly and let the agent find another
    #             way (default, safe; mirrors cron_mode deny)
    #   approve — auto-approve all dangerous commands on unattended platforms
    #
    # timeout — seconds to wait for the user's approve/deny before failing
    # closed (deny). Shared by the CLI prompt and gateway/messaging waits.
    # Messaging approvals arrive as a push notification the user may not see
    # immediately — 60s proved too tight on Telegram/Discord (the prompt
    # expired before the user reached their phone), so the default is 300.
    "approvals": {
        # single_query_mode — what to do when a single-query (-q) session hits a dangerous command. -q runs
        # export HERMES_INTERACTIVE=1 (for interactive sudo prompts) but have NO user waiting to answer
        # approval prompts — an unanswered prompt just waits the full timeout then fails closed, so the
        # agent is forced to work around the block (often via execute_code). This setting makes that intent
        # explicit: deny    — block the command and let the agent find another way (default, safe; mirrors
        # cron_mode deny) approve — auto-approve all dangerous commands in single-query mode These surfaces
        # bind a session platform like chat gateways do, but have no send_exec_approval and no /approve
        # channel — a pending approval there just blocks for the full timeout with nobody to answer (#37284,
        # #87509): deny    — block the command instantly and let the agent find another way (default, safe;
        # mirrors cron_mode deny) approve — auto-approve all dangerous commands on unattended platforms
        # Shared by the CLI prompt and gateway/messaging waits. Messaging approvals arrive as a push
        # notification the user may not see immediately — 60s proved too tight on Telegram/Discord (the
        # prompt expired before the user reached their phone), so the default is 300.
        "mode": "smart",
        "timeout": 300,
        "cron_mode": "deny",
        "single_query_mode": "deny",
        "unattended_mode": "deny",
        # Operator-customizable policy text for smart approvals. When
        # non-empty, this is appended to the smart-approval guardian's
        # SYSTEM prompt (trusted channel) as additional rules — e.g.
        # "Always ESCALATE commands touching /etc" or "APPROVE docker
        # compose restarts under ~/deploys". Inspired by ChatGPT Work's
        # customizable auto-review guardian policy.
        "smart_policy": "",
        # After this many consecutive guardian DENYs in a session, the deny message escalates to a
        # hard-stop (report to user / ask for /approve). Approval resets; 0 off.
        "denial_breaker_threshold": 3,
        # Case-insensitive fnmatch globs against terminal commands; a match blocks even under --yolo
        # / mode=off. Quote in YAML when starting with * or containing {}/!/: e.g. "git push
        # --force*".
        "deny": [],
        # /reload-mcp confirms before rebuilding the MCP tool set (it invalidates the prompt cache,
        # so the next message re-sends full input). "Always Approve" → false.
        "mcp_reload_confirm": True,
        # /clear, /new, /reset, /undo confirm before discarding state (Approve Once / Always Approve
        # / Cancel via tools.slash_confirm; native buttons on Telegram/ Discord/Slack). "Always
        # Approve" → false. HERMES_TUI_NO_CONFIRM=1 skips the TUI modal.
        "destructive_slash_confirm": True,
    },
    # Permanently allowed dangerous command patterns (added via "always" approval).
    "command_allowlist": [],
    # User-defined quick commands that bypass the agent loop (type: exec only).
    "quick_commands": {},
    # Per-platform system-prompt hint overrides, keyed by platform name (whatsapp, slack, telegram,
    # ...). Value: {"append": text} keeps the built-in hint and appends; {"replace": text}
    # substitutes it; a bare string is shorthand for append. `replace` wins over `append` if both
    # are given.
    "platform_hints": {},

    # Plugin system settings. ``enabled`` / ``disabled`` are written by
    # ``hermes plugins enable|disable`` and intentionally omitted here so an
    # empty default does not clobber a user's allow-list on merge.
    "plugins": {
        # Wall-clock cap (seconds) for a single in-process Python plugin hook
        # callback. Shell hooks keep their own per-entry ``timeout``. Set to 0
        # to disable the cap (sync call on the agent thread). Capped at 600.
        "hook_callback_timeout": 30,
    },

    # Shell-script hooks — declarative bridge that invokes shell scripts
    # on plugin-hook events (pre_tool_call, post_tool_call, pre_llm_call,
    # subagent_stop, etc.).  Each entry maps an event name to a list of
    # {matcher, command, timeout} dicts.  First registration of a new
    # command prompts the user for consent; subsequent runs reuse the
    # stored approval from ~/.hermes/shell-hooks-allowlist.json.
    # See `website/docs/user-guide/features/hooks.md` for schema + examples.
    "hooks": {},
    # Auto-accept shell-hook registrations without a TTY prompt (also --accept-hooks or
    # HERMES_ACCEPT_HOOKS=1). Gateway/cron/non-interactive runs need one of these to pick up
    # newly-added hooks.
    "hooks_auto_accept": False,
    # Custom personalities: {"name": "system prompt"} or {"name": {"description", "system_prompt",
    # "tone", "style"}}.
    "personalities": {},
    "security": {  # Security: pre-exec scanning via tirith plus related guards.
        "allow_private_urls": False,  # allow requests to private/internal IPs (OpenWrt, VPNs)
        "redact_secrets": True,
        # Persisted acknowledgement for unattended model overrides whose tier lets the vendor train
        # on prompts. The startup guard still warns every run; cost guards are unaffected.
        "allow_data_training_tiers_noninteractive": False,
        # Human-approval presentation transport. "builtin" = CLI/TUI/gateway/ACP surfaces; a plugin
        # transport is used only when named explicitly. Transport timeout/error/invalid response
        # DENIES unless transport_fallback is "builtin". Presentation only: plugins cannot detect,
        # suppress, or auto-approve commands outside a correlated human response.
        "approval": {"transport": "builtin", "transport_fallback": "deny"},
        # Writes to agent-instruction files (AGENTS.md/CLAUDE.md/SOUL.md/.cursorrules, project-local
        # .hermes config) always need human approval, even under yolo. Extra patterns are fnmatch
        # globs on the basename (e.g. "*.mdc").
        "protected_instruction_files": True,
        "protected_instruction_extra_patterns": [],
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
        "website_blocklist": {"enabled": False, "domains": [], "shared_files": []},
        # IDs of supply-chain advisories the user has read and acted on; acked ones stop the startup
        # banner. Add via `hermes doctor --ack <id>`; remove by editing the list. Catalog:
        # hermes_cli/security_advisories.py.
        "acked_advisories": [],
        # Lazy-install opt-in backend packages from PyPI when a backend that needs them is first
        # enabled (e.g. `elevenlabs`). False = require explicit pip install for everything beyond
        # the base set (restricted/audited/air-gapped environments).
        "allow_lazy_installs": True,
    },

    "cron": {
        "catch_up_missed": True,  # False skips recurring misses beyond the local grace window.
        # Let cron-spawned agents use the cronjob toolset (the "cron-librarian" pattern). Off by
        # default: policy-denied in cron context to prevent unattended scheduling loops. Jobs
        # created this way are user-owned in the same flat jobs table. Interactive toolsets
        # (messaging/clarify) stay denied in cron regardless.
        "allow_agent_scheduling": False,
        # Pre-dispatch validation: before building any agent machinery, verify the provider API key
        # resolves (unless a fallback chain exists), attached skills are ready, and delivery
        # platforms are configured. Failure -> last_status=blocked_config, ONE alert, no LLM call.
        # False = fail during the run instead.
        "preflight": True,
        # Default model for cron jobs (WHAT model runs). Fire-time resolution: per-job pin >
        # cron.model > the job's creation-time snapshot > model.default. An unpinned job keeps
        # running on the model it was created under when model.default later changes; cron.model
        # is the way to move the whole fleet at once. "" = fall through.
        "model": "",
        # Inference provider paired with cron.model (NOT the scheduler provider below). "" = resolve
        # from global config.
        "model_provider": "",
        # Cron SCHEDULER provider (WHEN a due job fires). "" = built-in in-process 60s ticker. Name
        # an installed provider (plugins/cron_providers/<name>/ or $HERMES_HOME/plugins/ <name>/),
        # e.g. "chronos" (NAS-mediated managed cron for scale-to-zero). An unknown or unavailable
        # provider falls back to the built-in so cron never loses its trigger.
        "provider": "",
        # Chronos settings; consulted only when provider == "chronos". All non-secret — the agent
        # holds NO scheduler credentials (provision reuses the Nous Portal token).
        "chronos": {
            # NAS/portal base URL that arms/cancels one-shots and mints the inbound fire JWT (used
            # as the expected issuer).
            "portal_url": "https://portal.nousresearch.com",
            # This agent's publicly reachable base URL; NAS POSTs {callback_url}/api/cron/fire. ""
            # -> Chronos unavailable, resolver falls back to the built-in ticker.
            "callback_url": "",
            "expected_audience": "",  # Expected JWT audience (e.g. "agent:{instance_id}").
            # NAS JWKS URL for verifying the fire JWT signature. "" -> the fire endpoint refuses all
            # tokens (never an unsigned decode).
            "nas_jwks_url": "",
        },
        # Wrap delivered cron responses with a task-name header and "The agent cannot see this
        # message" footer. False = clean output.
        "wrap_response": True,
        # Delivery behaviour for cron output sent through a live gateway adapter.
        "delivery": {
            # Mark cron deliveries as FINAL notifications so the platform pushes
            # them (Telegram's "important" notification mode otherwise sends
            # every non-notify message with disable_notification=True, and users
            # report the silent brief as "never delivered"). Set to false to
            # restore silent (no-push) cron deliveries.
            "notify": True,
        },
        # Make cron deliveries CONTINUABLE (user can reply to a brief with it in context). False
        # keeps deliveries isolated to the job's session; per-job `attach_to_session` overrides.
        # Thread-capable platforms (Telegram topics, Discord/Slack threads) get a seeded thread per
        # job via create_handoff_thread; DM-only platforms mirror the brief into the target DM
        # session. Appended at a turn boundary via mirror_to_session, cached system prompt
        # untouched. User-written bare platforms address home conversations, unlike `all`
        # broadcast expansions, which do not gain mirror eligibility.
        "mirror_delivery": False,
        # Max due jobs run in parallel per tick. None/0 = unbounded (thread count only); 1 = serial.
        # Env override: HERMES_CRON_MAX_PARALLEL.
        "max_parallel_jobs": None,
        # save_job_output keeps the N most recent .md files per job; 0 or negative disables pruning
        # (for externally managed cleanup).
        "output_retention": 50,
        # Timeout (seconds) for a no-agent cron script. Env: HERMES_CRON_SCRIPT_TIMEOUT. Keep in
        # sync with cron.scheduler._DEFAULT_SCRIPT_TIMEOUT.
        "script_timeout_seconds": 3600,
        # Timeout (seconds) for SessionDB() init inside cron jobs: state.db open/migrate has no
        # timeout of its own against a wedged sqlite3.connect, and an unbounded hang wedges the
        # job's dispatch guard forever. Env: HERMES_CRON_SESSION_DB_TIMEOUT. 0 = unlimited.
        "session_db_timeout_seconds": 10,
        # Timeout (seconds) per media attachment send during gateway delivery; large attachments
        # (long TTS audio, big exports) need more than 30s. Env: HERMES_CRON_MEDIA_SEND_TIMEOUT.
        # Keep in sync with cron.scheduler._DEFAULT_MEDIA_SEND_TIMEOUT.
        "media_send_timeout_seconds": 300,
        # Managed systemd gateway with no user session (containers, no linger): false runs
        # cron jobs as a direct external subprocess (warns once; no cgroup isolation), true
        # fails closed with the enable-linger remedy. Kanban always requires a scope.
        "require_restart_safe_scope": False,
    },
    # Kanban multi-agent coordination. The dispatcher ticks every N seconds, reclaims stale claims,
    # promotes dependency-satisfied todos to ready, and fires `hermes -p <assignee> chat -q ...` per
    # claimable task. Run ONE dispatcher per profile; two on the same kanban.db race for claims.
    "kanban": {
        # Auto-subscribe the originating gateway/TUI session to completion + block events when
        # kanban_create is called from a session with a persistent delivery channel. Disable for
        # profiles that prefer explicit kanban_notify-subscribe calls per task.
        "auto_subscribe_on_create": True,
        # Poll and deliver Kanban subscriptions from this gateway. Disable on profiles that do
        # not own notification subscriptions to avoid an idle five-second board probe.
        "notify_in_gateway": True,
        # Run the dispatcher inside the gateway process (~300µs per idle tick). False only if you
        # run it as a separate unit or don't want the gateway spawning workers.
        "dispatch_in_gateway": True,
        # Auto-claim tasks in the review column and spawn the assigned profile with the bundled
        # sdlc-review skill. Disable where every review is done manually from the dashboard.
        "review_dispatch": True,
        # Seconds between dispatcher ticks. Lower = snappier pickup; higher = less SQL pressure.
        "dispatch_interval_seconds": 60,
        # Auto-block after this many consecutive non-success attempts (spawn_failed, timed_out,
        # crashed) for the same task/profile. Reassignment resets the streak.
        "failure_limit": 2,
        # Worker stdout/stderr log rotation at spawn time (2 MiB + one backup). Raise to keep more
        # early failure evidence from long-running workers.
        "worker_log_rotate_bytes": 2 * 1024 * 1024,
        "worker_log_backup_count": 1,
        # Profile for the root/orchestration task after Triage decomposition; "" = default profile.
        # Does not control the decomposer LLM path (see auxiliary.kanban_decomposer).
        "orchestrator_profile": "",
        # Assignee when the orchestrator can't match one to an installed profile; "" = default
        # profile. A task never ends up with assignee=None.
        "default_assignee": "",
        # Global cap: positive int = the HOST never has more than N tasks 'running' across all
        # boards and both dispatch lanes. None = ~MemTotal / 512 MiB clamped to [2, 8]; where
        # MemTotal is unreadable (macOS/Windows) None means no cap.
        # Global concurrency cap (#33488): when set to a positive int, the HOST never has more than N tasks
        # in 'running' at once — counted across every active board and across both the ready and review
        # dispatch lanes (workers are OS processes sharing one machine's memory, so the cap bounds the
        # machine, not each board; OOF-30). Unset (None) means "derive from system memory" (OOF-30/OOF-77):
        # the dispatcher caps concurrency at roughly MemTotal / 512 MiB, clamped to [2, 8] — e.g. 2 workers
        # on a 1 GiB VM. On hosts where total memory can't be read (macOS/Windows), unset falls back to no
        # cap. Set an explicit value to override the derived default in either direction.
        "max_in_progress": None,
        # Per-profile cap: positive int = no single profile runs more than N workers even if the
        # global caps allow; blocked tasks defer to the next tick. None = no per-profile cap. Useful
        # when fan-out would saturate one profile's model/API quota/browser pool.
        # Unset (None) means "no per-profile cap" — backward-compatible with existing installs. Useful for
        # fan-out workflows that would otherwise saturate one profile's local model / API quota / browser
        # pool while leaving other profiles idle. See #21582.
        "max_in_progress_per_profile": None,
        # Per-home claim allowlist for boards shared across Hermes homes (#110995): profile names
        # this home's dispatcher may claim (list or comma-separated string). None = any existing
        # profile is claimable. Set = fail-closed (an empty list claims nothing). Every home has a
        # root profile named "default", so on a shared kanban.db every home can otherwise claim
        # default-assigned cards.
        "dispatch_profiles": None,
        # Auto-run the decomposer on Triage tasks every tick. False = manual via `hermes kanban
        # decompose <id>` or the dashboard's Decompose button.
        "auto_decompose": True,
        # Max triage tasks decomposed per tick, bounding the aux-LLM burst from a bulk load. Excess
        # defers to the next tick.
        "auto_decompose_per_tick": 3,
        # Running tasks with no heartbeat (last_heartbeat_at) for this many seconds are reclaimed to
        # ready on the next tick; a still-running local worker is terminated first. 0 = off.
        "dispatch_stale_timeout_seconds": 14400,
        # Each tick, requeue 'running' cards with broken claim bookkeeping (claim_lock or
        # claim_expires NULL with a dead worker) that TTL/crash/stale recovery can't see. False
        # keeps orphans frozen for manual forensics.
        "reconcile_orphans": True,
        # Notify subscriptions survive `done` (completion is reversible) and are removed on archive.
        # On boards that never archive, the notifier GC purges subscriptions for tasks done with no
        # activity for this many days so stale rows aren't scanned forever. 0 = off.
        "done_sub_retention_days": 30,
    },

    # Bot Mode cross-connection relay (tools/bot_relay.py). Envelopes queued
    # by message_agent for agents on other connections wait in an on-disk
    # outbox until the Desktop drains them.
    "bot_mode": {
        # Drain-time TTL (seconds): an envelope older than this is NOT
        # delivered when the Desktop finally drains the outbox — the sender
        # gets an error reply (reason 'queued_expired') instead, so a DM
        # written while the Desktop was away can't land hours late as a
        # confusing zombie message. 0 disables drain-time expiry (the 6h
        # stale-artifact sweep still applies).
        "envelope_ttl_seconds": 900,
        # How long a second delivery into an already-busy target profile
        # queues behind the current turn before failing with a structured
        # 'target_busy' error. Deliveries are serialized per profile with a
        # cross-process file lock so two turns never race one Bot Chat.
        "turn_wait_seconds": 120,
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
        # Kernel lifetime:
        #   per-call (default) — a fresh child process per execute_code call;
        #     no state carries over. Today's behavior.
        #   session            — one persistent kernel per (session owner,
        #     mode, interpreter, cwd, tool-set): variables, imports, and
        #     loaded data survive across calls AND across the user turns of
        #     one conversation (the owner is the conversation's approval
        #     session key; delegated subagent sessions get their own).
        #     Kernels are disposed with their session (session clear/new),
        #     reaped after kernel_idle_timeout seconds idle, and capped at
        #     max_session_kernels live children process-wide (LRU evicted).
        #     A timed-out or interrupted cell kills the kernel (state lost,
        #     next call starts fresh), and the child environment is frozen
        #     at kernel spawn — pass reset=true after changing env
        #     passthrough. Tool RPC authority is rebound to each cell: a
        #     later cell's tool calls run under that cell's approval and
        #     session context, never the first cell's. Security scrubbing,
        #     the tool whitelist, and output redaction are identical in
        #     both modes. NOTE for per-script static policy (see
        #     check_execute_code_guard): a persistent namespace lets cell
        #     N+1 reach objects cell N created, which a single-cell static
        #     scan cannot see — the runtime RPC boundary (allow-list, call
        #     budget, per-cell authority) is the operative cross-cell
        #     enforcement in this mode.
        # kernel_mode is retired: session kernels are always on for local
        # execution. Remote terminal backends still run per-call — their
        # file-based RPC path has no kernel host yet (tracked follow-up,
        # not a design limit). A leftover kernel_mode key in user config
        # is ignored.
        # Lifecycle bounds for session kernels.
        "kernel_idle_timeout": 1800,
        "max_session_kernels": 4,
    },
    "code_execution": {  # execute_code settings (programmatic tool calls).
        # project = run in the session cwd with the active venv/conda python so project deps and
        # relative paths resolve. strict = isolated temp dir with hermes-agent's own python
        # (sys.executable): max isolation, project deps/relative paths won't work. Env scrubbing
        # (*_API_KEY, *_TOKEN, *_SECRET, ...) and the tool whitelist apply in both modes.
        "mode": "project",
        # Session kernels are always on locally (`kernel_mode` is ignored) and remotely
        # (tools/code_kernel_remote.py; a backend that cannot spawn a kernel fails open to
        # per-call). One kernel per (session owner, mode, interpreter, cwd, tool-set) keeps state
        # across calls and turns; subagents get their own. Kernels die with the session, after
        # kernel_idle_timeout idle seconds, or by LRU eviction past max_session_kernels. A
        # timed-out/interrupted cell kills the kernel; env is frozen at spawn (reset=true after
        # changing passthrough). Tool RPC authority (approval, session, allow-list, call budget) is
        # rebound per cell — that runtime boundary is the cross-cell enforcement.
        "kernel_idle_timeout": 1800,
        "max_session_kernels": 4,
    },
    # Tool Search: deferrable (MCP / non-core plugin) tools are replaced in the model-facing array
    # by tool_search / tool_describe / tool_call bridges and surfaced on demand. Core Hermes tools
    # (terminal, file tools, todo, memory, browser_*, ...) are NEVER deferred.
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
            #   Today "auto" is an alias of "on"; it stays the default so a
            #   future budget-gated mode (inline schemas when they fit, defer
            #   only when they don't) can land on "auto" without changing
            #   behavior for anyone who pinned "on" or "off" explicitly.
            # "off" — disable entirely. Tools-array assembly is a pass-through.
            "enabled": "auto",
            # Listing budget as % of the model's context length; effective budget = min(this % of
            # context, listing_max_tokens). Range 0..100.
            "threshold_pct": 5,
            # When the model calls tool_search without a ``limit`` argument,
            # how many hits to return PER QUERY. Range 1..max_search_limit.
            "search_default_limit": 5,
            # Hard upper bound the model can request via ``limit`` (per
            # query). Range 1..50.
            "max_search_limit": 25,
            # Skills-style catalog listing embedded in the tool_search bridge
            # description: every deferred tool's name + first sentence of its
            # description (≤60 chars), grouped by MCP server / toolset. Keeps
            # capabilities discoverable while schemas stay deferred.
            # "auto" (default) — include when the listing fits the budget
            #   (falls back to names-only, then to the bare tier-2 bridge).
            # "on"  — same rendering, but explicit intent to always list.
            # "off" — always the bare bridge (tier 2 for every catalog).
            "listing": "auto",
            # Absolute cap on the embedded listing in tokens (chars/4), regardless of context size.
            # Range 200..60000.
            "listing_max_tokens": 4000,
        },
        # Remote connector discovery/lifecycle through the Nous tool gateway.
        # The flag is the user's off switch; availability additionally requires
        # the portal sign-in every managed tool gates on.
        "connectors": {"enabled": True},
    },
    "logging": {  # File logging to ~/.hermes/logs/: agent.log captures INFO+, errors.log WARNING+.
        "level": "INFO",       # minimum level for agent.log: DEBUG, INFO, WARNING
        "max_size_mb": 5,      # max size per log file before rotation
        "backup_count": 3,     # rotated backups to keep
    },
    # Remote model-catalog manifest: curated OpenRouter / Nous Portal model lists fetched from this
    # URL (falls back to the in-repo snapshot on network failure), so picker lists update without a
    # release. Default URL is served by the docs-site GitHub Pages deploy.
    "model_catalog": {
        "enabled": True,
        "url": "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json",
        # Disk cache TTL in minutes.  The gateway refreshes the catalogs on
        # this cadence in the background; the CLI refetches on the next
        # /model or `hermes model` invocation once the cache is older than
        # this.  Network failures silently fall back to the stale cache.
        # (Legacy `ttl_hours` is still honoured when set explicitly.)
        "ttl_minutes": 20,
        # Optional per-provider override URLs for third parties that want
        # to self-host their own curation list using the same schema.
        # Example:
        #   providers:
        #     openrouter:
        #       url: https://example.com/my-curation.json
        "providers": {},
    },
    # Per-model metadata overrides. Fields: context_window, supports_tools,
    # supports_vision, supports_reasoning, model_family. <provider>.<model_id> wins over
    # models.dev/OpenRouter/hardcoded defaults for the fields it sets (chain order in
    # agent/model_metadata.py). <provider>._default and top-level _default fill gaps ONLY for models
    # the catalog does not know, so they never clamp known models. Unknown ids start from safe
    # defaults (200K context, tools on, vision/reasoning off) and get patched. Provider keys: Hermes
    # or models.dev id; model ids match case-insensitively. Example: {"custom:my-local-vllm":
    # {"my-llava-model": {"context_window": 8192}}}
    # Semantics: 1. NOTE: an explicit model.context_length (global) and a custom_providers per-model
    # context_length are user settings at other layers and are consulted in the resolution chain order
    # documented in agent/model_metadata.py. 2. See #84482, #8731.
    "model_overrides": {},
    # models.dev registry (context windows, capabilities, pricing, modalities): fetched on startup,
    # served from cache, refreshed by a background daemon with ETag conditional GET. Override `url`
    # to point at a mirror (e.g. behind a corporate proxy).
    "models_dev": {
        "url": "",  # empty = default https://models.dev/api.json
    },

    "network": {
        # Force IPv4. With broken/unreachable IPv6, Python tries AAAA first and hangs for the full
        # TCP timeout before falling back. True skips IPv6 entirely.
        "force_ipv4": False,
    },
    # Gateway monitoring: service health + redacted operational diagnostics exported over OTLP to an
    # operator endpoint (OTEL Collector, DataDog, ...). Content-free by construction: no prompts,
    # messages, tool args/results, session history, usage analytics, audit logs, or trajectories.
    # Nothing is collected or sent until enabled with an endpoint.
    "monitoring": {
        # Stable install identifier on exported signals so operators can tell instances apart. "" =
        # mint a fresh UUID on first use; clear it to rotate. Carries no account identity.
        "install_id": "",
        "gateway_health_export": {  # Gateway health & diagnostics export.
            "enabled": False,
            "metrics_enabled": True,
            "diagnostic_events_enabled": True,
            "warning_error_events_enabled": True,
            "export_interval_seconds": 60,
            "logs_export_interval_seconds": 5,
            "resource_attributes": {
                "service.name": "hermes-gateway", "deployment.environment.name": "production"
            },
        },
        # OTLP destination. headers_env maps header names to ENVIRONMENT VARIABLE NAMES (never
        # secret values); values are read from the environment at export time.
        "export": {"otlp": {"enabled": False, "endpoint": "", "headers_env": {}}},
    },
    "gateway": {  # Gateway settings (messaging platforms: Telegram, Discord, Slack, ...).
        # Seconds to let a SIGTERM-interrupted gateway agent unwind before adapter/database
        # teardown. Keep short so service-manager shutdowns don't exhaust their stop budget.
        "signal_interrupt_grace_timeout": 1,
        # Durable delivery-obligation ledger: final responses are recorded in state.db around the
        # platform send; a gateway that died between finalize and platform ACK redelivers on next
        # boot (ambiguous cases carry a "recovered reply — may be a duplicate" marker;
        # at-least-once). Disable to lose in-flight final responses on crash/restart.
        "delivery_ledger": True,
        # Seconds to wait for one platform to connect at startup/reconnect; raise on "discord
        # connect timed out" loops (many slash commands to sync). 0/negative = wait forever. Bridged
        # to HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT, which wins if set explicitly.
        # Seconds the gateway waits for a single messaging platform to finish connecting during startup (and
        # on reconnect). Discord in particular can blow past the old fixed 30s when an account has many
        # slash commands to sync (#19776: 90-173 skills → ~28-31s sync). Raise this if your gateway hits
        # "discord connect timed out" / "Timeout waiting for connection to Discord" restart loops. ``0`` or
        # negative disables the timeout entirely (wait indefinitely).
        "platform_connect_timeout": 30,
        # Event-loop liveness watchdog: a daemon thread probes the asyncio loop; after consecutive
        # missed probes it dumps all-thread stacks and hard-exits with the service-restart code so
        # systemd/launchd revives the process instead of leaving a wedged-but-alive zombie.
        # Set to false to disable. See #69089.
        "loop_watchdog": True,

        # Loop-liveness watchdog tuning (defaults mirror
        # gateway/shutdown_watchdog.py constants). probe_interval = seconds
        # between liveness probes; probe_timeout = seconds a probe may go
        # unprocessed before counting as a miss; max_strikes = consecutive
        # misses before the watchdog hard-exits 75 for a service respawn
        # (~90-120s of sustained loop block at the defaults).
        "loop_watchdog_probe_interval_s": 30.0,
        "loop_watchdog_probe_timeout_s": 10.0,
        "loop_watchdog_max_strikes": 3,
        # Bot-to-bot loop guard: admitted bot messages per conversation before a cooldown.
        "bot_loop_guard": {"enabled": True, "max_events": 20, "window_seconds": 300, "cooldown_seconds": 600},
        # Startup-liveness watchdog: stdlib-only daemon thread armed at process entry that
        # hard-exits 75 if the loop isn't live within the deadline. Armed before config loads, so
        # run_gateway() bridges these to HERMES_STARTUP_WATCHDOG / HERMES_STARTUP_WATCHDOG_TIMEOUT_S
        # and re-arms the live handle; explicit env wins.
        "startup_watchdog": True,
        "startup_watchdog_timeout_seconds": 300,

        # Whether the gateway keeps writing the legacy sessions.json mirror of
        # its routing index. The primary copy lives in state.db (the
        # gateway_routing table). Default True for backward compatibility with
        # external tooling and downgrade safety; set to false to stop
        # producing ~/.hermes/sessions/sessions.json entirely.
        "write_sessions_json": True,
        # One gateway for every profile on this host: the DEFAULT profile's gateway also connects
        # each named profile's bots (their own .env / config.yaml, per-profile secret scope) and
        # stamps the profile into session keys. Flip with `hermes gateway migrate --multiplex`
        # (records a rollback manifest; `--standalone` undoes it) or `hermes config set
        # gateway.multiplex_profiles true` + `hermes gateway restart`. GATEWAY_MULTIPLEX_PROFILES
        # in the environment overrides. Two profiles configuring the same bot token cannot be
        # served together — the duplicate adapter is parked; `hermes profile create --clone`
        # therefore leaves messaging channels behind unless --clone-channels is passed.
        "multiplex_profiles": False,
        # Route inbound chats of the default profile's bots to another profile
        # (gateway/profile_routing.py): [{profile, platform, chat_id|user_id|guild_id|...}].
        # Most-specific match wins; only read by the multiplexing default gateway.
        "profile_routes": [],
        # Scale-to-zero idle TIMEOUT only. When an instance is opted in via the NAS "Labs" toggle
        # (HERMES_SCALE_TO_ZERO env stamp) AND messaging is relay-only/absent AND a wakeUrl is
        # registered, the relay transport goes dormant so the platform (e.g. Fly autostop) can
        # suspend the machine; it wakes on the wakeUrl poke. Enablement is the Labs toggle, never a
        # config key. 0/negative = default.
        "scale_to_zero": {"idle_timeout_minutes": 2},
        # Auto-resume restart-loop breaker. A supervisor-revived gateway auto-resumes the
        # SIGTERM-interrupted session; if that turn keeps triggering the kill, boots no more than
        # `max_gap_seconds` apart (floored by `window_seconds`) chain, and after `max_restarts`
        # auto-resume is SKIPPED for that boot (inbound messages still served). Gap-based chaining
        # also catches SLOW ~150s crash cycles. max_restarts=0 disables.
        "restart_loop_guard": {"max_restarts": 3, "window_seconds": 60, "max_gap_seconds": 300},
        # Respawn-storm circuit breaker (complements restart_loop_guard): counts (re)starts in a
        # sliding window and sleeps an exponential backoff before booting so a crash-looping
        # supervisor can't hammer the process. max_starts <= 0 disables. Env escape hatches:
        # HERMES_GATEWAY_MAX_STARTS / HERMES_GATEWAY_START_WINDOW_S.
        "respawn_storm": {"max_starts": 5, "window_seconds": 120},
        # Prefix user messages IN THE MODEL'S CONTEXT with a timestamp (e.g. "[Tue 2026-04-28
        # 13:40:53 CEST]") for temporal awareness. Persisted transcripts stay clean (timestamp is
        # message metadata regardless), so enabling later surfaces past send-times too.
        "message_timestamps": {"enabled": False},
        # Max bytes of inbound image/audio/video the gateway buffers into RAM and caches to disk.
        # Media is read fully into memory first, so unbounded uploads (Discord Nitro: 500 MB) or
        # huge remote URLs can OOM-kill constrained deployments. Enforced in
        # gateway/platforms/base.py for every adapter. 0 = no cap. Default 128 MiB.
        "max_inbound_media_bytes": 134217728,

        # Whether gateway platform adapters let aiohttp read proxy settings
        # (HTTP_PROXY / HTTPS_PROXY / NO_PROXY, plus SSL_CERT_FILE) from the
        # process environment, and whether generic proxy env / the macOS
        # system proxy are auto-detected for adapter clients. Set to false
        # when the gateway inherits a proxy it must not use — e.g. a Windows
        # Scheduled Task picking up a Clash/V2Ray HTTP_PROXY the interactive
        # shell never sees, producing "Cannot connect to host 127.0.0.1:7890"
        # poll loops (#48820). Explicit per-platform vars (DISCORD_PROXY,
        # TELEGRAM_PROXY, ...) are still honored. One knob for every adapter.
        "trust_env": True,

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
        # Extra roots (project/scratch dirs, mounted shares) from which bare file paths may be
        # uploaded; the Hermes cache is always trusted. List of absolute paths or one
        # os.pathsep-separated string; tildes expanded. Bridged to HERMES_MEDIA_ALLOW_DIRS. Honored
        # in both modes.
        "media_delivery_allow_dirs": [],
        # Trust files whose mtime is within trust_recent_files_seconds even outside the cache/
        # allowlist (e.g. `pandoc -o /tmp/report.pdf`); system paths stay blocked. False =
        # pure-allowlist mode. Bridged to HERMES_MEDIA_TRUST_RECENT_FILES. Only consulted when
        # strict is true.
        "trust_recent_files": True,
        # Recency window in seconds; 600 covers a multi-tool turn. Bridged to
        # HERMES_MEDIA_TRUST_RECENT_SECONDS. Only consulted when strict is true.
        "trust_recent_files_seconds": 600,
        "api_server": {  # OpenAI-compatible API server platform (gateway/platforms/api_server.py).
            # Max concurrent agent runs. Requests to /v1/chat/completions, /v1/responses, and
            # /v1/runs beyond this get HTTP 429 + Retry-After, bounding CPU/memory/LLM-quota
            # exhaustion from a request flood. 0 = no cap.
            "max_concurrent_runs": 10,
        },
    },
    # Real-time token streaming to messaging platforms (gateway; restart after enabling). Off by
    # default: costs extra edit/draft API calls per response.
    "streaming": {
        "enabled": False,  # When false, each response is delivered as one final message.
        # auto = native drafts where supported (Telegram DMs, Bot API 9.5+), edits elsewhere
        # (Discord, Slack, Matrix, Telegram groups); draft = drafts with edit fallback; edit =
        # progressive editMessageText only; off = disabled.
        "transport": "auto",
        # Minimum seconds between progressive edits (Telegram's ~1 edit/s flood envelope).
        "edit_interval": 0.8,
        # Flush the buffer once this many chars accumulate, so short replies feel instant.
        "buffer_threshold": 24,
        "cursor": " \u2589",  # Cursor glyph appended to the in-progress message.
        # Telegram only: when >0, if the preview was visible at least this many seconds the final
        # edit is sent as a fresh message so the timestamp reflects completion.
        "fresh_final_after_seconds": 0.0,
    },
    # Automatic cleanup of ~/.hermes/state.db, which otherwise grows without bound and slows FTS5
    # inserts, /resume listing, and insights queries.
    "sessions": {
        # When true, prune ENDED sessions inactive for retention_days once
        # per (roughly) min_interval_hours at CLI/gateway/cron startup.
        # Activity is the latest message timestamp, falling back to creation
        # time for empty sessions. Sessions that are still open, pinned, or
        # mid-turn are never deleted — the only open rows the sweep touches
        # are stale automation sessions (cron/kanban/subagent/one-shot CLI)
        # whose process died without closing them; those are *closed*, not
        # deleted, and get a further full retention window before removal.
        # Default true since #54189: without it state.db grows without bound
        # (multi-GB installs reported within weeks).  Set false to keep every
        # ended session forever.
        "auto_prune": True,
        # How many inactive days of ended-session history to keep. Matches
        # the default of ``hermes sessions prune``.
        "retention_days": 90,
        # Auto-archive (soft-hide, never delete) sessions with no activity for auto_archive_days,
        # once per min_interval_hours. Pinned sessions are exempt.
        "auto_archive": False,
        # Idle days before auto-archive hides a session (only when auto_archive is true).
        "auto_archive_days": 3,
        # VACUUM after a prune that actually deleted rows.  SQLite does not
        # reclaim disk space on DELETE — freed pages are just reused on
        # subsequent INSERTs — so without VACUUM the file stays bloated
        # even after pruning.  VACUUM blocks writes for a few seconds per
        # 100MB, so it only runs at startup, and only when prune deleted
        # ≥1 session AND the reclaimable fraction of the file
        # (PRAGMA freelist_count / page_count) exceeds 25% — a dense DB
        # never pays for a full rewrite to reclaim a few MB (#54189).
        "vacuum_after_prune": True,
        # Minimum days between VACUUM rewrites; pruning keeps its normal cadence.
        "min_vacuum_interval_days": 30,
        # Minimum hours between auto-maintenance runs (tracked in state.db state_meta, shared across
        # processes).
        "min_interval_hours": 24,

        # Notice about the compact FTS layout (reclaims ~60%+ of state.db). OPT-IN: legacy indexes
        # stay until `hermes sessions optimize-storage` runs, since the rebuild is disk-heavy on
        # large DBs. advise = `hermes update` prints a one-line notice with reclaimable size when a
        # legacy index is detected; require = shown as a REQUIRED upgrade (tooling may gate on it);
        # off = none.
        "fts_optimize_notice": "advise",
        # CJK-bigram search index (messages_fts_cjk). When the extension is built
        # (native/fts5_cjk/build.sh → ~/.hermes/lib/libfts5_cjk.so), 1-2 char CJK terms get exact
        # index matches instead of LIKE scans. True = use when present (inert otherwise); False =
        # never load/serve it. Bridged to HERMES_CJK_FTS.
        "cjk_fts": True,
        # Slow session-search threshold (ms): searches at/above it log one INFO line with the
        # routing path (fts_cjk / fts5 / trigram / like_scan). 0 logs every search. Bridged to
        # HERMES_SEARCH_SLOW_MS.
        "search_slow_ms": 1000,
        # Transcript guards (a runaway 100k+ row session can exhaust memory when materialized at
        # once; 0 disables). Max active messages (across the compression lineage) for interactive
        # resume.
        "max_resume_messages": 20000,
        # Max active messages per session for in-memory export (`hermes sessions export`); checked
        # per session, so full-DB backups of small sessions work.
        "max_export_messages": 20000,
    },
    # First-touch onboarding hints (agent/onboarding.py). Each hint shows once and is latched under
    # `seen`; wipe the section to re-see all hints.
    "onboarding": {
        "seen": {},
        # First-ever gateway message: ask = offer to build a user profile (consent- gated; never
        # reads connected accounts silently); off = plain intro only.
        "profile_build": "ask",
    },
    # Privacy-safe aggregate metrics in this profile's local telemetry dir. Collection (`enabled`)
    # and transmission to Nous (`send`) are SEPARATE opt-ins; see
    # website/docs/developer-guide/relay-shared-metrics.md Appendix A for consent/retention.
    "telemetry": {
        "shared_metrics": {
            "enabled": False,
            # Transmit exported packages to the Nous telemetry service.
            # Requires ``enabled``: it never switches collection on by itself,
            # and ``send`` without ``enabled`` is logged as an error rather
            # than silently doing nothing. A package is only sent when its
            # whole period falls inside a recorded consent window, so data
            # collected before consent — or while it was withdrawn — stays
            # local.
            "send": False,
            # Ingest endpoint. Production by default; override for staging or
            # a local test server. Deliberately NOT overridable by an
            # environment variable: that would let an inherited value silently
            # redirect telemetry a user consented to send to Nous. Non-HTTPS
            # is refused unless the host is localhost.
            "endpoint": "https://telemetry.nousresearch.com/v1/telemetry",
        },
    },

    "doctor": {
        # Per-probe timeout (seconds) for `hermes doctor --live` real-call probes.
        "live_probe_timeout": 10,
    },

    "updates": {
        # Passive version/banner checks only; explicit `hermes update --check` remains enabled.
        "check": True,
        # Pre-update backup. quick = snapshot small critical state (pairing JSONs, cron jobs,
        # config.yaml, .env, auth.json, profile DBs) into <HERMES_HOME>/state-snapshots/, skipping
        # files >1 GiB; restore via ``/snapshot``. full = quick PLUS a ``hermes backup`` zip in
        # <HERMES_HOME>/backups/ (``hermes import`` restores; slow on large homes; ``--backup``
        # forces once). off = none (``--no-backup`` forces once). Legacy booleans: true -> full,
        # false -> off.
        # Pre-update safety backup — ONE consolidated mechanism, three modes: Files over 1 GiB (e.g. a
        # bloated state.db) are skipped with a warning so the snapshot stays fast. This is the #48200
        # (wrong-path wipe) safety net.
        "pre_update_backup": "quick",
        # Full backup zips to retain (older pruned after each success; floored to 1 so the newest is
        # always kept). The quick snapshot always keeps exactly 1.
        "backup_keep": 5,
        # Uncommitted source-tree changes during NON-interactive updates (desktop, gateway — no TTY;
        # interactive updates always stash and ask). stash = stash, pull, restore on top (conflicts
        # stay in a git stash). discard = stash and drop after the pull (stash-and-drop, not reset
        # --hard + clean -fd, so ignored paths like node_modules/venv are never touched).
        "non_interactive_local_changes": "stash",
        # When `hermes update` finds the source checkout parked on a feature
        # branch (left behind by tooling or a manual checkout), switch back
        # to the update target automatically whenever the working tree is
        # clean. Committed-but-unmerged work is safe — `git checkout` never
        # discards commits; the branch keeps them and the update prints a
        # loud notice naming the branch and count. This keeps non-
        # interactive updates (desktop update button, gateway /update,
        # cron) working: they have no way to resolve a skip. Only a DIRTY
        # tree (uncommitted changes) blocks the switch — the code update is
        # then SKIPPED with a loud warning instead of pretending success
        # (2026-08-17 incident: "✓ Code updated!" printed while the
        # checkout stayed days behind main on a stale branch). Set false to
        # never auto-switch.
        "auto_switch_parked_branch": True,
        # HOW a clean parked branch with unmerged commits is handled:
        #   "switch" (default)  — switch to the update target; the commits
        #                         stay on the branch (git checkout never
        #                         discards committed work) and a loud notice
        #                         names the branch + count. Deterministic —
        #                         never conflicts — so desktop/gateway/cron
        #                         updates always land on current code.
        #   "update_in_place"   — for a deliberately maintained custom branch
        #                         (local patches on top of main): merge
        #                         origin/<target> INTO the branch instead.
        #                         The checkout never moves and local commits
        #                         survive; a conflict stops the update
        #                         cleanly with nothing changed. A safety tag
        #                         (pre-update-<stamp>) is left before the
        #                         merge. `hermes update --switch-branch`
        #                         overrides back to the switch path for one
        #                         run (e.g. a deep feature branch that must
        #                         not accumulate update merge commits).
        "parked_branch_strategy": "switch",
        # Refresh an already-installed cua-driver during `hermes update`.
        # The refresh is best-effort and macOS-only. Turn this off if the
        # upstream installer is not appropriate for the machine, for example
        # on non-admin accounts where `/Applications` is not writable.
        "refresh_cua_driver": True,
    },
    # LSP diagnostics (pyright, gopls, rust-analyzer...) in the post-write lint check of
    # write_file/patch. Runs only when the cwd or edited file is inside a git worktree; otherwise
    # dormant and the in-process syntax check is the only tier.
    "lsp": {
        "enabled": True,  # False disables the whole subsystem: no servers, no event loop, no cost.
        # document = wait up to wait_timeout seconds for the current file's diagnostics; full = also
        # request workspace-wide diagnostics (slower).
        "wait_mode": "document",
        "wait_timeout": 5.0,
        # Missing server binaries: auto = install via npm/go/pip into <HERMES_HOME>/lsp/bin/ on
        # first use; manual = only binaries on PATH; off = alias for manual.
        "install_strategy": "auto",
        # Idle seconds before a server is shut down (respawned on demand), so long- running
        # processes don't accumulate stale children (hundreds of MB + pipe FDs each) across
        # worktrees. 0 = keep servers for process lifetime.
        "idle_timeout": 600.0,
        # Per-server overrides keyed by registry server_id (pyright, gopls...): disabled: true;
        # command: ["path/to/server", "--stdio"] (bypasses auto- install); env: {...};
        # initialization_options: {...} (merged into LSP initializationOptions).
        "servers": {},
    },
    # X (Twitter) Search via xAI's x_search Responses tool. Registers when xAI creds exist
    # (SuperGrok OAuth or XAI_API_KEY) AND the toolset is enabled in `hermes tools`.
    "x_search": {
        # xAI model for the Responses call; any Grok model with x_search access works.
        "model": "grok-4.5",
        # Reasoning effort for models that support it; null keeps the model default.
        "reasoning_effort": None,
        # Request timeout in seconds (minimum 30); complex queries can take 60-120s.
        "timeout_seconds": 180,
        # Retries on 5xx / ReadTimeout / ConnectionError (backoff 1.5x attempt s, cap 5s).
        "retries": 2,
    },
    # External secret sources — pull credentials from secret managers at startup instead of storing
    # them in ~/.hermes/.env.
    # Browser credential vault: which login sources browser_vault_list/fill may draw from. The local
    # encrypted vault (`hermes vault add`, Desktop → Settings → Credential Vault) is always on.
    # External password managers are unlocked per session with a masked master-password prompt;
    # headless sessions (cron, webhook, API) never prompt and see them as locked.
    "vault": {
        "onepassword": {
            "enabled": False,       # `op` CLI: Login items with a website URL become fillable handles.
            "account": "",          # account shorthand for `op --account`; empty = default account.
            "binary_path": "",      # absolute path to op; empty = PATH.
            # Env var holding a service-account token (headless auth, no unlock prompt). Unset = prompt.
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
        },
        "bitwarden": {
            "enabled": False,       # `bw` CLI (Password Manager, not Secrets Manager); run `bw login` once first.
            "binary_path": "",      # absolute path to bw; empty = PATH.
        },
    },
    "secrets": {
        # Optional ordering of enabled sources (e.g. [onepassword, bitwarden]); default registration
        # order. Mapped sources (explicit VAR→ref) always beat bulk sources (BSM project dumps);
        # first claim on a var wins, later ones warn. "sources": [],
        "bitwarden": {
            "enabled": False,  # When false, BSM is never contacted and bws is never auto-installed.
            # Env var holding the machine-account token — the one bootstrap secret; lives in
            # ~/.hermes/.env (or the shell), never in config.yaml.
            "access_token_env": "BWS_ACCESS_TOKEN",
            "project_id": "",  # UUID of the BSM project to sync from.
            # Seconds to reuse a fresh disk/memory cache entry before contacting Bitwarden again. 0
            # disables fresh-cache reuse.
            "cache_ttl_seconds": 300,
            # Last-good fallback for NETWORK/TIMEOUT outages: AES-GCM cache under ~/.hermes/cache/,
            # reused up to max_stale_seconds. Auth failures never fall back.
            "encrypted_cache": {"enabled": False, "max_stale_seconds": 0},
            # BSM values overwrite existing env vars, so rotating in Bitwarden takes effect without
            # clearing the matching .env line.
            "override_existing": True,
            # Auto-download bws into ~/.hermes/bin/ on first use; False = bws must be on PATH.
            "auto_install": True,
            # Passed to bws as BWS_SERVER_URL. Empty = US Cloud (bws default);
            # https://vault.bitwarden.eu for EU; your own URL for self-hosted.
            "server_url": "",
        },
        "onepassword": {
            "enabled": False,  # When false, the op CLI is never invoked.
            "env": {},  # env-var name → op://vault/item/field; each resolved with one `op read`.
            # Account shorthand / sign-in address for `op read --account`; empty = default.
            "account": "",
            # Env var holding a service-account token for headless auth (exported to op as
            # OP_SERVICE_ACCOUNT_TOKEN). Unset = interactive/desktop op session.
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
            # Absolute path to op, used verbatim (avoids trusting PATH). Empty = PATH.
            "binary_path": "",
            # Seconds to cache values in-process and on disk; 0 disables BOTH layers.
            "cache_ttl_seconds": 300,
            # Overwrite existing env vars so rotation takes effect; False lets .env win.
            "override_existing": True,
        },
    },
    # Paste collapse thresholds (TUI + CLI); 0 disables each. threshold: bracketed pastes with this
    # many newlines collapse to a file reference. fallback: same test for terminals without
    # bracketed paste, gated by chars/newlines-added heuristics. char_threshold: single-line pastes
    # this long collapse too.
    "paste_collapse_threshold": 5,
    "paste_collapse_threshold_fallback": 5,
    "paste_collapse_char_threshold": 2000,

    "computer_use": {
        # cua-driver's upstream PostHog telemetry defaults ON; Hermes sets
        # CUA_DRIVER_RS_TELEMETRY_ENABLED=0 in every child env unless this is true.
        "cua_telemetry": False,
        "native_wayland": False,
        # Cap driver screenshot longest edge (pixels) via set_config on
        # session start. Shrinks SOM multimodal payloads; 0 disables.
        "max_image_dimension": 1456,
        # capture_after mode: som = screenshot + overlays; ax = elements only, no PNG (faster);
        # vision = pixels only.
        "capture_after_mode": "som",
        # Disable cua-driver's cursor overlay, which can peg a core when idle (macOS redraw loop;
        # Linux/WSL2 idle spin). None = auto (off on macOS + headless/ WSL2 Linux, on elsewhere);
        # True = always disable; False = always enable.
        # The overlay shows where agent actions land but can peg a core when idle (macOS vImage redraw loop
        # #47032; Linux/WSL2 idle spin #28152). cua-driver ≥ 0.6.x supports --no-overlay; Hermes also calls
        # set_agent_cursor_enabled(false) after start_session when this is on.
        "no_overlay": None,
        # cua-driver permission mode for each Hermes computer-use runtime.
        #   standard (default) — cua-driver's own approval boundary.
        #   bounded — repeatable automation under a user-reviewed session
        #     capability manifest (set capability_manifest below). No runtime
        #     prompts; anything outside the manifest fails closed inside
        #     cua-driver.
        # `unrestricted` is intentionally NOT accepted here: it stays bound to
        # the explicit per-session YOLO toggle so a config line can never
        # silently bypass approvals.
        "permission_mode": "standard",
        # Path (~ ok) to the reviewed manifest for permission_mode bounded; passed as
        # --capability-manifest. See cua.ai/docs/reference/cua-driver/permission-modes
        "capability_manifest": "",
        # macOS only: allow launching an UNSIGNED (ad-hoc / TeamIdentifier
        # not set) CuaDriver.app for the private-session daemon. The default
        # (false) fails closed unless the bundle is signed with the official
        # cua-driver identity (com.trycua.driver / an official signing team).
        # Enable only when developing the driver locally from source.
        "allow_unsigned_driver": False,
    },
    # Egress credential-injection proxy (iron-proxy) for remote terminal sandboxes (Docker today):
    # the sandbox sees opaque tokens and iron-proxy swaps in real credentials at egress, so a
    # compromised sandbox leaks only tokens that work behind the trusted proxy. Configure with
    # `hermes egress setup`.
    "proxy": {
        "enabled": False,  # When false, nothing starts, no docker mounts, no binary installs.
        # Tunnel listener port; sandboxes get HTTPS_PROXY=http://<host>:<port>.
        "tunnel_port": 9090,
        # Auto-download the pinned binary into ~/.hermes/bin/; False = iron-proxy on PATH.
        "auto_install": True,
        # Upstream secret source: env = process env; bitwarden = refetch via `bws secret list` on
        # each proxy restart (requires secrets.bitwarden.enabled).
        "credential_source": "env",
        # True: the Docker backend refuses to start a sandbox if the proxy is enabled but not
        # running. False: fall back to direct outbound with real credentials.
        "enforce_on_docker": True,
        # With credential_source bitwarden, a missing BWS token/project_id or an empty fetch makes
        # the daemon raise. True silently falls back to host env (useful mid-migration). A leftover
        # fail_on_uncovered_providers key is ignored.
        "allow_env_fallback": False,
        # SSRF deny list. None/empty = loopback, link-local (incl. 169.254.169.254 metadata),
        # RFC1918. Explicit [] opts out (only for hermetic loopback tests).
        "upstream_deny_cidrs": None,
        # Extra allowed upstream hosts beyond the bundled major-provider defaults; wildcards
        # (`*.foo.com`) supported.
        "extra_allowed_hosts": [],
    },
    "desktop": {  # Hermes Desktop (Electron) launch options; only affect `hermes desktop`.
        # CSS font-family for the app's chat and UI text (e.g. "OpenDyslexic"). Layered in front
        # of the active theme's own sans stack so missing glyphs still fall through. Empty = the
        # theme's face. The terminal pane is terminal.font_family.
        "font_family": "",
        # Git repo discovery for the Projects sidebar; empty roots = bounded scan of $HOME.
        "repo_scan_enabled": True,
        "repo_scan_roots": [],
        "repo_scan_exclude_paths": [],
        # Background update checks surfaced by Hermes Desktop. Applying an
        # update remains explicit; this only controls checks and notifications.
        "automatic_update_checks": True,
        # Extra Electron command-line flags appended to every desktop launch,
        # e.g. ["--ozone-platform=x11"] on headless/VM X11 hosts that need an
        # explicit ozone backend, or GPU workaround flags. A list of strings;
        # a single string is also accepted and shell-split.
        "electron_flags": [],
        # Linux Ozone backend hint, bridged to ELECTRON_OZONE_PLATFORM_HINT
        # at launch (an explicit env var still wins). "auto" is Chromium's
        # default — Wayland on a Wayland session, X11 otherwise.
        # Set "x11" to run under XWayland when a compositor ignores
        # always-on-top for native Wayland clients (COSMIC, issue #84011).
        # That also lands the HUD on the solid-window input path, because
        # setIgnoreMouseEvents is a one-way door on X11.
        # "wayland" forces a native Wayland surface.
        "ozone_platform_hint": "auto",
        # GPU hardware acceleration policy for the desktop app:
        #   "auto"  - let the app detect remote displays (SSH/VNC/RDP) and
        #             disable GPU only then (default; current behavior).
        #   true    - always disable GPU acceleration (software rendering).
        #             Use on no-GPU VMs / Proxmox hosts where the GPU path hangs.
        #   false   - always keep GPU acceleration on, even over a remote display.
        # Bridged to the HERMES_DESKTOP_DISABLE_GPU env var the Electron app reads.
        "disable_gpu": "auto",
        # Linux keychain for token storage (Chromium --password-store). auto = detect KWallet (KDE
        # env) or any org.freedesktop.secrets provider via D-Bus;
        # gnome-libsecret|kwallet|kwallet5|kwallet6|basic force one (basic = unencrypted). Bridged
        # to HERMES_DESKTOP_PASSWORD_STORE; ignored off-Linux.
        "password_store": "auto",
        # Linux: False preserves an existing custom XDG launcher entry; missing entries
        # are still created. True keeps the generated entry current on each launch.
        "manage_launcher_entry": True,
        # macOS only: code-signing identity (login-keychain cert; self-signed works) to re-sign
        # locally rebuilt apps so the Designated Requirement — and thus TCC grants — survives
        # updates. Empty = default ad-hoc identifier-pinned signing.
        "macos_signing_identity": "",
        # Auto-continue a turn killed by a crash: resuming re-submits the interrupted prompt if
        # fresh; a stale one just shows the recovered partial transcript.
        "auto_continue": {
            "enabled": True,
            # How recent the interruption must be to auto-continue (minutes).
            "freshness_minutes": 15,
            "max_attempts": 2,  # Crash-loop breaker: max automatic re-runs of one interrupted turn.
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
    "nous": {
        # Upper bound on the Nous auth keepalive tick, in seconds. The tick
        # actually used derives from the credential lifetime the server issued
        # and is capped by this value, so lowering it makes the keepalive more
        # frequent while raising it has no effect below the derived tick.
        # 0 disables the keepalive thread entirely.
        "keepalive_interval_seconds": 900,
        # anthropic_wire: which Portal route carries anthropic/* models. "chat" =
        # /v1/chat/completions (default for now); "native" = /v1/messages, the Anthropic
        # Messages wire (signed thinking passthrough, native cache_control scopes); "auto" =
        # start on chat and, per session, switch to native from the first response when the
        # Portal upstream serving the model is one where native is known clean. Native is the
        # better wire but on the OpenRouter-served path it re-writes the previous turn's cache on
        # 14-20% of consecutive calls in concurrent tool loops (measured 2026-09-06;
        # NousResearch/api#227), so chat is the default until that is fixed.
        "anthropic_wire": "chat",
        # Nous free tier: true opts into first-use provisioning (nous/welcome + connectors)
        # independently of the launcher; false disables it entirely. null preserves an older
        # launcher's explicit onboarding hint, but stays off without one. This must not default
        # to true: load_config_readonly merges defaults before the provisioning gate reads them.
        "guest": None,
    },

    "vertex": {
        # GCP project ID. Empty → project_id from the service-account JSON (or ADC).
        "project_id": "",
        # "global" is required for Gemini 3.x preview models (regional endpoints silently 404); use
        # e.g. "us-central1" only if your models are region-pinned.
        "region": "global",
    },

    # Managed llama.cpp local runtime (see docs: user-guide/local-models).
    # Hermes downloads official llama.cpp release binaries, then spawns and
    # supervises one llama-server in router mode. Context sizing is policy,
    # not preference: there are deliberately no context/VRAM knobs here.
    "local_runtime": {
        # Master switch for the managed runtime. Off = detection-only
        # (Hermes still finds an external llama-server you run yourself).
        "enabled": False,
        # Pinned llama.cpp release tag (rolling bNNNN). Bumped by Hermes
        # releases after the validation suite re-runs, not tracked live.
        "tag": "b10679",
        # Inference backend: auto = CUDA on NVIDIA, Metal on macOS, Vulkan on
        # other GPUs, else CPU. Explicit values: cuda|metal|vulkan|hip|cpu.
        "backend": "auto",
        # Router process: how many models may be resident at once.
        "models_max": 4,
        # Port for the managed server. 0 = pick a free port at spawn.
        "port": 0,
        # Extra ports detection probes for an external llama-server, in
        # addition to the default 8080.
        "detect_ports": [],
    },
    "_config_version": 44,  # Config schema version - bump this when adding new required fields
}


def _env(description, prompt, **keys):
    """One OPTIONAL_ENV_VARS entry; keyword order is preserved as dict key order."""
    return {"description": description, "prompt": prompt, **keys}


_OMIT = object()


def _category(category, password, advanced):
    """Entry factory for one category with its usual password/advanced defaults.

    ``url``/``help``/``tools`` are only written when passed; ``password=None`` omits the key;
    ``advanced`` is only written when true. Key order matches the plain ``_env`` entries.
    """
    def make(description, prompt, url=_OMIT, *, help=_OMIT, tools=_OMIT, password=password,
             advanced=advanced):
        d = {"description": description, "prompt": prompt}
        d.update((k, v) for k, v in (("help", help), ("url", url), ("tools", tools)) if v is not _OMIT)
        if password is not None:
            d["password"] = password
        d["category"] = category
        if advanced:
            d["advanced"] = True
        return d
    return make


_prov = _category("provider", password=True, advanced=True)
_tool = _category("tool", password=True, advanced=False)
_msg = _category("messaging", password=False, advanced=False)
_skill = _category("skill", password=True, advanced=True)
_setting = _category("setting", password=False, advanced=False)


def _base_url(name, prompt_name=None):
    """Provider ``*_BASE_URL`` override entry (advanced, not a secret)."""
    prompt = f"{prompt_name or name} base URL (leave empty for default)"
    return _prov(f"{name} base URL override", prompt, None, password=False)


# Optional environment variables that enhance functionality. Feeds the dashboard keys page and setup
# checklists; category: provider|tool|skill|messaging|setting, advanced=True hides from checklists,
# tools=[...] lists the model tools the key unlocks.
OPTIONAL_ENV_VARS = {
    # ── Provider (handled in provider selection, not shown in checklists) ──
    "NOUS_BASE_URL": _base_url("Nous Portal"),
    "HERMES_ANON_API_SECRET": _env(
        "Shared secret for the Nous free-tier sign-up endpoints while they are in their gated "
        "integration phase (not needed once the gate is removed)",
        "Nous free-tier shared secret (leave empty unless given one)", password=True,
        category="provider", advanced=True),
    "OPENROUTER_API_KEY": _env("OpenRouter API key (for vision, web scraping helpers, and MoA)",
        "OpenRouter API key", url="https://openrouter.ai/keys", password=True, tools=["vision_analyze"],
        category="provider", advanced=True),
    "GOOGLE_API_KEY": _prov("Google AI Studio API key (also recognized as GEMINI_API_KEY)",
        "Google AI Studio API key", "https://aistudio.google.com/app/apikey"),
    "GEMINI_API_KEY": _prov("Google AI Studio API key (alias for GOOGLE_API_KEY)", "Gemini API key",
        "https://aistudio.google.com/app/apikey"),
    "GEMINI_BASE_URL": _base_url("Google AI Studio", "Gemini"),
    "VERTEX_CREDENTIALS_PATH": _prov(
        "Path to a Google Cloud service account JSON for Vertex AI (Gemini). Vertex uses "
        "OAuth2, not a static API key — this points at the credentials Hermes mints short-lived "
        "tokens from. Falls back to GOOGLE_APPLICATION_CREDENTIALS, then to ADC (gcloud auth "
        "application-default login). Set project/region under vertex: in config.yaml.",
        "Vertex service account JSON path (leave empty to use ADC / "
        "GOOGLE_APPLICATION_CREDENTIALS)", "https://cloud.google.com/iam/docs/keys-create-delete",
        password=False),
    "XAI_API_KEY": _prov("xAI API key", "xAI API key", "https://console.x.ai/"),
    "XAI_BASE_URL": _base_url("xAI"),
    "NVIDIA_API_KEY": _prov("NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)",
        "NVIDIA NIM API key", "https://build.nvidia.com/"),
    "NVIDIA_BASE_URL": _prov(
        "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)",
        "NVIDIA NIM base URL (leave empty for default)", None, password=False),
    "LM_API_KEY": _prov("LM Studio bearer token for auth-enabled local servers",
        "LM Studio API key / bearer token", None),
    "LM_BASE_URL": _base_url("LM Studio"),
    "GLM_API_KEY": _prov("Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)",
        "Z.AI / GLM API key", "https://z.ai/"),
    "ZAI_API_KEY": _prov("Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", "https://z.ai/"),
    "Z_AI_API_KEY": _prov("Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", "https://z.ai/"),
    "GLM_BASE_URL": _base_url("Z.AI / GLM"),
    "KIMI_API_KEY": _prov("Kimi / Moonshot API key", "Kimi API key",
        "https://platform.moonshot.cn/"),
    "KIMI_BASE_URL": _base_url("Kimi / Moonshot", "Kimi"),
    "KIMI_CN_API_KEY": _prov("Kimi / Moonshot China API key", "Kimi (China) API key",
        "https://platform.moonshot.cn/"),
    "STEPFUN_API_KEY": _prov("StepFun Step Plan API key", "StepFun Step Plan API key",
        "https://platform.stepfun.com/"),
    "STEPFUN_BASE_URL": _base_url("StepFun Step Plan"),
    "ARCEEAI_API_KEY": _prov("Arcee AI API key", "Arcee AI API key", "https://chat.arcee.ai/"),
    "ARCEE_BASE_URL": _base_url("Arcee AI", "Arcee"),
    "GMI_API_KEY": _prov("GMI Cloud API key", "GMI Cloud API key", "https://www.gmicloud.ai/"),
    "GMI_BASE_URL": _base_url("GMI Cloud"),
    "ACTUAL_API_KEY": _prov("Actual Computer inference key (ac_...)",
        "Actual Computer inference key", "https://actual.inc/user/keys"),
    "FIREWORKS_API_KEY": _prov("Fireworks AI API key", "Fireworks AI API key",
        "https://app.fireworks.ai/settings/users/api-keys"),
    "MINIMAX_API_KEY": _prov("MiniMax API key (international)", "MiniMax API key",
        "https://www.minimax.io/"),
    "MINIMAX_BASE_URL": _base_url("MiniMax"),
    "MINIMAX_CN_API_KEY": _prov("MiniMax API key (China endpoint)", "MiniMax (China) API key",
        "https://www.minimaxi.com/"),
    "MINIMAX_CN_BASE_URL": _base_url("MiniMax (China)"),
    "DEEPSEEK_API_KEY": _prov("DeepSeek API key for direct DeepSeek access", "DeepSeek API Key",
        "https://platform.deepseek.com/api_keys", advanced=False),
    "DEEPSEEK_BASE_URL": _prov("Custom DeepSeek API base URL (advanced)", "DeepSeek Base URL", "",
        password=False, advanced=False),
    "DASHSCOPE_API_KEY": _prov("Alibaba Cloud DashScope API key (Qwen + multi-provider models)",
        "DashScope API Key", "https://modelstudio.console.alibabacloud.com/", advanced=False),
    "DASHSCOPE_BASE_URL": _prov(
        "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)",
        "DashScope Base URL", "", password=False),
    "HERMES_QWEN_BASE_URL": _prov(
        "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)",
        "Qwen Portal base URL (leave empty for default)", None, password=False),
    "OPENCODE_ZEN_API_KEY": _prov("OpenCode Zen API key (pay-as-you-go access to curated models)",
        "OpenCode Zen API key", "https://opencode.ai/auth"),
    "COMMANDCODE_API_KEY": _prov(
        "CommandCode API key (GOAT/Pro/Max/Provider plans — 30+ models via one key)",
        "CommandCode API key", "https://commandcode.ai/studio/"),
    "OPENCODE_ZEN_BASE_URL": _base_url("OpenCode Zen"),
    "OPENCODE_GO_API_KEY": _prov("OpenCode Go API key ($10/month subscription for open models)",
        "OpenCode Go API key", "https://opencode.ai/auth"),
    "OPENCODE_GO_BASE_URL": _base_url("OpenCode Go"),
    "HF_TOKEN": _prov(
        "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)",
        "Hugging Face Token", "https://huggingface.co/settings/tokens", advanced=False),
    "HF_BASE_URL": _base_url("Hugging Face Inference Providers", "HF"),
    "OLLAMA_API_KEY": _prov("Ollama Cloud API key (ollama.com — cloud-hosted open models)",
        "Ollama Cloud API key", "https://ollama.com/settings"),
    "OLLAMA_BASE_URL": _prov("Ollama Cloud base URL override (default: https://ollama.com/v1)",
        "Ollama base URL (leave empty for default)", None, password=False),
    "XIAOMI_API_KEY": _prov(
        "Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, "
        "mimo-v2-omni, mimo-v2-flash)", "Xiaomi MiMo API Key", "https://platform.xiaomimimo.com",
        advanced=False),
    "XIAOMI_BASE_URL": _prov(
        "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)",
        "Xiaomi base URL (leave empty for default)", None, password=False),
    "UPSTAGE_API_KEY": _prov("Upstage API key for Solar LLM models", "Upstage API Key",
        "https://console.upstage.ai/api-keys", advanced=False),
    "UPSTAGE_BASE_URL": _prov("Upstage base URL override (default: https://api.upstage.ai/v1)",
        "Upstage base URL (leave empty for default)", None, password=False),
    "AWS_REGION": _prov("AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)",
        "AWS Region", "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html",
        password=False),
    "AWS_PROFILE": _prov("AWS named profile for Bedrock authentication (from ~/.aws/credentials)",
        "AWS Profile", None, password=False),
    "AZURE_FOUNDRY_API_KEY": _prov("Azure Foundry API key for custom Azure endpoints",
        "Azure Foundry API Key", "https://ai.azure.com/", advanced=False),
    "AZURE_FOUNDRY_BASE_URL": _prov(
        "Azure Foundry base URL (set via 'hermes model' for endpoint-specific config)",
        "Azure Foundry base URL", None, password=False),
    # ── Tool API keys ──
    "EXA_API_KEY": _tool("Exa API key for AI-native web search and contents", "Exa API key",
        "https://exa.ai/", tools=["web_search", "web_extract"]),
    "PARALLEL_API_KEY": _tool("Parallel API key for AI-native web search and extract",
        "Parallel API key", "https://parallel.ai/", tools=["web_search", "web_extract"]),
    "FIRECRAWL_API_KEY": _tool("Firecrawl API key for web search and scraping", "Firecrawl API key",
        "https://firecrawl.dev/", tools=["web_search", "web_extract"]),
    "FIRECRAWL_API_URL": _tool("Firecrawl API URL for self-hosted instances (optional)",
        "Firecrawl API URL (leave empty for cloud)", None, password=False, advanced=True),
    "FIRECRAWL_GATEWAY_URL": _tool(
        "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)",
        "Firecrawl gateway URL (leave empty to derive from domain)", None, password=False,
        advanced=True),
    "TOOL_GATEWAY_URL": _tool(
        "Exact shared tool-gateway origin for on-origin vendors and media uploads (optional)",
        "Shared tool-gateway URL (leave empty to derive from domain)", None,
        password=False, advanced=True),
    "CONNECTOR_GATEWAY_URL": _tool(
        "Exact connector-gateway origin for the connectors API (optional)",
        "Connector-gateway URL (leave empty to derive from domain)", None,
        password=False, advanced=True),
    "TOOL_GATEWAY_DOMAIN": _tool(
        "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor "
        "hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com",
        "Tool-gateway domain suffix", None, password=False, advanced=True),
    "TOOL_GATEWAY_SCHEME": _tool(
        "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts "
        "(`https` by default, set `http` for local gateway testing)", "Tool-gateway URL scheme",
        None, password=False, advanced=True),
    "TOOL_GATEWAY_USER_TOKEN": _tool(
        "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise "
        "read from the Hermes auth store)", "Tool-gateway user token", None, advanced=True),
    "TAVILY_API_KEY": _tool(
        "Tavily API key for AI-native web search and extract (optional — keyless works when "
        "Tavily is selected)", "Tavily API key", "https://app.tavily.com/home",
        tools=["web_search", "web_extract"]),
    "PERPLEXITY_API_KEY": _tool(
        "Perplexity API key for the Search API web backend (ranked results + query-relevant page "
        "snippets)", "Perplexity API key", "https://www.perplexity.ai/account/api",
        tools=["web_search", "web_extract"]),
    "KEENABLE_API_KEY": _tool(
        "Keenable API key for fast independent-index web search and page fetch (optional — "
        "keyless free tier works without it)", "Keenable API key", "https://keenable.ai",
        tools=["web_search", "web_extract"]),
    "SEARXNG_URL": _tool("URL of your SearXNG instance for free self-hosted web search",
        "SearXNG URL (e.g. http://localhost:8080)", "https://searxng.github.io/searxng/",
        tools=["web_search"], password=False),
    "BRAVE_SEARCH_API_KEY": _tool(
        "Brave Search API subscription token (free tier: 2,000 queries/mo)",
        "Brave Search subscription token", "https://brave.com/search/api/", tools=["web_search"]),
    "BROWSERBASE_API_KEY": _tool(
        "Browserbase API key for cloud browser (optional — local browser works without this)",
        "Browserbase API key", "https://browserbase.com/",
        tools=["browser_navigate", "browser_click"]),
    "BROWSERBASE_PROJECT_ID": _tool(
        "Browserbase project ID (optional — only needed for cloud browser)",
        "Browserbase project ID", "https://browserbase.com/",
        tools=["browser_navigate", "browser_click"], password=False),
    "BROWSER_USE_API_KEY": _tool(
        "Browser Use API key for cloud browser (optional — local browser works without this)",
        "Browser Use API key", "https://browser-use.com/",
        tools=["browser_navigate", "browser_click"]),
    "FIRECRAWL_BROWSER_TTL": _tool(
        "Firecrawl browser session TTL in seconds (optional, default 300)",
        "Browser session TTL (seconds)", tools=["browser_navigate", "browser_click"],
        password=False),
    "AGENT_BROWSER_ENGINE": _env(
        "Local browser engine: auto (default Chrome), lightpanda (faster, no screenshots; Browser Use mode "
        "spawns lightpanda serve), chrome", "Browser engine (auto/lightpanda/chrome)",
        url="https://lightpanda.io/docs/run-locally/installation/one-liner",
        tools=["browser_exec", "browser_navigate", "browser_snapshot", "browser_click", "browser_vision"],
        password=False, category="tool", advanced=True),
    "CAMOFOX_URL": _tool(
        "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)",
        "Camofox server URL", "https://github.com/jo-inc/camofox-browser",
        tools=["browser_navigate", "browser_click"], password=False),
    "CAMOFOX_API_KEY": _tool(
        "Optional bearer token sent as Authorization header to a remote/authenticated Camofox "
        "server", "Camofox API key", "https://github.com/jo-inc/camofox-browser",
        tools=["browser_navigate", "browser_click"], advanced=True),
    "FAL_KEY": _tool("FAL API key for image and video generation", "FAL API key", "https://fal.ai/",
        tools=["image_generate", "video_generate"]),
    "KREA_API_KEY": _tool("Krea API key for Krea 2 image generation (Medium + Large)",
        "Krea API key", "https://www.krea.ai/settings/api-tokens", tools=["image_generate"]),
    "VOICE_TOOLS_OPENAI_KEY": _tool(
        "OpenAI API key for voice transcription (Whisper) and OpenAI TTS",
        "OpenAI API Key (for Whisper STT + TTS)", "https://platform.openai.com/api-keys",
        tools=["voice_transcription", "openai_tts"]),
    "ELEVENLABS_API_KEY": _tool(
        "ElevenLabs API key for premium text-to-speech voices and Scribe transcription",
        "ElevenLabs API key", "https://elevenlabs.io/",
        tools=["elevenlabs_tts", "voice_transcription"]),
    "MISTRAL_API_KEY": _tool("Mistral API key for Voxtral TTS and transcription (STT)",
        "Mistral API key", "https://console.mistral.ai/"),
    "PORCUPINE_ACCESS_KEY": _tool(
        "Picovoice access key for the Porcupine 'Hey Hermes' wake word engine (optional; "
        "openWakeWord is the free default)", "Picovoice access key",
        "https://console.picovoice.ai/"),
    "GITHUB_TOKEN": _tool("GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "GitHub Token", "https://github.com/settings/tokens"),
    # ── Bundled skills (opt-in) ── category="skill" (not "tool") so the sandbox env blocklist in
    # tools/environments/local.py does NOT rewrite them; skills need them passed through to curl
    # via tools/env_passthrough.py.
    "NOTION_API_KEY": _skill("Notion integration token (used by the `notion` skill)",
        "Notion API key", "https://www.notion.so/my-integrations"),
    "LINEAR_API_KEY": _skill("Linear personal API key (used by the `linear` skill)",
        "Linear API key", "https://linear.app/settings/account/security"),
    "AIRTABLE_API_KEY": _skill("Airtable personal access token (used by the `airtable` skill)",
        "Airtable API key", "https://airtable.com/create/tokens"),
    "TENOR_API_KEY": _skill("Tenor API key for GIF search (used by the `gif-search` skill)",
        "Tenor API key", "https://developers.google.com/tenor/guides/quickstart"),
    # ── Honcho ──
    "HONCHO_API_KEY": _tool("Honcho API key for AI-native persistent memory", "Honcho API key",
        "https://app.honcho.dev", tools=["honcho_context"]),
    "HONCHO_BASE_URL": _tool("Base URL for self-hosted Honcho instances (no API key needed)",
        "Honcho base URL (e.g. http://localhost:8000)", password=None),
    # ── Hindsight ──
    "HINDSIGHT_API_KEY": _tool("Hindsight API key for graph-aware persistent memory",
        "Hindsight API key", "https://hindsight.vectorize.io", tools=["hindsight_recall"]),
    "HINDSIGHT_API_URL": _tool(
        "Base URL for the Hindsight API (default: https://api.hindsight.vectorize.io)",
        "Hindsight API URL", password=None, advanced=True),
    # ── Supermemory ──
    "SUPERMEMORY_API_KEY": _tool("Supermemory API key for conversation-scoped persistent memory",
        "Supermemory API key", "https://supermemory.ai", tools=["supermemory_search"]),
    # ── Mem0 ──
    "MEM0_API_KEY": _tool("Mem0 Platform API key for semantic persistent memory", "Mem0 API key",
        "https://app.mem0.ai", tools=["mem0_search"]),
    # ── RetainDB ──
    "RETAINDB_API_KEY": _tool("RetainDB API key for persistent memory", "RetainDB API key",
        "https://retaindb.com", tools=["retaindb_search"]),
    "RETAINDB_BASE_URL": _tool(
        "Base URL for self-hosted RetainDB instances (default: https://api.retaindb.com)",
        "RetainDB base URL", password=None, advanced=True),
    # ── ByteRover ──
    "BRV_API_KEY": _tool("ByteRover API key (optional, for cloud sync — local-first by default)",
        "ByteRover API key", "https://app.byterover.dev", tools=["brv_query"]),
    # ── OpenViking ──
    "OPENVIKING_API_KEY": _tool("OpenViking API key (leave blank for local dev mode)",
        "OpenViking API key", tools=["viking_search"]),
    "OPENVIKING_ENDPOINT": _tool("OpenViking server URL (default: http://127.0.0.1:1933)",
        "OpenViking endpoint", password=None, advanced=True),
    # ── Langfuse observability ──
    "HERMES_LANGFUSE_PUBLIC_KEY": _tool("Langfuse project public key (pk-lf-...)",
        "Langfuse public key", "https://cloud.langfuse.com", password=False),
    "HERMES_LANGFUSE_SECRET_KEY": _tool("Langfuse project secret key (sk-lf-...)",
        "Langfuse secret key", "https://cloud.langfuse.com"),
    "HERMES_LANGFUSE_BASE_URL": _tool("Langfuse server URL (default: https://cloud.langfuse.com)",
        "Langfuse server URL (leave empty for cloud.langfuse.com)", None, password=False,
        advanced=True),
    # ── Messaging platforms ──
    "TELEGRAM_BOT_TOKEN": _msg(
        "Complete Telegram bot token created by @BotFather (numeric bot ID followed by a colon "
        "and secret)", "Telegram bot token", "https://t.me/BotFather", password=True),
    "TELEGRAM_ALLOWED_USERS": _msg(
        "Optional comma-separated numeric Telegram user IDs allowed immediately; leave blank to "
        "approve new users through DM pairing", "Allowed Telegram user IDs (comma-separated)",
        "https://t.me/userinfobot"),
    "TELEGRAM_PROXY": _msg(
        "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, "
        "https://, socks5://", "Telegram proxy URL (optional)"),
    "DISCORD_BOT_TOKEN": _msg("Discord bot token from Developer Portal", "Discord bot token",
        "https://discord.com/developers/applications", password=True),
    "DISCORD_ALLOWED_USERS": _msg("Comma-separated Discord user IDs allowed to use the bot",
        "Allowed Discord user IDs (comma-separated)", None),
    "DISCORD_REPLY_TO_MODE": _msg(
        "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first "
        "message only, default), 'all' (reply on every chunk)",
        "Discord reply mode (off/first/all)", None),
    "SLACK_BOT_TOKEN": _msg(
        "Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. "
        "Required scopes: chat:write, app_mentions:read, channels:history, groups:history, "
        "im:history, im:read, im:write, mpim:history, mpim:read, users:read, files:read, "
        "files:write", "Slack Bot Token (xoxb-...)", "https://api.slack.com/apps",
        help=("In your Slack app, add the required bot scopes, install the app to the workspace, "
        "then copy OAuth & Permissions > Bot User OAuth Token."), password=True),
    "SLACK_APP_TOKEN": _msg(
        "Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → App-Level "
        "Tokens. Also ensure Event Subscriptions include: message.im, message.channels, "
        "message.groups, message.mpim, app_mention", "Slack App Token (xapp-...)",
        "https://api.slack.com/apps",
        help=("In your Slack app, enable Socket Mode, then create Basic Information > App-Level "
        "Tokens with the connections:write scope."), password=True),
    "SLACK_ALLOWED_USERS": _msg(
        "Comma-separated Slack member IDs allowed to use Hermes, e.g. U01ABC2DEF3. Without "
        "this, Slack may connect but deny messages by default.", "Allowed Slack member IDs",
        "https://api.slack.com/apps",
        help=("In Slack, open your profile, choose More or the three-dot menu, then Copy member "
        "ID. Add multiple IDs comma-separated.")),
    "MATTERMOST_URL": _msg("Mattermost server URL (e.g. https://mm.example.com)",
        "Mattermost server URL", "https://mattermost.com/deploy/"),
    "MATTERMOST_TOKEN": _msg("Mattermost bot token or personal access token",
        "Mattermost bot token", None, password=True),
    "MATTERMOST_ALLOWED_USERS": _msg("Comma-separated Mattermost user IDs allowed to use the bot",
        "Allowed Mattermost user IDs (comma-separated)", None),
    "MATTERMOST_REQUIRE_MENTION": _msg(
        "Require @mention in Mattermost channels (default: true). Set to false to respond to "
        "all messages.", "Require @mention in channels", None),
    "MATTERMOST_FREE_RESPONSE_CHANNELS": _msg(
        "Comma-separated Mattermost channel IDs where bot responds without @mention",
        "Free-response channel IDs (comma-separated)", None),
    "MATRIX_HOMESERVER": _msg("Matrix homeserver URL (e.g. https://matrix.example.org)",
        "Matrix homeserver URL", "https://matrix.org/ecosystem/servers/"),
    "MATRIX_ACCESS_TOKEN": _msg("Matrix access token (preferred over password login)",
        "Matrix access token", None, password=True),
    "MATRIX_USER_ID": _msg("Matrix user ID (e.g. @hermes:example.org)",
        "Matrix user ID (@user:server)", None),
    "MATRIX_ALLOWED_USERS": _msg(
        "Comma-separated Matrix user IDs allowed to use the bot (@user:server format)",
        "Allowed Matrix user IDs (comma-separated)", None),
    "MATRIX_REQUIRE_MENTION": _msg(
        "Require @mention in Matrix rooms (default: true). Set to false to respond to all "
        "messages.", "Require @mention in rooms (true/false)", None, advanced=True),
    "MATRIX_FREE_RESPONSE_ROOMS": _msg(
        "Comma-separated Matrix room IDs where bot responds without @mention",
        "Free-response room IDs (comma-separated)", None, advanced=True),
    "MATRIX_AUTO_THREAD": _msg("Auto-create threads for messages in Matrix rooms (default: true)",
        "Auto-create threads in rooms (true/false)", None, advanced=True),
    "MATRIX_DM_AUTO_THREAD": _msg("Auto-create threads for DM messages in Matrix (default: false)",
        "Auto-create threads in DMs (true/false)", None, advanced=True),
    "MATRIX_DEVICE_ID": _msg(
        "Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)",
        "Matrix device ID (stable across restarts)", None, advanced=True),
    "MATRIX_RECOVERY_KEY": _msg(
        "Matrix recovery key for cross-signing verification after device key rotation (from "
        "Element: Settings → Security → Recovery Key)", "Matrix recovery key", None, password=True,
        advanced=True),
    "BLUEBUBBLES_SERVER_URL": _msg(
        "BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)",
        "BlueBubbles server URL", "https://bluebubbles.app/"),
    "BLUEBUBBLES_PASSWORD": _msg(
        "BlueBubbles server password (from BlueBubbles Server → Settings → API)",
        "BlueBubbles server password", None, password=True),
    "BLUEBUBBLES_ALLOWED_USERS": _msg(
        "Comma-separated iMessage addresses (email or phone) allowed to use the bot",
        "Allowed iMessage addresses (comma-separated)", None),
    "BLUEBUBBLES_ALLOW_ALL_USERS": _msg("Allow all BlueBubbles users without allowlist",
        "Allow All BlueBubbles Users", password=None),
    "QQ_APP_ID": _msg("QQ Bot App ID from QQ Open Platform (q.qq.com)", "QQ App ID",
        "https://q.qq.com", password=None),
    "QQ_CLIENT_SECRET": _msg("QQ Bot Client Secret from QQ Open Platform", "QQ Client Secret",
        password=True),
    "QQ_ALLOWED_USERS": _msg("Comma-separated QQ user IDs allowed to use the bot",
        "QQ Allowed Users", password=None),
    "QQ_GROUP_ALLOWED_USERS": _msg("Comma-separated QQ group IDs allowed to interact with the bot",
        "QQ Group Allowed Users", password=None),
    "QQ_ALLOW_ALL_USERS": _msg("Allow all QQ users without an allowlist (true/false)",
        "Allow All QQ Users", password=None),
    "QQBOT_HOME_CHANNEL": _msg("Default QQ channel/group for cron delivery and notifications",
        "QQ Home Channel", password=None),
    "QQBOT_HOME_CHANNEL_NAME": _msg("Display name for the QQ home channel", "QQ Home Channel Name",
        password=None),
    "QQ_SANDBOX": _msg("Enable QQ sandbox mode for development testing (true/false)",
        "QQ Sandbox Mode", password=None),
    "IRC_SERVER": _msg("IRC server hostname (e.g. irc.libera.chat)", "IRC server", None),
    "IRC_CHANNEL": _msg("IRC channel to join (e.g. #hermes)", "IRC channel", None),
    "IRC_NICKNAME": _msg("Bot nickname on IRC (default: hermes-bot)", "IRC nickname", None),
    "IRC_SERVER_PASSWORD": _msg("IRC server password (if required)", "IRC server password", None,
        password=True, advanced=True),
    "IRC_NICKSERV_PASSWORD": _msg("NickServ password for nick identification", "NickServ password",
        None, password=True, advanced=True),
    "GATEWAY_ALLOW_ALL_USERS": _msg(
        "Allow all users to interact with messaging bots (true/false). Default: false.",
        "Allow all users (true/false)", None, advanced=True),
    "API_SERVER_ENABLED": _msg(
        "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open "
        "WebUI, LobeChat, etc. to connect.", "Enable API server (true/false)", None, advanced=True),
    "API_SERVER_KEY": _msg(
        "Bearer token for API server authentication. Required whenever the API server is "
        "enabled; server refuses to start without it.", "API server auth key", None, password=True,
        advanced=True),
    "API_SERVER_PORT": _msg("Port for the API server (default: 8642).", "API server port", None,
        advanced=True),
    "API_SERVER_HOST": _msg(
        "Host/bind address for the API server (default: 127.0.0.1). API_SERVER_KEY is still "
        "required even on loopback binds.", "API server host", None, advanced=True),
    "API_SERVER_MODEL_NAME": _msg(
        "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' "
        "for the default profile). Useful for multi-user setups with OpenWebUI.",
        "API server model name", None, advanced=True),
    "GATEWAY_PROXY_URL": _msg(
        "URL of a remote Hermes API server to forward messages to (proxy mode). When set, the "
        "gateway handles platform I/O only — all agent work is delegated to the remote server. "
        "Use for Docker E2EE containers that relay to a host agent. Also configurable via "
        "gateway.proxy_url in config.yaml.",
        "Remote Hermes API server URL (e.g. http://192.168.1.100:8642)", None, advanced=True),
    "GATEWAY_PROXY_KEY": _msg(
        "Bearer token for authenticating with the remote Hermes API server (proxy mode). Must "
        "match the API_SERVER_KEY on the remote host.", "Remote API server auth key", None,
        password=True, advanced=True),
    "WEBHOOK_ENABLED": _msg(
        "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.",
        "Enable webhooks (true/false)", None),
    "WEBHOOK_PORT": _msg("Port for the webhook HTTP server (default: 8644).", "Webhook port", None),
    "WEBHOOK_SECRET": _msg(
        "Global HMAC secret for webhook signature validation (overridable per route in "
        "config.yaml).", "Webhook secret", None, password=True),
    # ── Agent settings ── (MESSAGING_CWD is gone: use terminal.cwd in config.yaml, which the
    # gateway bridges to TERMINAL_CWD.)
    "SUDO_PASSWORD": _setting(
        "Sudo password for terminal commands requiring root access; set to an explicit empty "
        "string to try empty without prompting", "Sudo password", None, password=True),
    # HERMES_TOOL_PROGRESS_MODE (deprecated; use display.tool_progress) is intentionally NOT listed:
    # this dict feeds user-facing surfaces (dashboard keys page, setup checklists), so deprecated
    # knobs stay in config._EXTRA_ENV_KEYS only. HERMES_TOOL_PROGRESS is unsupported.
    "HERMES_PREFILL_MESSAGES_FILE": _setting(
        "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "Prefill messages file path", None),
    "HERMES_EPHEMERAL_SYSTEM_PROMPT": _setting(
        "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "Ephemeral system prompt", None),
}
