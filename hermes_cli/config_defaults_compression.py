"""Pure-data defaults for context compression."""


def compression_defaults() -> dict:
    return {
        "enabled": True,
        # checkpoint_required: fail closed before lossy compaction unless an active memory provider
        # confirms checkpoint API compatibility and completes the checkpoint.
        "checkpoint_required": False,
        # progress_notices: when True, routine compression progress statuses (compacting/
        # preflight/pre-API/idle/retry) reach chat gateways instead of being filtered as noise.
        # Failure notices and manual /compress feedback are always visible.
        "progress_notices": False,
        # threshold: compress when context usage exceeds this ratio. Models with windows below 512K
        # are floored at 0.75 (raise-only) so compaction doesn't fire with half the window free; set
        # above 0.75 to override the floor.
        "threshold": 0.50,
        # threshold_tokens: optional absolute token cap — when set, compression triggers at the
        # lower of the ratio threshold and this count. Clamped to the model's context length.
        # Off by default: no single count suits windows from 64K to 1M+, so the ratio decides.
        "threshold_tokens": None,
        # "progress_notices": False,    # opt-in (#52995): when True, routine compression
        "target_ratio": 0.20,         # fraction of threshold to preserve as recent tail
        # Summary input source: "previous" preserves the iterative rolling summary behavior.
        # "original" is experimental and opt-in: re-derive from persisted raw turns for the
        # most recent windows instead of repeatedly feeding the newest summary back to itself.
        "summary_source": "previous",
        # Number of persisted compaction windows to re-read in original-source mode. Clamped 1..8.
        "summary_source_windows": 2,
        # tail_mode: "lean" = clamped 2.5%-of-window tail (10K floor / 25K cap) plus chunked
        # digests, anchor index, verbatim user messages and session_search pointers in the summary
        # (~3x fewer retained tokens; a few extra summarizer calls at the boundary). "legacy" =
        # 0.20×threshold verbatim tail (100-240K tokens on big windows).
        "tail_mode": "lean",
        # protect_last_n: minimum recent messages kept uncompressed, honoured up to a small count
        # floor; the verbatim tail is otherwise token-bounded and never above 20% of the window.
        "protect_last_n": 20,
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
        # (callers passing commit_fence, e.g. gateway hygiene, never use it). Floored at the auxiliary
        # compression request timeout (auxiliary.compression.timeout, min 300s): the host never judges
        # silence before the summary request itself would time out.
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
        # OpenAI direct, OpenRouter or another relay keep the global value.
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
        # session id (no parent_session_id chain, no `name #N` renumbering), avoiding the session-rotation
        # bug cluster. Pre-compaction turns are soft-archived under the same id (active=0, compacted=1) —
        # still session_search-able. False = legacy rotating-compaction path.
        "in_place": True,
        # Per-model threshold overrides: keys substring-match the model name (longest wins), values
        # replace the global `threshold`, e.g. {"glm-5.2": 0.40}. Prefix a key with "<provider>:" to
        # scope it to one route ({"openai-codex:astra": 0.85} leaves Astra on OpenRouter/Nous at the
        # global value). The <512K floor (0.75) still applies raise-only on top.
        "model_thresholds": {},
        # Opt-in idle compaction (0 = off): a session resuming after this many idle seconds compacts up
        # front, before the first reply. Time-based complement to `threshold`; skipped when already at/below
        # threshold × target_ratio; honors the same cooldown/anti-thrash/lock guards. Example: 1800 = 30 min.
        "idle_compact_after_seconds": 0,
        # Opt-in: write the next summary in the background near the trigger; compaction uses (or waits for) it if it still covers the same messages.
        "prepare_ahead": False,
    }
