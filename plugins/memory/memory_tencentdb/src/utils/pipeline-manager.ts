/**
 * MemoryPipelineManager: manages the L0→L1→L2→L3 memory extraction pipeline.
 *
 * ## Layered architecture
 *
 * - **L0 (capture)**: `auto-capture.ts` extracts new messages from each
 *   `agent_end` event, sanitizes them, and passes them to the pipeline via
 *   `notifyConversation(sessionKey, messages)`. Messages are buffered
 *   locally per-session — NO remote call happens at this stage.
 *
 * - **L1 (batch extraction / ingest)**: When the conversation count reaches
 *   `everyNConversations` OR the session goes idle for `l1IdleTimeoutSeconds`,
 *   the L1 Runner is invoked with all buffered messages. The runner receives
 *   `{ sessionKey, msg, bg_msg }` and is responsible for ingesting/extracting
 *   them (e.g. calling appendEvent, or running local extraction logic).
 *   `bg_msg` is reserved for background context; currently always empty.
 *
 * - **L2 (scene extraction)**: Per-session downward-only timer. After each
 *   L2 completion, the next fire time is set to `now + maxInterval`. When
 *   L1 completes (new memory event), the fire time is advanced (but never
 *   postponed) to `max(now + delay, lastL2 + minInterval)`. When the timer
 *   fires, if the session is cold (inactive > `sessionActiveWindowHours`),
 *   the timer is cancelled rather than triggering L2 — it will be re-armed
 *   by the next L1 event.
 *
 * - **L3 (persona generation)**: Global mutex (concurrency=1) + pending flag
 *   dedup. Triggered after L2 completes.
 *
 * ## Timer semantics
 *
 * L1 uses a **resettable timer** (classic idle/debounce): each conversation
 * resets the countdown to `l1IdleTimeoutSeconds`. When the timer fires,
 * buffered messages are flushed through L1.
 *
 * L2 uses a **downward-only timer**: the scheduled fire time can only be
 * moved earlier, never later. This ensures both the maxInterval guarantee
 * and the delay-after-L1 responsiveness, while minInterval acts as a floor.
 *
 * Both timer types are implemented via `ManagedTimer` to eliminate
 * repetitive clear→set→fire→clean boilerplate.
 *
 * ## Trigger paths for L1
 *   A. **Conversation threshold** (primary): when `conversation_count >=
 *      effectiveThreshold` in `notifyConversation()`, L1 is triggered
 *      immediately with all buffered messages. The effective threshold
 *      is influenced by warm-up mode (see below).
 *   B. **Idle timeout** (catch-up): when a session goes idle for
 *      `l1IdleTimeoutSeconds`, L1 fires with whatever messages have
 *      been buffered (below threshold).
 *   C. **Shutdown flush**: on graceful shutdown, all pending buffers
 *      are flushed through L1 then L2.
 *
 * ## Warm-up mode
 *
 * When `enableWarmup` is true (default), new sessions use an exponentially
 * increasing L1 trigger threshold instead of jumping straight to
 * `everyNConversations`. The sequence is: 1 → 2 → 4 → 8 → ... →
 * everyNConversations. This ensures early conversations are processed
 * quickly (first conversation triggers L1 immediately), while gradually
 * reducing processing frequency as the session matures.
 *
 * The `warmup_threshold` field in PipelineSessionState tracks the current
 * threshold. A value of 0 means warm-up is complete (graduated to
 * steady-state). The threshold doubles after each successful L1 run.
 *
 * ## Trigger paths for L2
 *   A. **Delay-after-L1**: L1 completes → timer advanced to
 *      `max(now + delay, lastL2 + min)` → fires → enqueue L2.
 *   B. **MaxInterval guarantee**: L2 completes → timer set to
 *      `now + maxInterval` → fires → enqueue L2 (if session active).
 *   C. **Shutdown flush**: all pending L2 timers are flushed.
 *
 * All queues use SerialQueue (concurrency=1) for serial execution.
 *
 * ## Design doc
 * See `docs/08-pipeline-refactor-design.md` for full architecture.
 */

import type { PipelineSessionState } from "./checkpoint.js";
import { SessionFilter } from "./session-filter.js";
import { ManagedTimer } from "./managed-timer.js";
import { SerialQueue } from "./serial-queue.js";
import { report } from "../core/report/reporter.js";
import type { Logger } from "../core/types.js";

// ============================
// Types
// ============================

/** A single captured message ready for L1 processing. */
export interface CapturedMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  /** ISO timestamp string */
  timestamp: string;
}

/** Pipeline configuration — all time values in seconds. */
export interface PipelineConfig {
  /**
   * Conversation count threshold to trigger L1 batch processing.
   * When a session's conversation_count reaches this value,
   * L1 is triggered immediately with all buffered messages.
   * Default: 5.
   */
  everyNConversations: number;

  /**
   * Enable warm-up mode for new sessions.
   * When enabled, the L1 trigger threshold starts at 1 and doubles after
   * each successful L1 run (1 → 2 → 4 → 8 → ... → everyNConversations),
   * allowing early sessions to be processed more aggressively.
   * Default: true.
   */
  enableWarmup: boolean;

  l1: {
    /** Idle timeout before triggering L1 (seconds, default: 60) */
    idleTimeoutSeconds: number;
  };

  l2: {
    /**
     * Delay after L1 completes before triggering L2 (seconds, default: 90).
     * Allows remote L1 to finish generating records asynchronously.
     */
    delayAfterL1Seconds: number;
    /** Minimum interval between L2 extractions per session (seconds, default: 900) */
    minIntervalSeconds: number;
    /**
     * Maximum interval between L2 extractions per session (seconds, default: 3600).
     * Even without new L1 completions, L2 will poll at this interval for active sessions.
     */
    maxIntervalSeconds: number;
    /**
     * Sessions inactive longer than this (hours, default: 24) stop L2 polling.
     * Prevents wasting resources on abandoned sessions.
     */
    sessionActiveWindowHours: number;
  };
}

/** Result returned by the L1 runner. */
export interface L1RunnerResult {
  /** Number of messages successfully processed */
  processedCount?: number;
  /** Agent/profile-level L2 keys affected by this L1 run. */
  profileScopes?: string[];
  /**
   * True iff there are still L0 rows past the cursor that this run did not
   * consume. The runner detects backlog by over-fetching (query 2N rows but
   * process at most N). Pipeline-manager uses this to decide whether to
   * schedule another L1 run.
   *
   * Semantics (let R = rows returned by the over-fetch, N = batch size):
   *   - R == 2N → `hasFullBacklog=true`  (DB likely has many more, drain ASAP)
   *   - N <  R <  2N → `hasMore=true`    (small tail, defer to next idle)
   *   - R <= N → both flags false        (fully consumed)
   */
  hasMore?: boolean;
  /** True iff the over-fetch returned exactly 2N rows — drain via direct enqueue. */
  hasFullBacklog?: boolean;
}

/** L1 runner — batch-processes buffered messages for a session. */
export type L1Runner = (params: {
  sessionKey: string;
  msg: CapturedMessage[];
  bg_msg: CapturedMessage[];
}) => Promise<L1RunnerResult | void>;

/** Result returned by the L2 extraction runner. */
export interface L2RunnerResult {
  /** The latest `updated_at` cursor from the processed batch. */
  latestCursor?: string;
  /** True if no new records were found and extraction was skipped. */
  skipped?: boolean;
}

/** L2 extraction runner — processes a single session's records. */
export type L2Runner = (sessionKey: string, cursor?: string) => Promise<L2RunnerResult | void>;

/** L3 runner — generates persona from all sessions' scene data. */
export type L3Runner = () => Promise<void>;

/** Callback to persist session states to checkpoint. */
export type PipelineStatePersister = (states: Record<string, PipelineSessionState>) => Promise<void>;

const TAG = "[memory-tdai] [pipeline]";

// ============================
// Per-session timer state (in memory only)
// ============================

interface SessionTimerState {
  /** L1 idle timer (resettable): debounces conversation activity. */
  l1Idle: ManagedTimer;
  /** L2 schedule timer (downward-only): next L2 fire time, only moves earlier. */
  l2Schedule: ManagedTimer;
  /** Whether an L1 task is already queued or running for this session. */
  l1Queued: boolean;
  /** Whether an L2 task is already queued or running for this session. */
  l2Queued: boolean;
  /** Consecutive L1 failure count for retry limiting. Reset on success or new conversation. */
  l1RetryCount: number;
}

export class MemoryPipelineManager {
  // Config (converted to ms internally)
  private readonly l1IdleTimeoutMs: number;
  private readonly everyNConversations: number;
  private readonly enableWarmup: boolean;
  private readonly l2DelayAfterL1Ms: number;
  private readonly l2MinIntervalMs: number;
  private readonly l2MaxIntervalMs: number;
  private readonly sessionActiveWindowMs: number;

  /** Delay before retrying a failed L1 (ms). */
  private readonly L1_RETRY_DELAY_MS = 30_000; // 30 seconds
  /** Max consecutive L1 retries per session before giving up. */
  private readonly L1_MAX_RETRIES = 5;

  // Queues (named for diagnostics)
  private readonly l1Queue = new SerialQueue("L1");
  private readonly l2Queue = new SerialQueue("L2");
  private readonly l3Queue = new SerialQueue("L3");

  // L3 dedup flag
  private l3Pending = false;
  private l3Running = false;

  // Per-session state
  private readonly sessionStates = new Map<string, PipelineSessionState>();
  private readonly sessionTimers = new Map<string, SessionTimerState>();

  // Per-session message buffer: messages accumulated since last L1 run
  private readonly messageBuffers = new Map<string, CapturedMessage[]>();

  // Per-session L2 last run time (epoch ms, for minInterval floor)
  private readonly l2LastRunTime = new Map<string, number>();

  // Callbacks
  private l1Runner: L1Runner | null = null;
  private l2Runner: L2Runner | null = null;
  private l3Runner: L3Runner | null = null;
  private persister: PipelineStatePersister | null = null;
  private logger: Logger | undefined;

  // Unified session filter (internal sessions + excludeAgents)
  private readonly sessionFilter: SessionFilter;

  // Lifecycle
  private destroyed = false;

  /** Plugin instance ID for metric reporting (set externally after async init). */
  instanceId?: string;

  // Session GC: runs periodically to evict cold sessions from memory
  /** Multiplier on sessionActiveWindowMs to determine GC eligibility. */
  private readonly SESSION_GC_INACTIVE_MULTIPLIER = 3;
  /** Run GC every N calls to notifyConversation. */
  private readonly SESSION_GC_EVERY_N_NOTIFICATIONS = 50;
  /** Counter for GC scheduling. */
  private notifyCounter = 0;

  constructor(config: PipelineConfig, logger?: Logger, sessionFilter?: SessionFilter) {
    this.l1IdleTimeoutMs = config.l1.idleTimeoutSeconds * 1000;
    this.everyNConversations = config.everyNConversations;
    this.enableWarmup = config.enableWarmup;
    this.l2DelayAfterL1Ms = config.l2.delayAfterL1Seconds * 1000;
    this.l2MinIntervalMs = config.l2.minIntervalSeconds * 1000;
    this.l2MaxIntervalMs = config.l2.maxIntervalSeconds * 1000;
    this.sessionActiveWindowMs = config.l2.sessionActiveWindowHours * 60 * 60 * 1000;
    this.logger = logger;
    this.sessionFilter = sessionFilter ?? new SessionFilter();

    this.logger?.debug?.(
      `${TAG} Initialized: everyNConversations=${config.everyNConversations}, ` +
      `warmup=${config.enableWarmup ? "enabled" : "disabled"}, ` +
      `l1IdleTimeout=${config.l1.idleTimeoutSeconds}s, ` +
      `l2DelayAfterL1=${config.l2.delayAfterL1Seconds}s, ` +
      `l2MinInterval=${config.l2.minIntervalSeconds}s, ` +
      `l2MaxInterval=${config.l2.maxIntervalSeconds}s, ` +
      `sessionActiveWindow=${config.l2.sessionActiveWindowHours}h`,
    );

    // Wire up queue debug logging
    if (this.logger?.debug) {
      const debugFn = (msg: string) => this.logger?.debug?.(`${TAG} ${msg}`);
      this.l1Queue.setDebugLogger(debugFn);
      this.l2Queue.setDebugLogger(debugFn);
      this.l3Queue.setDebugLogger(debugFn);
    }
  }

  // ============================
  // Setup
  // ============================

  setL1Runner(runner: L1Runner): void {
    this.l1Runner = runner;
  }

  setL2Runner(runner: L2Runner): void {
    this.l2Runner = runner;
  }

  setL3Runner(runner: L3Runner): void {
    this.l3Runner = runner;
  }

  setPersister(persister: PipelineStatePersister): void {
    this.persister = persister;
  }

  /**
   * Restore session states from checkpoint and start the pipeline.
   * Sessions with pending counts will be immediately re-enqueued.
   */
  start(restoredStates?: Record<string, PipelineSessionState>): void {
    if (this.destroyed) return;

    if (restoredStates) {
      let skipped = 0;
      for (const [sessionKey, state] of Object.entries(restoredStates)) {
        if (this.sessionFilter.shouldSkip(sessionKey)) {
          skipped++;
          continue;
        }
        // Backfill warmup_threshold for sessions persisted before warm-up feature.
        // Missing field → treat as graduated (warmup already complete).
        const patched = { ...state };
        if (patched.warmup_threshold == null) {
          patched.warmup_threshold = 0;
        }
        this.sessionStates.set(sessionKey, patched);
      }
      this.logger?.info(
        `${TAG} Restored ${this.sessionStates.size} session state(s) from checkpoint` +
        (skipped > 0 ? ` (filtered ${skipped} internal)` : ""),
      );
    }

    // Recovery: re-enqueue sessions with pending work
    this.recoverPendingSessions();

    this.logger?.info(`${TAG} Pipeline started`);
  }

  // ============================
  // L0→L1: Notify (called from auto-capture on agent_end)
  // ============================

  /**
   * Get the effective conversation threshold for a session, considering warm-up.
   *
   * When warm-up is enabled, new sessions start with threshold=1 and double
   * after each successful L1 run: 1 → 2 → 4 → 8 → ... → everyNConversations.
   * Once the threshold reaches everyNConversations, warm-up is considered complete
   * (warmup_threshold is set to 0) and the fixed config value is used.
   */
  private getEffectiveThreshold(state: PipelineSessionState): number {
    if (!this.enableWarmup) return this.everyNConversations;
    // warmup_threshold === 0 means warm-up completed; use steady-state config
    if (state.warmup_threshold <= 0) return this.everyNConversations;
    return Math.min(state.warmup_threshold, this.everyNConversations);
  }

  /**
   * Advance the warm-up threshold for a session after a successful L1 run.
   * Doubles the threshold until it reaches everyNConversations, then marks
   * warm-up as complete (warmup_threshold = 0).
   */
  private advanceWarmupThreshold(state: PipelineSessionState): void {
    if (!this.enableWarmup) return;
    if (state.warmup_threshold <= 0) return; // already graduated

    const next = state.warmup_threshold * 2;
    if (next >= this.everyNConversations) {
      // Graduated: switch to steady-state
      state.warmup_threshold = 0;
      this.logger?.debug?.(`${TAG} Warm-up graduated → using steady-state threshold ${this.everyNConversations}`);
    } else {
      state.warmup_threshold = next;
      this.logger?.debug?.(`${TAG} Warm-up advanced → next threshold ${next}`);
    }
  }

  /**
   * Notify the pipeline that a conversation round has ended for a session,
   * and buffer the captured messages for L1 batch processing.
   *
   * Two trigger paths start here:
   * - **Path A (threshold)**: if conversation_count >= effective threshold
   *   (warm-up or steady-state), trigger L1 immediately with all buffered messages.
   * - **Path B (idle)**: reset the L1 idle timer. When the timer fires (user
   *   stops chatting), L1 runs with whatever has been buffered.
   */
  async notifyConversation(sessionKey: string, messages: CapturedMessage[]): Promise<void> {
    if (this.destroyed) return;
    if (this.sessionFilter.shouldSkip(sessionKey)) return;

    const state = this.getOrCreateState(sessionKey);
    state.conversation_count += 1;
    state.last_active_time = Date.now();

    // Reset L1 retry count on new conversation (environment may have recovered)
    const timers = this.getOrCreateTimers(sessionKey);
    timers.l1RetryCount = 0;

    // Buffer messages for L1
    const buffer = this.messageBuffers.get(sessionKey) ?? [];
    buffer.push(...messages);
    this.messageBuffers.set(sessionKey, buffer);

    const effectiveThreshold = this.getEffectiveThreshold(state);
    const warmupInfo = this.enableWarmup && state.warmup_threshold > 0
      ? ` (warmup: ${state.warmup_threshold})`
      : "";

    this.logger?.debug?.(
      `${TAG} [${sessionKey}] notify: conversation_count=${state.conversation_count}/${effectiveThreshold}${warmupInfo}, ` +
      `buffered_messages=${buffer.length} (+${messages.length} new)`,
    );

    await this.persistStates();

    // Path A: conversation count reached effective threshold → trigger L1 batch
    if (state.conversation_count >= effectiveThreshold) {
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] Conversation threshold reached (${state.conversation_count}>=${effectiveThreshold}${warmupInfo}), triggering L1`,
      );
      this.enqueueL1(sessionKey);
      return; // skip idle timer reset — L1 is already triggered
    }

    // Path B: below threshold → reset L1 idle timer (catch residual later)
    timers.l1Idle.schedule(this.l1IdleTimeoutMs, () => this.onL1IdleTimeout(sessionKey));
    this.logger?.debug?.(
      `${TAG} [${sessionKey}] L1 idle timer reset (${this.l1IdleTimeoutMs / 1000}s)`,
    );

    // Periodic GC: evict cold sessions from memory
    this.notifyCounter += 1;
    if (this.notifyCounter >= this.SESSION_GC_EVERY_N_NOTIFICATIONS) {
      this.notifyCounter = 0;
      this.gcStaleSessions();
    }
  }

  // ============================
  // Graceful shutdown
  // ============================

  /**
   * Per-session flush — scoped end-of-session handling.
   *
   * Semantically different from {@link destroy}:
   *   - ``destroy`` tears down the *whole* scheduler (meant for process
   *     shutdown such as OpenClaw's ``gateway_stop``).
   *   - ``flushSession`` only processes the one session identified by
   *     ``sessionKey`` and leaves every other session's timers, buffers
   *     and pipeline state untouched.  This is the correct semantic for
   *     the Gateway's ``POST /session/end`` endpoint and for Hermes'
   *     ``on_session_end`` callback, which fire when one conversation
   *     ends while the process keeps serving other concurrent sessions.
   *
   * What it does:
   *   1. Cancel the session's pending L1 idle timer (no further idle
   *      fires for this key).
   *   2. If the session's message buffer still holds work, enqueue an
   *      immediate L1 run for this session (``triggerReason="flush"``).
   *   3. Await the shared ``l1Queue`` so the caller observes L1
   *      completion before returning.  We do not selectively wait
   *      because L1 is already a single-consumer SerialQueue — waiting
   *      for ``onIdle`` is the cheapest correct signal.
   *
   * What it deliberately does NOT do:
   *   - Touch other sessions' timers / buffers / pipeline state.
   *   - Destroy the scheduler or any of its queues.
   *   - Reset global fields such as ``destroyed``.
   *
   * Unknown session keys are a no-op: the scheduler may legitimately
   * have evicted the session earlier via GC, or the session may never
   * have produced any captures.
   */
  async flushSession(sessionKey: string): Promise<void> {
    if (this.destroyed) return;
    if (this.sessionFilter.shouldSkip(sessionKey)) return;

    const timers = this.sessionTimers.get(sessionKey);
    const buffer = this.messageBuffers.get(sessionKey);

    // Step 1: cancel the idle timer so it won't fire after we return.
    if (timers?.l1Idle.pending) {
      timers.l1Idle.cancel();
    }

    // Step 2: flush pending buffered messages through L1 if any.
    if (buffer && buffer.length > 0) {
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] flushSession: enqueuing L1 for ${buffer.length} buffered message(s)`,
      );
      this.enqueueL1(sessionKey, "flush");
    }

    // Step 3: wait for L1 to drain.  L1 is a single-consumer SerialQueue
    // so this is the cheapest correct signal; it will not starve other
    // sessions because any cross-session interleaving L1 work was either
    // already queued or will be queued concurrently by their own capture
    // paths.
    await this.l1Queue.onIdle();

    this.logger?.debug?.(`${TAG} [${sessionKey}] flushSession: complete`);
  }

  /**
   * Maximum time (ms) to wait for pipeline flush during destroy.
   * Must be shorter than the gateway_stop hook timeout (3 s) to leave
   * headroom for VectorStore / EmbeddingService cleanup that runs after.
   */
  private readonly DESTROY_TIMEOUT_MS = 2_000;

  /**
   * Graceful shutdown with timeout protection:
   * 1. Mark destroyed, stop accepting new work
   * 2. Attempt to flush pending L1/L2/L3 work within DESTROY_TIMEOUT_MS
   * 3. If flush times out or fails, persist current state for recovery on next startup
   * 4. Pending work is never lost — it will be recovered via checkpoint on next start()
   */
  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;

    this.logger?.info(
      `${TAG} Destroying pipeline (timeout=${this.DESTROY_TIMEOUT_MS}ms)...`,
    );

    try {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([
        this._doFlush(),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => reject(new Error("destroy timeout")), this.DESTROY_TIMEOUT_MS);
        }),
      ]).finally(() => {
        if (timeoutId !== undefined) clearTimeout(timeoutId);
      });
      this.logger?.info(`${TAG} Pipeline flushed successfully`);
    } catch (err) {
      this.logger?.warn(
        `${TAG} Pipeline flush timed out or failed: ${err instanceof Error ? err.message : String(err)}. ` +
        `Pending work will be recovered on next startup.`,
      );
    }

    // Always persist state — whether flush succeeded, timed out, or failed.
    // This ensures pending work (buffered messages, L2 pending counts) is
    // saved to checkpoint and can be recovered by recoverPendingSessions().
    try {
      await this.persistStates();
    } catch (err) {
      this.logger?.error(
        `${TAG} Failed to persist states during destroy: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    this.logger?.info(`${TAG} Pipeline destroyed`);
  }

  /**
   * Internal: attempt to flush all pending pipeline work (L1 → L2 → L3).
   * Extracted from destroy() so it can be wrapped with a timeout.
   */
  private async _doFlush(): Promise<void> {
    // Step 1: Flush all L1 idle timers — only enqueue if there are buffered messages
    for (const [sessionKey, timers] of this.sessionTimers) {
      if (timers.l1Idle.pending) {
        timers.l1Idle.cancel(); // don't fire the idle callback directly
        const buffer = this.messageBuffers.get(sessionKey);
        if (buffer && buffer.length > 0) {
          this.logger?.debug?.(`${TAG} [${sessionKey}] Flush: enqueuing L1 for ${buffer.length} buffered messages`);
          this.enqueueL1(sessionKey, "flush");
        }
      }
    }

    // Step 2: Wait for L1 queue to drain
    this.logger?.debug?.(`${TAG} Waiting for L1 queue to drain (size=${this.l1Queue.size})`);
    await this.l1Queue.onIdle();

    // Step 3: Flush all L2 schedule timers
    for (const [sessionKey, timers] of this.sessionTimers) {
      if (timers.l2Schedule.pending) {
        this.logger?.debug?.(`${TAG} [${sessionKey}] Flush: triggering L2 schedule timer`);
        timers.l2Schedule.flush();
      }
    }

    // Step 4: Wait for all remaining queues to drain
    this.logger?.debug?.(`${TAG} Waiting for queues to drain (l2=${this.l2Queue.size}, l3=${this.l3Queue.size})`);
    await Promise.all([
      this.l2Queue.onIdle(),
      this.l3Queue.onIdle(),
    ]);
  }

  // ============================
  // Internal: L1 idle timeout handler
  // ============================

  private onL1IdleTimeout(sessionKey: string): void {
    const buffer = this.messageBuffers.get(sessionKey);
    const state = this.sessionStates.get(sessionKey);

    // We deliberately do NOT early-return when in-memory buffer/conversation_count
    // are both zero. The runner now over-fetches L0 from the DB, so a "small tail"
    // backlog (see runL1's hasMore branch) may have re-armed this idle timer with
    // nothing in memory but rows past the cursor in the DB.  enqueueL1 →
    // runL1 → runner will detect "no rows past cursor" and early-return cheaply.
    this.logger?.debug?.(
      `${TAG} [${sessionKey}] L1 idle timeout fired (buffered=${buffer?.length ?? 0}, conversations=${state?.conversation_count ?? 0})`,
    );
    this.enqueueL1(sessionKey, "idle_timeout");
  }

  // ============================
  // Internal: L1 queue
  // ============================

  private enqueueL1(sessionKey: string, triggerReason: "threshold" | "idle_timeout" | "flush" = "threshold"): void {
    const timers = this.getOrCreateTimers(sessionKey);

    // Don't double-queue
    if (timers.l1Queued) {
      this.logger?.debug?.(`${TAG} [${sessionKey}] L1 already queued, skipping`);
      return;
    }

    // Cancel idle timer if running (threshold beat it)
    timers.l1Idle.cancel();

    timers.l1Queued = true;
    this.logger?.debug?.(`${TAG} [${sessionKey}] Enqueuing L1 (queue=${this.l1Queue.name})`);

    // ── pipeline_l1_trigger metric ──
    const state = this.sessionStates.get(sessionKey);
    const buffer = this.messageBuffers.get(sessionKey);
    if (this.instanceId && this.logger) {
      report("pipeline_l1_trigger", {
        sessionKey,
        triggerReason,
        conversationCount: state?.conversation_count ?? 0,
        bufferedMessageCount: buffer?.length ?? 0,
      });
    }

    this.l1Queue.add(async () => {
      await this.runL1(sessionKey);
    }).catch((err) => {
      this.logger?.error(
        `${TAG} [${sessionKey}] L1 task failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
    }).finally(() => {
      timers.l1Queued = false;
    });
  }

  /**
   * L1 runner: Takes all buffered messages for a session and passes them
   * to the L1Runner for batch processing (e.g. appendEvent, local extraction).
   *
   * After L1 completes successfully:
   * - conversation_count and message buffer are reset
   * - L2 timer is advanced (downward-only) to allow remote record generation
   *
   * If L1 fails, conversation_count and buffer are preserved for retry
   * on next idle timeout or threshold trigger.
   */
  private async runL1(sessionKey: string): Promise<void> {
    const state = this.sessionStates.get(sessionKey);
    if (!state) return;

    // Drain the message buffer (take ownership, clear the shared ref)
    const buffer = this.messageBuffers.get(sessionKey) ?? [];
    this.messageBuffers.set(sessionKey, []);

    // NOTE: we no longer early-return when buffer + conversation_count are both
    // zero. The runner is now the source of truth for "is there work to do?":
    // it queries L0 from the DB via the cursor, so an idle-timeout-triggered
    // run with empty in-memory state is still meaningful — it will pick up any
    // backlog past `last_l1_cursor` (see runL1's hasMore branch). The runner
    // itself cheaply early-returns when the DB query returns 0 rows.

    this.logger?.debug?.(
      `${TAG} [${sessionKey}] L1 running: messages=${buffer.length}, conversation_count=${state.conversation_count}`,
    );

    if (!this.l1Runner) {
      this.logger?.warn(`${TAG} [${sessionKey}] No L1 runner set, skipping`);
      state.l2_pending_l1_count = state.conversation_count;
      state.conversation_count = 0;
      this.advanceWarmupThreshold(state);
      await this.persistStates();
      this.advanceL2Timer(sessionKey);
      return;
    }

    let runnerResult: L1RunnerResult | void;
    try {
      runnerResult = await this.l1Runner({
        sessionKey,
        msg: buffer,
        bg_msg: [], // reserved for future use
      });

      this.logger?.debug?.(
        `${TAG} [${sessionKey}] L1 complete: processed ${buffer.length} messages`,
      );
    } catch (err) {
      this.logger?.error(
        `${TAG} [${sessionKey}] L1 runner failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
      // On failure: put messages back into the buffer for retry
      const currentBuffer = this.messageBuffers.get(sessionKey) ?? [];
      this.messageBuffers.set(sessionKey, [...buffer, ...currentBuffer]);
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] L1 failure: restored ${buffer.length} messages to buffer (total=${buffer.length + currentBuffer.length})`,
      );

      // Re-arm L1 idle timer for automatic retry (with max retry limit)
      const timers = this.getOrCreateTimers(sessionKey);
      timers.l1RetryCount += 1;
      if (timers.l1RetryCount <= this.L1_MAX_RETRIES) {
        timers.l1Idle.schedule(this.L1_RETRY_DELAY_MS, () => this.onL1IdleTimeout(sessionKey));
        this.logger?.debug?.(
          `${TAG} [${sessionKey}] L1 retry scheduled in ${this.L1_RETRY_DELAY_MS / 1000}s ` +
          `(attempt ${timers.l1RetryCount}/${this.L1_MAX_RETRIES})`,
        );
      } else {
        this.logger?.warn(
          `${TAG} [${sessionKey}] L1 max retries reached (${this.L1_MAX_RETRIES}), ` +
          `giving up auto-retry. ${buffer.length + currentBuffer.length} messages remain buffered. ` +
          `Will resume on next user conversation.`,
        );
      }

      return; // don't advance state or trigger L2
    }

    // Success: reset retry count and advance state
    const timers = this.getOrCreateTimers(sessionKey);
    timers.l1RetryCount = 0;
    state.conversation_count = 0;
    this.advanceWarmupThreshold(state);
    const l2Keys = runnerResult?.profileScopes?.length ? runnerResult.profileScopes : [sessionKey];
    for (const l2Key of l2Keys) {
      const l2State = this.getOrCreateState(l2Key);
      l2State.l2_pending_l1_count = 1;
    }
    await this.persistStates();

    // Advance agent/profile-level L2 timer(s), respecting minInterval
    for (const l2Key of l2Keys) this.advanceL2Timer(l2Key);

    // ── L0 backlog drain ──────────────────────────────────
    //
    // The runner over-fetches (queries 2N rows but processes at most N) so
    // backlog state is detectable from the runner's return value.
    //   - hasFullBacklog → DB likely has many more rows past the cursor; drain
    //     immediately by enqueueing another L1 round on the shared queue.
    //   - hasMore (small tail, <2N) → defer to the existing l1Idle timer so
    //     the residual rows are picked up either when the user pauses or on
    //     the next conversation. Reuses the same idle handler — no special
    //     drain semantics, just a normal idle re-trigger.
    if (runnerResult && typeof runnerResult === "object") {
      if (runnerResult.hasFullBacklog) {
        this.logger?.debug?.(
          `${TAG} [${sessionKey}] L0 backlog detected (full batch) — enqueueing next L1 round`,
        );
        this.enqueueL1(sessionKey, "idle_timeout");
      } else if (runnerResult.hasMore) {
        this.logger?.debug?.(
          `${TAG} [${sessionKey}] L0 backlog detected (small tail) — arming L1 idle timer (${this.l1IdleTimeoutMs / 1000}s)`,
        );
        timers.l1Idle.schedule(this.l1IdleTimeoutMs, () => this.onL1IdleTimeout(sessionKey));
      }
    }
  }

  // ============================
  // Internal: L2 timer management (downward-only)
  // ============================

  /**
   * Advance the per-session L2 timer after an L1 event (new memory generated).
   *
   * Computes the desired fire time as:
   *   T_desired = max(now + l2DelayAfterL1, lastL2Time + l2MinInterval)
   *
   * The timer is only moved if T_desired is earlier than the current schedule
   * (downward-only semantics). If no timer is pending, it's set unconditionally.
   */
  private advanceL2Timer(sessionKey: string): void {
    if (this.destroyed) return;

    const timers = this.getOrCreateTimers(sessionKey);
    const now = Date.now();

    // Compute the floor: lastL2 + minInterval (rate-limit protection)
    const lastL2 = this.l2LastRunTime.get(sessionKey) ?? 0;
    const minIntervalFloor = lastL2 > 0 ? lastL2 + this.l2MinIntervalMs : 0;

    // Desired fire time: delay after L1, but no earlier than minInterval floor
    const desiredTime = Math.max(now + this.l2DelayAfterL1Ms, minIntervalFloor);

    const advanced = timers.l2Schedule.tryAdvanceTo(desiredTime, () => this.onL2TimerFired(sessionKey, "delay-after-l1"));

    if (advanced) {
      const delaySec = Math.round((desiredTime - now) / 1000);
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] L2 timer advanced: firing in ${delaySec}s` +
        (timers.l2Schedule.scheduledTime > 0
          ? ` (was ${Math.round((timers.l2Schedule.scheduledTime - now) / 1000)}s)`
          : " (newly armed)"),
      );
    } else {
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] L2 timer not advanced: current schedule is already earlier`,
      );
    }
  }

  /**
   * Arm the L2 timer for the maxInterval guarantee after L2 completes.
   * Sets T = now + l2MaxInterval (unconditional, replaces any pending timer).
   */
  private armL2MaxInterval(sessionKey: string): void {
    if (this.destroyed) return;

    const timers = this.getOrCreateTimers(sessionKey);
    const fireAt = Date.now() + this.l2MaxIntervalMs;
    timers.l2Schedule.scheduleAt(fireAt, () => this.onL2TimerFired(sessionKey, "max-interval"));

    this.logger?.debug?.(
      `${TAG} [${sessionKey}] L2 maxInterval timer armed: ${Math.round(this.l2MaxIntervalMs / 1000)}s`,
    );
  }

  /**
   * Called when a per-session L2 timer fires.
   *
   * Checks session activity: if the session is cold (inactive > activeWindow),
   * the timer is NOT re-armed — it will be revived by the next L1 event.
   * Otherwise, enqueues L2.
   *
   * The `source` parameter distinguishes the trigger origin:
   * - "delay-after-l1": fired shortly after L1 completed — skip cold check
   *   because L1 completion itself proves recent activity.
   * - "max-interval": periodic timer — apply cold check normally.
   */
  private onL2TimerFired(sessionKey: string, source: "delay-after-l1" | "max-interval"): void {
    const state = this.sessionStates.get(sessionKey);
    if (!state) return;

    const now = Date.now();

    // Cold session check: only applies to periodic (maxInterval) triggers.
    // Delay-after-L1 triggers are exempt because L1 just completed, proving
    // the session was recently active.
    if (source === "max-interval" && now - state.last_active_time >= this.sessionActiveWindowMs) {
      this.logger?.debug?.(
        `${TAG} [${sessionKey}] L2 timer fired but session is cold ` +
        `(inactive ${Math.round((now - state.last_active_time) / 3600_000)}h), timer stopped. ` +
        `Will re-arm on next L1 event.`,
      );
      return; // timer not re-armed — advanceL2Timer() in runL1 will revive it
    }

    this.enqueueL2(sessionKey, `timer:${source}`);
  }

  // ============================
  // Internal: L2 queue
  // ============================

  private enqueueL2(sessionKey: string, trigger: string): void {
    const timers = this.getOrCreateTimers(sessionKey);

    // Cancel any pending L2 timer (we're about to run L2)
    timers.l2Schedule.cancel();

    // Conflict detection: warn if L2 is already queued
    if (timers.l2Queued) {
      this.logger?.warn(
        `${TAG} [${sessionKey}] L2 enqueue conflict on queue "${this.l2Queue.name}": ` +
        `task already queued/running (trigger=${trigger}), skipping`,
      );
      return;
    }

    timers.l2Queued = true;
    this.logger?.debug?.(`${TAG} [${sessionKey}] Enqueuing L2 (trigger=${trigger}, queue=${this.l2Queue.name})`);

    this.l2Queue.add(async () => {
      await this.runL2(sessionKey);
    }).catch((err) => {
      this.logger?.error(
        `${TAG} [${sessionKey}] L2 task failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
    }).finally(() => {
      timers.l2Queued = false;
    });
  }

  private async runL2(sessionKey: string): Promise<void> {
    const state = this.sessionStates.get(sessionKey);
    if (!state) return;

    if (!this.l2Runner) {
      this.logger?.warn(`${TAG} [${sessionKey}] No L2 runner set, skipping`);
      return;
    }

    this.logger?.debug?.(
      `${TAG} [${sessionKey}] L2 running: l2_pending_l1_count=${state.l2_pending_l1_count}`,
    );

    const cursor = state.last_extraction_updated_time || undefined;

    let result: L2RunnerResult | void;
    try {
      result = await this.l2Runner(sessionKey, cursor);
    } catch (err) {
      this.logger?.error(
        `${TAG} [${sessionKey}] L2 runner failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
      // Even on failure, arm maxInterval so we retry eventually
      this.armL2MaxInterval(sessionKey);
      return;
    }

    // After L2: update state
    const now = Date.now();
    state.l2_pending_l1_count = 0;

    // Cold-start optimization: if this is the very first L2 run for this session
    // and it was skipped (no new records), do NOT update l2LastRunTime.
    // This prevents l2MinIntervalSeconds from blocking the next L2 trigger
    // when the first L1 extraction produces actual memories shortly after.
    const isFirstL2 = !this.l2LastRunTime.has(sessionKey);
    const wasSkipped = result?.skipped === true;

    if (isFirstL2 && wasSkipped) {
      this.logger?.info?.(
        `${TAG} [${sessionKey}] L2 cold-start skip: not updating l2LastRunTime ` +
        `(minInterval won't block next trigger)`,
      );
      this.armL2MaxInterval(sessionKey);
      await this.persistStates();
      return;
    }

    state.last_extraction_time = new Date().toISOString();
    state.l2_last_extraction_time = new Date().toISOString();
    this.l2LastRunTime.set(sessionKey, now);

    // Advance cursor using the record timestamp returned by the runner
    if (result?.latestCursor) {
      state.last_extraction_updated_time = result.latestCursor;
    }

    await this.persistStates();

    this.logger?.debug?.(`${TAG} [${sessionKey}] L2 complete`);

    // Arm the maxInterval timer for the next cycle
    this.armL2MaxInterval(sessionKey);

    // Trigger L3
    this.triggerL3();
  }

  // ============================
  // Internal: L3 queue (global, dedup)
  // ============================

  private triggerL3(): void {
    if (this.destroyed) return;

    if (this.l3Running) {
      // L3 is in progress — mark pending so it runs again after current finishes
      this.l3Pending = true;
      this.logger?.debug?.(`${TAG} L3 already running, marking pending`);
      return;
    }

    this.logger?.debug?.(`${TAG} Triggering L3`);
    this.enqueueL3();
  }

  private enqueueL3(): void {
    this.l3Running = true;
    this.l3Pending = false;

    this.logger?.debug?.(`${TAG} Enqueuing L3 (queue=${this.l3Queue.name})`);

    this.l3Queue.add(async () => {
      await this.runL3();
    }).catch((err) => {
      this.logger?.error(
        `${TAG} L3 task failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
    }).finally(() => {
      this.l3Running = false;

      // If new L2 completions happened while L3 was running, run again
      if (this.l3Pending && !this.destroyed) {
        this.logger?.debug?.(`${TAG} L3 has pending work, re-running`);
        this.enqueueL3();
      }
    });
  }

  private async runL3(): Promise<void> {
    if (!this.l3Runner) {
      this.logger?.warn(`${TAG} No L3 runner set, skipping`);
      return;
    }

    this.logger?.debug?.(`${TAG} L3 running`);
    try {
      await this.l3Runner();
      this.logger?.debug?.(`${TAG} L3 complete`);
    } catch (err) {
      this.logger?.error(
        `${TAG} L3 runner failed: ${err instanceof Error ? err.stack ?? err.message : String(err)}`,
      );
    }
  }

  // ============================
  // Internal: state management
  // ============================

  private getOrCreateState(sessionKey: string): PipelineSessionState {
    let state = this.sessionStates.get(sessionKey);
    if (!state) {
      state = {
        conversation_count: 0,
        last_extraction_time: "",
        last_extraction_updated_time: "",
        last_active_time: Date.now(),
        l2_pending_l1_count: 0,
        warmup_threshold: this.enableWarmup ? 1 : 0,
        l2_last_extraction_time: "",
      };
      this.sessionStates.set(sessionKey, state);
      this.logger?.debug?.(`${TAG} [${sessionKey}] Created new session state`);
    }
    return state;
  }

  private getOrCreateTimers(sessionKey: string): SessionTimerState {
    let timers = this.sessionTimers.get(sessionKey);
    if (!timers) {
      const isDestroyed = () => this.destroyed;
      timers = {
        l1Idle: new ManagedTimer(`L1-idle:${sessionKey}`, isDestroyed),
        l2Schedule: new ManagedTimer(`L2-schedule:${sessionKey}`, isDestroyed),
        l1Queued: false,
        l2Queued: false,
        l1RetryCount: 0,
      };
      this.sessionTimers.set(sessionKey, timers);
    }
    return timers;
  }

  private async persistStates(): Promise<void> {
    if (!this.persister) return;

    // PipelineSessionState only contains pipeline-owned fields, so we can
    // safely persist the entire object without risk of overwriting runner state.
    const obj: Record<string, PipelineSessionState> = {};
    for (const [k, v] of this.sessionStates) {
      obj[k] = { ...v };
    }
    try {
      this.logger?.debug?.(`Persisting states: ${JSON.stringify(obj)}`);
      await this.persister(obj);
    } catch (err) {
      this.logger?.error(
        `${TAG} Failed to persist states: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  /**
   * Evict cold sessions from in-memory maps to prevent unbounded growth.
   *
   * A session is eligible for GC when:
   * 1. Inactive for > sessionActiveWindowMs * SESSION_GC_INACTIVE_MULTIPLIER
   * 2. No queued/running L1 or L2 tasks
   * 3. No buffered messages pending processing
   *
   * Evicted sessions can be fully restored from checkpoint on next
   * `notifyConversation()` (state) or `start()` (recovery).
   */
  private gcStaleSessions(): void {
    const now = Date.now();
    const maxInactiveMs = this.sessionActiveWindowMs * this.SESSION_GC_INACTIVE_MULTIPLIER;
    let evictedCount = 0;

    for (const [sessionKey, state] of this.sessionStates) {
      if (now - state.last_active_time < maxInactiveMs) continue;

      // Safety: don't evict sessions with active work
      const timers = this.sessionTimers.get(sessionKey);
      if (timers?.l1Queued || timers?.l2Queued) continue;

      const buffer = this.messageBuffers.get(sessionKey);
      if (buffer && buffer.length > 0) continue;

      // Evict: cancel any pending timers, then remove from all maps
      if (timers) {
        timers.l1Idle.cancel();
        timers.l2Schedule.cancel();
      }
      this.sessionStates.delete(sessionKey);
      this.sessionTimers.delete(sessionKey);
      this.messageBuffers.delete(sessionKey);
      this.l2LastRunTime.delete(sessionKey);
      evictedCount++;
    }

    if (evictedCount > 0) {
      this.logger?.debug?.(
        `${TAG} Session GC: evicted ${evictedCount} cold session(s), ` +
        `${this.sessionStates.size} remaining`,
      );
    }
  }

  /**
   * Recovery: re-enqueue sessions that have pending work from before restart.
   *
   * On restart, message buffers are empty (in-memory only). Sessions with
   * non-zero conversation_count had messages that were either:
   * 1. Already processed by L1 (l2_pending_l1_count > 0) → arm L2 timer
   * 2. Never reached L1 (conversation_count > 0, messages lost) → arm L2
   *    as best-effort recovery
   *
   * We arm L2 timers (with delay) rather than enqueuing immediately,
   * because the pipeline may be starting during management commands.
   */
  private recoverPendingSessions(): void {
    for (const [sessionKey, state] of this.sessionStates) {
      if (state.conversation_count === 0 && state.l2_pending_l1_count === 0) continue;

      this.logger?.debug?.(
        `${TAG} [${sessionKey}] Recovery: conversation_count=${state.conversation_count}, ` +
        `l2_pending_l1_count=${state.l2_pending_l1_count}, arming L2 timer`,
      );

      // Reset conversation_count since we can't recover the messages
      state.l2_pending_l1_count = Math.max(state.l2_pending_l1_count, state.conversation_count);
      state.conversation_count = 0;

      // Arm L2 timer with delay (gives the system time to fully start)
      this.advanceL2Timer(sessionKey);
    }
  }

  // ============================
  // Public accessors (for testing / status)
  // ============================

  /** Get the pipeline session state for a session (read-only copy). */
  getSessionState(sessionKey: string): PipelineSessionState | undefined {
    const state = this.sessionStates.get(sessionKey);
    return state ? { ...state } : undefined;
  }

  /** Get the buffered message count for a session. */
  getBufferedMessageCount(sessionKey: string): number {
    return this.messageBuffers.get(sessionKey)?.length ?? 0;
  }

  /** Get all session keys being tracked. */
  getSessionKeys(): string[] {
    return Array.from(this.sessionStates.keys());
  }

  /** Whether the pipeline has been destroyed. */
  get isDestroyed(): boolean {
    return this.destroyed;
  }

  /** Queue sizes and running state for monitoring. */
  getQueueSizes(): {
    l1: number; l2: number; l3: number;
    l1Pending: boolean; l2Pending: boolean; l3Pending: boolean;
    l1Idle: boolean; l2Idle: boolean; l3Idle: boolean;
  } {
    return {
      l1: this.l1Queue.size,
      l2: this.l2Queue.size,
      l3: this.l3Queue.size,
      l1Pending: this.l1Queue.pending,
      l2Pending: this.l2Queue.pending,
      l3Pending: this.l3Queue.pending,
      l1Idle: this.l1Queue.idle,
      l2Idle: this.l2Queue.idle,
      l3Idle: this.l3Queue.idle,
    };
  }
}
