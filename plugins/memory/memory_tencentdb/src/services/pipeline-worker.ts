/**
 * PipelineWorker — 竞争消费 Pipeline 任务
 *
 * 需求 #12 Worker 竞争消费 + #13 死信队列与失败处理
 *
 * 架构文档 §方案C:
 * - XREADGROUP Consumer Group 竞争消费 task (单一队列)
 * - 分布式锁保护：per-session (L1/L2), per-instance (L3)
 * - 锁续约：每 30s 续约，续约失败 abort
 * - LLM 执行 + 写入：取 buffer → 调 LLM → 写 VDB/COS
 * - 级联调度：L1→L2 (via onL1Complete timer推进), L2→L3 (直接入队)
 * - 死信队列：超过重试上限 → 写入死信
 * - 重试策略：抢锁失败 5s 重投, LLM 超时指数退避 5s/15s/45s
 * - 幂等：VDB upsert by record_id, COS 覆盖写
 */

import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import { serializeTraceContext } from "../core/report/trace-propagation.js";
import { obsLogger } from "../core/report/obs-logger.js";

// ============================
// Types
// ============================

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

/** L1/L2/L3 任务执行器 (由上层注入具体的 LLM + VDB 逻辑)
 *
 * H-11 Step 2: methods optionally accept an AbortSignal. When the worker loses
 * its distributed lock mid-execution, it aborts the signal so the executor
 * can promptly tear down in-flight LLM calls. Executors that ignore the
 * signal still work (the worker will skip ACK after lockLost), but they
 * waste compute / tokens until the LLM call naturally returns.
 */
export interface TaskExecutor {
  executeL1(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeL2(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeL3(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeFlush?(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeOffloadL1?(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeOffloadL15?(task: TaskPayload, signal?: AbortSignal): Promise<void>;
  executeOffloadL2?(task: TaskPayload, signal?: AbortSignal): Promise<void>;
}

export interface PipelineWorkerConfig {
  /** Worker 节点 ID */
  workerId?: string;
  /** 并发消费协程数 (default: 60). 每个协程独立消费任务，不同 session 并行执行。 */
  concurrency?: number;
  /** 消费轮询间隔 ms (default: 200) */
  pollIntervalMs?: number;
  /**
   * 锁 TTL ms (default: 240000 = 4min)。
   * 必须 ≥ 2 × max(LLM timeout)：默认 LLM timeout 是 120s，
   * 留 2x buffer 保证即便续约 timer 因 GC / 事件循环阻塞错过 1-2 次
   * tick，锁也不会过期被别人抢走。
   */
  lockTtlMs?: number;
  /** 锁续约间隔 ms (default: 30000 = TTL 的 1/8，避免续约失败) */
  lockRenewIntervalMs?: number;
  /** 最大重试次数 (default: 3) */
  maxRetries?: number;
  /** 重试基础延迟 ms (default: 5000, 指数退避) */
  retryBaseDelayMs?: number;
  /** Pending 消息回收间隔 ms (default: 30000) */
  pendingRecoveryIntervalMs?: number;
  /** Pending 消息超时判定 ms (default: 300000 = 5min, 必须 > lockTtlMs) */
  pendingStaleMs?: number;
  /** 死信任务持久化回调 */
  onDeadLetter?: (task: TaskPayload, error: string, retryCount: number) => Promise<void>;
  /**
   * L1 完成后的回调，用于推进 L2 timer（解决 L2 快路径）。
   * 由 server.ts 注入 statefulManager.advanceL2TimerAfterL1。
   * 不注入则 L2 只靠 maxInterval 兜底。
   */
  onL1Complete?: (sessionId: string, instanceId: string, teamId?: string, agentId?: string) => Promise<void>;
  /**
   * L2 完成后的回调，用于设置 L2 maxInterval timer。
   * 由 server.ts 注入 statefulManager.armL2MaxInterval。
   */
  onL2Complete?: (sessionId: string, instanceId: string, teamId?: string, agentId?: string) => Promise<void>;
  /**
   * 分布式锁粒度 (default: "session")
   * - "session": L1/L2 per-session 锁, L3 per-instance 锁 (原行为, 最大并发)
   * - "instance": L1/L2/L3 全部 per-instance 锁 (CR-1 临时缓解: 防止同 instance 不同 session
   *   并发 append 到 daily JSONL 共享 key. 代价是单 instance 内 task 完全串行.)
   *
   * 切换该值不影响持久状态 (lock 是 TTL=120s 的临时 key).
   * 灰度时必须在 lockTtlMs 时间内完成全 worker 同步切换, 避免新老 worker 用不同 key 同时持锁.
   */
  lockGranularity?: "session" | "instance";

  /**
   * PipelineWorker 内部的并发信号量。historically 跨 memory + skill V2 worker 共享,
   * 2026-07-17 skill 改造后 skill 侧不再用信号量做并发上限, 现在只有 memory
   * pipeline 一个 consumer。未注入时行为不变。processTask 入口 acquire、finally release。
   */
  permitPool?: import("./worker-permit-pool.js").WorkerPermitPool;
}

export interface DeadLetterEntry {
  task: TaskPayload;
  error: string;
  retryCount: number;
  deadAt: number;
}

const TAG = "[pipeline-worker]";

// ============================
// PipelineWorker
// ============================

export class PipelineWorker {
  private backend: IStateBackend;
  private executor: TaskExecutor;
  private config: Required<Omit<PipelineWorkerConfig, "onDeadLetter" | "onL1Complete" | "onL2Complete" | "permitPool">> & {
    onDeadLetter?: PipelineWorkerConfig["onDeadLetter"];
    onL1Complete?: PipelineWorkerConfig["onL1Complete"];
    onL2Complete?: PipelineWorkerConfig["onL2Complete"];
    permitPool?: PipelineWorkerConfig["permitPool"];
  };
  private logger: Logger;

  private running = false;
  private destroyed = false;
  private recoveryTimer: ReturnType<typeof setInterval> | null = null;

  // Active locks tracked for graceful shutdown
  private activeLocks = new Set<string>();

  // In-flight tasks (consumed but not yet completed/failed/dropped). Used by
  // standalone /v2/pipeline/status to compute per-L-type running stats.
  // Service mode never reads this — it just costs a Map.set/delete per task.
  private runningTasks = new Map<string, TaskPayload>();

  // Dead letter queue (进程内 + 可选回调持久化)
  private deadLetterQueue: DeadLetterEntry[] = [];

  // Metrics
  private metrics = {
    tasksConsumed: 0,
    tasksCompleted: 0,
    tasksFailed: 0,
    tasksRetried: 0,
    tasksDeadLettered: 0,
    lockConflicts: 0,
    /** H-11: number of times renewLock callback failed → lockLost=true was set. */
    lockRenewFailed: 0,
    /** H-11: number of times execution finished but lockLost was true → task left in PENDING for another worker. */
    lockLostDuringExecution: 0,
    /** H-11 Step 2: number of times an executor was aborted via AbortSignal due to lockLost. */
    executionAborted: 0,
  };

  constructor(backend: IStateBackend, executor: TaskExecutor, config?: PipelineWorkerConfig, logger?: Logger) {
    this.backend = backend;
    this.executor = executor;
    this.logger = logger ?? { info: console.log, warn: console.warn, error: console.error };
    this.config = {
      workerId: config?.workerId ?? `worker-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
      concurrency: config?.concurrency ?? 60,
      pollIntervalMs: config?.pollIntervalMs ?? 200,
      lockTtlMs: config?.lockTtlMs ?? 240000,
      lockRenewIntervalMs: config?.lockRenewIntervalMs ?? 30000,
      maxRetries: config?.maxRetries ?? 3,
      retryBaseDelayMs: config?.retryBaseDelayMs ?? 5000,
      pendingRecoveryIntervalMs: config?.pendingRecoveryIntervalMs ?? 30000,
      pendingStaleMs: config?.pendingStaleMs ?? 300000,
      onDeadLetter: config?.onDeadLetter,
      onL1Complete: config?.onL1Complete,
      onL2Complete: config?.onL2Complete,
      lockGranularity: config?.lockGranularity ?? "session",
      permitPool: config?.permitPool,
    };
  }

  // ============================
  // Lifecycle
  // ============================

  async start(): Promise<void> {
    if (this.destroyed || this.running) return;
    this.running = true;

    this.logger.info(`${TAG} Starting (workerId=${this.config.workerId}, concurrency=${this.config.concurrency})`);

    // 启动 pending 消息回收循环
    this.startPendingRecovery();

    // 启动 N 个并发消费协程
    for (let i = 0; i < this.config.concurrency; i++) {
      this.consumeLoop();
    }
  }

  async stop(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.running = false;

    // 停止 pending recovery
    if (this.recoveryTimer) { clearInterval(this.recoveryTimer); this.recoveryTimer = null; }

    // 释放所有活跃锁
    for (const lockKey of this.activeLocks) {
      try { await this.backend.releaseLock(lockKey, this.config.workerId); } catch { /* best effort */ }
    }
    this.activeLocks.clear();

    this.logger.info(
      `${TAG} Stopped (consumed=${this.metrics.tasksConsumed}, completed=${this.metrics.tasksCompleted}, ` +
      `failed=${this.metrics.tasksFailed}, deadLettered=${this.metrics.tasksDeadLettered})`,
    );
  }

  getMetrics() {
    return { ...this.metrics, workerId: this.config.workerId, deadLetterCount: this.deadLetterQueue.length };
  }

  /**
   * Snapshot of tasks currently being executed by this worker (after lock
   * acquisition, before completion/failure). Used by standalone
   * /v2/pipeline/status to compute per-L-type running stats. Service mode
   * never calls this. Returns a fresh array (Map values copy).
   */
  getRunningTasks(): TaskPayload[] {
    return Array.from(this.runningTasks.values());
  }

  getDeadLetterQueue(): readonly DeadLetterEntry[] {
    return this.deadLetterQueue;
  }

  // ============================
  // Consume Loop
  // ============================

  private async consumeLoop(): Promise<void> {
    while (this.running && !this.destroyed) {
      try {
        const task = await this.backend.consumeTask(this.config.workerId, this.config.pollIntervalMs);
        if (!task) continue;

        this.metrics.tasksConsumed++;
        await this.processTask(task);
      } catch (err) {
        if (!this.destroyed) {
          this.logger.error(`${TAG} Consume loop error: ${err instanceof Error ? err.message : String(err)}`);
          await this.sleep(1000); // 避免疯狂重试
        }
      }
    }
  }

  // ============================
  // Task Processing
  // ============================

  private async processTask(task: TaskPayload): Promise<void> {
    const lockKey = this.getLockKey(task);
    const retryCount = (task.data?.retryCount as number) ?? 0;

    // permitPool acquire：memory pipeline 内部并发限流 (历史上跨 skill V2 共用，
    // 2026-07 后 skill 侧不再用信号量做并发上限)。未注入 pool 时是 no-op。
    // lock-free 与 with-lock 两条分支都要 finally release —— 用 releasePermitOnce 幂等。
    const permitPool = this.config.permitPool;
    if (permitPool) {
      await permitPool.acquire();
    }
    let permitReleased = false;
    const releasePermitOnce = (): void => {
      if (permitReleased) return;
      permitReleased = true;
      if (permitPool) {
        try { permitPool.release(); }
        catch (err) {
          this.logger.warn(`${TAG} permitPool release error: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    };

    // Lock-free path: offload-l1 doesn't need distributed lock
    if (lockKey === null) {
      this.runningTasks.set(task.id, task);
      try {
        await this.executeTask(task, undefined);

        // ACK
        const msgId = (task as any)._msgId;
        if (msgId) await this.backend.ackTask(msgId);

        this.metrics.tasksCompleted++;
        this.logger?.debug?.(`${TAG} Task completed (lock-free): ${task.type} [${task.instanceId}/${task.sessionId}]`);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        this.metrics.tasksFailed++;

        if (retryCount < this.config.maxRetries) {
          const delay = this.config.retryBaseDelayMs * Math.pow(3, retryCount);
          this.logger.warn(`${TAG} Task failed (lock-free, retry ${retryCount + 1}/${this.config.maxRetries}, delay=${delay}ms): ${errMsg}`);
          const msgId = (task as any)._msgId;
          if (msgId) { try { await this.backend.ackTask(msgId); } catch { /* best effort */ } }
          await this.sleep(delay);
          await this.reEnqueue(task, retryCount + 1);
          this.metrics.tasksRetried++;
        } else {
          await this.moveToDeadLetter(task, errMsg, retryCount);
        }
      } finally {
        this.runningTasks.delete(task.id);
        releasePermitOnce();

        // Deferred enqueue (same as locked path)
        const deferred = (task as any)._deferredEnqueue as TaskPayload[] | undefined;
        if (deferred?.length) {
          for (const dTask of deferred) {
            try {
              await this.backend.enqueueTask(dTask);
              this.logger?.debug?.(`${TAG} Deferred enqueue: ${dTask.type} [${dTask.id}]`);
            } catch (err) {
              this.logger?.warn?.(`${TAG} Deferred enqueue failed: ${err instanceof Error ? err.message : String(err)}`);
            }
          }
        }
      }
      return;
    }

    // Step 1: 抢分布式锁
    const locked = await this.backend.acquireLock(lockKey, this.config.workerId, this.config.lockTtlMs);
    if (!locked) {
      this.metrics.lockConflicts++;

      // offload-l2: skip immediately on lock conflict (idempotent timer will re-trigger)
      if (task.type === "offload-l2") {
        this.logger?.debug?.(`${TAG} Lock conflict [offload-l2] (task=${task.id}): ${lockKey}, skip (timer will re-trigger)`);
        const msgId = (task as any)._msgId;
        if (msgId) {
          try { await this.backend.ackTask(msgId); } catch { /* best effort */ }
        }
        releasePermitOnce();
        return;
      }

      // Lock conflict: current coroutine waits locally (no re-enqueue to stream).
      // Exponential backoff: 200ms → 600ms → 1.8s → 5s (capped), retry until lockTtlMs exhausted.
      // 旧版本固定 sleep(5000) 在 instance 级锁下会造成排队体感差(同 instance 多 session 累积秒级延迟);
      // 改为指数退避后, 大多数冲突在 1s 内解决, 同时保留长尾退避避免后端压力.
      // Only this coroutine is occupied; other 9 continue consuming different sessions.
      const deadline = Date.now() + this.config.lockTtlMs;
      let acquired = false;
      let attempt = 0;
      let delay = 200;
      while (Date.now() < deadline && this.running) {
        attempt++;
        this.logger?.debug?.(`${TAG} Lock conflict [${task.type}] (task=${task.id}): ${lockKey}, retry ${attempt} after ${delay}ms`);
        await this.sleep(delay);
        acquired = await this.backend.acquireLock(lockKey, this.config.workerId, this.config.lockTtlMs);
        if (acquired) break;
        delay = Math.min(delay * 3, 5000);
      }
      if (!acquired) {
        this.logger?.warn?.(`${TAG} Lock conflict timeout [${task.type}] (task=${task.id}): ${lockKey}, dropping task`);
        // CR-1 fix: ACK to prevent stale recovery from re-claiming this message in an
        // infinite loop. Without it, XPENDING keeps returning this msgId every
        // pendingRecoveryIntervalMs, exhausting worker slots.
        const msgId = (task as any)._msgId;
        if (msgId) {
          try { await this.backend.ackTask(msgId); } catch { /* best effort */ }
        }
        releasePermitOnce();
        return;
      }
      // Fall through to execute with acquired lock
    }

    this.activeLocks.add(lockKey);
    // Track in-flight task — used by standalone /v2/pipeline/status. Done after
    // lock acquisition so lock-conflict drops don't pollute the running set.
    this.runningTasks.set(task.id, task);
    let lockLost = false;
    // H-11 Step 2: AbortController so renewLock failure can immediately interrupt
    // long-running LLM calls inside the executor (saves token cost and avoids
    // writing data after the lock has been transferred to another worker).
    const abortController = new AbortController();

    // Step 2: 启动锁续约 (局部 timer，per-task 独立)
    const renewTimer = setInterval(async () => {
      try {
        const renewed = await this.backend.renewLock(lockKey, this.config.workerId, this.config.lockTtlMs);
        if (!renewed) {
          this.metrics.lockRenewFailed++;
          this.logger.warn(
            `${TAG} Lock renew failed for ${lockKey} (worker=${this.config.workerId}); ` +
            `marking lockLost and aborting executor`,
          );
          lockLost = true;
          clearInterval(renewTimer);
          // H-11 Step 2: signal the executor to abort. Any in-flight LLM / VDB call
          // wired to this signal will throw an AbortError and tear down cleanly.
          if (!abortController.signal.aborted) {
            this.metrics.executionAborted++;
            abortController.abort(new Error("pipeline-worker: lock lost during execution"));
          }
        }
      } catch (e) {
        this.metrics.lockRenewFailed++;
        this.logger.warn(
          `${TAG} Lock renew threw for ${lockKey}: ${e instanceof Error ? e.message : String(e)}`,
        );
        lockLost = true;
        clearInterval(renewTimer);
        if (!abortController.signal.aborted) {
          this.metrics.executionAborted++;
          abortController.abort(new Error("pipeline-worker: lock renew exception"));
        }
      }
    }, this.config.lockRenewIntervalMs);

    // Step 3: 执行任务
    try {
      await this.executeTask(task, abortController.signal);

      // H-11 Step 1: re-check lockLost after successful executeTask.
      // If the lock was lost mid-execution we must NOT ack and NOT cascade
      // because another worker has already (or will) take over via XPENDING/XCLAIM
      // recovery, and ACK'ing here would cause a silent partial-failure where the
      // task is removed from the stream while only half of its side effects landed.
      if (lockLost) {
        this.metrics.lockLostDuringExecution++;
        this.logger.warn(
          `${TAG} Lock lost during execution but task body returned; ` +
          `skipping ACK + cascadeSchedule so another worker can re-process: ` +
          `${task.type} [${task.instanceId}/${task.sessionId}]`,
        );
        // NOTE: rely on L1/L2/L3 idempotency (vectorStore.upsert by memoryId
        // is idempotent; jsonl appends use ETag/append-position so concurrent
        // writers don't corrupt). Hard rollback not feasible for COS objects.
        return;
      }

      // Step 4: ACK
      const msgId = (task as any)._msgId;
      if (msgId) await this.backend.ackTask(msgId);

      this.metrics.tasksCompleted++;
      this.logger?.debug?.(`${TAG} Task completed: ${task.type} [${task.instanceId}/${task.sessionId}]`);

      // Step 5: 级联调度 (L1→L2, L2→L3)
      await this.cascadeSchedule(task);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);

      // 检查锁是否丢失 → 如果丢失，不重试（避免重复执行）
      if (lockLost) {
        this.logger.warn(`${TAG} Lock lost during execution, aborting: ${task.type} [${task.instanceId}/${task.sessionId}]`);
        this.metrics.tasksFailed++;
        return;
      }

      this.metrics.tasksFailed++;

      // 指数退避重试
      if (retryCount < this.config.maxRetries) {
        const delay = this.config.retryBaseDelayMs * Math.pow(3, retryCount); // 5s, 15s, 45s
        this.logger.warn(
          `${TAG} Task failed (retry ${retryCount + 1}/${this.config.maxRetries}, delay=${delay}ms): ${errMsg}`,
        );
        // CR-1 fix: ACK the original message before re-enqueue. Otherwise the original
        // msgId stays in XPENDING and gets re-claimed by stale recovery in parallel
        // with the retry, causing the same task to run twice.
        const msgId = (task as any)._msgId;
        if (msgId) {
          try { await this.backend.ackTask(msgId); } catch { /* best effort */ }
        }
        await this.sleep(delay);
        await this.reEnqueue(task, retryCount + 1);
        this.metrics.tasksRetried++;
      } else {
        await this.moveToDeadLetter(task, errMsg, retryCount);
      }
    } finally {
      // Step 6: 停止续约 + 释放锁
      clearInterval(renewTimer);
      this.activeLocks.delete(lockKey);
      this.runningTasks.delete(task.id);
      try { await this.backend.releaseLock(lockKey, this.config.workerId); } catch { /* best effort */ }
      releasePermitOnce();

      // Step 7: 延迟入队 — executor 可通过 task._deferredEnqueue 暂存需要在锁释放后才入队的任务，
      // 避免新任务立即被消费时因同 session 锁仍被持有而产生不必要的锁冲突。
      const deferred = (task as any)._deferredEnqueue as TaskPayload[] | undefined;
      if (deferred?.length) {
        for (const dTask of deferred) {
          try {
            await this.backend.enqueueTask(dTask);
            this.logger?.debug?.(`${TAG} Deferred enqueue: ${dTask.type} [${dTask.id}]`);
          } catch (err) {
            this.logger?.warn?.(`${TAG} Deferred enqueue failed: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      }
    }
  }

  private async executeTask(task: TaskPayload, signal?: AbortSignal): Promise<void> {
    switch (task.type) {
      case "L1": return this.executor.executeL1(task, signal);
      case "L2": return this.executor.executeL2(task, signal);
      case "L3": return this.executor.executeL3(task, signal);
      case "flush": return this.executor.executeFlush?.(task, signal) ?? this.executor.executeL1(task, signal);
      case "offload-l1": return this.executor.executeOffloadL1?.(task, signal);
      case "offload-l15": return this.executor.executeOffloadL15?.(task, signal);
      case "offload-l2": return this.executor.executeOffloadL2?.(task, signal);
      default:
        this.logger.warn(`${TAG} Unknown task type: ${task.type}`);
    }
  }

  // ============================
  // 级联调度
  // ============================

  private async cascadeSchedule(task: TaskPayload): Promise<void> {
    const now = Date.now();
    const tid = task.teamId ?? (task.data as any)?.teamId;
    const aid = task.agentId ?? (task.data as any)?.agentId;

    if (task.type === "L1" || task.type === "flush") {
      // L1 完成 → reset session-level L1 state, then advance agent/profile-level L2 timers.
      await this.backend.updateSessionState(task.instanceId, task.sessionId, {
        conversation_count: 0,
      }, tid, aid);
      const profileScopes = Array.isArray((task as any)._l2ProfileScopes)
        ? ((task as any)._l2ProfileScopes as string[]).filter(Boolean)
        : [];
      const l2Keys = profileScopes.length > 0 ? profileScopes : [task.sessionId];
      if (this.config.onL1Complete) {
        for (const l2Key of l2Keys) {
          try {
            await this.backend.updateSessionState(task.instanceId, l2Key, { l2_pending_l1_count: 1 }, tid, aid);
            await this.config.onL1Complete(l2Key, task.instanceId, tid, aid);
          } catch (err) {
            this.logger?.warn?.(`${TAG} onL1Complete failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      }
      this.logger?.debug?.(`${TAG} [${task.instanceId}/${task.sessionId}] L1 done → L2 timer advanced (${l2Keys.join(",")})`);
    }

    if (task.type === "L2") {
      // If L2 was skipped (no new L1 records), don't cascade to L3 or arm timer
      if ((task as any)._l2Skipped) {
        this.logger?.debug?.(`${TAG} [${task.instanceId}/${task.sessionId}] L2 skipped (no new data), not arming timer or enqueuing L3`);
        return;
      }

      // L2 完成 → 直接入队 L3（携带 trace context 用于跨异步链路关联）
      // L3 task 也带 team/agent，保锁粒度对齐
      await this.backend.enqueueTask({
        id: `L3-${task.instanceId}-${now}`,
        type: "L3",
        instanceId: task.instanceId,
        sessionId: task.sessionId,
        teamId: tid,
        agentId: aid,
        priority: 2,
        data: task.data ? { ...task.data, ...serializeTraceContext() } : { teamId: tid, agentId: aid, ...serializeTraceContext() },
        createdAt: now,
      });
      await this.backend.updateSessionState(task.instanceId, task.sessionId, {
        l2_pending_l1_count: 0,
        l2_last_extraction_time: new Date().toISOString(),
      }, tid, aid);
      // onL2Complete 由 server.ts 注入 statefulManager.armL2MaxInterval
      if (this.config.onL2Complete) {
        try {
          await this.config.onL2Complete(task.sessionId, task.instanceId, tid, aid);
        } catch (err) {
          this.logger?.warn?.(`${TAG} onL2Complete failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      this.logger?.debug?.(`${TAG} [${task.instanceId}/${task.sessionId}] L2 done → L3 enqueued`);
    }
  }

  // ============================
  // Lock Management
  // ============================

  /**
   * Lock key 设计：
   *
   * v2 pipeline 默认 (lockGranularity="session", 实际是 (instance, team, agent) 散开):
   *   - L1: pipeline:{inst:tid:aid}:s:{sess}   — session 级锁
   *           L1 数据按 (team,user,agent,session) 在 TCVDB 隔离，
   *           不同 session 真并发抽取
   *   - L2: pipeline:{inst:tid:aid}            — agent 级锁
   *           L2 落地是 profiles/team:T|agent:X/scene_blocks/ 共享目录，
   *           同 agent 不同 session 的 L2 必须互斥避免撞写 scene/index 文件
   *   - L3: pipeline:{inst:tid:aid}            — agent 级锁（同 L2）
   *           L3 写 profiles/team:T|agent:X/persona.md 一个 agent 一份
   *
   * 跨 agent 完全并发：不同 (tid, aid) 散到不同 Redis Cluster slot，
   * 避免单 instance 集中到一个 hash slot 形成大 key 热点。
   *
   * teamId / agentId 缺失（旧调用 / offload）时退化到 "_:_" 占位，
   * 等价于按 instance 维度落同一 slot —— 兼容老行为，不会破坏锁互斥。
   *
   * lockGranularity="instance" (legacy CR-1 缓解):
   *   - L1/L2/L3: pipeline:{instanceId}        — 全部 instance 级共享同一把锁
   *   不推荐使用，保留向后兼容。新部署用默认即可。
   *
   * Rolling upgrade caveat:
   *   升级窗口期，新旧 pod 看到的 lock key 格式不同，可能短暂破坏跨 pod
   *   互斥。本次升级已通过 keyPrefix 从 tdai_memory → tdai_memory_v2 物理隔离，
   *   新老 pod 走不同 redis 命名空间，无交叉。
   */
  private getLockKey(task: TaskPayload): string | null {
    // offload-l1 is lock-free: rename guarantees exclusive file ownership,
    // appendFile is atomic (O_APPEND), and state.json is read-only for L1.
    if (task.type === "offload-l1") return null;

    // offload-l2: per-MMD lock so different MMDs can be processed concurrently.
    if (task.type === "offload-l2") {
      const mmdFile = (task.data as any)?.targetMmdFile ?? "default";
      return `pipeline:{${task.instanceId}}:offload-l2:${mmdFile}`;
    }

    // offload-l15: lock-free at worker level. The executor acquires a short
    // lock only during the final write phase (state.json update), allowing
    // multiple L1.5 LLM calls to run concurrently without blocking each other.
    if (task.type === "offload-l15") return null;

    if (this.config.lockGranularity === "instance") {
      return `pipeline:{${task.instanceId}}`;
    }

    // v2 默认：按 (instance, team, agent) 散开 hash tag
    //
    // teamId/agentId 优先级：
    //   1. task.teamId / task.agentId（v2 入队时显式带）
    //   2. task.data.teamId / task.data.agentId（兼容老调用）
    //   3. 从 task.sessionId 解析（timer-scanner 入队的 L2/L3 task,
    //      sessionId 形如 "profile:team:T|agent:A" 或
    //      "profile:team:T|agent:A|session:S" 时从里面抠出 tid/aid）
    //   4. "_" 占位退化到 instance 级（不推荐，hash 集中）
    let tid = task.teamId || (task.data as any)?.teamId;
    let aid = task.agentId || (task.data as any)?.agentId;
    if (!tid || !aid) {
      const m = task.sessionId.match(/^profile:team:([^|]+)\|agent:([^|]+)(?:\|session:.+)?$/);
      if (m) {
        // profile scope 里的 team 字段实际是 (teamId || userId)，与 buildProfileIsolationScope 一致。
        // 这里直接当 teamId 用即可，hash 分桶维度对齐就行。
        // 如果 key 携带 source session，它只作为 L2 输入边界，不进入锁粒度。
        tid = tid || m[1];
        aid = aid || m[2];
      }
    }
    tid = tid || "_";
    aid = aid || "_";
    const ns = `{${task.instanceId}:${tid}:${aid}}`;

    if (task.type === "L2" || task.type === "L3") {
      // agent 级锁：同 agent 的 L2/L3 互斥，避免共享目录撞写
      return `pipeline:${ns}`;
    }
    // L1 + flush 仍是 session 级
    return `pipeline:${ns}:s:${task.sessionId}`;
  }

  // ============================
  // Dead Letter (#13)
  // ============================

  private async moveToDeadLetter(task: TaskPayload, error: string, retryCount: number): Promise<void> {
    const entry: DeadLetterEntry = { task, error, retryCount, deadAt: Date.now() };
    this.deadLetterQueue.push(entry);
    this.metrics.tasksDeadLettered++;

    this.logger.error(
      `${TAG} Dead letter: ${task.type} [${task.instanceId}/${task.sessionId}] after ${retryCount} retries: ${error}`,
    );

    // CR-1 fix: ACK the original message to prevent stale recovery from picking it up
    // again. Without this, a dead-lettered task remains in XPENDING and gets re-claimed
    // every pendingRecoveryIntervalMs, causing infinite retry loops that block the
    // worker pool (root-caused from a production incident).
    const msgId = (task as any)._msgId;
    if (msgId) {
      try { await this.backend.ackTask(msgId); } catch { /* best effort */ }
    }

    // Clean up timers for this session to prevent ghost triggers
    try {
      const tid = task.teamId ?? (task.data as any)?.teamId;
      const aid = task.agentId ?? (task.data as any)?.agentId;
      await this.backend.removeTimer(task.instanceId, buildPipelineTimerMember(task.sessionId, "L1_idle", { teamId: tid, agentId: aid }));
      await this.backend.removeTimer(task.instanceId, buildPipelineTimerMember(task.sessionId, "L2_schedule", { teamId: tid, agentId: aid }));
    } catch { /* best effort */ }

    // 关键节点日志：任务进入死信队列
    obsLogger.error("core.task.dead_letter", {
      instance_id: task.instanceId,
      session_id: task.sessionId,
      task_type: task.type,
      task_id: task.id,
      error,
      retry_count: retryCount,
    });

    // 持久化回调（写 COS / Stream）
    if (this.config.onDeadLetter) {
      try {
        await this.config.onDeadLetter(task, error, retryCount);
      } catch (err) {
        this.logger.error(`${TAG} onDeadLetter callback failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  private async reEnqueue(task: TaskPayload, newRetryCount: number): Promise<void> {
    await this.backend.enqueueTask({
      ...task,
      id: `${task.type}-${task.sessionId}-retry${newRetryCount}-${Date.now()}`,
      data: { ...task.data, retryCount: newRetryCount },
      createdAt: Date.now(),
    });
  }

  // ============================
  // Pending Message Recovery (#13.2: XPENDING 超时检测 + XCLAIM)
  // ============================

  /**
   * 定期扫描远程队列中超时未 ACK 的 pending 消息。
   *
   * 当某个 Worker 进程挂了，它消费过但未 ACK 的消息会卡在 pending 列表。
   * 存活的 Worker 通过后端的 claimStaleTasks 接管这些消息重新处理。
   *
   * 保证：
   * - 幂等: VDB upsert by record_id, COS 覆盖写
   * - 不重复: 后端原子转移所有权，同一消息只会被一个 Worker 认领
   */
  private startPendingRecovery(): void {
    if (!this.backend.claimStaleTasks) return; // LocalStateBackend 不需要

    this.recoveryTimer = setInterval(async () => {
      if (this.destroyed) return;
      try {
        const stale = await this.backend.claimStaleTasks!(
          this.config.workerId,
          this.config.pendingStaleMs,
          10, // 每次最多认领 10 条
        );
        if (stale.length > 0) {
          this.logger.info(`${TAG} Recovered ${stale.length} stale pending task(s)`);
          for (const task of stale) {
            this.metrics.tasksConsumed++;
            // 直接处理认领到的任务（走正常 processTask 流程）
            this.processTask(task).catch((err) => {
              this.logger.error(`${TAG} Recovery task failed: ${err instanceof Error ? err.message : String(err)}`);
            });
          }
        }
      } catch (err) {
        this.logger.warn(`${TAG} Pending recovery error: ${err instanceof Error ? err.message : String(err)}`);
      }
    }, this.config.pendingRecoveryIntervalMs);
  }

  // ============================
  // Util
  // ============================

  private sleep(ms: number): Promise<void> {
    return new Promise((r) => { const t = setTimeout(r, ms); t.unref(); });
  }
}
