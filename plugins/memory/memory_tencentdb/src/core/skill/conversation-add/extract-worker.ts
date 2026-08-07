/**
 * SkillConversationExtractWorker — §9。
 *
 * 常驻循环：
 *   ① BRPOP agent
 *   ② 抢 extract-lock；拿不到 → requeue + sleep
 *   ③ 抢 tasks-mutex 读队首 task
 *   ④ 读 archive:
 *        404 → 幽灵 task → 抢 mutex filter 删 task, 跳过
 *        成功 → SkillExtractor.extract → sink.applyCandidates → 抢 mutex filter 删 task
 *   ⑤ 判断是否重入队 (还有 task → requeue, 否则 removeAgent)
 *   ⑥ 释放 extract-lock
 *
 * 并发保护：
 *   - agent 级 extract-lock 保证同一 agent 只有一个 Worker 抽取
 *   - tasks-mutex 保护 _tasks.json 读改写（跟 Handler 归档段的 mutex 是同一把）
 */

import type {
  AgentTuple,
  ExtractLockHandle,
  ISkillAgentTaskQueue,
} from "./agent-task-queue.js";
import type {
  SkillBufferStorage,
  SkillDeadTaskEntry,
  SkillTaskEntry,
} from "./buffer-storage.js";
import type {
  ExtractedCandidate,
  ExtractorLogger,
  ISkillExtractor,
} from "../queue/types.js";
import { runInRootContext } from "../../report/otel-context.js";
import { obsLogger } from "../../report/obs-logger.js";
import { trace } from "../../report/trace.js";

/**
 * 把 candidates 落到业务侧（例如调 SkillCore.create/patch，或者直接写 SkillStore）。
 * 由 wiring 层注入，Worker 不关心具体实现。必须幂等（Client 重试可能导致 task 被抽多次）。
 */
export interface SkillCandidatesSink {
  applyCandidates(input: {
    task: SkillTaskEntry;
    candidates: ExtractedCandidate[];
    /** trace / logging 用途 */
    workerId: string;
  }): Promise<void>;
}

export interface SkillConversationExtractWorkerOptions {
  workerId: string;
  buffer: SkillBufferStorage;
  queue: ISkillAgentTaskQueue;
  extractor: ISkillExtractor;
  sink: SkillCandidatesSink;
  logger: ExtractorLogger;

  /** BRPOP 阻塞时长 ms，默认 5000。 */
  brpopBlockMs?: number;
  /** extract-lock TTL ms，默认 600_000 (10 min)。 */
  extractLockTtlMs?: number;
  /**
   * extract-lock 续约间隔 ms，默认 extractLockTtlMs / 4。
   *
   * 处理 skill extract 走 LLM tool-calling review agent，可能跨多轮 iteration，
   * 累计耗时接近或超过 lockTtl。续约保证 Worker 忙着抽的时候不会被别的 Worker
   * 抢锁。默认 lockRenewIntervalMs = ttl/4 (跟历史 V2 worker 参数对齐)。
   */
  extractLockRenewIntervalMs?: number;
  /** tasks-mutex 锁 TTL ms（进程崩溃兜底），默认 10000。 */
  tasksMutexLockTtlMs?: number;
  /** tasks-mutex 争抢等待最长时间，默认 30000。 */
  tasksMutexWaitDeadlineMs?: number;
  /** 抢 extract-lock 失败后重入队，睡多少 ms 再 dequeue。默认 2000 + 抖动。 */
  lockContentionSleepMs?: number;
  lockContentionSleepJitterMs?: number;
  /**
   * agent 一轮处理最多抽多少条 task 就 requeue 让位（保证公平），默认 1。
   * 设计文档里 Worker 每次处理一条 task 就把 agent 塞回队头。
   */
  tasksPerRound?: number;
  /** 时间源，测试注入。 */
  now?: () => number;

  // ── 失败处理（transient / permanent 分级 + DLQ） ────────────────────────
  //
  // 对齐设计文档 §3.6 (7) P0 修复：抽取失败原来 catch 里直接 requeue+break，
  // runLoop 立刻 dequeue 同 agent 形成 ~100 次/秒 的 hot retry，浪费 LLM
  // 额度、灌爆日志。现在按错误性质分两类：
  //
  //   A) transient (401/403/429/5xx/网络/timeout/fetch)
  //      → sleep(failureRequeueSleepMs) → requeue
  //      → retry_count 不变，不入 DLQ，无限重试等外部恢复
  //      → warn 采样：每 transientLogSampleEvery 次打一条 warn
  //   B) permanent (400/422/JSON parse/schema)
  //      → sleep → retry_count++ 回写 _tasks.json → requeue
  //      → retry_count >= permanentMaxRetries 时移到 _tasks_dlq.json
  //      → 无法分类的错误按 A 处理（兜底不丢数据）
  /**
   * 失败后 requeue 前 sleep 多少 ms，固定值（不做指数退避、不做 jitter）。
   * 默认 2000。
   */
  failureRequeueSleepMs?: number;
  /**
   * permanent 错累计多少次进 DLQ，默认 3。
   */
  permanentMaxRetries?: number;
  /**
   * transient 错误按 task_id 采样打 warn 的间隔（次）。第 1 次打 error，
   * 之后每 N 次打一条 warn，防日志刷屏。默认 60。
   */
  transientLogSampleEvery?: number;
}

export class SkillConversationExtractWorker {
  private readonly opts: SkillConversationExtractWorkerOptions;
  private readonly logger: ExtractorLogger;
  private closed = false;
  private started = false;
  private loopPromise: Promise<void> | null = null;
  /**
   * per-task_id transient 失败计数（进程内计数，不落盘）。用于 warn 采样：
   * 首次失败打 error，之后每 transientLogSampleEvery 次打一条 warn。进程重启
   * 后重新计数没关系——采样目的只是限日志频次，不是审计。
   */
  private readonly transientFailStreak = new Map<string, number>();

  constructor(opts: SkillConversationExtractWorkerOptions) {
    this.opts = opts;
    this.logger = opts.logger;
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.closed = false;
    this.logger.info(
      `[skill-conv-worker] start id=${this.opts.workerId} brpopBlockMs=${this.opts.brpopBlockMs ?? 5000} ` +
        `extractLockTtlMs=${this.opts.extractLockTtlMs ?? 600_000}`,
    );
    // 关键：把 runLoop 放到 OTel ROOT_CONTEXT 里启动。
    //
    // 本 worker 常在某个 HTTP 请求 handler 里被懒启动（resolveConversationAdd →
    // wireConversationAdd → start()）。若不脱离上下文，永不退出的 runLoop 会
    // 永久继承"启动那一刻"的 active span，导致之后每次抽取的 LLM span 都挂进
    // 那条请求 trace，被 Langfuse 合并成一条（tags 跨多 agent、sessionId 混乱）。
    // 详见 report/otel-context.ts。
    this.loopPromise = runInRootContext(() => this.runLoop());
    // 静默 unhandled rejection
    this.loopPromise.catch(() => { /* logged inside */ });
  }

  async stop(): Promise<void> {
    this.closed = true;
    if (this.loopPromise) {
      try {
        await this.loopPromise;
      } catch { /* swallow */ }
    }
  }

  /**
   * 单次消费一个 agent。测试专用（同步跑通）。返回处理结果，方便断言。
   */
  async runOnce(): Promise<{
    agent?: AgentTuple;
    processedTaskIds: string[];
    lockContended?: boolean;
    dropped?: string[]; // 幽灵 / 抽取失败被丢弃的 task
  }> {
    const agent = await this.opts.queue.dequeueAgent(this.opts.brpopBlockMs ?? 5000);
    if (!agent) return { processedTaskIds: [] };
    return this.consumeAgent(agent);
  }

  private async runLoop(): Promise<void> {
    const blockMs = this.opts.brpopBlockMs ?? 5000;
    while (!this.closed) {
      let agent: AgentTuple | null = null;
      try {
        agent = await this.opts.queue.dequeueAgent(blockMs);
      } catch (err) {
        if (this.closed) break;
        this.logger.warn(`[skill-conv-worker] dequeue error: ${(err as Error).message}`);
        await sleep(200);
        continue;
      }
      if (!agent) continue;
      try {
        await this.consumeAgent(agent);
      } catch (err) {
        this.logger.error(`[skill-conv-worker] consumeAgent error: ${(err as Error).message}`);
      }
    }
  }

  private async consumeAgent(agent: AgentTuple): Promise<{
    agent: AgentTuple;
    processedTaskIds: string[];
    lockContended?: boolean;
    dropped?: string[];
  }> {
    const q = this.opts.queue;
    const extractLockTtl = this.opts.extractLockTtlMs ?? 600_000;
    const mutexOpts = {
      lockTtlMs: this.opts.tasksMutexLockTtlMs ?? 10_000,
      waitDeadlineMs: this.opts.tasksMutexWaitDeadlineMs ?? 30_000,
    };
    const perRound = this.opts.tasksPerRound ?? 1;
    const processedTaskIds: string[] = [];
    const dropped: string[] = [];

    const agentKey = `${agent.space_id}|${agent.user_id}|${agent.team_id}|${agent.agent_id}`;
    // [obs] worker 段结构化事件：consume_start → acquire_lock → read_head →
    //   read_archive → extractor → apply_candidates → delete_task → consume_done。
    // 拿到 task_id 之前用 agent_key + worker_id 定位；拿到 task_id 之后每条事件
    // 都带 task_id —— handler 侧 skill.trigger.enqueue_agent 事件已带同一 task_id，
    // 后端按 task_id 就能拉全 handler + worker 双段。
    // obsLogger 内部 try/catch + 后端降级，不需要额外防御。
    const workerId = this.opts.workerId;
    const t0Consume = Date.now();
    obsLogger.info("skill.worker.consume_start", {
      worker_id: workerId, agent_key: agentKey,
    });
    this.logger.info(`[skill-conv-worker] dequeued agent=${agentKey}`);

    // ② 抢 extract-lock
    const t0Lock = Date.now();
    const handle = await q.acquireExtractLock(agent, extractLockTtl);
    obsLogger.info("skill.worker.acquire_lock", {
      worker_id: workerId, agent_key: agentKey,
      dur_ms: Date.now() - t0Lock, acquired: !!handle,
    });
    if (!handle) {
      this.logger.info(`[skill-conv-worker] extract-lock contended agent=${agentKey}, requeue+sleep`);
      await q.requeueAgent(agent);
      const jitter = Math.floor(Math.random() * (this.opts.lockContentionSleepJitterMs ?? 500));
      await sleep((this.opts.lockContentionSleepMs ?? 2000) + jitter);
      obsLogger.info("skill.worker.consume_done", {
        worker_id: workerId, agent_key: agentKey,
        outcome: "lock_contended", dur_ms: Date.now() - t0Consume,
      });
      return { agent, processedTaskIds: [], lockContended: true };
    }
    this.logger.info(`[skill-conv-worker] acquired extract-lock agent=${agentKey}`);

    // ②.5 启动续约定时器 —— 保证 LLM 长跑时锁不掉
    const renewInterval = this.opts.extractLockRenewIntervalMs ?? Math.floor(extractLockTtl / 4);
    let renewTimer: ReturnType<typeof setInterval> | undefined;
    if (renewInterval > 0) {
      renewTimer = setInterval(() => {
        void (async () => {
          try {
            const ok = await q.renewExtractLock(handle, extractLockTtl);
            if (!ok) {
              this.logger.warn(
                `[skill-conv-worker] renew extract-lock lost agent=${agentKey}`,
              );
              if (renewTimer) clearInterval(renewTimer);
              renewTimer = undefined;
            }
          } catch (err) {
            this.logger.warn(
              `[skill-conv-worker] renew extract-lock error: ${(err as Error).message}`,
            );
          }
        })();
      }, renewInterval);
    }

    try {
      for (let round = 0; round < perRound; round++) {
        // ③ 抢 mutex 读队首 task。
        //
        // 关键修复（幽灵任务 root cause）：判空之后的 removeAgent 必须在
        // 同一个 tasks-mutex 临界区内完成，不能等 mutex 释放后再调——否则
        // TriggerService.archive() 可能在这个无锁窗口里抢到 mutex 写入新
        // task，但此时 Redis Set 里 agent 还没被摘除（本函数的 removeAgent
        // 还没跑），enqueueAgent 的 SADD 会因为「Set 已存在」返回 0 而跳过
        // LPUSH——新 task 落地了但 Redis 队列毫无记录，永久卡死成幽灵任务。
        // 把 removeAgent 收进同一把锁，跟 trigger-service.ts 的 fix
        // （enqueueAgent 挪进写 task 的临界区）配合，保证两侧互斥。
        const t0Head = Date.now();
        const head = await q.withTasksMutex(agent, mutexOpts, async () => {
          const doc = await this.opts.buffer.readTasks(agent);
          const first = doc.tasks[0] ?? null;
          if (!first) {
            await q.removeAgent(agent);
          }
          return first;
        });
        obsLogger.info("skill.worker.read_head", {
          worker_id: workerId, agent_key: agentKey,
          dur_ms: Date.now() - t0Head, has_head: !!head,
        });

        if (!head) {
          // tasks 空 → agent 已在上面的临界区内下线，跳出
          obsLogger.info("skill.worker.consume_done", {
            worker_id: workerId, agent_key: agentKey,
            outcome: "empty", dur_ms: Date.now() - t0Consume,
          });
          return { agent, processedTaskIds, dropped };
        }

        // task_id 就位：从此往下所有事件都带 task_id，跟 handler 侧同 task_id
        // 关联（handler 的 skill.trigger.enqueue_agent 已经带过一次）。
        const t0Task = Date.now();
        obsLogger.info("skill.worker.task_start", {
          worker_id: workerId,
          task_id: head.task_id,
          task_ref_id: head.task_ref_id,
          session_id: head.session_id,
          team_id: head.team_id,
          agent_id: head.agent_id,
          archive_key: head.archive_key,
        });

        // ④ 读 archive
        let candidates: ExtractedCandidate[] | null = null;
        let isGhost = false;
        try {
          this.logger.info(
            `[skill-conv-worker] processing task_id=${head.task_id} archive_key=${head.archive_key}`,
          );
          const t0Arch = Date.now();
          const archive = await this.opts.buffer.readArchive(head.archive_key);
          obsLogger.info("skill.worker.read_archive", {
            worker_id: workerId, task_id: head.task_id,
            dur_ms: Date.now() - t0Arch,
            found: !!archive,
            msg_count: archive?.messages?.length ?? 0,
          });
          if (!archive) {
            isGhost = true;
            this.logger.warn(
              `[skill-conv-worker] ghost task, dropping task_id=${head.task_id} archive_key=${head.archive_key}`,
            );
            obsLogger.warn("skill.worker.ghost", {
              worker_id: workerId, task_id: head.task_id,
              archive_key: head.archive_key,
            });
          } else {
            const conversation = (archive.messages ?? []).map((m) => ({
              role: String(m.role ?? "user"),
              content: String(m.content ?? ""),
            }));
            this.logger.info(
              `[skill-conv-worker] extract start task_id=${head.task_id} messages=${conversation.length}`,
            );
            const t0Ext = Date.now();
            const result = await this.opts.extractor.extract({
              task_id: head.task_id,
              team_id: head.team_id,
              user_id: head.user_id,
              agent_id: head.agent_id,
              // Langfuse trace 绑定字段：不透传的话 skill.extract 的 trace
              // sessionId=null / instanceId=unknown，页面按 session 过滤就找不到。
              session_id: head.session_id,
              space_id: head.space_id,
              conversation,
              // direct-trigger (`/v3/skill/extract`) 独占字段透传；conversation/add 归档
              // 的 task 不带这两个字段, undefined 不影响 extractor (走默认)。
              reason: head.reason,
              options: head.max_iterations != null
                ? { max_iterations: head.max_iterations }
                : undefined,
            });
            candidates = result.candidates ?? [];
            obsLogger.info("skill.worker.extractor", {
              worker_id: workerId, task_id: head.task_id,
              dur_ms: Date.now() - t0Ext,
              candidates: candidates.length,
              msg_count: conversation.length,
            });
            this.logger.info(
              `[skill-conv-worker] extract done task_id=${head.task_id} candidates=${candidates.length}`,
            );
            const t0Sink = Date.now();
            await this.opts.sink.applyCandidates({
              task: head,
              candidates,
              workerId: this.opts.workerId,
            });
            obsLogger.info("skill.worker.apply_candidates", {
              worker_id: workerId, task_id: head.task_id,
              dur_ms: Date.now() - t0Sink,
              candidates: candidates.length,
            });
          }
        } catch (err) {
          // 失败分级（对齐设计文档 §3.6 (7) P0 修复）：
          //   transient → sleep + requeue，retry_count 不变
          //   permanent → sleep + retry_count++ 回写；达阈值移进 DLQ
          //   分类兜底 → 按 transient 处理（保守不丢数据）
          const errMsg = (err as Error).message ?? String(err);
          const category = classifyError(err as Error);
          if (category === "transient") {
            this.logTransientFailure(head.task_id, errMsg);
            // 注意：outcome 用 `retry_transient` 简写，不含 "transient" 关键字 ——
            // DLQ 单测用 .includes("transient") 判定 transient 采样计数，
            // 避免 obsLogger 事件也被算进去。
            obsLogger.info("skill.worker.task_done", {
              worker_id: workerId, task_id: head.task_id,
              outcome: "retry_transient", dur_ms: Date.now() - t0Task,
            });
            await sleep(this.opts.failureRequeueSleepMs ?? 2000);
            await q.requeueAgent(agent);
            break;
          }
          // permanent
          // 清掉 transient 计数器，避免历史 transient 干扰后续采样。
          this.transientFailStreak.delete(head.task_id);
          obsLogger.info("skill.worker.task_done", {
            worker_id: workerId, task_id: head.task_id,
            outcome: "dlq_or_retry", dur_ms: Date.now() - t0Task,
          });
          await sleep(this.opts.failureRequeueSleepMs ?? 2000);
          await this.handlePermanentFailure(agent, head, errMsg, mutexOpts);
          break;
        }

        // ⑤ 抽取成功 or 幽灵 task → CAS filter 删 task（按 task_id）。
        //
        // 关键修复：删 task 之后紧接着判断「本 agent 是否还有剩余任务 →
        // requeue 还是 remove」必须合并进同一个 tasks-mutex 临界区（原来是
        // 两次独立、其中第二次完全无锁的 readTasks），否则同样会跟
        // trigger-service.archive() 产生竞态：本函数在无锁间隙判定"空了"
        // 要 removeAgent，但 archive() 恰好在这之前抢到锁写入了新 task 且
        // 已经 enqueueAgent —— 之后本函数再 removeAgent 把 Set/List 清空，
        // 新写入的 task 就变成了 Redis 队列里彻底找不到记录的幽灵任务。
        const t0Del = Date.now();
        let remainingTasks = 0;
        await q.withTasksMutex(agent, mutexOpts, async () => {
          const doc = await this.opts.buffer.readTasks(agent);
          const before = doc.tasks.length;
          doc.tasks = doc.tasks.filter((t) => t.task_id !== head.task_id);
          if (doc.tasks.length !== before) {
            doc.updated_at_ms = this.opts.now?.() ?? Date.now();
            await this.opts.buffer.writeTasks(agent, doc);
          }
          remainingTasks = doc.tasks.length;
          if (doc.tasks.length > 0) {
            await q.requeueAgent(agent);
          } else {
            await q.removeAgent(agent);
          }
        });
        obsLogger.info("skill.worker.delete_task", {
          worker_id: workerId, task_id: head.task_id,
          dur_ms: Date.now() - t0Del,
          remaining: remainingTasks,
        });

        // trace.report 后端 span：跟 skill.extract / skill.conversation_add 对齐，
        // 按 task_id 就能在 clickhouse / langfuse 里拉到 handler + worker 双段。
        try {
          trace.report("skill.worker.task_done", {
            task_id: head.task_id,
            task_ref_id: head.task_ref_id,
            team_id: head.team_id,
            agent_id: head.agent_id,
            session_id: head.session_id,
            outcome: isGhost ? "ghost" : "ok",
            candidates: candidates?.length ?? 0,
            dur_ms: Date.now() - t0Task,
            success: true,
          });
        } catch { /* noop */ }

        obsLogger.info("skill.worker.task_done", {
          worker_id: workerId, task_id: head.task_id,
          outcome: isGhost ? "ghost" : "ok",
          candidates: candidates?.length ?? 0,
          dur_ms: Date.now() - t0Task,
        });

        if (isGhost) dropped.push(head.task_id);
        else processedTaskIds.push(head.task_id);
      }

      obsLogger.info("skill.worker.consume_done", {
        worker_id: workerId, agent_key: agentKey,
        outcome: "ok",
        processed: processedTaskIds.length,
        dropped: dropped.length,
        dur_ms: Date.now() - t0Consume,
      });
      return { agent, processedTaskIds, dropped };
    } finally {
      // ⑥ 停续约 + 释放 extract-lock
      if (renewTimer) {
        clearInterval(renewTimer);
        renewTimer = undefined;
      }
      try {
        await q.releaseExtractLock(handle);
      } catch (err) {
        this.logger.warn(
          `[skill-conv-worker] releaseExtractLock error: ${(err as Error).message}`,
        );
      }
    }
  }

  /**
   * transient 失败日志采样：首次以 error 级别打印，之后每 N 次以 warn 级
   * 打一条摘要，防日志刷屏。N 由 `transientLogSampleEvery` 控制（默认 60）。
   */
  private logTransientFailure(taskId: string, errMsg: string): void {
    const prev = this.transientFailStreak.get(taskId) ?? 0;
    const streak = prev + 1;
    this.transientFailStreak.set(taskId, streak);
    const every = this.opts.transientLogSampleEvery ?? 60;
    if (streak === 1) {
      this.logger.error(
        `[skill-conv-worker] transient extract failure task=${taskId}: ${errMsg}`,
      );
    } else if (every > 0 && streak % every === 0) {
      this.logger.warn(
        `[skill-conv-worker] transient extract failure (x${streak}) task=${taskId}: ${errMsg}`,
      );
    }
  }

  /**
   * permanent 失败：抢 tasks-mutex 读改写 `_tasks.json`。
   *
   *   - retry_count+1 < permanentMaxRetries：写回 task 条目、requeue agent
   *   - retry_count+1 >= permanentMaxRetries：从 `_tasks.json` 移除该 task、
   *     追加到 `_tasks_dlq.json`，剩余 task 决定 requeue / remove
   *
   * 关键：读改写 `_tasks.json` 必须在同一临界区里，跟成功路径同 pattern（避免
   * 和 trigger-service.archive() 竞态）。DLQ 写不占 mutex：Worker 已持
   * extract-lock，同一 agent 只有一个写者。
   */
  private async handlePermanentFailure(
    agent: AgentTuple,
    head: SkillTaskEntry,
    errMsg: string,
    mutexOpts: { lockTtlMs: number; waitDeadlineMs: number },
  ): Promise<void> {
    const q = this.opts.queue;
    const maxRetries = this.opts.permanentMaxRetries ?? 3;
    const truncated = errMsg.length > 1024 ? errMsg.slice(0, 1024) : errMsg;
    const nowMs = () => this.opts.now?.() ?? Date.now();

    let deadTask: SkillTaskEntry | null = null;

    await q.withTasksMutex(agent, mutexOpts, async () => {
      const doc = await this.opts.buffer.readTasks(agent);
      const idx = doc.tasks.findIndex((t) => t.task_id === head.task_id);
      if (idx < 0) {
        // task 已被别处清掉（幽灵回收 / DLQ 重跑），本次 permanent 视为无效。
        this.logger.warn(
          `[skill-conv-worker] permanent failure but task gone task=${head.task_id}`,
        );
        if (doc.tasks.length > 0) await q.requeueAgent(agent);
        else await q.removeAgent(agent);
        return;
      }
      const cur = doc.tasks[idx]!;
      const nextRetry = (cur.retry_count ?? 0) + 1;
      if (nextRetry >= maxRetries) {
        // → DLQ：从 _tasks.json 摘除
        deadTask = { ...cur, retry_count: nextRetry, last_error: truncated };
        doc.tasks.splice(idx, 1);
        doc.updated_at_ms = nowMs();
        await this.opts.buffer.writeTasks(agent, doc);
        this.logger.error(
          `[skill-conv-worker] permanent failure → DLQ task=${head.task_id} ` +
            `retries=${nextRetry}/${maxRetries} err=${truncated}`,
        );
      } else {
        doc.tasks[idx] = { ...cur, retry_count: nextRetry, last_error: truncated };
        doc.updated_at_ms = nowMs();
        await this.opts.buffer.writeTasks(agent, doc);
        this.logger.warn(
          `[skill-conv-worker] permanent failure task=${head.task_id} ` +
            `retries=${nextRetry}/${maxRetries} err=${truncated}`,
        );
      }
      if (doc.tasks.length > 0) await q.requeueAgent(agent);
      else await q.removeAgent(agent);
    });

    // 写 DLQ 放在 mutex 外：Worker 持 extract-lock 独占该 agent，DLQ 没有别的写者。
    if (deadTask) {
      const dead: SkillDeadTaskEntry = {
        ...(deadTask as SkillTaskEntry),
        dead_lettered_at_ms: nowMs(),
      };
      try {
        await this.opts.buffer.appendDlq(agent, dead);
      } catch (err) {
        this.logger.error(
          `[skill-conv-worker] appendDlq failed task=${head.task_id}: ${(err as Error).message}`,
        );
      }
    }
  }
}

/**
 * 把 extract/sink 抛出的 Error 分成 transient (会自愈) 或 permanent (数据/schema)
 * 两类。规则简单，按错误消息里的 HTTP 状态码 + 关键字匹配；识别不出的按 transient
 * 兜底 —— 保守，不丢数据。
 *
 * 完整分类矩阵见 docs/design/2026-07-21-memorycore-standalone-e2e.md §3.6 (7)。
 */
export function classifyError(err: Error): "transient" | "permanent" {
  const raw = `${err?.name ?? ""} ${err?.message ?? ""}`;
  const msg = raw.toLowerCase();
  // AbortError（LLM 请求被 signal cancel / timeout）视为 transient
  if ((err?.name ?? "") === "AbortError") return "transient";

  // ── permanent 优先匹配：显式的 4xx 数据/schema 错 ──
  // HTTP 400 / 422
  if (/(^|[^\d])(400|422)([^\d]|$)/.test(msg)) return "permanent";
  // JSON 解析错 / schema 校验错 / "invalid response" 类
  if (
    msg.includes("json.parse") ||
    msg.includes("unexpected token") ||
    msg.includes("invalid json") ||
    msg.includes("invalid response") ||
    msg.includes("schema") ||
    msg.includes("zod") ||
    msg.includes("validation failed")
  ) {
    return "permanent";
  }

  // ── transient 识别 ──
  // HTTP 401/403/429/5xx
  if (/(^|[^\d])(401|403|429|5\d{2})([^\d]|$)/.test(msg)) return "transient";
  // 网络错误常见错误码 / fetch 层
  if (
    msg.includes("econnrefused") ||
    msg.includes("etimedout") ||
    msg.includes("enotfound") ||
    msg.includes("eai_again") ||
    msg.includes("socket hang up") ||
    msg.includes("client network socket disconnected") ||
    msg.includes("fetch failed") ||
    msg.includes("und_err_") ||
    msg.includes("timeout") ||
    msg.includes("timed out")
  ) {
    return "transient";
  }

  // 兜底：识别不出的按 transient 处理（不丢数据）
  return "transient";
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
