/**
 * SkillTriggerService — 归档段（§7.4）。
 *
 * Handler 主流程判定"需要归档"后调用本模块，按以下顺序完成"**先落 archive
 * 再登记 task**"：
 *   ① 生成 archive_key / task_id / archived_at_ms
 *   ② 写 archive 文件（已存在 = 视为成功；这一步慢或失败都不会有 orphan task）
 *   ③ 抢 tasks-mutex → 读 `_tasks.json` → 追加 task → 写回 → 在同一临界区内 Redis 入队
 *
 * 顺序变更历史：
 *   2026-07-20 —— 原顺序是 ①→②(tasksMutex 内写 _tasks.json + 入队)→③(写 archive)，
 *   曾在测试环境触发：writeArchive 慢于 worker BRPOP（因 CoS 自签名 TLS + 内网
 *   DNS 抖动 单次 put ~10s），worker 抢先 readArchive 得 null 判 ghost 静默丢
 *   真任务。改成"先 archive 再 task"后：archive 慢/失败都在 handler 端原地
 *   报错，worker 永远看不到没有 archive 的 task，happy-path 下不会误判 ghost。
 *
 * 新顺序下的失败态：
 *   - writeArchive 抛错 → handler 拿到异常，_tasks.json 无残留、Redis 无 agent；
 *     Client 重试完整走一遍即可（archive 幂等：同 archived_at_ms 会 skip 不覆盖）。
 *   - writeArchive 成功 → mutex 内 writeTasks / enqueueAgent 任一失败 → 只留下
 *     "孤儿 archive"（一个 4-数十 KB 的 jsonl，worker 看不到，靠 CoS 生命周期
 *     或后续 GC 清）。这比现状好得多：不再有丢任务风险。
 */

import { randomUUID } from "node:crypto";

import type {
  AgentTuple,
  ISkillAgentTaskQueue,
} from "./agent-task-queue.js";
import type {
  BufferedMessages,
  SessionKey,
  SkillBufferStorage,
  SkillTaskEntry,
} from "./buffer-storage.js";
import { obsLogger } from "../../report/obs-logger.js";

export interface TriggerArchiveInput {
  session: SessionKey;
  bufferAtTrigger: BufferedMessages;
  /** 业务侧 task 引用（透传字段），可选。 */
  taskRefId?: string;
  /**
   * direct-trigger (`/v3/skill/extract`) 独占：主 Agent 的抽取提示，
   * 落到 SkillTaskEntry.reason，Worker 消费时透传给 extractor.extract。
   * conversation/add 路径不传。
   */
  reason?: string;
  /**
   * direct-trigger 独占：extractor LLM 迭代上限，落到 SkillTaskEntry.max_iterations。
   * conversation/add 路径不传（走 extractor 侧默认）。
   */
  maxIterations?: number;
  /**
   * 上游 handler 生成的 request_id。传入后 trigger 内部分段 obsLogger 事件
   * 会带上，方便按 req_id 过滤 handler + trigger + worker 全链路。缺省不影响功能。
   */
  perfRequestId?: string;
}

export interface TriggerArchiveResult {
  taskId: string;
  archivedAtMs: number;
  archiveKey: string;
}

export interface SkillTriggerServiceOptions {
  buffer: SkillBufferStorage;
  queue: ISkillAgentTaskQueue;
  /**
   * tasks-mutex 锁 TTL，默认 10000ms。**锁自身的过期时间**——只是为了兜底
   * 持锁进程崩溃后能自动释放。真正的临界区应该远小于这个值（COS 读改写通常 <300ms）。
   */
  tasksMutexLockTtlMs?: number;
  /**
   * tasks-mutex 争抢的最长等待时间，默认 30000ms（30s）。
   * 高并发同 agent 归档时（多 session 同时进临界区）排队时间会累积，
   * waitDeadline 必须 >> 单次临界区时长 × 期望排队数。
   */
  tasksMutexWaitDeadlineMs?: number;
  /** 时间源，用于测试注入。 */
  now?: () => number;
}

export class SkillTriggerService {
  private readonly buffer: SkillBufferStorage;
  private readonly queue: ISkillAgentTaskQueue;
  private readonly tasksMutexLockTtlMs: number;
  private readonly tasksMutexWaitDeadlineMs: number;
  private readonly now: () => number;

  constructor(opts: SkillTriggerServiceOptions) {
    this.buffer = opts.buffer;
    this.queue = opts.queue;
    this.tasksMutexLockTtlMs = opts.tasksMutexLockTtlMs ?? 10_000;
    this.tasksMutexWaitDeadlineMs = opts.tasksMutexWaitDeadlineMs ?? 30_000;
    this.now = opts.now ?? (() => Date.now());
  }

  /**
   * 触发一次归档。执行 §7.4 ①→②→③→④。
   * 失败会抛异常，让 Handler 转换成 500 交给 Client 重试。
   */
  async archive(input: TriggerArchiveInput): Promise<TriggerArchiveResult> {
    const { session, bufferAtTrigger, taskRefId } = input;

    // ① 生成标识
    const archivedAtMs = this.now();
    const archiveKey = this.buffer.archiveKey(session, archivedAtMs);
    // 前缀 `skill-extract-task-` 是内部 anchor（跟业务侧 task_id 明确区分）。
    // 一次归档 = 一个 SkillTaskEntry.task_id；handler 侧 `[skill-perf] phase=trigger.enqueueAgent
    // task_id=…` 与 worker 侧 `[skill-perf] kind=worker phase=consume.*` 共用同一
    // 值，grep 一次拉全 handler + worker 双段耗时。老数据前缀 `task-` 会被
    // worker 自然消费掉，无迁移风险（filter 按 task_id 值等价比较，不解析前缀）。
    const taskId = `skill-extract-task-${randomUUID().slice(0, 8)}`;
    const agent: AgentTuple = {
      space_id: session.space_id,
      user_id: session.user_id,
      team_id: session.team_id,
      agent_id: session.agent_id,
    };

    const entry: SkillTaskEntry = {
      task_id: taskId,
      session_id: session.session_id,
      user_id: session.user_id,
      team_id: session.team_id,
      agent_id: session.agent_id,
      space_id: session.space_id,
      task_ref_id: taskRefId,
      archive_key: archiveKey,
      archived_at_ms: archivedAtMs,
      enqueued_at_ms: archivedAtMs,
      reason: input.reason,
      max_iterations: input.maxIterations,
    };

    // [obs] 归档段五个关键 IO：writeArchive / mutex acquire / readTasks /
    // writeTasks / enqueueAgent。历史事故里就是 writeArchive 慢 10s 拖崩了
    // worker 侧的判空逻辑；这里逐段结构化事件，字段按 req_id / task_id 关联。
    // obsLogger 内部 try/catch + 后端降级，不需要额外防御。
    const rid = input.perfRequestId;

    // ② 先写 archive（已存在视为成功）
    //
    // 顺序在这里的 rationale：worker 侧 readArchive 得 null 会判 ghost 直接删
    // 对应 task；若 archive 段晚于 task 段，任何 CoS 抖动（自签名 TLS 慢握手、
    // 内网 DNS 重解析等）都会撑出一个"task 登记完但 archive 还没落"的窗口，
    // worker BRPOP 抢在这个窗口里就会把真任务当 ghost 丢掉（2026-07-20 生产
    // 现场：writeArchive 耗 10s，两台 core 分工，另一台在 5s 时就把 task
    // dropped 了）。改成先 archive 再 task 后：writeArchive 抛错/慢都不会
    // 造成 orphan task；worker 永远只在 archive 已经存在时才看到 task。
    //
    // 失败态：writeArchive 抛错 → 直接向 handler 抛异常，无残留。
    const t0Arch = Date.now();
    await this.buffer.writeArchive(session, archivedAtMs, bufferAtTrigger);
    obsLogger.info("skill.trigger.write_archive", {
      req_id: rid, task_id: taskId,
      dur_ms: Date.now() - t0Arch,
      archive_key: archiveKey,
    });

    // ③ 抢 mutex → CAS 追加 task 到队尾 → 在同一把锁内入队
    //
    // 关键保证（幽灵任务 root cause）：writeTasks 与 enqueueAgent 必须在同一个
    // tasks-mutex 临界区内完成，否则会跟 Worker 侧「判空→removeAgent」之间
    // 出现竞态窗口：Worker 在 mutex 外调用 removeAgent 之前，本函数的
    // enqueueAgent 可能已经因为 Set 里还残留旧 agent（SADD 返回 0）而跳过
    // LPUSH，导致 _tasks.json 里落地了新 task，但 Redis 队列对该 agent
    // 毫无记录 —— 任务永久卡死，且不会有任何后续机制重新触发。
    // 把 enqueueAgent 收进同一把锁，与 Worker 侧的 fix（见 extract-worker.ts）
    // 配合，保证「写任务+入队」与「判空+出队」互斥、不会交叉。
    const t0MutexEntry = Date.now();
    await this.queue.withTasksMutex(
      agent,
      { lockTtlMs: this.tasksMutexLockTtlMs, waitDeadlineMs: this.tasksMutexWaitDeadlineMs },
      async () => {
        const mutexAcquiredAt = Date.now();
        obsLogger.info("skill.trigger.mutex_acquire", {
          req_id: rid, task_id: taskId,
          dur_ms: mutexAcquiredAt - t0MutexEntry,
        });

        const t0Read = Date.now();
        const doc = await this.buffer.readTasks(agent);
        obsLogger.info("skill.trigger.read_tasks", {
          req_id: rid, task_id: taskId,
          dur_ms: Date.now() - t0Read,
          existing_tasks: doc.tasks.length,
        });

        doc.tasks.push(entry);
        doc.updated_at_ms = archivedAtMs;

        const t0Write = Date.now();
        await this.buffer.writeTasks(agent, doc);
        obsLogger.info("skill.trigger.write_tasks", {
          req_id: rid, task_id: taskId,
          dur_ms: Date.now() - t0Write,
          total_tasks: doc.tasks.length,
        });

        const t0Enq = Date.now();
        const enqueued = await this.queue.enqueueAgent(agent);
        obsLogger.info("skill.trigger.enqueue_agent", {
          req_id: rid, task_id: taskId,
          dur_ms: Date.now() - t0Enq,
          added: enqueued,
        });
      },
    );
    obsLogger.info("skill.trigger.mutex_total", {
      req_id: rid, task_id: taskId,
      dur_ms: Date.now() - t0MutexEntry,
    });

    return { taskId, archivedAtMs, archiveKey };
  }
}
