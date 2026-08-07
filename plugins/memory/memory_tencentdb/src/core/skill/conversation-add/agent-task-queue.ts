/**
 * SkillAgentTaskQueue — agent 级调度信号 + `_tasks.json` 短锁保护。
 *
 * 对应设计文档 §5、§9。
 *
 * 抽象接口 `ISkillAgentTaskQueue` 定义三组能力：
 *   1) agent 队列 (List + Set)：enqueueAgent / dequeueAgent / requeueAgent / removeAgent
 *   2) tasks-mutex（保护 `_tasks.json` 读改写，TTL 秒级）：withTasksMutex
 *   3) extract-lock（Worker 独占 agent 抽取权，TTL 10 min）：acquire/renew/releaseExtractLock
 *
 * 生产实现走 Redis（`RedisSkillAgentTaskQueue`，本文件），
 * 测试实现走内存（`LocalSkillAgentTaskQueue`，本文件）。
 *
 * agent tuple 序列化格式：
 *   `{space}|{user}|{team}|{agent}` — 与 Redis List 元素、锁 key 后缀完全一致。
 */

import { randomUUID } from "node:crypto";

// ── Common types ──────────────────────────────────────────────────────────

export interface AgentTuple {
  space_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
}

export function serializeAgentTuple(a: AgentTuple): string {
  return `${a.space_id}|${a.user_id}|${a.team_id}|${a.agent_id}`;
}

export function parseAgentTuple(raw: string): AgentTuple | null {
  const parts = raw.split("|");
  if (parts.length !== 4) return null;
  const [space_id, user_id, team_id, agent_id] = parts;
  if (!space_id || !user_id || !team_id || !agent_id) return null;
  return { space_id, user_id, team_id, agent_id };
}

export interface ExtractLockHandle {
  key: string;      // agent tuple 序列化后的字符串
  token: string;    // 释放/续约凭证
}

export interface ISkillAgentTaskQueue {
  // ── Agent queue ──
  /**
   * 幂等入队。Set 已含则不重复 LPUSH。返回本次是否是"新入队"（首次进入 Set 返回 true）。
   */
  enqueueAgent(tuple: AgentTuple): Promise<boolean>;
  /**
   * BRPOP-style 出队。阻塞直到有元素或超时；超时返回 null。
   * 注意：出队仅从 List 弹出，Set 不删除（Worker 会在处理完后决定 requeue 还是 remove）。
   */
  dequeueAgent(blockMs: number): Promise<AgentTuple | null>;
  /**
   * 处理完 tasks 仍非空时调用，塞回队头（等价排到队尾轮转）。Set 保持。
   */
  requeueAgent(tuple: AgentTuple): Promise<void>;
  /**
   * tasks 空时下线该 agent：SREM Set；如果 List 里还有残留则一并清理。
   */
  removeAgent(tuple: AgentTuple): Promise<void>;

  // ── tasks-mutex（短锁保护 `_tasks.json` 读改写） ──
  /**
   * 争抢 tasks-mutex 执行 fn，失败时**退避重试直到 waitDeadlineMs**，锁本身有 lockTtlMs 兜底防死锁。
   *
   * 两个参数分离的原因（旧实现把 deadline 和 ttl 用同一值 → 30 并发同 agent 全 timeout 500）：
   *   - `lockTtlMs`：锁自身在 Redis 上的过期时间；用于持锁进程崩溃时兜底自动释放（几秒足够）。
   *   - `waitDeadlineMs`：调用方最多愿意阻塞多久排队；应该 >> lockTtlMs，
   *     覆盖 N 个并发排队所需的累计临界区时长（例如 30 个 300ms 临界区 = 9s，
   *     waitDeadline 至少 15-30s 才不误报 timeout）。
   */
  withTasksMutex<T>(
    tuple: AgentTuple,
    opts: { lockTtlMs: number; waitDeadlineMs: number },
    fn: () => Promise<T>,
  ): Promise<T>;

  // ── extract-lock（Worker 独占 agent 抽取权） ──
  acquireExtractLock(tuple: AgentTuple, ttlMs: number): Promise<ExtractLockHandle | null>;
  renewExtractLock(handle: ExtractLockHandle, ttlMs: number): Promise<boolean>;
  releaseExtractLock(handle: ExtractLockHandle): Promise<void>;
}

// ────────────────────────────────────────────────────────────────────────────
// Local (in-memory) implementation — 单元测试专用
// ────────────────────────────────────────────────────────────────────────────

interface WaitingConsumer {
  resolve: (t: AgentTuple | null) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class LocalSkillAgentTaskQueue implements ISkillAgentTaskQueue {
  private readonly list: string[] = [];      // 头 = LPUSH, 尾 = RPOP —— 匹配 Redis 语义
  private readonly set = new Set<string>();
  private readonly tasksMutex = new Map<string, { token: string; expireAt: number }>();
  private readonly extractLocks = new Map<string, { token: string; expireAt: number }>();
  private readonly waiters: WaitingConsumer[] = [];

  private notify(): void {
    while (this.waiters.length > 0 && this.list.length > 0) {
      const w = this.waiters.shift()!;
      clearTimeout(w.timer);
      const raw = this.list.pop()!; // RPOP
      const parsed = parseAgentTuple(raw);
      w.resolve(parsed);
    }
  }

  async enqueueAgent(tuple: AgentTuple): Promise<boolean> {
    const key = serializeAgentTuple(tuple);
    const added = !this.set.has(key);
    if (added) {
      this.set.add(key);
      this.list.unshift(key); // LPUSH
    }
    this.notify();
    return added;
  }

  async dequeueAgent(blockMs: number): Promise<AgentTuple | null> {
    if (this.list.length > 0) {
      const raw = this.list.pop()!; // RPOP
      return parseAgentTuple(raw);
    }
    if (blockMs <= 0) return null;
    return new Promise<AgentTuple | null>((resolve) => {
      const timer = setTimeout(() => {
        const idx = this.waiters.findIndex((w) => w.resolve === resolve);
        if (idx >= 0) this.waiters.splice(idx, 1);
        resolve(null);
      }, blockMs);
      this.waiters.push({ resolve, timer });
    });
  }

  async requeueAgent(tuple: AgentTuple): Promise<void> {
    const key = serializeAgentTuple(tuple);
    // 塞回队头（LPUSH 语义）；Set 保持
    this.set.add(key);
    this.list.unshift(key);
    this.notify();
  }

  async removeAgent(tuple: AgentTuple): Promise<void> {
    const key = serializeAgentTuple(tuple);
    this.set.delete(key);
    for (let i = this.list.length - 1; i >= 0; i--) {
      if (this.list[i] === key) this.list.splice(i, 1);
    }
  }

  // ── mutex ──

  async withTasksMutex<T>(
    tuple: AgentTuple,
    opts: { lockTtlMs: number; waitDeadlineMs: number },
    fn: () => Promise<T>,
  ): Promise<T> {
    const key = `mutex:${serializeAgentTuple(tuple)}`;
    const deadline = Date.now() + opts.waitDeadlineMs;
    while (true) {
      const now = Date.now();
      const cur = this.tasksMutex.get(key);
      if (!cur || cur.expireAt <= now) {
        const token = randomUUID();
        this.tasksMutex.set(key, { token, expireAt: now + opts.lockTtlMs });
        try {
          return await fn();
        } finally {
          const held = this.tasksMutex.get(key);
          if (held && held.token === token) this.tasksMutex.delete(key);
        }
      }
      if (Date.now() > deadline) {
        throw new Error(`[skill-agent-queue] tasks-mutex wait timeout for ${key}`);
      }
      await sleep(10);
    }
  }

  // ── extract lock ──

  async acquireExtractLock(tuple: AgentTuple, ttlMs: number): Promise<ExtractLockHandle | null> {
    const key = serializeAgentTuple(tuple);
    const now = Date.now();
    const cur = this.extractLocks.get(key);
    if (cur && cur.expireAt > now) return null;
    const token = randomUUID();
    this.extractLocks.set(key, { token, expireAt: now + ttlMs });
    return { key, token };
  }

  async renewExtractLock(handle: ExtractLockHandle, ttlMs: number): Promise<boolean> {
    const cur = this.extractLocks.get(handle.key);
    if (!cur || cur.token !== handle.token) return false;
    cur.expireAt = Date.now() + ttlMs;
    return true;
  }

  async releaseExtractLock(handle: ExtractLockHandle): Promise<void> {
    const cur = this.extractLocks.get(handle.key);
    if (!cur) return;
    if (cur.token !== handle.token) return;
    this.extractLocks.delete(handle.key);
  }

  // ── test helpers ──

  /** For tests: peek current state. */
  _snapshot(): { list: string[]; set: string[] } {
    return { list: [...this.list], set: [...this.set] };
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ────────────────────────────────────────────────────────────────────────────
// Redis implementation
// ────────────────────────────────────────────────────────────────────────────

/**
 * 最小 ioredis 客户端子集，避免直接 import Redis 类型（保持与现有 redis-queue-v2 一致）。
 */
export interface RedisLike {
  sadd(key: string, ...members: string[]): Promise<number>;
  srem(key: string, ...members: string[]): Promise<number>;
  lpush(key: string, ...values: string[]): Promise<number>;
  lrem(key: string, count: number, value: string): Promise<number>;
  brpop(key: string, timeoutSec: number): Promise<[string, string] | null>;
  rpop(key: string): Promise<string | null>;
  set(key: string, value: string, ...args: (string | number)[]): Promise<"OK" | null>;
  get(key: string): Promise<string | null>;
  del(...keys: string[]): Promise<number>;
  eval(script: string, numKeys: number, ...args: (string | number)[]): Promise<unknown>;
  pexpire(key: string, ms: number): Promise<number>;
}

export interface RedisSkillAgentTaskQueueOptions {
  client: RedisLike;
  /** Redis key 前缀，默认 "skill"。 */
  keyPrefix?: string;
  /**
   * dequeueAgent 轮询间隔 ms，默认 200。
   *
   * 2026-07-21 —— 历史上 dequeueAgent 走的是 `BRPOP <key> 5`,零延迟唤醒是
   * 拿到了,但 BRPOP 是**阻塞命令**,共用同一条 ioredis 连接的其他 skill
   * 命令 (`SET NX PX` 抢 tasks-mutex、`SADD`/`LPUSH` enqueueAgent、`EVAL`
   * 释放锁) 全部要排在 BRPOP 后面等,handler 每次归档撞 3 次 BRPOP 窗口
   * ≈ 15s。改成非阻塞 `RPOP` + `setTimeout(pollIntervalMs)` 轮询后,主
   * 连接不再被卡,handler 侧 Redis IO 恢复毫秒级；代价是最坏 pollInterval
   * 的唤醒延迟(默认 200ms,skill 抽取本身 LLM 十几秒,这点延迟无感)。
   *
   * 语义跟 memory 侧 `RedisStateBackend.consumeTask` 对齐
   * (redis-backend.ts:312 —— XREADGROUP 不带 BLOCK,外层 sleep 200ms)。
   */
  pollIntervalMs?: number;
}

/**
 * Redis key 前缀默认值。
 *
 * 生产建议：wire 层传入形如 `${memoryPrefix}:skill-conv` 的前缀，跟 memory 的
 * `keyPrefix`（例如 `tdai_memory_prod_v3`）挂钩，避免不同环境的 Redis
 * key 撞。默认值 `"skill-conv"` 只在没显式传时兜底。
 */
const DEFAULT_PREFIX = "skill-conv";

const LUA_RENEW = `
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
`;

const LUA_RELEASE = `
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
`;

export class RedisSkillAgentTaskQueue implements ISkillAgentTaskQueue {
  private readonly client: RedisLike;
  private readonly listKey: string;
  private readonly setKey: string;
  private readonly extractLockPrefix: string;
  private readonly tasksMutexPrefix: string;
  private readonly pollIntervalMs: number;

  constructor(opts: RedisSkillAgentTaskQueueOptions) {
    this.client = opts.client;
    this.pollIntervalMs = opts.pollIntervalMs ?? 200;
    const prefix = opts.keyPrefix ?? DEFAULT_PREFIX;
    // key 布局（对齐 §5 & §21.5）：
    //   {prefix}:pending-agents         — List (LPUSH 入队 / RPOP 轮询出队)
    //   {prefix}:pending-agents-set     — Set (SADD/SREM 幂等去重)
    //   {prefix}:extract-lock:{tuple}   — Worker 独占 agent 抽取权 (10min TTL)
    //   {prefix}:tasks-mutex:{tuple}    — 保护 _tasks.json 读改写 (5s TTL)
    this.listKey = `${prefix}:pending-agents`;
    this.setKey = `${prefix}:pending-agents-set`;
    this.extractLockPrefix = `${prefix}:extract-lock:`;
    this.tasksMutexPrefix = `${prefix}:tasks-mutex:`;
  }

  async enqueueAgent(tuple: AgentTuple): Promise<boolean> {
    const raw = serializeAgentTuple(tuple);
    const added = await this.client.sadd(this.setKey, raw);
    if (added === 1) {
      await this.client.lpush(this.listKey, raw);
      return true;
    }
    return false;
  }

  async dequeueAgent(blockMs: number): Promise<AgentTuple | null> {
    // 非阻塞 RPOP + sleep 轮询,拒绝 BRPOP。见 pollIntervalMs 处的注释。
    //
    // blockMs=0 —— 只试一次,list 空立刻返回 null。
    // blockMs>0 —— 每 pollIntervalMs 试一次,直到拿到元素或超过 deadline。
    const deadline = Date.now() + Math.max(0, blockMs);
    while (true) {
      const raw = await this.client.rpop(this.listKey);
      if (raw !== null) return parseAgentTuple(raw);
      const remaining = deadline - Date.now();
      if (remaining <= 0) return null;
      // pollInterval 与 remaining 取小,保证 blockMs 小于 pollInterval 时能按预期尽快返回。
      await sleep(Math.min(this.pollIntervalMs, remaining));
    }
  }

  async requeueAgent(tuple: AgentTuple): Promise<void> {
    const raw = serializeAgentTuple(tuple);
    // 幂等：保证 Set 里也在
    await this.client.sadd(this.setKey, raw);
    await this.client.lpush(this.listKey, raw);
  }

  async removeAgent(tuple: AgentTuple): Promise<void> {
    const raw = serializeAgentTuple(tuple);
    await this.client.srem(this.setKey, raw);
    // List 里可能还有残留（Worker 已经 pop 就没了；此处兜底）
    await this.client.lrem(this.listKey, 0, raw);
  }

  async withTasksMutex<T>(
    tuple: AgentTuple,
    opts: { lockTtlMs: number; waitDeadlineMs: number },
    fn: () => Promise<T>,
  ): Promise<T> {
    const key = this.tasksMutexPrefix + serializeAgentTuple(tuple);
    const token = randomUUID();
    const deadline = Date.now() + opts.waitDeadlineMs;
    while (true) {
      // SET NX PX：短 TTL 兜底崩溃场景；waitDeadline 覆盖并发排队时间
      const ok = await this.client.set(key, token, "NX", "PX", opts.lockTtlMs);
      if (ok === "OK") {
        try {
          return await fn();
        } finally {
          try {
            await this.client.eval(LUA_RELEASE, 1, key, token);
          } catch {
            /* swallow */
          }
        }
      }
      if (Date.now() > deadline) {
        throw new Error(`[skill-agent-queue] tasks-mutex wait timeout for ${key}`);
      }
      await sleep(20 + Math.floor(Math.random() * 30));
    }
  }

  async acquireExtractLock(tuple: AgentTuple, ttlMs: number): Promise<ExtractLockHandle | null> {
    const key = this.extractLockPrefix + serializeAgentTuple(tuple);
    const token = randomUUID();
    const ok = await this.client.set(key, token, "NX", "PX", ttlMs);
    if (ok !== "OK") return null;
    return { key, token };
  }

  async renewExtractLock(handle: ExtractLockHandle, ttlMs: number): Promise<boolean> {
    const raw = handle.key.startsWith(this.extractLockPrefix)
      ? handle.key
      : this.extractLockPrefix + handle.key;
    const result = await this.client.eval(LUA_RENEW, 1, raw, handle.token, ttlMs);
    return result === 1;
  }

  async releaseExtractLock(handle: ExtractLockHandle): Promise<void> {
    const raw = handle.key.startsWith(this.extractLockPrefix)
      ? handle.key
      : this.extractLockPrefix + handle.key;
    try {
      await this.client.eval(LUA_RELEASE, 1, raw, handle.token);
    } catch {
      /* swallow */
    }
  }
}
