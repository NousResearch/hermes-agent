/**
 * LocalStateBackend — 进程内 Pipeline 状态后端 (开源单机版)
 *
 * 需求 #7.3: 基于进程内 Map/setTimeout/SerialQueue/文件 Checkpoint，零外部依赖。
 * 将现有 MemoryPipelineManager 中的状态管理逻辑封装为 IStateBackend 实现。
 */

import type {
  IStateBackend,
  PipelineSessionState,
  TimerEntry,
  TaskPayload,
  CaptureAtomicParams,
  CaptureAtomicResult,
} from "./types.js";
import { DEFAULT_PIPELINE_STATE } from "./types.js";

interface InternalTimer {
  member: string;
  fireAtMs: number;
  handle?: ReturnType<typeof setTimeout>;
}

export class LocalStateBackend implements IStateBackend {
  private sessionStates = new Map<string, PipelineSessionState>();
  private buffers = new Map<string, string[]>();
  private timers = new Map<string, InternalTimer>();
  private taskQueue: TaskPayload[] = [];
  private locks = new Map<string, { ownerId: string; expireAt: number }>();
  private consumeWaiters: Array<{ resolve: (task: TaskPayload | null) => void; timer: ReturnType<typeof setTimeout> }> = [];
  private onTimerExpired?: (entry: TimerEntry) => void;
  private destroyed = false;

  constructor(options?: { onTimerExpired?: (entry: TimerEntry) => void }) {
    this.onTimerExpired = options?.onTimerExpired;
  }

  /**
   * Map key 拼成 `{instanceId}:{teamId}:{agentId}:{sessionId}` —— 与 RedisStateBackend 的
   * hash tag 形态对齐：同一 (inst, tid, aid, sess) 必然 hash 到同一桶。
   * tid/aid 缺失（旧调用）时用 "_" 占位，等价于退化到 instance 维度。
   */
  private k(instanceId: string, sessionId: string, teamId?: string, agentId?: string): string {
    return `${instanceId}:${teamId || "_"}:${agentId || "_"}:${sessionId}`;
  }

  // ═══ Buffer ═══

  async appendBuffer(instanceId: string, sessionId: string, message: string, teamId?: string, agentId?: string): Promise<void> {
    const key = this.k(instanceId, sessionId, teamId, agentId);
    let buf = this.buffers.get(key);
    if (!buf) { buf = []; this.buffers.set(key, buf); }
    buf.push(message);
  }

  async drainBuffer(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<string[]> {
    const key = this.k(instanceId, sessionId, teamId, agentId);
    const buf = this.buffers.get(key);
    if (!buf || buf.length === 0) return [];
    const drained = buf.splice(0);
    this.buffers.delete(key);
    return drained;
  }

  async getBufferLength(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<number> {
    return this.buffers.get(this.k(instanceId, sessionId, teamId, agentId))?.length ?? 0;
  }

  // ═══ Session State ═══

  async getSessionState(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<PipelineSessionState | null> {
    return this.sessionStates.get(this.k(instanceId, sessionId, teamId, agentId)) ?? null;
  }

  async updateSessionState(instanceId: string, sessionId: string, patch: Partial<PipelineSessionState>, teamId?: string, agentId?: string): Promise<void> {
    const key = this.k(instanceId, sessionId, teamId, agentId);
    const current = this.sessionStates.get(key) ?? { ...DEFAULT_PIPELINE_STATE, last_active_time: Date.now() };
    this.sessionStates.set(key, { ...current, ...patch });
  }

  async deleteSessionState(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<void> {
    const key = this.k(instanceId, sessionId, teamId, agentId);
    this.sessionStates.delete(key);
    this.buffers.delete(key);
  }

  async listActiveSessions(instanceId: string): Promise<string[]> {
    const prefix = `${instanceId}:`;
    const sessions: string[] = [];
    for (const key of this.sessionStates.keys()) {
      if (key.startsWith(prefix)) {
        // key format: {inst}:{tid}:{aid}:{sess}  → 取最后一段作为 sessionId
        const parts = key.split(":");
        if (parts.length >= 4) sessions.push(parts.slice(3).join(":"));
      }
    }
    return sessions;
  }

  // ═══ Timer ═══

  async setTimer(instanceId: string, member: string, fireAtMs: number): Promise<void> {
    const key = `${instanceId}:${member}`;
    const existing = this.timers.get(key);
    if (existing?.handle) clearTimeout(existing.handle);

    const delay = Math.max(0, fireAtMs - Date.now());
    const handle = this.onTimerExpired
      ? setTimeout(() => { this.timers.delete(key); this.onTimerExpired!({ member, fireAtMs }); }, delay)
      : undefined;
    if (handle) handle.unref();
    this.timers.set(key, { member, fireAtMs, handle });
  }

  async setTimerIfEarlier(instanceId: string, member: string, fireAtMs: number): Promise<boolean> {
    const existing = this.timers.get(`${instanceId}:${member}`);
    if (existing && fireAtMs >= existing.fireAtMs) return false;
    await this.setTimer(instanceId, member, fireAtMs);
    return true;
  }

  async removeTimer(instanceId: string, member: string): Promise<void> {
    const key = `${instanceId}:${member}`;
    const existing = this.timers.get(key);
    if (existing?.handle) clearTimeout(existing.handle);
    this.timers.delete(key);
  }

  async getExpiredTimers(instanceId: string, nowMs: number): Promise<TimerEntry[]> {
    const prefix = `${instanceId}:`;
    const expired: TimerEntry[] = [];
    for (const [key, timer] of this.timers) {
      if (key.startsWith(prefix) && timer.fireAtMs <= nowMs) {
        expired.push({ member: timer.member, fireAtMs: timer.fireAtMs });
      }
    }
    for (const entry of expired) {
      const key = `${instanceId}:${entry.member}`;
      const t = this.timers.get(key);
      if (t?.handle) clearTimeout(t.handle);
      this.timers.delete(key);
    }
    return expired;
  }

  // ═══ Task Queue ═══

  async enqueueTask(task: TaskPayload): Promise<void> {
    const idx = this.taskQueue.findIndex(
      (t) => t.priority > task.priority || (t.priority === task.priority && t.createdAt > task.createdAt),
    );
    if (idx === -1) this.taskQueue.push(task);
    else this.taskQueue.splice(idx, 0, task);

    if (this.consumeWaiters.length > 0) {
      const waiter = this.consumeWaiters.shift()!;
      clearTimeout(waiter.timer);
      waiter.resolve(this.taskQueue.shift() ?? null);
    }
  }

  async consumeTask(_workerId: string, blockMs?: number): Promise<TaskPayload | null> {
    if (this.taskQueue.length > 0) return this.taskQueue.shift()!;
    if (!blockMs || blockMs <= 0) return null;

    return new Promise<TaskPayload | null>((resolve) => {
      const timer = setTimeout(() => {
        const idx = this.consumeWaiters.findIndex((w) => w.resolve === resolve);
        if (idx >= 0) this.consumeWaiters.splice(idx, 1);
        resolve(null);
      }, blockMs);
      timer.unref();
      this.consumeWaiters.push({ resolve, timer });
    });
  }

  async ackTask(_taskId: string): Promise<void> { /* no-op in local mode */ }

  async getQueueDepth(): Promise<{ high: number; low: number }> {
    let high = 0, low = 0;
    for (const t of this.taskQueue) { if (t.priority === 0) high++; else low++; }
    return { high, low };
  }

  /**
   * Snapshot of every task currently waiting in `taskQueue` (FIFO + priority order).
   * Returns a shallow copy so callers can safely iterate without holding a
   * reference into our internal array. Tasks already consumed by a worker
   * are NOT included (they live in PipelineWorker.runningTasks instead).
   */
  async listQueuedTasks(): Promise<TaskPayload[]> {
    return this.taskQueue.slice();
  }

  // ═══ Lock ═══

  private cleanExpiredLocks(): void {
    const now = Date.now();
    for (const [key, lock] of this.locks) {
      if (lock.expireAt <= now) this.locks.delete(key);
    }
  }

  async acquireLock(key: string, ownerId: string, ttlMs: number): Promise<boolean> {
    this.cleanExpiredLocks();
    const existing = this.locks.get(key);
    if (existing && existing.expireAt > Date.now()) return false;
    this.locks.set(key, { ownerId, expireAt: Date.now() + ttlMs });
    return true;
  }

  async renewLock(key: string, ownerId: string, ttlMs: number): Promise<boolean> {
    const existing = this.locks.get(key);
    if (!existing || existing.ownerId !== ownerId) return false;
    existing.expireAt = Date.now() + ttlMs;
    return true;
  }

  async releaseLock(key: string, ownerId: string): Promise<void> {
    const existing = this.locks.get(key);
    if (existing && existing.ownerId === ownerId) this.locks.delete(key);
  }

  // ═══ Atomic Capture ═══

  async captureAtomic(params: CaptureAtomicParams): Promise<CaptureAtomicResult> {
    const { instanceId, sessionId, teamId, agentId, messageJson, threshold, fireAtMs, timerMember, taskPayload, nowMs, rounds } = params;

    await this.appendBuffer(instanceId, sessionId, messageJson, teamId, agentId);

    const stateKey = this.k(instanceId, sessionId, teamId, agentId);
    let state = this.sessionStates.get(stateKey);
    if (!state) {
      state = { ...DEFAULT_PIPELINE_STATE, last_active_time: nowMs };
      this.sessionStates.set(stateKey, state);
    }

    state.conversation_count += rounds;
    state.last_active_time = nowMs;

    if (state.conversation_count >= threshold) {
      await this.enqueueTask(taskPayload);
      state.conversation_count = 0;
      await this.removeTimer(instanceId, timerMember);
      return { triggered: true, conversationCount: 0 };
    }

    await this.setTimer(instanceId, timerMember, fireAtMs);
    return { triggered: false, conversationCount: state.conversation_count };
  }

  // ═══ Instance Lifecycle ═══

  async purgeInstance(instanceId: string): Promise<{ sessions: number; timers: number; buffers: number }> {
    let sessions = 0;
    let timers = 0;
    let buffers = 0;
    const prefix = `${instanceId}:`;

    // Clear session states
    for (const key of [...this.sessionStates.keys()]) {
      if (key.startsWith(prefix)) {
        this.sessionStates.delete(key);
        sessions++;
      }
    }

    // Clear buffers
    for (const key of [...this.buffers.keys()]) {
      if (key.startsWith(prefix)) {
        this.buffers.delete(key);
        buffers++;
      }
    }

    // Clear timers
    for (const [key, timer] of [...this.timers.entries()]) {
      if (key.startsWith(prefix)) {
        if (timer.handle) clearTimeout(timer.handle);
        this.timers.delete(key);
        timers++;
      }
    }

    // Remove tasks belonging to this instance from the queue
    this.taskQueue = this.taskQueue.filter((t) => t.instanceId !== instanceId);

    return { sessions, timers, buffers };
  }

  // ═══ Lifecycle ═══

  async initialize(): Promise<void> { /* no-op */ }

  async destroy(): Promise<void> {
    this.destroyed = true;
    for (const [, timer] of this.timers) { if (timer.handle) clearTimeout(timer.handle); }
    this.timers.clear();
    for (const w of this.consumeWaiters) { clearTimeout(w.timer); w.resolve(null); }
    this.consumeWaiters = [];
    this.sessionStates.clear();
    this.buffers.clear();
    this.taskQueue = [];
    this.locks.clear();
  }

  getSnapshot() {
    return {
      sessions: this.sessionStates.size,
      buffers: this.buffers.size,
      timers: this.timers.size,
      queue: this.taskQueue.length,
      locks: this.locks.size,
    };
  }
}
